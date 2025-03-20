# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import traceback
from copy import deepcopy

import pytest
import tensorrt_llm
import torch
import transformers
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.mapping import Mapping

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.attention import SelfAttentionPairBias

_MOCK_MODEL_CONFIG = {
    "architectures": ["attention"],
    "torch_dtype": "float32",
}


def run_single_rank(tensor_parallel_size, single_rank_forward_func, s, z, mask,
                    num_attention_heads, c_s, c_z, q_weight, q_bias, k_weight,
                    v_weight, o_weight, g_weight, z_weights, z_biases):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(s, z, mask, num_attention_heads, c_s, c_z,
                                 tensor_parallel_size, rank, q_weight, q_bias,
                                 k_weight, v_weight, o_weight, g_weight,
                                 z_weights, z_biases)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def pairwise_attn_forward(s, z, mask, num_attention_heads, c_s, c_z,
                          tensor_parallel_size, tensor_parallel_rank, q_weight,
                          q_bias, k_weight, v_weight, o_weight, g_weight,
                          z_weights, z_biases):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    s = s.cuda()
    z = z.cuda()
    mask = mask.cuda()

    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    mapping = Mapping(world_size=tensor_parallel_size,
                      tp_size=tensor_parallel_size,
                      rank=tensor_parallel_rank)

    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
    )
    dtype = model_config.pretrained_config.torch_dtype
    metadata_cls = get_attention_backend("VANILLA").Metadata
    attn_metadata = metadata_cls(mapping=mapping)

    pairwise_attn = SelfAttentionPairBias(
        layer_idx=0,
        c_s=c_s,
        c_z=c_z,
        num_heads=num_attention_heads,
        dtype=dtype,
        config=model_config,
    )
    pairwise_attn.proj_q.load_weights([dict(weight=q_weight, bias=q_bias)])
    pairwise_attn.proj_k.load_weights([dict(weight=k_weight)])
    pairwise_attn.proj_v.load_weights([dict(weight=v_weight)])
    pairwise_attn.proj_o.load_weights([dict(weight=o_weight)])
    pairwise_attn.proj_g.load_weights([dict(weight=g_weight)])
    pairwise_attn.proj_z[0].weight.data.copy_(z_weights[0])
    pairwise_attn.proj_z[0].bias.data.copy_(z_biases[0])
    pairwise_attn.proj_z[1].load_weights([dict(weight=z_weights[1])])

    pairwise_attn.cuda()

    multi_dev_output = pairwise_attn.forward(s, z, mask, attn_metadata)

    mapping.enable_attention_dp = True
    single_model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
    )
    attn_metadata = metadata_cls(mapping=mapping)
    single_dev_pairwise_attn = SelfAttentionPairBias(
        layer_idx=0,
        c_s=c_s,
        c_z=c_z,
        num_heads=num_attention_heads,
        dtype=dtype,
        config=single_model_config,
    )
    single_dev_pairwise_attn.proj_q.load_weights(
        [dict(weight=q_weight, bias=q_bias)])
    single_dev_pairwise_attn.proj_k.load_weights([dict(weight=k_weight)])
    single_dev_pairwise_attn.proj_v.load_weights([dict(weight=v_weight)])
    single_dev_pairwise_attn.proj_o.load_weights([dict(weight=o_weight)])
    single_dev_pairwise_attn.proj_g.load_weights([dict(weight=g_weight)])
    single_dev_pairwise_attn.proj_z[0].weight.data.copy_(z_weights[0])
    single_dev_pairwise_attn.proj_z[0].bias.data.copy_(z_biases[0])
    single_dev_pairwise_attn.proj_z[1].load_weights([dict(weight=z_weights[1])])

    single_dev_pairwise_attn.cuda()

    if tensor_parallel_rank == 0:
        single_dev_output = single_dev_pairwise_attn.forward(
            s, z, mask, attn_metadata)
        torch.cuda.synchronize()
        assert multi_dev_output.shape == single_dev_output.shape
        torch.testing.assert_close(multi_dev_output,
                                   single_dev_output,
                                   atol=1e-4,
                                   rtol=1e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("num_attention_heads", [16, 17],
                         ids=["balanced", "unbalanced"])
def test_tp_pairwise_attention(num_attention_heads):
    torch.manual_seed(42)
    tensor_parallel_size = 2
    num_attention_heads = num_attention_heads
    c_s = 384
    c_z = 128
    seq_len = 117
    b = 1

    q_weight = torch.randn(c_s, c_s, dtype=torch.float32)
    q_bias = torch.randn(c_s, dtype=torch.float32)
    k_weight = torch.randn(c_s, c_s, dtype=torch.float32)
    v_weight = torch.randn(c_s, c_s, dtype=torch.float32)
    o_weight = torch.randn(c_s, c_s, dtype=torch.float32)
    g_weight = torch.randn(c_s, c_s, dtype=torch.float32)
    z_weights = [
        torch.randn(c_z, dtype=torch.float32),
        torch.randn(num_attention_heads, c_z, dtype=torch.float32),
    ]
    z_biases = [
        torch.randn(c_z, dtype=torch.float32),
        None,
    ]
    s = torch.randn(b, seq_len, c_s, dtype=torch.float32)
    z = torch.randn(b, seq_len, seq_len, c_z, dtype=torch.float32)
    mask = torch.randn(b, seq_len, dtype=torch.float32)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, pairwise_attn_forward, s, z, mask,
                    num_attention_heads, c_s, c_z, q_weight, q_bias, k_weight,
                    v_weight, o_weight, g_weight, z_weights, z_biases)] * 2))
        if num_attention_heads % 2 != 0:
            with pytest.raises(AssertionError):
                for r in results:
                    assert r is True
        else:
            for r in results:
                assert r is True
