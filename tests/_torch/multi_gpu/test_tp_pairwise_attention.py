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
from test_utils.create_and_load_weights import (
    create_self_pairwise_attention_weights,
    load_self_pairwise_attention_weights_torch)

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.attention import SelfAttentionPairBias
from tensorrt_bionemo.mapping import Mapping

_MOCK_MODEL_CONFIG = {
    "architectures": ["attention"],
    "torch_dtype": "float32",
}


def run_single_rank(tensor_parallel_size, single_rank_forward_func, s, z, mask,
                    num_attention_heads, c_s, c_z, weights_and_biases):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(s, z, mask, num_attention_heads, c_s, c_z,
                                 tensor_parallel_size, rank, weights_and_biases)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def pairwise_attn_forward(s, z, mask, num_attention_heads, c_s, c_z,
                          tensor_parallel_size, tensor_parallel_rank,
                          weights_and_biases):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
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
        max_attention_pairwise_tp_size=False)
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
    load_self_pairwise_attention_weights_torch(pairwise_attn,
                                               weights_and_biases,
                                               dtype=dtype)
    pairwise_attn.cuda()

    multi_dev_output = pairwise_attn(s, z, mask, attn_metadata)
    # create single mapping
    mapping = Mapping()
    single_model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
        max_attention_pairwise_tp_size=False)
    attn_metadata = metadata_cls(mapping=mapping)
    single_dev_pairwise_attn = SelfAttentionPairBias(
        layer_idx=0,
        c_s=c_s,
        c_z=c_z,
        num_heads=num_attention_heads,
        dtype=dtype,
        config=single_model_config,
    )
    load_self_pairwise_attention_weights_torch(single_dev_pairwise_attn,
                                               weights_and_biases,
                                               dtype=dtype)

    single_dev_pairwise_attn.cuda()

    single_dev_output = single_dev_pairwise_attn(s, z, mask, attn_metadata)
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

    s = torch.randn(b, seq_len, c_s, dtype=torch.float32)
    z = torch.randn(b, seq_len, seq_len, c_z, dtype=torch.float32)
    mask = torch.randn(b, seq_len, dtype=torch.float32)

    weights_and_biases = create_self_pairwise_attention_weights(
        c_s=c_s, c_z=c_z, num_attention_heads=num_attention_heads)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, pairwise_attn_forward, s, z, mask,
                    num_attention_heads, c_s, c_z, weights_and_biases)] * 2))
        if num_attention_heads % 2 != 0:
            with pytest.raises((AssertionError, RuntimeError)):
                for r in results:
                    assert r is True
        else:
            for r in results:
                assert r is True
