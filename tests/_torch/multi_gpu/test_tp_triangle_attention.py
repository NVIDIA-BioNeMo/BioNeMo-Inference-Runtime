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
from tensorrt_bionemo._torch.modules.attention import TriangleAttention

_MOCK_MODEL_CONFIG = {
    "architectures": ["attention"],
    "torch_dtype": "float32",
}


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    biases, num_attention_heads, qkv_weights, o_weights,
                    g_weights, hidden_size):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, biases, hidden_size,
                                 num_attention_heads, tensor_parallel_size,
                                 rank, qkv_weights, o_weights, g_weights)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def triangle_attn_forward(x, biases, hidden_size, num_attention_heads,
                          tensor_parallel_size, tensor_parallel_rank,
                          qkv_weights, o_weights, g_weights):
    x = x.cuda()
    biases = [bias.cuda() for bias in biases]

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

    tri_attn = TriangleAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        layer_idx=0,
        dtype=dtype,
        config=model_config,
    )
    tri_attn.qkv_proj.load_weights([
        dict(weight=qkv_weights[0]),
        dict(weight=qkv_weights[1]),
        dict(weight=qkv_weights[2])
    ])
    tri_attn.o_proj.load_weights([dict(weight=o_weights[0])])
    tri_attn.g_proj.load_weights([dict(weight=g_weights[0])])

    tri_attn.cuda()

    # tri_attn = torch.compile(tri_attn, fullgraph=True)
    multi_dev_output = tri_attn.forward(x, biases, attn_metadata)

    mapping.enable_attention_dp = True
    single_model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
    )
    attn_metadata = metadata_cls(mapping=mapping)
    single_dev_tri_attn = TriangleAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        layer_idx=0,
        dtype=dtype,
        config=single_model_config,
    )
    single_dev_tri_attn.qkv_proj.load_weights([
        dict(weight=qkv_weights[0]),
        dict(weight=qkv_weights[1]),
        dict(weight=qkv_weights[2])
    ])
    single_dev_tri_attn.o_proj.load_weights([dict(weight=o_weights[0])])
    single_dev_tri_attn.g_proj.load_weights([dict(weight=g_weights[0])])
    single_dev_tri_attn.cuda()

    if tensor_parallel_rank == 0:
        single_dev_output = single_dev_tri_attn.forward(x, biases,
                                                        attn_metadata)
        torch.cuda.synchronize()
        assert multi_dev_output.shape == single_dev_output.shape
        torch.testing.assert_close(multi_dev_output, single_dev_output)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("num_attention_heads", [8, 7],
                         ids=["balanced", "unbalanced"])
def test_triangle_attn_forward(num_attention_heads):
    torch.manual_seed(42)
    seq_len = 4
    hidden_size = 32
    tensor_parallel_size = 2
    biases = [
        torch.randn(seq_len, 1, 1, seq_len, dtype=torch.float32),
        torch.randn(1,
                    num_attention_heads,
                    seq_len,
                    seq_len,
                    dtype=torch.float32)
    ]
    hidden_states = torch.randn(seq_len,
                                seq_len,
                                hidden_size,
                                dtype=torch.float32)
    qkv_weights = [
        torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        torch.randn(hidden_size, hidden_size, dtype=torch.float32),
    ]
    o_weights = [
        torch.randn(hidden_size, hidden_size, dtype=torch.float32),
    ]
    g_weights = [
        torch.randn(hidden_size, hidden_size, dtype=torch.float32),
    ]

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, triangle_attn_forward, hidden_states,
                    biases, num_attention_heads, qkv_weights, o_weights,
                    g_weights, hidden_size)] * 2))
        if num_attention_heads % 2 != 0:
            with pytest.raises(AssertionError):
                for r in results:
                    assert r is True
        else:
            for r in results:
                assert r is True
