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
from dataclasses import dataclass

import pytest
import tensorrt_llm
import torch
import transformers
from mpi4py.futures import MPIPoolExecutor

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.triangle_nodes import (
    TriangleAttentionNode, TriangleAttentionNodeType)
from tensorrt_bionemo.mapping import Mapping

_MOCK_MODEL_CONFIG = {
    "architectures": ["attention"],
    "torch_dtype": "float32",
}


@dataclass(kw_only=True, frozen=True)
class Scenario:
    c_in: int = 32
    c_hidden: int = 8
    num_heads: int = 4
    chunk_size: int = 0
    torch_dtype: str = "float32"
    node_type: str = TriangleAttentionNodeType.STARTING
    tp_size: int = 1
    dp_size: int = 1
    seq_len: int = 128


def _generate_scenarios() -> list[Scenario]:
    ret = []
    total_devs = torch.cuda.device_count()

    for node_type in [
            TriangleAttentionNodeType.STARTING, TriangleAttentionNodeType.ENDING
    ]:
        ret.append(Scenario(tp_size=2, dp_size=1, node_type=node_type))
        ret.append(
            Scenario(tp_size=2, dp_size=1, chunk_size=16, node_type=node_type))
        ret.append(
            Scenario(tp_size=2, dp_size=1, chunk_size=8, node_type=node_type))

        ret.append(Scenario(tp_size=1, dp_size=2, node_type=node_type))
        ret.append(
            Scenario(tp_size=1, dp_size=2, chunk_size=16, node_type=node_type))
        ret.append(
            Scenario(tp_size=1, dp_size=2, chunk_size=8, node_type=node_type))

        if total_devs >= 4:
            ret.append(Scenario(tp_size=1, dp_size=4, node_type=node_type))
            ret.append(Scenario(tp_size=4, dp_size=1, node_type=node_type))
            ret.append(Scenario(tp_size=2, dp_size=2, node_type=node_type))
            ret.append(
                Scenario(tp_size=2,
                         dp_size=2,
                         chunk_size=16,
                         node_type=node_type))
            ret.append(
                Scenario(tp_size=2,
                         dp_size=2,
                         chunk_size=8,
                         node_type=node_type))

        if total_devs >= 8:
            ret.append(Scenario(tp_size=1, dp_size=8, node_type=node_type))
            ret.append(
                Scenario(tp_size=8,
                         dp_size=1,
                         node_type=node_type,
                         num_heads=8,
                         c_hidden=4))
            ret.append(Scenario(tp_size=4, dp_size=2, node_type=node_type))
            ret.append(
                Scenario(tp_size=4,
                         dp_size=2,
                         chunk_size=16,
                         node_type=node_type))
            ret.append(
                Scenario(tp_size=4,
                         dp_size=2,
                         chunk_size=8,
                         node_type=node_type))

            ret.append(Scenario(tp_size=2, dp_size=4, node_type=node_type))
            ret.append(
                Scenario(tp_size=2,
                         dp_size=4,
                         chunk_size=16,
                         node_type=node_type))
            ret.append(
                Scenario(tp_size=2,
                         dp_size=4,
                         chunk_size=8,
                         node_type=node_type))

    return ret


def run_triangle_attn_node_single_rank(single_rank_forward_func, x, mask,
                                       weights, biases, scenario):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(x, mask, weights, biases, scenario, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _load_weights(m, weights, biases):
    linear_weight = weights["linear"]
    norm_weight = weights["norm"]
    norm_bias = biases["norm"]

    mha_weights = weights["mha"]

    m.layer_norm.weight.data.copy_(norm_weight)
    m.layer_norm.bias.data.copy_(norm_bias)
    m.linear.load_weights([dict(weight=linear_weight)])

    qkv_weights = mha_weights["qkv"]
    o_weights = mha_weights["o"]
    g_weights = mha_weights["g"]

    m.mha.qkv_proj.load_weights([
        dict(weight=qkv_weights[0]),
        dict(weight=qkv_weights[1]),
        dict(weight=qkv_weights[2])
    ])
    m.mha.o_proj.load_weights([dict(weight=o_weights[0])])
    m.mha.g_proj.load_weights([dict(weight=g_weights[0])])


def _triangle_attn_node_forward(x, mask, weights, biases, scenario, rank):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    tp_size = scenario.tp_size
    dp_size = scenario.dp_size
    c_in = scenario.c_in
    c_hidden = scenario.c_hidden
    num_heads = scenario.num_heads
    node_type = scenario.node_type
    chunk_size = scenario.chunk_size
    x = x.cuda()
    mask = mask.cuda()
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    mapping = Mapping(world_size=tp_size * dp_size,
                      tp_size=tp_size,
                      dp_size=dp_size,
                      rank=rank)
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
        triangle_attn_node_chunk_size=chunk_size)
    dtype = model_config.pretrained_config.torch_dtype
    metadata_cls = get_attention_backend("VANILLA").Metadata
    attn_metadata = metadata_cls(mapping=mapping)

    multi_devs_tri_attn_node = TriangleAttentionNode(
        c_in=c_in,
        c_hidden=c_hidden,
        num_heads=num_heads,
        node_type=node_type,
        layer_idx=0,
        dtype=dtype,
        config=model_config,
    )
    multi_devs_tri_attn_node.cuda()
    multi_devs_tri_attn_node.eval()
    _load_weights(multi_devs_tri_attn_node, weights, biases)

    multi_devs_output = multi_devs_tri_attn_node.forward(x, mask, attn_metadata)

    mapping = Mapping()
    single_model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping,
        attn_backend="VANILLA",
    )
    attn_metadata = metadata_cls(mapping=mapping)

    single_dev_tri_attn_node = TriangleAttentionNode(
        c_in=c_in,
        c_hidden=c_hidden,
        num_heads=num_heads,
        node_type=node_type,
        layer_idx=0,
        dtype=dtype,
        config=single_model_config,
    )
    _load_weights(single_dev_tri_attn_node, weights, biases)
    single_dev_tri_attn_node.cuda()
    single_dev_tri_attn_node.eval()

    single_dev_output = single_dev_tri_attn_node.forward(x, mask, attn_metadata)
    torch.cuda.synchronize()
    torch.testing.assert_close(multi_devs_output,
                               single_dev_output,
                               atol=1e-3,
                               rtol=1e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario", _generate_scenarios())
def test_triangle_attn_node_parallelism(scenario: Scenario):
    torch.manual_seed(42)
    x = torch.randn(scenario.seq_len, scenario.seq_len, scenario.c_in)
    mask = torch.randn(scenario.seq_len, scenario.seq_len)
    weights = {
        "linear": torch.randn(scenario.num_heads, scenario.c_in),
        "norm": torch.randn(scenario.c_in),
        "mha": {
            "qkv": [
                torch.randn(scenario.c_in, scenario.c_in),
                torch.randn(scenario.c_in, scenario.c_in),
                torch.randn(scenario.c_in, scenario.c_in),
            ],
            "o": [torch.randn(scenario.c_in, scenario.c_in)],
            "g": [torch.randn(scenario.c_in, scenario.c_in)],
        }
    }
    biases = {
        "norm": torch.randn(scenario.c_in),
    }
    world_size = scenario.tp_size * scenario.dp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_triangle_attn_node_single_rank,
            *zip(*[(_triangle_attn_node_forward, x, mask, weights, biases,
                    scenario)] * world_size))
        for r in results:
            assert r is True
