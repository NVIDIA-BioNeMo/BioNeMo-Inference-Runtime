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
from test_utils.create_and_load_weights import (
    create_triangle_attention_node_weights,
    create_triangle_multiplication_node_weights,
    load_triangle_attention_node_weights_torch,
    load_triangle_multiplication_node_weights_torch)

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.triangle_nodes import (
    TriangleAttentionNode, TriangleAttentionNodeType,
    TriangleMultiplicationNode, TriangleMultiplicationNodeType)
from tensorrt_bionemo.mapping import Mapping

_MOCK_MODEL_CONFIG = {
    "architectures": ["triangle_nodes"],
    "torch_dtype": "float32",
}


@dataclass(kw_only=True, frozen=True)
class AttnNodeScenario:
    c_in: int = 32
    c_hidden: int = 8
    num_heads: int = 4
    chunk_size: int = 0
    torch_dtype: str = "float32"
    node_type: str = TriangleAttentionNodeType.STARTING
    tp_size: int = 1
    dcp_size: int = 1
    seq_len: int = 128


@dataclass(kw_only=True, frozen=True)
class MulNodeScenario:
    dim: int = 32
    seq_len: int = 16
    torch_dtype: str = "float32"
    node_type: str = TriangleMultiplicationNodeType.OUTGOING
    tp_size: int = 1
    dcp_size: int = 1


def _generate_attn_node_scenarios() -> list[AttnNodeScenario]:
    ret = []
    total_devs = torch.cuda.device_count()

    for node_type in [
            TriangleAttentionNodeType.STARTING, TriangleAttentionNodeType.ENDING
    ]:
        ret.append(AttnNodeScenario(tp_size=2, dcp_size=1, node_type=node_type))
        ret.append(
            AttnNodeScenario(tp_size=2,
                             dcp_size=1,
                             chunk_size=16,
                             node_type=node_type))
        ret.append(
            AttnNodeScenario(tp_size=2,
                             dcp_size=1,
                             chunk_size=8,
                             node_type=node_type))

        ret.append(AttnNodeScenario(tp_size=1, dcp_size=2, node_type=node_type))
        ret.append(
            AttnNodeScenario(tp_size=1,
                             dcp_size=2,
                             chunk_size=16,
                             node_type=node_type))
        ret.append(
            AttnNodeScenario(tp_size=1,
                             dcp_size=2,
                             chunk_size=8,
                             node_type=node_type))

        if total_devs >= 4:
            ret.append(
                AttnNodeScenario(tp_size=1, dcp_size=4, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=4, dcp_size=1, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=2, dcp_size=2, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=2,
                                 dcp_size=2,
                                 chunk_size=16,
                                 node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=2,
                                 dcp_size=2,
                                 chunk_size=8,
                                 node_type=node_type))

        if total_devs >= 8:
            ret.append(
                AttnNodeScenario(tp_size=1, dcp_size=8, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=8,
                                 dcp_size=1,
                                 node_type=node_type,
                                 num_heads=8,
                                 c_hidden=4))
            ret.append(
                AttnNodeScenario(tp_size=4, dcp_size=2, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=4,
                                 dcp_size=2,
                                 chunk_size=16,
                                 node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=4,
                                 dcp_size=2,
                                 chunk_size=8,
                                 node_type=node_type))

            ret.append(
                AttnNodeScenario(tp_size=2, dcp_size=4, node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=2,
                                 dcp_size=4,
                                 chunk_size=16,
                                 node_type=node_type))
            ret.append(
                AttnNodeScenario(tp_size=2,
                                 dcp_size=4,
                                 chunk_size=8,
                                 node_type=node_type))
    return ret


def _generate_mul_node_scenarios() -> list[MulNodeScenario]:
    ret = []
    ids = []
    total_devs = torch.cuda.device_count()

    for node_type in [
            TriangleMultiplicationNodeType.OUTGOING,
            TriangleMultiplicationNodeType.INCOMING
    ]:
        for tp_size, dcp_size in product([1, 2, 4], repeat=2):
            if tp_size * dcp_size > total_devs:
                continue
            ret.append(
                MulNodeScenario(tp_size=tp_size,
                                dcp_size=dcp_size,
                                node_type=node_type))
            ids.append(f"{node_type.name}_{tp_size}_{dcp_size}")

    return ret, ids


def run_triangle_attn_node_single_rank(single_rank_forward_func, x, mask,
                                       weights_and_biases, scenario):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(x, mask, weights_and_biases, scenario, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


def run_triangle_mul_node_single_rank(single_rank_forward_func, x, mask,
                                      weights_and_biases, scenario):
    import tensorrt_llm
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(x, mask, weights_and_biases, scenario, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _triangle_attn_node_forward(x, mask, weights_and_biases, scenario, rank):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    tp_size = scenario.tp_size
    dcp_size = scenario.dcp_size
    c_in = scenario.c_in
    c_hidden = scenario.c_hidden
    num_heads = scenario.num_heads
    node_type = scenario.node_type
    chunk_size = scenario.chunk_size
    x = x.cuda()
    mask = mask.cuda()
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    mapping = Mapping(world_size=tp_size * dcp_size,
                      tp_size=tp_size,
                      dcp_size=dcp_size,
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
    load_triangle_attention_node_weights_torch(multi_devs_tri_attn_node,
                                               weights_and_biases,
                                               dtype=dtype)

    with torch.inference_mode():
        multi_devs_output = multi_devs_tri_attn_node.forward(
            x, mask, attn_metadata)

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
    load_triangle_attention_node_weights_torch(single_dev_tri_attn_node,
                                               weights_and_biases,
                                               dtype=dtype)
    single_dev_tri_attn_node.cuda()
    single_dev_tri_attn_node.eval()

    with torch.inference_mode():
        single_dev_output = single_dev_tri_attn_node.forward(
            x, mask, attn_metadata)
    torch.cuda.synchronize()
    torch.testing.assert_close(multi_devs_output,
                               single_dev_output,
                               atol=1e-3,
                               rtol=1e-2)


def _triangle_mul_node_forward(x, mask, weights_and_biases, scenario, rank):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    tp_size = scenario.tp_size
    dcp_size = scenario.dcp_size
    dim = scenario.dim

    x = x.cuda()
    mask = mask.cuda()
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    mapping = Mapping(world_size=tp_size * dcp_size,
                      tp_size=tp_size,
                      dcp_size=dcp_size,
                      rank=rank)
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping)
    dtype = model_config.pretrained_config.torch_dtype

    multi_devs_tri_mul_node = TriangleMultiplicationNode(
        dim=dim,
        dtype=dtype,
        config=model_config,
        multiplication_type=scenario.node_type,
    )
    multi_devs_tri_mul_node.cuda()
    multi_devs_tri_mul_node.eval()
    load_triangle_multiplication_node_weights_torch(multi_devs_tri_mul_node,
                                                    weights_and_biases,
                                                    dtype=dtype)

    with torch.inference_mode():
        multi_devs_output = multi_devs_tri_mul_node.forward(x, mask)
    torch.cuda.synchronize(torch.cuda.current_device())
    mapping = Mapping()
    single_model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        mapping=mapping)

    single_dev_tri_mul_node = TriangleMultiplicationNode(
        dim=dim,
        dtype=dtype,
        config=single_model_config,
        multiplication_type=scenario.node_type,
    )
    single_dev_tri_mul_node.cuda()
    single_dev_tri_mul_node.eval()
    load_triangle_multiplication_node_weights_torch(single_dev_tri_mul_node,
                                                    weights_and_biases,
                                                    dtype=dtype)

    with torch.inference_mode():
        single_dev_output = single_dev_tri_mul_node.forward(x, mask)
    torch.cuda.synchronize(torch.cuda.current_device())
    torch.testing.assert_close(multi_devs_output,
                               single_dev_output,
                               atol=1e-3,
                               rtol=1e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_attn_node_scenarios()[0],
                         ids=_generate_attn_node_scenarios()[1])
def test_triangle_attn_node_parallelism(scenario: AttnNodeScenario):
    torch.manual_seed(42)
    x = torch.randn(scenario.seq_len, scenario.seq_len, scenario.c_in)
    mask = torch.randn(scenario.seq_len, scenario.seq_len)
    weights_and_biases = create_triangle_attention_node_weights(
        c_in=scenario.c_in,
        c_hidden=scenario.c_hidden,
        num_attention_heads=scenario.num_heads,
        torch_dtype=torch.float32,
    )
    world_size = scenario.tp_size * scenario.dcp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_triangle_attn_node_single_rank,
            *zip(*[(_triangle_attn_node_forward, x, mask, weights_and_biases,
                    scenario)] * world_size))
        for r in results:
            assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario", _generate_mul_node_scenarios())
def test_triangle_mul_node_parallelism(scenario: MulNodeScenario):
    torch.manual_seed(42)
    x = torch.rand(scenario.seq_len, scenario.seq_len, scenario.dim)
    mask = torch.randn(scenario.seq_len, scenario.seq_len)
    weights_and_biases = create_triangle_multiplication_node_weights(
        dim=scenario.dim,
        torch_dtype=torch.float32,
    )
    world_size = scenario.tp_size * scenario.dcp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_triangle_mul_node_single_rank,
            *zip(*[(_triangle_mul_node_forward, x, mask, weights_and_biases,
                    scenario)] * world_size))
        for r in results:
            assert r is True
