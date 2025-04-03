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
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
import transformers
from tensorrt_llm.models.convert_utils import split
from test_utils.ref_layers import (RefTriangleAttentionNode,
                                   RefTriangleMultiplicationNode)

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.attention import TriangleAttention
from tensorrt_bionemo._torch.modules.triangle_nodes import (
    TriangleAttentionNode, TriangleAttentionNodeType,
    TriangleMultiplicationNode, TriangleMultiplicationNodeType)
from tensorrt_bionemo.mapping import Mapping

_MOCK_MODEL_CONFIG = {
    "architectures": ["triangle-attention-nodes"],
    "torch_dtype": "float32",
}


def _load_attn_node_weights(ref_node: RefTriangleAttentionNode,
                            node: TriangleAttention,
                            dtype: torch.dtype,
                            bias: bool = False):
    qkv_weights = [
        {
            "weight": ref_node.mha.linear_q.weight.data.to(dtype),
            "bias": ref_node.mha.linear_q.bias.data.to(dtype) if bias else None
        },
        {
            "weight": ref_node.mha.linear_k.weight.data.to(dtype),
            "bias": ref_node.mha.linear_k.bias.data.to(dtype) if bias else None
        },
        {
            "weight": ref_node.mha.linear_v.weight.data.to(dtype),
            "bias": ref_node.mha.linear_v.bias.data.to(dtype) if bias else None
        },
    ]
    o_proj_weights = [{
        "weight":
        ref_node.mha.linear_o.weight.data.to(dtype),
        "bias":
        ref_node.mha.linear_o.bias.data.to(dtype) if bias else None
    }]
    g_proj_weights = [{
        "weight":
        ref_node.mha.linear_g.weight.data.to(dtype),
        "bias":
        ref_node.mha.linear_g.bias.data.to(dtype) if bias else None
    }]

    node.mha.qkv_proj.load_weights(qkv_weights)
    node.mha.o_proj.load_weights(o_proj_weights)
    node.mha.g_proj.load_weights(g_proj_weights)

    node.linear.load_weights([{
        "weight": ref_node.linear.weight.data.to(dtype),
    }])
    node.layer_norm.weight.data.copy_(ref_node.layer_norm.weight.data.to(dtype))
    node.layer_norm.bias.data.copy_(ref_node.layer_norm.bias.data.to(dtype))


def _load_mul_node_weights(ref_node: RefTriangleMultiplicationNode,
                           node: TriangleMultiplicationNode,
                           dtype: torch.dtype):
    node.norm_in.weight.data.copy_(ref_node.norm_in.weight.data.to(dtype))
    node.norm_in.bias.data.copy_(ref_node.norm_in.bias.data.to(dtype))

    p_in_weights = [
        {
            "weight": split(ref_node.p_in.weight.data.to(dtype), 2, 0,
                            0),  # tp_size=2, tp_rank=0, dim=0
        },
        {
            "weight": split(ref_node.p_in.weight.data.to(dtype), 2, 1,
                            0),  # tp_size=2, tp_rank=1, dim=0
        }
    ]
    g_in_weights = [
        {
            "weight": split(ref_node.g_in.weight.data.to(dtype), 2, 0,
                            0),  # tp_size=2, tp_rank=0, dim=0
        },
        {
            "weight": split(ref_node.g_in.weight.data.to(dtype), 2, 1,
                            0),  # tp_size=2, tp_rank=1, dim=0
        }
    ]
    node.p_in.load_weights(p_in_weights)
    node.g_in.load_weights(g_in_weights)

    node.norm_out.weight.data.copy_(
        ref_node.norm_out.weight.data.to(torch.float32))
    node.norm_out.bias.data.copy_(ref_node.norm_out.bias.data.to(torch.float32))

    p_out_weights = [{
        "weight": ref_node.p_out.weight.data.to(torch.float32),
    }]
    g_out_weights = [{
        "weight": ref_node.g_out.weight.data.to(torch.float32),
    }]
    node.p_out.load_weights(p_out_weights)
    node.g_out.load_weights(g_out_weights)


@dataclass(kw_only=True, frozen=True)
class AttnNodeScenario:
    backend: str
    seq_len: int = 16
    c_in: int = 128
    c_hidden: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    chunk_size: int = 0
    torch_dtype: str = "float32"
    starting: bool = True


@dataclass(kw_only=True, frozen=True)
class MulNodeScenario:
    seq_len: int = 16
    mul_type: TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.OUTGOING
    torch_dtype: str = "float32"


@pytest.mark.parametrize("s", [
    AttnNodeScenario(backend="VANILLA"),
    AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16"),
    AttnNodeScenario(backend="VANILLA", chunk_size=16),
    AttnNodeScenario(backend="VANILLA", chunk_size=8),
    AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=16),
    AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=8),
])
def test_triangle_attention_node(s: AttnNodeScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend).Metadata
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    config_dict["torch_dtype"] = s.torch_dtype
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        attn_backend=s.backend,
        triangle_attn_node_chunk_size=s.chunk_size,
    )
    dtype = model_config.pretrained_config.torch_dtype
    device = torch.device('cuda')

    ref_node = RefTriangleAttentionNode.load_weights(
        no_heads=s.num_attention_heads, starting=s.starting)
    ref_node.to(device)
    ref_node.eval()
    node = TriangleAttentionNode(
        c_in=s.c_in,
        c_hidden=s.c_hidden,
        num_heads=s.num_attention_heads,
        node_type=TriangleAttentionNodeType.STARTING
        if s.starting else TriangleAttentionNodeType.ENDING,
        dtype=dtype,
        config=model_config,
    )
    node.to(device)
    _load_attn_node_weights(ref_node, node, dtype, bias=False)
    attn_metadata = metadata_cls(chunk_size=None,
                                 chunk_dim=None,
                                 mapping=Mapping())
    x = torch.randn(s.seq_len, s.seq_len, s.c_in).cuda()
    mask = torch.randn(s.seq_len, s.seq_len).cuda()

    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_output = ref_node(x, mask)
        output = node(x, mask, attn_metadata)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    elif dtype == torch.bfloat16:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        assert abs(diff0_max - diff1_max) <= 0.3
        assert abs(diff0_mean - diff1_mean) <= 0.05
    # This go NaN for float16


@pytest.mark.parametrize("s", [
    MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING),
    MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING),
    MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING,
                    torch_dtype="bfloat16"),
    MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING,
                    torch_dtype="bfloat16"),
])
def test_triangle_multiplication_node(s: MulNodeScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device('cuda')

    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    config_dict["torch_dtype"] = s.torch_dtype
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict))
    dtype = model_config.pretrained_config.torch_dtype
    ref_node = RefTriangleMultiplicationNode.load_weights(
        outgoing=s.mul_type == TriangleMultiplicationNodeType.OUTGOING)
    ref_node.to(device)
    ref_node.eval()

    node = TriangleMultiplicationNode(
        dim=ref_node.dim,
        multiplication_type=s.mul_type,
        dtype=dtype,
        config=model_config,
    )
    node.to(device)
    _load_mul_node_weights(ref_node, node, dtype)

    x = torch.randn(1, s.seq_len, s.seq_len, ref_node.dim).cuda()
    mask = torch.randn(1, s.seq_len, s.seq_len).cuda()

    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)[0]
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_node.skip_cast()
        ref_output = ref_node(x, mask)[0]
        output = node(x, mask)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-4)
    elif dtype == torch.bfloat16:
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        assert abs(diff0_max - diff1_max) <= 0.3
        assert abs(diff0_mean - diff1_mean) <= 0.05
