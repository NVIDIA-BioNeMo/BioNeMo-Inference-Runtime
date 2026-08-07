# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_triangle_attention_node_weights,
    create_triangle_multiplication_node_weights,
    load_triangle_attention_node_weights_torch,
    load_triangle_multiplication_node_weights_torch,
)
from test_utils.boltz.ref_layers import RefTriangleAttentionNode, RefTriangleMultiplicationNode

from tensorrt_bionemo._torch.attention_backend import AttentionType, get_attention_backend
from tensorrt_bionemo._torch.layers.triangle_nodes import (
    TriangleAttentionNode,
    TriangleAttentionNodeType,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
)
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_pair_mask


@dataclass(kw_only=True, frozen=True)
class AttnNodeScenario:
    backend: str
    seq_len: int = 32
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
    high_precision: bool = True


@pytest.mark.parametrize(
    "s",
    [
        AttnNodeScenario(backend="VANILLA"),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16"),
        AttnNodeScenario(backend="SDPA"),
        AttnNodeScenario(backend="SDPA", torch_dtype="bfloat16"),
        AttnNodeScenario(backend="VANILLA", chunk_size=16),
        AttnNodeScenario(backend="VANILLA", chunk_size=8),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=16),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=8),
    ],
)
def test_triangle_attention_node(s: AttnNodeScenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend, AttentionType.TRIANGLE).Metadata
    bs = 1

    dtype = str_dtype_to_torch(s.torch_dtype)
    device = torch.device("cuda")

    ref_node = RefTriangleAttentionNode.load_weights(no_heads=s.num_attention_heads, starting=s.starting)
    ref_node.to(device)
    ref_node.eval()

    weights_and_biases = create_triangle_attention_node_weights(from_ref=ref_node)
    node = TriangleAttentionNode(
        c_in=s.c_in,
        c_hidden=s.c_hidden,
        num_heads=ref_node.num_heads,
        node_type=TriangleAttentionNodeType.STARTING if s.starting else TriangleAttentionNodeType.ENDING,
        dtype=dtype,
        attn_backend=s.backend,
        skip_create_weights=False,
    )
    node.to(device)
    load_triangle_attention_node_weights_torch(node, weights_and_biases, dtype)
    attn_metadata = metadata_cls()
    x = torch.randn(bs, s.seq_len, s.seq_len, s.c_in, dtype=torch.float32).cuda()
    mask = make_left_aligned_pair_mask(bs, s.seq_len, dtype=torch.float32, device="cuda")

    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_output = ref_node(x, mask)
        output = node(x, mask, attn_metadata=attn_metadata)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    elif dtype == torch.bfloat16:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert diff0_max <= diff1_max * 1.5 + 1e-3
        assert diff0_mean <= diff1_mean + 0.05


@pytest.mark.parametrize(
    "s",
    [
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING, torch_dtype="bfloat16"),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16"),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", high_precision=False),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", seq_len=512),
        MulNodeScenario(
            mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", seq_len=512, high_precision=False
        ),
    ],
)
def test_triangle_multiplication_node(s: MulNodeScenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(s.torch_dtype)
    ref_node = RefTriangleMultiplicationNode.load_weights(
        outgoing=s.mul_type == TriangleMultiplicationNodeType.OUTGOING
    )
    ref_node.to(device)
    ref_node.eval()
    bs = 1

    weights_and_biases = create_triangle_multiplication_node_weights(from_ref=ref_node)
    node = TriangleMultiplicationNode(
        dim=ref_node.dim,
        multiplication_type=s.mul_type,
        dtype=dtype,
        skip_create_weights=False,
        high_precision=s.high_precision,
    )
    node.to(device)
    load_triangle_multiplication_node_weights_torch(node, weights_and_biases, dtype)

    x = torch.randn(bs, s.seq_len, s.seq_len, ref_node.dim, device="cuda")
    mask = make_left_aligned_pair_mask(1, s.seq_len, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_output = ref_node(x, mask)
        output = node(x, mask)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-4)
    elif dtype == torch.bfloat16:
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.05
