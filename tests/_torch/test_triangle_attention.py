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
    create_triangle_attention_weights,
    load_triangle_attention_weights_torch,
)
from test_utils.boltz.ref_attn import RefTriangleAttention

from tensorrt_bionemo._torch.attention_backend import AttentionType, get_attention_backend
from tensorrt_bionemo._torch.layers.attention import TriangleAttention
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_pair_mask
from tests._torch import skip_cutedsl as _skip_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 16
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    gating: bool = True
    # FIXME: chunked attention (chunk_size / chunk_dim) is not covered by
    # this scenario type.
    torch_dtype: str = "float32"


def _make_backend_and_reference_biases(bs, seq_len, num_heads, dtype, device):
    """Build equivalent backend-specific and additive reference biases."""
    pair_mask = torch.ones(bs, seq_len, seq_len, dtype=torch.float32, device=device)
    pair_mask[..., -(seq_len // 4) :] = 0
    additive_mask = ((pair_mask - 1.0) * 1e9).unsqueeze(-2).unsqueeze(-2)
    triangle_bias = torch.randn(bs, num_heads, seq_len, seq_len, dtype=dtype, device=device)
    return ([additive_mask.to(dtype), triangle_bias], [additive_mask.to(dtype), triangle_bias])


@pytest.mark.parametrize(
    "s",
    [
        Scenario(backend="VANILLA"),
        Scenario(backend="VANILLA", torch_dtype="bfloat16"),
        Scenario(backend="SDPA"),
        Scenario(backend="SDPA", torch_dtype="bfloat16"),
        Scenario(backend="CUEQUIV"),
        Scenario(backend="CUEQUIV", torch_dtype="bfloat16"),
    ],
)
def test_triangle_attention_backend(s: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend, AttentionType.TRIANGLE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(s.torch_dtype)
    device = torch.device("cuda")

    ref_attn = RefTriangleAttention.load_weights(no_heads=s.num_attention_heads)
    ref_attn = ref_attn.to(device)

    weights_and_biases = create_triangle_attention_weights(from_ref=ref_attn)

    attn = TriangleAttention(
        layer_idx=0,
        hidden_size=s.hidden_size,
        head_dim=s.hidden_size // s.num_attention_heads,
        num_attention_heads=s.num_attention_heads,
        num_key_value_heads=s.num_key_value_heads,
        gating=s.gating,
        dtype=dtype,
        attn_backend=s.backend,
    )
    load_triangle_attention_weights_torch(attn, weights_and_biases, dtype=dtype)
    attn.to(device)
    attn_metadata = metadata_cls()
    hidden_states = torch.randn(bs, s.seq_len, s.seq_len, s.hidden_size, dtype=torch.float32, device=device)
    backend_biases, reference_biases = _make_backend_and_reference_biases(
        bs,
        s.seq_len,
        s.num_attention_heads,
        torch.float32,
        device,
    )

    with torch.inference_mode():
        ref_output_float = ref_attn(hidden_states, hidden_states, biases=reference_biases)
        hidden_states = hidden_states.to(dtype)
        backend_biases = [bias.to(dtype) for bias in backend_biases]
        reference_biases = [bias.to(dtype) for bias in reference_biases]
        ref_attn = ref_attn.to(dtype)
        ref_output = ref_attn(hidden_states, hidden_states, biases=reference_biases)
        output = attn(hidden_states, biases=backend_biases, attn_metadata=attn_metadata)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))
        assert diff0_max <= diff1_max * 1.6 + 1e-3
        assert diff0_mean <= diff1_mean + 0.2


@_skip_cutedsl
@pytest.mark.parametrize(
    "s",
    [
        Scenario(backend="CuTeDSL", torch_dtype="bfloat16"),
    ],
)
def test_triangle_attention_cutedsl(s: Scenario):
    """CuTeDSL uses per-row int32 KV lengths and supports fp16/bf16."""
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend, AttentionType.TRIANGLE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(s.torch_dtype)
    device = torch.device("cuda")

    ref_attn = RefTriangleAttention.load_weights(no_heads=s.num_attention_heads)
    ref_attn = ref_attn.to(device)
    weights_and_biases = create_triangle_attention_weights(from_ref=ref_attn)

    attn = TriangleAttention(
        layer_idx=0,
        hidden_size=s.hidden_size,
        head_dim=s.hidden_size // s.num_attention_heads,
        num_attention_heads=s.num_attention_heads,
        num_key_value_heads=s.num_key_value_heads,
        gating=s.gating,
        dtype=dtype,
        attn_backend=s.backend,
    )
    load_triangle_attention_weights_torch(attn, weights_and_biases, dtype=dtype)
    attn.to(device)
    attn_metadata = metadata_cls()

    hidden_states = torch.randn(bs, s.seq_len, s.seq_len, s.hidden_size, dtype=torch.float32, device=device)

    binary_mask = make_left_aligned_pair_mask(bs, s.seq_len, dtype=torch.float32, device=device)
    triangle_bias = torch.randn(bs, s.num_attention_heads, s.seq_len, s.seq_len, dtype=torch.float32, device=device)

    inf_val = 1e9
    ref_mask_bias = ((binary_mask - 1.0) * inf_val).unsqueeze(-2).unsqueeze(-3)
    ref_biases = [ref_mask_bias, triangle_bias]

    actual_s_kv = (binary_mask > 0.5).sum(dim=-1).to(dtype=torch.int32)
    cutedsl_biases = [actual_s_kv, triangle_bias.to(dtype)]

    with torch.inference_mode():
        ref_output_float = ref_attn(hidden_states, hidden_states, biases=ref_biases)

        ref_attn_typed = ref_attn.to(dtype)
        ref_biases_typed = [b.to(dtype) for b in ref_biases]
        ref_output_typed = ref_attn_typed(hidden_states.to(dtype), hidden_states.to(dtype), biases=ref_biases_typed)

        output = attn(hidden_states.to(dtype), biases=cutedsl_biases, attn_metadata=attn_metadata)

    assert output.shape == ref_output_typed.shape

    diff_ours = torch.max(torch.abs(output.float() - ref_output_float))
    diff_ref = torch.max(torch.abs(ref_output_typed.float() - ref_output_float))
    assert diff_ours <= 2.0 * diff_ref + 1e-3, f"CuTeDSL diff={diff_ours}, ref bf16 diff={diff_ref}"
