# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from tensorrt_bionemo._torch.custom_ops.gated_sigmoid import (
    GatedSigmoidCuTe, _classify_m_range, _invoke_vanilla_gated_sigmoid,
    get_gated_sigmoid_op)
from tests._torch import SM_VERSION, skip_if_no_cutedsl


def _ref_gated_sigmoid(
    s: torch.Tensor,
    weight: torch.Tensor,
    mha_out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks."""
    gate = torch.nn.functional.linear(
        s.float(), weight.float(),
        bias.float() if bias is not None else None)
    return (gate.sigmoid() * mha_out.float()).to(s.dtype)


# ---------------------------------------------------------------------------
# Parametrized M values
# ---------------------------------------------------------------------------

# Odd M values exercise CTA-tile predicate logic (partial tiles).
# CTA tiles by range: short=64x64, medium=64x128, long=128x64.

SHORT_M_VALUES = [
    1,  # single row
    33,  # odd, < one CTA row tile
    65,  # one full tile + 1 row
    127,  # just under 2 tiles (64*2=128)
    511,  # large odd
    999,  # odd, near boundary
    1024,  # exact boundary
]

MEDIUM_M_VALUES = [
    1025,  # boundary + 1
    1279,  # odd
    1537,  # odd
    2047,  # boundary - 1
    2048,  # exact boundary
]

LONG_M_VALUES = [
    2049,  # boundary + 1
    2561,  # odd
    3001,  # odd
    3455,  # odd, near upper benchmark limit
]

# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    K: int = 128
    N_out: int = 128
    M_values: list[int] = field(default_factory=lambda: [256])
    has_bias: bool = False
    dtype: torch.dtype = torch.bfloat16
    atol: float = 0.05
    rtol: float = 1e-2


# ---------------------------------------------------------------------------
# CuTe kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", [
    Scenario(M_values=SHORT_M_VALUES, has_bias=True, dtype=torch.bfloat16),
    Scenario(M_values=SHORT_M_VALUES, has_bias=False, dtype=torch.bfloat16),
    Scenario(M_values=MEDIUM_M_VALUES, has_bias=True, dtype=torch.bfloat16),
    Scenario(M_values=MEDIUM_M_VALUES, has_bias=False, dtype=torch.bfloat16),
    Scenario(M_values=LONG_M_VALUES, has_bias=True, dtype=torch.bfloat16),
    Scenario(M_values=LONG_M_VALUES, has_bias=False, dtype=torch.bfloat16),
    Scenario(M_values=SHORT_M_VALUES, has_bias=True, dtype=torch.float16),
    Scenario(M_values=SHORT_M_VALUES, has_bias=False, dtype=torch.float16),
    Scenario(M_values=MEDIUM_M_VALUES, has_bias=True, dtype=torch.float16),
    Scenario(M_values=MEDIUM_M_VALUES, has_bias=False, dtype=torch.float16),
    Scenario(M_values=LONG_M_VALUES, has_bias=True, dtype=torch.float16),
    Scenario(M_values=LONG_M_VALUES, has_bias=False, dtype=torch.float16),
],
                         ids=[
                             "short_bias_bf16",
                             "short_nobias_bf16",
                             "medium_bias_bf16",
                             "medium_nobias_bf16",
                             "long_bias_bf16",
                             "long_nobias_bf16",
                             "short_bias_fp16",
                             "short_nobias_fp16",
                             "medium_bias_fp16",
                             "medium_nobias_fp16",
                             "long_bias_fp16",
                             "long_nobias_fp16",
                         ])
def test_gated_sigmoid_cute_2d(sc: Scenario):
    """Test CuTeDSL kernel with 2-D [M, K] inputs across M-ranges."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    cute_op = GatedSigmoidCuTe()

    W = torch.randn(sc.N_out, sc.K, dtype=sc.dtype, device="cuda")
    bias = torch.randn(sc.N_out, dtype=sc.dtype,
                       device="cuda") if sc.has_bias else None

    for M in sc.M_values:
        s = torch.randn(M, sc.K, dtype=sc.dtype, device="cuda")
        mha = torch.randn(M, sc.N_out, dtype=sc.dtype, device="cuda")

        ref = _ref_gated_sigmoid(s, W, mha, bias)
        out = cute_op(s, W, mha, bias)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m: f"M={M}, range={_classify_m_range(M)}: {m}")


@pytest.mark.parametrize("sc", [
    Scenario(M_values=[33, 127, 511],
             has_bias=True,
             dtype=torch.bfloat16,
             N_out=64,
             K=64),
    Scenario(M_values=[33, 127, 511],
             has_bias=False,
             dtype=torch.bfloat16,
             N_out=64,
             K=64),
    Scenario(M_values=[33, 127, 511],
             has_bias=True,
             dtype=torch.bfloat16,
             N_out=256,
             K=128),
    Scenario(M_values=[33, 127, 511],
             has_bias=False,
             dtype=torch.bfloat16,
             N_out=256,
             K=128),
],
                         ids=[
                             "N64_K64_bias",
                             "N64_K64_nobias",
                             "N256_K128_bias",
                             "N256_K128_nobias",
                         ])
def test_gated_sigmoid_cute_varied_kn(sc: Scenario):
    """Test with non-default K and N_out dimensions."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    cute_op = GatedSigmoidCuTe()

    W = torch.randn(sc.N_out, sc.K, dtype=sc.dtype, device="cuda")
    bias = torch.randn(sc.N_out, dtype=sc.dtype,
                       device="cuda") if sc.has_bias else None

    for M in sc.M_values:
        s = torch.randn(M, sc.K, dtype=sc.dtype, device="cuda")
        mha = torch.randn(M, sc.N_out, dtype=sc.dtype, device="cuda")

        ref = _ref_gated_sigmoid(s, W, mha, bias)
        out = cute_op(s, W, mha, bias)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m: f"M={M}, K={sc.K}, N={sc.N_out}: {m}")


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 255, 128),  # 3-D, odd seq
        (2, 3, 171, 128),  # 4-D, odd seq (M = 2*3*171 = 1026 → medium)
        (1, 1, 65, 128),  # 4-D, M=65 (short, odd)
        (2, 1025, 128),  # 3-D, M=2050 (long, even)
    ],
    ids=["3d_M1020", "4d_M1026", "4d_M65", "3d_M2050"])
def test_gated_sigmoid_cute_batched(shape, has_bias):
    """Test with batched (3-D/4-D) inputs; kernel flattens leading dims."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K = shape[-1]
    N_out = 128

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(*shape, dtype=dtype, device="cuda")
    mha = torch.randn(*shape[:-1], N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = cute_op(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


# ---------------------------------------------------------------------------
# Vanilla fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("M", [1, 33, 127, 512, 1025, 2049])
def test_vanilla_gated_sigmoid(M, has_bias):
    """Test vanilla PyTorch fallback against fp32 reference."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K, N_out = 128, 128

    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None
    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = _invoke_vanilla_gated_sigmoid(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


# ---------------------------------------------------------------------------
# get_gated_sigmoid_op selector test
# ---------------------------------------------------------------------------


def test_get_gated_sigmoid_op_selector():
    """Verify that get_gated_sigmoid_op returns the correct backend."""
    op_bf16 = get_gated_sigmoid_op(torch.bfloat16)
    op_fp16 = get_gated_sigmoid_op(torch.float16)
    op_fp32 = get_gated_sigmoid_op(torch.float32)

    if SM_VERSION in (80, 86, 89, 90):
        assert isinstance(op_bf16, GatedSigmoidCuTe)
        assert isinstance(op_fp16, GatedSigmoidCuTe)
    else:
        assert op_bf16 is _invoke_vanilla_gated_sigmoid
        assert op_fp16 is _invoke_vanilla_gated_sigmoid

    assert op_fp32 is _invoke_vanilla_gated_sigmoid


def test_get_gated_sigmoid_op_runs():
    """End-to-end: get_gated_sigmoid_op returns a callable that produces correct results."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K, N_out, M = 128, 128, 513

    op = get_gated_sigmoid_op(dtype)
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda")
    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = op(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


# ---------------------------------------------------------------------------
# M-range classification unit test
# ---------------------------------------------------------------------------


def test_classify_m_range():
    assert _classify_m_range(1) == "short"
    assert _classify_m_range(1024) == "short"
    assert _classify_m_range(1025) == "medium"
    assert _classify_m_range(2048) == "medium"
    assert _classify_m_range(2049) == "long"
    assert _classify_m_range(10000) == "long"


# ---------------------------------------------------------------------------
# In-place gated sigmoid tests (output aliases mha_out)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("M", [33, 127, 390, 1025, 2049])
def test_gated_sigmoid_cute_inplace_2d(M, has_bias):
    """Verify in-place (output=mha_out) produces the same result as out-of-place."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    K, N_out = 768, 768
    dtype = torch.bfloat16

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")
    mha_copy = mha.clone()

    ref = cute_op(s, W, mha, bias)
    out = cute_op(s, W, mha_copy, bias, output=mha_copy)

    assert out.data_ptr() == mha_copy.data_ptr(
    ), "in-place should reuse mha_out memory"
    torch.testing.assert_close(
        out,
        ref,
        atol=0.0,
        rtol=0.0,
        msg=lambda m: f"M={M}, in-place != out-of-place: {m}")


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("shape_s,shape_mha", [
    ((4, 390, 768), (4, 390, 768)),
    ((2, 3, 171, 768), (2, 3, 171, 768)),
],
                         ids=["3d_DiT", "4d_batched"])
def test_gated_sigmoid_cute_inplace_batched(shape_s, shape_mha, has_bias):
    """In-place with batched inputs: flatten to 2D, pass as output=mha_flat."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    K = shape_s[-1]
    N_out = shape_mha[-1]
    dtype = torch.bfloat16

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(*shape_s, dtype=dtype, device="cuda")
    mha = torch.randn(*shape_mha, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)

    mha_flat = mha.reshape(-1, N_out)
    out = cute_op(s, W, mha_flat, bias, output=mha_flat)

    assert out.data_ptr() == mha_flat.data_ptr(
    ), "in-place should reuse mha_out memory"
    out_restored = out.view(shape_mha)
    torch.testing.assert_close(out_restored,
                               ref,
                               atol=0.05,
                               rtol=1e-2,
                               msg=lambda m: f"shapes={shape_s}: {m}")
