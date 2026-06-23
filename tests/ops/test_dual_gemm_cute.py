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
"""Tests for CuTe DSL dual GEMM kernels.

* ``x0_x1`` variant: ``sigmoid(X0 @ W0.T [+ bias0]) * (X1 @ W1.T [+ bias1])``
  -- two independent X tensors, no mask.
* ``x_x`` variant: ``sigmoid(X @ W0.T [+ bias0]) * (X @ W1.T [+ bias1])``
  -- single shared X tensor, optional left-aligned mask, optional
  transposed output.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from tensorrt_bionemo._torch.custom_ops.dual_gemm_x0_x1 import DualGemmX0X1CuTe
from tensorrt_bionemo._torch.custom_ops.dual_gemm_x_x import DualGemmXxCuTe
from tests._torch import (make_left_aligned_mask, skip_if_no_cutedsl,
                          skip_if_not_sm90)


def _ref_x0_x1_dual_gemm(
    X0: torch.Tensor,
    X1: torch.Tensor,
    W0: torch.Tensor,
    W1: torch.Tensor,
    bias0: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks."""
    d0 = torch.nn.functional.linear(
        X0.float(), W0.float(),
        bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(
        X1.float(), W1.float(),
        bias1.float() if bias1 is not None else None)
    return (d0.sigmoid() * d1).to(X0.dtype)


def _ref_x_x_dual_gemm(
    X: torch.Tensor,
    W0: torch.Tensor,
    W1: torch.Tensor,
    bias0: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks.

    Mirrors :func:`tensorrt_bionemo._torch.custom_ops.dual_gemm_x_x._invoke_vanilla_dual_gemm_x_x`
    bit-for-bit (mask multiply, optional transpose) but in fp32.
    """
    d0 = torch.nn.functional.linear(
        X.float(), W0.float(),
        bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(
        X.float(), W1.float(),
        bias1.float() if bias1 is not None else None)
    ret = (d0.sigmoid() * d1).to(X.dtype)
    if mask is not None:
        ret = ret * mask.unsqueeze(-1)
    if transpose_out:
        ret = ret.moveaxis(-1, 0)
    return ret.contiguous()


@dataclass
class Scenario:
    K: int = 128
    N: int = 128
    seq_lens: list[int] = field(
        default_factory=lambda: [100, 123, 512, 1023, 1024])
    has_bias: bool = False
    has_mask: bool = False
    transpose_out: bool = False
    dtype: torch.dtype = torch.bfloat16
    atol: float = 1e-2
    rtol: float = 1e-2


@pytest.mark.parametrize("sc", [
    Scenario(N=128, K=128, dtype=torch.bfloat16),
    Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
    Scenario(N=128, K=128, dtype=torch.float16),
    Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
],
                         ids=[
                             "sc_N128_K128_b0_bf16",
                             "sc_N128_K128_b1_bf16",
                             "sc_N128_K128_b0_fp16",
                             "sc_N128_K128_b1_fp16",
                         ])
def test_x0_x1_dual_gemm(sc: Scenario):
    """Test CuTe DSL dual GEMM x0_x1 against fp32 reference."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)

    cute_op = DualGemmX0X1CuTe()

    W0 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    W1 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    bias0 = torch.randn(sc.N, dtype=sc.dtype,
                        device="cuda") if sc.has_bias else None
    bias1 = torch.randn(sc.N, dtype=sc.dtype,
                        device="cuda") if sc.has_bias else None

    for seq_len in sc.seq_lens:
        X0 = torch.randn(1, seq_len, seq_len, sc.K,
                         device="cuda").contiguous().to(sc.dtype)
        X1 = torch.randn(1, seq_len, seq_len, sc.K,
                         device="cuda").contiguous().to(sc.dtype)

        ref = _ref_x0_x1_dual_gemm(X0, X1, W0, W1, bias0, bias1)
        out = cute_op(X0, X1, W0, W1, bias0, bias1)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m: f"seq_len={seq_len}, has_bias={sc.has_bias}: {m}",
        )


# ---------------------------------------------------------------------------
# x_x variant -- single shared X, optional left-aligned mask + transpose_out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        # Plain (no mask, no bias) -- the canonical fast path.
        Scenario(N=128, K=128, seq_lens=[100], dtype=torch.bfloat16),
        Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
        Scenario(N=128, K=128, dtype=torch.float16),
        Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
        # Masked variants (left-aligned in flat I*J view so the kernel's flat
        # `actual_seqlen[b] = mask.reshape(B, I*J).sum(-1)` predicate matches
        # the vanilla `ret * mask.unsqueeze(-1)` reference bit-for-bit).
        Scenario(N=128, K=128, dtype=torch.bfloat16, has_mask=True),
        Scenario(
            N=128, K=128, dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=128, K=128, dtype=torch.float16, has_mask=True),
        # Larger N path (most production trimul uses N=256).
        Scenario(N=256, K=128, seq_lens=[100, 512], dtype=torch.bfloat16),
        Scenario(N=256,
                 K=128,
                 seq_lens=[100, 512],
                 dtype=torch.bfloat16,
                 has_bias=True,
                 has_mask=True),
        # transpose_out=True -- exercises the col-major output allocator.
        Scenario(N=128,
                 K=128,
                 seq_lens=[100, 512],
                 dtype=torch.bfloat16,
                 transpose_out=True),
        # M = B*I*J must be padded to a multiple of 8 for the CuTe epilogue.
        Scenario(N=128,
                 K=128,
                 seq_lens=[123],
                 dtype=torch.bfloat16,
                 transpose_out=True),
    ],
    ids=[
        "sc_N128_K128_b0_m0_bf16",
        "sc_N128_K128_b1_m0_bf16",
        "sc_N128_K128_b0_m0_fp16",
        "sc_N128_K128_b1_m0_fp16",
        "sc_N128_K128_b0_m1_bf16",
        "sc_N128_K128_b1_m1_bf16",
        "sc_N128_K128_b0_m1_fp16",
        "sc_N256_K128_b0_m0_bf16",
        "sc_N256_K128_b1_m1_bf16",
        "sc_N128_K128_b0_m0_bf16_t1",
        "sc_N128_K128_b0_m0_bf16_t1_m_unaligned",
    ])
def test_x_x_dual_gemm(sc: Scenario):
    """Test CuTe DSL dual GEMM x_x against fp32 reference.

    Covers the four call-site axes of :class:`DualGemmXxCuTe`:
      * dtype: bf16 and fp16
      * has_bias: optional ``[N]`` biases on both gates
      * has_mask: left-aligned ``[B, I, J]`` row mask
      * transpose_out: optional output transpose
    """
    skip_if_no_cutedsl()
    torch.manual_seed(42)

    cute_op = DualGemmXxCuTe()

    W0 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    W1 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    bias0 = torch.randn(sc.N, dtype=sc.dtype,
                        device="cuda") if sc.has_bias else None
    bias1 = torch.randn(sc.N, dtype=sc.dtype,
                        device="cuda") if sc.has_bias else None

    for seq_len in sc.seq_lens:
        X = torch.randn(1, seq_len, seq_len, sc.K,
                        device="cuda").contiguous().to(sc.dtype)
        if sc.has_mask:
            # Left-aligned in the flat (B, I*J) view -- matches the kernel's
            # `actual_seqlen[b]` semantics so the masked rows zero out
            # identically in both reference and kernel.
            flat = make_left_aligned_mask(1,
                                          seq_len * seq_len,
                                          dtype=sc.dtype,
                                          device="cuda")
            mask = flat.reshape(1, seq_len, seq_len)
        else:
            mask = None

        ref = _ref_x_x_dual_gemm(X,
                                 W0,
                                 W1,
                                 bias0,
                                 bias1,
                                 mask=mask,
                                 transpose_out=sc.transpose_out)
        out = cute_op(X,
                      W0,
                      W1,
                      bias0=bias0,
                      bias1=bias1,
                      mask=mask,
                      transpose_out=sc.transpose_out)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m:
            (f"seq_len={seq_len}, has_bias={sc.has_bias}, "
             f"has_mask={sc.has_mask}, transpose_out={sc.transpose_out}: "
             f"{m}"),
        )


def test_x_x_dual_gemm_actual_seqlen_overrides_mask():
    """Pre-computed ``actual_seqlen`` should take precedence over ``mask``.

    Asserts that passing a deliberately-wrong ``mask`` together with the
    correct ``actual_seqlen`` still produces the same output as the
    correct mask alone, confirming the wrapper short-circuits the
    ``mask.sum(-1)`` reduction when ``actual_seqlen`` is supplied.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(42)

    cute_op = DualGemmXxCuTe()

    B, I, J, K, N = 1, 64, 64, 128, 128
    dtype = torch.bfloat16
    device = "cuda"

    W0 = torch.randn(N, K, dtype=dtype, device=device)
    W1 = torch.randn(N, K, dtype=dtype, device=device)
    X = torch.randn(B, I, J, K, device=device).contiguous().to(dtype)

    flat_correct = make_left_aligned_mask(B * I, J, dtype=dtype, device=device)
    mask_correct = flat_correct.reshape(B, I, J)
    # actual_seqlen is per-row (the kernel collapses (B, I) into the
    # leading batch dim, so kernel_B = B * I).
    actual_seqlen = mask_correct.reshape(B * I, J).sum(-1).to(torch.int32)

    # All-ones "wrong" mask -- would otherwise zero nothing.
    mask_wrong = torch.ones(B, I, J, dtype=dtype, device=device)

    out_with_mask = cute_op(X, W0, W1, mask=mask_correct)
    out_with_seqlen = cute_op(X,
                              W0,
                              W1,
                              mask=mask_wrong,
                              actual_seqlen=actual_seqlen)

    torch.testing.assert_close(out_with_seqlen,
                               out_with_mask,
                               atol=0.0,
                               rtol=0.0)


# ---------------------------------------------------------------------------
# SM90 (Hopper) kernel coverage
#
# On Hopper the wrappers resolve the persistent ping-pong kernel
# (``DualGemmSm90Pingpong``) instead of the SM80/86/89 universal kernel, so on
# this hardware every test above already exercises it. The tests below add the
# coverage those don't:
#   * explicit guards that the Hopper path is selected (not a silent SM80 /
#     cuEquivariance fallback), and
#   * correctness at the larger ``S=2048`` tuned bucket (``raster_factor=4``),
#     which the ``seq_len <= 1024`` cases above never reach.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [128, 256])
def test_x_x_dual_gemm_sm90_uses_hopper_kernel(N: int):
    """The x_x wrapper must resolve the SM90 ping-pong kernel on Hopper.

    Guards against a silent fall-back to an SM80 tile (wrong calling
    convention) or to the cuEquivariance path, which would make the
    ``is_sm90`` branch in ``__call__`` dead on this hardware.
    """
    skip_if_not_sm90()
    assert DualGemmXxCuTe()._kernel_is_sm90(128, N) is True


def test_x0_x1_dual_gemm_sm90_uses_hopper_kernel():
    """The x0_x1 wrapper must resolve the SM90 ping-pong kernel on Hopper."""
    skip_if_not_sm90()
    assert DualGemmX0X1CuTe()._kernel_is_sm90(128, 128) is True


@pytest.mark.parametrize(
    "N,transpose_out",
    [(128, False), (128, True), (256, False), (256, True)],
    ids=["N128_t0", "N128_t1", "N256_t0", "N256_t1"],
)
def test_x_x_dual_gemm_large_anchor(N: int, transpose_out: bool):
    """Exercise the ``S=2048`` tuned bucket for the x_x variant.

    Uses a rectangular ``[1, 4, 2048, K]`` input so the per-sample side
    length anchor ``S = J_outer = 2048`` selects the large bucket (a
    distinct compiled kernel -- ``raster_factor=4`` on SM90) while
    ``M = 4 * 2048`` stays small enough to run cheaply.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(0)
    K, dtype = 128, torch.bfloat16

    W0 = torch.randn(N, K, dtype=dtype, device="cuda")
    W1 = torch.randn(N, K, dtype=dtype, device="cuda")
    X = torch.randn(1, 4, 2048, K, device="cuda").contiguous().to(dtype)

    ref = _ref_x_x_dual_gemm(X, W0, W1, transpose_out=transpose_out)
    out = DualGemmXxCuTe()(X, W0, W1, transpose_out=transpose_out)

    torch.testing.assert_close(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda m: f"N={N}, transpose_out={transpose_out}: {m}",
    )


def test_x0_x1_dual_gemm_large_anchor():
    """Exercise the ``S=2048`` SM90 bucket for the x0_x1 variant.

    ``S = round(sqrt(M))`` for x0_x1 (no separate ``J`` axis), so a flat
    ``[1, M, K]`` input with ``M`` just past the 512<->2048 midpoint
    (``1280**2``) lands in the large bucket.
    """
    skip_if_not_sm90()
    torch.manual_seed(0)
    K, N, dtype = 128, 128, torch.bfloat16
    M = 1300 * 1300  # sqrt == 1300 > 1280 midpoint -> S=2048 bucket

    W0 = torch.randn(N, K, dtype=dtype, device="cuda")
    W1 = torch.randn(N, K, dtype=dtype, device="cuda")
    X0 = torch.randn(1, M, K, device="cuda").to(dtype)
    X1 = torch.randn(1, M, K, device="cuda").to(dtype)

    ref = _ref_x0_x1_dual_gemm(X0, X1, W0, W1)
    out = DualGemmX0X1CuTe()(X0, X1, W0, W1)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
