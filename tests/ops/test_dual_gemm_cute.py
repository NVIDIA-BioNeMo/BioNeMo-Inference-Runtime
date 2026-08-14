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

import importlib
from dataclasses import dataclass, field

import pytest
import torch

from bionemo_ir._torch import _cutedsl_kernel_library as library_runtime
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import DualGemmX0X1CuTe, get_dual_gemm_x0_x1_op
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import DualGemmXxCuTe, get_dual_gemm_x_x_op
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import cutedsl as dual_gemm_x_x_cutedsl
from tests._torch import SM_VERSION, cutedsl_test_modes, make_left_aligned_mask, skip_if_no_cutedsl, skip_if_not_sm90

_DUAL_GEMM_XX_SOURCE_MODULE = "bionemo_ir.dsl_kernels.cute.sm80_dual_gemm_x_x_k128"
_DUAL_GEMM_XX_TEST_MODES = cutedsl_test_modes(_DUAL_GEMM_XX_SOURCE_MODULE)
_DUAL_GEMM_XX_MODE_CACHES: dict[str, dict] = {
    "source": {},
    "cubin": {},
}


def _configure_dual_gemm_x_x_mode(mode: str, monkeypatch) -> None:
    """Force one x_x implementation without allowing silent fallback."""
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(DualGemmXxCuTe, "_compiled_cache", _DUAL_GEMM_XX_MODE_CACHES[mode])

    if mode == "cubin":
        try:
            importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
        except ImportError:
            pytest.fail("CUBIN test mode requires the _cutedsl_kernels extension")

        try:
            source_module = importlib.import_module("bionemo_ir._torch.custom_ops.dual_gemm_x_x._source")
        except ImportError:
            source_module = None
        if source_module is not None:

            def source_unavailable(_implementation):
                raise ModuleNotFoundError("CuTeDSL source disabled by CUBIN test mode")

            monkeypatch.setattr(source_module, "resolve_implementation", source_unavailable)
        monkeypatch.setattr(library_runtime, "_kernel_library", None)
        return

    def reject_cubin_fallback(*_args, **_kwargs):
        raise AssertionError("source test mode unexpectedly fell back to the CUBIN library")

    monkeypatch.setattr(dual_gemm_x_x_cutedsl, "populate_compiled_cache_from_library", reject_cubin_fallback)


def _ref_x0_x1_dual_gemm(
    X0: torch.Tensor,
    X1: torch.Tensor,
    W0: torch.Tensor,
    W1: torch.Tensor,
    bias0: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks."""
    d0 = torch.nn.functional.linear(X0.float(), W0.float(), bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(X1.float(), W1.float(), bias1.float() if bias1 is not None else None)
    return (d0.sigmoid() * d1).to(X0.dtype)


def _ref_x_x_dual_gemm(
    X: torch.Tensor,
    W0: torch.Tensor,
    W1: torch.Tensor,
    bias0: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks.

    Mirrors :func:`bionemo_ir._torch.custom_ops.dual_gemm_x_x._invoke_vanilla_dual_gemm_x_x`
    bit-for-bit (mask multiply, optional transpose) but in fp32.
    """
    d0 = torch.nn.functional.linear(X.float(), W0.float(), bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(X.float(), W1.float(), bias1.float() if bias1 is not None else None)
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
    seq_lens: list[int] = field(default_factory=lambda: [100, 123, 512, 1023, 1024])
    has_bias: bool = False
    has_mask: bool = False
    transpose_out: bool = False
    dtype: torch.dtype = torch.bfloat16
    atol: float = 1e-2
    rtol: float = 1e-2


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(N=128, K=128, dtype=torch.bfloat16),
        Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
        Scenario(N=128, K=128, dtype=torch.float16),
        Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
        # ProtenixV2 trimul out-proj (c_z=256) -- CuTeDSL on SM80/86/89/90.
        # ``|b=1`` reuses the tuned ``|b=0`` tiles (no separate bias tune).
        Scenario(N=256, K=256, seq_lens=[100, 256], dtype=torch.bfloat16),
        Scenario(N=256, K=256, seq_lens=[100, 256], dtype=torch.bfloat16, has_bias=True),
    ],
    ids=[
        "sc_N128_K128_b0_bf16",
        "sc_N128_K128_b1_bf16",
        "sc_N128_K128_b0_fp16",
        "sc_N128_K128_b1_fp16",
        "sc_N256_K256_b0_bf16",
        "sc_N256_K256_b1_bf16",
    ],
)
def test_x0_x1_dual_gemm(sc: Scenario):
    """Test CuTe DSL dual GEMM x0_x1 against fp32 reference."""
    skip_if_no_cutedsl()
    if (sc.N, sc.K) == (256, 256) and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"Protenix (N=256,K=256) x0_x1 CuTeDSL is SM80/86/89/90 only (current SM{SM_VERSION})")
    torch.manual_seed(42)

    cute_op = DualGemmX0X1CuTe()

    W0 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    W1 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    bias0 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None
    bias1 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None

    for seq_len in sc.seq_lens:
        X0 = torch.randn(1, seq_len, seq_len, sc.K, device="cuda").contiguous().to(sc.dtype)
        X1 = torch.randn(1, seq_len, seq_len, sc.K, device="cuda").contiguous().to(sc.dtype)

        ref = _ref_x0_x1_dual_gemm(X0, X1, W0, W1, bias0, bias1)
        out = cute_op(X0, X1, W0, W1, bias0, bias1)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m, seq_len=seq_len: f"seq_len={seq_len}, has_bias={sc.has_bias}: {m}",
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
        Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=128, K=128, dtype=torch.float16, has_mask=True),
        # Larger N path (most production trimul uses N=256).
        Scenario(N=256, K=128, seq_lens=[100, 512], dtype=torch.bfloat16),
        Scenario(N=256, K=128, seq_lens=[100, 512], dtype=torch.bfloat16, has_bias=True, has_mask=True),
        # ProtenixV2 trimul (c_z=256, hidden=256) -- CuTeDSL on SM80/86/89/90.
        Scenario(N=512, K=256, seq_lens=[100, 256], dtype=torch.bfloat16),
        Scenario(N=512, K=256, seq_lens=[100, 256], dtype=torch.bfloat16, has_mask=True),
        Scenario(N=512, K=256, seq_lens=[100, 256], dtype=torch.bfloat16, transpose_out=True),
        Scenario(N=512, K=256, seq_lens=[100, 256], dtype=torch.bfloat16, has_mask=True, transpose_out=True),
        # transpose_out=True -- exercises the col-major output allocator.
        Scenario(N=128, K=128, seq_lens=[100, 512], dtype=torch.bfloat16, transpose_out=True),
        # M = B*I*J must be padded to a multiple of 8 for the CuTe epilogue.
        Scenario(N=128, K=128, seq_lens=[123], dtype=torch.bfloat16, transpose_out=True),
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
        "sc_N512_K256_b0_m0_bf16",
        "sc_N512_K256_b0_m1_bf16",
        "sc_N512_K256_b0_m0_bf16_t1",
        "sc_N512_K256_b0_m1_bf16_t1",
        "sc_N128_K128_b0_m0_bf16_t1",
        "sc_N128_K128_b0_m0_bf16_t1_m_unaligned",
    ],
)
@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_dual_gemm(sc: Scenario, cutedsl_mode: str, monkeypatch):
    """Test CuTe DSL dual GEMM x_x against fp32 reference.

    Covers the four call-site axes of :class:`DualGemmXxCuTe`:
      * dtype: bf16 and fp16
      * has_bias: optional ``[N]`` biases on both gates
      * has_mask: left-aligned ``[B, I, J]`` row mask
      * transpose_out: optional output transpose
    """
    skip_if_no_cutedsl()
    if (sc.N, sc.K) == (512, 256) and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"Protenix (N=512,K=256) x_x CuTeDSL is SM80/86/89/90 only (current SM{SM_VERSION})")
    torch.manual_seed(42)
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)

    cute_op = DualGemmXxCuTe()

    W0 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    W1 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    bias0 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None
    bias1 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None

    for seq_len in sc.seq_lens:
        X = torch.randn(1, seq_len, seq_len, sc.K, device="cuda").contiguous().to(sc.dtype)
        if sc.has_mask:
            # Left-aligned in the flat (B, I*J) view -- matches the kernel's
            # `actual_seqlen[b]` semantics so the masked rows zero out
            # identically in both reference and kernel.
            flat = make_left_aligned_mask(1, seq_len * seq_len, dtype=sc.dtype, device="cuda")
            mask = flat.reshape(1, seq_len, seq_len)
        else:
            mask = None

        ref = _ref_x_x_dual_gemm(X, W0, W1, bias0, bias1, mask=mask, transpose_out=sc.transpose_out)
        out = cute_op(X, W0, W1, bias0=bias0, bias1=bias1, mask=mask, transpose_out=sc.transpose_out)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m, seq_len=seq_len: (
                f"seq_len={seq_len}, has_bias={sc.has_bias}, "
                f"has_mask={sc.has_mask}, transpose_out={sc.transpose_out}: "
                f"{m}"
            ),
        )


@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_dual_gemm_actual_seqlen_overrides_mask(cutedsl_mode: str, monkeypatch):
    """Pre-computed ``actual_seqlen`` should take precedence over ``mask``.

    Asserts that passing a deliberately-wrong ``mask`` together with the
    correct ``actual_seqlen`` still produces the same output as the
    correct mask alone, confirming the wrapper short-circuits the
    ``mask.sum(-1)`` reduction when ``actual_seqlen`` is supplied.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)

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
    out_with_seqlen = cute_op(X, W0, W1, mask=mask_wrong, actual_seqlen=actual_seqlen)

    torch.testing.assert_close(out_with_seqlen, out_with_mask, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# ProtenixV2 (N, K) dispatcher coverage
#
# ``(512, 256)`` / ``(256, 256)`` are gated to SM80/SM86/SM89/SM90. On those
# SMs the dispatcher must select the CuTe path; elsewhere it must fall back
# (vanilla / cuequiv) rather than try to load a missing JSON.
# ---------------------------------------------------------------------------


def test_x_x_protenix_shape_dispatches_cute_on_sm80_sm86_sm89_sm90():
    """Protenix ``(N=512, K=256)`` must hit CuTeDSL on SM80/86/89/90."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"requires SM80/86/89/90 (current SM{SM_VERSION})")
    op = get_dual_gemm_x_x_op(torch.bfloat16, N=512, K=256)
    assert op.__name__ == "_invoke_cute_dual_gemm_x_x"


def test_x0_x1_protenix_shape_dispatches_cute_on_sm80_sm86_sm89_sm90():
    """Protenix ``(N=256, K=256)`` must hit CuTeDSL on SM80/86/89/90."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"requires SM80/86/89/90 (current SM{SM_VERSION})")
    op = get_dual_gemm_x0_x1_op(torch.bfloat16, N=256, K=256)
    assert op.__name__ == "_invoke_cute_dual_gemm_x0_x1"


def test_x_x_protenix_sm90_uses_hopper_kernel():
    """Protenix x_x ``(512, 256)`` must resolve the SM90 ping-pong kernel."""
    skip_if_not_sm90()
    assert DualGemmXxCuTe()._kernel_is_sm90(256, 512) is True


def test_x0_x1_protenix_sm90_uses_hopper_kernel():
    """Protenix x0_x1 ``(256, 256)`` must resolve the SM90 ping-pong kernel."""
    skip_if_not_sm90()
    assert DualGemmX0X1CuTe()._kernel_is_sm90(256, 256) is True


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
@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_dual_gemm_large_anchor(
    N: int,
    transpose_out: bool,
    cutedsl_mode: str,
    monkeypatch,
):
    """Exercise the ``S=2048`` tuned bucket for the x_x variant.

    Uses a rectangular ``[1, 4, 2048, K]`` input so the per-sample side
    length anchor ``S = J_outer = 2048`` selects the large bucket (a
    distinct compiled kernel -- ``raster_factor=4`` on SM90) while
    ``M = 4 * 2048`` stays small enough to run cheaply.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(0)
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)
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


@pytest.mark.skipif(
    set(_DUAL_GEMM_XX_TEST_MODES) != {"source", "cubin"},
    reason="both implementations are required for a bitwise equivalence check",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True], ids=["bias0", "bias1"])
@pytest.mark.parametrize("has_mask", [False, True], ids=["mask0", "mask1"])
@pytest.mark.parametrize("transpose_out", [False, True], ids=["t0", "t1"])
def test_x_x_source_and_cubin_agree_bitwise(
    dtype: torch.dtype,
    has_bias: bool,
    has_mask: bool,
    transpose_out: bool,
    monkeypatch,
):
    """The direct launcher must reproduce the source host launch exactly."""
    skip_if_no_cutedsl()
    torch.manual_seed(123)
    batch, rows, cols, K, N = 1, 4, 31, 128, 128
    x = torch.randn(batch, rows, cols, K, dtype=dtype, device="cuda")
    w0 = torch.randn(N, K, dtype=dtype, device="cuda")
    w1 = torch.randn_like(w0)
    bias0 = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None
    bias1 = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None
    mask = make_left_aligned_mask(batch, rows, cols, dtype=dtype, device="cuda") if has_mask else None

    with monkeypatch.context() as source_patch:
        _configure_dual_gemm_x_x_mode("source", source_patch)
        source = DualGemmXxCuTe()(x, w0, w1, bias0=bias0, bias1=bias1, mask=mask, transpose_out=transpose_out)
    with monkeypatch.context() as cubin_patch:
        _configure_dual_gemm_x_x_mode("cubin", cubin_patch)
        cubin = DualGemmXxCuTe()(x, w0, w1, bias0=bias0, bias1=bias1, mask=mask, transpose_out=transpose_out)

    torch.testing.assert_close(cubin, source, atol=0.0, rtol=0.0)


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
