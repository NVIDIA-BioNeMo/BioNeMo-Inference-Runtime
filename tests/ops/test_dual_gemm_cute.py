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
* ``x_x`` variant: configurable sigmoid or silu gate over two projections of
  one shared X tensor, with an optional left-aligned mask and transposed output.
"""

import importlib
from dataclasses import dataclass, field

import pytest
import torch

from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import (
    DualGemmX0X1CuTe,
    _invoke_cuequiv_dual_gemm_x0_x1,
    _invoke_vanilla_dual_gemm_x0_x1,
    get_dual_gemm_x0_x1_op,
)
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import ops as dual_gemm_x0_x1_ops
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import (
    DualGemmXxCuTe,
    _invoke_cuequiv_dual_gemm_x_x,
    _invoke_vanilla_dual_gemm_x_x,
    get_dual_gemm_x_x_op,
)
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import cutedsl as dual_gemm_x_x_cutedsl
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import ops as dual_gemm_x_x_ops
from bionemo_ir._torch.custom_ops.dual_gemm_x_x._cubin import DualGemmXxCubinExecutable
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelVariantUnavailable
from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from tests._torch import SM_VERSION, cutedsl_test_modes, make_left_aligned_mask, skip_if_no_cutedsl, skip_if_not_sm90

_DUAL_GEMM_XX_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.dual_gemm_x_x._source"
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


def _skip_unless_cubin_has_x_x(
    K: int,
    N: int,
    *,
    transpose_out: bool = False,
    has_bias: bool = True,
    has_mask: bool = True,
    bucket: int = 128,
    gate: str = "sigmoid",
) -> None:
    """Skip cubin-mode tuned-shape cases until the artifact-only refresh lands."""
    try:
        library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
        launcher = library.dual_gemm_x_x
        launcher.make_kernel_config(
            SM_VERSION,
            K,
            N,
            bucket,
            launcher.DType.BFLOAT16,
            transpose_out,
            has_bias,
            has_mask,
            gate == "silu",
        )
    except ImportError:
        pytest.skip("CUBIN test mode requires the _cutedsl_kernels extension")
    except ValueError:
        pytest.skip(f"dual_gemm_x_x CUBIN corpus has no SM{SM_VERSION} K={K} N={N} yet")


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
    gate: str = "sigmoid",
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks.

    Mirrors :func:`bionemo_ir._torch.custom_ops.dual_gemm_x_x._invoke_vanilla_dual_gemm_x_x`
    bit-for-bit (mask multiply, optional transpose) but in fp32.
    """
    d0 = torch.nn.functional.linear(X.float(), W0.float(), bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(X.float(), W1.float(), bias1.float() if bias1 is not None else None)
    gate_fn = torch.sigmoid if gate == "sigmoid" else torch.nn.functional.silu
    ret = (gate_fn(d0) * d1).to(X.dtype)
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


@pytest.mark.parametrize(
    ("N", "K0", "K1"),
    [
        (256, 256, 200),
        (384, 384, 200),
        (384, 384, 256),
    ],
    ids=["k0-256-k1-200-n-256", "k0-384-k1-200-n-384", "k0-384-k1-256-n-384"],
)
def test_x0_x1_asymmetric_dual_gemm(N: int, K0: int, K1: int):
    """Asymmetric output gates keep independent pair/hidden K widths."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"Asymmetric tuning requires SM80/86/89/90, got SM{SM_VERSION}")

    torch.manual_seed(42)
    dtype = torch.bfloat16
    W0 = torch.randn(N, K0, device="cuda", dtype=dtype) * 0.1
    W1 = torch.randn(N, K1, device="cuda", dtype=dtype) * 0.1
    bias0 = torch.randn(N, device="cuda", dtype=dtype) * 0.1
    bias1 = torch.randn(N, device="cuda", dtype=dtype) * 0.1

    cute_op = DualGemmX0X1CuTe()
    # Includes M smaller than one CTA, odd/non-square M tails, and batch > 1.
    for leading_shape in ((1, 1, 1), (1, 32, 32), (1, 33, 35), (2, 3, 37)):
        X0 = torch.randn(*leading_shape, K0, device="cuda", dtype=dtype) * 0.1
        X1 = torch.randn(*leading_shape, K1, device="cuda", dtype=dtype) * 0.1
        ref = _ref_x0_x1_dual_gemm(X0, X1, W0, W1, bias0, bias1)
        out = cute_op(X0, X1, W0, W1, bias0, bias1)
        torch.testing.assert_close(
            out,
            ref,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda message, shape=leading_shape: f"leading_shape={shape}: {message}",
        )


@pytest.mark.parametrize(
    ("N", "K"),
    [(392, 256), (392, 384)],
    ids=["n392-k256", "n392-k384"],
)
@pytest.mark.parametrize("transpose_out", [False, True], ids=["normal", "transposed"])
@pytest.mark.parametrize("J", [35, 1537], ids=["odd-short", "odd-long-anchor"])
def test_x_x_tuned_odd_rectangular_masked(
    N: int,
    K: int,
    transpose_out: bool,
    J: int,
) -> None:
    """Exercise odd M/J tails and per-row masks for both tuned widths."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"Tuned x_x shapes requires SM80/86/89/90, got SM{SM_VERSION}")

    torch.manual_seed(43)
    dtype = torch.bfloat16
    B, I = 2, 3
    X = torch.randn(B, I, J, K, device="cuda", dtype=dtype) * 0.1
    W0 = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
    W1 = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
    bias0 = torch.randn(N, device="cuda", dtype=dtype) * 0.1
    bias1 = torch.randn(N, device="cuda", dtype=dtype) * 0.1

    lengths = torch.tensor(
        [[J, J - 1, J // 2], [1, 0, J - 7]],
        device="cuda",
        dtype=torch.int32,
    )
    positions = torch.arange(J, device="cuda")
    mask = positions < lengths.unsqueeze(-1)

    ref = _ref_x_x_dual_gemm(X, W0, W1, bias0, bias1, mask=mask, transpose_out=transpose_out)
    out = DualGemmXxCuTe()(
        X, W0, W1, bias0=bias0, bias1=bias1, mask=mask, transpose_out=transpose_out, actual_seqlen=lengths
    )
    torch.testing.assert_close(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda message: f"B={B}, I={I}, J={J}, N={N}, K={K}, transpose_out={transpose_out}: {message}",
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
        # The N=392 trimul input projections (SM80/86/89/90).
        Scenario(N=392, K=256, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=392, K=256, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True, transpose_out=True),
        Scenario(N=392, K=384, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=392, K=384, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True, transpose_out=True),
        # Pair-512 × K=384 is tuned on all shipped SMs.
        Scenario(N=512, K=384, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=512, K=384, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True, transpose_out=True),
        # The N=512, K=512 trimul input projection, tuned on all shipped SMs.
        Scenario(N=512, K=512, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True),
        Scenario(N=512, K=512, seq_lens=[32], dtype=torch.bfloat16, has_bias=True, has_mask=True, transpose_out=True),
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
        "sc_N392_K256_xx_t0",
        "sc_N392_K256_xx_t1",
        "sc_N392_K384_xx_t0",
        "sc_N392_K384_xx_t1",
        "sc_N512_K384_xx_t0",
        "sc_N512_K384_xx_t1",
        "sc_N512_K512_xx_t0",
        "sc_N512_K512_xx_t1",
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
    if (sc.N, sc.K) in {(392, 256), (392, 384)} and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"These x_x shapes is SM80/86/89/90 only (current SM{SM_VERSION})")
    if (sc.N, sc.K) == (512, 384) and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"The N=512, K=384 x_x shape requires SM80/86/89/90 (current SM{SM_VERSION})")
    if (sc.N, sc.K) == (512, 512) and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"The N=512, K=512 x_x shape requires SM80/86/89/90 (current SM{SM_VERSION})")
    if cutedsl_mode == "cubin" and (sc.N, sc.K) in {(392, 256), (392, 384), (512, 384), (512, 512)}:
        _skip_unless_cubin_has_x_x(
            sc.K, sc.N, transpose_out=sc.transpose_out, has_bias=sc.has_bias, has_mask=sc.has_mask
        )
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


@pytest.mark.parametrize(("N", "K"), [(256, 128), (512, 256)])
@pytest.mark.parametrize("has_bias", [False, True], ids=["bias0", "bias1"])
@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_silu_gate_matches_fused_swiglu(N: int, K: int, has_bias: bool, cutedsl_mode: str, monkeypatch):
    """The silu gate must be a fused SwiGLU, not just close to one.

    ``FusedSwiGLU`` is the kernel this gate exists to absorb, so it is the
    reference that matters. It reads a packed ``[x | gate]`` buffer and gates on
    the second half, which is why ``W0`` -- the kernel's gate weight -- supplies
    the *second* half here. Both are checked against the same fp32 reference
    rather than each other, because they round differently: the dual GEMM keeps
    the whole epilogue in fp32 and rounds once at the store, while FusedSwiGLU
    rounds ``silu(gate)`` to the working dtype before multiplying. The kernel is
    therefore expected to be at least as accurate, which the last assertion
    pins down so a regression cannot hide behind a loose tolerance.
    """
    skip_if_no_cutedsl()
    if (N, K) == (512, 256) and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"Protenix (N=512,K=256) x_x CuTeDSL is SM80/86/89/90 only (current SM{SM_VERSION})")
    from bionemo_ir.dsl_kernels.triton.fused_swiglu import FusedSwiGLU

    torch.manual_seed(42)
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)
    if cutedsl_mode == "cubin":
        _skip_unless_cubin_has_x_x(K, N, has_bias=has_bias, has_mask=False, gate="silu")

    dtype, device = torch.bfloat16, "cuda"
    cute_op = DualGemmXxCuTe()
    W0 = torch.randn(N, K, dtype=dtype, device=device)
    W1 = torch.randn(N, K, dtype=dtype, device=device)
    bias0 = torch.randn(N, dtype=dtype, device=device) if has_bias else None
    bias1 = torch.randn(N, dtype=dtype, device=device) if has_bias else None
    X = torch.randn(1, 96, 96, K, device=device).contiguous().to(dtype)

    out = cute_op(X, W0, W1, bias0=bias0, bias1=bias1, gate="silu")
    ref = _ref_x_x_dual_gemm(X, W0, W1, bias0, bias1, gate="silu")
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    a0 = torch.nn.functional.linear(X, W0, bias0)
    a1 = torch.nn.functional.linear(X, W1, bias1)
    swiglu = FusedSwiGLU(d=N, three_way=False, dtype=dtype)
    packed = torch.cat([a1, a0], dim=-1).contiguous()
    torch.testing.assert_close(out, swiglu(packed), atol=2e-2, rtol=2e-2)

    ref_f32 = ref.float()
    assert (out.float() - ref_f32).abs().max() <= (swiglu(packed).float() - ref_f32).abs().max()


@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_gate_selects_distinct_math(gate: str, cutedsl_mode: str, monkeypatch):
    """Each gate must produce its own activation, and sigmoid must not move.

    The gate joins ``has_bias`` and ``has_mask`` as a compile axis, so a stale
    cache entry or a dropped keyword would silently serve the other epilogue.
    Comparing against the matching fp32 reference catches that in both
    directions.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(0)
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)
    if cutedsl_mode == "cubin":
        _skip_unless_cubin_has_x_x(128, 256, has_bias=False, has_mask=False, gate=gate)

    dtype, device = torch.bfloat16, "cuda"
    N, K = 256, 128
    cute_op = DualGemmXxCuTe()
    W0 = torch.randn(N, K, dtype=dtype, device=device)
    W1 = torch.randn(N, K, dtype=dtype, device=device)
    X = torch.randn(1, 64, 64, K, device=device).contiguous().to(dtype)

    out = cute_op(X, W0, W1, gate=gate)
    torch.testing.assert_close(out, _ref_x_x_dual_gemm(X, W0, W1, gate=gate), atol=1e-2, rtol=1e-2)

    other = "silu" if gate == "sigmoid" else "sigmoid"
    assert not torch.allclose(out, _ref_x_x_dual_gemm(X, W0, W1, gate=other), atol=1e-2, rtol=1e-2)


def test_x_x_rejects_unknown_gate(monkeypatch):
    """An unsupported gate is a caller error, not a fallback to sigmoid."""
    skip_if_no_cutedsl()
    _configure_dual_gemm_x_x_mode("source", monkeypatch)

    dtype, device = torch.bfloat16, "cuda"
    W0 = torch.randn(128, 128, dtype=dtype, device=device)
    W1 = torch.randn(128, 128, dtype=dtype, device=device)
    X = torch.randn(1, 32, 32, 128, device=device).contiguous().to(dtype)

    with pytest.raises(ValueError, match="gate"):
        DualGemmXxCuTe()(X, W0, W1, gate="gelu")


def test_x_x_rejects_silu_without_declared_tuning(monkeypatch):
    skip_if_no_cutedsl()
    _configure_dual_gemm_x_x_mode("source", monkeypatch)

    dtype, device = torch.bfloat16, "cuda"
    W0 = torch.randn(128, 128, dtype=dtype, device=device)
    W1 = torch.randn(128, 128, dtype=dtype, device=device)
    X = torch.randn(1, 32, 32, 128, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="No dual_gemm x_x silu tuning"):
        DualGemmXxCuTe()(X, W0, W1, gate="silu")


def test_x_x_rank3_uses_sequence_length_for_tuning_bucket(monkeypatch):
    skip_if_no_cutedsl()
    op = DualGemmXxCuTe()
    captured = None

    def capture_variant(variant, _x):
        nonlocal captured
        captured = variant
        return object()

    monkeypatch.setattr(op, "_get_bucket_ranges", lambda *_args: [(128, "S=128|t=0"), (512, "S=512|t=0")])
    monkeypatch.setattr(op, "_get_executable", capture_variant)
    monkeypatch.setattr(dual_gemm_x_x_cutedsl, "launch_compiled_kernel", lambda *_args: None)

    dtype = torch.bfloat16
    x = torch.empty(1, 500, 128, device="cuda", dtype=dtype)
    weight = torch.empty(128, 128, device="cuda", dtype=dtype)
    op(x, weight, weight)

    assert captured is not None
    assert captured.bucket == 512


def test_x_x_cubin_adapter_rejects_an_unimplemented_gate_before_the_library():
    """A gate no image can carry must fail before any library work.

    The registry selects on the gate, so an unknown one would otherwise reach
    the launcher as a plain lookup miss and be reported as a missing shape.
    Refusing up front keeps a source-free build's error about the gate, which
    is why this passes ``None`` for both the library and the launcher: reaching
    either would raise ``AttributeError`` instead.
    """
    with pytest.raises(CuTeDSLKernelVariantUnavailable, match="gate"):
        DualGemmXxCubinExecutable(
            None,
            None,
            90,
            K=128,
            N=512,
            bucket=512,
            dtype=torch.bfloat16,
            transpose_out=False,
            has_bias=False,
            has_mask=False,
            gate="gelu",
        )


@pytest.mark.parametrize(("gate", "is_silu"), [("sigmoid", False), ("silu", True)])
def test_x_x_cubin_adapter_forwards_gate_axis_to_launcher(gate: str, is_silu: bool):
    class FakeDType:
        FLOAT16 = 0
        BFLOAT16 = 1

    class FakeLauncher:
        DType = FakeDType

        def __init__(self):
            self.args = None

        def make_kernel_config(self, *args):
            self.args = args
            return object()

    launcher = FakeLauncher()
    DualGemmXxCubinExecutable(
        None,
        launcher,
        90,
        K=128,
        N=512,
        bucket=512,
        dtype=torch.bfloat16,
        transpose_out=False,
        has_bias=False,
        has_mask=False,
        gate=gate,
    )

    assert launcher.args is not None
    assert launcher.args[-1] is is_silu


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


# Both released checkpoints share a trimul hidden width of 196: the upstream
# trunk builds ``PairReprUpdate`` without forwarding ``tri_mult_c``, so the
# module default wins and the configured value (256, or the 2026-07-22 release
# YAML's ``25`` typo) never reaches the layer.  ``N = 2 * 196`` for the fused
# input projection; ``K`` is the checkpoint's pair width.
@pytest.mark.parametrize(("N", "K"), [(392, 256), (392, 384), (512, 384)])
def test_x_x_tuned_shapes_dispatch_cute_on_supported_sms(N: int, K: int):
    skip_if_no_cutedsl()
    supported_sms = (80, 86, 89, 90)
    if SM_VERSION not in supported_sms:
        pytest.skip(f"shape N={N}, K={K} requires one of {supported_sms} (current SM{SM_VERSION})")
    op = get_dual_gemm_x_x_op(torch.bfloat16, N=N, K=K, pair_mask_left_aligned=True)
    assert op.__name__ == "_invoke_cute_dual_gemm_x_x"


def test_x_x_pair_swiglu_shape_is_reachable_only_through_the_silu_gate():
    """``(N=512, K=128)`` is a SwiGLU-only admission, not a general one.

    The shape has tuned silu coverage on SM80/86/89/90. Admitting it for
    sigmoid too would move trimul callers of that width onto a backend nobody
    measured them on, so the dispatcher must keep the old answer there.
    """
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"supported on SM80/86/89/90 only (current SM{SM_VERSION})")
    silu = get_dual_gemm_x_x_op(torch.bfloat16, N=512, K=128, gate="silu")
    assert silu.__name__ == "_invoke_cute_dual_gemm_x_x"
    sigmoid = get_dual_gemm_x_x_op(torch.bfloat16, N=512, K=128, gate="sigmoid")
    assert sigmoid.__name__ != "_invoke_cute_dual_gemm_x_x"


@pytest.mark.parametrize("cutedsl_mode", _DUAL_GEMM_XX_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_x_x_pair_swiglu_shape_matches_reference(cutedsl_mode: str, monkeypatch):
    """The K128_N512 tiles must compute the admitted SwiGLU."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"supported on SM80/86/89/90 only (current SM{SM_VERSION})")
    _configure_dual_gemm_x_x_mode(cutedsl_mode, monkeypatch)
    if cutedsl_mode == "cubin":
        _skip_unless_cubin_has_x_x(128, 512, has_bias=False, has_mask=False, bucket=256, gate="silu")
    torch.manual_seed(0)
    N, K, S = 512, 128, 256
    X = torch.randn(1, S, S, K, device="cuda", dtype=torch.bfloat16) * 0.1
    W0 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    W1 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    op = get_dual_gemm_x_x_op(torch.bfloat16, N=N, K=K, gate="silu")
    out = op(X, W0, W1, gate="silu")
    assert out.shape == (1, S, S, N)
    torch.testing.assert_close(out, _ref_x_x_dual_gemm(X, W0, W1, gate="silu"), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("K", "N"),
    [
        (64, 128),
        (64, 256),
        (128, 256),
        (128, 512),
        (256, 512),
        (256, 1024),
        (384, 768),
        (384, 1536),
        (768, 1536),
    ],
)
def test_x_x_model_swiglu_shapes_dispatch_cute_on_supported_sms(K: int, N: int):
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"supported on SM80/86/89/90 only (current SM{SM_VERSION})")
    op = get_dual_gemm_x_x_op(torch.bfloat16, N=N, K=K, gate="silu")
    assert op.__name__ == "_invoke_cute_dual_gemm_x_x"


def test_x0_x1_protenix_shape_dispatches_cute_on_sm80_sm86_sm89_sm90():
    """Protenix ``(N=256, K=256)`` must hit CuTeDSL on SM80/86/89/90."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"requires SM80/86/89/90 (current SM{SM_VERSION})")
    op = get_dual_gemm_x0_x1_op(torch.bfloat16, N=256, K=256)
    assert op.__name__ == "_invoke_cute_dual_gemm_x0_x1"


@pytest.mark.parametrize("sm", [80, 86, 90])
@pytest.mark.parametrize(
    ("N", "K0", "K1"),
    [(64, 64, 64), (256, 128, 128), (512, 512, 256)],
    ids=["template-64", "legacy-128x256", "n512-k0-512-k1-256"],
)
def test_x0_x1_new_shapes_dispatch_cute_on_tuned_sms(
    monkeypatch: pytest.MonkeyPatch,
    sm: int,
    N: int,
    K0: int,
    K1: int,
) -> None:
    monkeypatch.setattr(dual_gemm_x0_x1_ops, "get_sm_version", lambda: sm)
    op = get_dual_gemm_x0_x1_op(torch.bfloat16, N=N, K0=K0, K1=K1)
    assert op.__name__ == "_invoke_cute_dual_gemm_x0_x1"


@pytest.mark.parametrize("sm", [80, 86, 89, 90])
def test_x_x_z12_shape_dispatches_cute_on_tuned_sms(monkeypatch: pytest.MonkeyPatch, sm: int) -> None:
    monkeypatch.setattr(dual_gemm_x_x_ops, "get_sm_version", lambda: sm)
    op = get_dual_gemm_x_x_op(torch.bfloat16, N=512, K=512, gate="sigmoid")
    assert op.__name__ == "_invoke_cute_dual_gemm_x_x"


@pytest.mark.parametrize(
    ("N", "K0", "K1"),
    [(64, 64, 64), (256, 128, 128), (512, 512, 256)],
    ids=["template-64", "legacy-128x256", "n512-k0-512-k1-256"],
)
def test_x0_x1_additional_shapes_dispatch_cute_on_sm89(
    monkeypatch: pytest.MonkeyPatch,
    N: int,
    K0: int,
    K1: int,
) -> None:
    sm = 89
    monkeypatch.setattr(dual_gemm_x0_x1_ops, "get_sm_version", lambda: sm)
    op = get_dual_gemm_x0_x1_op(torch.bfloat16, N=N, K0=K0, K1=K1)
    assert op.__name__ == "_invoke_cute_dual_gemm_x0_x1"


@pytest.mark.parametrize(
    ("N", "K0", "K1"),
    # The trimul zero-extends its raw 196 hidden width to 200 before dispatch,
    # so every shipped asymmetric shape keeps 128-bit copies.
    [(256, 256, 200), (384, 384, 200), (384, 384, 256)],
)
def test_x0_x1_asymmetric_shapes_dispatch_cute_on_supported_sms(N: int, K0: int, K1: int):
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"requires SM80/86/89/90 (current SM{SM_VERSION})")
    op = get_dual_gemm_x0_x1_op(torch.bfloat16, N=N, K0=K0, K1=K1)
    assert op.__name__ == "_invoke_cute_dual_gemm_x0_x1"


@pytest.mark.parametrize(
    ("sm", "N", "K0", "K1", "transpose_out"),
    [
        (90, 256, 256, 200, True),
        (120, 256, 128, 64, False),
    ],
    ids=["transposed-cute-shape", "cuequiv-fallback-shape"],
)
def test_x0_x1_asymmetric_cuequiv_fallback_dispatches_vanilla(
    monkeypatch: pytest.MonkeyPatch,
    sm: int,
    N: int,
    K0: int,
    K1: int,
    transpose_out: bool,
) -> None:
    monkeypatch.setattr(dual_gemm_x0_x1_ops, "get_sm_version", lambda: sm)
    op = get_dual_gemm_x0_x1_op(
        torch.bfloat16,
        transpose_out=transpose_out,
        N=N,
        K0=K0,
        K1=K1,
    )
    assert op is _invoke_vanilla_dual_gemm_x0_x1


def test_x_x_protenix_sm90_uses_hopper_kernel():
    """Protenix x_x ``(512, 256)`` must resolve the SM90 ping-pong kernel."""
    skip_if_not_sm90()
    assert DualGemmXxCuTe()._kernel_is_sm90(256, 512) is True


def test_x0_x1_protenix_sm90_uses_hopper_kernel():
    """Protenix x0_x1 ``(256, 256)`` must resolve the SM90 ping-pong kernel."""
    skip_if_not_sm90()
    assert DualGemmX0X1CuTe()._kernel_is_sm90(256, 256) is True


@pytest.mark.parametrize(("K", "N"), [(256, 392), (384, 392), (384, 512)])
def test_x_x_tuned_sm90_uses_hopper_kernel(K: int, N: int):
    """The tuned x_x widths must resolve the SM90 ping-pong kernel on Hopper."""
    skip_if_not_sm90()
    assert DualGemmXxCuTe()._kernel_is_sm90(K, N) is True


@pytest.mark.parametrize(("K", "K1", "N"), [(256, 200, 256), (384, 200, 384), (384, 256, 384)])
def test_x0_x1_asymmetric_sm90_uses_hopper_kernel(K: int, K1: int, N: int):
    """The asymmetric x0_x1 widths must resolve the SM90 ping-pong kernel on Hopper."""
    skip_if_not_sm90()
    assert DualGemmX0X1CuTe()._kernel_is_sm90(K, N, K1) is True


@pytest.mark.parametrize(
    ("N", "K", "expected"),
    [
        (128, 128, _invoke_cuequiv_dual_gemm_x0_x1),
        (256, 256, _invoke_vanilla_dual_gemm_x0_x1),
    ],
)
def test_x0_x1_selector_rejects_unshipped_targets(monkeypatch, N: int, K: int, expected):
    """Off the tuned SMs the CuTe path is never selected, whatever the shape."""
    monkeypatch.setattr(dual_gemm_x0_x1_ops, "get_sm_version", lambda: 120)
    assert get_dual_gemm_x0_x1_op(torch.bfloat16, N=N, K=K) is expected


@pytest.mark.parametrize(
    ("N", "K", "expected"),
    [
        (128, 128, _invoke_cuequiv_dual_gemm_x_x),
        (512, 256, _invoke_vanilla_dual_gemm_x_x),
    ],
)
def test_x_x_selector_rejects_unshipped_targets(monkeypatch, N: int, K: int, expected):
    """Off the tuned SMs the CuTe path is never selected, whatever the shape."""
    monkeypatch.setattr(dual_gemm_x_x_ops, "get_sm_version", lambda: 120)
    assert get_dual_gemm_x_x_op(torch.bfloat16, N=N, K=K) is expected


def test_tuned_dual_gemms_are_cudagraph_safe():
    """TVM-FFI launches must be recorded on the capture stream."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"The tuned shapes requires SM80/86/89/90, got SM{SM_VERSION}")

    torch.manual_seed(42)
    dtype = torch.bfloat16
    length = 32

    def capture_and_compare(op, inputs, changed_indices):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        static_inputs = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = op(*static_inputs)

        for index in changed_indices:
            static_inputs[index].add_(0.125)
        graph.replay()
        replay = captured.clone()
        eager = op(*static_inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(replay, eager, atol=0, rtol=0)

    x0 = torch.randn(1, length, length, 256, device="cuda", dtype=dtype)
    x1 = torch.randn(1, length, length, 200, device="cuda", dtype=dtype)
    x1[..., 196:] = 0
    w0 = torch.randn(256, 256, device="cuda", dtype=dtype) * 0.05
    w1 = torch.randn(256, 200, device="cuda", dtype=dtype) * 0.05
    w1[:, 196:] = 0
    bias0 = torch.randn(256, device="cuda", dtype=dtype) * 0.05
    bias1 = torch.randn(256, device="cuda", dtype=dtype) * 0.05
    capture_and_compare(
        DualGemmX0X1CuTe(),
        (x0, x1, w0, w1, bias0, bias1),
        (0, 1),
    )

    x = torch.randn(1, length, length, 256, device="cuda", dtype=dtype)
    w0 = torch.randn(392, 256, device="cuda", dtype=dtype) * 0.05
    w1 = torch.randn(392, 256, device="cuda", dtype=dtype) * 0.05
    bias0 = torch.randn(392, device="cuda", dtype=dtype) * 0.05
    bias1 = torch.randn(392, device="cuda", dtype=dtype) * 0.05
    mask = torch.ones(1, length, length, device="cuda", dtype=torch.bool)
    capture_and_compare(
        DualGemmXxCuTe(),
        (x, w0, w1, bias0, bias1, mask),
        (0,),
    )


# ---------------------------------------------------------------------------
# SM90 (Hopper) kernel coverage
#
# On Hopper the wrappers resolve the persistent ping-pong kernel instead of the
# SM80/86/89 universal kernel, so on this hardware every test above already
# exercises it. The tests below add the coverage those don't:
#   * explicit guards that the Hopper path is selected (not a silent SM80 /
#     cuEquivariance fallback), and
#   * correctness at the larger ``S=2048`` tuned bucket (``raster_factor=4``),
#     which the ``seq_len <= 1024`` cases above never reach.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [128, 256, 392])
def test_x_x_dual_gemm_sm90_uses_hopper_kernel(N: int):
    """The x_x wrapper must resolve the SM90 ping-pong kernel on Hopper.

    Guards against a silent fall-back to an SM80 tile (wrong calling
    convention) or to the cuEquivariance path, which would make the
    ``is_sm90`` branch in ``__call__`` dead on this hardware.
    """
    skip_if_not_sm90()
    k = {128: 128, 256: 128, 392: 256}[N]
    assert DualGemmXxCuTe()._kernel_is_sm90(k, N) is True


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
