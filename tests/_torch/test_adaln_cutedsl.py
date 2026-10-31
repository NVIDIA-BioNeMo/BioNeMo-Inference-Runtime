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

"""Numerical tests for the CuTeDSL backend of AdaLN.

Compares the kernel-backed LayerNorm and RMSNorm variants against the torch
fallback (obtained by clearing ``_fused_op`` after construction) and exercises
the underlying custom op directly.
"""

import os
import warnings
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch

from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid import (
    SHIPPED_N,
    AdaLNLayerNormSigmoidCuTe,
    bucket_variants,
    get_adaln_layernorm_sigmoid_op,
    select_bucket,
)
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid import cutedsl as adaln_cutedsl
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid import ops as adaln_ops
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import (
    SUPPORTED_SMS as ADALN_SMS,
)
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import (
    config_identity as adaln_config_identity,
)
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._cubin import (
    AdaLNLayerNormSigmoidCubinExecutable,
)
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid.ops import (
    _invoke_vanilla_adaln_layernorm_sigmoid,
)
from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    cute_numeric,
)
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    init_module_weights,
    require_cubin_family,
    run_cutedsl_test_mode,
)

_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._source"
_CUTEDSL_MODES = cutedsl_test_modes(_SOURCE_MODULE)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    torch_dtype: str = "float32"
    seq_len: int = 128
    batch_size: int = 1


def _torch_adaln_layernorm_sigmoid(x, s_scale, s_bias, eps=1e-5, rms_norm=False):
    """Inline torch reference for ``sigmoid(s_scale) * Norm(x) + s_bias``."""
    N = x.shape[-1]
    normed = (
        torch.nn.functional.rms_norm(x, (N,), eps=eps) if rms_norm else torch.nn.functional.layer_norm(x, (N,), eps=eps)
    )
    return torch.sigmoid(s_scale) * normed + s_bias


def _skip_without_adaln_cubin() -> None:
    """Skip unless AdaLN ships a CUBIN for this GPU.

    Where it does not, AdaLN falls back to the torch
    path by design, so a test that requires the fused kernel is asserting
    something the support matrix never claimed.
    """
    if SM_VERSION not in ADALN_SMS:
        pytest.skip(f"AdaLN CUBINs do not target SM{SM_VERSION}")


def _init_adaln_weights(m: AdaLN) -> None:
    """Initialize the custom Linear inside AdaLN.

    ``Linear.create_weights`` uses ``torch.empty`` and expects callers to load
    real weights via ``load_weights``. If we skip that, the weights are
    uninitialized garbage and can contain NaN/Inf left over from prior CUDA
    allocations — which silently corrupts test results when forward is run.
    """
    with torch.no_grad():
        torch.nn.init.normal_(m.fused_s_scale_s_bias.weight, std=0.02)
        if m.fused_s_scale_s_bias.bias is not None:
            torch.nn.init.zeros_(m.fused_s_scale_s_bias.bias)
        # s_norm.weight is initialized to 1 by both normalization modules.


def _copy_weights(src: AdaLN, dst: AdaLN) -> None:
    """Mirror weights from ``src`` into ``dst`` (both AdaLN, same shapes)."""
    src_sd = src.state_dict()
    dst.load_state_dict(src_sd, strict=True)


def _max_rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return (diff / denom).max().item()


# ---------------------------------------------------------------------------
# AdaLN module-level: CuteDSL backend vs torch backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim=768, dim_single_cond=768, torch_dtype="float32"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
        Scenario(dim=384, dim_single_cond=384, torch_dtype="float32"),
        Scenario(dim=384, dim_single_cond=384, torch_dtype="bfloat16"),
        Scenario(dim=512, dim_single_cond=512, torch_dtype="float32"),
        Scenario(dim=512, dim_single_cond=512, torch_dtype="bfloat16"),
        Scenario(dim=1024, dim_single_cond=1024, torch_dtype="float32"),
        Scenario(dim=1024, dim_single_cond=1024, torch_dtype="bfloat16"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", batch_size=4),
    ],
)
@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
def test_adaln_cutedsl_matches_torch(sc: Scenario, norm_type: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    adaln_ref = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=dtype, norm_type=norm_type).to(device)
    # Force the inline torch path on the reference module.
    adaln_ref._fused_op = None
    _init_adaln_weights(adaln_ref)
    adaln_cute = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=dtype, norm_type=norm_type).to(device)
    _copy_weights(adaln_ref, adaln_cute)

    a = torch.randn(sc.batch_size, sc.seq_len, sc.dim, dtype=dtype, device=device)
    s = torch.randn(sc.batch_size, sc.seq_len, sc.dim_single_cond, dtype=dtype, device=device)

    # The CuteDSL backend is in-place on ``a``. Run each backend on its own
    # copy so the snapshots stay independent.
    a_ref = a.clone()
    a_cute = a.clone()
    a_fp32 = a.float()

    with torch.inference_mode():
        out_ref = adaln_ref.forward(a_ref, s)
        out_cute = adaln_cute.forward(a_cute, s)

    assert out_ref.shape == out_cute.shape
    assert out_ref.dtype == out_cute.dtype

    if dtype == torch.float32:
        torch.testing.assert_close(out_cute, out_ref, atol=1e-4, rtol=1e-4)
    else:
        # bf16: compare both backends against an fp32 ground truth.
        adaln_fp32 = AdaLN(
            dim=sc.dim,
            dim_single_cond=sc.dim_single_cond,
            dtype=torch.float32,
            norm_type=norm_type,
        ).to(device)
        adaln_fp32.load_state_dict({k: v.float() for k, v in adaln_ref.state_dict().items()}, strict=True)
        with torch.inference_mode():
            out_fp32 = adaln_fp32.forward(a_fp32, s.float())

        err_ref = (out_ref.float() - out_fp32).abs().max().item()
        err_cute = (out_cute.float() - out_fp32).abs().max().item()
        # Allow 2x the torch backend's quantization error.
        assert err_cute < max(err_ref * 2.0, 0.05), f"CuTe err {err_cute:.3e} too large vs torch err {err_ref:.3e}"


# ---------------------------------------------------------------------------
# Custom op: CuteDSL kernel vs vanilla python reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "dtype,M,N",
    [
        (torch.float16, 33, 128),
        (torch.bfloat16, 300, 768),
        (torch.float32, 2050, 1024),
    ],
    ids=["fp16_small", "bf16_second_bucket", "fp32_big_m"],
)
def test_adaln_source_and_cubin(mode, rms_norm, dtype, M, N, monkeypatch):
    _skip_without_adaln_cubin()
    torch.manual_seed(11)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    scale = torch.randn(M, N, device="cuda", dtype=dtype)
    bias = torch.randn(M, N, device="cuda", dtype=dtype)

    result = run_cutedsl_test_mode(
        mode,
        monkeypatch,
        AdaLNLayerNormSigmoidCuTe,
        adaln_cutedsl,
        lambda: AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)(x.clone(), scale, bias),
    )
    reference = _invoke_vanilla_adaln_layernorm_sigmoid(x.clone(), scale, bias, rms_norm=rms_norm)
    torch.testing.assert_close(result, reference, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "dtype,x_shape,s_shape",
    [
        (torch.bfloat16, (2, 37, 768), (2, 37, 768)),
        (torch.float16, (3, 64, 128), (3, 64, 128)),
        (torch.float32, (2, 3, 21, 256), (2, 1, 21, 256)),
    ],
    ids=["bf16", "fp16", "fp32_sample_broadcast"],
)
def test_adaln_mask_zeroes_its_rows_in_the_same_launch(
    mode: str,
    rms_norm: bool,
    dtype: torch.dtype,
    x_shape: tuple[int, ...],
    s_shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose s_scale row is masked out comes out +0, even from a non-finite input, and every other row keeps
    the maskless op's bits: the op with a mask is the op followed by ``masked_fill``."""
    _skip_without_adaln_cubin()
    torch.manual_seed(13)
    x = torch.randn(x_shape, device="cuda", dtype=dtype)
    scale = torch.randn(s_shape, device="cuda", dtype=dtype)
    bias = torch.randn(s_shape, device="cuda", dtype=dtype)
    # The caller's own form: [B, I], without the broadcast sample axis that s_scale carries as a 1.
    mask = torch.rand([d for d in s_shape[:-1] if d != 1], device="cuda") > 0.3
    excluded_rows = ~mask.reshape(s_shape[:-1]).unsqueeze(-1)
    x.masked_fill_(excluded_rows, float("nan"))
    op = AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)

    masked = run_cutedsl_test_mode(
        mode, monkeypatch, AdaLNLayerNormSigmoidCuTe, adaln_cutedsl, lambda: op(x.clone(), scale, bias, mask=mask)
    )
    unmasked = run_cutedsl_test_mode(
        mode, monkeypatch, AdaLNLayerNormSigmoidCuTe, adaln_cutedsl, lambda: op(x.clone(), scale, bias)
    )
    excluded = excluded_rows.expand_as(masked)
    assert torch.equal(masked, unmasked.masked_fill(excluded, 0))
    assert not masked[excluded].signbit().any()


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
def test_adaln_module_masks_inside_the_fused_op(norm_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """AdaLN hands a mask with one flag per row to the fused op instead of filling after it, bit for bit."""
    _skip_without_adaln_cubin()
    torch.manual_seed(17)
    adaln = init_module_weights(AdaLN(768, 256, dtype=torch.bfloat16, norm_type=norm_type).cuda())
    a = torch.randn(2, 37, 768, device="cuda").to(torch.bfloat16)
    s = torch.randn(2, 37, 256, device="cuda").to(torch.bfloat16)
    mask = torch.rand(2, 37, device="cuda") > 0.3

    def no_fill(*_: object) -> torch.Tensor:
        raise AssertionError("the mask should be applied inside the fused op")

    with torch.inference_mode():
        expected = adaln(a, s).masked_fill(~mask.unsqueeze(-1), 0)
        monkeypatch.setattr(AdaLN, "_maybe_mask", staticmethod(no_fill))
        actual = adaln(a, s, mask=mask)
    assert torch.equal(actual, expected)


def test_adaln_force_cubin_ignores_warmed_source(monkeypatch):
    backend = AdaLNLayerNormSigmoidCuTe()
    geometry = (256, 256, 256)
    key = (backend._sm_version, torch.bfloat16, 256, False, geometry, adaln_config_identity({}))
    cached_source, cubin = object(), object()
    monkeypatch.setattr(AdaLNLayerNormSigmoidCuTe, "_compiled_cache", {key: cached_source})
    monkeypatch.setattr(backend, "force_cubin", lambda: True)
    monkeypatch.setattr(backend, "_load_cubin_executable", lambda *_args, **_kwargs: cubin)
    assert backend._compile_bucket(torch.bfloat16, 256, {}, geometry) is cubin


def _lose_kernel(monkeypatch: pytest.MonkeyPatch, cause: str) -> None:
    """Leave only the CUBIN pack: force it as model tests do, or drop the kernel sources."""
    if cause == "force_cubin":
        monkeypatch.setenv(adaln_cutedsl.FORCE_CUBIN_ENV, "1")
        return

    def missing_source(_package: str) -> None:
        raise ModuleNotFoundError("CuTeDSL kernel source removed")

    monkeypatch.setattr(adaln_cutedsl, "load_source_module", missing_source)


def _hide_cubin_geometry(monkeypatch: pytest.MonkeyPatch, hidden: tuple[int, int]) -> list[int]:
    """Make the CUBIN pack lookup miss one ``(threads_per_row, num_threads)``; return the PyTorch-path row counts."""
    packed = adaln_cutedsl.AdaLNLayerNormSigmoidCubinExecutable

    def lookup(
        library: ModuleType,
        launcher: object,
        target_sm: int,
        dtype: torch.dtype,
        N: int,
        threads_per_row: int,
        num_threads: int,
        rms_norm: bool,
    ) -> AdaLNLayerNormSigmoidCubinExecutable:
        if (threads_per_row, num_threads) == hidden:
            raise CuTeDSLKernelVariantUnavailable(f"test hides the AdaLN CUBIN for geometry {hidden}")
        return packed(library, launcher, target_sm, dtype, N, threads_per_row, num_threads, rms_norm)

    pytorch_rows: list[int] = []
    pytorch_path = adaln_cutedsl._invoke_vanilla_adaln_layernorm_sigmoid

    def recording_pytorch_path(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        pytorch_rows.append(x.numel() // x.shape[-1])
        return pytorch_path(x, *args, **kwargs)

    monkeypatch.setattr(adaln_cutedsl, "AdaLNLayerNormSigmoidCubinExecutable", lookup)
    monkeypatch.setattr(adaln_cutedsl, "_invoke_vanilla_adaln_layernorm_sigmoid", recording_pytorch_path)
    monkeypatch.setattr(AdaLNLayerNormSigmoidCuTe, "_compiled_cache", {})
    monkeypatch.setattr(AdaLNLayerNormSigmoidCuTe, "_missing_variants_warned", set())
    return pytorch_rows


@pytest.mark.parametrize("cause", ["force_cubin", "no_sources"])
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("N", SHIPPED_N)
def test_adaln_runs_a_bucket_the_cubin_pack_lacks_in_pytorch(
    monkeypatch: pytest.MonkeyPatch, cause: str, rms_norm: bool, dtype: torch.dtype, N: int
) -> None:
    """A pack can lag the bucket schedule, as between a schedule change and its CUBIN refresh. With only the
    pack to run from, a bucket it lacks runs in PyTorch instead of raising mid-forward; the rest keep their
    CUBINs. Each bucket geometry is hidden in turn, after the op is built, as model tests set the flag."""
    _skip_without_adaln_cubin()
    require_cubin_family("adaln_layernorm_sigmoid")
    buckets = bucket_variants(SM_VERSION, N)
    first_rows = [1] + [m_max + 1 for m_max, _, _ in buckets[:-1]]
    torch.manual_seed(3)
    for hidden in dict.fromkeys(geometry for _, _, geometry in buckets):
        op = AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)
        with monkeypatch.context() as patch, pytest.warns(RuntimeWarning, match="runs in PyTorch"):
            _lose_kernel(patch, cause)
            pytorch_rows = _hide_cubin_geometry(patch, hidden)
            for rows, (_, _, geometry) in zip(first_rows, buckets, strict=True):
                x, scale, bias = (torch.randn(rows, N, device="cuda", dtype=dtype) for _ in range(3))
                expected = _torch_adaln_layernorm_sigmoid(x, scale, bias, rms_norm=rms_norm)
                torch.testing.assert_close(op(x, scale, bias), expected, atol=0.05, rtol=0.02)
                assert (rows in pytorch_rows) == (geometry == hidden)
            assert not any(value is None for value in AdaLNLayerNormSigmoidCuTe._compiled_cache.values())


@pytest.mark.parametrize("cause", ["force_cubin", "no_sources"])
def test_adaln_pack_gap_fallback_replays_in_a_cuda_graph(monkeypatch: pytest.MonkeyPatch, cause: str) -> None:
    """The case a CUBIN-only model test hit: fp32 ``N=128`` past the first bucket, with sample broadcast,
    under graph capture."""
    _skip_without_adaln_cubin()
    require_cubin_family("adaln_layernorm_sigmoid")
    N = 128
    m_max, _, _ = bucket_variants(SM_VERSION, N)[0]
    if m_max is None:
        pytest.skip(f"AdaLN SM{SM_VERSION} N={N} has a single bucket")
    samples, tokens = 5, m_max // 5 + 1
    op = AdaLNLayerNormSigmoidCuTe()
    _lose_kernel(monkeypatch, cause)
    pytorch_rows = _hide_cubin_geometry(monkeypatch, select_bucket(SM_VERSION, N, samples * tokens)[1])
    x = torch.randn(1, samples, tokens, N, device="cuda")
    scale, bias = (torch.randn(1, 1, tokens, N, device="cuda") for _ in range(2))
    out = torch.empty_like(x)
    with pytest.warns(RuntimeWarning, match="runs in PyTorch"):
        op(x, scale, bias, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, scale, bias, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert pytorch_rows == [samples * tokens] * 2
    torch.testing.assert_close(out, _torch_adaln_layernorm_sigmoid(x, scale, bias), atol=1e-4, rtol=1e-4)


def test_adaln_warns_once_per_missing_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variant the pack lacks warns once per process: not again on repeated calls, new instances, a switch
    between source-free and forced-CUBIN runs, or graph capture and replay, whatever the warning filter."""
    _skip_without_adaln_cubin()
    require_cubin_family("adaln_layernorm_sigmoid")
    N, rows = 128, 33
    _hide_cubin_geometry(monkeypatch, select_bucket(SM_VERSION, N, rows)[1])
    x, scale, bias = (torch.randn(rows, N, device="cuda") for _ in range(3))
    out = torch.empty_like(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cause in ("no_sources", "force_cubin"):
            with monkeypatch.context() as patch:
                _lose_kernel(patch, cause)
                for _ in range(2):
                    op = AdaLNLayerNormSigmoidCuTe()
                    op(x, scale, bias, out=out)
                    op(x, scale, bias, out=out)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        op(x, scale, bias, out=out)
                    graph.replay()
        torch.cuda.synchronize()
    missing = [w for w in caught if issubclass(w.category, RuntimeWarning) and "runs in PyTorch" in str(w.message)]
    assert len(missing) == 1, [str(w.message) for w in missing]
    torch.testing.assert_close(out, _torch_adaln_layernorm_sigmoid(x, scale, bias), atol=1e-4, rtol=1e-4)


def test_adaln_pack_covers_every_scheduled_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bucket a shipped width schedules on this GPU has a packaged CUBIN, so forcing CUBINs never falls
    back to PyTorch. A schedule change merged ahead of its CUBIN refresh fails here."""
    _skip_without_adaln_cubin()
    require_cubin_family("adaln_layernorm_sigmoid")
    monkeypatch.setenv(adaln_cutedsl.FORCE_CUBIN_ENV, "1")
    monkeypatch.setattr(AdaLNLayerNormSigmoidCuTe, "_compiled_cache", {})
    uncovered = []
    for rms_norm in (False, True):
        op = AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for N in SHIPPED_N:
                for (m_max, executable), (_, _, geometry) in zip(
                    op._get_or_build_schedule(dtype, N), bucket_variants(SM_VERSION, N), strict=True
                ):
                    if not isinstance(executable, CuTeDSLKernelLibraryExecutable):
                        uncovered.append((str(dtype), N, "rms" if rms_norm else "ln", m_max, geometry))
    assert not uncovered, f"SM{SM_VERSION} CUBIN packs lack (dtype, N, norm, bucket m_max, geometry): {uncovered}"


def test_adaln_selector_rejects_unshipped_targets(monkeypatch):
    monkeypatch.setattr(adaln_ops, "get_sm_version", lambda: 110)
    assert adaln_ops.get_adaln_layernorm_sigmoid_op(torch.bfloat16, N=128) is _invoke_vanilla_adaln_layernorm_sigmoid

    monkeypatch.setattr(adaln_ops, "get_sm_version", lambda: 90)
    assert adaln_ops.get_adaln_layernorm_sigmoid_op(torch.bfloat16, N=8192) is _invoke_vanilla_adaln_layernorm_sigmoid

    rms_fallback = adaln_ops.get_adaln_layernorm_sigmoid_op(torch.bfloat16, N=8192, rms_norm=True)
    x = torch.randn(2, 8)
    torch.testing.assert_close(
        rms_fallback(x.clone(), torch.zeros_like(x), torch.zeros_like(x)),
        torch.nn.functional.rms_norm(x, (8,), eps=1e-5) * 0.5,
    )


@pytest.mark.parametrize(
    ("launcher_error", "raised"),
    [
        (ValueError, CuTeDSLKernelVariantUnavailable),
        (TypeError, TypeError),
        (RuntimeError, RuntimeError),
    ],
    ids=["lookup_miss", "stale_binding", "launcher_failure"],
)
def test_cubin_adapter_reports_only_a_lookup_miss_as_a_missing_variant(
    launcher_error: type[Exception], raised: type[Exception]
) -> None:
    """The launcher signals a variant the pack lacks with ``ValueError``. A stale binding or a launcher failure
    propagates, so the call does not fall back to PyTorch."""

    class FakeDType:
        FLOAT16 = object()
        BFLOAT16 = object()
        FLOAT32 = object()

    class FailingLauncher:
        DType = FakeDType

        def make_kernel_config(self, *args: object) -> None:
            raise launcher_error("launcher refused")

    with pytest.raises(raised):
        AdaLNLayerNormSigmoidCubinExecutable(
            ModuleType("fake_adaln_cubin_library"), FailingLauncher(), 100, torch.float32, 128, 8, 128, False
        )


def test_cubin_adapter_selects_rms_norm_variant():
    class FakeDType:
        FLOAT16 = object()
        BFLOAT16 = object()
        FLOAT32 = object()

    class FakeLauncher:
        DType = FakeDType

        def __init__(self):
            self.args = None

        def make_kernel_config(self, *args):
            self.args = args
            return object()

    launcher = FakeLauncher()
    AdaLNLayerNormSigmoidCubinExecutable(
        ModuleType("fake_adaln_cubin_library"),
        launcher,
        86,
        torch.bfloat16,
        384,
        16,
        128,
        True,
    )

    assert launcher.args == (86, FakeDType.BFLOAT16, 384, 16, 128, True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 19, 1024),
        (1, 64, 768),
        (4, 32, 384),
        (3, 67, 128),
    ],
)
def test_custom_op_matches_vanilla(dtype: torch.dtype, rms_norm: bool, shape: tuple):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, D = shape

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    s_scale = torch.randn(B, S, D, dtype=dtype, device=device)
    s_bias = torch.randn(B, S, D, dtype=dtype, device=device)
    eps = 1e-5

    # The fused op is in-place: it mutates ``x``. Snapshot the input first
    # so the reference computation sees the original values.
    x_orig = x.clone()

    fused_op = get_adaln_layernorm_sigmoid_op(dtype, N=D, rms_norm=rms_norm)
    out_fused = fused_op(x, s_scale, s_bias, eps=eps)
    out_ref = _torch_adaln_layernorm_sigmoid(x_orig, s_scale, s_bias, eps=eps, rms_norm=rms_norm)

    assert out_fused.shape == out_ref.shape == x_orig.shape
    assert out_fused.dtype == out_ref.dtype == dtype

    out_ref_fp32 = _torch_adaln_layernorm_sigmoid(
        x_orig.float(),
        s_scale.float(),
        s_bias.float(),
        eps=eps,
        rms_norm=rms_norm,
    )

    diff_fused = (out_fused.float() - out_ref_fp32).abs().max().item()
    diff_torch = (out_ref.float() - out_ref_fp32).abs().max().item()

    if dtype == torch.float32:
        torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)
    else:
        # Allow 2x the torch backend's quantization error.
        assert diff_fused < max(diff_torch * 2.0, 0.05), (
            f"fused err {diff_fused:.3e} too large vs torch err {diff_torch:.3e}"
        )


# ---------------------------------------------------------------------------
# Forcing the torch fallback path must produce a bit-exact reference run
# ---------------------------------------------------------------------------


def test_adaln_torch_backend_explicit():
    """Clearing ``_fused_op`` forces the inline torch path; two such modules
    with identical weights must agree bit-for-bit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")

    a_dim, s_dim = 768, 768
    adaln_a = AdaLN(dim=a_dim, dim_single_cond=s_dim).to(device)
    adaln_a._fused_op = None
    _init_adaln_weights(adaln_a)
    adaln_b = AdaLN(dim=a_dim, dim_single_cond=s_dim).to(device)
    adaln_b._fused_op = None
    _copy_weights(adaln_a, adaln_b)

    a = torch.randn(1, 64, a_dim, device=device)
    s = torch.randn(1, 64, s_dim, device=device)

    with torch.inference_mode():
        out_a = adaln_a.forward(a, s)
        out_b = adaln_b.forward(a, s)

    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Default backend is CuteDSL with silent torch fallback
# ---------------------------------------------------------------------------


def test_adaln_default_backend_is_cutedsl():
    """No explicit ``backend`` ⇒ kernel-backed module (or transparent
    fallback if the kernel can't load on this SM/dtype). Either way the
    output must match a forced-torch reference within kernel tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dim = 768

    adaln_default = AdaLN(dim=dim, dim_single_cond=dim).to(device)  # CuteDSL
    adaln_torch = AdaLN(dim=dim, dim_single_cond=dim).to(device)
    # Force the inline torch path on the reference module.
    adaln_torch._fused_op = None
    _init_adaln_weights(adaln_torch)
    _copy_weights(adaln_torch, adaln_default)

    a = torch.randn(1, 64, dim, device=device)
    s = torch.randn(1, 64, dim, device=device)

    # CuteDSL backend mutates ``a`` in place — give each forward its own copy.
    a_default = a.clone()
    a_torch = a.clone()

    with torch.inference_mode():
        out_default = adaln_default.forward(a_default, s)
        out_torch = adaln_torch.forward(a_torch, s)

    # fp32 default dtype — CuTe kernel matches torch closely.
    torch.testing.assert_close(out_default, out_torch, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Multiplicity broadcast — s_scale / s_bias have size 1 on one leading dim
# (e.g. x is [B, S, I, D] and s is [B, 1, I, D]).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "x_shape, s_shape",
    [
        # Multi-sample case: multiplicity dim is second from front.
        ((1, 5, 128, 768), (1, 1, 128, 768)),
        ((2, 8, 128, 384), (2, 1, 128, 384)),
        ((1, 10, 256, 768), (1, 1, 256, 768)),
        ((1, 4, 512, 768), (1, 1, 512, 768)),
        # Long seq, large multiplicity.
        ((1, 32, 128, 256), (1, 1, 128, 256)),
        # Broadcast on the batch dim.
        ((4, 128, 768), (1, 128, 768)),
        ((4, 5, 64, 256), (1, 5, 64, 256)),
        # 5-D layout ``(B, S, I, J, D)`` with multiplicity at dim 1.
        ((1, 5, 28, 32, 128), (1, 1, 28, 32, 128)),
        # 5-D layout but broadcast on an interior dim (dim 2).
        ((2, 4, 7, 128, 384), (2, 4, 1, 128, 384)),
        ((2, 6, 8, 384), (2, 1, 8, 384)),
        # Small / odd ``inner = prod(x.shape[bcast_dim + 1:-1])``.
        ((1, 4, 3, 768), (1, 1, 3, 768)),  # inner=3
        ((1, 5, 7, 384), (1, 1, 7, 384)),  # inner=7
        ((2, 4, 5, 128), (2, 1, 5, 128)),  # inner=5
        ((1, 8, 1, 768), (1, 1, 1, 768)),  # inner=1
        ((1, 3, 6, 384), (1, 1, 6, 384)),  # inner=6
    ],
)
def test_custom_op_broadcast(dtype: torch.dtype, rms_norm: bool, x_shape: tuple, s_shape: tuple):
    """Kernel must produce torch-equivalent output when s_scale / s_bias
    broadcast against one leading dim of x."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")

    x = torch.randn(*x_shape, dtype=dtype, device=device)
    s_scale = torch.randn(*s_shape, dtype=dtype, device=device)
    s_bias = torch.randn(*s_shape, dtype=dtype, device=device)
    eps = 1e-5

    # Run on a clone since the kernel may write in-place when out is None.
    x_orig = x.clone()
    out = torch.empty_like(x)

    fused_op = get_adaln_layernorm_sigmoid_op(dtype, N=x_shape[-1], rms_norm=rms_norm)
    out_fused = fused_op(x_orig.clone(), s_scale, s_bias, out=out, eps=eps)

    # Reference: torch handles size-1 broadcast natively.
    out_ref = _torch_adaln_layernorm_sigmoid(x_orig, s_scale, s_bias, eps=eps, rms_norm=rms_norm)

    assert out_fused.shape == out_ref.shape == x_orig.shape
    assert out_fused.dtype == dtype

    out_ref_fp32 = _torch_adaln_layernorm_sigmoid(
        x_orig.float(),
        s_scale.float(),
        s_bias.float(),
        eps=eps,
        rms_norm=rms_norm,
    )

    if dtype == torch.float32:
        torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)
    else:
        diff_fused = (out_fused.float() - out_ref_fp32).abs().max().item()
        diff_torch = (out_ref.float() - out_ref_fp32).abs().max().item()
        assert diff_fused < max(diff_torch * 2.0, 0.05), (
            f"fused err {diff_fused:.3e} too large vs torch err {diff_torch:.3e}"
        )


@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
def test_adaln_multisample_does_not_fall_back(norm_type: str):
    """Regression: with x=[B,S,I,D] and s=[B,1,I,D], the CuTeDSL kernel must
    actually run rather than get disabled inside ``AdaLN.forward``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _skip_without_adaln_cubin()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dim, dim_cond = 768, 768
    B, S, I = 1, 5, 128

    module = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=torch.bfloat16, norm_type=norm_type).to(device)
    _init_adaln_weights(module)
    # isinstance, not "is not None": the dispatcher now returns a torch
    # fallback rather than raising, so a None check would pass even when the
    # kernel silently degraded.
    assert isinstance(module._fused_op, AdaLNLayerNormSigmoidCuTe), "kernel must initialize on supported SM"

    a = torch.randn(B, S, I, dim, dtype=torch.bfloat16, device=device)
    s = torch.randn(B, 1, I, dim_cond, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        out = module(a.clone(), s)

    assert isinstance(module._fused_op, AdaLNLayerNormSigmoidCuTe), (
        "AdaLN.forward disabled the kernel mid-forward — multi-sample "
        "broadcast regression (kernel should handle s_scale/s_bias with "
        "size-1 multiplicity dim)"
    )
    assert out.shape == a.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_samples", [5, 10])
@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
def test_adaln_multisample_matches_torch(dtype: torch.dtype, num_samples: int, norm_type: str):
    """End-to-end: AdaLN with multi-sample input should give same numerics
    via CuteDSL backend as via torch backend."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Without a CUBIN there is no CuTeDSL side to compare the torch side to.
    _skip_without_adaln_cubin()

    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    device = torch.device("cuda")
    dim, dim_cond = 768, 768
    B, I = 1, 128

    adaln_ref = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=dtype, norm_type=norm_type).to(device)
    # Force the inline torch path on the reference module.
    adaln_ref._fused_op = None
    _init_adaln_weights(adaln_ref)
    adaln_cute = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=dtype, norm_type=norm_type).to(device)
    _copy_weights(adaln_ref, adaln_cute)

    a = torch.randn(B, num_samples, I, dim, dtype=dtype, device=device)
    s = torch.randn(B, 1, I, dim_cond, dtype=dtype, device=device)

    a_ref = a.clone()
    a_cute = a.clone()

    with torch.inference_mode():
        out_ref = adaln_ref.forward(a_ref, s)
        out_cute = adaln_cute.forward(a_cute, s)

    assert isinstance(adaln_cute._fused_op, AdaLNLayerNormSigmoidCuTe), (
        "kernel disabled — broadcast fallback regression"
    )
    assert out_ref.shape == out_cute.shape == a.shape

    if dtype == torch.float32:
        torch.testing.assert_close(out_cute, out_ref, atol=1e-4, rtol=1e-4)
    else:
        adaln_fp32 = AdaLN(
            dim=dim,
            dim_single_cond=dim_cond,
            dtype=torch.float32,
            norm_type=norm_type,
        ).to(device)
        adaln_fp32.load_state_dict({k: v.float() for k, v in adaln_ref.state_dict().items()}, strict=True)
        with torch.inference_mode():
            out_fp32 = adaln_fp32.forward(a.float(), s.float())
        err_ref = (out_ref.float() - out_fp32).abs().max().item()
        err_cute = (out_cute.float() - out_fp32).abs().max().item()
        assert err_cute < max(err_ref * 2.0, 0.05), f"CuTe err {err_cute:.3e} vs torch err {err_ref:.3e}"


# ---------------------------------------------------------------------------
# Shared-memory accounting
#
# ``dynamic_smem_bytes()`` once counted only the reduction buffer while
# ``kernel`` also allocated ``sX``/``sS``/``sSb``, so a staged-mode launch asked
# for 32 B against a real 98,336 B and faulted.
#
# Staged mode (``N > 8192``) is no longer covered here. It is unreachable from
# the models (their N are 128 / 384 / 768, and every shipped N is <= 1024), and
# in fp32 it needs 196,640 B -- over SM80's 166,912 B opt-in ceiling but under
# SM90's 232,448 B, so the case passed or failed depending on which GPU the CI
# pool handed out. ``AdaLNLayerNormSigmoidCuTe._assert_smem_fits`` still rejects
# an over-budget variant with the numbers if anyone reaches staged mode.
# ---------------------------------------------------------------------------

# All direct mode: below the N <= 8192 boundary that flips direct_load/async_s_copy.
_SMEM_N = [128, 1024, 8192]


def _make_fusion(dtype: torch.dtype, N: int, **kwargs):
    source = _require_source()
    return source.kernel_cls(cute_numeric(dtype), N, **kwargs)


def _reduction_only_bytes(kernel) -> int:
    """The reduction buffer + mbarriers, i.e. the base-class contribution."""
    return _require_source().reduction_base_cls.dynamic_smem_bytes(kernel)


def _require_source() -> ModuleType:
    """Skip source-only checks when the source adapter is absent."""
    return pytest.importorskip(_SOURCE_MODULE, reason="the AdaLN CuTeDSL source adapter is unavailable")


@pytest.mark.parametrize("N", _SMEM_N)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
def test_dynamic_smem_bytes_matches_direct_mode(dtype: torch.dtype, N: int, rms_norm: bool):
    """Every covered N is direct mode, which needs only the reduction buffer.

    The staged-mode arm (sX/sS/sSb) is deliberately not covered -- see the
    section comment above.
    """
    kernel = _make_fusion(dtype, N, rms_norm=rms_norm)
    assert kernel.direct_load and not kernel.async_s_copy, (
        f"N={N} is expected to be direct mode: direct_load={kernel.direct_load} async_s_copy={kernel.async_s_copy}"
    )
    reported = kernel.dynamic_smem_bytes()
    assert reported == _reduction_only_bytes(kernel), (
        f"direct mode should need only the reduction buffer, got {reported}"
    )


@pytest.mark.parametrize("N", _SMEM_N)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
def test_launch_requests_enough_smem_no_cuda_error(dtype: torch.dtype, N: int, rms_norm: bool):
    """Run each N end to end: no CuTe under-subscription warning, no CUDA fault.

    Skipped under ``CUTEDSL_FORCE_CUBIN=1``: ``N`` is compiled in, and the staged
    ``N`` here is deliberately outside ``SHIPPED_N`` so no payload exists.
    """
    import warnings

    _require_source()
    force_cubin_env = adaln_cutedsl.FORCE_CUBIN_ENV

    if os.getenv(force_cubin_env, "").strip().lower() in ("1", "true", "yes", "on"):
        pytest.skip(f"{force_cubin_env}=1: unshipped N has no CUBIN payload")

    M = 64
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    s_scale = torch.randn(M, N, device="cuda", dtype=dtype)
    s_bias = torch.randn(M, N, device="cuda", dtype=dtype)
    expected = _torch_adaln_layernorm_sigmoid(
        x.float(),
        s_scale.float(),
        s_bias.float(),
        rms_norm=rms_norm,
    )

    op = AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = op(x, s_scale, s_bias, torch.empty_like(x))
        torch.cuda.synchronize()  # surface an illegal access here, not in a later test

    undersized = [str(w.message) for w in caught if "smem" in str(w.message).lower()]
    assert not undersized, f"CuTe reports the launch under-requests smem: {undersized}"

    tol = 1e-4 if dtype == torch.float32 else 0.05
    assert (out.float() - expected).abs().max().item() < tol


def test_explicit_staged_flags_are_accounted():
    """A tuning config may set these; both builder and source path forward them."""
    N = 128
    direct = _make_fusion(torch.bfloat16, N)
    staged = _make_fusion(torch.bfloat16, N, direct_load=False)

    assert direct.dynamic_smem_bytes() == _reduction_only_bytes(direct)
    # direct_load=False also defaults async_s_copy on, so all three tiles land in smem.
    assert staged.async_s_copy is True
    assert staged.dynamic_smem_bytes() > direct.dynamic_smem_bytes()


# ---------------------------------------------------------------------------
# Compile-cache identity
#
# The bucket cache key was ``(dtype, N, geometry)``, and geometry resolves only
# tpr_override / num_threads_override. Every other knob was absent from the key,
# so two machine-distinct configs shared one entry -- in memory and on disk --
# and the second silently reused the first's kernel.
# ---------------------------------------------------------------------------


def test_config_identity_separates_every_kernel_knob():
    """Each knob make_kernel forwards must change the identity."""
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import (
        _KERNEL_CFG_DEFAULTS,
        config_identity,
    )

    base = config_identity({})
    for knob, default in _KERNEL_CFG_DEFAULTS.items():
        altered = 1 if not isinstance(default, bool) and default is None else not default
        assert config_identity({knob: altered}) != base, f"{knob} does not reach the cache key"


def test_config_identity_is_default_insensitive_and_extensible():
    """Omitted knobs equal explicit defaults; unknown knobs still participate."""
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import (
        config_identity,
        is_cubin_representable,
    )

    assert config_identity({}) == config_identity({"single_pass": True})
    assert is_cubin_representable({}) and is_cubin_representable({"single_pass": True})
    # A knob added to make_kernel but not to _KERNEL_CFG_DEFAULTS must not alias.
    assert config_identity({"some_future_knob": 7}) != config_identity({})
    assert not is_cubin_representable({"single_pass": False})


def test_shipped_bucket_configs_stay_cubin_representable():
    """Every config the scheduler can hand out must be loadable from a CUBIN.

    ``_BUCKET_BIG_M`` sets tpr/num_threads overrides and is shipped, so a guard
    that treats *any* non-empty config as unrepresentable breaks the whole CUBIN
    path -- those two knobs resolve into ``geometry``, which the registry keys on.
    """
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import (
        SHIPPED_N,
        SUPPORTED_SMS,
        bucket_variants,
        is_cubin_representable,
    )

    for sm in SUPPORTED_SMS:
        for N in SHIPPED_N:
            for _m_max, cfg, _geometry in bucket_variants(sm, N):
                assert is_cubin_representable(cfg), f"sm{sm} N={N} ships {cfg}, which no payload can select"


def test_distinct_configs_do_not_share_a_compiled_cache_entry():
    """Two configs with identical geometry must compile to separate entries.

    Exercises ``_compile_bucket`` itself rather than re-deriving the key, so the
    test fails if the key ever drops back to ``(dtype, N, geometry)``.
    """
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import resolve_geometry
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid.cutedsl import AdaLNLayerNormSigmoidCuTe

    _require_source()
    force_cubin_env = adaln_cutedsl.FORCE_CUBIN_ENV

    if os.getenv(force_cubin_env, "").strip().lower() in ("1", "true", "yes", "on"):
        pytest.skip(f"{force_cubin_env}=1: non-default knobs have no CUBIN payload by design")

    N = 768
    default_cfg, tuned_cfg = {}, {"single_pass": False}
    geometry = resolve_geometry(N, default_cfg)
    assert geometry == resolve_geometry(N, tuned_cfg), "geometry alone cannot tell them apart"

    cache = AdaLNLayerNormSigmoidCuTe()
    saved = dict(AdaLNLayerNormSigmoidCuTe._compiled_cache)
    AdaLNLayerNormSigmoidCuTe._compiled_cache.clear()
    try:
        first = cache._compile_bucket(torch.bfloat16, N, default_cfg, geometry)
        second = cache._compile_bucket(torch.bfloat16, N, tuned_cfg, geometry)
        assert len(AdaLNLayerNormSigmoidCuTe._compiled_cache) == 2, (
            "both configs landed on one cache entry; the second reused the first's kernel"
        )
        assert first is not second
    finally:
        AdaLNLayerNormSigmoidCuTe._compiled_cache.clear()
        AdaLNLayerNormSigmoidCuTe._compiled_cache.update(saved)


def test_cubin_path_refuses_non_default_knobs():
    """No payload encodes a non-default knob, so loading one must fail loudly."""
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import resolve_geometry
    from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid.cutedsl import AdaLNLayerNormSigmoidCuTe

    N = 768
    cache = AdaLNLayerNormSigmoidCuTe()
    with pytest.raises(RuntimeError, match="no AdaLN CUBIN can represent"):
        cache._load_cubin_executable(
            ("BFloat16", N, resolve_geometry(N, {}), ()),
            torch.bfloat16,
            N,
            resolve_geometry(N, {}),
            cfg={"single_pass": False},
        )
