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
"""Source/CUBIN tests for dual-GEMM ``x0_x1``."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
import torch

from bionemo_ir._torch import _cutedsl_kernel_library as library_runtime
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import DualGemmX0X1CuTe
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import _config as dg_config
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import cutedsl as dg_cutedsl
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1._cubin import DualGemmX0X1CubinExecutable
from tests._torch import SM_VERSION, cutedsl_test_modes, skip_if_no_cutedsl

_SOURCE_MODULE = "bionemo_ir.dsl_kernels.cute.sm80_dualgemm_x0x1_splitkv1"
_MODES = cutedsl_test_modes(_SOURCE_MODULE)
# Configs select Ampere split-K or Hopper ping-pong by device.
_CUBIN_SMS = (80, 86, 89, 90)
_MODE_CACHES: dict[str, dict] = {"source": {}, "cubin": {}}

CONFIG_DIR = Path(dg_config._CONFIGS_DIR)


def _configure_mode(mode: str, monkeypatch) -> None:
    """Force one implementation path without allowing silent fallback."""
    monkeypatch.setattr(DualGemmX0X1CuTe, "_compiled_cache", _MODE_CACHES[mode])

    if mode == "cubin":
        if SM_VERSION not in _CUBIN_SMS:
            pytest.skip(f"dual_gemm x0_x1 CUBINs cover SM{_CUBIN_SMS} (current SM{SM_VERSION})")
        try:
            importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
        except ImportError:
            pytest.fail("CUBIN test mode requires the _cutedsl_kernels extension")

        def source_unavailable(_implementation):
            raise ModuleNotFoundError("CuTeDSL source disabled by CUBIN test mode")

        monkeypatch.setattr(dg_config, "resolve_implementation", source_unavailable)
        monkeypatch.setattr(library_runtime, "_kernel_library", None)
        return

    def reject_cubin_fallback(*_args, **_kwargs):
        raise AssertionError("source test mode unexpectedly fell back to the CUBIN library")

    monkeypatch.setattr(dg_cutedsl, "populate_compiled_cache_from_library", reject_cubin_fallback)


def _reference(X0, X1, W0, W1, bias0, bias1) -> torch.Tensor:
    """fp32 reference mirroring the vanilla backend."""
    d0 = torch.nn.functional.linear(X0.float(), W0.float(), None if bias0 is None else bias0.float())
    d1 = torch.nn.functional.linear(X1.float(), W1.float(), None if bias1 is None else bias1.float())
    return (d0.sigmoid() * d1).to(X0.dtype).contiguous()


def _run(op, K, N, M, dtype, has_bias, *, row_stride_pad=0):
    W0 = torch.randn(N, K, dtype=dtype, device="cuda")
    W1 = torch.randn(N, K, dtype=dtype, device="cuda")
    bias0 = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None
    bias1 = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None
    if row_stride_pad:
        # Preserve inner contiguity while changing the row stride.
        X0 = torch.randn(M, K + row_stride_pad, device="cuda").to(dtype)[:, :K]
        X1 = torch.randn(M, K + row_stride_pad, device="cuda").to(dtype)[:, :K]
        assert X0.stride(0) == K + row_stride_pad and X0.stride(1) == 1
    else:
        X0 = torch.randn(M, K, device="cuda").to(dtype)
        X1 = torch.randn(M, K, device="cuda").to(dtype)
    return op(X0, X1, W0, W1, bias0, bias1), _reference(X0, X1, W0, W1, bias0, bias1)


@pytest.mark.parametrize("cutedsl_mode", _MODES, ids=lambda mode: f"impl-{mode}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize(
    ("K", "N", "M"),
    [
        (128, 128, 64),  # minimum: one tile in each dimension
        (128, 128, 100 * 100),  # tail rows, mid bucket
        (128, 128, 1024 * 1024),  # aligned, large bucket
        (256, 256, 128 * 128),  # ProtenixV2 width at a bucket boundary
        (256, 256, 257 * 257),  # tail rows just past a bucket boundary
    ],
    ids=["K128-min", "K128-tail", "K128-aligned", "K256-boundary", "K256-tail"],
)
def test_matches_fp32_reference(cutedsl_mode, dtype, has_bias, K, N, M, monkeypatch):
    skip_if_no_cutedsl()
    _configure_mode(cutedsl_mode, monkeypatch)
    torch.manual_seed(42)
    out, ref = _run(DualGemmX0X1CuTe(), K, N, M, dtype, has_bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("cutedsl_mode", _MODES, ids=lambda mode: f"impl-{mode}")
def test_mismatched_row_stride_is_rejected_by_both_paths(cutedsl_mode, monkeypatch):
    """Both paths reject the shared row-stride ABI mismatch."""
    skip_if_no_cutedsl()
    _configure_mode(cutedsl_mode, monkeypatch)
    torch.manual_seed(7)
    with pytest.raises(ValueError, match="stride"):
        _run(DualGemmX0X1CuTe(), 128, 128, 512, torch.bfloat16, True, row_stride_pad=64)


def test_cubin_mode_selects_the_cubin_adapter(monkeypatch):
    """CUBIN mode must actually reach the library, not a cached source kernel."""
    skip_if_no_cutedsl()
    _configure_mode("cubin", monkeypatch)
    torch.manual_seed(0)
    op = DualGemmX0X1CuTe()
    _run(op, 128, 128, 512, torch.bfloat16, True)
    executable = op._last_exe
    assert isinstance(executable, DualGemmX0X1CubinExecutable)
    config = executable._config
    assert config.variant_id
    assert config.spec.target_sm == SM_VERSION
    assert config.spec.K == 128 and config.spec.N == 128
    assert config.has_bias is True
    assert config.dynamic_smem_bytes > 0


def test_cubin_and_python_agree_on_the_bucket(monkeypatch):
    """Nearest-anchor selection is implemented twice; they must not diverge."""
    skip_if_no_cutedsl()
    _configure_mode("cubin", monkeypatch)
    library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    launcher = library.dual_gemm_x0_x1
    for K, N in ((128, 128), (256, 256)):
        anchors = dg_config.bucket_anchors(dg_config.load_bundle(SM_VERSION, K, N).configs, True)
        assert anchors
        for S in [1, 64, 200, 400, 900, 1500, 5000]:
            expected = dg_config.get_nearest_bucket(SM_VERSION, K, N, S, True)
            config = launcher.make_kernel_config(SM_VERSION, K, N, S, launcher.DType.BFLOAT16, True)
            assert config.spec.bucket == expected, f"K={K} N={N} S={S}"


def test_every_shipped_config_is_reachable_through_the_cubin_path(monkeypatch):
    """A tuned entry the CUBIN path cannot select is a variant nobody runs."""
    skip_if_no_cutedsl()
    _configure_mode("cubin", monkeypatch)
    library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    launcher = library.dual_gemm_x0_x1
    dtypes = {"bf16": launcher.DType.BFLOAT16, "fp16": launcher.DType.FLOAT16}
    path = CONFIG_DIR / f"K{128}_N{128}_sm{SM_VERSION}.json"
    checked = 0
    for path in sorted(CONFIG_DIR.glob(f"K*_N*_sm{SM_VERSION}.json")):
        match = re.fullmatch(r"K(\d+)_N(\d+)_sm(\d+)\.json", path.name)
        K, N, _ = map(int, match.groups())
        bundle = json.loads(path.read_text())
        for key in bundle["configs"]:
            key_match = re.fullmatch(r"S=(\d+)(?:\|b=([01]))?", key)
            bucket = int(key_match.group(1))
            raw_bias = key_match.group(2)
            bias_modes = (False, True) if raw_bias is None else (bool(int(raw_bias)),)
            for name, library_dtype in dtypes.items():
                for has_bias in bias_modes:
                    config = launcher.make_kernel_config(SM_VERSION, K, N, bucket, library_dtype, has_bias)
                    assert config.spec.bucket == bucket, f"{path.name} {key} {name}"
                    assert config.has_bias == has_bias
                    checked += 1
    assert checked > 0


def test_unavailable_variant_raises_instead_of_falling_back(monkeypatch):
    """An unsupported shape must fail loudly, not silently pick another kernel."""
    skip_if_no_cutedsl()
    _configure_mode("cubin", monkeypatch)
    library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    launcher = library.dual_gemm_x0_x1
    # nanobind maps the launcher's std::invalid_argument onto ValueError.
    with pytest.raises(ValueError, match="No embedded dual-GEMM x0_x1 CUBIN"):
        launcher.make_kernel_config(SM_VERSION, 512, 512, 128, launcher.DType.BFLOAT16, True)


def test_force_cubin_names_the_flag_when_a_variant_is_missing(monkeypatch):
    """``CUTEDSL_FORCE_CUBIN`` must not be mistaken for missing sources."""
    skip_if_no_cutedsl()
    monkeypatch.setattr(DualGemmX0X1CuTe, "_compiled_cache", {})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    op = DualGemmX0X1CuTe()
    variant = dg_cutedsl._DualGemmX0X1Variant(dtype=torch.bfloat16, K=512, N=512, bucket=128, has_bias=True)
    with pytest.raises(RuntimeError, match="CUTEDSL_FORCE_CUBIN"):
        op._load_cubin_executable(variant)
