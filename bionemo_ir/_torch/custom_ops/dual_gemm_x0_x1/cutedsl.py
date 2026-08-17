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
"""Source-or-CUBIN backend for fused dual-GEMM ``x0_x1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import torch

from bionemo_ir._torch._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryError,
    launch_compiled_kernel,
    populate_compiled_cache_from_library,
)
from bionemo_ir._torch._kernel_source_loader import load_source_module
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import (
    _CONFIGS_DIR,
    DualGemmX0X1KernelConfig,
    bucket_anchors,
    compute_S,
    get_kernel_config,
    kernel_is_sm90,
    load_bundle,
)
from ._cubin import DualGemmX0X1CubinExecutable

__all__ = [
    "DualGemmX0X1CuTe",
    "DualGemmX0X1KernelConfig",
    "_CONFIGS_DIR",
    "get_kernel_config",
]

_TORCH_TO_DTYPE_STR = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _cutlass_dtype(tensor: torch.Tensor) -> type:
    dtype = _TORCH_TO_CUTLASS_DTYPE.get(tensor.dtype)
    if dtype is None:
        raise TypeError(f"dual_gemm x0_x1 CuTe expects float16 or bfloat16; got {tensor.dtype}")
    return dtype


def _dtype_str(dtype: torch.dtype) -> str:
    name = _TORCH_TO_DTYPE_STR.get(dtype)
    if name is None:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return name


@dataclass(frozen=True)
class _DualGemmX0X1Variant:
    """Every property that selects a distinct compiled kernel."""

    dtype: torch.dtype
    K: int
    N: int
    bucket: int
    has_bias: bool


class DualGemmX0X1CuTe(CuteKernelCache):
    """Cache architecture-specific source or CUBIN executables."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_exe = None
        self._last_variant: _DualGemmX0X1Variant | None = None
        # Parsed tuning anchors by (K, N, has_bias).
        self._bucket_ranges: dict[tuple, list[tuple[int, str]] | None] = {}
        self._is_sm90: dict[tuple[int, int], bool] = {}

    def _disk_cache_key(self, variant: _DualGemmX0X1Variant) -> tuple:
        # v2 uses 64-bit activation/output row strides; bump on ABI changes.
        return (
            "dual_gemm_x0_x1_cute_asym_v2",
            self._sm_version,
            variant.dtype,
            variant.K,
            variant.N,
            variant.bucket,
            variant.has_bias,
        )

    def _kernel_is_sm90(self, K: int, N: int) -> bool:
        """Whether ``(K, N)`` resolves to the Hopper calling convention."""
        cached = self._is_sm90.get((K, N))
        if cached is None:
            cached = kernel_is_sm90(self._sm_version, K, N)
            self._is_sm90[(K, N)] = cached
        return cached

    def _nearest_bucket(self, S: int, K: int, N: int, has_bias: bool) -> int:
        """Return the tuned ``S`` anchor nearest ``S`` for this call site."""
        cache_key = (K, N, bool(has_bias))
        ranges = self._bucket_ranges.get(cache_key)
        if cache_key not in self._bucket_ranges:
            try:
                bundle = load_bundle(self._sm_version, K, N)
            except ValueError:
                ranges = None
            else:
                ranges = bucket_anchors(bundle.configs, has_bias) or None
            self._bucket_ranges[cache_key] = ranges
        if not ranges:
            return S
        return min(ranges, key=lambda anchor: (abs(anchor[0] - S), anchor[0]))[0]

    def _resolve_source_kernel(
        self,
        variant: _DualGemmX0X1Variant,
        ct_dtype: type,
        dtype_str: str,
    ) -> tuple[DualGemmX0X1KernelConfig, Any]:
        config = get_kernel_config(
            self._sm_version,
            K=variant.K,
            N=variant.N,
            S=variant.bucket,
            has_bias=variant.has_bias,
            dtype_str=dtype_str,
        )
        if not config.can_implement(ct_dtype, variant.K, variant.N):
            raise RuntimeError(
                f"dual_gemm x0_x1 kernel cannot implement: dtype={ct_dtype}, K={variant.K}, N={variant.N}"
            )
        return config, config.kernel_factory()

    def _load_cubin_executable(
        self,
        variant: _DualGemmX0X1Variant,
        source_error: Exception | None = None,
    ):
        cache_key = (self._sm_version, variant)
        try:
            executable = populate_compiled_cache_from_library(
                DualGemmX0X1CuTe._compiled_cache,
                cache_key,
                "dual_gemm_x0_x1",
                lambda library, launcher: DualGemmX0X1CubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    variant.K,
                    variant.N,
                    variant.bucket,
                    variant.dtype,
                    variant.has_bias,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel "
                    f"library cannot provide this variant: {library_error}"
                ) from library_error
            raise RuntimeError(
                "CuTeDSL dual_gemm x0_x1 source is unavailable and the "
                f"precompiled kernel library cannot provide this variant: {library_error}"
            ) from source_error

        logger.info(
            f"CuTeDSL dual_gemm x0_x1: using precompiled CUBIN for "
            f"SM{self._sm_version}, dtype={variant.dtype}, K={variant.K}, "
            f"N={variant.N}, bucket={variant.bucket}, has_bias={variant.has_bias}"
        )
        return executable

    def _load_or_compile_source(
        self,
        variant: _DualGemmX0X1Variant,
        config: DualGemmX0X1KernelConfig,
        kernel: Any,
        compile_source: Any,
        ct_dtype: type,
    ):
        cache_key = (self._sm_version, variant)
        disk_key = self._disk_cache_key(variant)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL dual_gemm x0_x1: loaded cached kernel for SM{self._sm_version}, key={cache_key}")
            DualGemmX0X1CuTe._compiled_cache[cache_key] = executable
            return executable

        logger.info(
            f"CuTeDSL dual_gemm x0_x1: compiling kernel for SM{self._sm_version} "
            f"({config.arch}), K={variant.K}, N={variant.N}, "
            f"S_anchor~{variant.bucket}, has_bias={variant.has_bias}, "
            f"dt={variant.dtype}, picked={config.chosen_key}, tile={config.tile_params}"
        )
        executable = compile_source(
            self.compile,
            kernel,
            config.arch,
            ct_dtype,
            variant.has_bias,
        )
        DualGemmX0X1CuTe._compiled_cache[cache_key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL dual_gemm x0_x1: compilation done")
        return executable

    def _get_executable(self, variant: _DualGemmX0X1Variant, ct_dtype: type, dtype_str: str):
        cache_key = (self._sm_version, variant)
        executable = DualGemmX0X1CuTe._compiled_cache.get(cache_key)
        if executable is not None:
            return executable

        if self.force_cubin():
            return self._load_cubin_executable(variant)

        try:
            source = load_source_module(__package__)
            config, kernel = self._resolve_source_kernel(variant, ct_dtype, dtype_str)
        except ImportError as source_error:
            return self._load_cubin_executable(variant, source_error)

        return self._load_or_compile_source(
            variant,
            config,
            kernel,
            source.compile_dual_gemm_x0_x1_source,
            ct_dtype,
        )

    def __call__(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        W0: torch.Tensor,
        W1: torch.Tensor,
        bias0: torch.Tensor | None = None,
        bias1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the fused output.

        Args:
            X0: First activation [*, K].
            X1: Second activation [*, K], matching ``X0``.
            W0: First weight [N, K].
            W1: Second weight [N, K], matching ``W0``.
            bias0: Optional first bias [N].
            bias1: Optional second bias [N].

        Returns:
            Output [*, N].
        """
        w0_shape = W0.shape
        N, K = w0_shape  # dim_out, dim_in
        x0_orig_shape = X0.shape  # e.g. [B, R, R, K]
        x0_ndim = X0.ndim
        device = X0.device
        dtype = X0.dtype

        if W1.shape != w0_shape:
            raise ValueError(f"W1.shape expected {w0_shape}, got {W1.shape}")
        if X1.shape != x0_orig_shape:
            raise ValueError(f"X1.shape expected {x0_orig_shape}, got {X1.shape}")
        if (bias0 is None) != (bias1 is None):
            raise ValueError("bias0 and bias1 must both be supplied or both None.")

        X0_2d = X0.reshape(-1, K) if x0_ndim != 2 else X0
        X1_2d = X1.reshape(-1, K) if x0_ndim != 2 else X1
        M = X0_2d.shape[0]
        has_bias = bias0 is not None

        dtype_str = _dtype_str(dtype)
        variant = _DualGemmX0X1Variant(
            dtype=dtype,
            K=K,
            N=N,
            bucket=self._nearest_bucket(compute_S(M), K, N, has_bias),
            has_bias=has_bias,
        )

        if variant == self._last_variant:
            exe = self._last_exe
        else:
            exe = self._get_executable(variant, _cutlass_dtype(X0), dtype_str)
            self._last_variant = variant
            self._last_exe = exe

        out_2d = torch.empty((M, N), dtype=dtype, device=device)
        if self._kernel_is_sm90(K, N):
            # Hopper carries unused mask and I_dim slots.
            launch_compiled_kernel(exe, X0_2d, X1_2d, W0, W1, bias0, bias1, None, out_2d, 1)
        else:
            launch_compiled_kernel(exe, X0_2d, X1_2d, W0, W1, bias0, bias1, out_2d)

        if x0_ndim == 2:
            return out_2d
        return out_2d.view(x0_orig_shape[:-1] + (N,))
