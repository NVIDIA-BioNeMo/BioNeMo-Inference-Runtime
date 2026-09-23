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

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
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
    needs_independent_operand_strides,
    supports_fused_residual,
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
    # Second GEMM inner dim; equal to ``K`` for the symmetric case.
    K1: int
    fused_residual: bool = False


class DualGemmX0X1CuTe(CuteKernelCache):
    """Cache architecture-specific source or CUBIN executables."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_exe = None
        self._last_variant: _DualGemmX0X1Variant | None = None
        # Parsed tuning anchors by (K0, K1, N, has_bias).
        self._bucket_ranges: dict[tuple, list[tuple[int, str]] | None] = {}
        self._is_sm90: dict[tuple[int, int, int], bool] = {}

    def _disk_cache_key(self, variant: _DualGemmX0X1Variant) -> tuple:
        # Bump when the compiled ABI changes; a stale entry would reuse a
        # kernel with shared ``K0``/``K1`` layout or the old SM90 path.
        return (
            "dual_gemm_x0_x1_cute_asym_v9",
            self._sm_version,
            variant.dtype,
            variant.K,
            variant.K1,
            variant.N,
            variant.bucket,
            variant.has_bias,
            variant.fused_residual,
        )

    def _kernel_is_sm90(self, K: int, N: int, K1: int | None = None) -> bool:
        """Whether ``(K, K1, N)`` resolves to the Hopper calling convention."""
        K1 = K if K1 is None else K1
        cached = self._is_sm90.get((K, K1, N))
        if cached is None:
            cached = kernel_is_sm90(self._sm_version, K, N, K1)
            self._is_sm90[(K, K1, N)] = cached
        return cached

    def _nearest_bucket(self, S: int, K: int, N: int, has_bias: bool, K1: int | None = None) -> int:
        """Return the tuned ``S`` anchor nearest ``S`` for this call site."""
        K1 = K if K1 is None else K1
        cache_key = (K, K1, N, bool(has_bias))
        ranges = self._bucket_ranges.get(cache_key)
        if cache_key not in self._bucket_ranges:
            try:
                bundle = load_bundle(self._sm_version, K, N, K1)
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
            has_mask=variant.fused_residual,
            fused_residual=variant.fused_residual,
            K1=variant.K1,
        )
        # Both inner dims share one tile, so each must satisfy the kernel's
        # alignment and shared-memory limits.
        for inner in {variant.K, variant.K1}:
            if not config.can_implement(ct_dtype, inner, variant.N):
                raise RuntimeError(
                    f"dual_gemm x0_x1 kernel cannot implement: dtype={ct_dtype}, "
                    f"K0={variant.K}, K1={variant.K1}, N={variant.N}"
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
                    variant.K1,
                    variant.fused_residual,
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
            asymmetric=needs_independent_operand_strides(variant.K, variant.K1, variant.N),
            has_mask=variant.fused_residual,
            fused_residual=variant.fused_residual,
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
        actual_seqlen: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the fused output.

        Args:
            X0: First activation [*, K0].
            X1: Second activation [*, K1]; leading dims must match ``X0``.
            W0: First weight [N, K0].
            W1: Second weight [N, K1].
            bias0: Optional first bias [N].
            bias1: Optional second bias [N].
            actual_seqlen: Optional int32 valid-J counts for every ``(B, I)`` row.
            residual: Optional residual with the input leading shape and ``N`` channels.

        Returns:
            Output [*, N].
        """
        if X0.ndim < 1 or X1.ndim < 1:
            raise ValueError("X0 and X1 must each have at least one dimension")
        if W0.ndim != 2 or W1.ndim != 2:
            raise ValueError(f"W0 and W1 must be rank 2, got {W0.ndim} and {W1.ndim}")

        N, K = W0.shape  # dim_out, first dim_in (K0)
        N1, K1 = W1.shape  # second dim_in may differ from K0
        x0_orig_shape = X0.shape  # e.g. [B, R, R, K0]
        x0_ndim = X0.ndim
        device = X0.device
        dtype = X0.dtype

        if not X0.is_cuda:
            raise ValueError("dual_gemm x0_x1 CuTe operands must be CUDA tensors")
        if N1 != N:
            raise ValueError(f"W1 output dimension must be {N}, got {tuple(W1.shape)}")
        if x0_orig_shape[-1] != K:
            raise ValueError(f"X0 trailing dimension must match W0 K0={K}, got {tuple(x0_orig_shape)}")
        if X1.shape[-1] != K1 or X1.shape[:-1] != x0_orig_shape[:-1]:
            raise ValueError(
                f"X1 expected leading shape {tuple(x0_orig_shape[:-1])} and K1={K1}, got {tuple(X1.shape)}"
            )
        if X1.dtype != dtype or X1.device != device:
            raise ValueError("X0 and X1 must have matching dtype and device")
        for name, tensor in (("W0", W0), ("W1", W1)):
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError(f"{name} must match X0 dtype and device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if (bias0 is None) != (bias1 is None):
            raise ValueError("bias0 and bias1 must both be supplied or both None.")
        if bias0 is not None and bias1 is not None:
            for name, bias in (("bias0", bias0), ("bias1", bias1)):
                if bias.shape != (N,):
                    raise ValueError(f"{name} must have shape ({N},), got {tuple(bias.shape)}")
                if bias.dtype != dtype or bias.device != device:
                    raise ValueError(f"{name} must match X0 dtype and device")
                if not bias.is_contiguous():
                    raise ValueError(f"{name} must be contiguous")
        if K % 8 != 0 or K1 % 8 != 0 or N % 8 != 0:
            raise ValueError("dual_gemm x0_x1 requires K0, K1 and N to be multiples of 8")

        X0_2d = X0.reshape(-1, K) if x0_ndim != 2 else X0
        X1_2d = X1.reshape(-1, K1) if x0_ndim != 2 else X1
        residual_2d = None
        M = X0_2d.shape[0]
        if M == 0:
            raise ValueError("dual_gemm x0_x1 requires at least one row")
        for name, tensor, width in (("X0", X0_2d, K), ("X1", X1_2d, K1)):
            if tensor.stride(1) != 1 or tensor.stride(0) < width:
                raise ValueError(f"{name} must have contiguous rows")
            if tensor.stride(0) % 8 != 0:
                raise ValueError(f"{name} row stride must be a multiple of 8 elements")
        # The compact signature was traced with one row-stride symbol for both
        # inputs and the output, so it can only bind inputs whose row stride is
        # already N. Shapes on the wide signature carry three independent
        # strides and may therefore consume aligned row-padded inputs.
        if not needs_independent_operand_strides(K, K1, N) and X0_2d.stride(0) != N:
            raise ValueError(f"X0 row stride must equal output width N={N}, got {X0_2d.stride(0)}")
        has_bias = bias0 is not None
        fused_residual = residual is not None
        if fused_residual != (actual_seqlen is not None):
            raise ValueError("actual_seqlen and residual must both be supplied for fused residual output")
        if fused_residual:
            if not supports_fused_residual(self._sm_version, K, N, K1):
                raise ValueError("fused dual_gemm x0_x1 residual output is unavailable for this shape")
            if residual.shape != (*x0_orig_shape[:-1], N):
                raise ValueError(f"residual must have shape {(*x0_orig_shape[:-1], N)}, got {tuple(residual.shape)}")
            if residual.dtype != dtype or residual.device != device:
                raise ValueError("residual must match X0 dtype and device")
            if not residual.is_contiguous():
                raise ValueError("residual must be contiguous")
            residual_2d = residual.view(-1, N)
            if residual_2d.stride(1) != 1 or residual_2d.stride(0) % 8 != 0:
                raise ValueError("residual rows must be contiguous and 16-byte aligned")

        dtype_str = _dtype_str(dtype)
        variant = _DualGemmX0X1Variant(
            dtype=dtype,
            K=K,
            N=N,
            bucket=self._nearest_bucket(compute_S(M), K, N, has_bias, K1),
            has_bias=has_bias,
            K1=K1,
            fused_residual=fused_residual,
        )

        if variant == self._last_variant:
            exe = self._last_exe
        else:
            exe = self._get_executable(variant, _cutlass_dtype(X0), dtype_str)
            self._last_variant = variant
            self._last_exe = exe

        out_2d = torch.empty((M, N), dtype=dtype, device=device)
        i_dim = 1
        if fused_residual:
            i_dim = x0_orig_shape[-2]
            kernel_B = M // i_dim
            if actual_seqlen.numel() != kernel_B:
                raise ValueError(f"actual_seqlen must have {kernel_B} elements, got shape {tuple(actual_seqlen.shape)}")
            actual_seqlen = actual_seqlen.to(device=device, dtype=torch.int32).reshape(kernel_B).contiguous()
        if self._kernel_is_sm90(K, N, K1):
            # Hopper carries unused mask and I_dim slots.
            launch_compiled_kernel(
                exe,
                X0_2d,
                X1_2d,
                W0,
                W1,
                bias0,
                bias1,
                actual_seqlen,
                residual_2d,
                out_2d,
                i_dim,
            )
        else:
            if fused_residual:
                launch_compiled_kernel(
                    exe,
                    X0_2d,
                    X1_2d,
                    W0,
                    W1,
                    bias0,
                    bias1,
                    actual_seqlen,
                    residual_2d,
                    out_2d,
                    i_dim,
                )
            else:
                launch_compiled_kernel(exe, X0_2d, X1_2d, W0, W1, bias0, bias1, out_2d)

        if x0_ndim == 2:
            return out_2d
        return out_2d.view(x0_orig_shape[:-1] + (N,))
