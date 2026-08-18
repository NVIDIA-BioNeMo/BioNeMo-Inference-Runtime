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
"""Source-or-CUBIN CuTeDSL backend for dual-GEMM ``x_x``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    launch_compiled_kernel,
    populate_compiled_cache_from_library,
)
from bionemo_ir._torch._kernel_source_loader import load_source_module
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import (
    _get_bucket_ranges,
    _get_config_selection,
    _kernel_is_sm90,
    _variant_key,
)
from ._cubin import DualGemmXxCubinExecutable

_TORCH_TO_DTYPE_STR = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@dataclass(frozen=True)
class _DualGemmXxVariant:
    """Every runtime axis that selects machine code or argument packing."""

    dtype: torch.dtype
    K: int
    N: int
    bucket: int
    transpose_out: bool
    has_bias: bool
    has_mask: bool


def _compute_S(I_dim: int) -> int:
    """Map a flattened row count to the per-side tuning axis."""
    return int(round(math.sqrt(max(I_dim, 1))))


def _dtype_str(dtype: torch.dtype) -> str:
    """Return the config dtype token for a supported torch dtype."""
    try:
        return _TORCH_TO_DTYPE_STR[dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported dtype: {dtype}") from error


class DualGemmXxCuTe(CuteKernelCache):
    """Cached CuTeDSL backend for the dual-GEMM ``x_x`` variant.

    Source-enabled builds resolve and compile the implementation the ``_source``
    adapter selects for the JSON tuning bundle. Source-free builds populate the
    same executable cache with a direct adapter for
    ``_cutedsl_kernels.dual_gemm_x_x``.
    """

    _compiled_cache: dict[tuple[int, _DualGemmXxVariant], Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_exe = None
        self._last_key: _DualGemmXxVariant | None = None
        self._bucket_ranges: dict[tuple[int, int, bool], list[tuple[int, str]] | None] = {}

    def _disk_cache_key(self, variant: _DualGemmXxVariant) -> tuple:
        """Return the unchanged source-object disk cache key."""
        source_key = (
            variant.dtype,
            variant.K,
            variant.N,
            _variant_key(variant.bucket, variant.transpose_out),
            variant.transpose_out,
            variant.has_bias,
            variant.has_mask,
        )
        return ("dual_gemm_x_x_cute_v2", self._sm_version) + source_key

    def _kernel_is_sm90(self, K: int, N: int) -> bool:
        """Whether this shape uses the duplicated-X SM90 source signature."""
        return _kernel_is_sm90(self._sm_version, K, N)

    def _resolve_source_kernel(
        self,
        variant: _DualGemmXxVariant,
        x: torch.Tensor,
    ) -> tuple[ModuleType, Any]:
        """Import, resolve, and instantiate one development-time source."""
        source_module = load_source_module(__package__)

        selection = _get_config_selection(
            self._sm_version,
            variant.K,
            variant.N,
            variant.bucket,
            variant.transpose_out,
        )
        source = source_module.resolve_source_kernel(
            selection,
            x,
            has_bias=variant.has_bias,
            has_mask=variant.has_mask,
            transpose_out=variant.transpose_out,
            dtype_str=_dtype_str(variant.dtype),
        )
        return source_module, source

    def _load_cubin_executable(
        self,
        variant: _DualGemmXxVariant,
        source_error: ImportError | None = None,
    ) -> DualGemmXxCubinExecutable:
        """Populate the shared cache from the packaged kernel library."""
        cache_key = (self._sm_version, variant)
        try:
            executable = populate_compiled_cache_from_library(
                DualGemmXxCuTe._compiled_cache,
                cache_key,
                "dual_gemm_x_x",
                lambda library, launcher: DualGemmXxCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    variant.K,
                    variant.N,
                    variant.bucket,
                    variant.dtype,
                    variant.transpose_out,
                    variant.has_bias,
                    variant.has_mask,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel library "
                    f"cannot provide this variant: {library_error}"
                ) from library_error
            raise RuntimeError(
                "CuTeDSL dual_gemm_x_x source is unavailable and the precompiled "
                f"kernel library cannot provide this variant: {library_error}"
            ) from source_error

        logger.info(
            f"CuTeDSL dual_gemm x_x: using precompiled CUBIN for SM{self._sm_version}, "
            f"K={variant.K}, N={variant.N}, bucket={variant.bucket}, "
            f"transpose_out={variant.transpose_out}, has_bias={variant.has_bias}, "
            f"has_mask={variant.has_mask}, dtype={variant.dtype}"
        )
        return executable

    def _load_or_compile_source(
        self,
        variant: _DualGemmXxVariant,
        source_module: ModuleType,
        source: Any,
    ) -> Any:
        """Load the persistent source object or compile and save it."""
        cache_key = (self._sm_version, variant)
        disk_key = self._disk_cache_key(variant)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL dual_gemm x_x: loaded cached kernel for SM{self._sm_version}, key={variant}")
            DualGemmXxCuTe._compiled_cache[cache_key] = executable
            return executable

        logger.info(
            f"CuTeDSL dual_gemm x_x: compiling kernel for SM{self._sm_version}, "
            f"K={variant.K}, N={variant.N}, bucket={variant.bucket}, "
            f"transpose_out={variant.transpose_out}, has_bias={variant.has_bias}, "
            f"has_mask={variant.has_mask}, dtype={variant.dtype}, tile={source.config.tile_params}"
        )
        executable = source_module.compile_dual_gemm_x_x_source(
            self.compile,
            source,
            K=variant.K,
            transpose_out=variant.transpose_out,
            has_bias=variant.has_bias,
            has_mask=variant.has_mask,
            is_sm90=self._kernel_is_sm90(variant.K, variant.N),
        )
        DualGemmXxCuTe._compiled_cache[cache_key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL dual_gemm x_x: compilation done")
        return executable

    def _get_executable(self, variant: _DualGemmXxVariant, x: torch.Tensor) -> Any:
        """Resolve a cached source executable or direct CUBIN adapter."""
        cache_key = (self._sm_version, variant)
        executable = DualGemmXxCuTe._compiled_cache.get(cache_key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        if force_cubin:
            # A process may set the flag after compiling a source variant. Do
            # not let that cache entry defeat the explicit CUBIN request.
            DualGemmXxCuTe._compiled_cache.pop(cache_key, None)
            return self._load_cubin_executable(variant)

        try:
            source_module, source = self._resolve_source_kernel(variant, x)
        except ImportError as source_error:
            return self._load_cubin_executable(variant, source_error)

        return self._load_or_compile_source(variant, source_module, source)

    def __call__(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        bias0: torch.Tensor | None = None,
        bias1: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        transpose_out: bool = False,
        actual_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a sigmoid-gated dual GEMM with an optional left-aligned mask.

        Args:
            x: ``[B, I, J, K]``, ``[B, I, K]``, or ``[M, K]`` activation.
            w0: ``[N, K]`` sigmoid-gate weight.
            w1: ``[N, K]`` value-gate weight.
            bias0: Optional ``[N]`` sigmoid-gate bias.
            bias1: Optional ``[N]`` value-gate bias.
            mask: Optional binary mask reduced to int32 leading lengths.
            transpose_out: Move the output ``N`` dimension to the front.
            actual_seqlen: Optional precomputed int32 leading lengths.

        Returns:
            The gated output with the same layout as the existing backend.
        """
        x = x.contiguous()
        if x.ndim == 2:
            kernel_B, I_dim = 1, x.shape[0]
            S = _compute_S(I_dim)
            leading = (x.shape[0],)
        elif x.ndim == 3:
            kernel_B, I_dim, _ = x.shape
            S = _compute_S(I_dim)
            leading = x.shape[:-1]
        elif x.ndim == 4:
            batch, i_outer, j_outer, _ = x.shape
            kernel_B = batch * i_outer
            I_dim = j_outer
            S = j_outer
            leading = x.shape[:-1]
        else:
            raise ValueError(f"x must be 2-D / 3-D / 4-D, got shape {tuple(x.shape)}")

        K = x.shape[-1]
        N = w0.shape[0]
        M = kernel_B * I_dim
        device = x.device

        if (bias0 is None) != (bias1 is None):
            raise ValueError("bias0 and bias1 must both be supplied or both None.")
        has_bias = bias0 is not None
        has_mask = mask is not None or actual_seqlen is not None
        _dtype_str(x.dtype)

        ranges = self._get_bucket_ranges(K, N, transpose_out)
        bucket = min(ranges, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))[0] if ranges else S
        variant = _DualGemmXxVariant(
            dtype=x.dtype,
            K=K,
            N=N,
            bucket=bucket,
            transpose_out=transpose_out,
            has_bias=has_bias,
            has_mask=has_mask,
        )

        force_cubin = self.force_cubin()
        if variant == self._last_key and (
            not force_cubin or isinstance(self._last_exe, CuTeDSLKernelLibraryExecutable)
        ):
            executable = self._last_exe
        else:
            executable = self._get_executable(variant, x)
            self._last_key = variant
            self._last_exe = executable

        x_2d = x.reshape(M, K)
        if has_mask:
            actual_seqlen = self._get_actual_seqlen(
                actual_seqlen=actual_seqlen,
                mask=mask,
                kernel_B=kernel_B,
                I_dim=I_dim,
                device=device,
            )
        else:
            actual_seqlen = None

        is_sm90 = self._kernel_is_sm90(K, N)

        if transpose_out:
            m_padded = (M + 7) // 8 * 8
            output_storage = torch.empty((N, m_padded), dtype=x.dtype, device=device)
            output = output_storage[:, :M].T
        else:
            output = torch.empty((M, N), dtype=x.dtype, device=device)
            output_storage = output

        if is_sm90:
            launch_compiled_kernel(
                executable,
                x_2d,
                x_2d,
                w0,
                w1,
                bias0,
                bias1,
                actual_seqlen,
                output,
                I_dim,
            )
        else:
            launch_compiled_kernel(
                executable,
                x_2d,
                w0,
                w1,
                bias0,
                bias1,
                actual_seqlen,
                output,
                I_dim,
            )

        if transpose_out:
            result = output_storage[:, :M]
            if x.ndim == 2:
                return result
            return result.view(N, *leading)
        if x.ndim == 2:
            return output
        return output.view(*leading, N)

    def _get_bucket_ranges(self, K: int, N: int, transpose_out: bool) -> list[tuple[int, str]] | None:
        """Return cached sorted anchors for a shape and output layout."""
        cache_key = (K, N, bool(transpose_out))
        if cache_key not in self._bucket_ranges:
            self._bucket_ranges[cache_key] = _get_bucket_ranges(
                self._sm_version,
                K,
                N,
                transpose_out,
            )
        return self._bucket_ranges[cache_key]

    def _nearest_anchor_key(self, S: int, K: int, N: int, transpose_out: bool) -> str:
        """Return the nearest JSON key, preferring the lower anchor on ties."""
        ranges = self._get_bucket_ranges(K, N, transpose_out)
        if not ranges:
            return f"S~{S}"
        return min(ranges, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))[1]

    @staticmethod
    def _get_actual_seqlen(
        actual_seqlen: torch.Tensor | None,
        mask: torch.Tensor | None,
        kernel_B: int,
        I_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize a mask or explicit length tensor to contiguous int32."""
        if actual_seqlen is not None:
            if actual_seqlen.device != device:
                actual_seqlen = actual_seqlen.to(device)
            if actual_seqlen.dtype != torch.int32:
                actual_seqlen = actual_seqlen.to(torch.int32)
            if actual_seqlen.numel() != kernel_B:
                raise ValueError(
                    f"actual_seqlen must have {kernel_B} elements (kernel batch count), "
                    f"got shape {tuple(actual_seqlen.shape)}"
                )
            return actual_seqlen.reshape(kernel_B).contiguous()
        if mask is not None:
            return (mask > 0).sum(-1).reshape(kernel_B).to(dtype=torch.int32, device=device)
        return torch.full((kernel_B,), I_dim, dtype=torch.int32, device=device)
