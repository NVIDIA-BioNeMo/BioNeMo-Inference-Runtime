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
"""Source-or-CUBIN CuTeDSL backend for pair-weighted averaging."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import (
    _SUPPORTED_SM,
    PWAConfigParams,
    PWAConfigSelection,
    _select_pwa_config_selection_bucket,
    default_params,
    is_profitable_shape,
)
from ._cubin import PairWeightedAveragingCubinExecutable

_TORCH_TO_DTYPE_STR = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

_CompileKey = tuple[int, tuple[object, ...], torch.dtype]


def _dtype_str(dtype: torch.dtype) -> str:
    """Return the source config token for one supported torch dtype."""
    try:
        return _TORCH_TO_DTYPE_STR[dtype]
    except KeyError as error:
        raise TypeError(f"Pair-weighted averaging expects float16 or bfloat16; got {dtype}") from error


def _compile_cache_key(
    sm_version: int,
    params: PWAConfigParams,
    dtype: torch.dtype,
) -> _CompileKey:
    """Key one always-predicated executable."""
    return sm_version, params.machine_key(), dtype


class PairWeightedAveragingCuTe(CuteKernelCache):
    """Cached source-or-CUBIN backend for fused pair-weighted averaging."""

    # Source and CUBIN executables intentionally share this one cache.
    _compiled_cache: dict[_CompileKey, Any] = {}

    def __init__(self) -> None:
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor

    def _disk_cache_key(self, key: _CompileKey) -> tuple[object, ...]:
        """Version the always-predicated dynamic signature."""
        return ("pwa_cute_always_predicated_v1", *key)

    def _load_cubin_executable(
        self,
        key: _CompileKey,
        params: PWAConfigParams,
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
        source_error: ImportError | None = None,
    ) -> PairWeightedAveragingCubinExecutable:
        """Populate the shared cache from the packaged PWA family."""
        try:
            executable = populate_compiled_cache_from_library(
                type(self)._compiled_cache,
                key,
                "pair_weighted_averaging",
                lambda library, launcher: PairWeightedAveragingCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    I,
                    J,
                    S,
                    params.D,
                    params.c_m,
                    dtype,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel library "
                    f"cannot provide this variant: {library_error}"
                ) from library_error
            raise RuntimeError(
                "CuTeDSL pair_weighted_averaging source is unavailable and the "
                f"precompiled kernel library cannot provide this variant: {library_error}"
            ) from source_error

        logger.info(
            f"CuTeDSL PWA: using precompiled CUBIN for SM{self._sm_version}, I={I}, J={J}, S={S}, "
            f"D={params.D}, c_m={params.c_m}, dtype={dtype}"
        )
        return executable

    def _load_or_compile_source(
        self,
        key: _CompileKey,
        selection: PWAConfigSelection,
        source_module: ModuleType,
        source: Any,
    ) -> Any:
        """Load one persistent dynamic source object or compile and save it."""
        executable = type(self)._compiled_cache.get(key)
        if executable is not None:
            return executable

        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL PWA: loaded cached kernel for SM{self._sm_version}, key={key}")
            type(self)._compiled_cache[key] = executable
            return executable

        logger.info(f"CuTeDSL PWA: compiling kernel for SM{self._sm_version}, key={key}, params={selection.params}")
        executable = source_module.compile_pair_weighted_averaging_source(self.compile, source)
        type(self)._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL PWA: compilation done")
        return executable

    def _get_bucket_executable(
        self,
        selected: PWAConfigSelection,
        bucket: tuple[PWAConfigSelection, ...],
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
    ) -> Any:
        """Resolve the selected executable, preloading all source bucket configs."""
        selected_key = _compile_cache_key(self._sm_version, selected.params, dtype)
        cached = type(self)._compiled_cache.get(selected_key)

        if self.force_cubin():
            if isinstance(cached, CuTeDSLKernelLibraryExecutable):
                return cached
            # A source executable cached before the flag was set must not
            # defeat the explicit request for a packaged CUBIN.
            type(self)._compiled_cache.pop(selected_key, None)
            return self._load_cubin_executable(selected_key, selected.params, I, J, S, dtype)

        bucket_variants = list(bucket)
        if cached is not None and all(
            _compile_cache_key(self._sm_version, selection.params, dtype) in type(self)._compiled_cache
            for selection in bucket_variants
        ):
            return cached

        try:
            source_module = load_source_module(__package__)

            unresolved_sources = [
                (
                    selection,
                    source_module.resolve_pwa_source(selection, _dtype_str(dtype)),
                )
                for selection in bucket_variants
                if _compile_cache_key(self._sm_version, selection.params, dtype) not in type(self)._compiled_cache
            ]
        except ImportError as source_error:
            if cached is not None:
                return cached
            return self._load_cubin_executable(selected_key, selected.params, I, J, S, dtype, source_error)

        # Compilation stays outside the source-resolution try block so a real
        # compiler failure is never hidden behind the CUBIN fallback.
        for selection, source in unresolved_sources:
            key = _compile_cache_key(self._sm_version, selection.params, dtype)
            self._load_or_compile_source(key, selection, source_module, source)
        return type(self)._compiled_cache[selected_key]

    @staticmethod
    def _normalize_inputs(
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize only layouts required by the source and CUBIN ABIs."""
        w = w.contiguous()
        weight = weight.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()
        if g.stride(-1) != 1:
            g = g.contiguous()
        return w, v, g, weight

    @staticmethod
    def _validate_launch_dtypes(
        dtype: torch.dtype,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        """Require every source/CUBIN tensor operand to use one scalar dtype."""
        for name, tensor in (("v", v), ("g", g), ("weight", weight)):
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must use {dtype}; got {tensor.dtype}")

    def _launch_selection(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        selected: PWAConfigSelection,
        bucket: tuple[PWAConfigSelection, ...],
    ) -> torch.Tensor:
        """Allocate output, resolve one dynamic variant, and launch it."""
        B, _, I, _ = w.shape
        S, J = v.shape[1], v.shape[2]
        dtype = w.dtype
        self._validate_launch_dtypes(dtype, v, g, weight)
        executable = self._get_bucket_executable(selected, bucket, I, J, S, dtype)
        w, v, g, weight = self._normalize_inputs(w, v, g, weight)
        output = torch.empty(B, S, I, weight.shape[0], dtype=dtype, device=w.device)
        launch_compiled_kernel(executable, w, v, g, weight, output)
        return output

    def is_supported(self, w: torch.Tensor, v: torch.Tensor, Wo: torch.Tensor) -> bool:
        """Check dtype, registered dimensions, SM, and w-padding constraints."""
        if self._sm_version not in _SUPPORTED_SM:
            return False
        if w.dtype not in _TORCH_TO_DTYPE_STR or w.dim() != 4 or v.dim() != 4 or Wo.dim() != 2:
            return False
        H, I, Jp = w.shape[1], w.shape[2], w.shape[3]
        N = v.shape[2]
        hidden = v.shape[-1]
        D = hidden // H if H else 0
        c_m = Wo.shape[0]
        return hidden == H * D and Jp >= N and Jp % 8 == 0 and is_profitable_shape(H, D, c_m, max(I, N))

    def __call__(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        Wo: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused PWA or use the vanilla fallback outside its fixed envelope."""
        if not self.is_supported(w, v, Wo):
            from .ops import _invoke_vanilla_pwa

            return _invoke_vanilla_pwa(w, v, g, Wo)

        _, H, I, _ = w.shape
        S, J = v.shape[1], v.shape[2]
        D = v.shape[-1] // H
        c_m = Wo.shape[0]
        dtype_str = _dtype_str(w.dtype)
        selected, bucket = _select_pwa_config_selection_bucket(
            self._sm_version,
            I,
            J,
            S,
            dtype_str,
            H=H,
            D=D,
            c_m=c_m,
        )
        return self._launch_selection(w, v, g, Wo, selected, bucket)

    def _call_explicit(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        Wo: torch.Tensor,
        config: object | None,
    ) -> torch.Tensor:
        """Compatibility launch with one explicit or default source config."""
        if w.dim() != 4 or v.dim() != 4 or Wo.dim() != 2:
            raise ValueError("PWA expects rank-4 w/v and rank-2 output weight tensors")
        _, H, _, Jp = w.shape
        N = v.shape[2]
        if Jp < N or Jp % 8 != 0:
            raise ValueError(f"PWA: w j-extent Jp={Jp} must cover N={N} and be a multiple of 8")
        dtype_str = _dtype_str(w.dtype)
        D = v.shape[-1] // H
        c_m = Wo.shape[0]
        if config is None:
            params = default_params(dtype_str, H, D, c_m)
        else:
            config_dtype = config.ab_dtype
            if config_dtype != dtype_str:
                raise TypeError(f"PWA config dtype {config_dtype!r} does not match tensor dtype {w.dtype}")
            params = PWAConfigParams.from_config(config)
        selection = PWAConfigSelection(params=params)
        return self._launch_selection(w, v, g, Wo, selection, (selection,))


def pwa_cute(
    w: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    Wo: torch.Tensor,
    config: object | None = None,
) -> torch.Tensor:
    """Launch PWA with an explicit source-compatible config or its defaults."""
    return PairWeightedAveragingCuTe()._call_explicit(w, v, g, Wo, config)
