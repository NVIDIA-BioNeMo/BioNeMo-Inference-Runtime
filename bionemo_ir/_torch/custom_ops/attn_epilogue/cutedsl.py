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
"""Source and CUBIN selection for the fused attention epilogue."""

from __future__ import annotations

import json
from typing import Any

import torch
from cutlass.cute.runtime import make_fake_stream

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import TunedConfig, tuned_configs
from ._cubin import AttnEpilogueCubinExecutable

__all__ = ["AttnEpilogueCuTe"]


class AttnEpilogueCuTe(CuteKernelCache):
    """Compiled or packaged executables for one layer shape and tuning anchor."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(
        self,
        heads: int,
        head_dim: int,
        channels: int,
        has_bias: bool = False,
        has_output_gate: bool = False,
        anchor: int | None = None,
        has_residual: bool = True,
    ) -> None:
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._shape = (heads, head_dim, channels)
        self._has_bias = has_bias
        self._has_output_gate = has_output_gate
        self._has_residual = has_residual
        # The ``R=<rows>`` tuning this backend serves; None takes the lowest.
        self._anchor = anchor

    def _tuning(self) -> TunedConfig | None:
        configs = tuned_configs(self._sm_version, *self._shape)
        if self._anchor is None:
            return configs[0] if configs else None
        return next((config for config in configs if config.rows == self._anchor), None)

    def _rows(self) -> int | None:
        # Every launch keys its executable on this; an anchor is already its tuning's rows.
        if self._anchor is not None:
            return self._anchor
        tuning = self._tuning()
        return None if tuning is None else tuning.rows

    def _describe(self) -> str:
        heads, head_dim, channels = self._shape
        return (
            f"SM{self._sm_version}, heads={heads}, head_dim={head_dim}, channels={channels}, "
            f"has_bias={self._has_bias}, has_output_gate={self._has_output_gate}, "
            f"has_residual={self._has_residual}, rows={self._rows()}"
        )

    def _load_cubin_executable(self, key: tuple, source_error: Exception | None = None) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                AttnEpilogueCuTe._compiled_cache,
                key,
                "attn_epilogue",
                lambda library, launcher: AttnEpilogueCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    *self._shape,
                    self._has_bias,
                    self._has_output_gate,
                    self._rows() or 0,
                    has_residual=self._has_residual,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the attention epilogue CUBIN path, "
                    f"but no CUBIN is available for {self._describe()}"
                ) from library_error
            raise library_error from source_error
        logger.info(f"CuTeDSL attention epilogue: using CUBIN kernel for {self._describe()}")
        return executable

    def _load_or_compile_source(self, key: tuple, device_index: int) -> Any:
        source = load_source_module(__package__)
        tuning = self._tuning()
        if tuning is None:
            raise ImportError(f"no attention epilogue tuning for {self._describe()}")
        kernel_abi, kernel_variant, tile_params = tuning.kernel_abi, tuning.kernel_variant, tuning.tile_params
        # Bump the tag when the call ABI changes. The anchor stays out, so
        # anchors that share a tuning share one compiled kernel.
        disk_key = (
            "attn_epilogue_cute_v3",
            self._sm_version,
            *self._shape,
            self._has_bias,
            self._has_output_gate,
            self._has_residual,
            device_index,
            kernel_abi,
            kernel_variant,
            json.dumps(tile_params, sort_keys=True),
        )
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(f"CuTeDSL attention epilogue: compiling kernel for {self._describe()}")
            # The kernel sizes its persistent grid from the device it is built on.
            with torch.cuda.device(device_index):
                kernel = source.make_kernel(
                    kernel_abi,
                    tile_params,
                    *self._shape[:2],
                    has_bias=self._has_bias,
                    has_output_gate=self._has_output_gate,
                    channels=self._shape[2],
                    kernel_variant=kernel_variant,
                    residual=self._has_residual,
                )
                executable = source.compile_attn_epilogue_source(
                    self.compile, kernel, make_fake_stream(use_tvm_ffi_env_stream=True)
                )
            self.save_to_cache(disk_key, executable)
        AttnEpilogueCuTe._compiled_cache[key] = executable
        return executable

    def executable(self, device_index: int) -> Any:
        """Return the executable for ``device_index``, compiling or loading it once."""
        key = (
            self._sm_version,
            *self._shape,
            self._has_bias,
            self._has_output_gate,
            self._has_residual,
            self._rows(),
            device_index,
        )
        executable = AttnEpilogueCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            AttnEpilogueCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key)
        try:
            return self._load_or_compile_source(key, device_index)
        except ImportError as source_error:
            return self._load_cubin_executable(key, source_error)
