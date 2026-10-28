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
"""Source and CUBIN selection for the windowed attention kernels."""

from __future__ import annotations

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

from ._config import config_kernel_abi
from ._cubin import WindowAttentionCubinExecutable

__all__ = ["WindowAttentionCuTe"]

_LOG2_E = 1.4426950408889634


class WindowAttentionCuTe(CuteKernelCache):
    """The compiled or packaged kernel for one window shape and dtype on one SM.

    A build with the kernel sources compiles the kernel its config names; one without them, or any build under
    ``CUTEDSL_FORCE_CUBIN``, launches the packaged CUBIN for the SM.
    """

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self, sm_version: int, dtype_str: str, head_dim: int, n_queries: int, n_keys: int):
        self._sm_version = sm_version
        self._dtype_str = dtype_str
        self._shape = (head_dim, n_queries, n_keys)

    def _describe(self) -> str:
        head_dim, n_queries, n_keys = self._shape
        return f"SM{self._sm_version}, {self._dtype_str}, D={head_dim}, windows {n_queries}x{n_keys}"

    def _load_cubin_executable(self, key: tuple, source_error: Exception | None = None) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                WindowAttentionCuTe._compiled_cache,
                key,
                "window_attention",
                lambda library, launcher: WindowAttentionCubinExecutable(
                    library, launcher, self._sm_version, self._dtype_str, *self._shape
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the window attention CUBIN path, "
                    f"but no CUBIN is available for {self._describe()}"
                ) from library_error
            raise library_error from source_error
        logger.info(f"CuTeDSL window attention: using CUBIN kernel for {self._describe()}")
        return executable

    def _load_or_compile_source(self, key: tuple) -> Any:
        source = load_source_module(__package__)
        kernel_abi = config_kernel_abi(self._sm_version, *self._shape)
        if kernel_abi is None:
            raise ImportError(f"no window attention config for {self._describe()}")
        if not source.can_implement(kernel_abi, self._dtype_str, *self._shape, self._sm_version):
            raise RuntimeError(f"the {kernel_abi} window attention kernel cannot serve {self._describe()}")
        # Bump the tag when the call ABI changes.
        disk_key = ("window_attention_cute", 2, self._sm_version, kernel_abi, self._dtype_str, *self._shape)
        executable = self.load_from_cache(disk_key)
        if executable is None:
            from cutlass.cute.runtime import make_fake_stream

            logger.info(f"CuTeDSL window attention: compiling the {kernel_abi} kernel for {self._describe()}")
            kernel = source.make_kernel(kernel_abi, *self._shape)
            executable = source.compile_window_attention_source(
                self.compile, kernel, self._dtype_str, *self._shape, make_fake_stream(use_tvm_ffi_env_stream=True)
            )
            self.save_to_cache(disk_key, executable)
        WindowAttentionCuTe._compiled_cache[key] = executable
        return executable

    def executable(self) -> Any:
        """Return the executable, compiling or loading it once per process."""
        key = (self._sm_version, self._dtype_str, *self._shape)
        executable = WindowAttentionCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            WindowAttentionCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key)
        try:
            return self._load_or_compile_source(key)
        except ImportError as source_error:
            return self._load_cubin_executable(key, source_error)

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, out: torch.Tensor, scale: float
    ) -> torch.Tensor:
        launch_compiled_kernel(self.executable(), q, k, v, bias, out, scale * _LOG2_E, scale)
        return out
