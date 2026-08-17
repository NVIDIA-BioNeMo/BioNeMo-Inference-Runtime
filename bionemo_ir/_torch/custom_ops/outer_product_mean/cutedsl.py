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
"""Source-or-CUBIN CuTeDSL interface for fused outer-product-mean."""

from __future__ import annotations

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
    _KERNEL_C,
    _KERNEL_CZ,
    _KERNEL_D,
    _OPM_CONFIGS_DIR,
    KernelConfig,
    _parse_key,
    _select_opm_config_bucket,
    config_identity,
    select_opm_config,
)
from ._cubin import OuterProductMeanCubinExecutable
from .ops import _invoke_vanilla_opm

__all__ = [
    "OuterProductMeanCuTe",
    "_KERNEL_C",
    "_KERNEL_CZ",
    "_KERNEL_D",
    "_OPM_CONFIGS_DIR",
    "_invoke_vanilla_opm",
    "_parse_key",
    "_select_opm_config_bucket",
    "config_identity",
    "select_opm_config",
]

# SM86/89 use a smaller tile to fit their shared-memory limit.
_SUPPORTED_SM = (80, 86, 89, 90, 100, 103)
_DTYPE_STR = {torch.float16: "fp16", torch.bfloat16: "bf16"}


class OuterProductMeanCuTe(CuteKernelCache):
    """Cached CuTeDSL backend for the fused outer-product-mean projection."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor

    def _disk_cache_key(self, key: tuple) -> tuple:
        return ("opm_cute",) + key

    def _load_cubin_executable(
        self,
        key: tuple,
        config: KernelConfig,
        dtype: torch.dtype,
        has_bias: bool,
        norm_before: bool,
        source_error: Exception | None = None,
    ):
        identity = config_identity(config)
        try:
            executable = populate_compiled_cache_from_library(
                OuterProductMeanCuTe._compiled_cache,
                key,
                "outer_product_mean",
                lambda library, launcher: OuterProductMeanCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    dtype,
                    has_bias,
                    norm_before,
                    identity,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the outer-product-mean CUBIN "
                    f"path, but no CUBIN is available for SM{self._sm_version}, "
                    f"dtype={dtype}, has_bias={has_bias}, "
                    f"norm_before={norm_before}, config={identity}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL OPM: using CUBIN kernel for SM{self._sm_version}, "
            f"dtype={dtype}, has_bias={has_bias}, norm_before={norm_before}"
        )
        return executable

    @staticmethod
    def _resolve_source(config: KernelConfig, has_bias: bool, norm_before: bool):
        source = load_source_module(__package__)

        needed = source.dynamic_smem_bytes(config)
        limit = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        if needed > limit:
            raise RuntimeError(
                f"outer-product-mean needs {needed} B dynamic shared memory, but the device allows {limit} B"
            )
        return source.make_kernel(config, has_bias, norm_before), source.compile_opm_source

    def _load_or_compile_source(
        self, key: tuple, kernel: Any, compile_source: Any, config: KernelConfig, has_bias: bool
    ):
        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL OPM: loaded cached kernel for SM{self._sm_version}, key={key}")
            OuterProductMeanCuTe._compiled_cache[key] = executable
            return executable

        logger.info(f"CuTeDSL OPM: compiling kernel for SM{self._sm_version}, key={key}")
        executable = compile_source(self.compile, kernel, config, has_bias)
        OuterProductMeanCuTe._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        return executable

    def _get_or_compile(
        self,
        config: KernelConfig,
        dtype: torch.dtype,
        has_bias: bool,
        norm_before: bool,
        key: tuple,
    ):
        executable = OuterProductMeanCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        if force_cubin:
            OuterProductMeanCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, config, dtype, has_bias, norm_before)

        try:
            kernel, compile_source = self._resolve_source(config, has_bias, norm_before)
        except ImportError as source_error:
            return self._load_cubin_executable(key, config, dtype, has_bias, norm_before, source_error)
        return self._load_or_compile_source(key, kernel, compile_source, config, has_bias)

    def is_supported(self, a: torch.Tensor, b: torch.Tensor, W_o: torch.Tensor) -> bool:
        """Whether the CuTeDSL kernel can run this problem (else vanilla)."""
        if self._sm_version not in _SUPPORTED_SM:
            return False
        if a.dtype not in _DTYPE_STR or a.dim() != 4 or b.dim() != 4:
            return False
        C, D, C_z = a.shape[-1], b.shape[-1], W_o.shape[0]
        return C == _KERNEL_C and D == _KERNEL_D and C_z == _KERNEL_CZ

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        num_mask: torch.Tensor,
        W_o: torch.Tensor,
        bias: torch.Tensor | None = None,
        norm_before: bool = True,
    ) -> torch.Tensor:
        """Run fused OPM, falling back to PyTorch when unsupported.

        Args:
            a: Left input with shape ``[B, S, I, C]``.
            b: Right input with shape ``[B, S, J, D]``.
            num_mask: Normalization counts with shape ``[B, I, J]``.
            W_o: Projection weight with shape ``[C_z, C * D]``.
            bias: Optional projection bias with shape ``[C_z]``.
            norm_before: Apply normalization before the projection.

        Returns:
            Tensor with shape ``[B, I, J, C_z]``.
        """
        if not self.is_supported(a, b, W_o):
            return _invoke_vanilla_opm(a, b, num_mask, W_o, bias, norm_before)

        B, S, I, C = a.shape
        J = b.shape[2]
        C_z = W_o.shape[0]
        has_bias = bias is not None
        dtype_str = _DTYPE_STR[a.dtype]

        config, bucket_configs = _select_opm_config_bucket(
            self._sm_version, I, J, S, norm_before, has_bias, dtype_str, C=C, D=b.shape[-1], C_z=C_z
        )

        # Preload the selected N bucket.
        bucket_executables = {}
        for variant in bucket_configs:
            variant_key = (self._sm_version, variant.config_key(), has_bias, norm_before)
            bucket_executables[variant_key] = self._get_or_compile(variant, a.dtype, has_bias, norm_before, variant_key)
        key = (self._sm_version, config.config_key(), has_bias, norm_before)
        exe = bucket_executables[key]

        a = a.contiguous()
        b = b.contiguous()
        num_mask = num_mask.to(torch.float32).contiguous()
        W_o = W_o.contiguous()
        t_bias = bias.contiguous() if has_bias else None
        out = torch.empty(B, I, J, C_z, dtype=a.dtype, device=a.device)

        launch_compiled_kernel(exe, a, b, num_mask, W_o, t_bias, out)
        return out
