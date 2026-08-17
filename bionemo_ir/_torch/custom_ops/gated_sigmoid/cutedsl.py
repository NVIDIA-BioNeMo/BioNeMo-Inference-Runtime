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
"""Source-or-CUBIN interface for fused gated sigmoid."""

from __future__ import annotations

from typing import Any

import cutlass
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
    _GS_CONFIGS_DIR,
    M_MEDIUM_THRESHOLD,
    M_SHORT_THRESHOLD,
    GatedSigmoidKernelConfig,
    _build_kernel_config,
    _classify_m_range,
    _make_sm80_config,
    _params_from_json,
    get_kernel_config,
    get_m_bucket,
    get_tile_params,
)
from ._cubin import GatedSigmoidCubinExecutable
from .ops import _invoke_vanilla_gated_sigmoid

__all__ = [
    "GatedSigmoidCuTe",
    "GatedSigmoidKernelConfig",
    "M_MEDIUM_THRESHOLD",
    "M_SHORT_THRESHOLD",
    "_GS_CONFIGS_DIR",
    "_build_kernel_config",
    "_classify_m_range",
    "_invoke_vanilla_gated_sigmoid",
    "_make_sm80_config",
    "_params_from_json",
    "get_kernel_config",
]

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _cutlass_dtype(t: torch.Tensor) -> type:
    ty = _TORCH_TO_CUTLASS_DTYPE.get(t.dtype)
    if ty is None:
        raise TypeError(f"SM80 gated sigmoid expects float16 or bfloat16; got {t.dtype}")
    return ty


class GatedSigmoidCuTe(CuteKernelCache):
    """Cached backend for ``sigmoid(s @ W.T + bias) * mha_out``."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_exe = None
        self._last_key: tuple = ()

    def _disk_cache_key(self, key: tuple) -> tuple:
        # Bias changes the compiled signature; bump the tag for ABI changes.
        return ("gated_sigmoid_cute_v2",) + key

    def _load_cubin_executable(
        self,
        key: tuple,
        dtype: torch.dtype,
        has_bias: bool,
        K: int,
        N: int,
        M: int,
        source_error: Exception | None = None,
    ):
        tile_params = get_tile_params(self._sm_version, K, N, M)
        m_bucket = get_m_bucket(M)
        try:
            executable = populate_compiled_cache_from_library(
                GatedSigmoidCuTe._compiled_cache,
                key,
                "gated_sigmoid",
                lambda library, launcher: GatedSigmoidCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    m_bucket,
                    dtype,
                    has_bias,
                    tile_params,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the gated sigmoid CUBIN path, "
                    f"but no CUBIN is available for SM{self._sm_version}, "
                    f"dtype={dtype}, has_bias={has_bias}, K={K}, N={N}, "
                    f"m_bucket={m_bucket}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL gated sigmoid: using CUBIN kernel for SM{self._sm_version}, "
            f"dtype={dtype}, has_bias={has_bias}, K={K}, N={N}, m_bucket={m_bucket}"
        )
        return executable

    def _resolve_source_kernel(self, ct_dtype: type, has_bias: bool, K: int, N: int, M: int):
        source = load_source_module(__package__)
        config = get_kernel_config(self._sm_version, K, N, M)
        if not config.can_implement(ct_dtype):
            raise RuntimeError(f"Gated sigmoid kernel cannot implement dtype={ct_dtype}")

        kernel = config.kernel_factory(ct_dtype, has_bias)
        needed = kernel.dynamic_smem_bytes(
            ct_dtype,
            kernel.bM,
            kernel.bN,
            kernel.bK,
            kernel.num_stages,
            kernel.n_mhaout_stages,
        )
        limit = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        if needed > limit:
            raise RuntimeError(f"gated sigmoid needs {needed} B dynamic shared memory, but the device allows {limit} B")
        return kernel, source.compile_gated_sigmoid_source

    def _load_or_compile_source(
        self,
        key: tuple,
        kernel: Any,
        compile_source: Any,
        ct_dtype: type,
        has_bias: bool,
        K: int,
        N: int,
        M: int,
    ):
        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL gated sigmoid: loaded cached kernel for SM{self._sm_version}, key={key}")
            GatedSigmoidCuTe._compiled_cache[key] = executable
            return executable

        logger.info(
            f"CuTeDSL gated sigmoid: compiling kernel for SM{self._sm_version}, "
            f"K={K}, N={N}, m_range={_classify_m_range(M)!r}, has_bias={has_bias}"
        )
        executable = compile_source(self.compile, kernel, ct_dtype, has_bias)
        GatedSigmoidCuTe._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL gated sigmoid: compilation done")
        return executable

    def _get_or_compile(self, ct_dtype: type, dtype: torch.dtype, has_bias: bool, K: int, N: int, M: int, key: tuple):
        executable = GatedSigmoidCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        if force_cubin:
            GatedSigmoidCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, dtype, has_bias, K, N, M)

        try:
            kernel, compile_source = self._resolve_source_kernel(ct_dtype, has_bias, K, N, M)
        except ImportError as source_error:
            return self._load_cubin_executable(key, dtype, has_bias, K, N, M, source_error)
        return self._load_or_compile_source(key, kernel, compile_source, ct_dtype, has_bias, K, N, M)

    def __call__(
        self,
        s: torch.Tensor,
        weight: torch.Tensor,
        mha_out: torch.Tensor,
        bias: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fused gated sigmoid GEMM.

        Args:
            s: Gate input with shape ``[..., K]``.
            weight: Projection weight with shape ``[N, K]``.
            mha_out: Attention output with shape ``[..., N]``. Its leading
                dimensions may differ from ``s`` in one dimension where
                ``s`` has size 1.
            bias: Optional projection bias with shape ``[N]``.
            output: Optional output tensor matching ``mha_out``.

        Returns:
            Tensor with the same shape as ``mha_out``.
        """
        K = s.shape[-1]
        N_out = weight.shape[0]

        # Allow one broadcast leading dimension.
        s_lead = s.shape[:-1]
        mha_lead = mha_out.shape[:-1]
        mult = 1
        inner: int | None = None
        if s_lead != mha_lead:
            mismatched = [i for i in range(len(mha_lead)) if i >= len(s_lead) or s_lead[i] != mha_lead[i]]
            if len(s_lead) != len(mha_lead) or len(mismatched) != 1 or s_lead[mismatched[0]] != 1:
                return _invoke_vanilla_gated_sigmoid(s, weight, mha_out, bias, output)
            bcast_dim = mismatched[0]
            mult = int(mha_lead[bcast_dim])
            inner = 1
            for d in range(bcast_dim + 1, len(mha_lead)):
                inner *= int(mha_lead[d])

        s_2d = s.reshape(-1, K)
        M_s = s_2d.shape[0]
        mha_2d = mha_out.reshape(-1, N_out)
        M_out = mha_2d.shape[0]
        if inner is None:
            inner = M_s

        has_bias = bias is not None
        m_range = _classify_m_range(M_s)
        compile_key = (self._sm_version, s.dtype, has_bias, K, N_out, m_range)

        force_cubin = self.force_cubin()
        if compile_key == self._last_key and (
            not force_cubin or isinstance(self._last_exe, CuTeDSLKernelLibraryExecutable)
        ):
            exe = self._last_exe
        else:
            ct_dtype = _cutlass_dtype(s_2d)
            exe = self._get_or_compile(ct_dtype, s.dtype, has_bias, K, N_out, M_s, compile_key)
            self._last_key = compile_key
            self._last_exe = exe

        # No-bias kernels omit the bias argument from their ABI.
        out_provided = (
            output is not None
            and output.is_contiguous()
            and output.shape[-1] == N_out
            and output.numel() == M_out * N_out
        )
        if out_provided:
            out_2d = output.view(M_out, N_out)
        else:
            out_2d = torch.empty_like(mha_2d)

        launch_compiled_kernel(exe, s_2d, weight, bias, mha_2d, out_2d, mult, inner)

        if out_provided:
            return output
        if mha_out.ndim == 2:
            return out_2d
        return out_2d.view(mha_out.shape[:-1] + (N_out,))
