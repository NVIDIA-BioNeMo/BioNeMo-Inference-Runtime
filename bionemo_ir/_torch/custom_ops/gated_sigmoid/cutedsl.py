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
    _GS_CONFIGS_DIR,
    M_MEDIUM_THRESHOLD,
    M_SHORT_THRESHOLD,
    GatedSigmoidKernelConfig,
    _build_kernel_config,
    _classify_m_range,
    _make_sm80_config,
    _params_from_json,
    get_kernel_abi,
    get_kernel_config,
    get_m_bucket,
    get_tile_params,
    kernel_smem_bytes,
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
    """Cached backend for ``[residual +] [mask *] sigmoid(s @ W.T + bias) * mha_out``."""

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_exe = None
        self._last_key: tuple = ()
        self._last_abi = ""

    def _disk_cache_key(self, key: tuple) -> tuple:
        # Bias, residual and mask change the compiled signature; bump the tag for ABI changes.
        return ("gated_sigmoid_cute_v3",) + key

    def _load_cubin_executable(
        self,
        key: tuple,
        dtype: torch.dtype,
        has_bias: bool,
        has_residual: bool,
        has_mask: bool,
        K: int,
        N: int,
        M: int,
        source_error: Exception | None = None,
    ):
        tile_params = get_tile_params(self._sm_version, K, N, M)
        kernel_abi = get_kernel_abi(self._sm_version, K, N)
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
                    has_residual,
                    has_mask,
                    kernel_abi,
                    tile_params,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the gated sigmoid CUBIN path, "
                    f"but no CUBIN is available for SM{self._sm_version}, "
                    f"dtype={dtype}, has_bias={has_bias}, has_residual={has_residual}, "
                    f"has_mask={has_mask}, K={K}, N={N}, m_bucket={m_bucket}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL gated sigmoid: using CUBIN kernel for SM{self._sm_version}, "
            f"dtype={dtype}, has_bias={has_bias}, has_residual={has_residual}, has_mask={has_mask}, "
            f"K={K}, N={N}, m_bucket={m_bucket}"
        )
        return executable

    def _resolve_source_kernel(
        self, ct_dtype: type, has_bias: bool, has_residual: bool, has_mask: bool, K: int, N: int, M: int
    ):
        source = load_source_module(__package__)
        config = get_kernel_config(self._sm_version, K, N, M)
        if not config.can_implement(ct_dtype):
            raise RuntimeError(f"Gated sigmoid kernel cannot implement dtype={ct_dtype}")

        kernel = config.kernel_factory(ct_dtype, has_bias, has_residual, has_mask)
        needed = kernel_smem_bytes(kernel, ct_dtype)
        limit = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        if needed > limit:
            raise RuntimeError(f"gated sigmoid needs {needed} B dynamic shared memory, but the device allows {limit} B")
        return kernel, source.compile_gated_sigmoid_source, config.kernel_abi

    def _load_or_compile_source(
        self,
        key: tuple,
        kernel: Any,
        compile_source: Any,
        kernel_abi: str,
        ct_dtype: type,
        has_bias: bool,
        has_residual: bool,
        has_mask: bool,
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
            f"K={K}, N={N}, m_range={_classify_m_range(M)!r}, has_bias={has_bias}, "
            f"has_residual={has_residual}, has_mask={has_mask}, kernel_abi={kernel_abi}"
        )
        executable = compile_source(self.compile, kernel, ct_dtype, has_bias, has_residual, has_mask, kernel_abi)
        GatedSigmoidCuTe._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL gated sigmoid: compilation done")
        return executable

    def _get_or_compile(
        self,
        ct_dtype: type,
        dtype: torch.dtype,
        has_bias: bool,
        has_residual: bool,
        has_mask: bool,
        K: int,
        N: int,
        M: int,
        key: tuple,
    ):
        executable = GatedSigmoidCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        flags = (dtype, has_bias, has_residual, has_mask, K, N, M)
        if force_cubin:
            GatedSigmoidCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, *flags)

        try:
            kernel, compile_source, kernel_abi = self._resolve_source_kernel(
                ct_dtype, has_bias, has_residual, has_mask, K, N, M
            )
        except ImportError as source_error:
            return self._load_cubin_executable(key, *flags, source_error)
        return self._load_or_compile_source(
            key, kernel, compile_source, kernel_abi, ct_dtype, has_bias, has_residual, has_mask, K, N, M
        )

    def __call__(
        self,
        s: torch.Tensor,
        weight: torch.Tensor,
        mha_out: torch.Tensor,
        bias: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fused gated sigmoid GEMM.

        Args:
            s: Gate input with shape ``[..., K]``.
            weight: Projection weight with shape ``[N, K]``.
            mha_out: Attention output with shape ``[..., N]``. Its leading
                dimensions may differ from ``s`` in one dimension where
                ``s`` has size 1, or may have a different shape with the same
                flattened row count.
            bias: Optional projection bias with shape ``[N]``.
            output: Optional output tensor matching ``mha_out``; it may alias
                ``mha_out`` or ``residual``.
            residual: Optional tensor matching ``mha_out``, added to the result.
            mask: Optional row mask with one element per ``mha_out`` row,
                multiplied into the gated product.

        Returns:
            Tensor with the same shape as ``mha_out``.
        """
        K = s.shape[-1]
        N_out = weight.shape[0]

        def vanilla() -> torch.Tensor:
            return _invoke_vanilla_gated_sigmoid(s, weight, mha_out, bias, output, residual, mask)

        if (
            weight.ndim != 2
            or weight.shape[1] != K
            or mha_out.ndim < 1
            or mha_out.shape[-1] != N_out
            or (bias is not None and (bias.ndim != 1 or bias.shape[0] != N_out))
        ):
            return vanilla()

        s_2d = s.reshape(-1, K)
        M_s = s_2d.shape[0]
        mha_2d = mha_out.reshape(-1, N_out)
        M_out = mha_2d.shape[0]

        def supports_row_major_2d(tensor: torch.Tensor) -> bool:
            return tensor.ndim == 2 and tensor.layout == torch.strided and tensor.stride(1) == 1

        def view_like_mha(tensor: torch.Tensor) -> torch.Tensor | None:
            if (
                tensor.shape != mha_out.shape
                or tensor.dtype != mha_out.dtype
                or tensor.device != mha_out.device
                or tensor.layout != torch.strided
            ):
                return None
            try:
                return tensor.view(M_out, N_out)
            except RuntimeError:
                return None

        output_2d: torch.Tensor | None = None
        if output is not None:
            output_2d = view_like_mha(output)
            if output_2d is None:
                return vanilla()
        residual_2d: torch.Tensor | None = None
        if residual is not None:
            residual_2d = view_like_mha(residual)
            if residual_2d is None:
                return vanilla()
        mask_rows: torch.Tensor | None = None
        if mask is not None:
            if mask.numel() != M_out or mask.device != s.device:
                return vanilla()
            mask_rows = mask.reshape(M_out).to(dtype=s.dtype).contiguous()

        operands = (s_2d, weight, mha_2d)
        if (
            not all(supports_row_major_2d(tensor) for tensor in operands)
            or any(tensor.dtype != s.dtype or tensor.device != s.device for tensor in operands[1:])
            or (
                bias is not None
                and (bias.ndim != 1 or bias.stride(0) != 1 or bias.dtype != s.dtype or bias.device != s.device)
            )
            or (output_2d is not None and not supports_row_major_2d(output_2d))
            or (residual_2d is not None and not supports_row_major_2d(residual_2d))
        ):
            return vanilla()

        # Allow one broadcast leading dimension.
        s_lead = s.shape[:-1]
        mha_lead = mha_out.shape[:-1]
        mult = 1
        inner: int | None = None
        # Triangle attention retains the batched gate input while flattening
        # its reusable attention-output buffer. Equal flattened row counts are
        # still the same row-wise operation and need no broadcast.
        if M_s == M_out:
            inner = M_s
        elif s_lead != mha_lead:
            mismatched = [i for i in range(len(mha_lead)) if i >= len(s_lead) or s_lead[i] != mha_lead[i]]
            if len(s_lead) != len(mha_lead) or len(mismatched) != 1 or s_lead[mismatched[0]] != 1:
                return vanilla()
            bcast_dim = mismatched[0]
            mult = int(mha_lead[bcast_dim])
            inner = 1
            for d in range(bcast_dim + 1, len(mha_lead)):
                inner *= int(mha_lead[d])

        if inner is None:
            inner = M_s

        has_bias = bias is not None
        has_residual = residual is not None
        has_mask = mask is not None
        m_range = _classify_m_range(M_s)
        compile_key = (self._sm_version, s.dtype, has_bias, has_residual, has_mask, K, N_out, m_range)

        force_cubin = self.force_cubin()
        if compile_key == self._last_key and (
            not force_cubin or isinstance(self._last_exe, CuTeDSLKernelLibraryExecutable)
        ):
            exe = self._last_exe
        else:
            ct_dtype = _cutlass_dtype(s_2d)
            exe = self._get_or_compile(ct_dtype, s.dtype, has_bias, has_residual, has_mask, K, N_out, M_s, compile_key)
            self._last_key = compile_key
            self._last_exe = exe
            self._last_abi = get_kernel_abi(self._sm_version, K, N_out)

        # Absent operands are omitted from the kernel ABI.
        out_2d = output_2d if output_2d is not None else torch.empty_like(mha_2d)
        args = (s_2d, weight, bias, mha_2d, residual_2d, mask_rows, out_2d, mult, inner)
        if self._last_abi == "sm90":
            # Every sample of a gate tile runs in one CTA.
            args += (mult,)
        launch_compiled_kernel(exe, *args)

        if output is not None:
            return output
        if mha_out.ndim == 2:
            return out_2d
        return out_2d.view(mha_out.shape[:-1] + (N_out,))
