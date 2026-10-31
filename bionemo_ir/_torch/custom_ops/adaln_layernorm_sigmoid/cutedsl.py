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
"""Source-or-CUBIN interface for fused LayerNorm/RMSNorm and sigmoid gating."""

from __future__ import annotations

import warnings
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    cute_numeric,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import (
    _BUCKET_BIG_M,
    _DTYPES,
    SHIPPED_N,
    SUPPORTED_SMS,
    bucket_variants,
    config_identity,
    is_cubin_representable,
)
from ._cubin import AdaLNLayerNormSigmoidCubinExecutable
from .ops import _invoke_vanilla_adaln_layernorm_sigmoid

#: Per device, the empty mask a maskless call passes; the kernel keeps every row for it.
_NO_MASK: dict[torch.device, torch.Tensor] = {}

__all__ = [
    "AdaLNLayerNormSigmoidCuTe",
    "SHIPPED_N",
    "SUPPORTED_SMS",
    "_BUCKET_BIG_M",
    "_invoke_vanilla_adaln_layernorm_sigmoid",
]


class AdaLNLayerNormSigmoidCuTe(CuteKernelCache):
    """Cached backend for fused LayerNorm/RMSNorm and sigmoid gating."""

    _compiled_cache: dict[tuple, Any] = {}
    # Variants the CUBIN pack lacked, warned about once per process whatever the instance or mode.
    _missing_variants_warned: set[tuple] = set()

    def __init__(self, rms_norm: bool = False):
        if not isinstance(rms_norm, bool):
            raise TypeError(f"rms_norm must be bool, got {type(rms_norm).__name__}")
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._rms_norm = rms_norm
        self._schedule_cache: dict[tuple, list[tuple[int | None, Any]]] = {}

    def _disk_cache_key(self, key: tuple) -> tuple:
        return ("adaln_layernorm_sigmoid_cute",) + key

    def _load_cubin_executable(
        self,
        key: tuple,
        dtype: torch.dtype,
        N: int,
        geometry: tuple[int, int],
        source_error: Exception | None = None,
        cfg: dict | None = None,
    ) -> Any | None:
        """Load the packaged CUBIN for one bucket, or ``None`` when the pack has no image for it.

        The pack can lag the bucket schedule, for example between a schedule
        change and its CUBIN refresh. ``None`` sends that bucket to PyTorch, as
        a source-free build does on an SM without a payload, and warns once per
        variant, since the pack should cover every scheduled variant.
        """
        threads_per_row, num_threads = geometry
        # Reject knobs that are not represented by the CUBIN registry key.
        if cfg is not None and not is_cubin_representable(cfg):
            unrepresentable = sorted(set(config_identity(cfg)) - set(config_identity({})))
            raise RuntimeError(
                f"no AdaLN CUBIN can represent these kernel knobs: {unrepresentable}. "
                f"Payloads are selected by (sm, dtype, N, geometry) only, so this config is "
                f"indistinguishable from the default build; add the knob to the builder's "
                f"shipping matrix and registry key before tuning it."
            )
        try:
            executable = populate_compiled_cache_from_library(
                AdaLNLayerNormSigmoidCuTe._compiled_cache,
                key,
                "adaln_layernorm_sigmoid",
                lambda library, launcher: AdaLNLayerNormSigmoidCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    dtype,
                    N,
                    threads_per_row,
                    num_threads,
                    self._rms_norm,
                ),
            )
        except CuTeDSLKernelVariantUnavailable as missing_variant:
            if key not in AdaLNLayerNormSigmoidCuTe._missing_variants_warned:
                AdaLNLayerNormSigmoidCuTe._missing_variants_warned.add(key)
                warnings.warn(
                    f"{missing_variant}; this bucket runs in PyTorch until the CUBIN packs are refreshed",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the AdaLN CUBIN path, but the CuTeDSL kernel "
                    f"library cannot load the adaln_layernorm_sigmoid family for SM{self._sm_version}"
                ) from library_error
            raise library_error from source_error
        logger.info(f"CuTeDSL adaln_layernorm_sigmoid: using CUBIN kernel for SM{self._sm_version}, key={key}")
        return executable

    def _assert_smem_fits(self, kernel: Any, ct_dtype: type, N: int) -> None:
        """Reject source variants that exceed the device shared-memory limit."""
        needed = kernel.dynamic_smem_bytes()
        limit = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        if needed > limit:
            raise RuntimeError(
                f"adaln_layernorm_sigmoid needs {needed} B of dynamic shared memory for "
                f"dtype={ct_dtype.__name__}, N={N}, but SM{self._sm_version} allows only "
                f"{limit} B. N > 8192 selects staged mode, whose sX/sS/sSb tiles scale with "
                f"the element width; use a narrower dtype, a smaller N, or a device with a "
                f"larger opt-in shared-memory limit."
            )

    @staticmethod
    def _resolve_source_tools():
        source = load_source_module(__package__)
        return source.make_kernel, source.compile_adaln_source

    def _load_or_compile_source(
        self,
        key: tuple,
        ct_dtype: type,
        N: int,
        cfg: dict,
        make_kernel: Any,
        compile_source: Any,
    ):
        # Validate shared memory even for disk-cached executables.
        kernel = make_kernel(ct_dtype, N, cfg, self._rms_norm)
        self._assert_smem_fits(kernel, ct_dtype, N)

        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL adaln_layernorm_sigmoid: loaded cached kernel for SM{self._sm_version}, key={key}")
            AdaLNLayerNormSigmoidCuTe._compiled_cache[key] = executable
            return executable

        logger.info(
            f"CuTeDSL adaln_layernorm_sigmoid: compiling kernel for "
            f"SM{self._sm_version}, dtype={ct_dtype.__name__}, N={N}, cfg={cfg}"
        )
        executable = compile_source(self.compile, kernel, ct_dtype, N)
        AdaLNLayerNormSigmoidCuTe._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL adaln_layernorm_sigmoid: compilation done")
        return executable

    def _compile_bucket(self, dtype: torch.dtype, N: int, cfg: dict, geometry: tuple[int, int]) -> Any:
        # Include all config knobs, not only resolved launch geometry.
        key = (self._sm_version, dtype, N, self._rms_norm, geometry, config_identity(cfg))
        executable = AdaLNLayerNormSigmoidCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        if force_cubin:
            AdaLNLayerNormSigmoidCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, dtype, N, geometry, cfg=cfg)

        try:
            make_kernel, compile_source = self._resolve_source_tools()
        except ImportError as source_error:
            return self._load_cubin_executable(key, dtype, N, geometry, source_error, cfg=cfg)
        return self._load_or_compile_source(key, cute_numeric(dtype), N, cfg, make_kernel, compile_source)

    def _get_or_build_schedule(self, dtype: torch.dtype, N: int) -> list[tuple[int | None, Any]]:
        if dtype not in _DTYPES:
            raise TypeError(f"adaln_layernorm_sigmoid expects float16, bfloat16 or float32; got {dtype}")
        shape_key = (self._sm_version, dtype, N, self._rms_norm, self.force_cubin())
        cached = self._schedule_cache.get(shape_key)
        if cached is not None:
            return cached
        schedule: list[tuple[int | None, Any]] = []
        for m_max, cfg, geometry in bucket_variants(self._sm_version, N):
            schedule.append((m_max, self._compile_bucket(dtype, N, cfg, geometry)))
        self._schedule_cache[shape_key] = schedule
        return schedule

    @staticmethod
    def _pick(schedule: list[tuple[int | None, Any]], M: int) -> Any:
        for m_max, executable in schedule:
            if m_max is None or M <= m_max:
                return executable
        raise RuntimeError(f"No bucket matched M={M} (last bucket must be catch-all)")

    def __call__(
        self,
        x: torch.Tensor,
        s_scale: torch.Tensor,
        s_bias: torch.Tensor,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run fused LayerNorm/RMSNorm and sigmoid gating.

        A bucket whose variant the CUBIN pack lacks runs in PyTorch when the
        CUBIN path is forced (``CUTEDSL_FORCE_CUBIN=1``) or the kernel sources
        are absent.

        Args:
            x: Contiguous input with shape ``[..., N]``.
            s_scale: Sigmoid scale with shape ``[..., N]``. It may differ from
                ``x`` in one leading dimension whose size is 1.
            s_bias: Bias with the same shape as ``s_scale``.
            out: Optional contiguous output matching the shape and dtype of
                ``x``. The operation writes to ``x`` when omitted.
            eps: Normalization epsilon.
            mask: Optional contiguous boolean keep flag per row of ``s_scale``, such as ``[B, I]`` for
                ``s_scale`` ``[B, I, N]`` or ``[B, 1, I, N]``. The kernel reads it with ``s_scale``'s own row
                index, so a sample axis of ``x`` broadcasts as ``s_scale`` does, and writes each row whose
                flag is false as zeros, as ``out.masked_fill_(~mask, 0)`` after the op would.

        Returns:
            The output tensor with shape ``[..., N]``.
        """
        if out is None:
            out = x
        elif out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError(
                f"out must match x in shape/dtype; got out={tuple(out.shape)}/{out.dtype} vs "
                f"x={tuple(x.shape)}/{x.dtype}"
            )

        # Allow one broadcast leading dimension.
        mult = 1
        inner = 1
        if s_scale.shape != x.shape:
            # Reject unsupported broadcast patterns explicitly.
            mismatched = (
                [i for i in range(x.ndim - 1) if s_scale.shape[i] != x.shape[i]] if s_scale.ndim == x.ndim else None
            )
            if (
                mismatched is None
                or s_scale.shape[-1] != x.shape[-1]
                or len(mismatched) != 1
                or s_scale.shape[mismatched[0]] != 1
            ):
                raise ValueError(
                    f"s_scale must match x.shape or differ in exactly one "
                    f"leading dim of size 1; got s_scale={tuple(s_scale.shape)} vs x={tuple(x.shape)}"
                )
            bcast_dim = mismatched[0]
            mult = int(x.shape[bcast_dim])
            inner = 1
            for d in range(bcast_dim + 1, x.ndim - 1):
                inner *= int(x.shape[d])

        # Writable tensors must not be silently copied by reshape.
        for name, t in (("x", x), ("out", out)):
            if not t.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous (kernel writes in-place); "
                    f"got stride={t.stride()}. Call .contiguous() before passing."
                )

        N = x.shape[-1]
        x_2d = x.view(-1, N)
        out_2d = out.view(-1, N)
        s_scale_2d = s_scale.reshape(-1, N)
        s_bias_2d = s_bias.reshape(-1, N)
        M = x_2d.shape[0]
        if mask is None:
            mask_rows = _NO_MASK.get(x.device)
            if mask_rows is None:
                mask_rows = _NO_MASK.setdefault(x.device, torch.empty(0, dtype=torch.uint8, device=x.device))
        elif mask.dtype != torch.bool or mask.numel() != s_scale_2d.shape[0] or not mask.is_contiguous():
            raise ValueError(
                f"mask must be a contiguous bool tensor with one flag per s_scale row ({s_scale_2d.shape[0]}); "
                f"got {tuple(mask.shape)}/{mask.dtype}"
            )
        else:
            # The bool storage read as bytes: no copy, no launch.
            mask_rows = mask.view(-1).view(torch.uint8)

        schedule = self._get_or_build_schedule(x.dtype, N)
        executable = self._pick(schedule, M)
        if executable is None:
            return _invoke_vanilla_adaln_layernorm_sigmoid(
                x, s_scale, s_bias, out=out, eps=eps, rms_norm=self._rms_norm, mask=mask
            )

        launch_compiled_kernel(executable, x_2d, s_scale_2d, s_bias_2d, out_2d, mask_rows, eps, mult, inner)
        return out
