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
"""Source-or-CUBIN interface for the fused transition MLP."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
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
    KERNEL_ABIS,
    TransitionMlpVariant,
    buckets,
    get_tile_params,
    nearest_bucket,
    pseudo_seqlen,
    shipped_variants,
)
from ._cubin import TransitionMlpCubinExecutable

__all__ = ["TransitionMlpCuTe", "TransitionMlpOp"]

_TORCH_TO_CUTLASS_DTYPE = {torch.bfloat16: cutlass.BFloat16}
# TMA and cp.async move these operands in 16-byte vectors, so base addresses and row strides need 16 bytes.
_ROW_ALIGNMENT_BYTES = 16


def _rows_aligned(tensor: torch.Tensor) -> bool:
    row_bytes = tensor.stride(0) * tensor.element_size()
    return (
        tensor.stride(-1) == 1
        and tensor.data_ptr() % _ROW_ALIGNMENT_BYTES == 0
        and row_bytes % _ROW_ALIGNMENT_BYTES == 0
    )


def _w1_rows(variant: TransitionMlpVariant) -> int:
    return 2 * variant.hidden if variant.activation == "silu_gate" else variant.hidden


class TransitionMlpCuTe(CuteKernelCache):
    """Cached backend for ``[residual +] mask * (act(x @ w1.T + b1) @ w2.T + b2)``.

    Each call runs the tile its variant's configs tune for the bucket nearest the call's pseudo
    sequence length, ``round(sqrt(rows))``.

    Args:
        sm_version: The SM whose configs and kernel to use; defaults to this GPU's. An SM90 GPU can
            also run the SM80 kernel.
    """

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self, sm_version: int | None = None) -> None:
        if sm_version is None:
            major, minor = torch.cuda.get_device_capability()
            sm_version = major * 10 + minor
        self._sm_version = sm_version
        self._variants = shipped_variants(sm_version)
        self._anchors: dict[TransitionMlpVariant, tuple[int, ...]] = {}
        self._unavailable: set[tuple] = set()

    def bucket(self, variant: TransitionMlpVariant, rows: int) -> int:
        """The bucket a call of ``variant`` over ``rows`` flattened rows runs."""
        anchors = self._anchors.get(variant)
        if anchors is None:
            anchors = self._anchors[variant] = buckets(self._sm_version, variant)
        return nearest_bucket(anchors, pseudo_seqlen(rows))

    def _key(self, dtype: torch.dtype, variant: TransitionMlpVariant, bucket: int) -> tuple:
        return (self._sm_version, dtype, *variant, bucket)

    def ships(self, dtype: torch.dtype, variant: TransitionMlpVariant) -> bool:
        """Whether this build can run the variant, from source or from a packaged CUBIN.

        A source-free build, or ``CUTEDSL_FORCE_CUBIN``, needs the packaged image. Without one the
        caller keeps its own path rather than failing on a family that has not shipped.
        """
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        anchors = buckets(self._sm_version, variant)
        if not anchors:
            return False
        try:
            self._load_cubin_executable(self._key(dtype, variant, anchors[0]), dtype, variant, anchors[0])
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _load_cubin_executable(
        self,
        key: tuple,
        dtype: torch.dtype,
        variant: TransitionMlpVariant,
        bucket: int,
        source_error: Exception | None = None,
    ) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                TransitionMlpCuTe._compiled_cache,
                key,
                "transition_mlp",
                lambda library, launcher: TransitionMlpCubinExecutable(
                    library, launcher, self._sm_version, dtype, variant, bucket
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the transition MLP CUBIN path, but no CUBIN is "
                    f"available for SM{self._sm_version}, dtype={dtype}, {variant}, bucket={bucket}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL transition MLP: using CUBIN kernel for SM{self._sm_version}, dtype={dtype}, {variant}, "
            f"bucket={bucket}"
        )
        return executable

    def _get_or_compile(self, dtype: torch.dtype, variant: TransitionMlpVariant, bucket: int) -> Any:
        key = self._key(dtype, variant, bucket)
        executable = TransitionMlpCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TransitionMlpCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, dtype, variant, bucket)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, dtype, variant, bucket, source_error)

        tile_params = get_tile_params(self._sm_version, variant, bucket)
        if tile_params is None:
            raise ValueError(f"transition MLP has no SM{self._sm_version} config for {variant} at bucket {bucket}")
        kernel_abi = KERNEL_ABIS[self._sm_version]
        # Buckets that share a tile share its compile.
        disk_key = ("transition_mlp_cute_v4", *key[:-1], kernel_abi, tuple(sorted(tile_params.items())))
        executable = TransitionMlpCuTe._compiled_cache.get(disk_key) or self.load_from_cache(disk_key)
        if executable is None:
            logger.info(f"CuTeDSL transition MLP: compiling kernel for SM{self._sm_version}, {variant}, {tile_params}")
            executable = source.compile_transition_mlp_source(
                self.compile, _TORCH_TO_CUTLASS_DTYPE[dtype], variant, tile_params, kernel_abi
            )
            self.save_to_cache(disk_key, executable)
        TransitionMlpCuTe._compiled_cache[disk_key] = executable
        TransitionMlpCuTe._compiled_cache[key] = executable
        return executable

    def accepts(
        self,
        variant: TransitionMlpVariant,
        like: torch.Tensor,
        mask: torch.Tensor | None,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
    ) -> bool:
        """Whether a call with these operands can run, checked before the caller computes ``x``.

        ``like`` is the residual, or for a variant without one any tensor shaped like ``x``.
        """
        width, hidden = variant.width, variant.hidden
        dtype = like.dtype
        biases = (b1, b2)
        if (
            variant not in self._variants
            or not like.is_cuda
            or dtype not in _TORCH_TO_CUTLASS_DTYPE
            or like.shape[-1] != width
            or w1.shape != (_w1_rows(variant), width)
            or w2.shape != (width, hidden)
            or any((bias is not None) != variant.has_bias for bias in biases)
            or (mask is not None) != variant.has_mask
        ):
            return False
        present = [t for t in (w1, w2, *biases) if t is not None]
        if any(t.dtype != dtype or t.device != like.device or not t.is_contiguous() for t in present):
            return False
        if variant.has_bias and (b1.shape != (w1.shape[0],) or b2.shape != (width,)):
            return False
        # ``run`` casts the mask's dtype but not its device.
        if variant.has_mask and (mask.device != like.device or mask.numel() * width != like.numel()):
            return False
        read = [w1, w2]
        if variant.has_residual:
            read.append(like.reshape(like.numel() // width, width))
        return all(_rows_aligned(t) for t in read)

    def run(
        self,
        variant: TransitionMlpVariant,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run ``variant``, or return ``None`` when this call needs the caller's own path."""
        width = variant.width
        like = residual if variant.has_residual else x
        if (
            (residual is not None) != variant.has_residual
            or x.shape != like.shape
            or x.dtype != like.dtype
            or x.device != like.device
            or not self.accepts(variant, like, mask, w1, b1, w2, b2)
        ):
            return None
        dtype = like.dtype
        rows = like.numel() // width
        x_rows = x.reshape(rows, width)
        residual_rows = None if residual is None else residual.reshape(rows, width)
        if out is None:
            out = torch.empty(like.shape, dtype=dtype, device=like.device)
        elif out.shape != like.shape or out.dtype != dtype or out.device != like.device:
            return None
        out_rows = out.view(rows, width) if out.is_contiguous() else None
        if out_rows is None or not _rows_aligned(x_rows) or not _rows_aligned(out_rows):
            return None
        mask_rows = None if mask is None else mask.reshape(rows).to(dtype=dtype).contiguous()
        if rows == 0:
            return out

        bucket = self.bucket(variant, rows)
        key = (dtype, *variant, bucket)
        if key in self._unavailable:
            return None
        try:
            executable = self._get_or_compile(dtype, variant, bucket)
        except CuTeDSLKernelLibraryError as error:
            logger.warning(
                f"CuTeDSL transition MLP unavailable for {variant} at bucket {bucket}; using the unfused path: {error}"
            )
            self._unavailable.add(key)
            return None
        # A source-backed launch takes its stream from the current device, not the operands'. Switch
        # only when they differ: an unconditional guard costs several microseconds a call.
        on_device = like.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(like.device):
            launch_compiled_kernel(executable, x_rows, w1, b1, w2, b2, residual_rows, mask_rows, out_rows)
        return out


@dataclass(frozen=True)
class TransitionMlpOp:
    """The fused op bound to one variant; see :func:`get_transition_mlp_op`."""

    backend: TransitionMlpCuTe
    variant: TransitionMlpVariant

    def accepts(
        self,
        like: torch.Tensor,
        mask: torch.Tensor | None,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
    ) -> bool:
        """Whether a call with these operands can run, checked before the caller computes ``x``.

        ``like`` is the residual, or for a variant without one any tensor shaped like ``x``.
        """
        return self.backend.accepts(self.variant, like, mask, w1, b1, w2, b2)

    def __call__(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """``[residual +] mask * (act(x @ w1.T + b1) @ w2.T + b2)``, or ``None`` if this call can't run.

        Args:
            x: Normalized input with shape ``[..., C]``.
            w1: ``[H, C]``, or ``[2H, C]`` holding value rows then gate rows for ``"silu_gate"``.
            b1: One bias per ``w1`` row, present exactly when the variant has biases.
            w2: Second weight with shape ``[C, H]``.
            b2: ``[C]`` bias, present exactly when the variant has biases.
            residual: Residual with the shape of ``x``, present exactly when the variant adds one.
            mask: Row mask whose elements match ``x``'s leading dimensions, present exactly when
                the variant is masked.
            out: Output buffer shaped like ``x``; it may be ``residual`` itself.
        """
        return self.backend.run(self.variant, x, w1, b1, w2, b2, residual, mask, out)
