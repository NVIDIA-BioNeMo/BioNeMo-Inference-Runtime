# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Triangle attention backend using precompiled or source CuTe left-mask kernels.

The kernel variant is selected from ``torch.cuda.get_device_capability()`` and
the tuning bundle's ``kernel_abi``. If the source implementation is absent, the
executable cache is populated from the packaged ``_cutedsl_kernels`` library
instead.

Requires the per-row pair mask to be left-aligned (``1...1 0...0``), as produced
by ``pair_mask = seq_mask[..., None] * seq_mask[..., None, :]`` under the
OpenFold/Boltz padding convention. The kernel takes one ``actual_s_kv`` count of
leading 1s per ``(B, I)`` row instead of a ``[B, I, J]`` mask.

Logical input shapes, and the flattened / padded forms passed to the kernel:
  Q, K, V, O  : [B, I, J, H, D]   ->  [B*I, J, H, D]
  bias        : [B, H, J, J]      ->  [B, H, J, ceil(J / align) * align]
  actual_s_kv : [B, I] int32      ->  [B*I]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cutlass
import torch
import torch.nn.functional as F

from bionemo_ir._torch._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryError,
    launch_compiled_kernel,
    populate_compiled_cache_from_library,
)
from bionemo_ir._torch._kernel_source_loader import load_source_module
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ..interface import AttentionBackend, AttentionMetadata
from ._config import (
    _TRI_CONFIGS_DIR,
    TriangleAttentionLeftMaskKernelConfig,
    _build_sm80_config,
    _build_sm90_config,
    get_kernel_config,
    get_nearest_bucket,
)
from ._cubin import TriangleAttentionCubinExecutable

__all__ = [
    "TriangleAttentionCuTeLeftMask",
    "TriangleAttentionCuTeLeftMaskMetadata",
    "TriangleAttentionLeftMaskKernelConfig",
    "_TRI_CONFIGS_DIR",
    "_build_sm80_config",
    "_build_sm90_config",
    "get_kernel_config",
]

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


class TriangleAttentionCuTeLeftMaskMetadata(AttentionMetadata):
    qkv_packed: bool = True


# The Hopper kernel softmaxes via exp2 fastmath, so it takes a log2-form scale.
_LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class _TriangleAttentionVariant:
    dtype: torch.dtype
    head_dim: int
    bucket: int
    qkv_packed: bool


@dataclass(frozen=True)
class _TriangleAttentionLaunchInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    bias: torch.Tensor
    actual_s_kv: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    variant: _TriangleAttentionVariant
    ct_dtype: type[cutlass.Numeric]
    align_elems: int
    softmax_scale: float
    batch_size: int
    i_dim: int
    seqlen: int
    num_heads: int


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _pad_last_dim(t: torch.Tensor, new_size: int, value: float = 0.0) -> torch.Tensor:
    cur = t.size(-1)
    if cur >= new_size:
        return t
    return F.pad(t, (0, new_size - cur), value=value)


def _cutlass_dtype(t: torch.Tensor) -> type[cutlass.Numeric]:
    ty = _TORCH_TO_CUTLASS_DTYPE.get(t.dtype)
    if ty is None:
        raise TypeError(f"SM80 triangle attention expects float16 or bfloat16; got {t.dtype}")
    return ty


def _resolve_lse_buffer(
    output_lse: torch.Tensor | None,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Reuse ``output_lse`` if it matches the kernel contract, else allocate.

    The contract is fixed: ``shape``, dtype float32 (qk_acc_dtype), ``device``.
    A mismatch falls back to an internal allocation rather than failing, as the
    ``output`` parameter does.
    """
    if (
        output_lse is not None
        and tuple(output_lse.shape) == shape
        and output_lse.dtype == torch.float32
        and output_lse.device == device
    ):
        return output_lse
    return torch.empty(*shape, dtype=torch.float32, device=device)


def _to_actual_s_kv_int32(actual_s_kv: torch.Tensor, batch_size: int, i_dim: int) -> torch.Tensor:
    """Normalize ``actual_s_kv`` to a contiguous ``[B*I]`` int32 tensor.

    Accepts ``[B, I]`` (per-row leading-1s count), ``[B*I]`` (already flat), or
    ``[B]`` (broadcast across I; typical for OpenFold2, where ``pair_mask`` is
    the outer product of a single ``seq_mask`` per batch).
    """
    if actual_s_kv.dtype != torch.int32:
        actual_s_kv = actual_s_kv.to(torch.int32)
    if actual_s_kv.ndim == 1 and actual_s_kv.shape[0] == batch_size * i_dim:
        return actual_s_kv.contiguous()
    if actual_s_kv.ndim == 2 and actual_s_kv.shape == (batch_size, i_dim):
        return actual_s_kv.contiguous().view(batch_size * i_dim)
    if actual_s_kv.ndim == 1 and actual_s_kv.shape[0] == batch_size:
        return (
            actual_s_kv.contiguous().view(batch_size, 1).expand(batch_size, i_dim).contiguous().view(batch_size * i_dim)
        )
    raise ValueError(
        f"actual_s_kv must be [B,I], [B*I], or [B] "
        f"(with B={batch_size}, I={i_dim}); "
        f"got shape {tuple(actual_s_kv.shape)}"
    )


class TriangleAttentionCuTeLeftMask(CuteKernelCache, AttentionBackend[TriangleAttentionCuTeLeftMaskMetadata]):
    """Triangle attention backend for left-aligned pair masks.

    Source kernels use the existing CuTeDSL JIT/disk cache. Public builds can
    omit those source classes and transparently use packaged CUBIN launchers
    through the same in-process executable cache. ``CUTEDSL_FORCE_CUBIN=1``
    takes the CUBIN path even when the sources are importable.

    Implements lines 5 and 6 of Algorithm 14 from
    https://www.nature.com/articles/s41586-024-07487-w#Sec19.

    Expects inputs in the kernel's logical shapes:
      Q, K, V     : [B, I, J, H, D]
      biases      : [actual_s_kv, pair_bias]
        actual_s_kv: [B, I] / [B*I] / [B]    int32 (count of leading 1s)
        pair_bias  : [B, H, J, J]
    """

    Metadata = TriangleAttentionCuTeLeftMaskMetadata

    _compiled_cache: dict[tuple[int, _TriangleAttentionVariant], Any] = {}

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_executable = None
        self._last_variant: _TriangleAttentionVariant | None = None

    def _disk_cache_key(self, variant: _TriangleAttentionVariant) -> tuple:
        return (
            "triangle_attn_cute_left_mask",
            self._sm_version,
            variant.dtype,
            variant.head_dim,
            variant.bucket,
            variant.qkv_packed,
        )

    def _resolve_source_kernel(
        self,
        variant: _TriangleAttentionVariant,
        ct_dtype: type[cutlass.Numeric],
    ) -> tuple[TriangleAttentionLeftMaskKernelConfig, Any]:
        config = get_kernel_config(self._sm_version, variant.head_dim, variant.bucket)
        if not config.can_implement(ct_dtype, variant.head_dim):
            raise RuntimeError(
                f"Triangle attention (left-mask) kernel cannot implement: dtype={ct_dtype}, D={variant.head_dim}"
            )
        return config, config.kernel_factory(variant.head_dim)

    def _load_cubin_executable(
        self,
        variant: _TriangleAttentionVariant,
        source_error: Exception | None = None,
    ):
        cache_key = (self._sm_version, variant)
        try:
            executable = populate_compiled_cache_from_library(
                TriangleAttentionCuTeLeftMask._compiled_cache,
                cache_key,
                "triangle_attention",
                lambda library, launcher: TriangleAttentionCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    variant.head_dim,
                    variant.bucket,
                    variant.dtype,
                    variant.qkv_packed,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel "
                    f"library cannot provide this variant: {library_error}"
                ) from library_error
            raise RuntimeError(
                "CuTeDSL triangle attention source is unavailable and the "
                "precompiled kernel library cannot provide this variant: "
                f"{library_error}"
            ) from source_error

        logger.info(
            f"CuTeDSL triangle attention (left-mask): using precompiled "
            f"CUBIN for SM{self._sm_version}, dtype={variant.dtype}, "
            f"head_dim={variant.head_dim}, bucket={variant.bucket}, "
            f"qkv_packed={variant.qkv_packed}"
        )
        return executable

    def _load_or_compile_source(
        self,
        variant: _TriangleAttentionVariant,
        config: TriangleAttentionLeftMaskKernelConfig,
        kernel,
        compile_source: Any,
        ct_dtype: type[cutlass.Numeric],
        align_elems: int,
        sm_scale: float,
        i_dim: int,
    ):
        cache_key = (self._sm_version, variant)
        disk_key = self._disk_cache_key(variant)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(
                f"CuTeDSL triangle attention (left-mask): loaded cached kernel "
                f"for SM{self._sm_version} ({config.arch}), "
                f"dtype={variant.dtype}, head_dim={variant.head_dim}, "
                f"bucket={variant.bucket}, "
                f"qkv_packed={variant.qkv_packed}"
            )
            TriangleAttentionCuTeLeftMask._compiled_cache[cache_key] = executable
            return executable

        logger.info(
            f"CuTeDSL triangle attention (left-mask): compiling kernel for "
            f"layer={self.layer_idx}, SM{self._sm_version} ({config.arch}), "
            f"dtype={variant.dtype}, head_dim={variant.head_dim}, "
            f"bucket={variant.bucket}, "
            f"qkv_packed={variant.qkv_packed}"
        )
        executable = compile_source(
            self.compile,
            kernel,
            config.arch,
            variant.head_dim,
            variant.qkv_packed,
            ct_dtype,
            align_elems,
            sm_scale,
            i_dim,
        )
        TriangleAttentionCuTeLeftMask._compiled_cache[cache_key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info(f"CuTeDSL triangle attention (left-mask): compilation done for layer={self.layer_idx}")
        return executable

    def _get_executable(
        self,
        variant: _TriangleAttentionVariant,
        ct_dtype: type[cutlass.Numeric],
        align_elems: int,
        sm_scale: float,
        i_dim: int,
    ):
        cache_key = (self._sm_version, variant)
        executable = TriangleAttentionCuTeLeftMask._compiled_cache.get(cache_key)
        if executable is not None:
            return executable

        if self.force_cubin():
            return self._load_cubin_executable(variant)

        try:
            source = load_source_module(__package__)
            config, kernel = self._resolve_source_kernel(variant, ct_dtype)
        except ImportError as source_error:
            return self._load_cubin_executable(variant, source_error)

        return self._load_or_compile_source(
            variant,
            config,
            kernel,
            source.compile_triangle_attention_source,
            ct_dtype,
            align_elems,
            sm_scale,
            i_dim,
        )

    def _prepare_launch_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        actual_s_kv: torch.Tensor,
        bias: torch.Tensor,
        qkv_packed: bool,
        output: torch.Tensor | None,
        output_lse: torch.Tensor | None,
    ) -> _TriangleAttentionLaunchInputs:
        if q.ndim == 4:
            leading_shape = q.shape[:-1]
            q = q.view(*leading_shape, self.num_heads, self.head_dim)
            k = k.view(*leading_shape, self.num_heads, self.head_dim)
            v = v.view(*leading_shape, self.num_heads, self.head_dim)

        padded_head_dim = _align_up(self.head_dim, 32)
        if padded_head_dim != self.head_dim:
            q = _pad_last_dim(q, padded_head_dim)
            k = _pad_last_dim(k, padded_head_dim)
            v = _pad_last_dim(v, padded_head_dim)

        if q.ndim != 5:
            raise ValueError(f"Q must be [B, I, J, H, D]; got ndim={q.ndim}")
        if k.shape != q.shape or v.shape != q.shape:
            raise ValueError("Q, K, and V must have the same shape")

        batch_size, i_dim, seqlen, num_heads, padded_head_dim = q.shape
        if num_heads != self.num_heads:
            raise ValueError(f"num_heads mismatch: tensor H={num_heads}, expected {self.num_heads}")

        ct_dtype = _cutlass_dtype(q)
        align_elems = 128 // ct_dtype.width
        flat_shape = (batch_size * i_dim, seqlen, num_heads, padded_head_dim)
        q_flat = q.reshape(flat_shape)
        k_flat = k.reshape(flat_shape)
        v_flat = v.reshape(flat_shape)

        if output is not None and output.shape == flat_shape and output.dtype == q.dtype and output.device == q.device:
            output_flat = output
        else:
            output_flat = torch.empty(flat_shape, dtype=q.dtype, device=q.device)

        padded_seqlen = _align_up(seqlen, align_elems)
        bias_padded = _pad_last_dim(bias.contiguous(), padded_seqlen)
        actual_s_kv_flat = _to_actual_s_kv_int32(actual_s_kv, batch_size, i_dim)
        if actual_s_kv_flat.device != q.device:
            raise ValueError(f"actual_s_kv must be on {q.device}; got {actual_s_kv_flat.device}")

        lse_flat = _resolve_lse_buffer(
            output_lse,
            (batch_size * i_dim, seqlen, num_heads, 1),
            q.device,
        )
        S = int(round(math.sqrt(max(i_dim * seqlen, 1))))
        variant = _TriangleAttentionVariant(
            dtype=q_flat.dtype,
            head_dim=padded_head_dim,
            bucket=get_nearest_bucket(self._sm_version, padded_head_dim, S),
            qkv_packed=qkv_packed,
        )
        return _TriangleAttentionLaunchInputs(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            bias=bias_padded,
            actual_s_kv=actual_s_kv_flat,
            output=output_flat,
            lse=lse_flat,
            variant=variant,
            ct_dtype=ct_dtype,
            align_elems=align_elems,
            softmax_scale=float(self.head_dim**-0.5),
            batch_size=batch_size,
            i_dim=i_dim,
            seqlen=seqlen,
            num_heads=num_heads,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: TriangleAttentionCuTeLeftMaskMetadata | None = None,
        output: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run triangle attention via the CuTe left-mask kernel.

        Args:
            q: [B, I, J, H*D] or [B, I, J, H, D] (may be non-contiguous when qkv_packed)
            k: same logical shape as q
            v: same logical shape as q
            biases: [actual_s_kv, pair_bias]
                actual_s_kv: [B, I] / [B*I] / [B] int32 — count of leading 1s
                             along the KV axis (must be ``<= J``).
                pair_bias  : [B, H, J, J]
            output: Optional ``[B*I, J, H, D]`` buffer written in place.
            output_lse: Optional ``[B*I, J, H, 1]`` float32 buffer written in
                place, honored by both the Ampere and Hopper kernels. A
                shape/dtype/device mismatch falls back to an internal
                allocation.
        Returns:
            o: [B, I, J, H, D]
        """
        if biases is None or len(biases) < 2:
            raise ValueError("CuTeDSL triangle attention (left-mask) expects biases=[actual_s_kv, pair_bias]")
        actual_s_kv = biases[0]
        bias = biases[1]
        if metadata is None:
            metadata = TriangleAttentionCuTeLeftMaskMetadata()

        launch_inputs = self._prepare_launch_inputs(
            q,
            k,
            v,
            actual_s_kv,
            bias,
            getattr(metadata, "qkv_packed", True),
            output,
            output_lse,
        )
        if launch_inputs.variant == self._last_variant:
            executable = self._last_executable
        else:
            executable = self._get_executable(
                launch_inputs.variant,
                launch_inputs.ct_dtype,
                launch_inputs.align_elems,
                launch_inputs.softmax_scale,
                launch_inputs.i_dim,
            )
            self._last_variant = launch_inputs.variant
            self._last_executable = executable

        launch_compiled_kernel(
            executable,
            launch_inputs.q,
            launch_inputs.k,
            launch_inputs.v,
            launch_inputs.bias,
            launch_inputs.actual_s_kv,
            launch_inputs.output,
            launch_inputs.lse,
            launch_inputs.softmax_scale * _LOG2_E,
            launch_inputs.softmax_scale,
            launch_inputs.i_dim,
        )
        output_shape = (
            launch_inputs.batch_size,
            launch_inputs.i_dim,
            launch_inputs.seqlen,
            launch_inputs.num_heads,
            launch_inputs.variant.head_dim,
        )
        return launch_inputs.output.view(output_shape)[..., : self.head_dim]
