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
"""Pairwise attention backend using precompiled or source CuTe left-mask kernels.

Dispatches to :class:`FlashAttentionForwardAmpere` (SM80/86/89) or
:class:`HopperFusedMultiHeadAttentionForward` (SM90) on
``torch.cuda.get_device_capability()``. If the source implementation is absent,
the executable cache is populated from the packaged ``_cutedsl_kernels``
library instead.

Requires the KV-side mask to be left-aligned (``1...1 0...0``). The kernel then
takes one ``actual_s_kv`` count of leading 1s per batch instead of a ``[B, Sk]``
mask, so trailing blocks are skipped with no GMEM or SMEM mask traffic. The
tuned variant is the ``S=<anchor>`` entry nearest ``S = round(sqrt(Sq * Sk))``.

Logical input shapes, and the flattened / padded forms passed to the kernel:
  Q, K, V, O  : [*, S, H, D]      ->  [B*mult, S, H, ceil(D / 16) * 16]
  bias        : [*, H, Sq, Sk]    ->  [B, H, Sq, ceil(Sk / align) * align]
  actual_s_kv : [B] int32         ->  [B]            broadcasts over mult
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
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ..interface import AttentionBackend, AttentionMetadata
from ._config import (
    _PW_CONFIGS_DIR,
    PairwiseAttentionLeftMaskKernelConfig,
    _build_sm80_config,
    _build_sm90_config,
    get_kernel_config,
    get_nearest_bucket,
)
from ._cubin import PairwiseAttentionCubinExecutable
from ._source import compile_pairwise_attention_source

__all__ = [
    "PairwiseAttentionCuTeLeftMask",
    "PairwiseAttentionCuTeLeftMaskMetadata",
    "PairwiseAttentionLeftMaskKernelConfig",
    "_PW_CONFIGS_DIR",
    "_build_sm80_config",
    "_build_sm90_config",
    "get_kernel_config",
    "get_nearest_bucket",
]

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


class PairwiseAttentionCuTeLeftMaskMetadata(AttentionMetadata):
    kv_packed: bool = True


# The Hopper kernel softmaxes via exp2 fastmath, so it takes a log2-form scale.
_LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class _PairwiseAttentionVariant:
    dtype: torch.dtype
    head_dim: int
    bucket: int
    kv_packed: bool


@dataclass(frozen=True)
class _PairwiseAttentionLaunchInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    bias: torch.Tensor
    actual_s_kv: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    variant: _PairwiseAttentionVariant
    ct_dtype: type[cutlass.Numeric]
    align_elems: int
    softmax_scale: float
    batch_shape: tuple[int, ...]
    mult: int
    seqlen_q: int
    num_heads: int


def _compute_S(seqlen_q: int, seqlen_kv: int) -> int:
    """Map a possibly rectangular attention problem to its side-length axis."""
    return int(round(math.sqrt(max(seqlen_q * seqlen_kv, 1))))


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
        raise TypeError(f"Pairwise attention expects float16 or bfloat16; got {t.dtype}")
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


def _batch_size(actual_s_kv: torch.Tensor) -> int:
    """Infer the per-batch count carried by ``actual_s_kv``.

    A float tensor is the ``[*, Sk]`` binary-mask convenience form, whose
    leading dimensions collapse to the batch axis. An integer tensor is already
    one leading-1s count per batch.
    """
    if actual_s_kv.is_floating_point():
        if actual_s_kv.ndim < 2:
            raise ValueError(f"binary mask must have ndim >= 2 (last dim = Sk); got shape {tuple(actual_s_kv.shape)}")
        return actual_s_kv.reshape(-1, actual_s_kv.shape[-1]).shape[0]
    return actual_s_kv.reshape(-1).shape[0]


def _to_actual_s_kv_int32(actual_s_kv: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize ``actual_s_kv`` to a contiguous ``[B]`` int32 tensor.

    Accepts a ``[B]`` integer count of leading 1s, or a left-aligned ``[*, Sk]``
    float binary mask whose count is reduced with ``(mask > 0.5).sum(-1)``.
    """
    if actual_s_kv.is_floating_point():
        flat = actual_s_kv.reshape(-1, actual_s_kv.shape[-1])
        out = (flat > 0.5).sum(dim=-1).to(torch.int32).contiguous()
    else:
        if actual_s_kv.dtype != torch.int32:
            actual_s_kv = actual_s_kv.to(torch.int32)
        out = actual_s_kv.reshape(-1).contiguous()
    if out.numel() != batch_size:
        raise ValueError(
            f"actual_s_kv must have B={batch_size} entries (got {out.numel()}). "
            f"Original shape: {tuple(actual_s_kv.shape)}"
        )
    return out


class PairwiseAttentionCuTeLeftMask(CuteKernelCache, AttentionBackend[PairwiseAttentionCuTeLeftMaskMetadata]):
    """Pairwise attention backend for left-aligned KV masks.

    Source kernels use the existing CuTeDSL JIT/disk cache. Public builds can
    omit those source classes and transparently use packaged CUBIN launchers
    through the same in-process executable cache. ``CUTEDSL_FORCE_CUBIN=1``
    takes the CUBIN path even when the sources are importable.

    Expects inputs in the kernel's logical shapes:
      Q, K, V     : [*, Sq, H*D] or [B_flat, Sq, H, D]
      biases      : [actual_s_kv, pair_bias]
        actual_s_kv: [B] int32 (count of leading 1s), or a left-aligned
                     ``[*, Sk]`` float binary mask reduced internally.
        pair_bias  : [*, H, Sq, Sk]

    Multiplicity (``mult``) is inferred from the batch-dimension ratio between
    Q and the per-batch ``actual_s_kv``.
    """

    Metadata = PairwiseAttentionCuTeLeftMaskMetadata

    _compiled_cache: dict[tuple[int, _PairwiseAttentionVariant], Any] = {}

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
        self._last_variant: _PairwiseAttentionVariant | None = None

    def _disk_cache_key(self, variant: _PairwiseAttentionVariant) -> tuple:
        return (
            "attn_pair_bias_cute_left_mask",
            self._sm_version,
            variant.dtype,
            variant.head_dim,
            variant.bucket,
            variant.kv_packed,
        )

    def _resolve_source_kernel(
        self,
        variant: _PairwiseAttentionVariant,
        ct_dtype: type[cutlass.Numeric],
    ) -> tuple[PairwiseAttentionLeftMaskKernelConfig, Any]:
        config = get_kernel_config(self._sm_version, variant.head_dim, variant.bucket)
        if not config.can_implement(ct_dtype, variant.head_dim):
            raise RuntimeError(
                f"Pairwise attention (left-mask) kernel cannot implement: dtype={ct_dtype}, D={variant.head_dim}"
            )
        return config, config.kernel_factory(variant.head_dim)

    def _load_cubin_executable(
        self,
        variant: _PairwiseAttentionVariant,
        source_error: Exception | None = None,
    ):
        cache_key = (self._sm_version, variant)
        try:
            executable = populate_compiled_cache_from_library(
                PairwiseAttentionCuTeLeftMask._compiled_cache,
                cache_key,
                "pairwise_attention",
                lambda library, launcher: PairwiseAttentionCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    variant.head_dim,
                    variant.bucket,
                    variant.dtype,
                    variant.kv_packed,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel "
                    f"library cannot provide this variant: {library_error}"
                ) from library_error
            raise RuntimeError(
                "CuTeDSL pairwise attention source is unavailable and the "
                "precompiled kernel library cannot provide this variant: "
                f"{library_error}"
            ) from source_error

        logger.info(
            f"CuTeDSL pairwise attention (left-mask): using precompiled "
            f"CUBIN for SM{self._sm_version}, dtype={variant.dtype}, "
            f"head_dim={variant.head_dim}, bucket={variant.bucket}, "
            f"kv_packed={variant.kv_packed}"
        )
        return executable

    def _load_or_compile_source(
        self,
        variant: _PairwiseAttentionVariant,
        config: PairwiseAttentionLeftMaskKernelConfig,
        kernel,
        ct_dtype: type[cutlass.Numeric],
        align_elems: int,
        sm_scale: float,
        mult: int,
    ):
        cache_key = (self._sm_version, variant)
        disk_key = self._disk_cache_key(variant)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(
                f"CuTeDSL pairwise attention (left-mask): loaded cached kernel "
                f"for SM{self._sm_version} ({config.arch}), "
                f"dtype={variant.dtype}, head_dim={variant.head_dim}, "
                f"bucket={variant.bucket}, kv_packed={variant.kv_packed}"
            )
            PairwiseAttentionCuTeLeftMask._compiled_cache[cache_key] = executable
            return executable

        logger.info(
            f"CuTeDSL pairwise attention (left-mask): compiling kernel for "
            f"layer={self.layer_idx}, SM{self._sm_version} ({config.arch}), "
            f"dtype={variant.dtype}, head_dim={variant.head_dim}, "
            f"bucket={variant.bucket}, kv_packed={variant.kv_packed}"
        )
        executable = compile_pairwise_attention_source(
            self.compile,
            kernel,
            config.arch,
            variant.head_dim,
            variant.kv_packed,
            ct_dtype,
            align_elems,
            sm_scale,
            mult,
        )
        PairwiseAttentionCuTeLeftMask._compiled_cache[cache_key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info(f"CuTeDSL pairwise attention (left-mask): compilation done for layer={self.layer_idx}")
        return executable

    def _get_executable(
        self,
        variant: _PairwiseAttentionVariant,
        ct_dtype: type[cutlass.Numeric],
        align_elems: int,
        sm_scale: float,
        mult: int,
    ):
        cache_key = (self._sm_version, variant)
        executable = PairwiseAttentionCuTeLeftMask._compiled_cache.get(cache_key)
        if executable is not None:
            return executable

        if self.force_cubin():
            return self._load_cubin_executable(variant)

        try:
            config, kernel = self._resolve_source_kernel(variant, ct_dtype)
        except (ImportError, AttributeError) as source_error:
            return self._load_cubin_executable(variant, source_error)

        return self._load_or_compile_source(
            variant,
            config,
            kernel,
            ct_dtype,
            align_elems,
            sm_scale,
            mult,
        )

    def _prepare_launch_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        actual_s_kv: torch.Tensor,
        bias: torch.Tensor,
        kv_packed: bool,
        output: torch.Tensor | None,
        output_lse: torch.Tensor | None,
    ) -> _PairwiseAttentionLaunchInputs:
        if q.shape[-1] == self.num_heads * self.head_dim:
            q_leading, kv_leading = q.shape[:-1], k.shape[:-1]
            q = q.view(*q_leading, self.num_heads, self.head_dim)
            k = k.view(*kv_leading, self.num_heads, self.head_dim)
            v = v.view(*kv_leading, self.num_heads, self.head_dim)

        padded_head_dim = _align_up(self.head_dim, 16)
        if padded_head_dim != self.head_dim:
            q = _pad_last_dim(q, padded_head_dim)
            k = _pad_last_dim(k, padded_head_dim)
            v = _pad_last_dim(v, padded_head_dim)

        batch_shape = tuple(q.shape[:-3])
        batch_flat = math.prod(batch_shape) if batch_shape else 1
        seqlen_q, num_heads = q.shape[-3], q.shape[-2]
        seqlen_kv = k.shape[-3]
        if num_heads != self.num_heads:
            raise ValueError(f"num_heads mismatch: tensor H={num_heads}, expected {self.num_heads}")
        if k.shape != v.shape:
            raise ValueError("K and V must have the same shape")

        ct_dtype = _cutlass_dtype(q)
        align_elems = 128 // ct_dtype.width
        q_flat = q.reshape(batch_flat, seqlen_q, num_heads, padded_head_dim)
        k_flat = k.reshape(batch_flat, seqlen_kv, num_heads, padded_head_dim)
        v_flat = v.reshape(batch_flat, seqlen_kv, num_heads, padded_head_dim)

        output_shape = (batch_flat, seqlen_q, num_heads, padded_head_dim)
        if (
            output is not None
            and output.shape == output_shape
            and output.dtype == q.dtype
            and output.device == q.device
        ):
            output_flat = output
        else:
            output_flat = torch.empty(output_shape, dtype=q.dtype, device=q.device)

        # B comes from actual_s_kv; mult = B_flat // B is the sample multiplicity.
        batch_size = _batch_size(actual_s_kv)
        if batch_size == 0 or batch_flat % batch_size != 0:
            raise ValueError(f"Q batch dim {batch_flat} is not a multiple of the actual_s_kv batch dim {batch_size}")
        mult = batch_flat // batch_size
        actual_s_kv_flat = _to_actual_s_kv_int32(actual_s_kv, batch_size)
        if actual_s_kv_flat.device != q.device:
            raise ValueError(f"actual_s_kv must be on {q.device}; got {actual_s_kv_flat.device}")

        padded_seqlen_kv = _align_up(seqlen_kv, align_elems)
        bias_padded = _pad_last_dim(bias.contiguous(), padded_seqlen_kv).view(-1, num_heads, seqlen_q, padded_seqlen_kv)
        if bias_padded.shape[0] != batch_size:
            raise ValueError(f"Pair bias batch dim {bias_padded.shape[0]} != actual_s_kv batch dim {batch_size}")

        lse_flat = _resolve_lse_buffer(
            output_lse,
            (batch_flat, seqlen_q, num_heads, 1),
            q.device,
        )
        S = _compute_S(seqlen_q, seqlen_kv)
        variant = _PairwiseAttentionVariant(
            dtype=q_flat.dtype,
            head_dim=padded_head_dim,
            bucket=get_nearest_bucket(self._sm_version, padded_head_dim, S),
            kv_packed=kv_packed,
        )
        return _PairwiseAttentionLaunchInputs(
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
            batch_shape=batch_shape,
            mult=mult,
            seqlen_q=seqlen_q,
            num_heads=num_heads,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: PairwiseAttentionCuTeLeftMaskMetadata | None = None,
        output: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run pairwise attention via the CuTe left-mask kernel.

        Args:
            q: ``[*, Sq, H*D]`` or ``[B_flat, Sq, H, D]``.
            k: ``[*, Sk, H*D]`` or ``[B_flat, Sk, H, D]`` (may be
                non-contiguous when kv_packed).
            v: same shape as *k*.
            biases: ``[actual_s_kv, pair_bias]``
                actual_s_kv: ``[B]`` int32 — count of leading 1s along Sk. A
                             left-aligned ``[*, Sk]`` float binary mask is
                             accepted as a convenience.
                pair_bias  : ``[*, H, Sq, Sk]``.
            output: Optional ``[B_flat, Sq, H, D_padded]`` buffer written in
                place.
            output_lse: Optional ``[B_flat, Sq, H, 1]`` float32 buffer written
                in place, honored by both the Ampere and Hopper kernels. A
                shape/dtype/device mismatch falls back to an internal
                allocation.

        Returns:
            Output tensor ``[*, Sq, H, D]``.
        """
        if biases is None or len(biases) < 2:
            raise ValueError("CuTeDSL pairwise attention (left-mask) expects biases=[actual_s_kv, pair_bias]")
        actual_s_kv = biases[0]
        bias = biases[1]
        if metadata is None:
            metadata = PairwiseAttentionCuTeLeftMaskMetadata()

        launch_inputs = self._prepare_launch_inputs(
            q,
            k,
            v,
            actual_s_kv,
            bias,
            getattr(metadata, "kv_packed", True),
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
                launch_inputs.mult,
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
            launch_inputs.mult,
        )
        output_shape = (
            *launch_inputs.batch_shape,
            launch_inputs.seqlen_q,
            launch_inputs.num_heads,
            launch_inputs.variant.head_dim,
        )
        return launch_inputs.output.view(output_shape)[..., : self.head_dim]
