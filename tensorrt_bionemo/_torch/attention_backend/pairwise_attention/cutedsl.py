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
"""Pairwise attention (attention with pair bias) backend using the CuTe
left-mask kernels.

Unified Ampere (SM80/86/89) + Hopper (SM90) wrapper. Dispatches between
:class:`FlashAttentionForwardAmpere` (``sm80_attn_pb_left_mask``) and
:class:`HopperFusedMultiHeadAttentionForward` (``sm90_attn_pb_left_mask``)
at runtime based on ``torch.cuda.get_device_capability()``. The two
kernels expose slightly different launch signatures (SM90 also produces
an LSE output and takes both ``softmax_scale`` and
``softmax_scale * log2(e)``); the dispatch is handled transparently here.

Specialized for the case where the KV-side mask is *left-aligned*
(``1...1 0...0``), as opposed to a general per-key mask.  Instead of a
``[B, Sk]`` float32 binary mask, the kernel takes a single ``actual_s_kv``
integer per batch giving the count of leading 1s.  Blocks past
``actual_s_kv`` are skipped entirely; the partial tail block is guarded by a
column-index compare against ``actual_s_kv``.  No GMEM mask traffic, no SMEM
mask buffer.

Wraps :class:`FlashAttentionForwardAmpere` from ``sm80_attn_pb_left_mask.py``.

Kernel tensor shapes:
  Q, K, V, O  : [B*mult, Sq, H, D]
  bias        : [B, H, Sq, Sk_padded]   broadcasts over ``mult`` dimension
  actual_s_kv : [B]                     int32, broadcasts over ``mult``

The backend expects biases in the following format::

    biases = [actual_s_kv, pair_bias]
      actual_s_kv: [B] int32  (count of leading 1s along Sk).
                   For convenience the wrapper also accepts a left-aligned
                   binary mask ``[*, Sk]`` (float) and computes the
                   leading-1s count via ``(mask > 0.5).sum(-1)``.
      pair_bias  : [*, H, Sq, Sk]

The backend pads the last dimension of *pair_bias* to the kernel's alignment
requirement and flattens batch dims before invoking the kernel.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
import tvm_ffi
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from tensorrt_bionemo._torch._kernel_config_loader import (
    get_config_file_name,
    load_kernel_configs,
    resolve_implementation,
)
from tensorrt_bionemo.dsl_kernels.cute_cache import CuteKernelCache
from tensorrt_bionemo.logger import logger

from ..interface import AttentionBackend, AttentionMetadata

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
_VARIANT_KEY_RE = re.compile(r"^S=(\d+)$")


class PairwiseAttentionCuTeLeftMaskMetadata(AttentionMetadata):
    kv_packed: bool = True


# log2(e), used to convert softmax scale to its log2 form for the Hopper
# kernel which performs softmax via exp2 fastmath.
_LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class PairwiseAttentionLeftMaskKernelConfig:
    """Compilation config for a pairwise attention CuTe left-mask kernel.

    Attributes:
        arch: Architecture tag of the underlying kernel. Currently
            ``"sm80"`` (Ampere ``FlashAttentionForwardAmpere``) or
            ``"sm90"`` (Hopper ``HopperFusedMultiHeadAttentionForward``).
            Used by the backend to dispatch to the right compile-arg /
            launch-arg layout (the two kernels have different ``__call__``
            signatures and the SM90 path also produces an LSE output).
        kernel_factory: ``Callable(head_dim) -> kernel instance``.
        can_implement:  ``Callable(cutlass_dtype, head_dim) -> bool``.
    """

    arch: str
    kernel_factory: Callable[[int], Any]
    can_implement: Callable[[type, int], bool]


def _build_sm80_config(
    kernel_cls: type,
    m_block_size: int,
    n_block_size: int,
    num_threads: int = 128,
    swizzle_b: int = 3,
    load_bias_before_gemm: bool = True,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Wrap an Ampere pair-bias left-mask kernel class + tile params.

    The kernel class is resolved from the JSON ``implementation`` field at
    call time, so the same builder can construct both the production
    ``sm80_attn_pb_left_mask.FlashAttentionForwardAmpere`` and (on SM90 +
    small head_dim, where Hopper smem doesn't fit) the same Ampere class
    reused as a fallback.
    """

    def factory(head_dim: int):
        return kernel_cls(
            head_dim,
            m_block_size,
            n_block_size,
            num_threads,
            swizzle_b=swizzle_b,
            load_bias_before_gemm=load_bias_before_gemm,
        )

    def can_impl(ct_dtype: type, head_dim: int) -> bool:
        return kernel_cls.can_implement(
            ct_dtype,
            head_dim,
            m_block_size,
            n_block_size,
            num_threads,
        )

    return PairwiseAttentionLeftMaskKernelConfig(arch="sm80", kernel_factory=factory, can_implement=can_impl)


def _build_sm90_config(
    kernel_cls: type,
    mma_tiler_mn: tuple[int, int],
    is_persistent: bool,
    kv_stage: int = 5,
    raster_factor: int = 0,
    qk_acc_dtype: type = cutlass.Float32,
    pv_acc_dtype: type = cutlass.Float32,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Wrap a Hopper pair-bias left-mask kernel class + tile params.

    The Hopper kernel uses a TMA + warp-specialised pipeline; the relevant
    knobs are the MMA tile shape ``(M, N)`` (the ``K`` dim is the head dim
    and is filled in by ``factory(head_dim)``), the persistent-kernel mode,
    the K/V pipeline depth, and the persistent tile-scheduler raster factor:

      ``raster_factor == 0``: default M-fast iteration (best L2 reuse on
        K/V/bias for one ``(b, h)`` before advancing).
      ``raster_factor > 0``:  block-raster on the M axis — split M into
        chunks of ``raster_factor`` and interleave bh within each chunk.
        Trades K/V/bias L2 reuse for greater wavefront diversity at small
        head_dim. Caller must ensure ``raster_factor <= M_tiles`` at
        launch time. Ignored when ``is_persistent=False``.
    """
    mma_mn_tuple = tuple(mma_tiler_mn)

    def factory(head_dim: int):
        mma_tiler = (mma_mn_tuple[0], mma_mn_tuple[1], head_dim)
        return kernel_cls(
            qk_acc_dtype,
            pv_acc_dtype,
            mma_tiler,
            is_persistent,
            kv_stage=kv_stage,
            raster_factor=raster_factor,
        )

    def can_impl(ct_dtype: type, head_dim: int) -> bool:
        # SM90 ``can_implement`` validates against shapes and scale; for
        # config-selection time we don't have shapes yet.  Use placeholder
        # shapes that satisfy the b/h divisibility checks; the dtype and
        # mma_tiler / persistent constraints (the things that actually
        # depend on this config) are still exercised.
        ok, _ = kernel_cls.can_implement(
            (1, 64, 1, head_dim),
            (1, 64, 1, head_dim),
            ct_dtype,
            qk_acc_dtype,
            pv_acc_dtype,
            mma_mn_tuple,
            is_persistent,
            1.0,
            1,
        )
        return ok

    return PairwiseAttentionLeftMaskKernelConfig(arch="sm90", kernel_factory=factory, can_implement=can_impl)


# Tuned tile configs: attention_backend/configs/pairwise_attention/
#   D{D}_sm{sm}.json — each file carries
#   {implementation, configs: {"S=<anchor>": params}}. The nearest per-side
#   sequence-length anchor is selected from ``S = round(sqrt(Sq * Sk))``.
#   The ``implementation`` field selects the Ampere
#   (``sm80_attn_pb_left_mask.FlashAttentionForwardAmpere``) or Hopper
#   (``sm90_attn_pb_left_mask.HopperFusedMultiHeadAttentionForward``)
#   kernel class — Hopper-schema tile params are distinguished by the
#   presence of ``mma_tiler_mn``.
_PW_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "pairwise_attention")


def _load_config_bundle(sm_version: int, head_dim: int):
    bundle = load_kernel_configs(
        _PW_CONFIGS_DIR,
        get_config_file_name(sm_version, D=head_dim),
    )
    if bundle is None:
        raise ValueError(f"No pairwise-attention config file for SM{sm_version}, head_dim={head_dim}.")
    return bundle


def _nearest_variant(configs: dict[str, Any], S: int) -> tuple[int, dict[str, Any]]:
    """Return the tile whose ``S=<anchor>`` key is nearest to ``S``."""
    if S < 0:
        raise ValueError(f"Pairwise-attention S must be non-negative; got {S}")

    candidates: list[tuple[int, str]] = []
    for key in configs:
        match = _VARIANT_KEY_RE.fullmatch(key)
        if match is not None:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise ValueError(f"No pairwise-attention S anchors are registered; available keys: {sorted(configs)}")

    anchor, key = min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))
    return anchor, dict(configs[key])


def get_nearest_bucket(sm_version: int, head_dim: int, S: int) -> int:
    """Return the nearest tuned per-side sequence-length anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    bucket, _ = _nearest_variant(bundle.configs, S)
    return bucket


def get_kernel_config(
    sm_version: int,
    head_dim: int,
    S: int,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Resolve the source kernel at the nearest tuned ``S`` anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, S)
    kernel_cls = resolve_implementation(bundle.implementation)
    if "mma_tiler_mn" in tile_params:
        return _build_sm90_config(kernel_cls, **tile_params)
    return _build_sm80_config(kernel_cls, **tile_params)


def _compute_S(Sq: int, Sk: int) -> int:
    """Map a possibly rectangular attention problem to its side-length axis."""
    return int(round(math.sqrt(max(Sq * Sk, 1))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_up(x: int, align: int) -> int:
    "Maps integer x to the next integer multiple of align"
    return ((x + align - 1) // align) * align


def _pad_last_dim(t: torch.Tensor, new_size: int, value: float = 0.0) -> torch.Tensor:
    cur = t.size(-1)
    if cur >= new_size:
        return t
    return F.pad(t, (0, new_size - cur), value=value)


def _cutlass_dtype(t: torch.Tensor) -> type[cutlass.Numeric]:
    ty = _TORCH_TO_CUTLASS_DTYPE.get(t.dtype)
    if ty is None:
        raise TypeError(f"SM80 pairwise attention expects float16 or bfloat16; got {t.dtype}")
    return ty


def _resolve_lse_buffer(
    output_lse: torch.Tensor | None,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Reuse a caller-supplied LSE buffer when it matches the kernel's
    expected shape/dtype/device contract, else allocate a fresh float32
    tensor of ``shape`` on ``device``.

    The kernel's contract is fixed: ``shape`` rows, last dim 1, dtype
    ``float32`` (qk_acc_dtype). A mismatch on any of these falls back to an
    internal allocation rather than failing — same pattern used by the
    ``output`` parameter for the attention output tensor.
    """
    if (
        output_lse is not None
        and tuple(output_lse.shape) == shape
        and output_lse.dtype == torch.float32
        and output_lse.device == device
    ):
        return output_lse
    return torch.empty(*shape, dtype=torch.float32, device=device)


def _to_actual_s_kv_int32(actual_s_kv: torch.Tensor, B: int) -> torch.Tensor:
    """Normalize ``actual_s_kv`` to a contiguous ``[B]`` int32 tensor.

    Accepts:
      * ``[B]`` int (any int dtype) — leading-1s count per batch.
      * ``[B, Sk]`` (or higher-rank ``[*, Sk]``) float binary mask — the
        leading-1s count is computed via ``(mask > 0.5).sum(dim=-1)`` along
        the masked axis.  Leading dims are flattened to a single ``B``.
    """
    if actual_s_kv.is_floating_point():
        flat = actual_s_kv
        if flat.ndim > 2:
            flat = flat.view(-1, flat.shape[-1])
        elif flat.ndim < 2:
            raise ValueError(f"binary mask must have ndim >= 2 (last dim = Sk); got shape {tuple(actual_s_kv.shape)}")
        out = (flat > 0.5).sum(dim=-1).to(torch.int32).contiguous()
    else:
        if actual_s_kv.dtype != torch.int32:
            actual_s_kv = actual_s_kv.to(torch.int32)
        out = actual_s_kv.reshape(-1).contiguous()
    if out.numel() != B:
        raise ValueError(
            f"actual_s_kv must have B={B} entries (got {out.numel()}). Original shape: {tuple(actual_s_kv.shape)}"
        )
    return out


class PairwiseAttentionCuTeLeftMask(CuteKernelCache, AttentionBackend[PairwiseAttentionCuTeLeftMaskMetadata]):
    """Pairwise attention backend using the CuTe DSL left-mask kernels.

    Specialized for the common case where the KV-side mask is
    left-aligned (``1...1 0...0``).
    Instead of a per-batch binary mask the kernel reads a single integer
    ``actual_s_kv[b]`` giving the count of leading 1s, skipping fully-masked
    KV blocks and avoiding the GMEM/SMEM mask traffic.

    Dispatches between the SM80 (Ampere) ``FlashAttentionForwardAmpere`` and
    the SM90 (Hopper) ``HopperFusedMultiHeadAttentionForward`` kernels based
    on the device's compute capability. The two kernels expose slightly
    different launch signatures (SM90 also produces an LSE output and takes
    both ``softmax_scale`` and ``softmax_scale * log2(e)``); this class
    handles both transparently.

    Inherits from :class:`~...dsl_kernels.cute_cache.CuteKernelCache` for the
    unified compile / save / load interface.

    Expects:
      Q, K, V : ``[*, Sq, H*D]`` or ``[B_flat, Sq, H, D]``
      biases  : ``[actual_s_kv, pair_bias]``
        actual_s_kv: ``[B]`` int32  (count of leading 1s along Sk).
                     A left-aligned ``[*, Sk]`` float binary mask is also
                     accepted as a convenience and reduced internally.
        pair_bias  : ``[*, H, Sq, Sk]``

    Multiplicity (``mult``) is inferred automatically from the batch
    dimension ratio between Q and the per-batch ``actual_s_kv``.
    """

    Metadata = PairwiseAttentionCuTeLeftMaskMetadata

    _compiled_cache: dict[tuple, Any] = {}

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
        self._last_exe = None
        self._last_key: tuple = ()

    def _disk_cache_key(self, key: tuple) -> tuple:
        """Build the disk-cache key by prefixing the SM version."""
        return ("attn_pair_bias_cute_left_mask", self._sm_version) + key

    def _get_or_compile(
        self,
        kernel,
        arch: str,
        ct_dtype: type[cutlass.Numeric],
        D: int,
        align_elems: int,
        sm_scale: float,
        mult: int,
        key: tuple,
    ):
        exe = PairwiseAttentionCuTeLeftMask._compiled_cache.get(key)
        if exe is not None:
            return exe

        dtype, head_dim, bucket, kv_packed = key

        disk_key = self._disk_cache_key(key)
        exe = self.load_from_cache(disk_key)
        if exe is not None:
            logger.info(
                f"CuTeDSL pairwise attention (left-mask): loaded cached kernel "
                f"for SM{self._sm_version} ({arch}), dtype={dtype}, "
                f"head_dim={head_dim}, bucket={bucket}, kv_packed={kv_packed}"
            )
            PairwiseAttentionCuTeLeftMask._compiled_cache[key] = exe
            return exe

        logger.info(
            f"CuTeDSL pairwise attention (left-mask): compiling kernel for "
            f"layer={self.layer_idx}, SM{self._sm_version} ({arch}), "
            f"dtype={dtype}, head_dim={head_dim}, "
            f"bucket={bucket}, kv_packed={kv_packed}"
        )

        div = align_elems
        b_flat_sym = cute.sym_int()
        b_sym = cute.sym_int()
        sq_sym = cute.sym_int()
        sk_sym = cute.sym_int()
        h_sym = cute.sym_int()
        q_fake = make_fake_tensor(
            ct_dtype,
            (b_flat_sym, sq_sym, h_sym, D),
            stride=(cute.sym_int64(divisibility=div), cute.sym_int64(divisibility=div), D, 1),
            assumed_align=16,
        )
        kv_fake = make_fake_tensor(
            ct_dtype,
            (b_flat_sym, sk_sym, h_sym, D),
            stride=(cute.sym_int64(divisibility=div), cute.sym_int64(divisibility=div), D, 1),
            assumed_align=16,
        )
        if kv_packed:
            o_fake = make_fake_tensor(
                ct_dtype,
                (cute.sym_int(), cute.sym_int(), cute.sym_int(), D),
                stride=(cute.sym_int64(divisibility=div), cute.sym_int64(divisibility=div), D, 1),
                assumed_align=16,
            )
        else:
            o_fake = q_fake
        bias_fake = make_fake_tensor(
            ct_dtype,
            (b_sym, h_sym, sq_sym, cute.sym_int()),
            stride=(
                cute.sym_int64(divisibility=div),
                cute.sym_int64(divisibility=div),
                cute.sym_int64(divisibility=div),
                1,
            ),
            assumed_align=16,
        )
        # actual_s_kv: [B] int32, contiguous (stride 1).
        actual_s_kv_fake = make_fake_tensor(
            cutlass.Int32,
            (b_sym,),
            stride=(1,),
            assumed_align=4,
        )
        stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

        # Unified call signature for both Ampere (SM80/86/89) and Hopper (SM90):
        #   (q, k, v, bias, actual_s_kv, o, lse,
        #    sm_log2, sm_scale, mult, stream)
        # LSE shape [B*mult, Sq, H, 1] Float32 on both paths.
        if arch not in ("sm80", "sm90"):
            raise ValueError(
                f"Unsupported pairwise attention left-mask kernel arch {arch!r}; expected 'sm80' or 'sm90'."
            )

        lse_fake = make_fake_tensor(
            cutlass.Float32,
            (b_flat_sym, sq_sym, h_sym, 1),
            stride=(cute.sym_int64(divisibility=1), cute.sym_int64(divisibility=1), 1, 1),
            assumed_align=4,
        )
        sm_log2 = float(sm_scale * _LOG2_E)
        exe = self.compile(
            kernel,
            q_fake,
            kv_fake,
            kv_fake,
            bias_fake,
            actual_s_kv_fake,
            o_fake,
            lse_fake,
            sm_log2,
            float(sm_scale),
            mult,
            stream_fake,
        )

        PairwiseAttentionCuTeLeftMask._compiled_cache[key] = exe
        self.save_to_cache(disk_key, exe)
        logger.info(f"CuTeDSL pairwise attention (left-mask): compilation done for layer={self.layer_idx}")
        return exe

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

        Dispatches to the SM80 (Ampere) ``FlashAttentionForwardAmpere`` or
        the SM90 (Hopper) ``HopperFusedMultiHeadAttentionForward`` kernel
        based on the device's compute capability. The two kernels expose
        slightly different launch signatures (SM90 also produces an LSE
        output and takes both ``softmax_scale`` and
        ``softmax_scale * log2(e)``); this method handles both
        transparently.

        Args:
            q: ``[*, Sq, H*D]`` or ``[B_flat, Sq, H, D]``.
            k: ``[*, Sk, H*D]`` or ``[B_flat, Sk, H, D]``
                (may be non-contiguous when kv_packed).
            v: same shape as *k*.
            biases: ``[actual_s_kv, pair_bias]``
                actual_s_kv: ``[B]`` int32 — count of leading 1s along Sk.
                             A left-aligned binary mask ``[*, Sk]`` (float)
                             is accepted as a convenience.
                pair_bias: ``[*, H, Sq, Sk]``.
            output: Optional pre-allocated buffer shaped
                ``[B_flat, Sq, H, D_padded]``. When supplied the kernel
                writes directly into it, avoiding a per-call allocation.
            output_lse: Optional pre-allocated LSE buffer shaped
                ``[B_flat, Sq, H, 1]`` with dtype float32. Honored by both
                the SM80/86/89 (Ampere) and SM90 (Hopper) kernels. When
                supplied the kernel writes directly into it, avoiding a
                per-call allocation. Caller is responsible for the
                shape/dtype/device matching exactly; a mismatch falls back
                to an internal allocation.

        Returns:
            Output tensor ``[*, Sq, H, D]``.
        """
        if biases is None or len(biases) < 2:
            raise ValueError("CuTeDSL pairwise attention (left-mask) expects biases=[actual_s_kv, pair_bias]")
        actual_s_kv_input = biases[0]
        pair_bias = biases[1]
        if metadata is None:
            metadata = PairwiseAttentionCuTeLeftMaskMetadata()

        kv_packed = getattr(metadata, "kv_packed", True)

        # --- Reshape Q/K/V from [*, S, H*D] to [*, S, H, D] ---------------
        if q.shape[-1] == self.num_heads * self.head_dim:
            *q_leading, _ = q.shape
            *k_leading, _ = k.shape
            q = q.view(*q_leading, self.num_heads, self.head_dim)
            k = k.view(*k_leading, self.num_heads, self.head_dim)
            v = v.view(*k_leading, self.num_heads, self.head_dim)

        batch_shape = q.shape[:-3]
        B_flat = 1
        for d in batch_shape:
            B_flat *= d
        Sq, H, D = q.shape[-3], q.shape[-2], q.shape[-1]
        Sk = k.shape[-3]

        # Pad D to a multiple of 16 for safe SMEM access
        D_padded = _align_up(D, 16)
        if D_padded != D:
            q = _pad_last_dim(q, D_padded)
            k = _pad_last_dim(k, D_padded)
            v = _pad_last_dim(v, D_padded)
            D = D_padded

        q_flat = q.reshape(B_flat, Sq, H, D)
        k_flat = k.reshape(B_flat, Sk, H, D)
        v_flat = v.reshape(B_flat, Sk, H, D)

        if (
            output is not None
            and output.shape == (B_flat, Sq, H, D)
            and output.dtype == q.dtype
            and output.device == q.device
        ):
            o_flat = output
        else:
            o_flat = torch.empty(B_flat, Sq, H, D, dtype=q.dtype, device=q.device)

        ct_dtype = _cutlass_dtype(q_flat)
        align_elems = 128 // ct_dtype.width  # 8 for bf16/fp16
        Sk_padded = _align_up(Sk, align_elems)

        # --- B / mult inference --------------------------------------------
        # B is inferred from actual_s_kv (or its binary-mask flattened form);
        # mult = B_flat // B is the diffusion-sample multiplicity.
        if actual_s_kv_input.is_floating_point():
            # Binary-mask form: leading dims collapse to B, last dim is Sk.
            mask_view = actual_s_kv_input
            if mask_view.ndim > 2:
                mask_view = mask_view.view(-1, mask_view.shape[-1])
            elif mask_view.ndim < 2:
                raise ValueError(
                    f"binary mask must have ndim >= 2 (last dim = Sk); got shape {tuple(actual_s_kv_input.shape)}"
                )
            B = mask_view.shape[0]
        else:
            B = actual_s_kv_input.reshape(-1).shape[0]
        mult = B_flat // B
        actual_s_kv_flat = _to_actual_s_kv_int32(actual_s_kv_input, B)
        assert actual_s_kv_flat.device == q.device, f"actual_s_kv must be on {q.device}; got {actual_s_kv_flat.device}"

        # --- Pair bias -----------------------------------------------------
        # [*, H, Sq, Sk] → [B, H, Sq, Sk_padded]
        pb_padded = _pad_last_dim(pair_bias.contiguous(), Sk_padded)
        pb = pb_padded.view(-1, H, Sq, Sk_padded)
        assert pb.shape[0] == B, f"Pair bias batch dim {pb.shape[0]} != actual_s_kv batch dim {B}"
        bias_padded = pb

        # --- Pick kernel config --------------------------------------------
        softmax_scale = float(self.head_dim**-0.5)
        S = _compute_S(Sq, Sk)
        bucket = get_nearest_bucket(self._sm_version, D, S)
        compile_key = (q_flat.dtype, D, bucket, kv_packed)

        if compile_key == self._last_key:
            exe = self._last_exe
        else:
            cfg = get_kernel_config(self._sm_version, D, S)
            if not cfg.can_implement(ct_dtype, D):
                raise RuntimeError(f"Pairwise attention (left-mask) kernel cannot implement: dtype={ct_dtype}, D={D}")
            kernel = cfg.kernel_factory(D)
            exe = self._get_or_compile(kernel, cfg.arch, ct_dtype, D, align_elems, softmax_scale, mult, compile_key)
            self._last_key = compile_key
            self._last_exe = exe

        # LSE shape/dtype is fixed by the kernel: [B*mult, Sq, H, 1] f32.
        lse_flat = _resolve_lse_buffer(output_lse, (B_flat, Sq, H, 1), q.device)

        # Unified call signature for both Ampere (SM80/86/89) and Hopper
        # (SM90):
        #   (q, k, v, bias, actual_s_kv, o, lse, sm_log2, sm_scale, mult)
        sm_log2 = float(softmax_scale * _LOG2_E)
        # The kernel is compiled with use_tvm_ffi_env_stream=True, so it reads
        # its launch stream from the TVM-FFI environment. Sync that env stream
        # to torch's current stream so the kernel runs on the active stream
        # (e.g. a side/capture stream), not the default stream.
        with tvm_ffi.use_torch_stream():
            exe(q_flat, k_flat, v_flat, bias_padded, actual_s_kv_flat, o_flat, lse_flat, sm_log2, softmax_scale, mult)

        # [B_flat, Sq, H, D] → [*, Sq, H, D]
        o = o_flat.view(*batch_shape, Sq, H, D)
        return o[..., : self.head_dim]
