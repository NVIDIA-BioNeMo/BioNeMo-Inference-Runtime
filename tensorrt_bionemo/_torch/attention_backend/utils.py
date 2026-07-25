# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from typing import Callable, Optional, Type

import torch

from tensorrt_bionemo.utils import get_sm_version

from .cuequiv import CuEquivAttention
from .interface import AttentionBackend, AttentionType
from .pairwise_attention_cute_left_mask import PairwiseAttentionCuTeLeftMask
from .sdpa import SDPAPairwiseAttention
from .triangle_attention_cute_left_mask import TriangleAttentionCuTeLeftMask
from .trifast import TrifastAttention
from .vanilla import VanillaPairwiseAttention, VanillaTriangleAttention


def auto_select_triangle_attention_backend(
    dtype: torch.dtype = torch.float32, ) -> str:
    """Return the fastest available triangle attention backend name.

    Selection priority (highest to lowest):
      1. **CuTeDSL** — SM80/SM86/SM89/SM90, fp16/bf16 only.
      2. **CUEQUIV** — all SKUs, all dtypes (fp32, fp16, bf16).
      3. **TRIFAST** — Triton-based fallback.
      4. **VANILLA** — pure-PyTorch reference.
    """
    sm = get_sm_version()
    is_half = dtype in (torch.float16, torch.bfloat16)

    if sm in (80, 86, 89, 90) and is_half:
        return "CuTeDSL"

    try:
        import cuequivariance_ops_torch  # noqa: F401
        return "CUEQUIV"
    except ImportError:
        pass

    try:
        from .trifast import TrifastAttention as _  # noqa: F401
        return "TRIFAST"
    except ImportError:
        pass

    return "VANILLA"


def auto_select_pairwise_attention_backend(
    dtype: torch.dtype = torch.float32, ) -> str:
    """Return the fastest available pairwise attention backend name.

    Selection priority (highest to lowest):
      1. **CuTeDSL** — SM80/SM86/SM89/SM90, fp16/bf16 only.
      2. **SDPA** — PyTorch scaled-dot-product attention, all SKUs/dtypes.
      3. **VANILLA** — pure-PyTorch reference.
    """
    sm = get_sm_version()
    is_half = dtype in (torch.float16, torch.bfloat16)

    if sm in (80, 86, 89, 90) and is_half:
        return "CuTeDSL"

    return "SDPA"


def get_attention_backend(
    backend_name: str,
    attention_type: AttentionType = AttentionType.TRIANGLE
) -> Type[AttentionBackend]:
    """Get the attention backend class based on the backend name and attention type."""
    if attention_type == AttentionType.TRIANGLE:
        if backend_name == "VANILLA":
            return VanillaTriangleAttention
        elif backend_name == "TRIFAST":
            return TrifastAttention
        elif backend_name == "CUEQUIV":
            return CuEquivAttention
        elif backend_name == "CuTeDSL":
            return TriangleAttentionCuTeLeftMask
    elif attention_type == AttentionType.PAIRWISE:
        if backend_name == "VANILLA":
            return VanillaPairwiseAttention
        elif backend_name == "SDPA":
            return SDPAPairwiseAttention
        elif backend_name == "CuTeDSL":
            return PairwiseAttentionCuTeLeftMask
        else:
            raise ValueError(f"Invalid backend name: {backend_name}")
    else:
        raise ValueError(f"Invalid backend name: {backend_name}")


def create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    attention_type: AttentionType = AttentionType.TRIANGLE
) -> AttentionBackend:
    """Create an attention backend based on the backend name and attention type."""
    attn_cls = get_attention_backend(backend_name, attention_type)
    return attn_cls(layer_idx, num_heads, head_dim, num_kv_heads)


# ---------------------------------------------------------------------------
# Precomputed pair-mask registry
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedPairMasks:
    """Precomputed per-row attention masks for triangle attention nodes.

    Computing the per-row mask from ``pair_mask`` is identical across layers,
    so doing it once before the layer loop avoids redundant work. The exact
    representation depends on the active triangle attention backend.

    Attributes:
        pair_mask: Original ``[B, I, J]`` mask for TriangleMultiplicationNode.
        mask_bias: Per-row mask payload for TriangleAttentionStartingNode.
            Shape / semantics depend on the backend:
              * default backends (VANILLA / CUEQUIV / TRIFAST): additive
                bias of shape ``[B, I, 1, 1, J]``;
              * CuTeDSL left-mask kernel: ``int32`` count of valid KV
                positions per row (``actual_s_kv``), shape ``[B, I]``,
                ``mask_bias[b, i] = (pair_mask[b, i, :] > 0.5).sum()``.
                Assumes a left-aligned ``1...1 0...0`` mask, which holds when
                ``pair_mask`` is the outer product of a left-aligned
                ``seq_mask``.

            For the CuTeDSL backend, this tensor doubles as the dual_gemm_x_x
            ``actual_seqlen`` consumed by ``TriangleMultiplicationNode``
            (OUTGOING / ``tri_mul_out``): the LM dual_gemm kernel treats each
            ``(b, i)`` pair-tensor row as a separate kernel batch of length
            ``J`` and masks row ``g_m`` via
            ``g_m % J < mask_bias[g_m // J]``.
        mask_bias_transposed: Per-row mask payload for TriangleAttentionEndingNode
            (whose input is transposed before the kernel call).
            Shape / semantics depend on the backend:
              * default backends: additive bias of shape ``[B, J, 1, 1, I]``;
              * CuTeDSL left-mask kernel: ``int32`` count along the
                transposed axis (``actual_s_kv``), shape ``[B, J]``,
                ``mask_bias_transposed[b, j] = (pair_mask[b, :, j] > 0.5).sum()``.

            For the CuTeDSL backend, this tensor doubles as the dual_gemm_x_x
            ``actual_seqlen`` consumed by ``TriangleMultiplicationNode``
            (INCOMING / ``tri_mul_in``) -- the per-column counterpart of
            ``mask_bias``. For the typical outer-product symmetric pair mask
            ``mask_bias`` and ``mask_bias_transposed`` are numerically equal,
            but threading the transposed view through ``tri_mul_in`` keeps
            the wiring symmetric with the rest of the precompute consumers
            and handles non-square pair tensors correctly.
    """
    pair_mask: torch.Tensor
    mask_bias: torch.Tensor
    mask_bias_transposed: torch.Tensor


PrecomputePairMasksFn = Callable[[torch.Tensor, float, Optional[torch.dtype]],
                                 PrecomputedPairMasks]

_PRECOMPUTE_PAIR_MASKS_REGISTRY: dict[str, PrecomputePairMasksFn] = {}


def register_precompute_pair_masks(
    backend_name: str,
    fn: PrecomputePairMasksFn,
) -> None:
    """Register a precompute-pair-masks callable for *backend_name*."""
    _PRECOMPUTE_PAIR_MASKS_REGISTRY[backend_name] = fn


def precompute_pair_masks(
    backend_name: str,
    pair_mask: torch.Tensor,
    inf: float = 1e9,
    dtype: Optional[torch.dtype] = None,
) -> PrecomputedPairMasks:
    """Dispatch to the registered precompute function for *backend_name*.

    Args:
        backend_name: Triangle attention backend identifier (e.g. ``"CUEQUIV"``).
        pair_mask: ``[B, I, J]`` mask tensor.
        inf: Large value used to mask out invalid positions.
        dtype: Target dtype for the mask bias tensors.

    Returns:
        A :class:`PrecomputedPairMasks` bundle ready to be passed through
        pairformer / evoformer layers.
    """
    fn = _PRECOMPUTE_PAIR_MASKS_REGISTRY.get(backend_name)
    if fn is None:
        raise ValueError(
            f"No precompute_pair_masks registered for backend '{backend_name}'. "
            f"Available: {sorted(_PRECOMPUTE_PAIR_MASKS_REGISTRY.keys())}")
    return fn(pair_mask, inf, dtype)


# ---------------------------------------------------------------------------
# Default implementation (shared by VANILLA / CUEQUIV / TRIFAST)
# ---------------------------------------------------------------------------


def _default_precompute_pair_masks(
    pair_mask: torch.Tensor,
    inf: float = 1e9,
    dtype: Optional[torch.dtype] = None,
) -> PrecomputedPairMasks:
    mask_typed = (pair_mask.to(dtype) if dtype is not None
                  and pair_mask.dtype != dtype else pair_mask)
    bias = (inf * (mask_typed - 1))[..., :, None, None, :]
    bias_transposed = (
        inf * (mask_typed.transpose(-2, -1) - 1))[..., :, None,
                                                  None, :].contiguous()
    return PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=bias,
        mask_bias_transposed=bias_transposed,
    )


register_precompute_pair_masks("VANILLA", _default_precompute_pair_masks)
register_precompute_pair_masks("CUEQUIV", _default_precompute_pair_masks)
register_precompute_pair_masks("TRIFAST", _default_precompute_pair_masks)

# ---------------------------------------------------------------------------
# CuTeDSL implementation — left-mask kernel (int32 ``actual_s_kv`` per row)
# ---------------------------------------------------------------------------
#
# The CuTeDSL triangle-attention kernel takes a per-row count of leading 1s
# (``actual_s_kv``) instead of a binary mask tensor.  This is valid whenever
# ``pair_mask[b, i, :]`` is left-aligned (``1...1 0...0``), which holds for
# OpenFold/Boltz-style ``pair_mask = seq_mask[..., None] * seq_mask[..., None, :]``
# with a left-aligned ``seq_mask``.
#
# Under that assumption, the leading-1s count is simply ``sum(>0.5)`` along
# the masked axis.  No padding is needed: the kernel handles the partial tail
# block via a column-index compare against ``actual_s_kv``.


def _cutedsl_precompute_pair_masks(
    pair_mask: torch.Tensor,
    inf: float = 1e9,
    dtype: Optional[torch.dtype] = None,
) -> PrecomputedPairMasks:
    """Build ``actual_s_kv`` tensors for the CuTeDSL left-mask kernel.

    For the start node the masked axis is the last dim (J); for the end node
    the input is transposed first, so the masked axis is the second-to-last
    dim (I) of the original ``pair_mask``.  Both reductions are computed on
    the original ``pair_mask`` (no explicit transpose) and returned as
    contiguous int32 tensors.
    """
    del inf, dtype  # unused by the left-mask kernel
    mask_bool = pair_mask > 0.5
    # The CuTeDSL left-mask kernel assumes each row along the masked axis is
    # ``1...1 0...0``. Equivalently, the row must be non-increasing. Verify
    # this along both axes since we reduce over each independently below.
    # ``torch.all(...)`` in a bool context forces a device->host sync, which is
    # illegal while a CUDA graph is capturing (``operation not permitted when
    # stream is capturing``) -- and this precompute runs inside the graphed
    # region for graph-optimized modules (e.g. the Protenix trunk pairformer,
    # called once per recycling cycle). Skip the (debug) validation during
    # capture; it already ran during the graph tracker's eager warmup for this
    # shape, and the mask is left-aligned by construction in inference.
    if not torch.cuda.is_current_stream_capturing():
        assert torch.all(mask_bool[..., :-1] >= mask_bool[..., 1:]) and \
            torch.all(mask_bool[..., :-1, :] >= mask_bool[..., 1:, :]), (
                "CuTeDSL precompute_pair_masks requires a left-aligned "
                "(``1...1 0...0``) pair_mask along both the last and "
                "second-to-last dims (e.g. the outer product of a left-aligned "
                "seq_mask). Got a pair_mask with interior zeros.")
    # ``actual_s_kv`` (per-row valid J count, int32 ``[B, I]``) doubles as
    # the dual_gemm_x_x ``actual_seqlen`` for ``tri_mul_out`` -- the LM
    # dual_gemm kernel masks row ``g_m`` via
    # ``g_m % J < actual_s_kv[g_m // J]``. Likewise ``actual_s_kv_t``
    # (``[B, J]``) is the ``tri_mul_in`` counterpart. The two are exposed
    # through ``mask_bias`` / ``mask_bias_transposed`` directly so no
    # separate field is needed.
    actual_s_kv = mask_bool.sum(dim=-1).to(dtype=torch.int32).contiguous()
    actual_s_kv_t = mask_bool.sum(dim=-2).to(dtype=torch.int32).contiguous()
    return PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=actual_s_kv,
        mask_bias_transposed=actual_s_kv_t,
    )


register_precompute_pair_masks("CuTeDSL", _cutedsl_precompute_pair_masks)

# ---------------------------------------------------------------------------
# Precomputed single-mask registry (for pairwise / DiffusionTransformer)
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedSingleMasks:
    """Precomputed mask tensors for pairwise attention (AttentionPairBias).

    In DiffusionTransformer / AtomTransformer layers the single mask ``[*, I]``
    is converted to an additive bias identically in every layer.  Precomputing
    it once before the layer loop avoids redundant work.

    Attributes:
        single_mask: Original ``[*, I]`` mask.
        mask_bias: Additive mask bias ready for the attention kernel.
            Shape depends on backend (default: ``[*, 1, 1, I]``).
        mask_bias_local: Additive mask bias after ``query_to_keys``
            transformation for sequence-local atom attention.
            ``None`` when ``query_to_keys`` is not used.
    """
    single_mask: torch.Tensor
    mask_bias: torch.Tensor
    mask_bias_local: Optional[torch.Tensor] = None


PrecomputeSingleMasksFn = Callable[[torch.Tensor, float],
                                   PrecomputedSingleMasks]

_PRECOMPUTE_SINGLE_MASKS_REGISTRY: dict[str, PrecomputeSingleMasksFn] = {}


def register_precompute_single_masks(
    backend_name: str,
    fn: PrecomputeSingleMasksFn,
) -> None:
    """Register a precompute-single-masks callable for *backend_name*."""
    _PRECOMPUTE_SINGLE_MASKS_REGISTRY[backend_name] = fn


def precompute_single_masks(
    backend_name: str,
    single_mask: torch.Tensor,
    inf: float = 1e9,
    query_to_keys: Optional[Callable] = None,
) -> PrecomputedSingleMasks:
    """Dispatch to the registered precompute function for *backend_name*.

    Args:
        backend_name: Pairwise attention backend identifier
            (e.g. ``"VANILLA"``, ``"SDPA"``).
        single_mask: ``[*, I]`` mask tensor (1 = valid, 0 = masked-out).
        inf: Large value used to mask out invalid positions.
        query_to_keys: Optional callable that transforms the mask for
            sequence-local atom attention.  When provided, the transformed
            mask bias is also precomputed and stored in
            ``mask_bias_local``.

    Returns:
        A :class:`PrecomputedSingleMasks` bundle ready to be passed through
        DiffusionTransformer / AtomTransformer layers.
    """
    fn = _PRECOMPUTE_SINGLE_MASKS_REGISTRY.get(backend_name)
    if fn is None:
        raise ValueError(
            f"No precompute_single_masks registered for backend '{backend_name}'. "
            f"Available: {sorted(_PRECOMPUTE_SINGLE_MASKS_REGISTRY.keys())}")
    result = fn(single_mask, inf)
    if query_to_keys is not None:
        local_mask = query_to_keys(single_mask.unsqueeze(-1)).squeeze(-1)
        local_mask = (local_mask > 0).to(local_mask.dtype)
        local_result = fn(local_mask, inf)
        result.mask_bias_local = local_result.mask_bias
    return result


# ---------------------------------------------------------------------------
# Default implementation (shared by VANILLA / SDPA)
# ---------------------------------------------------------------------------


def _default_precompute_single_masks(
    single_mask: torch.Tensor,
    inf: float = 1e9,
) -> PrecomputedSingleMasks:
    mask_bias = (1 - single_mask.float()) * -inf
    mask_bias = mask_bias[..., None, None, :]
    return PrecomputedSingleMasks(
        single_mask=single_mask,
        mask_bias=mask_bias,
    )


def _cutedsl_precompute_single_masks(
    single_mask: torch.Tensor,
    inf: float = 1e9,
) -> PrecomputedSingleMasks:
    mask_bias = single_mask.float()
    return PrecomputedSingleMasks(
        single_mask=single_mask,
        mask_bias=mask_bias,
    )


register_precompute_single_masks("VANILLA", _default_precompute_single_masks)
register_precompute_single_masks("SDPA", _default_precompute_single_masks)
register_precompute_single_masks("CuTeDSL", _cutedsl_precompute_single_masks)
