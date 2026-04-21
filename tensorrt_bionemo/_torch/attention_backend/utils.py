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
import torch.nn.functional as F
from tensorrt_llm_lite._utils import get_sm_version

from .cuequiv import CuEquivAttention
from .interface import AttentionBackend, AttentionType
from .pairwise_attention_cute import PairwiseAttentionCuTe
from .sdpa import SDPAPairwiseAttention
from .triangle_attention_cute import TriangleAttentionCuTe
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
            return TriangleAttentionCuTe
    elif attention_type == AttentionType.PAIRWISE:
        if backend_name == "VANILLA":
            return VanillaPairwiseAttention
        elif backend_name == "SDPA":
            return SDPAPairwiseAttention
        elif backend_name == "CuTeDSL":
            return PairwiseAttentionCuTe
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
    """Precomputed mask tensors for triangle attention / multiplication nodes.

    Computing mask_bias from pair_mask is identical across layers, so doing it
    once before the layer loop avoids redundant work.

    Attributes:
        pair_mask: Original [B, I, J] mask for TriangleMultiplicationNode.
        mask_bias: Mask for TriangleAttentionStartingNode.
            Shape depends on the backend: [B, I, 1, 1, J] additive bias for
            default backends, [B, I, J_padded] binary float32 mask for CuTeDSL.
        mask_bias_transposed: Mask for TriangleAttentionEndingNode.
            Shape depends on the backend: [B, J, 1, 1, I] additive bias for
            default backends, [B, J, I_padded] binary float32 mask for CuTeDSL.
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
    mask_bias = (inf * (mask_typed - 1))[..., :, None, None, :]
    mask_bias_transposed = (
        inf * (mask_typed.transpose(-2, -1) - 1))[..., :, None,
                                                  None, :].contiguous()
    return PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=mask_bias,
        mask_bias_transposed=mask_bias_transposed,
    )


register_precompute_pair_masks("VANILLA", _default_precompute_pair_masks)
register_precompute_pair_masks("CUEQUIV", _default_precompute_pair_masks)
register_precompute_pair_masks("TRIFAST", _default_precompute_pair_masks)

# ---------------------------------------------------------------------------
# CuTeDSL implementation — float32 binary mask (0/1), last dim padded to multiple of 8
# ---------------------------------------------------------------------------

_CUTEDSL_ALIGN = 8


def _pad_last_dim_to_multiple(
    t: torch.Tensor,
    align: int,
    value: float = 0.0,
) -> torch.Tensor:
    cur = t.size(-1)
    padded = (cur + align - 1) // align * align
    if padded > cur:
        return F.pad(t, (0, padded - cur), value=value)
    return t


def _cutedsl_precompute_pair_masks(
    pair_mask: torch.Tensor,
    inf: float = 1e9,
    dtype: Optional[torch.dtype] = None,
) -> PrecomputedPairMasks:
    mask_f32 = pair_mask.to(torch.float32)

    mask_bias = _pad_last_dim_to_multiple(mask_f32, _CUTEDSL_ALIGN)

    mask_t = mask_f32.transpose(-2, -1)
    mask_bias_transposed = _pad_last_dim_to_multiple(mask_t, _CUTEDSL_ALIGN)

    return PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=mask_bias,
        mask_bias_transposed=mask_bias_transposed,
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
