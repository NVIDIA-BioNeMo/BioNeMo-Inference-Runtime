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
"""A design step's initial pair track in one pass: projected pair features, normalized and conditioned.

The pair features are one-hot classes, the distance bin, the bond order and the bond-path bin, plus a few
scalar channels. A one-hot class selects one column of the projection, so per pair ``(b, i, j)``

    v = W_dist[bin(|x_i - x_j|)] + W_bond[bond] + W_path[path] + sum_s scalar_s * W_scalar[s]
        (+ active[b] * W_self[bin(|x'_i - x'_j|)] for a schema that bins the self-conditioning coordinates x')
    y = round(norm(v))                                 # the pair RMSNorm or LayerNorm, stored in the output dtype
    o = sigmoid(scale[b]) * rms_norm(y) + bias[b]      # the step's AdaLN
    o = 0 outside the pair mask

which equals projecting the concatenated features, up to the order of the FP32 sum. The pair is written once and
nothing else of its size is read.

Credit: kit fast's fused pair-init kernel showed that this step fits in one pass over memory. This kernel follows
its structure and reads the static features as class indices and scalars instead of a dense projection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["rfd4_pair_init"]

#: Pairs per program, along ``j``.
_BLOCK_PAIRS = 16


@triton.jit
def _distance_bin(coords_ptr, row, b, j, tokens, in_range, edges_ptr, EDGES: tl.constexpr, BLOCK_EDGES: tl.constexpr):
    """``torch.bucketize(|x_i - x_j|, edges)`` for one row ``i`` and the pairs' ``j``."""
    # torch.norm reduces a 3-vector as (x + z) + y, rounding each square and each sum in FP32. A fused
    # multiply-add rounds once instead, which can carry a distance on an edge into the next bin, so the squares
    # and sums go through mul_rn and add_rn, which the compiler never contracts.
    squared = tl.zeros(j.shape, tl.float32)
    for step in tl.static_range(3):
        axis = step * 2 % 3
        delta = tl.load(coords_ptr + row * 3 + axis) - tl.load(coords_ptr + (b * tokens + j) * 3 + axis, in_range, 0.0)
        squared = libdevice.add_rn(squared, libdevice.mul_rn(delta, delta))
    distance = tl.sqrt_rn(squared)
    # bucketize counts the edges strictly below the value.
    edge = tl.arange(0, BLOCK_EDGES)
    edges = tl.load(edges_ptr + edge, edge < EDGES, float("inf"))
    return tl.sum((edges[None, :] < distance[:, None]).to(tl.int32), 1)


@triton.jit(do_not_specialize=["tokens", "padded", "norm_eps", "adaln_eps", "self_conditioned", "second"])
def rfd4_pair_init_kernel(
    coords_ptr,
    classes_ptr,
    scalars_ptr,
    mask_ptr,
    edges_ptr,
    w_dist_ptr,
    w_bond_ptr,
    w_path_ptr,
    w_scalar_ptr,
    norm_w_ptr,
    norm_b_ptr,
    scale_ptr,
    bias_ptr,
    self_coords_ptr,
    self_active_ptr,
    w_self_ptr,
    out_ptr,
    out2_ptr,
    tokens,
    padded,
    norm_eps,
    adaln_eps,
    self_conditioned,
    second,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EDGES: tl.constexpr,
    BLOCK_EDGES: tl.constexpr,
    SCALARS: tl.constexpr,
    LAYER_NORM: tl.constexpr,
    BP: tl.constexpr,
):
    """Program ``(b * P + i, k)`` writes pairs ``(b, i, j)`` for ``j`` in block ``k`` of the ``P = padded`` tokens.

    Pairs with a token at or past ``tokens`` are padding and get zeros. A nonzero ``self_conditioned`` adds the
    self-conditioning distance bin, and a nonzero ``second`` writes the pairs to ``out2_ptr`` as well; otherwise
    those pointers are not read.
    """
    out_row = tl.program_id(0).to(tl.int64)
    b = out_row // padded
    i = out_row % padded
    j = tl.program_id(1) * BP + tl.arange(0, BP)
    stored = j < padded
    in_range = (j < tokens) & (i < tokens)
    # Row i of the unpadded inputs; a padding row reads the last real one and keeps none of it.
    row = b * tokens + tl.minimum(i, tokens - 1)
    pair = row * tokens + j
    channel = tl.arange(0, BLOCK_C)
    channels = channel < C
    live = in_range[:, None] & channels[None, :]

    bin_index = _distance_bin(coords_ptr, row, b, j, tokens, in_range, edges_ptr, EDGES, BLOCK_EDGES)
    v = tl.load(w_dist_ptr + bin_index[:, None] * C + channel[None, :], live, 0.0)
    if self_conditioned != 0:
        self_bin = _distance_bin(self_coords_ptr, row, b, j, tokens, in_range, edges_ptr, EDGES, BLOCK_EDGES)
        active = tl.load(self_active_ptr + b)
        v += active * tl.load(w_self_ptr + self_bin[:, None] * C + channel[None, :], live, 0.0)
    bond = tl.load(classes_ptr + pair * 2, in_range, -1)
    v += tl.load(w_bond_ptr + bond[:, None] * C + channel[None, :], live & (bond >= 0)[:, None], 0.0)
    path = tl.load(classes_ptr + pair * 2 + 1, in_range, 0)
    v += tl.load(w_path_ptr + path[:, None] * C + channel[None, :], live, 0.0)
    for s in tl.static_range(SCALARS):
        value = tl.load(scalars_ptr + pair * SCALARS + s, in_range, 0.0)
        v += value[:, None] * tl.load(w_scalar_ptr + s * C + channel, channels, 0.0)[None, :]

    weight = tl.load(norm_w_ptr + channel, channels, 0.0)
    if LAYER_NORM:
        centered = tl.where(channels[None, :], v - (tl.sum(v, 1) / C)[:, None], 0.0)
        rstd = tl.rsqrt(tl.sum(centered * centered, 1) / C + norm_eps)
        y = centered * rstd[:, None] * weight[None, :] + tl.load(norm_b_ptr + channel, channels, 0.0)[None, :]
    else:
        y = v * tl.rsqrt(tl.sum(v * v, 1) / C + norm_eps)[:, None] * weight[None, :]
    # The norm stores the AdaLN's dtype, which the AdaLN reads back.
    y = y.to(out_ptr.dtype.element_ty).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(y * y, 1) / C + adaln_eps)
    scale = tl.sigmoid(tl.load(scale_ptr + b * C + channel, channels, 0.0).to(tl.float32))
    bias = tl.load(bias_ptr + b * C + channel, channels, 0.0).to(tl.float32)
    out = scale[None, :] * (y * rstd[:, None]) + bias[None, :]
    keep = in_range & (tl.load(mask_ptr + row) != 0) & (tl.load(mask_ptr + b * tokens + j, in_range, 0) != 0)
    out = tl.where(keep[:, None], out, 0.0).to(out_ptr.dtype.element_ty)
    offsets = (out_row * padded + j)[:, None] * C + channel[None, :]
    store = stored[:, None] & channels[None, :]
    tl.store(out_ptr + offsets, out, store)
    if second != 0:
        tl.store(out2_ptr + offsets, out, store)


def rfd4_pair_init(
    coordinates: torch.Tensor,
    classes: torch.Tensor,
    scalars: torch.Tensor,
    padding_mask: torch.Tensor,
    edges: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    norm_eps: float,
    adaln_eps: float,
    out_dtype: torch.dtype,
    self_conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Build a step's ``[B, N, N, C]`` pair track: projection, pair norm, AdaLN and mask in one kernel.

    Args:
        coordinates: FP32 ``[B, N, 3]`` noisy backbone coordinates.
        classes: int32 ``[B, N, N, 2]`` bond-order class (``-1`` for none) and bond-path bin per pair.
        scalars: FP32 ``[B, N, N, S]`` scalar pair channels.
        padding_mask: ``[B, N]`` mask of real tokens; pairs with a padding token are written as zeros.
        edges: FP32 ``[K - 1]`` sorted distance-bin edges, binned as ``torch.bucketize`` does.
        weights: FP32 projection rows ``[K, C]`` per distance bin, ``[num_bond_orders, C]`` per bond order,
            ``[path_bins, C]`` per bond-path bin and ``[S, C]`` per scalar channel.
        norm_weight: FP32 ``[C]`` pair-norm weight.
        norm_bias: FP32 ``[C]`` pair-norm bias for a LayerNorm, or ``None`` for an RMSNorm.
        scale: ``[B, C]`` AdaLN scale before its sigmoid, one row per design.
        bias: ``[B, C]`` AdaLN bias, one row per design.
        norm_eps: Pair-norm epsilon.
        adaln_eps: AdaLN RMSNorm epsilon.
        out_dtype: Output dtype, which the pair norm also rounds to.
        self_conditioning: For a schema that also bins the self-conditioning coordinates: those FP32 ``[B, N, 3]``
            coordinates, FP32 ``[B]`` weights of that bin (zero while self-conditioning is off) and its projection
            rows ``[K, C]``.
        out: Optional two contiguous ``[B, P, P, C]`` buffers in ``out_dtype`` with ``P >= N``. Both receive the
            pair track zero-padded to ``P`` tokens, which is the masked pair a padding trunk would build.

    Returns:
        ``[B, N, N, C]`` pair track in ``out_dtype``, a view of the first of ``out`` when given.
    """
    batch, tokens, _ = coordinates.shape
    # The kernel indexes every tensor operand as a dense array.
    w_dist, w_bond, w_path, w_scalar = (weight.contiguous() for weight in weights)
    channels = w_dist.shape[1]
    if out is None:
        # One new tensor, which both output pointers name and the kernel stores once.
        out = (torch.empty((batch, tokens, tokens, channels), device=coordinates.device, dtype=out_dtype),) * 2
    first, second = out
    padded = first.shape[1]
    if padded < tokens or any(
        t.shape != (batch, padded, padded, channels) or t.dtype != out_dtype or not t.is_contiguous()
        for t in (first, second)
    ):
        raise ValueError(
            f"out must be two contiguous [{batch}, P, P, {channels}] {out_dtype} buffers with P >= {tokens}"
        )
    if not first.numel():
        return first[:, :tokens, :tokens]
    self_coords, self_active, w_self = (coordinates, edges, w_dist) if self_conditioning is None else self_conditioning
    rfd4_pair_init_kernel[(batch * padded, triton.cdiv(padded, _BLOCK_PAIRS))](
        coordinates.contiguous(),
        classes.contiguous(),
        scalars.contiguous(),
        padding_mask.contiguous(),
        edges.contiguous(),
        w_dist,
        w_bond,
        w_path,
        w_scalar,
        norm_weight.contiguous(),
        (norm_weight if norm_bias is None else norm_bias).contiguous(),
        scale.contiguous(),
        bias.contiguous(),
        self_coords.contiguous(),
        self_active.contiguous(),
        w_self.contiguous(),
        first,
        second,
        tokens,
        padded,
        norm_eps,
        adaln_eps,
        int(self_conditioning is not None),
        int(second is not first),
        C=channels,
        BLOCK_C=triton.next_power_of_2(channels),
        EDGES=edges.numel(),
        BLOCK_EDGES=triton.next_power_of_2(max(edges.numel(), 1)),
        SCALARS=scalars.shape[-1],
        LAYER_NORM=norm_bias is not None,
        BP=_BLOCK_PAIRS,
        num_warps=4,
    )
    return first[:, :tokens, :tokens]
