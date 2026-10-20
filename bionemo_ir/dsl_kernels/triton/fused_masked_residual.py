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
"""Triton masked residual.

Adds ``update`` into ``value`` over a single or pair representation in place,
then writes exact zero where ``mask`` is false. ``update`` may be strided
across token axes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: Rows per program. Four keeps the tile wide enough to saturate the loads
#: without pushing a 512-channel block past the register budget.
_ROWS_PER_PROGRAM = 4


@triton.jit
def _masked_residual_kernel(
    pair_ptr,
    update_ptr,
    mask_ptr,
    n_rows,
    tokens,
    update_stride_b,
    update_stride_i,
    update_stride_j,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    PAIR_LAYOUT: tl.constexpr,
):
    """Accumulate one tile of rows, reading ``update`` through its own strides."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_valid = rows < n_rows
    offs_d = tl.arange(0, BLOCK_D)
    live = row_valid[:, None] & (offs_d < D)[None, :]

    packed = rows.to(tl.int64)[:, None] * D + offs_d[None, :]
    if PAIR_LAYOUT:
        j = rows % tokens
        bi = rows // tokens
        update_row = (
            (bi // tokens).to(tl.int64) * update_stride_b
            + (bi % tokens).to(tl.int64) * update_stride_i
            + j.to(tl.int64) * update_stride_j
        )
    else:
        update_row = (rows // tokens).to(tl.int64) * update_stride_b + (rows % tokens).to(tl.int64) * update_stride_i
    strided = update_row[:, None] + offs_d[None, :]

    m = tl.load(mask_ptr + rows, mask=row_valid, other=0.0).to(tl.float32)
    p = tl.load(pair_ptr + packed, mask=live, other=0.0).to(tl.float32)
    u = tl.load(update_ptr + strided, mask=live, other=0.0).to(tl.float32)
    accumulated = p + u
    tl.store(pair_ptr + packed, tl.where(m[:, None] != 0, accumulated, 0.0), mask=live)


def supports_masked_residual(pair: torch.Tensor, update: torch.Tensor, mask: torch.Tensor) -> bool:
    """Whether the kernel can take these operands as given.

    Single ``[B, N, C]`` and pair ``[B, N, N, C]`` layouts are supported.
    The update may be strided across token axes but its channels must be
    contiguous. The destination and mask must be packed.
    """
    if pair.device.type != "cuda" or pair.dtype not in _SUPPORTED_DTYPES:
        return False
    if update.device != pair.device or mask.device != pair.device:
        return False
    if update.dtype != pair.dtype or mask.dtype != pair.dtype:
        return False
    if pair.ndim not in (3, 4) or update.shape != pair.shape:
        return False
    if pair.ndim == 3:
        batch, tokens, _ = pair.shape
        if mask.shape != (batch, tokens, 1):
            return False
    else:
        batch, tokens, tokens_j, _ = pair.shape
        if tokens != tokens_j or mask.shape != (batch, tokens, tokens, 1):
            return False
    return pair.is_contiguous() and mask.is_contiguous() and update.stride(-1) == 1


def fused_masked_residual(pair: torch.Tensor, update: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Add ``update`` into ``pair`` and re-apply ``mask``, in place.

    Args:
        pair: Contiguous ``[B, N, C]`` or ``[B, N, N, C]`` destination.
        update: Same shape as ``pair``. Token axes may be strided.
        mask: Contiguous ``[B, N, 1]`` or ``[B, N, N, 1]`` mask.

    Returns:
        ``pair``, updated.

    Raises:
        ValueError: If the operands are outside what the kernel addresses.
    """
    if not supports_masked_residual(pair, update, mask):
        raise ValueError(
            "unsupported masked-residual operands: "
            f"pair {tuple(pair.shape)} {pair.dtype} contiguous={pair.is_contiguous()}, "
            f"update {tuple(update.shape)} strides {tuple(update.stride())}, "
            f"mask {tuple(mask.shape)}"
        )
    if pair.ndim == 4:
        _, tokens, _, channels = pair.shape
        update_stride_b, update_stride_i, update_stride_j = update.stride()[:3]
    else:
        _, tokens, channels = pair.shape
        update_stride_b, update_stride_i = update.stride()[:2]
        update_stride_j = 0
    n_rows = pair.numel() // channels
    _masked_residual_kernel[(triton.cdiv(n_rows, _ROWS_PER_PROGRAM),)](
        pair,
        update,
        mask,
        n_rows,
        tokens,
        update_stride_b,
        update_stride_i,
        update_stride_j,
        D=channels,
        BLOCK_D=triton.next_power_of_2(channels),
        ROWS=_ROWS_PER_PROGRAM,
        PAIR_LAYOUT=pair.ndim == 4,
        num_warps=8,
        num_stages=2,
    )
    return pair
