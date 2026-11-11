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
"""Pair-weighted averaging prologue: the masked softmax of the pair-weight logits in one pass, J-padded.

Computes ``softmax(b + (1 - mask) * -inf, dim=-1)`` for ``b`` of shape ``[B, H, N, N]`` and ``mask`` of shape
``[B, N, N]``, rounding the bias terms to the input dtype as the PyTorch expression does, and writes
``[B, H, N, Jp]`` with zeros in ``[N, Jp)``. PyTorch spends a pass on the bias, a temporary for the broadcast mask
and two or three passes on a long-row softmax; this reads ``b`` and ``mask`` once and writes once.
"""

import torch
import triton
import triton.language as tl

# Rows longer than this (in elements) take the PyTorch path: one row lives in registers.
MAX_ROW = 16384


@triton.jit
def _pwa_prologue_kernel(
    b_ptr,
    mask_ptr,
    w_ptr,
    N,
    JP,
    H,
    stride_b_row,
    stride_mask_b,
    stride_mask_i,
    stride_w_row,
    NEG_INF,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = row % N
    bb = row // N // H
    dtype = w_ptr.dtype.element_ty
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(b_ptr + row * stride_b_row + cols, mask=valid, other=0.0).to(tl.float32)
    m = tl.load(mask_ptr + bb * stride_mask_b + i * stride_mask_i + cols, mask=valid, other=1.0).to(tl.float32)
    # b + (1 - mask) * -inf, each step rounded to the input dtype like the eager expression.
    bias = (1.0 - m).to(dtype).to(tl.float32)
    bias = (bias * NEG_INF).to(dtype).to(tl.float32)
    x = (x + bias).to(dtype).to(tl.float32)
    x = tl.where(valid, x, -float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    e = tl.where(valid, e, 0.0)
    out = e / tl.sum(e, axis=0)
    tl.store(w_ptr + row * stride_w_row + cols, out.to(dtype), mask=cols < JP)


def pwa_prologue(b: torch.Tensor, mask: torch.Tensor, inf: float) -> torch.Tensor | None:
    """``softmax(b + (1 - mask[:, None]) * -inf, -1)`` zero-padded to a multiple of 8 columns, the PWA kernels' w pitch.

    Triton specializes ``N`` and the strides per call: a multiple of 16 lets the masked row loads move 16 bytes at a
    time, which a precompiled kernel with unspecialized integers loses (2-4x slower at 4096-8192 tokens on B200).

    Args:
        b: ``[B, H, N, N]`` contiguous bf16 or fp16 logits.
        mask: ``[B, N, N]`` pair mask, last dim contiguous, same dtype as ``b``.
        inf: The large value masked logits are pushed down by.

    Returns:
        ``[B, H, N, Jp]`` weights, or ``None`` when this kernel does not take the shape (the caller then runs the
        PyTorch expression).
    """
    if (
        not b.is_cuda
        or b.dim() != 4
        or mask.dim() != 3
        or b.dtype not in (torch.bfloat16, torch.float16)
        or mask.dtype != b.dtype
        or b.shape[-1] != b.shape[-2]
        or b.shape[-1] > MAX_ROW
        or not b.is_contiguous()
        or mask.stride(-1) != 1
    ):
        return None
    B, H, N, _ = b.shape
    if mask.shape != (B, N, N):
        return None
    jp = (N + 7) // 8 * 8
    w = torch.empty(B, H, N, jp, dtype=b.dtype, device=b.device)
    if B * H * N == 0:
        return w
    block = triton.next_power_of_2(jp)
    _pwa_prologue_kernel[(B * H * N,)](
        b,
        mask,
        w,
        N,
        jp,
        H,
        b.stride(-2),
        mask.stride(0),
        mask.stride(1),
        w.stride(-2),
        -float(inf),
        BLOCK=block,
        num_warps=16 if block >= 8192 else (8 if block >= 2048 else 4),
    )
    return w
