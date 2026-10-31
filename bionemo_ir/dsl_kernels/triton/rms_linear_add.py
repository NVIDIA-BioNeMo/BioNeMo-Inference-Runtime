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
"""An RMSNorm and a bias-free linear layer in one GEMM, plus a residual.

RMSNorm followed by a bias-free linear layer factors as
``rmsnorm(x) @ W^T = rstd * (x @ (W * gamma)^T)``, with ``rstd = rsqrt(sum_d x^2 / D + eps)``
one scale per row. When the row sums of squares are already known, :func:`rms_linear_add`
reads ``x`` once, in the GEMM, and applies each row's ``rstd`` and adds the residual in its
epilogue, so the product is never rounded on its own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["rms_linear_add"]

#: Row tile and pipeline stages for calls of ``_LARGE_ROWS`` rows or more, and for smaller ones. On H100 the 128-row
#: tile is 1.45x cuBLAS plus a separate rescale at 8x400 pair sizes, and the 64-row tile keeps calls of a few
#: thousand rows level with them.
_LARGE_TILE, _SMALL_TILE = (128, 3), (64, 4)
_LARGE_ROWS = 65536
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit(do_not_specialize=["rows", "use_active"])
def rms_linear_add_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    row_sumsq_ptr,
    out_ptr,
    active_ptr,
    rows,
    use_active,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    """Program ``p`` writes row block ``p // (N // BLOCK_N)`` and column block ``p % (N // BLOCK_N)``.

    Consecutive programs share a row block, so its ``x`` rows come from L2 after the first of them reads them.
    With ``use_active`` set, a zero at ``active_ptr`` makes every program return before it reads anything.
    """
    if (use_active != 0) & (tl.load(active_ptr) == 0):
        return
    num_n: tl.constexpr = N // BLOCK_N
    pid = tl.program_id(0)
    offs_m = (pid // num_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = (pid % num_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    row_offsets = offs_m.to(tl.int64)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + row_offsets[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        weight = tl.load(weight_ptr + offs_n[None, :] * K + offs_k[:, None])
        acc = tl.dot(x, weight, acc)
    rstd = tl.rsqrt(tl.load(row_sumsq_ptr + offs_m, mask=mask_m, other=0.0) / K + EPS)
    out_offsets = row_offsets[:, None] * N + offs_n[None, :]
    residual = tl.load(residual_ptr + out_offsets, mask=mask_m[:, None], other=0.0)
    out = residual.to(tl.float32) + acc * rstd[:, None]
    tl.store(out_ptr + out_offsets, out.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


def _block(width: int, largest: int) -> int | None:
    """The widest power-of-two tile from 16 to ``largest`` that divides ``width``."""
    block = largest
    while block >= 16:
        if width % block == 0:
            return block
        block //= 2
    return None


def rms_linear_add(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    row_sumsq: torch.Tensor,
    *,
    eps: float,
    out: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """``residual + rsqrt(row_sumsq / K + eps) * (x @ weight^T)``, each row scaled by its own factor.

    With ``weight`` a linear layer's weight with an RMSNorm's gamma folded in and ``row_sumsq`` each row's
    ``sum_d x^2``, this is ``residual + linear(rmsnorm(x))``. The kernel accumulates the product in FP32 and rounds
    once, after the residual.

    Args:
        residual: ``[*, N]`` tensor added to the scaled product.
        x: ``[*, K]`` rows to project, in ``residual``'s dtype.
        weight: ``[N, K]`` projection.
        row_sumsq: FP32 ``[*]`` sums of squares of ``x``'s rows.
        eps: RMSNorm epsilon.
        out: Optional output; it may hold ``residual`` but not ``x``, which the GEMM still reads while
            it writes. An ``out`` sharing ``x``'s storage receives a copy.
        active: Optional ``[1]`` int32 flag on ``x``'s device. A zero skips the work, read on the device, and
            leaves the result undefined; the PyTorch form computes it regardless.

    Returns:
        ``out``, or a new tensor shaped like ``residual``.
    """
    rows, (n, k) = row_sumsq.numel(), weight.shape
    block_n, block_k = _block(n, 128), _block(k, 64)
    if not (
        x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.dtype == weight.dtype == residual.dtype
        and row_sumsq.dtype == torch.float32
        and block_n is not None
        and block_k is not None
        and x.shape[:-1] == residual.shape[:-1] == row_sumsq.shape
        and x.shape[-1] == k
        and residual.shape[-1] == n
        and all(t.is_contiguous() and t.device == x.device for t in (x, weight, residual, row_sumsq))
        and (active is None or (active.dtype == torch.int32 and active.numel() == 1 and active.device == x.device))
        and (out is None or (out.shape == residual.shape and out.dtype == residual.dtype and out.is_contiguous()))
    ):
        rstd = torch.rsqrt(row_sumsq.float() / k + eps).unsqueeze(-1)
        result = residual.float() + rstd * F.linear(x, weight).float()
        return result.to(residual.dtype) if out is None else out.copy_(result)
    shares_x = out is not None and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    target = torch.empty_like(residual) if out is None or shares_x else out
    if rows:
        block_m, stages = _LARGE_TILE if rows >= _LARGE_ROWS else _SMALL_TILE
        rms_linear_add_kernel[(triton.cdiv(rows, block_m) * (n // block_n),)](
            x,
            weight,
            residual,
            row_sumsq,
            target,
            # Without a flag the kernel ignores what it loads here; an int32 view keeps one compiled variant.
            row_sumsq.view(torch.int32) if active is None else active,
            rows,
            int(active is not None),
            N=n,
            K=k,
            EPS=eps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            STAGES=stages,
            num_warps=4,
            num_stages=stages,
        )
    return out.copy_(target) if shares_x else target
