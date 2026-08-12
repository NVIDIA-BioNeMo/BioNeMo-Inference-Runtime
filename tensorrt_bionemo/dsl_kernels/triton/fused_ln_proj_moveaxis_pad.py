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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel: fused LayerNorm + Linear projection + moveaxis + pad.

Problem
-------
The pair bias path in AttentionPairBias does:

    z: [B, I, J, D]  ->  LayerNorm(z)  ->  Linear(z) -> [B, I, J, H]  ->  moveaxis+pad -> [B, H, I, J_padded]

Unfused, this is 3 separate kernels:
    1. LayerNorm: read [B,I,J,D], write [B,I,J,D]       -- 2 passes over O(N^2*D) data
    2. Linear:    read [B,I,J,D], write [B,I,J,H]       -- 1 pass
    3. Moveaxis+pad: read [B,I,J,H], write [B,H,I,J_pad] -- 1 pass

The LayerNorm alone accounts for 85-89% of total attention time because the pair
tensor is O(N^2) and memory-bandwidth-bound.

This fused kernel does it all in one pass:
    - Reads z [B, I, J, D] once
    - Computes LayerNorm statistics in registers
    - Applies LayerNorm affine transform + Linear projection (tiled matmul)
    - Writes output directly in [B, H, I, J_padded] layout with padding

Based on cuequivariance's pair_bias_norm_linear_mask_forward_kernel, adapted
for TRT-BNM's layout conventions (no mask application — mask is handled by
the downstream attention kernel).

Launch strategy
---------------
Grid: (ceil_div(J_padded, TILE_J), I, ceil_div(H, HEADS_PER_BLK) * B)

Each program handles a tile of [TILE_J, D] input, computes LN stats,
then accumulates the projection in [TILE_J, HEADS_PER_BLK] tiles over D,
and writes the transposed result to [HEADS_PER_BLK, TILE_J] in the output.

Alignment note
--------------
This kernel is only used when the caller passes ``multiple >= 0`` (i.e. the
CuTeDSL path, typically ``multiple=8``).  In that case J_padded is always a
multiple of 8, so out_stride_h = I * J_padded is divisible by 8, satisfying
the 16-byte alignment required by Triton's vectorised 128-bit stores for every
head h >= 1.  When ``multiple < 0`` callers fall back to the unfused sequence
to avoid potential misalignment for arbitrary sequence lengths.
"""

import math

import torch
import triton
import triton.language as tl

from tensorrt_bionemo.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_ln_proj_moveaxis_pad_kernel(
    # Pointers
    z_ptr,  # input:  [B, I, J, D] contiguous
    w_ln_ptr,  # LN weight: [D]
    b_ln_ptr,  # LN bias:   [D]
    w_proj_ptr,  # projection weight: [H, D]
    out_ptr,  # output: [B, H, I, J_padded] contiguous
    # Dims
    J,
    J_padded,
    # Input strides
    z_stride_b,  # = I * J * D
    z_stride_i,  # = J * D
    z_stride_j,  # = D
    # Output strides
    out_stride_b,  # = H * I * J_padded
    out_stride_h,  # = I * J_padded
    out_stride_i,  # = J_padded
    # Tile constants
    TILE_J: tl.constexpr,
    TILE_K: tl.constexpr,  # tile over D dimension
    DIM_D: tl.constexpr,  # full D (known at compile time)
    NUM_HEADS: tl.constexpr,  # total number of heads
    HEADS_PER_BLK: tl.constexpr,  # heads per program (power of 2)
    EPS: tl.constexpr,
    ELEMENTWISE_AFFINE: tl.constexpr,
):
    """Fused LayerNorm + Linear + moveaxis + pad.

    Input:  z[b, i, j, :D]        -- contiguous [B, I, J, D]
    Output: out[b, h, i, j_padded] -- contiguous [B, H, I, J_padded]

    Each program handles one (b, i, j_tile, head_block).
    """
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1).to(tl.int64)
    head_batch_idx = tl.program_id(2)

    num_head_blks = tl.cdiv(NUM_HEADS, HEADS_PER_BLK)
    batch_idx = (head_batch_idx // num_head_blks).to(tl.int64)
    head_blk_idx = head_batch_idx % num_head_blks

    offs_j = pid_j * TILE_J + tl.arange(0, TILE_J)
    offs_d = tl.arange(0, DIM_D)
    offs_h = head_blk_idx * HEADS_PER_BLK + tl.arange(0, HEADS_PER_BLK)
    # ``pid_i`` and ``batch_idx`` are promoted to i64 above because at
    # OF3 token-transformer scale (I=J=4832, D=128) the per-row offset
    #   pid_i * z_stride_i = 4831 * (J*D) = 2.99e9
    # overflows i32 (INT32_MAX = 2.15e9). The same applies to
    #   batch_idx * z_stride_b
    # for B>1 or H growth on the output side. ``pid_j`` and offsets in
    # the J/D/H dims stay i32 (max ~5k) — no overflow risk there.

    mask_j = offs_j < J
    mask_h = offs_h < NUM_HEADS

    # ── Pass 1: compute LayerNorm statistics ──────────────────
    # Load full z_tile [TILE_J, DIM_D] for mean/var computation
    z_base = z_ptr + batch_idx * z_stride_b + pid_i * z_stride_i
    z_stat_ptrs = z_base + offs_j[:, None] * z_stride_j + offs_d[None, :]

    z_full = tl.load(z_stat_ptrs, mask=mask_j[:, None], other=0.0).to(tl.float32)

    mean = tl.sum(z_full, axis=1) / DIM_D  # [TILE_J]
    var = z_full - mean[:, None]
    var = tl.sum(var * var, axis=1) / DIM_D  # [TILE_J]
    rstd = tl.rsqrt(var + EPS)  # [TILE_J]

    # ── Pass 2: fused LN + projection ───────────────────────
    # Accumulate: out[j, h] = sum_k( LN(z[j, k]) * w_proj[h, k] )
    acc = tl.zeros((TILE_J, HEADS_PER_BLK), dtype=tl.float32)

    if TILE_K == DIM_D:
        # Fast path: TILE_K covers the full D dimension, so z_full
        # (already in registers from the stats pass) can be reused
        # directly — no second global memory read.
        z_full = (z_full - mean[:, None]) * rstd[:, None]

        if ELEMENTWISE_AFFINE:
            w_ln_full = tl.load(w_ln_ptr + offs_d).to(tl.float32)
            b_ln_full = tl.load(b_ln_ptr + offs_d).to(tl.float32)
            z_full = z_full * w_ln_full + b_ln_full

        w_proj_ptrs = w_proj_ptr + offs_h[None, :] * DIM_D + offs_d[:, None]
        w_tile = tl.load(w_proj_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)

        acc = tl.dot(z_full, w_tile, acc, input_precision="tf32")
    else:
        # Tiled path: iterate over D in TILE_K chunks, reloading z
        # from global memory (necessary when D > TILE_K).
        k_offset = 0
        for _tile in range(DIM_D // TILE_K):
            tile_k = tl.arange(0, TILE_K) + k_offset

            z_tile_ptrs = z_base + offs_j[:, None] * z_stride_j + tile_k[None, :]
            z_tile = tl.load(z_tile_ptrs, mask=mask_j[:, None], other=0.0).to(tl.float32)

            z_tile = (z_tile - mean[:, None]) * rstd[:, None]

            if ELEMENTWISE_AFFINE:
                w_ln_tile = tl.load(w_ln_ptr + tile_k).to(tl.float32)
                b_ln_tile = tl.load(b_ln_ptr + tile_k).to(tl.float32)
                z_tile = z_tile * w_ln_tile + b_ln_tile

            w_proj_ptrs = w_proj_ptr + offs_h[None, :] * DIM_D + tile_k[:, None]
            w_tile = tl.load(w_proj_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)

            acc = tl.dot(z_tile, w_tile, acc, input_precision="tf32")

            k_offset += TILE_K

    # ── Store: transposed layout [B, H, I, J_padded] ─────────
    # acc is [TILE_J, HEADS_PER_BLK]
    # Output location: out[batch_idx, offs_h, pid_i, offs_j]
    out_base = out_ptr + batch_idx * out_stride_b + pid_i * out_stride_i
    out_ptrs = out_base + offs_h[None, :] * out_stride_h + offs_j[:, None]

    store_mask = (offs_j[:, None] < J_padded) & mask_h[None, :]
    out = tl.where(mask_j[:, None], acc, 0.0)
    tl.store(out_ptrs, out, mask=store_mask)


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

_TILE_J_DEFAULT = 32


class FusedLNProjMoveaxisPad(TritonKernelCache):
    """Fused LayerNorm + Linear projection + moveaxis(-1,-3) + pad.

    Replaces the sequence:
        z_normed = layer_norm(z)           # [B, I, J, D]
        proj = linear(z_normed)            # [B, I, J, H]
        out = moveaxis_pad(proj)           # [B, H, I, J_padded]

    with a single kernel that reads z once and writes the transposed output.

    Only used when ``multiple >= 0`` (CuTeDSL path) so that J_padded is
    always a multiple of 8, guaranteeing 16-byte aligned stores in the kernel.

    Args:
        D: pair feature dimension (e.g. 64 or 128)
        H: number of attention heads
        tile_j: J-dimension tile size (default 64)
        dtype: compute dtype
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(self, D: int, H: int, tile_j: int = _TILE_J_DEFAULT, dtype: torch.dtype = torch.bfloat16):
        self._dim_d = D
        self._num_heads = H
        self._tile_j = tile_j
        self._tile_k = min(D, 128) if D > 16 else 16
        self._heads_per_blk = min(triton.next_power_of_2(H), 16)
        self._dtype = dtype
        self._kernels: dict[torch.dtype | tuple[torch.dtype, ...], CachedKernel] = {}

        if torch.cuda.is_available():
            self._ensure_compiled()

    def _ensure_compiled(self):
        if self._kernels:
            return
        base_key = (self._dim_d, self._num_heads, self._tile_j, self._tile_k, self._heads_per_blk)
        dtypes = [self._dtype, torch.bfloat16, torch.float32]
        common_kwargs = {
            "TILE_J": self._tile_j,
            "TILE_K": self._tile_k,
            "DIM_D": self._dim_d,
            "NUM_HEADS": self._num_heads,
            "HEADS_PER_BLK": self._heads_per_blk,
            "EPS": 1e-5,
            "ELEMENTWISE_AFFINE": True,
        }

        cached = FusedLNProjMoveaxisPad._global_cache.get(base_key)
        if cached:
            self._kernels = cached
        else:
            result = self.compile_for_dtypes(
                _fused_ln_proj_moveaxis_pad_kernel,
                dtypes=dtypes,
                make_dummy_args=lambda dt: self._make_dummy_args(dt),
                grid=(1, 2, 1),
                **common_kwargs,
            )
            mixed_result = self.compile_for_dtypes(
                _fused_ln_proj_moveaxis_pad_kernel,
                dtypes=[dt for dt in dtypes if dt != torch.float32],
                make_dummy_args=lambda dt: self._make_dummy_args(dt, ln_dtype=torch.float32),
                grid=(1, 2, 1),
                **common_kwargs,
            )
            result.update(
                {
                    (dtype, torch.float32, torch.float32, dtype): kernel
                    for dtype, kernel in mixed_result.items()
                }
            )
            self._kernels = result
            FusedLNProjMoveaxisPad._global_cache[base_key] = result

    def _make_dummy_args(self, dtype: torch.dtype, ln_dtype: torch.dtype | None = None) -> tuple:
        D, H, tj = self._dim_d, self._num_heads, self._tile_j
        ln_dtype = ln_dtype or dtype
        z = torch.empty(1, 2, tj, D, dtype=dtype, device="cuda")
        w_ln = torch.empty(D, dtype=ln_dtype, device="cuda")
        b_ln = torch.empty(D, dtype=ln_dtype, device="cuda")
        w_proj = torch.empty(H, D, dtype=dtype, device="cuda")
        out = torch.empty(1, H, 2, tj, dtype=dtype, device="cuda")
        return (
            z,
            w_ln,
            b_ln,
            w_proj,
            out,
            tj,
            tj,
            z.stride(0),
            z.stride(1),
            z.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )

    def __call__(
        self,
        z: torch.Tensor,
        w_ln: torch.Tensor,
        b_ln: torch.Tensor,
        w_proj: torch.Tensor,
        multiple: int = 8,
    ) -> torch.Tensor:
        """Fused LN + projection + moveaxis + pad.

        Args:
            z: input pair tensor [*, I, J, D] (contiguous)
            w_ln: LayerNorm weight [D]
            b_ln: LayerNorm bias [D]
            w_proj: projection weight [H, D]
            multiple: pad J to next multiple (must be >= 0)

        Returns:
            Projected output [*, H, I, J_padded]
        """
        *lead, I, J, D = z.shape
        H = w_proj.shape[0]
        B = math.prod(lead) if lead else 1
        z3 = z.reshape(B, I, J, D)

        if multiple <= 0:
            J_padded = J
        else:
            J_padded = ((J + multiple - 1) // multiple) * multiple

        out = torch.empty(B, H, I, J_padded, device=z.device, dtype=z.dtype)

        grid_j = triton.cdiv(J_padded, self._tile_j)
        num_head_blks = triton.cdiv(H, self._heads_per_blk)

        signature = (z.dtype, w_ln.dtype, b_ln.dtype, w_proj.dtype)
        kernel = self._kernels.get(z.dtype if len(set(signature)) == 1 else signature)

        if kernel is not None and kernel.driver is not None:
            drv = kernel.driver
            drv.params[0].value = z3.data_ptr()
            drv.params[1].value = w_ln.data_ptr()
            drv.params[2].value = b_ln.data_ptr()
            drv.params[3].value = w_proj.data_ptr()
            drv.params[4].value = out.data_ptr()
            drv.params[5].value = J
            drv.params[6].value = J_padded
            drv.params[7].value = z3.stride(0)
            drv.params[8].value = z3.stride(1)
            drv.params[9].value = z3.stride(2)
            drv.params[10].value = out.stride(0)
            drv.params[11].value = out.stride(1)
            drv.params[12].value = out.stride(2)
            drv.launch(grid_j, I, num_head_blks * B)
        else:
            _fused_ln_proj_moveaxis_pad_kernel[(grid_j, I, num_head_blks * B)](
                z3,
                w_ln,
                b_ln,
                w_proj,
                out,
                J,
                J_padded,
                z3.stride(0),
                z3.stride(1),
                z3.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                TILE_J=self._tile_j,
                TILE_K=self._tile_k,
                DIM_D=self._dim_d,
                NUM_HEADS=self._num_heads,
                HEADS_PER_BLK=self._heads_per_blk,
                EPS=1e-5,
                ELEMENTWISE_AFFINE=True,
            )

        out_shape = list(lead) + [H, I, J_padded] if lead else [H, I, J_padded]
        return out.view(out_shape)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def fused_ln_proj_moveaxis_pad(
    z: torch.Tensor,
    w_ln: torch.Tensor,
    b_ln: torch.Tensor,
    w_proj: torch.Tensor,
    multiple: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused LayerNorm + Linear projection + moveaxis(-1,-3) + pad.

    Args:
        z: input [*, I, J, D] contiguous
        w_ln: LayerNorm weight [D]
        b_ln: LayerNorm bias [D]
        w_proj: projection weight [H, D]
        multiple: pad J to next multiple (must be >= 0)
        eps: LayerNorm epsilon

    Returns:
        [*, H, I, J_padded] contiguous
    """
    *lead, I, J, D = z.shape
    H = w_proj.shape[0]
    B = math.prod(lead) if lead else 1
    z3 = z.reshape(B, I, J, D)

    if multiple <= 0:
        J_padded = J
    else:
        J_padded = ((J + multiple - 1) // multiple) * multiple
    tile_k = min(D, 128) if D > 16 else 16
    heads_per_blk = min(triton.next_power_of_2(H), 16)
    tile_j = _TILE_J_DEFAULT

    out = torch.empty(B, H, I, J_padded, device=z.device, dtype=z.dtype)

    grid_j = triton.cdiv(J_padded, tile_j)
    num_head_blks = triton.cdiv(H, heads_per_blk)

    _fused_ln_proj_moveaxis_pad_kernel[(grid_j, I, num_head_blks * B)](
        z3,
        w_ln,
        b_ln,
        w_proj,
        out,
        J,
        J_padded,
        z3.stride(0),
        z3.stride(1),
        z3.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TILE_J=tile_j,
        TILE_K=tile_k,
        DIM_D=D,
        NUM_HEADS=H,
        HEADS_PER_BLK=heads_per_blk,
        EPS=eps,
        ELEMENTWISE_AFFINE=True,
    )

    out_shape = list(lead) + [H, I, J_padded] if lead else [H, I, J_padded]
    return out.view(out_shape)
