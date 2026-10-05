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

"""Triton kernel: fused LayerNorm + Linear projection + moveaxis + pad.

The pair bias path in ``AttentionPairBias`` normalizes, projects and transposes::

    z: [B, I, J, D] -> LayerNorm -> Linear -> [B, I, J, H] -> [B, H, I, J_padded]

Unfused that is three kernels over an O(N^2) tensor, dominated by the
bandwidth-bound LayerNorm. This reads ``z`` once, keeps the LayerNorm statistics in
registers, folds the affine transform into a tiled matmul against the projection
weight, and writes straight into the padded transposed layout.

Grid: ``(ceil_div(J_padded, TILE_J), I, ceil_div(H, HEADS_PER_BLK) * B)``, one
``[TILE_J, D]`` tile per program accumulated into ``[TILE_J, HEADS_PER_BLK]``.

Low-precision inputs with ``D % 64 == 0`` take a persistent streaming kernel instead: the
norm folds into the projection, so ``z`` is read once in exact 64-wide slices and fed to a
bf16/fp16 dot without materializing the normalized tile.

Based on NVIDIA cuEquivariance's Apache-2.0
``pair_bias_norm_linear_mask_forward_kernel`` and adapted to BioIR layouts; the
mask is left to the downstream attention kernel. BioIR currently integrates
cuEquivariance 0.11.1: https://github.com/NVIDIA/cuEquivariance/tree/v0.11.1
"""

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


# One CUBIN serves every runtime shape, so no argument may carry a specialization
# from the value it held at compile time. ``_make_dummy_args`` makes J, J_padded and
# all six strides multiples of 16, which Triton turns into a divisible_by_16
# assumption; launching that CUBIN with J=30 then corrupts and overruns the output.
@triton.jit(
    do_not_specialize=[
        "J",
        "J_padded",
        "z_stride_b",
        "z_stride_i",
        "z_stride_j",
        "out_stride_b",
        "out_stride_h",
        "out_stride_i",
    ]
)
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
    BLOCK_D: tl.constexpr,  # next power of two >= DIM_D, for reductions
    DIM_D: tl.constexpr,  # full D (known at compile time)
    NUM_HEADS: tl.constexpr,  # total number of heads
    HEADS_PER_BLK: tl.constexpr,  # heads per program (power of 2)
    EPS: tl.constexpr,
    ELEMENTWISE_AFFINE: tl.constexpr,
    RMS_NORM: tl.constexpr,  # RMSNorm (no mean subtraction, no bias) vs LayerNorm
):
    """Fused LayerNorm/RMSNorm + Linear + moveaxis + pad.

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
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = head_blk_idx * HEADS_PER_BLK + tl.arange(0, HEADS_PER_BLK)
    # ``pid_i`` and ``batch_idx`` are i64 above because at OF3 token-transformer
    # scale (I=J=4832, D=128) the per-row offset pid_i * z_stride_i reaches 2.99e9
    # and overflows i32; ``batch_idx * z_stride_b`` does the same for B>1 or more
    # heads. Offsets along J/D/H stay i32 -- they never exceed ~5k.

    mask_j = offs_j < J
    mask_d = offs_d < DIM_D
    mask_h = offs_h < NUM_HEADS

    # ── Pass 1: compute LayerNorm statistics ──────────────────
    # Triton reductions require a power-of-two extent. Padded lanes must be
    # masked both at load and after centering, otherwise LayerNorm variance
    # would include ``BLOCK_D - DIM_D`` copies of ``-mean``.
    z_base = z_ptr + batch_idx * z_stride_b + pid_i * z_stride_i
    z_stat_ptrs = z_base + offs_j[:, None] * z_stride_j + offs_d[None, :]

    z_full = tl.load(z_stat_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # RMSNorm skips mean subtraction (mean fixed to 0); LayerNorm centers.
    if RMS_NORM:
        mean = tl.zeros([TILE_J], dtype=tl.float32)  # [TILE_J]
    else:
        mean = tl.sum(z_full, axis=1) / DIM_D  # [TILE_J]
    cent = tl.where(mask_d[None, :], z_full - mean[:, None], 0.0)
    var = tl.sum(cent * cent, axis=1) / DIM_D  # [TILE_J]
    rstd = tl.rsqrt(var + EPS)  # [TILE_J]

    # ── Pass 2: fused LN + projection ───────────────────────
    # Accumulate: out[j, h] = sum_k( LN(z[j, k]) * w_proj[h, k] )
    acc = tl.zeros((TILE_J, HEADS_PER_BLK), dtype=tl.float32)

    if TILE_K == BLOCK_D:
        # Fast path: TILE_K covers the padded reduction dimension, so z_full
        # (already in registers from the stats pass) can be reused
        # directly — no second global memory read.
        z_full = cent * rstd[:, None]

        if ELEMENTWISE_AFFINE:
            w_ln_full = tl.load(w_ln_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            if RMS_NORM:
                z_full = z_full * w_ln_full
            else:
                b_ln_full = tl.load(b_ln_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                z_full = z_full * w_ln_full + b_ln_full

        w_proj_ptrs = w_proj_ptr + offs_h[None, :] * DIM_D + offs_d[:, None]
        w_tile = tl.load(w_proj_ptrs, mask=mask_d[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        acc = tl.dot(z_full, w_tile, acc, input_precision="tf32")
    else:
        # Tiled path: iterate over D in TILE_K chunks, reloading z
        # from global memory (necessary when D > TILE_K).
        k_offset = 0
        for _tile in range((DIM_D + TILE_K - 1) // TILE_K):
            tile_k = tl.arange(0, TILE_K) + k_offset
            mask_k = tile_k < DIM_D

            z_tile_ptrs = z_base + offs_j[:, None] * z_stride_j + tile_k[None, :]
            z_tile = tl.load(z_tile_ptrs, mask=mask_j[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

            z_tile = (z_tile - mean[:, None]) * rstd[:, None]
            z_tile = tl.where(mask_k[None, :], z_tile, 0.0)

            if ELEMENTWISE_AFFINE:
                w_ln_tile = tl.load(w_ln_ptr + tile_k, mask=mask_k, other=0.0).to(tl.float32)
                if RMS_NORM:
                    z_tile = z_tile * w_ln_tile
                else:
                    b_ln_tile = tl.load(b_ln_ptr + tile_k, mask=mask_k, other=0.0).to(tl.float32)
                    z_tile = z_tile * w_ln_tile + b_ln_tile

            w_proj_ptrs = w_proj_ptr + offs_h[None, :] * DIM_D + tile_k[:, None]
            w_tile = tl.load(w_proj_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

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


@triton.jit(
    do_not_specialize=[
        "I",
        "J",
        "J_padded",
        "num_j_tiles",
        "num_tiles",
        "out_stride_b",
        "out_stride_h",
        "out_stride_i",
    ]
)
def _fused_ln_proj_moveaxis_pad_streaming_kernel(
    z_ptr,  # input:  [B, I, J, D] contiguous, bf16 or fp16
    w_ln_ptr,  # LN weight: [D]
    b_ln_ptr,  # LN bias:   [D]
    w_proj_ptr,  # projection weight: [H, D]
    out_ptr,  # output: [B, H, I, J_padded] contiguous
    I,
    J,
    J_padded,
    num_j_tiles,  # ceil_div(J_padded, TILE_J)
    num_tiles,  # B * I * num_j_tiles
    out_stride_b,
    out_stride_h,
    out_stride_i,
    TILE_J: tl.constexpr,
    CHUNK: tl.constexpr,  # divides DIM_D
    DIM_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,  # next power of two >= NUM_HEADS, at least 16
    EPS: tl.constexpr,
    RMS_NORM: tl.constexpr,
):
    """Persistent LayerNorm/RMSNorm + Linear + moveaxis + pad over ``[TILE_J, D]`` tiles.

    With ``W' = gamma * W`` the norm folds into the projection,
    ``out = rstd * (z @ W'^T - mean * sum_d W') + W @ beta``, so each tile's ``CHUNK``-wide
    slices of ``z`` go straight into the dot. ``W'`` is split into high and low halves in the
    input dtype, which keeps TF32-level accuracy. Slice statistics merge with Chan's update.
    ``z`` must be contiguous: row addresses come from the constexpr ``DIM_D``, which proves the
    alignment vectorized loads need without specializing the CUBIN on runtime strides.
    """
    offs_c = tl.arange(0, CHUNK)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS

    # Per program: column sums of the W' the dots use, and the folded LN bias.
    col_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    shift = tl.zeros([BLOCK_H], dtype=tl.float32)
    if not RMS_NORM:
        for k in tl.static_range(DIM_D // CHUNK):
            d_k = k * CHUNK + offs_c
            w_k = tl.load(w_proj_ptr + offs_h[None, :] * DIM_D + d_k[:, None], mask=mask_h[None, :], other=0.0)
            w_k = w_k.to(tl.float32)
            scaled_k = w_k * tl.load(w_ln_ptr + d_k).to(tl.float32)[:, None]
            hi_k = scaled_k.to(z_ptr.dtype.element_ty).to(tl.float32)
            col_sum += tl.sum(hi_k + (scaled_k - hi_k).to(z_ptr.dtype.element_ty).to(tl.float32), axis=0)
            shift += tl.sum(w_k * tl.load(b_ln_ptr + d_k).to(tl.float32)[:, None], axis=0)

    for tile in tl.range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        row = (tile // num_j_tiles).to(tl.int64)
        batch_idx = row // I
        pid_i = row % I
        offs_j = (tile % num_j_tiles) * TILE_J + tl.arange(0, TILE_J)
        mask_j = offs_j < J
        z_rows = z_ptr + (row * J + offs_j[:, None]) * DIM_D

        acc = tl.zeros([TILE_J, BLOCK_H], dtype=tl.float32)
        mean = tl.zeros([TILE_J], dtype=tl.float32)
        m2 = tl.zeros([TILE_J], dtype=tl.float32)
        for c in tl.static_range(DIM_D // CHUNK):
            d_c = c * CHUNK + offs_c
            z = tl.load(z_rows + d_c[None, :], mask=mask_j[:, None], other=0.0)
            zf = z.to(tl.float32)
            if RMS_NORM:
                m2 += tl.sum(zf * zf, axis=1)
            else:
                chunk_mean = tl.sum(zf, axis=1) / CHUNK
                dev = zf - chunk_mean[:, None]
                delta = chunk_mean - mean
                mean += delta / (c + 1)
                m2 += tl.sum(dev * dev, axis=1) + delta * delta * (CHUNK * c / (c + 1))
            w_c = tl.load(w_proj_ptr + offs_h[None, :] * DIM_D + d_c[:, None], mask=mask_h[None, :], other=0.0)
            scaled = w_c.to(tl.float32) * tl.load(w_ln_ptr + d_c).to(tl.float32)[:, None]
            hi = scaled.to(z.dtype)
            acc = tl.dot(z, hi, acc)
            acc = tl.dot(z, (scaled - hi.to(tl.float32)).to(z.dtype), acc)

        rstd = tl.rsqrt(m2 / DIM_D + EPS)
        if RMS_NORM:
            out = rstd[:, None] * acc
        else:
            out = rstd[:, None] * (acc - mean[:, None] * col_sum[None, :]) + shift[None, :]
        out = tl.where(mask_j[:, None], out, 0.0)
        out_ptrs = (
            out_ptr + batch_idx * out_stride_b + pid_i * out_stride_i + offs_h[None, :] * out_stride_h + offs_j[:, None]
        )
        tl.store(out_ptrs, out, mask=(offs_j[:, None] < J_padded) & mask_h[None, :])


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

_TILE_J_DEFAULT = 32

_STREAMING_CHUNK = 64
_STREAMING_NUM_STAGES = 1
#: CTAs per SM the kernel needs to keep enough loads in flight. A variant whose natural register
#: count admits fewer (the bf16 LayerNorm build at D=384, TILE_J=128: 145 regs, 1.5x slower) is
#: rebuilt under the matching ``maxnreg`` cap.
_STREAMING_MIN_CTAS_PER_SM = 2
_STREAMING_MAX_HEADS = 64
_STREAMING_DTYPES = (torch.bfloat16, torch.float16)


class _StreamingTuning(NamedTuple):
    """Streaming-kernel launch constants for one SM version."""

    #: ``(TILE_J, num_warps)`` variants, all compiled up front; each call takes the one that
    #: pads J least, the wider on a tie.
    tiles: tuple[tuple[int, int], ...]
    #: Persistent programs per SM.
    programs_per_sm: int
    #: Width of the ``z`` slices each step loads; a D it does not divide takes ``_STREAMING_CHUNK``.
    chunk: int = _STREAMING_CHUNK


#: SM versions without an entry in :data:`_STREAMING_TUNING`.
_STREAMING_DEFAULT_TUNING = _StreamingTuning(tiles=((64, 4),), programs_per_sm=4)
#: Constants are per SM, so one entry serves every SM count of that version.
_STREAMING_TUNING: dict[int, _StreamingTuning] = {
    # A100 (108 SMs), D in {128, 256, 384}, N up to 2048, B up to 32: the 4-warp 128-row tile is
    # 1.1-1.4x the default at every shape, even where it pads J further; within 1.05x of the swept best.
    80: _StreamingTuning(tiles=((128, 4),), programs_per_sm=4),
    # RTX A6000 (84 SMs), D in {128, 256, 384}, 4-16 heads, N up to 2048, B up to 32: the 4-warp 128-row
    # tile is never slower than the default and up to 1.26x faster (12 heads at D=256 and 384); within 1.04x
    # of the best timed arm at every shape. Two programs per SM time the same.
    86: _StreamingTuning(tiles=((128, 4),), programs_per_sm=4),
    # L40S (142 SMs), D in {128, 256, 384}, 4-16 heads, N up to 2048, B up to 32: 128-wide chunks time
    # 1.05x the 64-wide ones, and with them the 4-warp 32-row tile is 1.07x the default (up to 1.12x; at
    # most 4% slower below 40k rows) and within 1.06x of the best timed arm at every shape.
    89: _StreamingTuning(tiles=((32, 4),), programs_per_sm=16, chunk=128),
    # H200 (132 SMs), D in {256, 384}, N up to 2048, B up to 32: the 128-row tile is 4-7% faster
    # unless it pads J further; every pick within 1.06x of the swept best.
    90: _StreamingTuning(tiles=((64, 4), (128, 8)), programs_per_sm=4),
}
#: Device index -> ``(sm_version, sm_count)``.
_DEVICE_INFO: dict[int, tuple[int, int]] = {}


def _streaming_supported(D: int, H: int) -> bool:
    return D % _STREAMING_CHUNK == 0 and H <= _STREAMING_MAX_HEADS


def _streaming_block_h(H: int) -> int:
    return max(16, triton.next_power_of_2(H))


def _device_info(device: torch.device) -> tuple[int, int]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _DEVICE_INFO:
        props = torch.cuda.get_device_properties(index)
        _DEVICE_INFO[index] = (props.major * 10 + props.minor, props.multi_processor_count)
    return _DEVICE_INFO[index]


def _streaming_tuning(device: torch.device) -> _StreamingTuning:
    return _STREAMING_TUNING.get(_device_info(device)[0], _STREAMING_DEFAULT_TUNING)


def _streaming_chunk(D: int, tuning: _StreamingTuning) -> int:
    return tuning.chunk if D % tuning.chunk == 0 else _STREAMING_CHUNK


def _streaming_tile(J_padded: int, tiles: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Pick the ``(TILE_J, num_warps)`` variant that pads ``J`` least, the wider on a tie."""
    return min(tiles, key=lambda tile: (triton.cdiv(J_padded, tile[0]) * tile[0], -tile[0]))


def _streaming_grid(device: torch.device, num_tiles: int, programs_per_sm: int) -> int:
    return min(num_tiles, _device_info(device)[1] * programs_per_sm)


class FusedLNProjMoveaxisPad(TritonKernelCache):
    """Fused LayerNorm + Linear projection + moveaxis(-1,-3) + pad.

    Compiles once for ``(D, H, tile_j, ...)`` across the common dtypes so each call
    is a bare launch; instances sharing that key reuse one compilation through a
    class-level cache. Low-precision inputs with ``D % 64 == 0`` launch the persistent
    streaming kernel. Callers reach it via ``LNProjMoveaxisPad``.

    Args:
        D: Pair feature dimension, e.g. 64 or 128.
        H: Number of attention heads.
        tile_j: Tile size along J.
        dtype: Compute dtype.
        rms_norm: Use RMSNorm (no mean subtraction or bias) instead of
            LayerNorm.
        eps: Epsilon added to the normalization variance.
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(
        self,
        D: int,
        H: int,
        tile_j: int = _TILE_J_DEFAULT,
        dtype: torch.dtype = torch.bfloat16,
        rms_norm: bool = False,
        eps: float = 1e-5,
    ) -> None:
        self._dim_d = D
        self._num_heads = H
        self._tile_j = tile_j
        self._block_d = triton.next_power_of_2(D)
        self._tile_k = max(16, min(self._block_d, 128))
        self._heads_per_blk = min(triton.next_power_of_2(H), 16)
        self._dtype = dtype
        self._rms_norm = rms_norm
        self._eps = eps
        self._kernels: dict[torch.dtype | tuple[torch.dtype, ...], CachedKernel] = {}
        self._streaming = _streaming_supported(D, H)
        self._streaming_tuning = _STREAMING_DEFAULT_TUNING
        self._streaming_kernels: dict[tuple[torch.dtype | tuple[torch.dtype, ...], int], CachedKernel] = {}

        if torch.cuda.is_available():
            self._ensure_compiled()

    def _ensure_compiled(self):
        if self._kernels:
            return
        base_key = (
            self._dim_d,
            self._num_heads,
            self._tile_j,
            self._tile_k,
            self._heads_per_blk,
            self._rms_norm,
            self._eps,
        )
        dtypes = [self._dtype, torch.bfloat16, torch.float32]
        common_kwargs = {
            "TILE_J": self._tile_j,
            "TILE_K": self._tile_k,
            "BLOCK_D": self._block_d,
            "DIM_D": self._dim_d,
            "NUM_HEADS": self._num_heads,
            "HEADS_PER_BLK": self._heads_per_blk,
            "EPS": self._eps,
            "ELEMENTWISE_AFFINE": True,
            "RMS_NORM": self._rms_norm,
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
                {(dtype, torch.float32, torch.float32, dtype): kernel for dtype, kernel in mixed_result.items()}
            )
            self._kernels = result
            FusedLNProjMoveaxisPad._global_cache[base_key] = result
        if self._streaming:
            self._ensure_streaming_compiled(dtypes)

    def _ensure_streaming_compiled(self, dtypes: list[torch.dtype]) -> None:
        tuning = _streaming_tuning(torch.device("cuda", torch.cuda.current_device()))
        self._streaming_tuning = tuning
        # The key names the dtypes compiled, so a bfloat16 op cannot leave a float16 one without kernels.
        low_precision = [dt for dt in dict.fromkeys(dtypes) if dt in _STREAMING_DTYPES]
        key = ("streaming", tuning, self._dim_d, self._num_heads, self._rms_norm, self._eps, *low_precision)
        cached = FusedLNProjMoveaxisPad._global_cache.get(key)
        if cached is not None:
            self._streaming_kernels = cached
            return
        result = {}
        for tile_j, num_warps in tuning.tiles:
            common_kwargs = {
                "TILE_J": tile_j,
                "CHUNK": _streaming_chunk(self._dim_d, tuning),
                "DIM_D": self._dim_d,
                "NUM_HEADS": self._num_heads,
                "BLOCK_H": _streaming_block_h(self._num_heads),
                "EPS": self._eps,
                "RMS_NORM": self._rms_norm,
                "num_warps": num_warps,
                "num_stages": _STREAMING_NUM_STAGES,
            }
            plain = self._compile_streaming(low_precision, None, common_kwargs)
            mixed = self._compile_streaming(low_precision, torch.float32, common_kwargs)
            result.update({(dtype, tile_j): kernel for dtype, kernel in plain.items()})
            result.update(
                {((dtype, torch.float32, torch.float32, dtype), tile_j): kernel for dtype, kernel in mixed.items()}
            )
        self._streaming_kernels = result
        FusedLNProjMoveaxisPad._global_cache[key] = result

    def _compile_streaming(
        self, dtypes: list[torch.dtype], ln_dtype: torch.dtype | None, kwargs: dict
    ) -> dict[torch.dtype, CachedKernel]:
        def make_dummy_args(dtype: torch.dtype) -> tuple:
            return self._make_streaming_dummy_args(dtype, ln_dtype=ln_dtype, tile_j=kwargs["TILE_J"])

        kernels = self.compile_for_dtypes(
            _fused_ln_proj_moveaxis_pad_streaming_kernel,
            dtypes=dtypes,
            make_dummy_args=make_dummy_args,
            grid=(1,),
            **kwargs,
        )
        registers = torch.cuda.get_device_properties(torch.cuda.current_device()).regs_per_multiprocessor
        budget = registers // (_STREAMING_MIN_CTAS_PER_SM * kwargs["num_warps"] * 32)
        heavy = [dtype for dtype, kernel in kernels.items() if getattr(kernel.compiled, "n_regs", 0) > budget]
        if heavy:
            kernels.update(
                self.compile_for_dtypes(
                    _fused_ln_proj_moveaxis_pad_streaming_kernel,
                    dtypes=heavy,
                    make_dummy_args=make_dummy_args,
                    grid=(1,),
                    maxnreg=budget,
                    **kwargs,
                )
            )
        return kernels

    def streams(self, dtype: torch.dtype) -> bool:
        """Whether inputs of ``dtype`` take the streaming kernel."""
        return self._streaming and dtype in _STREAMING_DTYPES

    def _make_streaming_dummy_args(
        self, dtype: torch.dtype, ln_dtype: torch.dtype | None = None, tile_j: int = 64
    ) -> tuple:
        D, H, tj = self._dim_d, self._num_heads, tile_j
        ln_dtype = ln_dtype or dtype
        z = torch.empty(1, 2, tj, D, dtype=dtype, device="cuda")
        out = torch.empty(1, H, 2, tj, dtype=dtype, device="cuda")
        return (
            z,
            torch.empty(D, dtype=ln_dtype, device="cuda"),
            torch.empty(D, dtype=ln_dtype, device="cuda"),
            torch.empty(H, D, dtype=dtype, device="cuda"),
            out,
            2,
            tj,
            tj,
            1,
            2,
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )

    def _launch_streaming(
        self,
        kernel: CachedKernel,
        tile_j: int,
        num_warps: int,
        z3: torch.Tensor,
        w_ln: torch.Tensor,
        b_ln: torch.Tensor,
        w_proj: torch.Tensor,
        out: torch.Tensor,
        J: int,
        J_padded: int,
    ) -> None:
        B, I = z3.shape[0], z3.shape[1]
        num_j_tiles = triton.cdiv(J_padded, tile_j)
        num_tiles = B * I * num_j_tiles
        if num_tiles == 0:
            return
        grid = _streaming_grid(z3.device, num_tiles, self._streaming_tuning.programs_per_sm)
        args = (
            z3,
            w_ln,
            b_ln,
            w_proj,
            out,
            I,
            J,
            J_padded,
            num_j_tiles,
            num_tiles,
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
        # The driver's scalar slots are i32 (compiled from small dummy values); the JIT launch
        # re-types scalars from their runtime values.
        if kernel.driver is not None and all(isinstance(value, torch.Tensor) or value < 2**31 for value in args):
            values = tuple(value.data_ptr() if isinstance(value, torch.Tensor) else value for value in args)
            kernel.driver.launch_with(values, grid)
            return
        _fused_ln_proj_moveaxis_pad_streaming_kernel[(grid,)](
            *args,
            TILE_J=tile_j,
            CHUNK=_streaming_chunk(self._dim_d, self._streaming_tuning),
            DIM_D=self._dim_d,
            NUM_HEADS=self._num_heads,
            BLOCK_H=_streaming_block_h(self._num_heads),
            EPS=self._eps,
            RMS_NORM=self._rms_norm,
            num_warps=num_warps,
            num_stages=_STREAMING_NUM_STAGES,
        )

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
        b_ln: torch.Tensor | None,
        w_proj: torch.Tensor,
        multiple: int = 8,
    ) -> torch.Tensor:
        """Fused LN/RMS + projection + moveaxis + pad.

        Args:
            z: input pair tensor [*, I, J, D] (contiguous)
            w_ln: LayerNorm/RMSNorm weight [D]
            b_ln: LayerNorm bias [D]; ``None`` for RMSNorm (not read).
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

        # RMSNorm never reads b_ln; pass w_ln as a valid placeholder pointer.
        # Resolve it before the signature so the mixed-dtype lookup below does
        # not dereference a missing bias.
        b_ln_ptr_src = b_ln if b_ln is not None else w_ln
        signature = (z.dtype, w_ln.dtype, b_ln_ptr_src.dtype, w_proj.dtype)
        kernel_key = z.dtype if len(set(signature)) == 1 else signature
        tile_j, num_warps = _streaming_tile(J_padded, self._streaming_tuning.tiles)
        streaming = self._streaming_kernels.get((kernel_key, tile_j)) if self.streams(z.dtype) else None
        kernel = self._kernels.get(kernel_key)

        if streaming is not None:
            self._launch_streaming(streaming, tile_j, num_warps, z3, w_ln, b_ln_ptr_src, w_proj, out, J, J_padded)
        elif kernel is not None and kernel.driver is not None:
            values = (
                z3.data_ptr(),
                w_ln.data_ptr(),
                b_ln_ptr_src.data_ptr(),
                w_proj.data_ptr(),
                out.data_ptr(),
                J,
                J_padded,
                z3.stride(0),
                z3.stride(1),
                z3.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
            )
            kernel.driver.launch_with(values, grid_j, I, num_head_blks * B)
        else:
            _fused_ln_proj_moveaxis_pad_kernel[(grid_j, I, num_head_blks * B)](
                z3,
                w_ln,
                b_ln_ptr_src,
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
                BLOCK_D=self._block_d,
                DIM_D=self._dim_d,
                NUM_HEADS=self._num_heads,
                HEADS_PER_BLK=self._heads_per_blk,
                EPS=self._eps,
                ELEMENTWISE_AFFINE=True,
                RMS_NORM=self._rms_norm,
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
    block_d = triton.next_power_of_2(D)
    tile_k = max(16, min(block_d, 128))
    heads_per_blk = min(triton.next_power_of_2(H), 16)
    tile_j = _TILE_J_DEFAULT

    out = torch.empty(B, H, I, J_padded, device=z.device, dtype=z.dtype)
    out_shape = list(lead) + [H, I, J_padded] if lead else [H, I, J_padded]

    if z3.dtype in _STREAMING_DTYPES and _streaming_supported(D, H):
        tuning = _streaming_tuning(z3.device)
        stream_tile_j, stream_warps = _streaming_tile(J_padded, tuning.tiles)
        num_j_tiles = triton.cdiv(J_padded, stream_tile_j)
        num_tiles = B * I * num_j_tiles
        if num_tiles:
            _fused_ln_proj_moveaxis_pad_streaming_kernel[
                (_streaming_grid(z3.device, num_tiles, tuning.programs_per_sm),)
            ](
                z3,
                w_ln,
                b_ln,
                w_proj,
                out,
                I,
                J,
                J_padded,
                num_j_tiles,
                num_tiles,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                TILE_J=stream_tile_j,
                CHUNK=_streaming_chunk(D, tuning),
                DIM_D=D,
                NUM_HEADS=H,
                BLOCK_H=_streaming_block_h(H),
                EPS=eps,
                RMS_NORM=False,
                num_warps=stream_warps,
                num_stages=_STREAMING_NUM_STAGES,
            )
        return out.view(out_shape)

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
        BLOCK_D=block_d,
        DIM_D=D,
        NUM_HEADS=H,
        HEADS_PER_BLK=heads_per_blk,
        EPS=eps,
        ELEMENTWISE_AFFINE=True,
        RMS_NORM=False,
    )

    return out.view(out_shape)
