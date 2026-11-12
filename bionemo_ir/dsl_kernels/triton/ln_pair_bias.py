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
"""Fused LayerNorm and triangle-bias projection for triangle-attention nodes.

One pass over the pair rows writes ``normed = LayerNorm(x)``, in ``(b, j, i)`` order for the ending node, and
``bias = normed @ weight.T`` with fp32 accumulation, in the attention's ``[B, H, R, S_padded]`` layout
(``[B, H, S, R_padded]`` for a transposed bias) with zero padding.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

_BLOCK_M = 32
_NUM_WARPS = 4
# tl.dot needs at least 16 columns; heads past H load as zeros.
_BLOCK_H = 16
_WIDTHS = (16, 32, 64, 128, 256)
_DTYPES = (torch.bfloat16, torch.float16)
# Registers per thread that let _PROGRAMS_PER_SM programs of _NUM_WARPS warps
# reside on an SM. Without the cap, C = 256 takes 136 and only three fit; with
# four, it runs about 1.15x faster on B200, with no spills.
_MAX_REGISTERS = 128
# Resident programs per SM. Each loops over row tiles and prefetches the next
# tile's rows while it normalizes the current one, which a one-tile program
# cannot do; the plain kernel reached about a third of the HBM rate.
_PROGRAMS_PER_SM = 4
# The ending node reads pair rows transposed. Its tiles cover _TILE_COLUMNS
# output rows by _BLOCK_M // _TILE_COLUMNS columns, and consecutive tiles step
# along the output rows, so resident programs share a few input rows instead
# of each touching a different stretch of the whole pair. On B200 the
# row-major order lost its whole gain at some sizes, e.g. 1736 tokens.
_TILE_COLUMNS = 4
# Software-pipeline depth of the tile loop: rows of 2 tiles in flight for the
# starting node, 3 for the ending node's shorter contiguous reads.
_STAGES = 2
_STAGES_SWAPPED = 3
# Key padding, in elements, that lets bias rows start 16-byte aligned.
_KEY_ALIGN = 8


# One CUBIN serves every shape: no scalar may specialize on its compile-time value.
@triton.jit(do_not_specialize=["rows", "dim_r", "dim_s", "key_groups", "eps"])
def _ln_pair_bias_kernel(
    x_ptr,
    ln_w_ptr,
    ln_b_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    rows,
    dim_r,
    dim_s,
    key_groups,
    eps,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SWAP_IJ: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    TILE_COLUMNS: tl.constexpr,
    STAGES: tl.constexpr,
    KEY_ALIGN: tl.constexpr,
):
    # The bias rows hold key_groups * KEY_ALIGN keys. A KEY_ALIGN of 8 tells the
    # compiler the rows are 16-byte aligned, so bias stores vectorize.
    pad_keys = key_groups * KEY_ALIGN
    offs_c = tl.arange(0, C)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    w = tl.load(w_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None], other=0.0)
    # A tile covers TILE_COLUMNS output rows r by TILE_S columns s. With one
    # row, its columns are provably contiguous whatever the shape, which lets
    # the bias stores vectorize.
    TILE_S: tl.constexpr = BLOCK_M // TILE_COLUMNS
    tiles_r = tl.cdiv(dim_r, TILE_COLUMNS)
    tiles_s = tl.cdiv(dim_s, TILE_S)
    tiles = tiles_r * tiles_s * (rows // (dim_r * dim_s))
    local = tl.arange(0, BLOCK_M)
    for tile in tl.range(tl.program_id(0), tiles, tl.num_programs(0), num_stages=STAGES):
        index = tile.to(tl.int64)
        if SWAP_IJ:
            # Consecutive tiles step along r, so resident programs read a few
            # rows of x[b, i, :] rather than the whole pair.
            tile_r = index % tiles_r
            tile_s = index // tiles_r % tiles_s
        else:
            tile_s = index % tiles_s
            tile_r = index // tiles_s % tiles_r
        b = index // (tiles_r * tiles_s)
        r = tile_r * TILE_COLUMNS + local % TILE_COLUMNS
        s = tile_s * TILE_S + local // TILE_COLUMNS
        mask_m = (r < dim_r) & (s < dim_s)
        offs_m = (b * dim_r + r) * dim_s + s
        if SWAP_IJ:
            # Ending node: row (b, j, i) reads x[b, i, j].
            x_rows = (b * dim_s + s) * dim_r + r
        else:
            x_rows = offs_m
        x = tl.load(x_ptr + x_rows[:, None] * C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        x_centred = x - mean[:, None]
        rstd = tl.rsqrt(tl.sum(x_centred * x_centred, axis=1) / C + eps)
        x_hat = x_centred * rstd[:, None]
        normed = tl.math.fma(x_hat, ln_w[None, :], ln_b[None, :]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs_m[:, None] * C + offs_c[None, :], normed, mask=mask_m[:, None])

        acc = tl.dot(normed, tl.trans(w))
        # bias[b, h, r, s], or bias[b, h, s, r] when TRANSPOSED.
        if TRANSPOSED:
            bias_rows = (b[:, None] * H + offs_h[None, :]) * dim_s + s[:, None]
            keys, key_count, queries_valid = r, dim_r, s < dim_s
        else:
            bias_rows = (b[:, None] * H + offs_h[None, :]) * dim_r + r[:, None]
            keys, key_count, queries_valid = s, dim_s, r < dim_r
        if KEY_ALIGN > 1:
            # Write zeros to the padded keys rather than masking them, so the
            # mask is uniform over each aligned group of keys.
            acc = tl.where((keys < key_count)[:, None], acc, 0.0)
            mask = (queries_valid & (keys < pad_keys))[:, None] & mask_h[None, :]
        else:
            mask = mask_m[:, None] & mask_h[None, :]
        tl.store(bias_ptr + bias_rows * pad_keys + keys[:, None], acc.to(bias_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _has_tensor_core_dot(device_index: int) -> bool:
    """Whether ``tl.dot`` takes bf16 and fp16 on this device (SM80+)."""
    return torch.cuda.get_device_capability(device_index)[0] >= 8


def _resident_programs(device_index: int, kernel: CachedKernel) -> int:
    """Programs of the persistent grid: as many per SM as the kernel's registers and shared memory let reside.

    A grid larger than what resides runs its last programs as a tail wave, which costs more than the
    pipelining gains: at ``C = 256`` the kernel holds 136 registers a thread, so three programs fit an
    SM, not four.
    """
    props = torch.cuda.get_device_properties(device_index)
    threads = _NUM_WARPS * props.warp_size
    per_sm = _PROGRAMS_PER_SM
    registers = getattr(kernel.compiled, "n_regs", 0)
    if registers:
        # Registers are allocated per warp in 256-register units.
        per_warp = -(-registers * props.warp_size // 256) * 256
        per_sm = min(per_sm, props.regs_per_multiprocessor // (per_warp * _NUM_WARPS))
    shared = getattr(getattr(kernel.compiled, "metadata", None), "shared", 0)
    if shared:
        # The driver reserves 1 KB of shared memory per resident block.
        per_sm = min(per_sm, props.shared_memory_per_multiprocessor // (shared + 1024))
    per_sm = min(per_sm, props.max_threads_per_multi_processor // threads)
    return max(1, per_sm) * props.multi_processor_count


class LNPairBias(TritonKernelCache):
    """The kernel compiled once per ``(C, H, swap_ij, transposed, dtype)`` and device, launched through
    :class:`~bionemo_ir.dsl_kernels.cache_base.DriverLauncher`. Without a kernel for the configuration (fp32,
    widths outside 16-256, more than 16 heads, pre-SM80) calls return ``None``.

    Args:
        C: Pair width.
        H: Bias heads.
        swap_ij: Rows in ``(b, j, i)`` order, for the ending node.
        transposed: Bias from the transposed pair, for OpenFold-3's ending node.
        dtype: Pair dtype.
    """

    _global_cache: dict[tuple, CachedKernel] = {}

    def __init__(
        self,
        C: int,
        H: int,
        *,
        swap_ij: bool = False,
        transposed: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._channels = C
        self._heads = H
        self._swap_ij = swap_ij
        self._transposed = transposed
        self._dtype = dtype
        self._tile_columns = _TILE_COLUMNS if swap_ij else 1
        self._device: int | None = None
        self._kernel: CachedKernel | None = None
        # Grid size per kernel, by its key alignment.
        self._programs: dict[int, int] = {}
        # Serves calls whose padded key count is a multiple of _KEY_ALIGN.
        self._aligned_kernel: CachedKernel | None = None
        if dtype in _DTYPES and C in _WIDTHS and H <= _BLOCK_H and torch.cuda.is_available():
            device = torch.cuda.current_device()
            if _has_tensor_core_dot(device):
                self._device = device
                self._kernel = self._compiled(1)
                # Only the starting node's tiles hold contiguous keys, which aligned bias rows let it store
                # as vectors; elsewhere the variant would only cost registers.
                if not swap_ij and not transposed:
                    self._aligned_kernel = self._compiled(_KEY_ALIGN)
                # Read the registers only after queuing both variants: the read waits for a cold compile,
                # which would otherwise run before the second variant is queued.
                self._programs[1] = _resident_programs(device, self._kernel)
                if self._aligned_kernel is not None:
                    self._programs[_KEY_ALIGN] = _resident_programs(device, self._aligned_kernel)

    def _compiled(self, key_align: int) -> CachedKernel:
        key = (self._channels, self._heads, self._swap_ij, self._transposed, self._dtype, self._device, key_align)
        kernel = LNPairBias._global_cache.get(key)
        if kernel is None:
            kernel = self.compile_for_dtypes(
                _ln_pair_bias_kernel,
                dtypes=[self._dtype],
                make_dummy_args=functools.partial(self._make_dummy_args, key_align=key_align),
                # Compile without launching on these uninitialized dummies.
                grid=(0,),
                C=self._channels,
                H=self._heads,
                BLOCK_M=_BLOCK_M,
                BLOCK_H=_BLOCK_H,
                SWAP_IJ=self._swap_ij,
                TRANSPOSED=self._transposed,
                TILE_COLUMNS=self._tile_columns,
                STAGES=_STAGES_SWAPPED if self._swap_ij else _STAGES,
                KEY_ALIGN=key_align,
                num_warps=_NUM_WARPS,
                maxnreg=_MAX_REGISTERS,
            )[self._dtype]
            LNPairBias._global_cache[key] = kernel
        return kernel

    def _make_dummy_args(self, dtype: torch.dtype, key_align: int = 1) -> tuple:
        C, H, side = self._channels, self._heads, 4
        x = torch.zeros(1, side, side, C, dtype=dtype, device="cuda")
        affine = torch.zeros(C, dtype=dtype, device="cuda")
        weight = torch.zeros(H, C, dtype=dtype, device="cuda")
        normed = torch.empty_like(x)
        bias = torch.empty(1, H, side, 2 * side, dtype=dtype, device="cuda")
        return (x, affine, affine, weight, normed, bias, side * side, side, side, 2 * side // key_align, 1e-5)

    def __call__(
        self,
        x: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        weight: torch.Tensor,
        pad_multiple: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(normed, bias)`` for ``x`` ``[B, I, J, C]``, the bias keys padded to ``pad_multiple``; ``None``
        without a kernel."""
        if self._kernel is None:
            return None
        x = x.contiguous()
        B, I, J, C = x.shape
        H = self._heads
        R, S = (J, I) if self._swap_ij else (I, J)
        queries, keys = (S, R) if self._transposed else (R, S)
        pad_keys = keys if pad_multiple <= 0 else -(-keys // pad_multiple) * pad_multiple
        normed = torch.empty((B, R, S, C), device=x.device, dtype=x.dtype)
        # Padded keys stay zero.
        alloc = torch.zeros if pad_keys > keys else torch.empty
        bias = alloc((B, H, queries, pad_keys), device=x.device, dtype=x.dtype)
        rows = B * R * S
        tiles = B * triton.cdiv(R, self._tile_columns) * triton.cdiv(S, _BLOCK_M // self._tile_columns)
        key_align = _KEY_ALIGN if self._aligned_kernel is not None and pad_keys % _KEY_ALIGN == 0 else 1
        grid = min(tiles, self._programs[key_align])
        kernel = self._aligned_kernel if key_align > 1 else self._kernel
        if kernel.driver is not None:
            values = (
                x.data_ptr(),
                ln_weight.data_ptr(),
                ln_bias.data_ptr(),
                weight.data_ptr(),
                normed.data_ptr(),
                bias.data_ptr(),
                rows,
                R,
                S,
                pad_keys // key_align,
                eps,
            )
            kernel.driver.launch_with(values, grid)
        else:
            kernel.launch(
                (grid,),
                x,
                ln_weight,
                ln_bias,
                weight,
                normed,
                bias,
                rows,
                R,
                S,
                pad_keys // key_align,
                eps,
                C,
                H,
                _BLOCK_M,
                _BLOCK_H,
                self._swap_ij,
                self._transposed,
                self._tile_columns,
                _STAGES_SWAPPED if self._swap_ij else _STAGES,
                key_align,
            )
        return normed, bias


def ln_pair_bias(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    weight: torch.Tensor,
    *,
    swap_ij: bool = False,
    transposed: bool = False,
    pad_multiple: int = -1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One :class:`LNPairBias` call for ``x`` and ``weight``'s shapes."""
    op = LNPairBias(x.shape[-1], weight.shape[0], swap_ij=swap_ij, transposed=transposed, dtype=x.dtype)
    return op(x, ln_weight, ln_bias, eps, weight, pad_multiple=pad_multiple)
