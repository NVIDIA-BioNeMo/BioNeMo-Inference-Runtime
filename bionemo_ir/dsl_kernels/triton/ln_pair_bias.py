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


# One CUBIN serves every shape: no scalar may specialize on its compile-time value.
@triton.jit(do_not_specialize=["rows", "dim_r", "dim_s", "pad_keys", "eps"])
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
    pad_keys,
    eps,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SWAP_IJ: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < rows
    s = offs_m % dim_s
    rest = offs_m // dim_s
    r = rest % dim_r
    b = rest // dim_r
    if SWAP_IJ:
        # Ending node: row (b, j, i) reads x[b, i, j].
        x_rows = (b * dim_s + s) * dim_r + r
    else:
        x_rows = offs_m
    offs_c = tl.arange(0, C)
    x = tl.load(x_ptr + x_rows[:, None] * C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    x_centred = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(x_centred * x_centred, axis=1) / C + eps)
    x_hat = x_centred * rstd[:, None]
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    normed = tl.math.fma(x_hat, ln_w[None, :], ln_b[None, :]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * C + offs_c[None, :], normed, mask=mask_m[:, None])

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    w = tl.load(w_ptr + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None], other=0.0)
    acc = tl.dot(normed, tl.trans(w))
    # bias[b, h, r, s], or bias[b, h, s, r] when TRANSPOSED.
    if TRANSPOSED:
        bias_rows = (b[:, None] * H + offs_h[None, :]) * dim_s + s[:, None]
        keys = r
    else:
        bias_rows = (b[:, None] * H + offs_h[None, :]) * dim_r + r[:, None]
        keys = s
    tl.store(
        bias_ptr + bias_rows * pad_keys + keys[:, None],
        acc.to(bias_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_h[None, :],
    )


@functools.cache
def _has_tensor_core_dot(device_index: int) -> bool:
    """Whether ``tl.dot`` takes bf16 and fp16 on this device (SM80+)."""
    return torch.cuda.get_device_capability(device_index)[0] >= 8


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
        self._device: int | None = None
        self._kernel: CachedKernel | None = None
        if dtype in _DTYPES and C in _WIDTHS and H <= _BLOCK_H and torch.cuda.is_available():
            device = torch.cuda.current_device()
            if _has_tensor_core_dot(device):
                self._device = device
                self._kernel = self._compiled()

    def _compiled(self) -> CachedKernel:
        key = (self._channels, self._heads, self._swap_ij, self._transposed, self._dtype, self._device)
        kernel = LNPairBias._global_cache.get(key)
        if kernel is None:
            kernel = self.compile_for_dtypes(
                _ln_pair_bias_kernel,
                dtypes=[self._dtype],
                make_dummy_args=self._make_dummy_args,
                grid=(1,),
                C=self._channels,
                H=self._heads,
                BLOCK_M=_BLOCK_M,
                BLOCK_H=_BLOCK_H,
                SWAP_IJ=self._swap_ij,
                TRANSPOSED=self._transposed,
                num_warps=_NUM_WARPS,
            )[self._dtype]
            LNPairBias._global_cache[key] = kernel
        return kernel

    def _make_dummy_args(self, dtype: torch.dtype) -> tuple:
        C, H, side = self._channels, self._heads, 4
        x = torch.zeros(1, side, side, C, dtype=dtype, device="cuda")
        affine = torch.zeros(C, dtype=dtype, device="cuda")
        weight = torch.zeros(H, C, dtype=dtype, device="cuda")
        normed = torch.empty_like(x)
        bias = torch.empty(1, H, side, 2 * side, dtype=dtype, device="cuda")
        return (x, affine, affine, weight, normed, bias, side * side, side, side, 2 * side, 1e-5)

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
        grid = triton.cdiv(rows, _BLOCK_M)
        kernel = self._kernel
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
                pad_keys,
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
                pad_keys,
                eps,
                C,
                H,
                _BLOCK_M,
                _BLOCK_H,
                self._swap_ij,
                self._transposed,
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
