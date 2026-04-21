# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel: fused moveaxis(-1, -3) + zero-pad J to multiple of 8.

Problem
-------
After proj_z (a Linear over the pairwise tensor), we have:

    pair_bias : [B, I, J, H]   (contiguous, H is the inner/fastest dim)

The custom attention kernel wants:

    pair_bias : [B, H, I, J_padded]   (contiguous)

where J_padded = ceil(J / 8) * 8. or J_padded = J if multiple < 0.

The naive PyTorch sequence:
    pair_bias = torch.moveaxis(pair_bias, -1, -3)   # non-contiguous view
    pair_bias = F.pad(pair_bias, (0, J_padded - J)) # still non-contiguous
    pair_bias = pair_bias.contiguous()              # copy

does this in three kernel launches and touches HBM twice (once for the copy,
once for the pad).  The fused kernel does it in a single pass.

Launch strategy
---------------
When ``cuda.bindings`` is available, :class:`MoveaxisPad` launches via
:class:`~tensorrt_bionemo.dsl_kernels.cache_base.DriverLauncher`
(``cuLaunchKernel`` — ~17 µs overhead).  Otherwise it falls back to
Triton's C-level ``.run()`` path (~20 µs overhead).

Tiling strategy
---------------
Grid: (B, I, ceil_div(J_padded, BLOCK_J))

Using a 3D grid makes I and J fully dynamic: each axis is an independent
program dimension so no integer division is needed inside the kernel to
recover (b, i) from a flat pid.

Each program loads a [BLOCK_J, BLOCK_H] tile of inp[b, i, j0:j0+BLOCK_J, 0:H].
Because H is the fastest dimension in the input, this tile is a contiguous block
in memory -> coalesced loads.

The tile is then transposed in registers (tl.trans) and stored as a
[BLOCK_H, BLOCK_J] tile at out[b, 0:H, i, j0:j0+BLOCK_J].
J positions in [J, J_padded) receive 0 (other= in masked load).

Usage
-----
    from tensorrt_bionemo.dsl_kernels.triton.moveaxis_pad import (
        MoveaxisPad, moveaxis_pad)

    # Class-based (preferred for repeated calls with same H):
    op = MoveaxisPad(H=16)
    out = op(pair_bias)

    # Functional:
    pair_bias = moveaxis_pad(pair_bias)   # [B, H, I, J_padded]  -- contiguous
"""

import math

import torch
import triton
import triton.language as tl

from tensorrt_bionemo.dsl_kernels.triton_cache import (CachedKernel,
                                                       TritonKernelCache)

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=[
    'J',
    'J_padded',
    'inp_stride_b',
    'inp_stride_i',
    'inp_stride_j',
    'out_stride_b',
    'out_stride_h',
    'out_stride_i',
])
def _moveaxis_pad_kernel(
    inp_ptr,  # [B, I, J, H]  contiguous
    out_ptr,  # [B, H, I, J_padded]  contiguous (pre-allocated)
    J,
    J_padded,  # I and B are implicit in the 3D grid; only J/J_padded needed for masks
    # input strides  (H=innermost so inp_stride_h=1 — not passed)
    inp_stride_b,  # = I * J * H
    inp_stride_i,  # = J * H
    inp_stride_j,  # = H
    # output strides  (J=innermost so out_stride_j=1 — not passed)
    out_stride_b,  # = H * I * J_padded
    out_stride_h,  # = I * J_padded
    out_stride_i,  # = J_padded
    BLOCK_J: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 3D grid: axis-0 = b, axis-1 = i, axis-2 = J tile
    b_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    pid_j = tl.program_id(2)

    j_off = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # [BLOCK_J]
    h_off = tl.arange(0, BLOCK_H)  # [BLOCK_H]

    inp_base = inp_ptr + b_idx * inp_stride_b + i_idx * inp_stride_i
    inp_ptrs = inp_base + j_off[:, None] * inp_stride_j + h_off[None, :]

    load_mask = j_off[:, None] < J  # [BLOCK_J, BLOCK_H]
    vals = tl.load(inp_ptrs, mask=load_mask, other=0.0)  # [BLOCK_J, BLOCK_H]

    out_base = out_ptr + b_idx * out_stride_b + i_idx * out_stride_i
    out_ptrs = out_base + h_off[:, None] * out_stride_h + j_off[None, :]

    store_mask = j_off[None, :] < J_padded  # [BLOCK_H, BLOCK_J]
    tl.store(out_ptrs, tl.trans(vals), mask=store_mask)


# ---------------------------------------------------------------------------
# Class-based API — pre-compile per (dtype, BLOCK_H)
# ---------------------------------------------------------------------------

_BLOCK_J_DEFAULT = 128


class MoveaxisPad(TritonKernelCache):
    """Pre-compiled fused moveaxis(-1, -3) + pad with ``cuda.bindings`` launch.

    Inherits from :class:`~tensorrt_bionemo.dsl_kernels.triton_cache.TritonKernelCache`
    for the unified compile / save / load interface.

    Pre-compiles the kernel for a known ``H`` (number of heads) and
    common dtypes, enabling fast dispatch via
    :class:`~tensorrt_bionemo.dsl_kernels.cache_base.DriverLauncher`
    with fallback to Triton's C-level ``.run()``.

    A class-level cache ensures that multiple instances with the same
    ``(BLOCK_H, BLOCK_J)`` share compiled kernels.
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(self,
                 H: int,
                 block_j: int = _BLOCK_J_DEFAULT,
                 dtype: torch.dtype = torch.bfloat16):
        self._block_h = triton.next_power_of_2(H)
        self._block_j = block_j
        self._kernels: dict[torch.dtype, CachedKernel] = {}
        self._dtype = dtype

        if torch.cuda.is_available():
            self._ensure_compiled()

    def _ensure_compiled(self):
        if self._kernels:
            return
        cache_key = (self._block_h, self._block_j)
        cached = MoveaxisPad._global_cache.get(cache_key)
        if cached:
            self._kernels = cached
            return

        bj, bh = self._block_j, self._block_h
        result = self.compile_for_dtypes(
            _moveaxis_pad_kernel,
            dtypes=[self._dtype, torch.float32, torch.bfloat16, torch.float16],
            make_dummy_args=lambda dt: self._make_dummy_args(dt),
            grid=(1, 2, 1),
            BLOCK_J=bj,
            BLOCK_H=bh,
        )

        if result:
            self._kernels = result
            MoveaxisPad._global_cache[cache_key] = result

    def _make_dummy_args(self, dtype: torch.dtype) -> tuple:
        bj, bh = self._block_j, self._block_h
        dummy_inp = torch.empty(1, 2, bj, bh, dtype=dtype, device="cuda")
        dummy_out = torch.empty(1, bh, 2, bj, dtype=dtype, device="cuda")
        return (
            dummy_inp,
            dummy_out,
            bj,
            bj,
            dummy_inp.stride(0),
            dummy_inp.stride(1),
            dummy_inp.stride(2),
            dummy_out.stride(0),
            dummy_out.stride(1),
            dummy_out.stride(2),
        )

    def __call__(self, x: torch.Tensor, multiple: int = -1) -> torch.Tensor:
        """Fused moveaxis(-1, -3) + optional zero-pad.

        Args:
            x: contiguous tensor ``[..., I, J, H]``
            multiple: pad J to next multiple (negative = no padding).

        Returns:
            contiguous tensor ``[..., H, I, J_padded]``
        """
        *lead, I, J, H = x.shape
        B = math.prod(lead) if lead else 1
        x3 = x.reshape(B, I, J, H)

        J_padded = J if multiple < 0 else (
            (J + multiple - 1) // multiple) * multiple

        out = torch.empty(B, H, I, J_padded, device=x.device, dtype=x.dtype)

        grid_z = triton.cdiv(J_padded, self._block_j)

        kernel = self._kernels[x.dtype]
        drv = kernel.driver
        if drv is not None:
            drv.params[0].value = x3.data_ptr()
            drv.params[1].value = out.data_ptr()
            drv.params[2].value = J
            drv.params[3].value = J_padded
            drv.params[4].value = x3.stride(0)
            drv.params[5].value = x3.stride(1)
            drv.params[6].value = x3.stride(2)
            drv.params[7].value = out.stride(0)
            drv.params[8].value = out.stride(1)
            drv.params[9].value = out.stride(2)
            drv.launch(B, I, grid_z)
        else:
            kernel.launch(
                (B, I, grid_z),
                x3,
                out,
                J,
                J_padded,
                x3.stride(0),
                x3.stride(1),
                x3.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                self._block_j,
                self._block_h,
            )

        out_shape = list(lead) + [H, I, J_padded] if lead else [H, I, J_padded]
        return out.view(out_shape)


# ---------------------------------------------------------------------------
# Functional API (original interface, standard Triton dispatch)
# ---------------------------------------------------------------------------


def moveaxis_pad(x: torch.Tensor,
                 multiple: int = -1,
                 BLOCK_J: int = _BLOCK_J_DEFAULT) -> torch.Tensor:
    """Fused moveaxis(-1, -3) + optional zero-pad last dim to a multiple.

    Args:
        x: contiguous tensor of shape [..., I, J, H]
        multiple: pad J to the next multiple of this value.
                  If multiple < 0, no padding is applied (J_padded = J).
        BLOCK_J: tile size along J (must be power-of-two, default 128)

    Returns:
        contiguous tensor of shape [..., H, I, J_padded]  where:
          multiple >= 0 -> J_padded = ceil(J / multiple) * multiple
          multiple <  0 -> J_padded = J  (no padding)
    """
    assert x.is_contiguous(), "input must be contiguous"
    *lead, I, J, H = x.shape
    B = math.prod(lead) if lead else 1
    x3 = x.reshape(B, I, J, H)

    J_padded = J if multiple < 0 else (
        (J + multiple - 1) // multiple) * multiple
    BLOCK_H = triton.next_power_of_2(H)

    out = torch.empty(B, H, I, J_padded, device=x.device, dtype=x.dtype)

    grid = (B, I, triton.cdiv(J_padded, BLOCK_J))
    _moveaxis_pad_kernel[grid](
        x3,
        out,
        J,
        J_padded,
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_J=BLOCK_J,
        BLOCK_H=BLOCK_H,
    )

    out_shape = list(lead) + [H, I, J_padded] if lead else [H, I, J_padded]
    return out.view(out_shape)
