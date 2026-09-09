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

"""Triton kernel: fused moveaxis(-1, -3) + zero-pad J.

``proj_z`` produces pair bias as ``[B, I, J, H]`` but the attention kernels want
``[B, H, I, J_padded]``, where ``J_padded = ceil(J / multiple) * multiple`` (or
``J`` when ``multiple <= 0``). In PyTorch that is ``moveaxis`` then ``pad`` then
``contiguous`` -- three launches and two passes over HBM. This does it in one.

Grid ``(B, I, ceil_div(J_padded, BLOCK_J))`` keeps I and J dynamic without
recovering ``(b, i)`` from a flat pid. Each program loads a ``[BLOCK_J, BLOCK_H]``
tile, which is contiguous because H is the fastest input dimension, transposes it
in registers and stores it transposed. Columns in ``[J, J_padded)`` get zero from
the masked load's ``other=``.

:class:`MoveaxisPad` launches through
:class:`~bionemo_ir.dsl_kernels.cache_base.DriverLauncher` when ``cuda.bindings``
is available and otherwise through Triton's ``.run()``::

    op = MoveaxisPad(H=16)      # preferred: pre-compiled, reused across calls
    out = op(pair_bias, multiple=8)

    out = moveaxis_pad(pair_bias, multiple=8)   # functional, compiles per call
"""

import math

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "J",
        "J_padded",
        "H",
        "inp_stride_b",
        "inp_stride_i",
        "inp_stride_j",
        "out_stride_b",
        "out_stride_h",
        "out_stride_i",
    ]
)
def _moveaxis_pad_kernel(
    inp_ptr,  # [B, I, J, H]  contiguous
    out_ptr,  # [B, H, I, J_padded]  contiguous (pre-allocated)
    J,
    J_padded,  # I and B are implicit in the 3D grid; only J/J_padded needed for masks
    H,  # BLOCK_H rounds H up to a power of two, so the h axis needs a mask
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
    # i64 so the base offsets cannot wrap: inp_stride_b = I * J * H already reaches
    # 3.74e8 at OF3 scale (I=J=4832, H=16) and crosses INT32_MAX once any of
    # (B>5, H>=64, I/J>=8k) holds. Free when the offset already fits.
    b_idx = tl.program_id(0).to(tl.int64)
    i_idx = tl.program_id(1).to(tl.int64)
    pid_j = tl.program_id(2)

    j_off = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # [BLOCK_J]
    h_off = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_off < H  # [BLOCK_H]

    inp_base = inp_ptr + b_idx * inp_stride_b + i_idx * inp_stride_i
    inp_ptrs = inp_base + j_off[:, None] * inp_stride_j + h_off[None, :]

    load_mask = (j_off[:, None] < J) & h_mask[None, :]  # [BLOCK_J, BLOCK_H]
    vals = tl.load(inp_ptrs, mask=load_mask, other=0.0)  # [BLOCK_J, BLOCK_H]

    out_base = out_ptr + b_idx * out_stride_b + i_idx * out_stride_i
    out_ptrs = out_base + h_off[:, None] * out_stride_h + j_off[None, :]

    store_mask = (j_off[None, :] < J_padded) & h_mask[:, None]  # [BLOCK_H, BLOCK_J]
    tl.store(out_ptrs, tl.trans(vals), mask=store_mask)


# ---------------------------------------------------------------------------
# Class-based API — pre-compile per (dtype, BLOCK_H)
# ---------------------------------------------------------------------------

_BLOCK_J_DEFAULT = 128


class MoveaxisPad(TritonKernelCache):
    """Pre-compiled fused moveaxis(-1, -3) + pad with ``cuda.bindings`` launch.

    Compiles once for a known ``H`` (number of heads) across the common dtypes so
    each call is a bare launch. Instances sharing ``(BLOCK_H, BLOCK_J)`` reuse one
    compilation through a class-level cache.
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(self, H: int, block_j: int = _BLOCK_J_DEFAULT, dtype: torch.dtype = torch.bfloat16):
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
            bh,
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
            multiple: pad J to next multiple (non-positive = no padding).

        Returns:
            contiguous tensor ``[..., H, I, J_padded]``

        Raises:
            ValueError: If ``x`` has more heads than the instance was compiled for.
        """
        *lead, I, J, H = x.shape
        if H > self._block_h:
            raise ValueError(
                f"input has {H} heads but this MoveaxisPad was built for at most {self._block_h}; "
                f"construct it with H={H}"
            )
        B = math.prod(lead) if lead else 1
        x3 = x.reshape(B, I, J, H)

        J_padded = J if multiple <= 0 else ((J + multiple - 1) // multiple) * multiple

        out = torch.empty(B, H, I, J_padded, device=x.device, dtype=x.dtype)

        grid_z = triton.cdiv(J_padded, self._block_j)

        kernel = self._kernels[x.dtype]
        drv = kernel.driver
        if drv is not None:
            drv.params[0].value = x3.data_ptr()
            drv.params[1].value = out.data_ptr()
            drv.params[2].value = J
            drv.params[3].value = J_padded
            drv.params[4].value = H
            drv.params[5].value = x3.stride(0)
            drv.params[6].value = x3.stride(1)
            drv.params[7].value = x3.stride(2)
            drv.params[8].value = out.stride(0)
            drv.params[9].value = out.stride(1)
            drv.params[10].value = out.stride(2)
            drv.launch(B, I, grid_z)
        else:
            kernel.launch(
                (B, I, grid_z),
                x3,
                out,
                J,
                J_padded,
                H,
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


def moveaxis_pad(x: torch.Tensor, multiple: int = -1, BLOCK_J: int = _BLOCK_J_DEFAULT) -> torch.Tensor:
    """Fused moveaxis(-1, -3) + optional zero-pad of J.

    Args:
        x: contiguous ``[..., I, J, H]``.
        multiple: pad J up to this multiple; ``<= 0`` leaves J unpadded.
        BLOCK_J: power-of-two tile size along J.

    Returns:
        Contiguous ``[..., H, I, J_padded]``.
    """
    assert x.is_contiguous(), "input must be contiguous"
    *lead, I, J, H = x.shape
    B = math.prod(lead) if lead else 1
    x3 = x.reshape(B, I, J, H)

    J_padded = J if multiple <= 0 else ((J + multiple - 1) // multiple) * multiple
    BLOCK_H = triton.next_power_of_2(H)

    out = torch.empty(B, H, I, J_padded, device=x.device, dtype=x.dtype)

    grid = (B, I, triton.cdiv(J_padded, BLOCK_J))
    _moveaxis_pad_kernel[grid](
        x3,
        out,
        J,
        J_padded,
        H,
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
