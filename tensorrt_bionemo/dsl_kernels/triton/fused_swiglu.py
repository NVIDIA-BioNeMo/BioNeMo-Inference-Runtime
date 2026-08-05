# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton kernel: fused SwiGLU activation.

Computes one of two patterns in a single kernel launch, selected by
the compile-time flag ``THREE_WAY``:

    2-way:  out = silu(gate) * x            (standard SwiGLU)
    3-way:  out = silu(gate) * x * n        (gated SwiGLU with extra multiplier - Boltz DiT)

The input ``z`` is a contiguous tensor whose last dimension packs the
components end-to-end:

    2-way:  z[..., 0:d] = x,   z[..., d:2d] = gate          → shape [..., 2*d]
    3-way:  z[..., 0:d] = x,   z[..., d:2d] = gate,
            z[..., 2d:3d] = n                                 → shape [..., 3*d]

By reading from the packed buffer directly the kernel avoids the
non-contiguous views that ``z.split(...)`` would produce and eliminates
two extra kernel launches (silu + mul / mul).

Adapted from vLLM's ``_silu_and_mul_kernel`` (Apache-2.0).

Launch strategy
---------------
When ``cuda.bindings`` is available, :class:`FusedSwiGLU` launches via
:class:`~tensorrt_bionemo.dsl_kernels.cache_base.DriverLauncher`
(``cuLaunchKernel`` — ~17 µs overhead).  Otherwise it falls back to
Triton's C-level ``.run()`` path (~20 µs overhead).

Usage
-----
    from tensorrt_bionemo.dsl_kernels.triton.fused_swiglu import FusedSwiGLU

    # Class-based (preferred — pre-compiled, ~17 µs total with cuda.bindings):
    swiglu = FusedSwiGLU(d=768, three_way=False)
    out = swiglu(z)
    swiglu(z, output=buf)                  # with pre-allocated buffer

    # Functional (convenience, ~30 µs overhead):
    from tensorrt_bionemo.dsl_kernels.triton.fused_swiglu import fused_swiglu
    out = fused_swiglu(z, three_way=False)
"""

import torch
import triton
import triton.language as tl

from tensorrt_bionemo.dsl_kernels.triton_cache import TritonKernelCache

# ---------------------------------------------------------------------------
# Triton kernel — d is constexpr for optimal codegen (mask elimination,
# constant-folded pointer arithmetic).
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["out_row_stride", "z_row_stride"])
def _fused_swiglu_kernel(
    out_ptr,
    out_row_stride,
    z_ptr,
    z_row_stride,
    d: tl.constexpr,
    THREE_WAY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)

    z_row = z_ptr + row * z_row_stride
    o_row = out_ptr + row * out_row_stride

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < d

    x = tl.load(z_row + offs, mask=mask)
    gate = tl.load(z_row + offs + d, mask=mask)

    # sigmoid requires fp32; cast silu(gate) back to native dtype so
    # the multiply with x matches PyTorch nn.SiLU precision.
    gate_f32 = gate.to(tl.float32)
    silu_gate = (tl.sigmoid(gate_f32) * gate_f32).to(x.dtype)
    result = silu_gate * x

    if THREE_WAY:
        n = tl.load(z_row + offs + 2 * d, mask=mask)
        result = result * n

    tl.store(o_row + offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Block-size heuristic
# ---------------------------------------------------------------------------


def _pick_block_size(d: int) -> int:
    if d <= 256:
        return 128
    if d <= 1024:
        return 512
    return 1024


# ---------------------------------------------------------------------------
# Class-based API — pre-compile, fast dispatch via cuda.bindings
# ---------------------------------------------------------------------------


class FusedSwiGLU(TritonKernelCache):
    """Fused SwiGLU with pre-compiled kernel and ``cuda.bindings`` launch.

    Inherits from :class:`~tensorrt_bionemo.dsl_kernels.triton_cache.TritonKernelCache`
    for the unified compile / save / load interface.

    At construction the kernel is compiled for ``(d, three_way, block_size)``
    across multiple dtypes.  ``__call__`` dispatches via
    :class:`~tensorrt_bionemo.dsl_kernels.cache_base.DriverLauncher`
    (cuda.bindings ``cuLaunchKernel``) when available, falling back to
    Triton's C-level ``.run()`` (~20 µs vs ~17 µs).

    A class-level cache ensures that multiple instances with the same
    ``(d, three_way)`` share compiled kernels — avoiding redundant
    kernel launches and GPU syncs during model construction.
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(self, d: int, three_way: bool, dtype: torch.dtype = torch.bfloat16):
        self.d = d
        self.three_way = three_way
        self._block_size = _pick_block_size(d)
        self._divisor = 3 if three_way else 2
        self._tiles = triton.cdiv(d, self._block_size)
        self._kernels: dict = {}
        self._dtype = dtype

        if torch.cuda.is_available():
            self._ensure_compiled()

    def _ensure_compiled(self):
        if self._kernels:
            return
        cache_key = (self.d, self.three_way, self._block_size)
        cached = FusedSwiGLU._global_cache.get(cache_key)
        if cached:
            self._kernels = cached
            return

        mul = self._divisor
        d = self.d
        bs = self._block_size
        result = self.compile_for_dtypes(
            _fused_swiglu_kernel,
            dtypes=[self._dtype, torch.float32, torch.bfloat16, torch.float16],
            make_dummy_args=lambda dt: (
                torch.empty(1, d, dtype=dt, device="cuda"),
                d,
                torch.empty(1, mul * d, dtype=dt, device="cuda"),
                mul * d,
            ),
            grid=(1, self._tiles),
            d=d,
            THREE_WAY=self.three_way,
            BLOCK_SIZE=bs,
        )

        if result:
            self._kernels = result
            FusedSwiGLU._global_cache[cache_key] = result

    def __call__(
        self,
        z: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d = self.d
        num_rows = z.numel() // (self._divisor * d)

        z_flat = z.reshape(num_rows, -1)

        if output is not None:
            out_flat = output.reshape(num_rows, d)
        else:
            out_flat = torch.empty((num_rows, d), dtype=z.dtype, device=z.device)

        kernel = self._kernels[z.dtype]
        drv = kernel.driver
        if drv is not None:
            drv.params[0].value = out_flat.data_ptr()
            drv.params[1].value = out_flat.stride(0)
            drv.params[2].value = z_flat.data_ptr()
            drv.params[3].value = z_flat.stride(0)
            drv.launch(num_rows, self._tiles)
        else:
            kernel.launch(
                (num_rows, self._tiles),
                out_flat,
                out_flat.stride(0),
                z_flat,
                z_flat.stride(0),
                d,
                self.three_way,
                self._block_size,
            )

        if output is not None:
            return output
        return out_flat.reshape(z.shape[:-1] + (d,))


# ---------------------------------------------------------------------------
# Functional API (convenience wrapper)
# ---------------------------------------------------------------------------


def fused_swiglu(
    z: torch.Tensor,
    three_way: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused SwiGLU activation on packed input.

    For hot loops prefer :class:`FusedSwiGLU` which pre-compiles and
    caches the kernel for direct cuda.bindings launch.

    Args:
        z: Contiguous tensor ``[..., K*d]`` where K is 2 or 3.
        three_way: If ``True``, compute ``silu(gate) * x * n`` from a
            ``[..., 3*d]`` input.  If ``False``, compute ``silu(gate) * x``
            from a ``[..., 2*d]`` input.
        output: Optional pre-allocated ``[..., d]`` buffer.

    Returns:
        Tensor of shape ``[..., d]``.
    """
    last = z.shape[-1]
    divisor = 3 if three_way else 2
    d = last // divisor
    num_rows = z.numel() // last

    z_flat = z.reshape(num_rows, last)

    if output is not None:
        out_flat = output.reshape(num_rows, d)
    else:
        out_flat = torch.empty((num_rows, d), dtype=z.dtype, device=z.device)

    if num_rows == 0:
        return out_flat.reshape(z.shape[:-1] + (d,))

    block_size = _pick_block_size(d)
    grid = (num_rows, triton.cdiv(d, block_size))

    _fused_swiglu_kernel[grid](
        out_flat,
        out_flat.stride(0),
        z_flat,
        z_flat.stride(0),
        d=d,
        THREE_WAY=three_way,
        BLOCK_SIZE=block_size,
    )

    if output is not None:
        return output
    return out_flat.reshape(z.shape[:-1] + (d,))
