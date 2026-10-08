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

"""Fuse inference-only FP32 rotation products."""

from threading import Lock

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache


@triton.jit(do_not_specialize=["A", "B", "rows"])
def _rotation_product(A, B, O, M, rows, RANK: tl.constexpr, MATRIX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    remaining = row
    ao = tl.full((BLOCK,), 0, tl.int64)
    bo = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(RANK - 1, -1, -1):
        size = tl.load(M + axis)
        coordinate = remaining % size
        remaining = remaining // size
        ao += coordinate * tl.load(M + RANK + axis)
        bo += coordinate * tl.load(M + 2 * RANK + 2 + axis)
    a_row = tl.load(M + 2 * RANK)
    a_col = tl.load(M + 2 * RANK + 1)
    b_row = tl.load(M + 3 * RANK + 2)
    if MATRIX:
        b_col = tl.load(M + 3 * RANK + 3)
    for i in tl.static_range(3):
        a0 = tl.load(A + ao + i * a_row, valid, 0)
        a1 = tl.load(A + ao + i * a_row + a_col, valid, 0)
        a2 = tl.load(A + ao + i * a_row + 2 * a_col, valid, 0)
        for j in tl.static_range(3 if MATRIX else 1):
            if MATRIX:
                b0 = tl.load(B + bo + j * b_col, valid, 0)
                b1 = tl.load(B + bo + b_row + j * b_col, valid, 0)
                b2 = tl.load(B + bo + 2 * b_row + j * b_col, valid, 0)
                offset = row * 9 + i * 3 + j
            else:
                b0 = tl.load(B + bo, valid, 0)
                b1 = tl.load(B + bo + b_row, valid, 0)
                b2 = tl.load(B + bo + 2 * b_row, valid, 0)
                offset = row * 3 + i
            tl.store(O + offset, (a0 * b0 + a1 * b1) + a2 * b2, valid)


class _RotationProductKernel(TritonKernelCache):
    def __init__(self, rank: int, matrix: bool) -> None:
        self.kernel = self.compile_for_dtypes(
            _rotation_product,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(9, device="cuda", dtype=dtype),
                torch.empty(9, device="cuda", dtype=dtype),
                torch.empty(9, device="cuda", dtype=dtype),
                torch.empty(3 * rank + 2 + (2 if matrix else 1), device="cuda", dtype=torch.int64),
                2**63 - 1,
            ),
            grid=(0,),
            RANK=rank,
            MATRIX=matrix,
            BLOCK=128,
            enable_fp_fusion=False,
        )[torch.float32]


_KERNELS: dict[tuple[int, int, bool], _RotationProductKernel] = {}
# Keep admitted buffers alive: captured graphs retain their device pointers.
_MAX_LAYOUTS = 256
_LAYOUTS: dict[tuple, torch.Tensor] = {}
_LAYOUT_LOCK = Lock()


def rotation_product(a: torch.Tensor, b: torch.Tensor, *, matrix: bool) -> torch.Tensor | None:
    """Fuse CUDA FP32 inference products; return None otherwise.

    Args:
        a: Rotation matrices shaped [..., 3, 3].
        b: Broadcastable matrices [..., 3, 3] or vectors [..., 3].
        matrix: Whether the right operand contains matrices.
    """
    tail = 2 if matrix else 1
    if (
        not a.is_cuda
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or torch.compiler.is_compiling()
        or a.is_neg()
        or b.is_neg()
    ):
        return None
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-tail])
    rows = shape.numel()
    if rows == 0:
        return None
    left = a.expand(*shape, 3, 3)
    right = b.expand(*shape, *((3, 3) if matrix else (3,)))
    key = (a.device.index, len(shape), matrix)
    layout = (a.device.index, tuple(shape), left.stride(), right.stride(), matrix)
    with torch.cuda.device(a.device):
        cached = _KERNELS.get(key)
        metadata = _LAYOUTS.get(layout)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and (cached is None or metadata is None):
            return None
        if metadata is None:
            with _LAYOUT_LOCK:
                metadata = _LAYOUTS.get(layout)
                if metadata is None:
                    if len(_LAYOUTS) >= _MAX_LAYOUTS:
                        return None
                    metadata = torch.tensor(
                        (*shape, *left.stride(), *right.stride()), device=a.device, dtype=torch.int64
                    )
                    _LAYOUTS[layout] = metadata
        if cached is None:
            cached = _RotationProductKernel(len(shape), matrix)
            _KERNELS[key] = cached
        kernel = cached.kernel
        if capturing and isinstance(kernel.compiled, JITFunction):
            return None
        output = torch.empty((*shape, *((3, 3) if matrix else (3,))), device=a.device, dtype=a.dtype)
        grid = triton.cdiv(rows, 128)
        driver = kernel.driver
        if driver is not None:
            driver.launch_with((a.data_ptr(), b.data_ptr(), output.data_ptr(), metadata.data_ptr(), rows), grid)
        else:
            kernel.launch((grid,), a, b, output, metadata, rows, len(shape), matrix, 128)
    return output
