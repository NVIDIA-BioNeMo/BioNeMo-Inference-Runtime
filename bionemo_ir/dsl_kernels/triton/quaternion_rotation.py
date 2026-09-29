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

# Quaternion formulas derive from PyTorch3D.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause: see LICENSES/BSD-3-Clause.txt.

"""Fuse inference quaternion conversion in float32."""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache


@triton.jit(do_not_specialize=["rows"])
def _quaternion_matrix(Q, O, rows, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    r = tl.load(Q + row * 4, valid, 0)
    i = tl.load(Q + row * 4 + 1, valid, 0)
    j = tl.load(Q + row * 4 + 2, valid, 0)
    k = tl.load(Q + row * 4 + 3, valid, 0)
    two_s = tl.div_rn(2.0, (r * r + j * j) + (i * i + k * k))
    tl.store(O + row * 9, 1 - two_s * (j * j + k * k), valid)
    tl.store(O + row * 9 + 1, two_s * (i * j - k * r), valid)
    tl.store(O + row * 9 + 2, two_s * (i * k + j * r), valid)
    tl.store(O + row * 9 + 3, two_s * (i * j + k * r), valid)
    tl.store(O + row * 9 + 4, 1 - two_s * (i * i + k * k), valid)
    tl.store(O + row * 9 + 5, two_s * (j * k - i * r), valid)
    tl.store(O + row * 9 + 6, two_s * (i * k - j * r), valid)
    tl.store(O + row * 9 + 7, two_s * (j * k + i * r), valid)
    tl.store(O + row * 9 + 8, 1 - two_s * (i * i + j * j), valid)


class _QuaternionRotationKernel(TritonKernelCache):
    def __init__(self) -> None:
        self.kernel = self.compile_for_dtypes(
            _quaternion_matrix,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(4, device="cuda", dtype=dtype),
                torch.empty(9, device="cuda", dtype=dtype),
                1,
            ),
            grid=(0,),
            BLOCK=128,
            enable_fp_fusion=False,
        )[torch.float32]


@cache
def _quaternion_rotation_kernel(device: int) -> _QuaternionRotationKernel:
    with torch.cuda.device(device):
        return _QuaternionRotationKernel()


def quaternion_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert contiguous, aligned CUDA FP32 [..., 4] quaternions."""
    output = torch.empty((*quaternions.shape[:-1], 3, 3), device=quaternions.device, dtype=quaternions.dtype)
    rows = quaternions.numel() // 4
    if rows:
        with torch.cuda.device(quaternions.device):
            kernel = _quaternion_rotation_kernel(quaternions.device.index).kernel
            grid = triton.cdiv(rows, 128)
            driver = kernel.driver
            if driver is not None:
                driver.params[0].value = quaternions.data_ptr()
                driver.params[1].value = output.data_ptr()
                driver.params[2].value = rows
                driver.launch(grid)
            else:
                kernel.launch((grid,), quaternions, output, rows, 128)
    return output
