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
"""Zero invalid rows without reading or rewriting valid output values."""

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_masked_rows_kernel(
    output,
    mask,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    if INT64:
        block = block.to(tl.int64)
    index = block * BLOCK + tl.arange(0, BLOCK)
    if ROW_STRIDE == 1:
        row, channel = index % M, index // M
    else:
        row, channel = index // N, index % N
    valid = tl.load(mask + row, mask=index < M * N, other=1)
    tl.store(output + row * ROW_STRIDE + channel * CHANNEL_STRIDE, 0, mask=(index < M * N) & (valid == 0))


def zero_masked_rows_(output: torch.Tensor, mask: torch.Tensor) -> None:
    """Set invalid rows of a CUDA matrix to zero in place, including NaNs.

    Args:
        output: Writable ``[rows, channels]`` matrix; either axis may be contiguous.
        mask: Binary mask with one element per row, on the same device.
    """
    if output.ndim != 2 or mask.numel() != output.shape[0] or mask.device != output.device or not output.is_cuda:
        raise ValueError("Expected a CUDA matrix and a same-device mask with one element per row")
    if not output.numel():
        return
    rows, channels = output.shape
    row_stride, channel_stride = output.stride()
    max_offset = (rows - 1) * row_stride + (channels - 1) * channel_stride
    _zero_masked_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        output,
        mask.contiguous(),
        rows,
        channels,
        row_stride,
        channel_stride,
        max(max_offset, output.numel()) >= 2**31,
        1024,
    )
