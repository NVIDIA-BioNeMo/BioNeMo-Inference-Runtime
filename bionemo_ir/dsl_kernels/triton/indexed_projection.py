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

"""Grouped FP32 projection without per-row weight materialization."""

from dataclasses import dataclass
from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache


@dataclass(frozen=True)
class IndexedRows:
    """Input-dependent row order shared by confidence samples."""

    indices: torch.Tensor
    offsets: torch.Tensor
    max_rows: int


def prepare_indexed_rows(index: torch.Tensor, groups: int) -> IndexedRows | None:
    """Group a CUDA index vector once, outside graph capture.

    Negative indices retain PyTorch's wraparound semantics. Invalid indices
    return to the reference path, which reports the indexing error.
    """
    if not index.is_cuda or index.ndim != 1 or torch.cuda.is_current_stream_capturing():
        return None
    if index.numel() == 0 or index.dtype not in (torch.int32, torch.int64):
        return None
    low, high = torch.aminmax(index)
    if low.item() < -groups or high.item() >= groups:
        return None
    index = index.remainder(groups)
    counts = torch.bincount(index, minlength=groups)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return IndexedRows(index.argsort(), offsets, int(counts.max().item()))


@triton.jit(do_not_specialize=["n_rows"])
def _indexed_projection(
    features,
    weights,
    indices,
    offsets,
    output,
    n_rows: tl.int64,
    channels: tl.constexpr,
    bins: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    group = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)
    start = tl.load(offsets + group)
    end = tl.load(offsets + group + 1)
    if start + tl.program_id(0) * block_m < end:
        rows = start + tl.program_id(0) * block_m + tl.arange(0, block_m)
        atom = tl.load(indices + rows, rows < end, 0)
        k = tl.arange(0, block_k)
        b = tl.arange(0, block_n)
        total = tl.zeros((block_m, block_n), tl.float32)
        for tile in range(tl.cdiv(channels, block_k)):
            c = tile * block_k + k
            x = tl.load(
                features + (batch * n_rows + atom[:, None]) * channels + c[None, :],
                (rows < end)[:, None] & (c < channels)[None, :],
                0,
            )
            w = tl.load(
                weights + (group * channels + c[:, None]) * bins + b[None, :],
                (c < channels)[:, None] & (b < bins)[None, :],
                0,
            )
            total += tl.dot(x, w, input_precision="tf32x3")
        tl.store(
            output + (batch * n_rows + atom[:, None]) * bins + b[None, :],
            total,
            (rows < end)[:, None] & (b < bins)[None, :],
        )


class _IndexedProjection(TritonKernelCache):
    def __init__(self, channels: int, bins: int) -> None:
        self.kernel = self.compile_for_dtypes(
            _indexed_projection,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=torch.int64),
                torch.empty(1, device="cuda", dtype=torch.int64),
                torch.empty(1, device="cuda", dtype=dtype),
                2,
            ),
            grid=(0,),
            channels=channels,
            bins=bins,
            block_m=32,
            block_n=triton.next_power_of_2(bins),
            block_k=32,
            num_warps=4,
        )[torch.float32]


@cache
def _cached_projection(device: int, channels: int, bins: int) -> CachedKernel:
    with torch.cuda.device(device):
        return _IndexedProjection(channels, bins).kernel


def indexed_projection(features: torch.Tensor, weights: torch.Tensor, rows: IndexedRows) -> torch.Tensor:
    """Project contiguous FP32 ``[..., A, C]`` through grouped ``[G, C, O]`` weights.

    The caller supplies validated row groups and disables autograd. FP32
    accumulation uses three TF32 products on Ampere, Hopper and Blackwell.

    Raises:
        ValueError: Features or weights are not contiguous.
    """
    if not features.is_contiguous() or not weights.is_contiguous():
        raise ValueError("Projection features and weights must be contiguous")
    n_rows, channels = features.shape[-2:]
    groups, _, bins = weights.shape
    output = features.new_empty((*features.shape[:-1], bins))
    batches = features.numel() // (n_rows * channels)
    tensors = (features, weights, rows.indices, rows.offsets, output)
    constants = (channels, bins, 32, triton.next_power_of_2(bins), 32)
    grid = (triton.cdiv(rows.max_rows, 32), groups, batches)
    with torch.cuda.device(features.device):
        # Unaligned views invalidate compiled pointer alignment.
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            _indexed_projection[grid](*tensors, n_rows, *constants, num_warps=4)
            return output
        kernel = _cached_projection(features.device.index, channels, bins)
        driver = kernel.driver
        if driver is not None:
            driver.launch_with((*(tensor.data_ptr() for tensor in tensors), n_rows), *grid)
        else:
            kernel.launch(grid, *tensors, n_rows, *constants)
    return output
