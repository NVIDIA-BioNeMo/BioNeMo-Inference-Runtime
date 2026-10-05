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

"""Normalize and project only token pairs used by local atom windows."""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache


@triton.jit
def _normalize_project(pair, n_pairs, index, valid, norm_weight, weight, eps):
    channels = tl.arange(0, 128)
    bins = tl.arange(0, 16)
    # Addresses outside the pair tensor read zeros, which project to zero.
    valid &= (index >= 0) & (index < n_pairs)
    x = tl.load(pair + index[:, None] * 128 + channels[None, :], valid[:, None], 0)
    centered = x - (tl.sum(x, 1) / 128)[:, None]
    variance = tl.sum(centered * centered, 1) / 128
    x = centered * tl.rsqrt(variance[:, None] + eps) * tl.load(norm_weight + channels)[None, :]
    w = tl.load(weight + bins[None, :] * 128 + channels[:, None])
    return tl.dot(x, w, input_precision="tf32x3")


@triton.jit
def _store_pairs(value, rows, n_rows, mask, output):
    value *= tl.load(mask + rows, rows < n_rows, 0)[:, None]
    bins = tl.arange(0, 16)
    tl.store(output + rows[:, None] * 16 + bins[None, :], value, (rows < n_rows)[:, None])


@triton.jit(do_not_specialize=["n_pairs", "n_rows", "eps"])
def _project_pairs(
    pair,
    indices,
    norm_weight,
    weight,
    mask,
    output,
    n_pairs: tl.int64,
    n_rows: tl.int64,
    eps,
    block_m: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64) * block_m
    lane = tl.arange(0, block_m)
    rows = start + lane
    index = tl.load(indices + rows, rows < n_rows, 0)
    previous = tl.load(indices + rows - 1, (lane > 0) & (rows < n_rows), 0)
    first = (lane == 0) | (index != previous)
    group = tl.cumsum(first.to(tl.int32), 0) - 1
    # Reuse consecutive pairs within each tile. A 128-key protein window spans
    # about 17 tokens, so 16 groups would send most tiles down the slow path.
    if tl.max(group, 0) < 32:
        counts = tl.histogram(group, 32)
        compact_rows = start + tl.cumsum(counts, 0) - counts
        compact = tl.load(indices + compact_rows, (counts > 0) & (compact_rows < n_rows), 0)
        projected = _normalize_project(pair, n_pairs, compact, counts > 0, norm_weight, weight, eps)
        value = tl.gather(projected, tl.broadcast_to(group[:, None], (block_m, 16)), 0)
        _store_pairs(value, rows, n_rows, mask, output)
    else:
        # Bound shared memory for irregular windows.
        for tile in range(block_m // 32):
            tile_rows = start + tile * 32 + tl.arange(0, 32)
            tile_index = tl.load(indices + tile_rows, tile_rows < n_rows, 0)
            value = _normalize_project(pair, n_pairs, tile_index, tile_rows < n_rows, norm_weight, weight, eps)
            _store_pairs(value, tile_rows, n_rows, mask, output)


class _PairProjections(TritonKernelCache):
    def __init__(self, mask_dtype: torch.dtype) -> None:
        self.project = self.compile_for_dtypes(
            _project_pairs,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=torch.int64),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=mask_dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                2,
                2,
                1e-5,
            ),
            grid=(0,),
            block_m=128,
            num_warps=2,
        )[torch.float32]


@cache
def _cached_projections(device: int, mask_dtype: torch.dtype) -> _PairProjections:
    with torch.cuda.device(device):
        return _PairProjections(mask_dtype)


def project_pair_windows(
    pair: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize, project, and mask indexed pairs in one kernel.

    Capturable in CUDA graphs: addresses are bounded on the device, so the
    call never synchronizes. The caller retains the original path for
    other layouts. Safe to call from concurrent threads.

    Args:
        pair: Contiguous CUDA FP32 ``[B, N, N, 128]`` trunk pair.
        indices: Flattened pair addresses shaped ``[B, K, Q, H]``.
        mask: Contiguous FP32 or bool atom-pair mask shaped like ``indices``.
        norm_weight: Contiguous FP32 LayerNorm scale, ``[128]``; no bias.
        weight: Contiguous FP32 projection weight, ``[16, 128]``.
        eps: LayerNorm variance offset.

    Returns:
        FP32 ``[B, K, Q, H, 16]`` projections, zero where ``mask`` is zero or
        the address lies outside ``pair``.

    Raises:
        ValueError: An input is not contiguous or its shape does not match.
    """
    # Flat addressing requires contiguous inputs.
    if not all(tensor.is_contiguous() for tensor in (pair, mask, norm_weight, weight)):
        raise ValueError("Pair projection inputs must be contiguous")
    if pair.shape[-1] != 128 or norm_weight.shape != (128,) or weight.shape != (16, 128):
        raise ValueError("Pair projection expects 128 channels and 16 bins")
    if mask.numel() != indices.numel():
        raise ValueError("Pair mask must match the addresses")
    output = pair.new_empty((*indices.shape, 16))
    indices = indices.flatten().contiguous()
    if output.numel() == 0:
        return output
    tensors = (pair, indices, norm_weight, weight, mask, output)
    n_pairs, n_rows = pair.numel() // 128, indices.numel()
    grid = (triton.cdiv(n_rows, 128),)
    with torch.cuda.device(pair.device):
        # Unaligned views invalidate compiled pointer alignment.
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            _project_pairs[grid](*tensors, n_pairs, n_rows, eps, 128, num_warps=2)
            return output
        kernel = _cached_projections(pair.device.index, mask.dtype).project
        driver = kernel.driver
        if driver is not None:
            driver.launch_with((*(tensor.data_ptr() for tensor in tensors), n_pairs, n_rows, float(eps)), *grid)
        else:
            kernel.launch(grid, *tensors, n_pairs, n_rows, float(eps), 128)
    return output
