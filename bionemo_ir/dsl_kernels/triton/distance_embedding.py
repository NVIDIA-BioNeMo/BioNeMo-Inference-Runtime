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

import math
from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

_BLOCK_PAIRS = 16


@triton.jit(do_not_specialize=["I", "J"])
def _distance_embedding(
    X: tl.tensor,
    Y: tl.tensor,
    L: tl.tensor,
    U: tl.tensor,
    W: tl.tensor,
    B: tl.tensor,
    R: tl.tensor,
    O: tl.tensor,
    I: tl.int64,
    J: tl.int64,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
    EUCLIDEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RAW: tl.constexpr,
) -> None:
    pair = tl.program_id(0).to(tl.int64) * BP + tl.arange(0, BP)
    batch = tl.program_id(1).to(tl.int64)
    in_range = pair < I * J
    i, j = pair // J, pair % J
    distance = tl.zeros((BP,), tl.float32)
    for step in tl.static_range(3):
        # Axes 0, 2, 1: PyTorch reduces a 3-vector as (x + z) + y.
        axis = step * 2 % 3
        x = tl.load(X + (batch * I + i) * 3 + axis, in_range, 0).to(tl.float32)
        y = tl.load(Y + (batch * J + j) * 3 + axis, in_range, 0).to(tl.float32)
        # Subtract in the coordinate dtype, square and sum in FP32.
        delta = (x - y).to(X.dtype.element_ty).to(tl.float32)
        distance += delta * delta
    if EUCLIDEAN:
        distance = tl.sqrt_rn(distance)
    bins = tl.arange(0, BK)
    lower = tl.load(L + bins, bins < K, float("inf"))
    index = tl.sum((distance[:, None] > lower[None, :]).to(tl.int32), 1) - 1
    upper = tl.load(U + index, index >= 0, 0)
    hit = (index >= 0) & (distance < upper)
    channel = tl.arange(0, BC)
    weight = tl.load(W + channel[None, :] * K + index[:, None], hit[:, None] & (channel < C)[None, :], 0)
    result = weight.to(tl.float32)
    if HAS_BIAS:
        result += tl.load(B + channel, channel < C, 0).to(tl.float32)[None, :]
    if HAS_RAW:
        result += distance[:, None] * tl.load(R + channel, channel < C, 0).to(tl.float32)[None, :]
    tl.store(
        O + (batch * I * J + pair[:, None]) * C + channel[None, :],
        result,
        in_range[:, None] & (channel < C)[None, :],
    )


class _DistanceEmbedding(TritonKernelCache):
    def __init__(self, dtypes: tuple[torch.dtype, ...], constants: tuple[tuple[str, int | bool], ...]) -> None:
        self.kernel = self.compile_for_dtypes(
            _distance_embedding,
            dtypes=[dtypes[0]],
            make_dummy_args=lambda dtype: (*(torch.empty(1, device="cuda", dtype=dt) for dt in dtypes), 2, 2),
            grid=(0,),
            enable_fp_fusion=False,
            **dict(constants),
        )[dtypes[0]]


@cache
def _cached_embedding(
    device: int, dtypes: tuple[torch.dtype, ...], constants: tuple[tuple[str, int | bool], ...]
) -> CachedKernel:
    with torch.cuda.device(device):
        return _DistanceEmbedding(dtypes, constants).kernel


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    # The cached CUBIN assumes 16-byte aligned pointers.
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def project_distance_bins(
    rows: torch.Tensor,
    coordinates: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    distance_weight: torch.Tensor | None = None,
    euclidean: bool = False,
) -> torch.Tensor:
    """Project one-hot distance bins without materializing them.

    Matches ``linear(one_hot, weight, bias)`` with FP32 math: coordinates
    subtract in their own dtype; squares, sums and the output are FP32.

    Args:
        rows: CUDA coordinates [..., I, 3].
        coordinates: Coordinates [..., J, 3] with the leading dims and dtype of ``rows``.
        lower: Sorted exclusive lower bounds [K].
        upper: Exclusive upper bounds [K]; ``upper[k] == lower[k + 1]`` except the last.
        weight: Projection weights [C, K].
        bias: Optional bias [C].
        distance_weight: Optional raw-distance projection [C], added as ``distance * distance_weight``.
        euclidean: Bin the Euclidean distance instead of its square.

    Returns:
        FP32 features [..., I, J, C]; ``bias`` alone where no bin matches.
    """
    *leading, i, _ = rows.shape
    j, channels, bins = coordinates.shape[-2], weight.shape[0], lower.numel()
    output = rows.new_empty((*leading, i, j, channels), dtype=torch.float32)
    if not output.numel():
        return output
    inputs = (rows, coordinates, lower, upper, weight, weight if bias is None else bias)
    inputs += (weight if distance_weight is None else distance_weight.reshape(channels),)
    tensors = (*(_aligned(t) for t in inputs), output)
    constants = {
        "K": bins,
        "C": channels,
        "BK": triton.next_power_of_2(bins),
        "BP": _BLOCK_PAIRS,
        "BC": triton.next_power_of_2(channels),
        "EUCLIDEAN": euclidean,
        "HAS_BIAS": bias is not None,
        "HAS_RAW": distance_weight is not None,
    }
    grid = (triton.cdiv(i * j, _BLOCK_PAIRS), math.prod(leading))
    with torch.cuda.device(rows.device):
        kernel = _cached_embedding(rows.device.index, tuple(t.dtype for t in tensors), tuple(constants.items()))
        if kernel.driver is not None:
            kernel.driver.launch_with((*(t.data_ptr() for t in tensors), i, j), *grid)
        else:
            kernel.launch(grid, *tensors, i, j, *constants.values())
    return output
