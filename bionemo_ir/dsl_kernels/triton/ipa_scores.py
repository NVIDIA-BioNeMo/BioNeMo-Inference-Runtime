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


"""FP32 invariant point attention scores without quadratic point arrays."""

import math
from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache


@triton.jit
def _point_sum(
    Q,
    K,
    weight,
    batch,
    query,
    head,
    key,
    valid,
    N,
    H: tl.constexpr,
    P: tl.constexpr,
    FIRST: tl.constexpr,
    STRIDE: tl.constexpr,
    COUNT: tl.constexpr,
):
    if COUNT == 1:
        q_offset = ((batch * N + query) * H + head) * P * 3 + FIRST * 3
        k_offset = ((batch * H + head) * N + key) * P * 3 + FIRST * 3
        dx = tl.load(Q + q_offset) - tl.load(K + k_offset, valid, 0)
        dy = tl.load(Q + q_offset + 1) - tl.load(K + k_offset + 1, valid, 0)
        dz = tl.load(Q + q_offset + 2) - tl.load(K + k_offset + 2, valid, 0)
        return ((dx * dx + dy * dy) + dz * dz) * weight
    else:
        # Spell out the descending-offset tree. A generic tl.sum may choose
        # another association, magnified by rounding at the finite mask value.
        left = _point_sum(Q, K, weight, batch, query, head, key, valid, N, H, P, FIRST, STRIDE * 2, COUNT // 2)
        right = _point_sum(
            Q, K, weight, batch, query, head, key, valid, N, H, P, FIRST + STRIDE, STRIDE * 2, COUNT // 2
        )
        return left + right


# N is a runtime argument: one compile serves every length in a BLOCK bucket.
@triton.jit(do_not_specialize=["N"])
def _ipa_scores(
    A,
    B,
    Q,
    K,
    W,
    M,
    O,
    N,
    H: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    INF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    query = row % N
    head = (row // N) % H
    batch = row // (N * H)
    key = tl.arange(0, BLOCK)
    valid = key < N
    weight = tl.load(W + head)
    distance = _point_sum(Q, K, weight, batch, query, head, key, valid, N, H, P, 0, 1, P)
    scalar = tl.load(A + row * N + key, valid, 0) * math.sqrt(1.0 / (3 * C))
    pair = tl.load(B + ((batch * N + query) * N + key) * H + head, valid, 0) * math.sqrt(1.0 / 3)
    square_mask = tl.load(M + batch * N + query) * tl.load(M + batch * N + key, valid, 0)
    score = ((scalar + pair) + distance * (-0.5)) + INF * (square_mask - 1)
    score = tl.where(valid, score, -float("inf"))
    numerator = tl.exp(score - tl.max(score, 0))
    probability = numerator / tl.sum(numerator, 0)
    tl.store(O + row * N + key, probability, valid)


class _IPAScoresKernel(TritonKernelCache):
    def __init__(self, block: int, heads: int, points: int, channels: int, inf: float) -> None:
        self.kernel = self.compile_for_dtypes(
            _ipa_scores,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (*(torch.empty(1, device="cuda", dtype=dtype) for _ in range(7)), block),
            grid=(0,),
            H=heads,
            P=points,
            C=channels,
            INF=inf,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )[torch.float32]


@cache
def _scores_kernel(device: int, block: int, heads: int, points: int, channels: int, inf: float) -> _IPAScoresKernel:
    with torch.cuda.device(device):
        return _IPAScoresKernel(block, heads, points, channels, inf)


def _materialized(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.resolve_neg().contiguous()
    # The cached launcher specializes on the dummy tensors' 16-byte alignment.
    # A contiguous view can still start at an unaligned storage offset.
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def ipa_scores(
    scalar_scores: torch.Tensor,
    pair_bias: torch.Tensor,
    query_points: torch.Tensor,
    key_points: torch.Tensor,
    point_weights: torch.Tensor,
    mask: torch.Tensor,
    channels: int,
    inf: float,
) -> torch.Tensor:
    """Compute masked IPA probabilities in FP32 on CUDA.

    Args:
        scalar_scores: Contiguous scalar products [..., H, N, N].
        pair_bias: Contiguous projected pair features [..., N, N, H].
        query_points: Global query points [..., N, H, P, 3].
        key_points: Global key points with the same shape as query_points.
        point_weights: Positive weights [H], including the point scale.
        mask: Sequence mask [..., N].
        channels: Scalar query/key channel count.
        inf: Finite mask penalty, matching the eager path.

    Returns:
        Contiguous attention probabilities [..., H, N, N].
    """
    n, heads = scalar_scores.shape[-1], scalar_scores.shape[-3]
    points = query_points.shape[-2]
    if points not in (4, 8):
        raise ValueError("IPA score fusion supports four or eight query/key points")
    scalar_scores = _materialized(scalar_scores)
    batch_shape = scalar_scores.shape[:-3]
    point_shape = (*batch_shape, n, heads, points, 3)
    query_points = _materialized(query_points.expand(point_shape))
    # Adjacent keys must be adjacent in memory for each head. Otherwise a
    # warp fetches one point from each of H interleaved head records.
    key_points = _materialized(key_points.expand(point_shape).transpose(-4, -3))
    pair_bias = _materialized(pair_bias.expand(*batch_shape, n, n, heads))
    mask = _materialized(mask.expand(*batch_shape, n))
    point_weights = _materialized(point_weights)
    output = torch.empty_like(scalar_scores)
    if output.numel() == 0:
        return output
    tensors = (scalar_scores, pair_bias, query_points, key_points, point_weights, mask, output)
    block = triton.next_power_of_2(n)
    with torch.cuda.device(scalar_scores.device):
        kernel = _scores_kernel(scalar_scores.device.index, block, heads, points, channels, inf).kernel
        grid = (scalar_scores.numel() // n,)
        if kernel.driver is not None:
            kernel.driver.launch_with((*(t.data_ptr() for t in tensors), n), *grid)
        else:
            kernel.launch(grid, *tensors, n, heads, points, channels, inf, block)
    return output
