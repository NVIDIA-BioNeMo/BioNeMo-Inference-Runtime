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

import math
from collections.abc import Callable

import torch

from .auto_chunk import iter_chunks


def compact_inference(enabled: bool, tensor: torch.Tensor) -> bool:
    """Allow compact outputs outside CUDA graph capture."""
    return enabled and (not tensor.is_cuda or not torch.cuda.is_current_stream_capturing())


def pair_expectations(
    pair: torch.Tensor,
    project: Callable[[torch.Tensor, int, int], torch.Tensor],
    weights: tuple[torch.Tensor, ...],
    *,
    symmetric: bool = False,
    rows: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Project pair rows and share their softmax across model-specific expectations.

    Args:
        pair: Floating point pair embeddings shaped ``[..., N, N, C]``.
        project: Map a row block and its start/stop indices to binned logits.
        weights: Bin weights broadcastable to ``[..., rows, N, bins]``.
            Softmax retains the projected logits' dtype, matching raw heads.
        symmetric: Add transposed pair rows before projection.
        rows: Optional positive row count; defaults to a bounded byte budget.

    Returns:
        One ``[..., N, N]`` expectation per set of bin weights.
    """
    n_tokens = pair.shape[-2]
    if rows is None:
        budget = (128 if pair.is_cuda else 8) << 20
        row_bytes = math.prod(pair.shape[:-3]) * n_tokens * weights[0].shape[-1] * 4
        rows = max(1, budget // max(row_bytes, 1))
    if rows < 1:
        raise ValueError("Confidence rows must be positive")
    outputs = None
    for start, length in iter_chunks(n_tokens, rows):
        stop = start + length
        block = pair[..., start:stop, :, :]
        if symmetric:
            block = block + pair[..., :, start:stop, :].transpose(-3, -2)
        logits = project(block, start, stop)
        probabilities = torch.softmax(logits, dim=-1)
        del logits, block
        reduced = tuple((probabilities * weight).sum(dim=-1) for weight in weights)
        if outputs is None:
            outputs = tuple(value.new_empty(pair.shape[:-1]) for value in reduced)
        for output, value in zip(outputs, reduced, strict=True):
            output[..., start:stop, :] = value
        del probabilities, reduced, value
    if outputs is None:
        return tuple(pair.new_empty(pair.shape[:-1], dtype=weight.dtype) for weight in weights)
    return outputs
