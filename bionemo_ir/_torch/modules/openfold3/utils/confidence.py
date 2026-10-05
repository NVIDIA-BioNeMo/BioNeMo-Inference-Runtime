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
"""Shared PAE, pTM and ipTM reductions over OpenFold3 PAE logits.

One FP32 softmax per row block feeds the expected aligned error and the
expected TM term, so a caller never holds a full ``[N, N, n_bins]``
probability tensor. The scores follow AF3 SI 5.9.1 Eqs. (17-18), as the
dense reference in the OpenFold3 postprocessor does. Chunking preserves the
equations; floating-point reduction order can change the last bits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from bionemo_ir._torch.utils import iter_chunks

__all__ = [
    "PAE_BIN_MAX",
    "PAE_BIN_MIN",
    "PAE_BLOCK_BYTES",
    "PaeReduction",
    "bin_centers",
    "pae_reduction_rows",
    "reduce_pae_logits",
    "tm_d0",
    "tm_max",
    "tm_per_bin",
    "tm_rows",
]

PAE_BIN_MIN = 0.0
PAE_BIN_MAX = 32.0
# FP32 bytes per softmax block. Measured on an H100 host (EPYC 7413, 128 MiB
# L3): CPU blocks past ~16 MiB fall out of cache and run 5x slower, while
# CUDA blocks under ~64 MiB start paying per-launch overhead. At 1734 tokens
# and 64 bins these give 18 CPU rows and 302 CUDA rows; the whole matrix is
# 734 MiB.
PAE_BLOCK_BYTES = {"cpu": 8 << 20, "cuda": 128 << 20}


def bin_centers(bin_min: float, bin_max: float, n_bins: int) -> torch.Tensor:
    """Midpoints of ``n_bins`` equal-width bins spanning ``[bin_min, bin_max]``.

    A binned head predicts a distribution over bins, so its expectation weights
    each bin by that bin's midpoint: 0.25, 0.75, ... 31.75 A for the 64-bin PAE
    head and 0.01, 0.03, ... 0.99 for the 50-bin pLDDT head. Mirrors upstream
    ``openfold3/core/metrics/confidence.py::get_bin_centers``.

    ``torch.linspace(bin_min, bin_max, n_bins)`` returns bin *boundaries*
    instead, starting on the bottom edge and ending on the top one. That
    stretches the grid by ``n_bins / (n_bins - 1)``, so it misplaces every
    weight and biases the expectation it feeds.
    """
    width = (bin_max - bin_min) / n_bins
    boundaries = torch.linspace(bin_min, bin_max, n_bins + 1, dtype=torch.float32)
    return boundaries[:-1] + 0.5 * width


def tm_d0(n_tokens: int) -> float:
    """TM-score ``d0`` for ``n_tokens``; the floor of 19 keeps it positive."""
    return 1.24 * (max(n_tokens, 19) - 15) ** (1.0 / 3.0) - 1.8


def tm_per_bin(n_bins: int, n_tokens: int) -> torch.Tensor:
    """Expected TM term of each PAE bin, ``1 / (1 + (e / d0)^2)`` at the bin center."""
    centers = bin_centers(PAE_BIN_MIN, PAE_BIN_MAX, n_bins)
    return 1.0 / (1.0 + (centers / tm_d0(n_tokens)) ** 2)


def pae_reduction_rows(n_tokens: int, n_bins: int, device: torch.device) -> int:
    """Token rows per softmax block that keep one FP32 block within the device's byte budget."""
    budget = PAE_BLOCK_BYTES["cuda" if device.type == "cuda" else "cpu"]
    return max(1, budget // max(n_tokens * n_bins * 4, 1))


def tm_rows(
    tm_per_pair: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-aligned-token TM averages and their eligibility.

    Args:
        tm_per_pair: ``[..., rows, N]`` expected TM term per pair.
        pair_mask: optional ``[..., rows, N]`` 0/1 mask of the scored pairs.
        has_frame: optional ``[..., rows]`` mask of tokens with a valid frame.

    Returns:
        ``(scores, eligible)``, each ``[..., rows]``. A row is eligible when it
        scores at least one pair and, when given, has a frame.
    """
    if pair_mask is None:
        pair_mask = torch.ones_like(tm_per_pair)
    n_scored = pair_mask.sum(dim=-1)
    scores = (tm_per_pair * pair_mask).sum(dim=-1) / n_scored.clamp(min=1)
    eligible = n_scored > 0
    if has_frame is not None:
        eligible = eligible & has_frame.to(device=eligible.device, dtype=torch.bool)
    return scores, eligible


def tm_max(scores: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    """Maximum of ``scores`` over eligible rows along the last axis, NaN when none is."""
    if scores.shape[-1] == 0:
        return scores.new_full(scores.shape[:-1], float("nan"))
    best = scores.masked_fill(~eligible, -float("inf")).amax(dim=-1)
    return best.masked_fill(~eligible.any(dim=-1), float("nan"))


@dataclass(frozen=True)
class PaeReduction:
    """The three consumers of one sample's PAE logits.

    Attributes:
        pae: ``[N, N]`` FP32 expected aligned error.
        scores: ``[2]`` FP32 ``(pTM, ipTM)``; NaN marks an unavailable score.
    """

    pae: torch.Tensor
    scores: torch.Tensor

    @property
    def ptm(self) -> torch.Tensor:
        return self.scores[0]

    @property
    def iptm(self) -> torch.Tensor:
        return self.scores[1]


def reduce_pae_logits(
    logits: torch.Tensor,
    n_tokens: int,
    chain_indices: torch.Tensor,
    has_frame: torch.Tensor | None,
    *,
    rows: int | None = None,
    pae_out: torch.Tensor | None = None,
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> PaeReduction:
    """Reduce one sample's PAE logits to PAE, pTM and ipTM in bounded row blocks.

    Every row block is normalized once in FP32; the block's probabilities then
    feed both expectations and are released before the next block. pTM averages
    over every token, ipTM over the tokens of other chains, and both take the
    maximum over aligned tokens that have a valid frame.

    Args:
        logits: ``[N_pad, N_pad, n_bins]`` PAE logits, any float dtype, on CPU or
            CUDA. Only the leading ``n_tokens`` rows and columns are read. With
            ``project``, the ``[N_pad, N_pad, C]`` pair embedding instead.
        n_tokens: real token count ``N``; sets ``d0`` and the crop.
        chain_indices: ``[>= N]`` integer chain id per token.
        has_frame: optional ``[>= N]`` mask of tokens eligible as the aligned
            token; ``None`` treats every token as eligible.
        rows: token rows per softmax block; ``None`` sizes the block from
            :data:`PAE_BLOCK_BYTES` and ``<= 0`` reduces in one block.
        pae_out: optional ``[N, N]`` FP32 destination for the PAE.
        project: optional map from a ``[rows, N, C]`` pair block to its
            ``[rows, N, n_bins]`` logits, so no sample ever holds its full
            logits. Needs an explicit ``rows``.

    Returns:
        The reduction on the logits' device.
    """
    device = logits.device
    if pae_out is None:
        pae_out = torch.empty((n_tokens, n_tokens), dtype=torch.float32, device=device)
    scores = torch.empty((2, n_tokens), dtype=torch.float32, device=device)
    eligible = torch.empty((2, n_tokens), dtype=torch.bool, device=device)
    chains = torch.as_tensor(chain_indices, device=device)[:n_tokens]
    frames = None if has_frame is None else torch.as_tensor(has_frame, device=device)[:n_tokens].bool()
    centers = tm_bins = None

    if rows is None:
        if project is not None:
            raise ValueError("rows is required with project: the bin count is unknown before projecting")
        rows = pae_reduction_rows(n_tokens, logits.shape[-1], device)
    block_rows = rows if rows > 0 else max(n_tokens, 1)
    for start, length in iter_chunks(n_tokens, block_rows):
        stop = start + length
        block = logits[start:stop, :n_tokens]
        if project is not None:
            block = project(block)
        if centers is None or tm_bins is None:
            n_bins = block.shape[-1]
            centers = bin_centers(PAE_BIN_MIN, PAE_BIN_MAX, n_bins).to(device=device)
            tm_bins = tm_per_bin(n_bins, n_tokens).to(device=device)
        probs = torch.softmax(block.float(), dim=-1)
        del block
        pae_out[start:stop] = (probs * centers).sum(dim=-1)
        tm_pairs = (probs * tm_bins).sum(dim=-1)
        del probs
        block_frames = None if frames is None else frames[start:stop]
        interface = (chains[start:stop, None] != chains[None, :]).to(dtype=torch.float32)
        for index, pair_mask in enumerate((None, interface)):
            scores[index, start:stop], eligible[index, start:stop] = tm_rows(tm_pairs, pair_mask, block_frames)
        del tm_pairs, interface

    return PaeReduction(pae=pae_out, scores=tm_max(scores, eligible))
