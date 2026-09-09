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

import pytest
import torch

from bionemo_ir._torch.modules.boltz.confidence_utils import (
    NUM_CONTACT_BINS,
    compute_contact_prob,
    repeat_with_multiplicity,
)
from bionemo_ir._torch.utils import ChunkPolicy


def _reference_contact_prob(logits: torch.Tensor, num_contact_bins: int = NUM_CONTACT_BINS) -> torch.Tensor:
    """The pre-reduction confidence-head expression: softmax over bins, mask, sum.

    Mirrors the original ``(softmax(logits, -1) * contacts).sum(-1)`` with ``contacts`` a full-width
    0/1 mask, which is what ``compute_contact_prob`` must reproduce.
    """
    prob = torch.softmax(logits.float(), dim=-1)
    contacts = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
    contacts[:num_contact_bins] = 1.0
    return (prob * contacts).sum(-1)


@pytest.mark.parametrize("n_tokens", [7, 64, 129])
@pytest.mark.parametrize("num_bins", [64, 38])
def test_matches_head_reduction(n_tokens, num_bins):
    torch.manual_seed(0)
    logits = torch.randn(2, n_tokens, n_tokens, num_bins)

    torch.testing.assert_close(
        compute_contact_prob(logits),
        _reference_contact_prob(logits),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 16, 512])
def test_row_chunking_is_exact(chunk_size):
    """Row-chunking must not perturb the result: softmax is independent per pair row."""
    torch.manual_seed(0)
    logits = torch.randn(1, 33, 33, 64)

    dense = compute_contact_prob(logits, policy=ChunkPolicy(chunk_size=chunk_size, min_size=10**9, dim=1))
    chunked = compute_contact_prob(logits, policy=ChunkPolicy(chunk_size=chunk_size, min_size=0, dim=1))

    torch.testing.assert_close(chunked, dense, rtol=0, atol=0)
    assert chunked.shape == (1, 33, 33)


def test_reduces_pair_dim_and_is_a_probability():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 12, 64)
    out = compute_contact_prob(logits)

    assert out.shape == (1, 12, 12)
    assert out.dtype == torch.float32
    assert ((out >= 0) & (out <= 1)).all()


def test_num_contact_bins_selects_leading_bins():
    """All mass in a bin inside/outside the cutoff drives the probability to 1/0."""
    logits = torch.full((1, 2, 2, 64), -1e4)
    logits[..., 0] = 1e4  # nearest bin -> a contact
    torch.testing.assert_close(compute_contact_prob(logits), torch.ones(1, 2, 2))

    logits = torch.full((1, 2, 2, 64), -1e4)
    logits[..., NUM_CONTACT_BINS] = 1e4  # first bin beyond the cutoff -> not a contact
    torch.testing.assert_close(compute_contact_prob(logits), torch.zeros(1, 2, 2))


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_reduce_then_repeat_matches_repeat_then_reduce(multiplicity):
    """The heads now repeat an already-reduced ``prob_contact`` instead of repeating the logits.

    ``repeat_interleave`` copies exactly, so hoisting the reduction ahead of the sample repeat is
    value-preserving -- and it drops the per-sample ``[B, mult, N, N, num_bins]`` softmax entirely.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 5, 64)

    reduce_then_repeat = repeat_with_multiplicity(compute_contact_prob(logits), multiplicity)
    repeat_then_reduce = _reference_contact_prob(repeat_with_multiplicity(logits, multiplicity))

    torch.testing.assert_close(reduce_then_repeat, repeat_then_reduce, rtol=0, atol=0)
    assert reduce_then_repeat.shape == (2, multiplicity, 5, 5)


def test_bf16_logits_reduce_in_fp32():
    """bf16 logits are promoted before the softmax, so the result keeps fp32 resolution."""
    torch.manual_seed(0)
    logits = torch.randn(1, 8, 8, 64).bfloat16()
    out = compute_contact_prob(logits)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _reference_contact_prob(logits), rtol=0, atol=0)
