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
"""Batched per-residue augmentation matches one augmentation call per residue."""

from __future__ import annotations

import pytest
import torch

from bionemo_ir._torch.layers.random_augmentation import centre_random_augmentation
from bionemo_ir.pipeline.models.openfold3.common import centre_random_augmentation_blocks


def _per_block(pos: torch.Tensor, block_sizes: list[int]) -> torch.Tensor:
    out = torch.empty_like(pos)
    offset = 0
    for size in block_sizes:
        block = pos[offset : offset + size]
        out[offset : offset + size] = centre_random_augmentation(
            block, torch.ones(size), normalize_quaternions_first=True, mask_denominator_min=1.0
        )
        offset += size
    return out


@pytest.mark.parametrize("block_sizes", [[], [1], [5, 1, 14, 1, 1, 23], [8] * 40])
def test_blocks_match_per_block_calls_and_rng_stream(block_sizes: list[int]) -> None:
    pos = torch.randn(sum(block_sizes), 3, generator=torch.Generator().manual_seed(0)) * 10

    torch.manual_seed(7)
    expected = _per_block(pos, block_sizes)
    expected_next = torch.randn(4)
    torch.manual_seed(7)
    got = centre_random_augmentation_blocks(pos, block_sizes)
    got_next = torch.randn(4)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)
    assert torch.equal(got_next, expected_next)
