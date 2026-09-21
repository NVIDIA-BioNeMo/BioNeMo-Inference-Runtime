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

from bionemo_ir._torch.modules.openfold3.utils.relpos import relpos_complex
from bionemo_ir._torch.utils import dist_one_hot


def _legacy_relpos(batch: dict[str, torch.Tensor], relative_clip: int, chain_clip: int) -> torch.Tensor:
    """Reference the original nearest-bin encoding and feature column order."""
    same_chain = batch["asym_id"][..., None] == batch["asym_id"][..., None, :]
    same_residue = batch["residue_index"][..., None] == batch["residue_index"][..., None, :]
    same_entity = batch["entity_id"][..., None] == batch["entity_id"][..., None, :]

    def encode(name: str, condition: torch.Tensor, clip: int) -> torch.Tensor:
        indices = batch[name]
        offset = (indices[..., None] - indices[..., None, :] + clip).clamp(0, 2 * clip)
        offset = torch.where(condition, offset, torch.full_like(offset, 2 * clip + 1))
        return dist_one_hot(offset, torch.arange(2 * clip + 2))

    return torch.cat(
        [
            encode("residue_index", same_chain, relative_clip),
            encode("token_index", same_chain & same_residue, relative_clip),
            same_entity[..., None].float(),
            encode("sym_id", same_entity, chain_clip),
        ],
        dim=-1,
    )


@pytest.mark.parametrize("batch_shape", [(2,), (2, 1)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("relative_clip,chain_clip", [(0, 0), (2, 1), (32, 2)])
def test_relative_position_rows_match_legacy_exactly(
    batch_shape: tuple[int, ...], index_dtype: torch.dtype, relative_clip: int, chain_clip: int
) -> None:
    generator = torch.Generator().manual_seed(42)
    tokens = 11
    batch = {
        name: torch.randint(-40, 40, (*batch_shape, tokens), generator=generator, dtype=index_dtype)
        for name in ("residue_index", "token_index", "sym_id")
    }
    for name in ("asym_id", "entity_id"):
        batch[name] = torch.randint(0, 3, (*batch_shape, tokens), generator=generator, dtype=index_dtype)
    # Same-residue atomized tokens and heterogeneous chains exercise every gate.
    batch["residue_index"][..., 1] = batch["residue_index"][..., 0]
    batch["asym_id"][..., 1] = batch["asym_id"][..., 0]

    expected = _legacy_relpos(batch, relative_clip, chain_clip)
    actual = relpos_complex(batch, relative_clip, chain_clip)
    chunks = [
        relpos_complex(batch, relative_clip, chain_clip, row_slice=slice(start, start + 4))
        for start in range(0, tokens, 4)
    ]

    assert torch.equal(actual, expected)
    assert torch.equal(torch.cat(chunks, dim=-3), expected)
