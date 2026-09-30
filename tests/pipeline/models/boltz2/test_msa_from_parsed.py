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
"""Boltz-2 A3M rows decode like the per-character loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.boltz2.const import prot_letter_to_token, token_ids
from bionemo_ir.pipeline.models.boltz2.featurizer import _msa_from_parsed

RAW = ["ACD", "KaLbM", "AC", "ccPQRd", "A-CD", "ıſé", "AéCD", "ÄCD", "ACßD", "KLM", "WYV", "xxWYVx"]


def _token(char: str) -> int:
    return token_ids.get(prot_letter_to_token.get(char.upper(), "UNK"), token_ids["UNK"])


def _reference(raw: list[str], width: int) -> tuple[list[list[int]], list[list[int]], list[str]]:
    rows, dels, keys, visited = [], [], [], set()
    for s in raw:
        key = s.replace("-", "").upper()
        if key in visited:
            continue
        visited.add(key)
        row, counts, run = [], [], 0
        for char in s:
            if char.islower():
                run += 1
            else:
                row.append(_token(char))
                counts.append(run)
                run = 0
        if len(row) == width:
            rows.append(row)
            dels.append(counts)
            keys.append(key)
    return rows, dels, keys


def test_rows_match_per_character_reference() -> None:
    msa, deletion, paired, keys = _msa_from_parsed({"sequences": RAW, "raw": RAW}, 3, prot_letter_to_token)
    rows, dels, ref_keys = _reference(RAW, 3)
    assert msa[1:].tolist() == rows
    assert torch.equal(deletion[1:], torch.tensor(dels, dtype=torch.float32))
    assert keys[1:] == ref_keys
    assert paired.shape == msa.shape


@pytest.mark.parametrize("limit", [0, 1, 2, 5, 100])
def test_limit_keeps_the_unlimited_prefix(limit: int) -> None:
    parsed = {"sequences": RAW, "raw": RAW}
    full = _msa_from_parsed(parsed, 3, prot_letter_to_token, visited=set())
    capped = _msa_from_parsed(parsed, 3, prot_letter_to_token, visited=set(), limit=limit)
    n = 1 + min(limit, full[0].shape[0] - 1)
    for a, b in zip(full[:3], capped[:3], strict=True):
        assert torch.equal(a[:n], b)
    assert capped[3] == full[3][:n]


def test_zero_residue_rows() -> None:
    msa, deletion, _, _ = _msa_from_parsed({"sequences": ["", "a", "bb"]}, 0, prot_letter_to_token, default_query=[])
    assert msa.shape == (4, 0) and deletion.shape == (4, 0)
    assert np.array_equal(msa.numpy(), np.zeros((4, 0), dtype=np.int64))
