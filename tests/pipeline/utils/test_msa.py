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
"""Vectorized A3M decoding matches the per-character loops."""

from __future__ import annotations

import numpy as np
import pytest

from bionemo_ir.pipeline.utils.msa import a3m_columns, code_points, map_code_points, ragged_mask, truncate_rows


def _reference(rows: list[str]) -> tuple[list[int], list[int], list[int]]:
    codes, deletions, counts = [], [], []
    for row in rows:
        run, count = 0, 0
        for char in row:
            if char.islower():
                run += 1
                continue
            codes.append(ord(char))
            deletions.append(run)
            run, count = 0, count + 1
        counts.append(count)
    return codes, deletions, counts


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [""],
        ["ACDEF", "aaACdeF-.gG", "abc", "", "-.-.", "xyzA"],
        ["AébÉC", "ıſacD", "ßA", "\ud800a"],
    ],
)
def test_a3m_columns_match_reference(rows: list[str]) -> None:
    codes, deletions, counts = a3m_columns(rows)
    expected = _reference(rows)
    assert (codes.tolist(), deletions.tolist(), counts.tolist()) == expected


def test_code_points_widen_only_for_unicode() -> None:
    assert code_points("AC-").dtype == np.uint8
    assert code_points("Aé").tolist() == [ord("A"), ord("é")]


def test_map_code_points_resolves_in_first_seen_order() -> None:
    table = np.arange(128, dtype=np.int64)
    table[ord("X")] = -1
    seen: list[str] = []

    def resolve(char: str) -> int:
        seen.append(char)
        return 1000 + ord(char)

    values = map_code_points(code_points("AéXéΔX"), table, resolve)
    assert values.tolist() == [
        ord("A"),
        1000 + ord("é"),
        1000 + ord("X"),
        1000 + ord("é"),
        1000 + ord("Δ"),
        1000 + ord("X"),
    ]
    assert seen == ["é", "X", "Δ"]


def test_truncate_rows_and_ragged_mask() -> None:
    values = np.arange(7)
    counts = np.array([3, 0, 4])
    kept = truncate_rows(values, counts, np.array([2, 0, 4]))
    assert kept.tolist() == [0, 1, 3, 4, 5, 6]
    out = np.full((3, 4), -1)
    out[ragged_mask(np.array([2, 0, 4]), 4)] = kept
    assert out.tolist() == [[0, 1, -1, -1], [-1, -1, -1, -1], [3, 4, 5, 6]]
