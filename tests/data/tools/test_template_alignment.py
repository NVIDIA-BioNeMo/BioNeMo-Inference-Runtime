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

import numpy as np
import pytest

from bionemo_ir.data.tools.template_alignment import calculate_ids_hit, seq_identity_and_coverage


def _chars(text: str) -> np.ndarray:
    return np.asarray(list(text), dtype="<U1")


@pytest.mark.parametrize(
    "query_aln, template_aln, expected_query_ids, expected_template_ids",
    [
        # Mutually aligned with an internal gap on each side.
        ("AC-D", "A-CD", [1, 2, -1, 3], [1, -1, 2, 3]),
        # No gaps: straight 1-based enumeration.
        ("MR", "MR", [1, 2], [1, 2]),
        # A one-sided leading gap is kept and encoded as -1 on the gap side.
        ("-A", "MA", [-1, 1], [1, 2]),
    ],
)
def test_calculate_ids_hit(query_aln, template_aln, expected_query_ids, expected_template_ids):
    query_ids, template_ids = calculate_ids_hit(_chars(query_aln), _chars(template_aln))
    np.testing.assert_array_equal(query_ids, np.asarray(expected_query_ids, dtype=np.int64))
    np.testing.assert_array_equal(template_ids, np.asarray(expected_template_ids, dtype=np.int64))


def test_calculate_ids_hit_drops_double_gap_columns():
    # A column that is a gap on both sides carries no residue and is removed.
    query_ids, template_ids = calculate_ids_hit(_chars("A-C"), _chars("A.C"))
    np.testing.assert_array_equal(query_ids, np.asarray([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(template_ids, np.asarray([1, 2], dtype=np.int64))


@pytest.mark.parametrize(
    "query_aln, template_aln, query_seq, expected_identity, expected_coverage",
    [
        ("AC-D", "A-CD", "ACD", 2 / 3, 2 / 3),
        ("MR", "MR", "MR", 1.0, 1.0),
        ("MR", "LK", "MR", 0.0, 1.0),
    ],
)
def test_seq_identity_and_coverage(query_aln, template_aln, query_seq, expected_identity, expected_coverage):
    identity, coverage = seq_identity_and_coverage(_chars(query_aln), _chars(template_aln), query_seq)
    assert identity == pytest.approx(expected_identity)
    assert coverage == pytest.approx(expected_coverage)


def test_seq_identity_and_coverage_all_gap_query_is_zero():
    identity, coverage = seq_identity_and_coverage(_chars("--"), _chars("MR"), "MR")
    assert identity == 0.0
    assert coverage == 0.0
