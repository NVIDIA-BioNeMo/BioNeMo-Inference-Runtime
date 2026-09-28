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

import random
from unittest.mock import patch

import numpy as np

from bionemo_ir.pipeline.models.openfold3.feature_context import (
    _build_nucleotide_rdkit_mol,
    _build_structure_from_polymers,
)


def test_parallel_nucleotide_embedding_preserves_seed_and_coordinates() -> None:
    sequence = "ACGU" * 10
    codes = {"A": "A", "C": "C", "G": "G", "U": "U"}
    saved_state = random.getstate()
    try:
        random.seed(42)
        expected = [_build_nucleotide_rdkit_mol(codes[char]) for char in sequence]
        expected_state = random.getstate()

        random.seed(42)
        structure = _build_structure_from_polymers([{"sequence": sequence, "chain_id": "A", "polymer_type": "rna"}])

        assert random.getstate() == expected_state
        for (mol, crop_mask), actual_mol, actual_mask in zip(
            expected, structure["residue_mols"], structure["residue_crop_masks"], strict=True
        ):
            assert mol is not None and actual_mol is not None
            np.testing.assert_array_equal(actual_mol.GetConformer().GetPositions(), mol.GetConformer().GetPositions())
            np.testing.assert_array_equal(actual_mask, crop_mask)
    finally:
        random.setstate(saved_state)


def test_cached_topology_keeps_distinct_conformers_and_masks() -> None:
    saved_state = random.getstate()
    try:
        random.seed(42)
        first_mol, first_mask = _build_nucleotide_rdkit_mol("A")
        second_mol, second_mask = _build_nucleotide_rdkit_mol("A")

        assert first_mol is not None and second_mol is not None
        assert first_mol is not second_mol
        assert first_mask is not second_mask
        assert not np.array_equal(first_mol.GetConformer().GetPositions(), second_mol.GetConformer().GetPositions())
        np.testing.assert_array_equal(first_mask, second_mask)
    finally:
        random.setstate(saved_state)


def test_topology_prefetch_failure_uses_residue_fallback() -> None:
    sequence = "A" * 32
    with patch(
        "bionemo_ir.pipeline.models.openfold3.feature_context._nucleotide_rdkit_topology",
        side_effect=ValueError("missing topology"),
    ) as topology:
        structure = _build_structure_from_polymers([{"sequence": sequence, "chain_id": "A", "polymer_type": "rna"}])

    assert topology.call_count == len(sequence) + 1
    assert structure["residue_mols"] == [None] * len(sequence)
    assert all(mask.size == 0 for mask in structure["residue_crop_masks"])
