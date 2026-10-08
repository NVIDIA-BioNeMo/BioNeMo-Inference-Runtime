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
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedParameters

from bionemo_ir.pipeline.models.openfold3 import feature_context as fc
from bionemo_ir.pipeline.models.openfold3.feature_context import (
    _build_nucleotide_rdkit_mol,
    _build_structure_from_polymers,
)


@pytest.mark.parametrize("length", [7, 8, 29, 31, 32, 40])
@pytest.mark.parametrize("polymer_type", ["rna", "dna"])
def test_parallel_nucleotide_embedding_preserves_seed_and_coordinates(length: int, polymer_type: str) -> None:
    sequence = (("ACGU" if polymer_type == "rna" else "ACGT") * 10)[:length]
    codes = {char: char if polymer_type == "rna" else "D" + char for char in "ACGUT"}
    saved_state = random.getstate()
    try:
        random.seed(42)
        expected = [_build_nucleotide_rdkit_mol(codes[char]) for char in sequence]
        expected_state = random.getstate()

        random.seed(42)
        structure = _build_structure_from_polymers(
            [{"sequence": sequence, "chain_id": "A", "polymer_type": polymer_type}]
        )

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


@pytest.mark.parametrize("failure", ["primary", "both", "raise"])
def test_short_retry_parity(failure: str) -> None:
    codes = ["A", "C", "G", "U"] * 3
    embed = fc.AllChem.EmbedMolecule
    atom_count = Chem.AddHs(fc._nucleotide_rdkit_topology("C")[0]).GetNumAtoms()

    def failing_embed(mol: Chem.Mol, params: EmbedParameters) -> int:
        if mol.GetNumAtoms() == atom_count:
            if failure == "raise":
                raise ValueError("Injected embedding failure")
            if failure == "both" or not params.useRandomCoords:
                return -1
        return embed(mol, params)

    saved_state = random.getstate()
    try:
        with patch.object(fc.AllChem, "EmbedMolecule", side_effect=failing_embed):
            random.seed(42)
            expected = [_build_nucleotide_rdkit_mol(code) for code in codes]
            expected_state = random.getstate()
            random.seed(42)
            actual = fc._prefetch_serial_mols(codes, fc._nucleotide_rdkit_topology, _build_nucleotide_rdkit_mol)
        assert actual is not None
        assert random.getstate() == expected_state
        for (mol, mask), (reference, reference_mask) in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(mask, reference_mask)
            if reference is None:
                assert mol is None
            else:
                assert mol is not None
                assert Chem.MolToMolBlock(mol) == Chem.MolToMolBlock(reference)
                assert mol.GetNumConformers() == reference.GetNumConformers()
                if mol.GetNumConformers():
                    np.testing.assert_array_equal(
                        mol.GetConformer().GetPositions(), reference.GetConformer().GetPositions()
                    )
    finally:
        random.setstate(saved_state)
