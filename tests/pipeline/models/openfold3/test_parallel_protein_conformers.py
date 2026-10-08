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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedParameters

from bionemo_ir.pipeline.models.openfold3 import feature_context as fc
from bionemo_ir.pipeline.models.openfold3.const import _PROTEIN_1TO3


@pytest.fixture(autouse=True)
def preserve_rng() -> Iterator[None]:
    state = random.getstate()
    yield
    random.setstate(state)


def legacy_residue(ccd_code: str) -> tuple[Chem.Mol | None, np.ndarray]:
    res = fc._get_residue_from_ccd_with_oxt(ccd_code)
    try:
        mol = fc.biotite_to_mol(res, kekulize=True)
        Chem.SanitizeMol(mol)
        mol.RemoveConformer(0)
        mol = Chem.AddHs(mol)
        fc._embed_conformer_inplace(mol)
        return Chem.RemoveHs(mol), np.asarray(res.atom_name != "OXT", dtype=bool)
    except Exception:
        return None, np.array([], dtype=bool)


def assert_mols_equal(actual: list, expected: list) -> None:
    for (mol, mask), (reference, reference_mask) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(mask, reference_mask)
        if reference is None:
            assert mol is None
            continue
        assert mol is not None
        assert Chem.MolToMolBlock(mol) == Chem.MolToMolBlock(reference)
        assert mol.GetNumConformers() == reference.GetNumConformers()
        if mol.GetNumConformers():
            np.testing.assert_array_equal(mol.GetConformer().GetPositions(), reference.GetConformer().GetPositions())


@pytest.mark.parametrize("seed", [0, 42, 12345])
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_prefetch_matches_legacy(seed: int, workers: int) -> None:
    codes = list(_PROTEIN_1TO3.values()) * 2
    random.seed(seed)
    expected = [legacy_residue(code) for code in codes]
    expected_state = random.getstate()
    random.seed(seed)
    with patch.object(fc, "ThreadPoolExecutor", side_effect=lambda **kwargs: ThreadPoolExecutor(workers)):
        actual = fc._prefetch_protein_mols(codes)
    assert actual is not None
    assert_mols_equal(actual, expected)
    assert random.getstate() == expected_state


@pytest.mark.parametrize("failure", ["primary", "both", "raise"])
def test_retry_preserves_rng(failure: str) -> None:
    codes = ["ALA", "LYS", "GLY", "TRP"] * 8
    embed = fc.AllChem.EmbedMolecule

    def failing_embed(mol: Chem.Mol, params: EmbedParameters) -> int:
        if mol.GetNumAtoms() == Chem.AddHs(fc._residue_rdkit_topology("LYS")[0]).GetNumAtoms():
            if failure == "raise":
                raise ValueError("Injected embedding failure")
            if failure == "both" or not params.useRandomCoords:
                return -1
        return embed(mol, params)

    with patch.object(fc.AllChem, "EmbedMolecule", side_effect=failing_embed):
        random.seed(42)
        expected = [legacy_residue(code) for code in codes]
        expected_state = random.getstate()
        random.seed(42)
        actual = fc._prefetch_protein_mols(codes)
    assert actual is not None
    assert_mols_equal(actual, expected)
    assert random.getstate() == expected_state


def test_mixed_structure_parity() -> None:
    polymers = [
        {"sequence": "ACDEFGHIKLMNPQRSTVWYX" * 2, "chain_id": ["A", "B"], "polymer_type": "protein"},
        {"sequence": "ACGU" * 8, "chain_id": "C", "polymer_type": "rna"},
        {"sequence": "ACGT" * 8, "chain_id": "D", "polymer_type": "dna"},
        {"sequence": "CCO", "chain_id": "E", "polymer_type": "smiles_ligand"},
    ]
    random.seed(42)
    with patch.object(fc, "_prefetch_protein_mols", return_value=None):
        expected = fc._build_structure_from_polymers(polymers)
    expected_state = random.getstate()
    random.seed(42)
    actual = fc._build_structure_from_polymers(polymers)
    assert random.getstate() == expected_state
    assert_mols_equal(
        list(zip(actual.pop("residue_mols"), actual.pop("residue_crop_masks"), strict=True)),
        list(zip(expected.pop("residue_mols"), expected.pop("residue_crop_masks"), strict=True)),
    )
    assert actual == expected


def test_topology_is_isolated() -> None:
    first, mask = fc._build_residue_rdkit_mol("ALA", 42)
    assert first is not None
    first.GetConformer().SetAtomPosition(0, (100, 200, 300))
    mask[:] = False
    second, other_mask = fc._build_residue_rdkit_mol("ALA", 42)
    random.seed(42)
    expected = legacy_residue("ALA")
    random.seed(42)
    third = fc._build_residue_rdkit_mol("ALA")
    assert_mols_equal([third], [expected])
    assert second is not first
    assert other_mask.any()
    assert fc._residue_rdkit_topology("ALA")[0].GetNumConformers() == 0


@pytest.mark.parametrize("target", ["_residue_rdkit_topology", "ThreadPoolExecutor"])
def test_prefetch_failure_rng(target: str) -> None:
    random.seed(42)
    before = random.getstate()
    with patch.object(fc, target, side_effect=RuntimeError("Injected prefetch failure")):
        assert fc._prefetch_protein_mols(["ALA"] * 32) is None
    assert random.getstate() == before


@pytest.mark.parametrize("sequence", ["A" * 7, "A" * 31 + "J"])
def test_serial_gate(sequence: str) -> None:
    with patch.object(fc, "_prefetch_protein_mols") as prefetch:
        fc._build_structure_from_polymers([{"sequence": sequence, "chain_id": "A", "polymer_type": "protein"}])
    prefetch.assert_not_called()


def test_external_draw_survives() -> None:
    codes = ["ALA", "GLY"] * 16
    entered, release = Event(), Event()
    build = fc._build_residue_rdkit_mol

    def paused_build(
        ccd_code: str,
        random_seed: int | None = None,
        *,
        retry_seed: Callable[[], int] = fc._draw_conformer_seed,
    ) -> tuple[Chem.Mol | None, np.ndarray]:
        if retry_seed is fc._defer_conformer_retry:
            entered.set()
            assert release.wait(timeout=10)
        return build(ccd_code, random_seed, retry_seed=retry_seed)

    random.seed(42)
    external_seed = random.randint(0, 10**9)
    expected = [legacy_residue(code) for code in codes]
    expected_state = random.getstate()
    random.seed(42)
    with patch.object(fc, "_build_residue_rdkit_mol", side_effect=paused_build):
        with patch.object(random, "setstate", side_effect=AssertionError("Global RNG rewind")):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fc._prefetch_protein_mols, codes)
                try:
                    assert entered.wait(timeout=10)
                    assert random.randint(0, 10**9) == external_seed
                finally:
                    release.set()
                actual = future.result(timeout=30)
    assert actual is not None
    assert_mols_equal(actual, expected)
    assert random.getstate() == expected_state


def test_concurrent_rng_draws() -> None:
    codes = ["ALA", "GLY"] * 16
    random.seed(42)
    for _ in range(2 * len(codes)):
        random.randint(0, 10**9)
    expected_state = random.getstate()
    random.seed(42)
    with patch.object(random, "setstate", side_effect=AssertionError("Global RNG rewind")):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(fc._prefetch_protein_mols, [codes, codes]))
    assert all(result is not None for result in results)
    assert random.getstate() == expected_state
