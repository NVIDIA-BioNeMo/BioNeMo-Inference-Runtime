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
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedParameters

from bionemo_ir.pipeline.models.openfold3 import feature_context as fc

_POLYMERS = [
    {"sequence": "ACDEFGHIKLMNPQRSTVWY", "chain_id": ["A", "B"], "polymer_type": "protein"},
    {"sequence": "ACGU" * 3, "chain_id": "C", "polymer_type": "rna"},
    {"sequence": "ACGT" * 9, "chain_id": "D", "polymer_type": "dna"},
    {"sequence": "SAH", "chain_id": "E", "polymer_type": "ccd_ligand"},
    {"sequence": "ZN", "chain_id": "F", "polymer_type": "ccd_ligand"},
    {"sequence": "GA", "chain_id": "G", "polymer_type": "protein"},
]


@pytest.fixture(autouse=True)
def preserve_rng() -> Iterator[None]:
    state = random.getstate()
    yield
    random.setstate(state)


def fresh_structure(seed: int) -> tuple[dict, tuple]:
    random.seed(seed)
    with patch.object(fc._CONFORMERS, "maxsize", 0):
        fc._CONFORMERS.clear()
        structure = fc._build_structure_from_polymers(_POLYMERS)
    return structure, random.getstate()


def cached_structure(seed: int) -> tuple[dict, tuple]:
    random.seed(seed)
    structure = fc._build_structure_from_polymers(_POLYMERS)
    return structure, random.getstate()


def assert_structures_equal(actual: dict, expected: dict) -> None:
    actual, expected = dict(actual), dict(expected)
    flags = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    for mol, reference in zip(actual.pop("residue_mols"), expected.pop("residue_mols"), strict=True):
        assert mol.ToBinary(flags) == reference.ToBinary(flags)
    for mask, reference in zip(actual.pop("residue_crop_masks"), expected.pop("residue_crop_masks"), strict=True):
        np.testing.assert_array_equal(mask, reference)
    assert actual == expected


@pytest.mark.parametrize("seed", [0, 42])
def test_cached_requests_match_fresh_embeddings(seed: int) -> None:
    expected, expected_state = fresh_structure(seed)
    first, first_state = cached_structure(seed)
    assert len(fc._CONFORMERS) > 0
    with patch.object(fc.AllChem, "EmbedMolecule", side_effect=AssertionError("cache miss")):
        second, second_state = cached_structure(seed)
    assert first_state == second_state == expected_state
    assert_structures_equal(first, expected)
    assert_structures_equal(second, expected)


def test_cached_request_skips_the_conformer_pool() -> None:
    expected, expected_state = cached_structure(42)
    with patch.object(fc, "ThreadPoolExecutor", side_effect=AssertionError("pool used")):
        actual, actual_state = cached_structure(42)
    assert actual_state == expected_state
    assert_structures_equal(actual, expected)


def test_pool_workers_leave_the_cache_to_the_caller() -> None:
    caller = threading.get_ident()
    put_many = fc._CONFORMERS.put_many

    def from_caller(items: list) -> None:
        assert threading.get_ident() == caller
        put_many(items)

    with patch.object(fc._CONFORMERS, "put_many", side_effect=from_caller):
        results = fc._build_parallel("residue", fc._build_residue_rdkit_mol, ["ALA", "TRP", "ALA"], [1, 2, 1])
    assert all(mol is not None for mol, _ in results)
    assert len(fc._CONFORMERS) == 2


@contextmanager
def counted_embeddings() -> Iterator[list[EmbedParameters]]:
    embed = fc.AllChem.EmbedMolecule
    calls: list[EmbedParameters] = []

    def counting(mol: Chem.Mol, params: EmbedParameters) -> int:
        calls.append(params)
        return embed(mol, params)

    with patch.object(fc.AllChem, "EmbedMolecule", side_effect=counting):
        yield calls


def test_other_seed_misses_and_matches_fresh() -> None:
    expected, expected_state = fresh_structure(7)
    cached_structure(42)
    warmed = len(fc._CONFORMERS)
    with counted_embeddings() as calls:
        actual, actual_state = cached_structure(7)
    assert calls
    assert len(fc._CONFORMERS) > warmed
    assert actual_state == expected_state
    assert_structures_equal(actual, expected)
    with counted_embeddings() as calls:
        cached_structure(42)
    assert not calls


def test_key_covers_seed_code_and_kind() -> None:
    fingerprint = fc._etkdg_fingerprint()
    key = fc._conformer_key("residue", "ALA", 1, fingerprint)
    assert key != fc._conformer_key("residue", "ALA", 2, fingerprint)
    assert key != fc._conformer_key("residue", "GLY", 1, fingerprint)
    assert key != fc._conformer_key("nucleotide", "ALA", 1, fingerprint)


def _scalar_settings() -> list[str]:
    params = EmbedParameters()
    return [
        name
        for name in dir(params)
        if not name.startswith("_") and name != "randomSeed" and isinstance(getattr(params, name), (bool, int, float))
    ]


def _changed(value: bool | int | float) -> bool | int | float:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return value + 0.5


@pytest.mark.parametrize("name", _scalar_settings())
def test_every_embedding_setting_change_misses(name: str) -> None:
    factory = fc._etkdg_params

    def changed(random_seed: int) -> EmbedParameters:
        params = factory(random_seed)
        setattr(params, name, _changed(getattr(params, name)))
        return params

    fc._build_parallel("residue", fc._build_residue_rdkit_mol, ["ALA"], [5])
    fc._build_residue_rdkit_mol("GLY", 5)
    with counted_embeddings() as calls:
        fc._build_parallel("residue", fc._build_residue_rdkit_mol, ["ALA"], [5])
        fc._build_residue_rdkit_mol("GLY", 5)
    assert not calls
    with patch.object(fc, "_etkdg_params", side_effect=changed), counted_embeddings() as calls:
        fc._build_parallel("residue", fc._build_residue_rdkit_mol, ["ALA"], [5])
        fc._build_residue_rdkit_mol("GLY", 5)
    assert len(calls) >= 2
    assert all(getattr(params, name) != getattr(factory(5), name) for params in calls)


def test_structure_reads_the_settings_once_and_misses_on_a_change() -> None:
    factory = fc._etkdg_params

    def changed(random_seed: int) -> EmbedParameters:
        params = factory(random_seed)
        params.useRandomCoords = not params.useRandomCoords
        return params

    cached_structure(42)
    with patch.object(fc, "_etkdg_fingerprint", wraps=fc._etkdg_fingerprint) as fingerprint:
        with counted_embeddings() as calls:
            cached_structure(42)
    assert not calls
    assert fingerprint.call_count == 1
    with patch.object(fc, "_etkdg_params", side_effect=changed), counted_embeddings() as calls:
        cached_structure(42)
    assert calls
    assert all(params.useRandomCoords != factory(42).useRandomCoords for params in calls)


def test_parameter_change_embeds_with_the_new_settings() -> None:
    fc._build_residue_rdkit_mol("ALA", 5)

    def etkdg_v2(random_seed: int) -> EmbedParameters:
        params = fc.AllChem.ETKDGv2()
        params.randomSeed = random_seed
        params.clearConfs = False
        return params

    with patch.object(fc, "_etkdg_params", side_effect=etkdg_v2):
        mol, _ = fc._build_residue_rdkit_mol("ALA", 5)
    reference = Chem.AddHs(Chem.Mol(fc._residue_rdkit_topology("ALA")[0]))
    assert fc.AllChem.EmbedMolecule(reference, etkdg_v2(5)) == 0
    np.testing.assert_array_equal(
        mol.GetConformer().GetPositions(), Chem.RemoveHs(reference).GetConformer().GetPositions()
    )


def test_retry_result_is_not_cached() -> None:
    embed = fc.AllChem.EmbedMolecule

    def failing_primary(mol: Chem.Mol, params: EmbedParameters) -> int:
        return -1 if not params.useRandomCoords else embed(mol, params)

    with patch.object(fc.AllChem, "EmbedMolecule", side_effect=failing_primary):
        mol, _ = fc._build_residue_rdkit_mol("LYS", 3, retry_seed=lambda: 11)
    assert mol is not None and mol.GetNumConformers() == 1
    assert len(fc._CONFORMERS) == 0


def test_callers_get_copies() -> None:
    first, mask = fc._build_residue_rdkit_mol("TRP", 9)
    expected = first.GetConformer().GetPositions().copy()
    first.GetConformer().SetAtomPosition(0, (100.0, 200.0, 300.0))
    mask[:] = False
    second, other_mask = fc._build_residue_rdkit_mol("TRP", 9)
    np.testing.assert_array_equal(second.GetConformer().GetPositions(), expected)
    assert other_mask.any()


def embedded(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert fc.AllChem.EmbedMolecule(mol, fc._etkdg_params(seed)) == 0
    return Chem.RemoveHs(mol)


def test_lru_is_bounded() -> None:
    cache = fc._ConformerCache(maxsize=2)
    mask = np.ones(3, dtype=bool)
    for seed in (1, 2):
        cache.put(("residue", "SER", seed), embedded("OCC", seed), mask)
    assert cache.get(("residue", "SER", 1)) is not None
    cache.put(("residue", "SER", 3), embedded("OCC", 3), mask)
    assert len(cache) == 2
    assert cache.get(("residue", "SER", 2)) is None
    assert cache.get(("residue", "SER", 1)) is not None and cache.get(("residue", "SER", 3)) is not None


def test_hit_is_the_template_with_the_entry_coordinates() -> None:
    cache = fc._ConformerCache(maxsize=4)
    mask = np.ones(3, dtype=bool)
    first, second = embedded("OCC", 1), embedded("OCC", 2)
    cache.put(("residue", "SER", 1), first, mask)
    cache.put(("residue", "SER", 2), second, mask)
    flags = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    for seed, reference in ((1, first), (2, second)):
        mol, _ = cache.get(("residue", "SER", seed))
        assert mol.ToBinary(flags) == reference.ToBinary(flags)


def test_template_leaves_with_its_last_entry() -> None:
    cache = fc._ConformerCache(maxsize=2)
    cache.put(("residue", "SER", 1), embedded("OCC", 1), np.ones(3, dtype=bool))
    cache.put(("residue", "SER", 1), embedded("OCC", 1), np.ones(3, dtype=bool))
    cache.put(("residue", "GLY", 1), embedded("NCC(=O)O", 1), np.ones(5, dtype=bool))
    cache.put(("residue", "GLY", 2), embedded("NCC(=O)O", 2), np.ones(5, dtype=bool))
    assert set(cache._templates) == {("residue", "GLY")}
    cache.clear()
    assert len(cache) == 0 and not cache._templates


def test_concurrent_builds_match_serial() -> None:
    codes = ["ALA", "GLY", "TRP", "LYS"] * 16
    seeds = list(range(len(codes)))
    with patch.object(fc._CONFORMERS, "maxsize", 0):
        expected = [fc._build_residue_rdkit_mol(code, seed) for code, seed in zip(codes, seeds, strict=True)]
    with ThreadPoolExecutor(8) as executor:
        for _ in range(2):
            actual = list(executor.map(fc._build_residue_rdkit_mol, codes, seeds))
            for (mol, mask), (reference, reference_mask) in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(mask, reference_mask)
                assert mol.GetConformer().GetPositions().tobytes() == reference.GetConformer().GetPositions().tobytes()
    assert len(fc._CONFORMERS) == len(set(zip(codes, seeds, strict=True)))
