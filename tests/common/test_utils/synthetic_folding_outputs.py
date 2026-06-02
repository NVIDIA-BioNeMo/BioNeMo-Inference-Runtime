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
"""Synthetic ``FoldingOutput`` builders covering RNA / DNA / ligand chains.

The on-disk ``sample_folding_output.npy`` only covers a protein structure
(OpenFold2 era). These builders produce small, hand-crafted FoldingOutput
dicts that exercise every chain-kind code path in ``CIFWriter`` /
``PDBWriter``:

* pure RNA chain (restypes RA/RG/RC/RU; nucleic backbone atoms)
* pure DNA chain (restypes DA/DG/DC/DT)
* per-atom-tokenised non-polymer chain (every residue ``X``)
* multi-polymer combos (protein + RNA + DNA + nonpoly in one structure)

Tests can opt into ``residue_names`` / ``mol_types`` to exercise the
"producer supplied CCD codes" path, or leave them out to exercise the
``_classify_chain`` heuristic that the writers fall back on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tensorrt_bionemo.data.schemas.basic import FoldingOutput
from tensorrt_bionemo.data.utils import get_all_atom_types, get_all_residue_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _of3_indices() -> tuple[dict[str, int], dict[str, int]]:
    """Build (restype-name → index, atom-name → index) lookups for openfold3."""
    res_types = get_all_residue_types("openfold3")
    atom_types = get_all_atom_types("openfold3")
    return (
        {r.name: i for i, r in enumerate(res_types)},
        {a.name: i for i, a in enumerate(atom_types)},
    )


def of3_mappings() -> tuple[dict[int, object], dict[int, object]]:
    """Return ``(res_type_mapping, atom_type_mapping)`` for openfold3."""
    res_types = get_all_residue_types("openfold3")
    atom_types = get_all_atom_types("openfold3")
    return (
        {i: r for i, r in enumerate(res_types)},
        {i: a for i, a in enumerate(atom_types)},
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_folding(
    *,
    res_indices: list[int],
    residue_indices: list[int],
    chain_indices: list[int],
    per_residue_atom_names: list[list[str]],
    residue_names: Optional[list[str]] = None,
    mol_types: Optional[list[int]] = None,
    coord_offset: float = 0.0,
) -> FoldingOutput:
    """Low-level synthetic FoldingOutput constructor.

    Each residue gets a sparse atom_mask covering only the atom names
    listed in ``per_residue_atom_names[i]``; positions are deterministic
    so test assertions can find them.
    """
    res_idx_map, atom_idx_map = _of3_indices()
    n_tokens = len(res_indices)
    n_atom_types = len(atom_idx_map)
    atom_positions = np.zeros((n_tokens, n_atom_types, 3), dtype=np.float32)
    atom_mask = np.zeros((n_tokens, n_atom_types), dtype=np.float32)
    b_factors = np.zeros((n_tokens, n_atom_types), dtype=np.float32)
    for i, names in enumerate(per_residue_atom_names):
        for j, name in enumerate(names):
            slot = atom_idx_map[name]
            atom_positions[i, slot] = [
                coord_offset + i + j * 0.1,
                coord_offset + i * 1.5 + j * 0.2,
                coord_offset + i * 2.0 + j * 0.3,
            ]
            atom_mask[i, slot] = 1.0
            b_factors[i, slot] = 50.0 + i + j

    kwargs = dict(
        atom_positions=atom_positions,
        residue_types=np.asarray(res_indices, dtype=np.int64),
        atom_mask=atom_mask,
        residue_indices=np.asarray(residue_indices, dtype=np.int64),
        b_factors=b_factors,
        chain_indices=np.asarray(chain_indices, dtype=np.int64),
    )
    if residue_names is not None:
        kwargs["residue_names"] = residue_names
    if mol_types is not None:
        kwargs["mol_types"] = np.asarray(mol_types, dtype=np.int64)
    return FoldingOutput(**kwargs)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def rna_only_folding(
    *,
    sequence: str = "AGCU",
    chain_index: int = 0,
    with_residue_names: bool = False,
    with_mol_types: bool = False,
) -> FoldingOutput:
    """RNA-only chain with ``len(sequence)`` residues using nucleic backbone atoms.

    When ``with_residue_names=True`` we populate the PDB/CIF-standard
    1-letter RNA codes (``A``/``G``/``C``/``U``) — that's what real
    producers like OF3's postprocessor emit, sourced from biotite CCD
    via ``token_resnames``. The internal ``RA``/``RG``/... short codes
    are reserved for the res_types enum; if a producer ever sets
    residue_names to those instead, the writer faithfully emits them
    verbatim (the ``_IHM_REMAP`` shorthand is only consulted on the
    fallback path where residue_names is absent).
    """
    res_idx_map, _ = _of3_indices()
    short_to_restype = {"A": "RA", "G": "RG", "C": "RC", "U": "RU"}
    res_indices = [res_idx_map[short_to_restype[s]] for s in sequence]
    per_res_atoms = [["P", "C5'", "C4'", "C3'", "O3'", "C1'"]] * len(sequence)
    # PDB-standard 1-letter RNA residue names (NOT the internal "RA" short).
    residue_names = list(sequence) if with_residue_names else None
    mol_types = [1] * len(sequence) if with_mol_types else None
    return _build_folding(
        res_indices=res_indices,
        residue_indices=list(range(1, len(sequence) + 1)),
        chain_indices=[chain_index] * len(sequence),
        per_residue_atom_names=per_res_atoms,
        residue_names=residue_names,
        mol_types=mol_types,
    )


def dna_only_folding(
    *,
    sequence: str = "ACGT",
    chain_index: int = 0,
    with_residue_names: bool = False,
    with_mol_types: bool = False,
) -> FoldingOutput:
    """DNA-only chain. Restypes DA/DG/DC/DT."""
    res_idx_map, _ = _of3_indices()
    short_to_index = {"A": "DA", "G": "DG", "C": "DC", "T": "DT"}
    res_indices = [res_idx_map[short_to_index[s]] for s in sequence]
    per_res_atoms = [["P", "C5'", "C4'", "C3'", "O3'", "C1'"]] * len(sequence)
    residue_names = (
        [short_to_index[s] for s in sequence] if with_residue_names else None
    )
    mol_types = [2] * len(sequence) if with_mol_types else None
    return _build_folding(
        res_indices=res_indices,
        residue_indices=list(range(1, len(sequence) + 1)),
        chain_indices=[chain_index] * len(sequence),
        per_residue_atom_names=per_res_atoms,
        residue_names=residue_names,
        mol_types=mol_types,
    )


def nonpoly_ligand_folding(
    *,
    atom_names: list[str] = ("C1", "C2", "N2"),
    chain_index: int = 0,
    ccd_code: Optional[str] = None,
    with_mol_types: bool = False,
) -> FoldingOutput:
    """Per-atom-tokenised non-polymer chain. Every residue is ``X`` (idx 20).

    When ``ccd_code`` is given, every token's ``residue_name`` is set to
    that code (e.g. ``NAG``); otherwise ``residue_names`` is omitted and
    the writer falls back to the all-``X`` → ``nonpoly`` heuristic.
    """
    res_idx_map, _ = _of3_indices()
    n = len(atom_names)
    res_indices = [res_idx_map["X"]] * n
    # Each ligand atom-token has exactly one atom of its own name.
    per_res_atoms = [[name] for name in atom_names]
    residue_names = [ccd_code] * n if ccd_code is not None else None
    mol_types = [3] * n if with_mol_types else None
    return _build_folding(
        res_indices=res_indices,
        residue_indices=[1] * n,
        chain_indices=[chain_index] * n,
        per_residue_atom_names=per_res_atoms,
        residue_names=residue_names,
        mol_types=mol_types,
    )


def multi_polymer_folding(
    *,
    with_residue_names: bool = True,
    with_mol_types: bool = True,
) -> FoldingOutput:
    """Multi-chain structure exercising every chain-kind in one FoldingOutput.

    Layout:
      * chain 0: 2-residue protein (ALA, TYR)
      * chain 1: 3-residue RNA (A, G, C)
      * chain 2: 3-residue DNA (DA, DG, DC)
      * chain 3: 2-atom non-polymer ligand (NAG)
    """
    res_idx_map, _ = _of3_indices()
    # tokens, in order
    rec = [
        # (restype, residue_idx, chain_idx, atom_names, residue_name, mol_type)
        (res_idx_map["A"], 1, 0, ["N", "CA", "C", "O", "CB"], "ALA", 0),
        (res_idx_map["Y"], 2, 0, ["N", "CA", "C", "O", "CB"], "TYR", 0),
        (res_idx_map["RA"], 1, 1, ["P", "C5'", "C4'", "C1'"], "A", 1),
        (res_idx_map["RG"], 2, 1, ["P", "C5'", "C4'", "C1'"], "G", 1),
        (res_idx_map["RC"], 3, 1, ["P", "C5'", "C4'", "C1'"], "C", 1),
        (res_idx_map["DA"], 1, 2, ["P", "C5'", "C4'", "C1'"], "DA", 2),
        (res_idx_map["DG"], 2, 2, ["P", "C5'", "C4'", "C1'"], "DG", 2),
        (res_idx_map["DC"], 3, 2, ["P", "C5'", "C4'", "C1'"], "DC", 2),
        (res_idx_map["X"], 1, 3, ["C1"], "NAG", 3),
        (res_idx_map["X"], 1, 3, ["N2"], "NAG", 3),
    ]
    return _build_folding(
        res_indices=[r[0] for r in rec],
        residue_indices=[r[1] for r in rec],
        chain_indices=[r[2] for r in rec],
        per_residue_atom_names=[r[3] for r in rec],
        residue_names=[r[4] for r in rec] if with_residue_names else None,
        mol_types=[r[5] for r in rec] if with_mol_types else None,
    )


def many_chains_folding(n_chains: int) -> FoldingOutput:
    """``n_chains`` single-residue protein chains, all alanine.

    Used to exercise the chain-index → asym-id mapping for both writers
    (CIF supports unlimited chains via multi-char asym_id; PDB caps at 62).
    """
    res_idx_map, _ = _of3_indices()
    return _build_folding(
        res_indices=[res_idx_map["A"]] * n_chains,
        residue_indices=[1] * n_chains,
        chain_indices=list(range(n_chains)),
        per_residue_atom_names=[["N", "CA", "C", "O", "CB"]] * n_chains,
    )
