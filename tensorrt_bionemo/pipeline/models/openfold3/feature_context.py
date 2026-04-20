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
"""OpenFold3 ContextGenerator: parse InputParsed into structural context.

Builds per-token and per-atom structural data from protein sequences using
Biotite CCD for accurate atom arrays, parses MSA files, and stores
everything in a context dict for downstream feature generators.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import numpy as np

import biotite.structure as struc
import biotite.structure.info as struc_info
from biotite.interface.rdkit import to_mol as biotite_to_mol
from rdkit import Chem
from rdkit.Chem import AllChem

from tensorrt_bionemo.data.schemas.basic import InputParsed
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase

from .const import (
    GAP_IDX,
    MSA_CHAR_TO_IDX,
    POLYMER_TYPE_TO_MOL_TYPE,
    UNK_IDX,
    _PROTEIN_1TO3,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biotite CCD helpers (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=500)
def _get_residue_from_ccd(ccd_code: str) -> struc.AtomArray:
    """Get a residue AtomArray from Biotite CCD, heavy atoms only, no OXT."""
    res = struc_info.residue(ccd_code)
    res = res[res.element != "H"]
    res = res[res.atom_name != "OXT"]
    return res


@lru_cache(maxsize=500)
def _get_residue_from_ccd_with_oxt(ccd_code: str) -> struc.AtomArray:
    """Get a residue AtomArray with OXT kept (for RDKit conformer gen)."""
    res = struc_info.residue(ccd_code)
    res = res[res.element != "H"]
    return res


def _build_residue_rdkit_mol(
    ccd_code: str,
) -> tuple[Optional[Chem.Mol], np.ndarray]:
    """Convert a CCD residue to an RDKit Mol with 3D conformer.

    Matches OSS: keeps OXT in the mol for correct bond topology during
    conformer generation, then returns a mask indicating which atoms to
    include in the final features (OXT masked out).

    Returns:
        (mol, in_crop_mask): mol with conformer, boolean mask over mol atoms.
        in_crop_mask[i] = True for atoms to keep (non-OXT).
    """
    import random as _random

    # Use the full residue (with OXT) for conformer generation
    res_full = _get_residue_from_ccd_with_oxt(ccd_code)

    try:
        # Use biotite.interface.rdkit.to_mol (matches OSS exactly)
        mol = biotite_to_mol(res_full, kekulize=True)
        Chem.SanitizeMol(mol)
        # Remove CCD conformer (OSS: mol.RemoveConformer(0))
        mol.RemoveConformer(0)
        # Generate fresh conformer using ETKDGv3
        # OSS uses random.randint(0, 10**9) as seed
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = _random.randint(0, 10**9)
        params.clearConfs = False
        conf_id = AllChem.EmbedMolecule(mol_h, params)
        if conf_id == -1:
            params.useRandomCoords = True
            params.randomSeed = _random.randint(0, 10**9)
            conf_id = AllChem.EmbedMolecule(mol_h, params)
        mol_h = Chem.RemoveHs(mol_h)

        # Build mask: exclude OXT
        in_crop_mask = np.array(
            [res_full.atom_name[i] != "OXT"
             for i in range(len(res_full))], dtype=bool)

        return mol_h, in_crop_mask
    except Exception as e:
        _logger.debug("Failed to build RDKit mol for %s: %s", ccd_code, e)
        return None, np.array([], dtype=bool)


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------

def _build_structure_from_polymers(
    polymers: list[dict],
) -> dict[str, Any]:
    """Build atom-level and token-level structure data from polymer list
    using Biotite CCD for accurate atom arrays.

    For protein-only inputs, each standard amino acid residue = one token.
    Atom names, elements, and counts come directly from the CCD.
    """
    token_resnames: list[str] = []
    token_chain_ids: list[str] = []
    token_entity_ids: list[int] = []
    token_mol_types: list[int] = []
    token_res_ids: list[int] = []
    atom_names: list[str] = []
    atom_elements: list[str] = []
    atom_token_idx: list[int] = []
    atoms_per_token: list[int] = []
    token_start_atoms: list[int] = []

    # Per-residue RDKit mols for conformer generation
    residue_mols: list[Optional[Chem.Mol]] = []
    residue_crop_masks: list[np.ndarray] = []
    residue_atom_charges: list[list[int]] = []

    token_idx = 0
    atom_idx = 0

    # Assign entity IDs: same sequence → same entity
    seq_to_entity: dict[str, int] = {}
    entity_counter = 1

    for poly in polymers:
        sequence = poly.get("sequence", "")
        chain_id = poly.get("chain_id")
        polymer_type = poly.get("polymer_type", "protein")
        mol_type = POLYMER_TYPE_TO_MOL_TYPE.get(polymer_type, 0)

        if isinstance(chain_id, list):
            chain_ids = chain_id
        elif chain_id is not None:
            chain_ids = [chain_id]
        else:
            chain_ids = ["A"]

        if sequence not in seq_to_entity:
            seq_to_entity[sequence] = entity_counter
            entity_counter += 1
        entity_id = seq_to_entity[sequence]

        for cid in chain_ids:
            for res_idx, aa_char in enumerate(sequence):
                resname_3 = _PROTEIN_1TO3.get(aa_char, "UNK")

                # Get atom array from Biotite CCD
                try:
                    ccd_res = _get_residue_from_ccd(resname_3)
                    res_atom_names = list(ccd_res.atom_name)
                    res_elements = list(ccd_res.element)
                except Exception as e:
                    # Fallback for unknown residues
                    _logger.debug(
                        "CCD lookup failed for %s, using backbone fallback: %s",
                        resname_3, e,
                    )
                    res_atom_names = ["N", "CA", "C", "O"]
                    res_elements = ["N", "C", "C", "O"]

                # Build RDKit mol for conformer generation
                # Keep OXT in mol for proper topology; mask it out later
                mol, in_crop_mask = _build_residue_rdkit_mol(resname_3)
                residue_mols.append(mol)
                residue_crop_masks.append(in_crop_mask)

                # Extract formal charges from RDKit mol (only for kept atoms)
                charges = []
                if mol is not None:
                    for ai, atom in enumerate(mol.GetAtoms()):
                        if ai < len(in_crop_mask) and in_crop_mask[ai]:
                            charges.append(atom.GetFormalCharge())
                if len(charges) != len(res_atom_names):
                    charges = [0] * len(res_atom_names)
                residue_atom_charges.append(charges)

                token_resnames.append(resname_3)
                token_chain_ids.append(cid)
                token_entity_ids.append(entity_id)
                token_mol_types.append(mol_type)
                token_res_ids.append(res_idx + 1)
                token_start_atoms.append(atom_idx)

                n_atoms_token = len(res_atom_names)
                atoms_per_token.append(n_atoms_token)

                for aname, elem in zip(res_atom_names, res_elements,
                                       strict=True):
                    atom_names.append(aname)
                    atom_elements.append(elem)
                    atom_token_idx.append(token_idx)
                    atom_idx += 1

                token_idx += 1

    return {
        "token_resnames": token_resnames,
        "token_chain_ids": token_chain_ids,
        "token_entity_ids": token_entity_ids,
        "token_mol_types": token_mol_types,
        "token_res_ids": token_res_ids,
        "atom_names": atom_names,
        "atom_elements": atom_elements,
        "atom_token_idx": atom_token_idx,
        "n_tokens": token_idx,
        "n_atoms": atom_idx,
        "atoms_per_token": atoms_per_token,
        "token_start_atoms": token_start_atoms,
        "residue_mols": residue_mols,
        "residue_crop_masks": residue_crop_masks,
        "residue_atom_charges": residue_atom_charges,
    }


def _compute_sym_ids(
    entity_ids: list[int],
    chain_ids: list[str],
) -> list[int]:
    """Compute symmetry IDs: enumerate chains within each entity."""
    entity_chain_counter: dict[int, dict[str, int]] = {}
    sym_ids = []
    for eid, cid in zip(entity_ids, chain_ids, strict=True):
        if eid not in entity_chain_counter:
            entity_chain_counter[eid] = {}
        chain_map = entity_chain_counter[eid]
        if cid not in chain_map:
            chain_map[cid] = len(chain_map) + 1
        sym_ids.append(chain_map[cid])
    return sym_ids


def _renumber_chain_ids(chain_ids: list[str]) -> list[int]:
    """Renumber chain IDs to 1-based integers."""
    chain_to_num: dict[str, int] = {}
    result = []
    for cid in chain_ids:
        if cid not in chain_to_num:
            chain_to_num[cid] = len(chain_to_num) + 1
        result.append(chain_to_num[cid])
    return result


def _parse_msa_entry(msas: list | None) -> dict | None:
    """Load and parse a single MSA entry."""
    if not msas:
        return None
    first = msas[0]
    if first is None:
        return None
    if first.get("sequences") or first.get("raw"):
        return first
    content = first.get("content") if isinstance(first, dict) else None
    if content is None and isinstance(first, dict) and first.get("path"):
        path = first["path"]
        try:
            with open(path, "r") as f:
                content = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            _logger.warning("Failed to read MSA file %s: %s", path, e)
            return None
    if content:
        from io import StringIO
        from tensorrt_bionemo.data.parsers.a3m import parse_a3m_content
        return parse_a3m_content(StringIO(content))
    return None


class OpenFold3ContextGenerator(ContextGeneratorBase):
    """Build OpenFold3 context dict from InputParsed using Biotite CCD."""

    def __init__(
        self,
        config: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(config, metadata)
        self._required_kwargs = ["parsed"]

    @property
    def required_kwargs(self) -> list[str]:
        return self._required_kwargs

    @required_kwargs.setter
    def required_kwargs(self, value: list[str]) -> None:
        self._required_kwargs = value

    def __call__(self, parsed: InputParsed) -> dict[str, Any]:
        polymers = parsed.get("polymers")
        if not polymers:
            raise ValueError("No polymers in input")

        struct = _build_structure_from_polymers(polymers)

        msa_per_chain: list[dict | None] = []
        paired_msa_per_chain: list[dict | None] = []
        for poly in polymers:
            chain_id = poly.get("chain_id")
            n_chains = (len(chain_id) if isinstance(chain_id, list) else 1)
            msa_entry = _parse_msa_entry(poly.get("msas"))
            paired_entry = _parse_msa_entry(poly.get("paired_msas"))
            for _ in range(n_chains):
                msa_per_chain.append(msa_entry)
                paired_msa_per_chain.append(paired_entry)

        chain_sequences: list[str] = []
        for poly in polymers:
            sequence = poly.get("sequence", "")
            chain_id = poly.get("chain_id")
            n_chains = (len(chain_id) if isinstance(chain_id, list) else 1)
            for _ in range(n_chains):
                chain_sequences.append(sequence)

        return {
            "structure": struct,
            "msa_per_chain": msa_per_chain,
            "paired_msa_per_chain": paired_msa_per_chain,
            "chain_sequences": chain_sequences,
        }
