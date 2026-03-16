# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Structure from InputParsed and CCD."""

from __future__ import annotations

import logging

import numpy as np

from .const import (Atom, Bond, Chain, EnsembleDtype, Residue, Structure,
                    chain_type_ids, chirality_type_ids, prot_letter_to_token,
                    ref_atoms, res_to_center_atom_id, res_to_disto_atom_id,
                    token_ids, unk_chirality_type)

logger = logging.getLogger(__name__)


def _get_conformer(mol):
    """Return first conformer: prefer 'Computed' or 'Ideal', else conformer 0."""
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:
            pass
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:
            pass
    if mol.GetNumConformers() > 0:
        return mol.GetConformer(0)
    raise ValueError("No conformer in molecule")


def _parse_protein_residue(res_name: str,
                           ccd: dict) -> tuple[list[Atom], int, int]:
    """
    Parse a single standard protein residue from CCD.
    Returns (atoms, atom_center_offset, atom_disto_offset) for this residue's atoms.
    """
    from rdkit.Chem import AllChem

    unk_chirality = chirality_type_ids[unk_chirality_type]
    ref_mol = ccd.get(res_name)
    if ref_mol is None:
        raise ValueError(f"CCD missing residue: {res_name}")
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    conformer = _get_conformer(ref_mol)
    ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
    atom_names = ref_atoms[res_name]
    atoms = []
    for atom_name in atom_names:
        ref_atom = ref_name_to_atom[atom_name]
        idx = ref_atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)
        ref_coords = (float(pos.x), float(pos.y), float(pos.z))
        chirality = chirality_type_ids.get(str(ref_atom.GetChiralTag()),
                                           unk_chirality)
        atoms.append(
            Atom(
                name=atom_name,
                element=ref_atom.GetAtomicNum(),
                charge=ref_atom.GetFormalCharge(),
                coords=(0.0, 0.0, 0.0),
                conformer=ref_coords,
                is_present=True,
                chirality=chirality,
            ))
    center_id = res_to_center_atom_id.get(res_name, 0)
    disto_id = res_to_disto_atom_id.get(res_name, 0)
    return atoms, center_id, disto_id


def build_structure_from_input(
    input_parsed: dict,
    ccd: dict,
) -> Structure:
    """Build a Structure from InputParsed and CCD.

    Supports protein monomers and multimers only (no ligands/templates).

    Args:
        input_parsed: Parsed input dict containing ``"polymers"`` list.
        ccd: CCD dictionary mapping residue names to RDKit molecules.

    Returns:
        Fully populated :class:`Structure`.
    """
    polymers = input_parsed.get("polymers") or []
    if not polymers:
        raise ValueError("No polymers in input")

    # Group by (polymer_type, sequence) for entity_id
    entity_keys: list[tuple[str, str]] = []
    seen = {}
    for p in polymers:
        pt = (p.get("polymer_type") or "protein").lower()
        seq = p.get("sequence") or ""
        key = (pt, seq)
        if key not in seen:
            seen[key] = len(entity_keys)
            entity_keys.append(key)
        p["_entity_id"] = seen[key]

    all_atoms: list[Atom] = []
    all_residues: list[Residue] = []
    all_chains: list[Chain] = []
    all_bonds: list[Bond] = []
    sym_count: dict[int, int] = {}
    global_atom_idx = 0
    global_res_idx = 0
    chain_idx = 0

    for poly in polymers:
        polymer_type = (poly.get("polymer_type") or "protein").lower()
        if polymer_type != "protein":
            raise ValueError("Only protein polymers are supported")
        sequence = poly.get("sequence") or ""
        chain_ids = poly.get("chain_id")
        if chain_ids is None:
            chain_ids = ["A"]
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        entity_id = poly["_entity_id"]
        mol_type = chain_type_ids["PROTEIN"]
        seq_tokens = [prot_letter_to_token.get(c, "UNK") for c in sequence]
        # One chain per chain_id
        for ch_name in chain_ids:
            sym_id = sym_count.get(entity_id, 0)
            sym_count[entity_id] = sym_id + 1
            chain_atom_start = global_atom_idx
            chain_res_start = global_res_idx
            chain_atom_count = 0
            chain_res_count = 0
            for res_idx_in_chain, res_name in enumerate(seq_tokens):
                if res_name not in ref_atoms or not ref_atoms[res_name]:
                    raise ValueError(f"Unsupported residue: {res_name}")
                atoms, center_off, disto_off = _parse_protein_residue(
                    res_name, ccd)
                res_type = token_ids.get(res_name, token_ids["UNK"])
                atom_center_global = global_atom_idx + center_off
                atom_disto_global = global_atom_idx + disto_off
                # Per-chain 0-based residue index (matches Boltz2 OSS / CCD npz pipeline)
                all_residues.append(
                    Residue(
                        name=res_name,
                        res_type=res_type,
                        res_idx=res_idx_in_chain,
                        atom_idx=global_atom_idx,
                        atom_num=len(atoms),
                        atom_center=atom_center_global,
                        atom_disto=atom_disto_global,
                        is_standard=True,
                        is_present=True,
                    ))
                for a in atoms:
                    all_atoms.append(a)
                global_atom_idx += len(atoms)
                global_res_idx += 1
                chain_atom_count += len(atoms)
                chain_res_count += 1
            all_chains.append(
                Chain(
                    name=ch_name,
                    mol_type=mol_type,
                    entity_id=entity_id,
                    sym_id=sym_id,
                    asym_id=chain_idx,
                    atom_idx=chain_atom_start,
                    atom_num=chain_atom_count,
                    res_idx=chain_res_start,
                    res_num=chain_res_count,
                    cyclic_period=0,
                ))
            chain_idx += 1

    n_atoms = len(all_atoms)
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    # OSS schema uses atoms["coords"] for StructureV2 (=(0,0,0) in parse_polymer); conformer is ref only.
    for i, a in enumerate(all_atoms):
        coords[i, 0], coords[i, 1], coords[i, 2] = a.coords
    n_chains = len(all_chains)
    mask = np.ones(n_chains, dtype=bool)
    # OSS-compatible: one conformer starting at index 0 with n_atoms (same as types.py StructureV2)
    ensemble = np.array([(0, n_atoms)], dtype=EnsembleDtype)
    bfactor = np.zeros(n_atoms, dtype=np.float32)
    plddt = np.ones(
        n_atoms, dtype=np.float32
    )  # OSS schema sets plddt=1.0 for sequence-only (schema.py boltz_2)

    return Structure(
        atoms=all_atoms,
        bonds=all_bonds,
        residues=all_residues,
        chains=all_chains,
        coords=coords,
        ensemble=ensemble,
        mask=mask,
        bfactor=bfactor,
        plddt=plddt,
    )
