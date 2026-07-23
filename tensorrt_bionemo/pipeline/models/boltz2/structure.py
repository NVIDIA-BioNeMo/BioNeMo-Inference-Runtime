# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a :class:`Structure` from :class:`InputParsed`.

Supports the full basic-schema polymer set used by Boltz2:

* ``protein`` / ``rna`` / ``dna`` — standard polymer residues parsed from the
  CCD using the canonical atom ordering in :mod:`const.ref_atoms`.
* ``ccd_ligand`` — one or more CCD codes (joined with ``_``); each component
  becomes a non-standard residue inside a single ``NONPOLYMER`` chain.
* ``smiles_ligand`` — a SMILES string; RDKit is used to embed a 3D conformer
  and to enumerate heavy atoms; the resulting molecule is added to a per-call
  ``extra_mols`` cache so feature generators can look it up by name.

Intra-residue bonds for ligands come from RDKit; inter-residue (peptide /
phosphodiester) bonds are *not* added here — Boltz2 derives the connectivity
from positional embeddings, matching the OSS pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .const import (Atom, Bond, Chain, EnsembleDtype, Residue, Structure,
                    bond_type_ids, chain_type_ids, chirality_type_ids,
                    dna_letter_to_token, prot_letter_to_token, ref_atoms,
                    res_to_center_atom_id, res_to_disto_atom_id,
                    rna_letter_to_token, token_ids, unk_bond_type,
                    unk_chirality_type, unk_token_ids)

logger = logging.getLogger(__name__)

# Polymer types we route through ``_parse_polymer_residue``.
_POLYMER_TYPES = {"protein", "rna", "dna"}
_LIGAND_TYPES = {"ccd_ligand", "smiles_ligand"}


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


def _seq_to_tokens(polymer_type: str, sequence: str) -> list[str]:
    """Map a 1-letter polymer sequence to per-residue 3-letter (or 2-letter) CCD codes."""
    if polymer_type == "protein":
        unk = "UNK"
        mapping = prot_letter_to_token
    elif polymer_type == "rna":
        unk = "N"
        mapping = rna_letter_to_token
    elif polymer_type == "dna":
        unk = "DN"
        mapping = dna_letter_to_token
    else:
        raise ValueError(
            f"Unknown polymer type for sequence mapping: {polymer_type}")
    return [mapping.get(c, unk) for c in sequence]


def _parse_polymer_residue(res_name: str, ccd: dict) -> list[Atom]:
    """Parse a standard polymer residue using ``const.ref_atoms`` ordering."""
    from rdkit.Chem import AllChem

    unk_chirality = chirality_type_ids[unk_chirality_type]
    ref_mol = ccd.get(res_name)
    if ref_mol is None:
        raise ValueError(f"CCD missing residue: {res_name}")
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    conformer = _get_conformer(ref_mol)
    ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
    atom_names = ref_atoms[res_name]
    atoms: list[Atom] = []
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
    return atoms


def _compute_rdkit_bounds_constraints(mol, idx_map):
    """Compute RDKit distance-bounds constraints for a ligand residue.

    Returns a list of dicts: ``{atom_idxs, is_bond, is_angle, upper_bound, lower_bound}``
    using the surviving heavy-atom indices from ``idx_map``.
    """
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    if mol.GetNumAtoms() <= 1:
        return []
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    bounds = GetMoleculeBoundsMatrix(mol,
                                     set15bounds=True,
                                     scaleVDW=True,
                                     doTriangleSmoothing=True,
                                     useMacrocycle14config=False)
    bonds_set = {
        tuple(sorted(b))
        for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    }
    angles_set = {
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    }
    constraints = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
        i, j = int(i), int(j)
        if i in idx_map and j in idx_map:
            constraints.append({
                "atom_idxs": (idx_map[i], idx_map[j]),
                "is_bond": tuple(sorted([i, j])) in bonds_set,
                "is_angle": tuple(sorted([i, j])) in angles_set,
                "upper_bound": float(bounds[i, j]),
                "lower_bound": float(bounds[j, i]),
            })
    return constraints


def _compute_chiral_atom_constraints(mol, idx_map):
    """Compute per-chiral-center reference + permuted constraints (OSS algorithm)."""
    from rdkit import Chem
    from rdkit.Chem import HybridizationType

    constraints = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints
    for center_idx, orientation in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False):
        center = mol.GetAtomWithIdx(center_idx)
        neighbors = [(neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                     for neighbor in center.GetNeighbors()]
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        neighbors = tuple(n[0] for n in neighbors)
        is_r = orientation == "R"
        if len(neighbors) > 4 or center.GetHybridization(
        ) != HybridizationType.SP3:
            continue
        ref_idxs = (*neighbors[:3], center_idx)
        if all(i in idx_map for i in ref_idxs):
            constraints.append({
                "atom_idxs": tuple(idx_map[i] for i in ref_idxs),
                "is_reference": True,
                "is_r": is_r,
            })
        if len(neighbors) == 4:
            for skip_idx in range(3):
                chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1:]
                if skip_idx % 2 == 0:
                    atom_idxs = chiral_set[::-1] + (center_idx, )
                else:
                    atom_idxs = chiral_set + (center_idx, )
                if all(i in idx_map for i in atom_idxs):
                    constraints.append({
                        "atom_idxs":
                        tuple(idx_map[i] for i in atom_idxs),
                        "is_reference":
                        False,
                        "is_r":
                        is_r,
                    })
    return constraints


def _compute_stereo_bond_constraints(mol, idx_map):
    """Compute E/Z stereo-bond constraints (OSS algorithm).

    Each STEREOE/STEREOZ bond produces a quadruple
    ``(start_neighbor, start_atom, end_atom, end_neighbor)`` where the
    neighbors are CIP-ranked. A "check" constraint uses the highest-rank
    neighbors; a non-reference constraint uses the second-rank pair when
    both sides have exactly 2 neighbors.
    """
    from rdkit.Chem.rdchem import BondStereo

    constraints = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in (BondStereo.STEREOE, BondStereo.STEREOZ):
            continue
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        start_neighbors = [
            (n.GetIdx(), int(n.GetProp("_CIPRank")))
            for n in mol.GetAtomWithIdx(start_idx).GetNeighbors()
            if n.GetIdx() != end_idx
        ]
        start_neighbors = [
            n[0]
            for n in sorted(start_neighbors, key=lambda x: x[1], reverse=True)
        ]
        end_neighbors = [(n.GetIdx(), int(n.GetProp("_CIPRank")))
                         for n in mol.GetAtomWithIdx(end_idx).GetNeighbors()
                         if n.GetIdx() != start_idx]
        end_neighbors = [
            n[0]
            for n in sorted(end_neighbors, key=lambda x: x[1], reverse=True)
        ]
        is_e = stereo == BondStereo.STEREOE
        if not start_neighbors or not end_neighbors:
            continue

        ref_idxs = (start_neighbors[0], start_idx, end_idx, end_neighbors[0])
        if all(i in idx_map for i in ref_idxs):
            constraints.append({
                "atom_idxs": tuple(idx_map[i] for i in ref_idxs),
                "is_reference": True,
                "is_e": is_e,
            })
        if len(start_neighbors) == 2 and len(end_neighbors) == 2:
            alt = (start_neighbors[1], start_idx, end_idx, end_neighbors[1])
            if all(i in idx_map for i in alt):
                constraints.append({
                    "atom_idxs": tuple(idx_map[i] for i in alt),
                    "is_reference": False,
                    "is_e": is_e,
                })
    return constraints


_PLANAR_DOUBLE_BOND_SMARTS = "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)"
_AROMATIC_RING_5_SMARTS = "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1"
_AROMATIC_RING_6_SMARTS = "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"


def _compute_flatness_constraints(mol, idx_map):
    """Compute planar-bond + aromatic ring (5/6) constraints (OSS algorithm).

    Uses RDKit's substructure matcher with the same SMARTS patterns as
    OSS ``compute_flatness_constraints``.

    Prep: OSS ``compute_flatness_constraints`` runs the SMARTS matcher on the
    mol with its *shipped* aromaticity/hybridization state and does NOT
    re-perceive (no ``SetAromaticity`` / ``SetHybridization``). We match that:
    initialize ring info + valence cache only. Re-perceiving aromaticity here
    surfaces aromatic rings the OSS reference does not (e.g. an extra
    ``planar_ring_5`` on ccd.pkl ligands) and diverges from OSS. The
    consequence is that ``[ar5^2]`` / ``[ar6^2]`` only match where the mol
    already carries the perceived state — exactly OSS behavior.

    Returns three lists of dicts (planar bonds, aromatic 5-rings,
    aromatic 6-rings); each dict has ``atom_idxs`` mapped through
    ``idx_map`` so callers can shift them to global indices.
    """
    from rdkit import Chem

    try:
        Chem.GetSSSR(mol)
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        # Best effort — fall through to SMARTS matching with whatever
        # state the mol has. Worst case: 0 constraints, same as before.
        pass

    planar_bond_smarts = Chem.MolFromSmarts(_PLANAR_DOUBLE_BOND_SMARTS)
    ring5_smarts = Chem.MolFromSmarts(_AROMATIC_RING_5_SMARTS)
    ring6_smarts = Chem.MolFromSmarts(_AROMATIC_RING_6_SMARTS)

    planar_bonds: list[dict] = []
    rings5: list[dict] = []
    rings6: list[dict] = []
    for match in mol.GetSubstructMatches(planar_bond_smarts):
        if all(i in idx_map for i in match):
            planar_bonds.append(
                {"atom_idxs": tuple(idx_map[i] for i in match)})
    for match in mol.GetSubstructMatches(ring5_smarts):
        if all(i in idx_map for i in match):
            rings5.append({"atom_idxs": tuple(idx_map[i] for i in match)})
    for match in mol.GetSubstructMatches(ring6_smarts):
        if all(i in idx_map for i in match):
            rings6.append({"atom_idxs": tuple(idx_map[i] for i in match)})
    return planar_bonds, rings5, rings6


def _parse_ccd_ligand_residue(
    ref_mol,
    drop_leaving_atoms: bool = False
) -> tuple[list[Atom], list[tuple[int, int, int]], dict[str, list[dict]]]:
    """Parse a CCD ligand residue: heavy atoms + bonds + per-residue constraints.

    Mirrors OSS ``parse_ccd_residue``: iterates non-H atoms (filtering leaving
    atoms when requested), converts bonds via the surviving-atom index map,
    and computes RDKit bounds, chiral, stereo-bond, and flatness constraints
    over the same index map.

    The third return value is a dict of constraint lists with keys
    ``rdkit_bounds``, ``chiral_atoms``, ``stereo_bonds``, ``planar_bonds``,
    ``planar_ring_5``, ``planar_ring_6``.
    """
    from rdkit.Chem import AllChem

    unk_chirality = chirality_type_ids[unk_chirality_type]
    unk_bond = bond_type_ids[unk_bond_type]
    # Single-heavy-atom CCD residues need a placeholder name (matches OSS).
    from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

    empty_constraints: dict[str, list[dict]] = {
        "rdkit_bounds": [],
        "chiral_atoms": [],
        "stereo_bonds": [],
        "planar_bonds": [],
        "planar_ring_5": [],
        "planar_ring_6": [],
    }

    # Initialize ring info + valence cache (needed by the distance-bounds
    # matrix and SMARTS matching), but do NOT re-perceive aromaticity or
    # hybridization. OSS boltz1 ``parse_ccd_residue`` computes these
    # constraints on the ccd.pkl mol using its *shipped* aromaticity flags —
    # it never calls ``SetAromaticity`` / ``SetHybridization``. Re-perceiving
    # changes bond orders (shifting RDKit distance bounds ~0.1-0.2 A) and
    # surfaces aromatic rings the OSS reference does not, so it must not be
    # done here. Match OSS exactly: cache only, no re-perception.
    try:
        from rdkit import Chem as _Chem
        _Chem.GetSSSR(ref_mol)
        ref_mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    if CalcNumHeavyAtoms(ref_mol) == 1:
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality = chirality_type_ids.get(str(ref_atom.GetChiralTag()),
                                           unk_chirality)
        try:
            name = ref_atom.GetProp("name")
        except KeyError:
            name = ref_atom.GetSymbol().upper()
        return [
            Atom(
                name=name,
                element=ref_atom.GetAtomicNum(),
                charge=ref_atom.GetFormalCharge(),
                coords=(0.0, 0.0, 0.0),
                conformer=(0.0, 0.0, 0.0),
                is_present=True,
                chirality=chirality,
            )
        ], [], empty_constraints

    conformer = _get_conformer(ref_mol)
    atoms: list[Atom] = []
    idx_map: dict[int, int] = {}
    atom_idx = 0
    for i, ref_atom in enumerate(ref_mol.GetAtoms()):
        if ref_atom.GetAtomicNum() == 1:
            continue
        if drop_leaving_atoms:
            try:
                if int(ref_atom.GetProp("leaving_atom")):
                    continue
            except KeyError:
                pass
        try:
            atom_name = ref_atom.GetProp("name")
        except KeyError:
            atom_name = ref_atom.GetSymbol().upper()
        pos = conformer.GetAtomPosition(ref_atom.GetIdx())
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
        idx_map[i] = atom_idx
        atom_idx += 1

    bonds: list[tuple[int, int, int]] = []
    for bond in ref_mol.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        if i1 not in idx_map or i2 not in idx_map:
            continue
        a1 = idx_map[i1]
        a2 = idx_map[i2]
        start = min(a1, a2)
        end = max(a1, a2)
        bond_type = bond_type_ids.get(bond.GetBondType().name, unk_bond)
        bonds.append((start, end, bond_type))

    rdkit_bounds = _compute_rdkit_bounds_constraints(ref_mol, idx_map)
    chiral_atoms = _compute_chiral_atom_constraints(ref_mol, idx_map)
    stereo_bonds = _compute_stereo_bond_constraints(ref_mol, idx_map)
    planar_bonds, planar_ring_5, planar_ring_6 = _compute_flatness_constraints(
        ref_mol, idx_map)
    constraints = {
        "rdkit_bounds": rdkit_bounds,
        "chiral_atoms": chiral_atoms,
        "stereo_bonds": stereo_bonds,
        "planar_bonds": planar_bonds,
        "planar_ring_5": planar_ring_5,
        "planar_ring_6": planar_ring_6,
    }
    return atoms, bonds, constraints


def _parse_modified_residue(name: str, ref_mol, gemmi_res, res_idx: int) -> dict:
    """Parse a modified/non-standard polymer residue, mirroring OSS
    ``parse_ccd_residue`` (``mmcif.py:371``, ``is_covalent=True``).

    Sibling of :func:`_parse_ccd_ligand_residue` / :func:`_parse_polymer_residue`,
    but overlays real coordinates from a template gemmi residue (by atom name)
    instead of the CCD conformer, and emits a plain residue dict with token type
    ``UNK`` and no center/distogram atom — exactly OSS's modified-residue
    behaviour (e.g. CSO -> atoms N,CA,CB,SG,C,O,OD, center=N). Consumed by the
    template featurizer in ``template_logic``.
    """
    from rdkit import Chem
    ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)
    is_present = gemmi_res is not None
    pdb_pos: dict[str, tuple[float, float, float]] = {}
    if is_present:
        for a in gemmi_res:
            pdb_pos[a.name] = (float(a.pos.x), float(a.pos.y), float(a.pos.z))

    atoms: list[tuple] = []
    ref_atom_list = list(ref_mol.GetAtoms())
    if len(ref_atom_list) == 1:
        nm = ref_atom_list[0].GetProp("name")
        coords = pdb_pos.get(nm)
        atoms.append((nm, coords or (0.0, 0.0, 0.0),
                      bool(coords is not None and is_present)))
    else:
        for a in ref_atom_list:
            nm = a.GetProp("name")
            # Skip covalent leaving atoms not present in the PDB (OSS rule).
            if (a.HasProp("leaving_atom")
                    and int(a.GetProp("leaving_atom")) == 1
                    and nm not in pdb_pos):
                continue
            coords = pdb_pos.get(nm)
            atoms.append((nm, coords or (0.0, 0.0, 0.0),
                          bool(coords is not None and is_present)))

    return {
        "name": name,                       # keep CCD name (token metadata)
        "res_type": token_ids["UNK"],       # OSS types modified residues as UNK
        "res_idx": res_idx,
        "atoms": atoms,
        "atom_center": 0,                   # OSS parse_ccd_residue: no center
        "atom_disto": 0,
        "is_present": is_present,
        "is_standard": False,               # OSS: modified residue -> no frame
    }


def _build_smiles_mol(smiles: str, name: str):
    """Build an RDKit Mol from SMILES with canonical atom names + 3D conformer.

    Follows the OSS ``parse_boltz_schema`` SMILES branch: parse, add Hs,
    canonical rank, assign stereo, set per-atom names, embed 3D, then remove Hs.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES for {name}: {smiles!r}")
    mol = Chem.AddHs(mol)
    # OSS schema.py sets atom names from CanonicalRankAtoms, then assigns
    # stereochemistry (this order matters; do not swap).
    canonical_order = Chem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(int(can_idx) + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"SMILES {smiles!r} has an atom with a name longer than 4 chars: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    # Replicate OSS schema.compute_3d_conformer exactly: ETKDGv3 embed with a
    # random-coords fallback, then UFF-optimize (maxIters=1000). The prior
    # implementation used a plain EmbedMolecule(randomSeed=42) with no UFF
    # relaxation, which produced a different conformer geometry than the OSS
    # reference (ref_pos diverged). The conformer is named "Computed" so the
    # downstream _get_conformer / _select_conformer picks it deterministically.
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    conf_id = AllChem.EmbedMolecule(mol, options)
    if conf_id == -1:
        options.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, options)
    if conf_id == -1:
        raise ValueError(
            f"Failed to compute 3D conformer for SMILES {smiles!r}")
    try:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
    except (RuntimeError, ValueError):
        pass  # force-field / sanitization issue — keep the embedded coords
    conformer = mol.GetConformer(conf_id)
    conformer.SetProp("name", "Computed")
    mol_no_h = Chem.RemoveHs(mol, sanitize=False)
    # ``RemoveHs(sanitize=False)`` strips the ``_CIPRank`` properties that
    # ``compute_chiral_atom_constraints`` and ``compute_stereo_bond_constraints``
    # require. Re-run stereo assignment on the heavy-atom-only mol so chiral
    # centers and E/Z stereo bonds are picked up downstream.
    Chem.AssignStereochemistry(mol_no_h, force=True, cleanIt=True)
    return mol_no_h


def build_structure_from_input(
    input_parsed: dict,
    ccd: dict,
) -> tuple[Structure, dict[str, Any], dict[str, list[dict]]]:
    """Build a Structure from InputParsed and CCD.

    Supports protein, RNA, DNA polymer chains and CCD/SMILES ligand chains.

    Args:
        input_parsed: Parsed input dict containing ``"polymers"`` list.
        ccd: CCD dictionary mapping residue names to RDKit molecules.

    Returns:
        Tuple of (Structure, extra_mols, constraints). ``extra_mols`` maps
        generated ligand identifiers (e.g. ``"LIG1"``) to the RDKit Mol so
        the feature generator can resolve atoms whose names are not in the
        CCD. ``constraints`` is a dict of per-residue RDKit-derived geometry
        constraints (``rdkit_bounds``, ``chiral_atoms``, ``stereo_bonds``,
        ``planar_bonds``, ``planar_ring_5``, ``planar_ring_6``) keyed by
        constraint type with global atom indices ready for downstream
        featurization.
    """
    polymers = input_parsed.get("polymers") or []
    if not polymers:
        raise ValueError("No polymers in input")

    # Group by (polymer_type, sequence) for entity_id.
    entity_keys: list[tuple[str, str]] = []
    seen: dict[tuple[str, str], int] = {}
    for p in polymers:
        pt = (p.get("polymer_type") or "protein").lower()
        seq = p.get("sequence") or ""
        key = (pt, seq)
        if key not in seen:
            seen[key] = len(entity_keys)
            entity_keys.append(key)
        p["_entity_id"] = seen[key]

    # Emit chains in ENTITY-GROUPED order (stable sort by entity id), matching
    # OSS ``parse_boltz_schema`` asym_id assignment. No-op for entity-contiguous
    # inputs; only reorders interleaved homo-oligomers (e.g. 1a3n A,C,B,D).
    ordered_polymers = sorted(polymers, key=lambda p: p["_entity_id"])

    all_atoms: list[Atom] = []
    all_residues: list[Residue] = []
    all_chains: list[Chain] = []
    all_bonds: list[Bond] = []
    sym_count: dict[int, int] = {}
    global_atom_idx = 0
    global_res_idx = 0
    chain_idx = 0
    ligand_counter = 0
    extra_mols: dict[str, Any] = {}
    rdkit_bounds_global: list[dict] = []
    chiral_atoms_global: list[dict] = []
    stereo_bonds_global: list[dict] = []
    planar_bonds_global: list[dict] = []
    planar_ring_5_global: list[dict] = []
    planar_ring_6_global: list[dict] = []

    for poly in ordered_polymers:
        polymer_type = (poly.get("polymer_type") or "protein").lower()
        sequence = poly.get("sequence") or ""
        chain_ids = poly.get("chain_id")
        if chain_ids is None:
            chain_ids = ["A"]
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        entity_id = poly["_entity_id"]

        _empty_constraints: dict[str, list[dict]] = {
            "rdkit_bounds": [],
            "chiral_atoms": [],
            "stereo_bonds": [],
            "planar_bonds": [],
            "planar_ring_5": [],
            "planar_ring_6": [],
        }

        if polymer_type in _POLYMER_TYPES:
            mol_type = chain_type_ids[polymer_type.upper()]
            seq_tokens = _seq_to_tokens(polymer_type, sequence)
            residues_atoms_bonds = [
                (res_name, _parse_polymer_residue(res_name, ccd), [],
                 _empty_constraints, True,
                 token_ids.get(res_name, token_ids[_unk_for(polymer_type)]),
                 res_to_center_atom_id.get(res_name, 0),
                 res_to_disto_atom_id.get(res_name, 0))
                for res_name in seq_tokens
            ]
        elif polymer_type in _LIGAND_TYPES:
            unk_prot_id = unk_token_ids["PROTEIN"]
            mol_type = chain_type_ids["NONPOLYMER"]
            residues_atoms_bonds = []
            if polymer_type == "ccd_ligand":
                codes = sequence.split("_") if sequence else []
                for code in codes:
                    ref_mol = ccd.get(code)
                    if ref_mol is None:
                        raise ValueError(
                            f"CCD missing ligand component: {code!r} in polymer"
                            f" entity {entity_id}")
                    atoms, bonds, residue_constraints = (
                        _parse_ccd_ligand_residue(ref_mol))
                    residues_atoms_bonds.append(
                        (code, atoms, bonds, residue_constraints, False,
                         unk_prot_id, 0, 0))
            else:  # smiles_ligand
                ligand_counter += 1
                lig_name = f"LIG{ligand_counter}"
                mol = _build_smiles_mol(sequence, lig_name)
                extra_mols[lig_name] = mol
                atoms, bonds, residue_constraints = _parse_ccd_ligand_residue(
                    mol)
                residues_atoms_bonds.append(
                    (lig_name, atoms, bonds, residue_constraints, False,
                     unk_prot_id, 0, 0))
        else:
            raise ValueError(f"Unsupported polymer_type: {polymer_type!r}")

        # Create one chain per chain_id (sharing the entity_id).
        for ch_name in chain_ids:
            sym_id = sym_count.get(entity_id, 0)
            sym_count[entity_id] = sym_id + 1
            chain_atom_start = global_atom_idx
            chain_res_start = global_res_idx
            chain_atom_count = 0
            chain_res_count = 0
            for res_idx_in_chain, (
                    res_name, atoms, bonds, residue_constraints, is_standard,
                    res_type, center_off,
                    disto_off) in enumerate(residues_atoms_bonds):
                if not atoms:
                    raise ValueError(
                        f"Residue {res_name} has no atoms (mol_type={polymer_type})"
                    )
                atom_center_global = global_atom_idx + center_off
                atom_disto_global = global_atom_idx + disto_off
                all_residues.append(
                    Residue(
                        name=res_name,
                        res_type=res_type,
                        res_idx=res_idx_in_chain,
                        atom_idx=global_atom_idx,
                        atom_num=len(atoms),
                        atom_center=atom_center_global,
                        atom_disto=atom_disto_global,
                        is_standard=is_standard,
                        is_present=True,
                    ))
                all_atoms.extend(atoms)
                for (a1, a2, btype) in bonds:
                    all_bonds.append(
                        Bond(
                            chain_1=chain_idx,
                            chain_2=chain_idx,
                            res_1=global_res_idx,
                            res_2=global_res_idx,
                            atom_1=global_atom_idx + a1,
                            atom_2=global_atom_idx + a2,
                            type=btype,
                        ))
                for c in residue_constraints["rdkit_bounds"]:
                    a1, a2 = c["atom_idxs"]
                    rdkit_bounds_global.append({
                        "atom_idxs":
                        (global_atom_idx + a1, global_atom_idx + a2),
                        "is_bond":
                        c["is_bond"],
                        "is_angle":
                        c["is_angle"],
                        "upper_bound":
                        c["upper_bound"],
                        "lower_bound":
                        c["lower_bound"],
                    })
                for c in residue_constraints["chiral_atoms"]:
                    a = c["atom_idxs"]
                    chiral_atoms_global.append({
                        "atom_idxs":
                        tuple(global_atom_idx + i for i in a),
                        "is_reference":
                        c["is_reference"],
                        "is_r":
                        c["is_r"],
                    })
                for c in residue_constraints["stereo_bonds"]:
                    a = c["atom_idxs"]
                    stereo_bonds_global.append({
                        "atom_idxs":
                        tuple(global_atom_idx + i for i in a),
                        "is_reference":
                        c["is_reference"],
                        "is_e":
                        c["is_e"],
                    })
                for c in residue_constraints["planar_bonds"]:
                    a = c["atom_idxs"]
                    planar_bonds_global.append({
                        "atom_idxs":
                        tuple(global_atom_idx + i for i in a),
                    })
                for c in residue_constraints["planar_ring_5"]:
                    a = c["atom_idxs"]
                    planar_ring_5_global.append({
                        "atom_idxs":
                        tuple(global_atom_idx + i for i in a),
                    })
                for c in residue_constraints["planar_ring_6"]:
                    a = c["atom_idxs"]
                    planar_ring_6_global.append({
                        "atom_idxs":
                        tuple(global_atom_idx + i for i in a),
                    })
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
    for i, a in enumerate(all_atoms):
        coords[i, 0], coords[i, 1], coords[i, 2] = a.coords
    n_chains = len(all_chains)
    mask = np.ones(n_chains, dtype=bool)
    ensemble = np.array([(0, n_atoms)], dtype=EnsembleDtype)
    bfactor = np.zeros(n_atoms, dtype=np.float32)
    plddt = np.ones(n_atoms, dtype=np.float32)

    structure = Structure(
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
    constraints = {
        "rdkit_bounds": rdkit_bounds_global,
        "chiral_atoms": chiral_atoms_global,
        "stereo_bonds": stereo_bonds_global,
        "planar_bonds": planar_bonds_global,
        "planar_ring_5": planar_ring_5_global,
        "planar_ring_6": planar_ring_6_global,
    }
    return structure, extra_mols, constraints


def _unk_for(polymer_type: str) -> str:
    """Return the canonical UNK token name for a polymer type."""
    if polymer_type == "protein":
        return "UNK"
    if polymer_type == "rna":
        return "N"
    if polymer_type == "dna":
        return "DN"
    return "UNK"
