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

import biotite.structure as struc
import biotite.structure.info as struc_info
import numpy as np
from biotite.interface.rdkit import to_mol as biotite_to_mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from tensorrt_bionemo.data.schemas.basic import InputParsed
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase

from .const import (_PROTEIN_1TO3, DNA_RESTYPE_1TO3, GAP_IDX, MOL_TYPE_DNA,
                    MOL_TYPE_LIGAND, MOL_TYPE_PROTEIN, MOL_TYPE_RNA,
                    MSA_CHAR_TO_IDX, POLYMER_TYPE_TO_MOL_TYPE, RNA_1_TO_IDX,
                    RNA_RESTYPE_1TO3, UNK_IDX)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MSA character dispatch (D-05)
# ---------------------------------------------------------------------------

# Module-level set to suppress repeated warnings for the same (char, mol_type)
# pair (T-02-06: DoS via unbounded warning spam).
_seen_unknown_msa_chars: set[tuple[str, int]] = set()


def _resolve_msa_char(char: str, mol_type: int) -> int:
    """Polymer-type-aware MSA char -> 32-class restype index dispatch (D-05).

    Args:
        char: Single MSA character (e.g. 'A', 'U', '-').
        mol_type: Molecule type integer (MOL_TYPE_PROTEIN/RNA/DNA/LIGAND).

    Returns:
        Restype index in the 32-class vocabulary.

    Dispatch rules (D-05):
      - Gap chars ('-', '.') -> GAP_IDX for all mol_types.
      - MOL_TYPE_PROTEIN -> MSA_CHAR_TO_IDX.get(char.upper(), UNK_IDX)
        (backward compat).
      - MOL_TYPE_RNA -> RNA_1_TO_IDX.get(char.upper(), UNK_IDX); unknown
        chars log a WARNING once per unique (char, mol_type) pair.
      - MOL_TYPE_DNA / MOL_TYPE_LIGAND -> GAP_IDX unconditionally (zero-fill
        per D-05: DNA never has MSA; ligand is always atomized).
    """
    if char in ("-", "."):
        return GAP_IDX

    c_upper = char.upper()

    if mol_type == MOL_TYPE_PROTEIN:
        return MSA_CHAR_TO_IDX.get(c_upper, UNK_IDX)

    if mol_type == MOL_TYPE_RNA:
        idx = RNA_1_TO_IDX.get(c_upper)
        if idx is not None:
            return idx
        # Unknown RNA MSA char — warn once per unique (char, mol_type) (T-02-06)
        key = (c_upper, mol_type)
        if key not in _seen_unknown_msa_chars:
            _seen_unknown_msa_chars.add(key)
            _logger.warning("Unknown RNA MSA char %r; falling back to UNK_IDX",
                            char)
        return UNK_IDX

    if mol_type in (MOL_TYPE_DNA, MOL_TYPE_LIGAND):
        # D-05: DNA/LIGAND positions are always zero-filled — no MSA for these.
        return GAP_IDX

    # Unknown mol_type — warn and return UNK_IDX
    _logger.warning(
        "Unknown mol_type %d in _resolve_msa_char; returning UNK_IDX",
        mol_type)
    return UNK_IDX


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


# ---------------------------------------------------------------------------
# Nucleotide CCD helpers (D-01, D-02, D-03)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=500)
def _get_nucleotide_from_ccd(ccd_code: str) -> struc.AtomArray:
    """Heavy atoms only, phosphate leaving groups removed (OP3, O3P).

    Per D-02: RNA/DNA polymer linkage leaving atoms are OP3 and O3P.
    Per D-03: @lru_cache bounds biotite CCD disk-read overhead.
    """
    res = struc_info.residue(ccd_code)
    res = res[res.element != "H"]
    res = res[~np.isin(res.atom_name, ["OP3", "O3P"])]
    return res


@lru_cache(maxsize=500)
def _get_nucleotide_from_ccd_with_leaving(ccd_code: str) -> struc.AtomArray:
    """Heavy atoms only, leaving groups kept (for RDKit topology).

    Per D-02: Leaving groups (OP3/O3P) are retained for correct bond topology
    during RDKit conformer generation, then filtered out in structure building.
    Per D-03: @lru_cache bounds biotite CCD disk-read overhead.
    """
    res = struc_info.residue(ccd_code)
    res = res[res.element != "H"]
    return res


def _embed_conformer_inplace(mol_h: Chem.Mol) -> int:
    """Embed one 3-D conformer into *mol_h*; return its id (-1 on failure).

    Monatomic species — e.g. metal-ion ligands like CD (cadmium), CO
    (cobalt), ZN, NA, K, MG — have no geometry to embed. Calling
    ``AllChem.EmbedMolecule`` on them still "succeeds" (the lone atom lands
    at the origin) but makes RDKit's UFF typer log noisy
    ``UFFTYPER: Unrecognized atom type`` warnings for the unparameterised
    metal. Short-circuit single-atom mols by placing the atom at the origin
    directly — numerically identical to what ETKDG produces for one atom,
    without the warning or the wasted embedding attempt.
    """
    import random as _random

    if mol_h.GetNumAtoms() == 1:
        conf = Chem.Conformer(1)
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        mol_h.RemoveAllConformers()
        return mol_h.AddConformer(conf, assignId=True)

    params = AllChem.ETKDGv3()
    params.randomSeed = _random.randint(0, 10**9)
    params.clearConfs = False
    conf_id = AllChem.EmbedMolecule(mol_h, params)
    if conf_id == -1:
        params.useRandomCoords = True
        params.randomSeed = _random.randint(0, 10**9)
        conf_id = AllChem.EmbedMolecule(mol_h, params)
    return conf_id


def _build_nucleotide_rdkit_mol(
    ccd_code: str, ) -> tuple[Optional[Chem.Mol], np.ndarray]:
    """Convert a CCD nucleotide residue to an RDKit Mol with 3D conformer.

    Matches OSS: keeps OP3/O3P in the mol for correct bond topology during
    conformer generation, then returns a mask indicating which atoms to
    include in the final features (OP3/O3P masked out). Mirrors
    _build_residue_rdkit_mol but uses the nucleotide leaving-atom policy.

    Returns:
        (mol, in_crop_mask): mol with conformer, boolean mask over mol atoms.
        in_crop_mask[i] = True for atoms to keep (non-leaving).
    """
    # Use the full residue (with OP3/O3P) for conformer generation
    res_full = _get_nucleotide_from_ccd_with_leaving(ccd_code)

    try:
        mol = biotite_to_mol(res_full, kekulize=True)
        Chem.SanitizeMol(mol)
        mol.RemoveConformer(0)
        mol_h = Chem.AddHs(mol)
        _embed_conformer_inplace(mol_h)
        mol_h = Chem.RemoveHs(mol_h)

        # Build mask: exclude OP3 and O3P leaving atoms
        in_crop_mask = np.array([
            res_full.atom_name[i] not in ("OP3", "O3P")
            for i in range(len(res_full))
        ],
                                dtype=bool)

        return mol_h, in_crop_mask
    except Exception as e:
        _logger.debug("Failed to build RDKit mol for nucleotide %s: %s",
                      ccd_code, e)
        return None, np.array([], dtype=bool)


def _embed_smiles_mol(mol: Chem.Mol) -> Chem.Mol:
    """Add ETKDGv3 conformer to an RDKit Mol built from SMILES.

    Matches the ETKDGv3 seeded pattern from _build_residue_rdkit_mol (lines
    88-116 of the original protein conformer helper). Adds Hs, embeds, removes
    Hs.  Falls back to useRandomCoords if initial embedding fails.

    Args:
        mol: Heavy-atom RDKit Mol (no conformer, no Hs).

    Returns:
        mol with one 3-D conformer (heavy atoms only).

    Raises:
        ValueError: If embedding fails even with useRandomCoords.
    """
    mol_h = Chem.AddHs(mol)
    conf_id = _embed_conformer_inplace(mol_h)
    if conf_id == -1:
        raise ValueError(
            "ETKDGv3 conformer embedding failed for SMILES molecule")
    return Chem.RemoveHs(mol_h)


def _build_residue_rdkit_mol(
    ccd_code: str, ) -> tuple[Optional[Chem.Mol], np.ndarray]:
    """Convert a CCD residue to an RDKit Mol with 3D conformer.

    Matches OSS: keeps OXT in the mol for correct bond topology during
    conformer generation, then returns a mask indicating which atoms to
    include in the final features (OXT masked out).

    Returns:
        (mol, in_crop_mask): mol with conformer, boolean mask over mol atoms.
        in_crop_mask[i] = True for atoms to keep (non-OXT).
    """
    # Use the full residue (with OXT) for conformer generation
    res_full = _get_residue_from_ccd_with_oxt(ccd_code)

    try:
        # Use biotite.interface.rdkit.to_mol (matches OSS exactly)
        mol = biotite_to_mol(res_full, kekulize=True)
        Chem.SanitizeMol(mol)
        # Remove CCD conformer (OSS: mol.RemoveConformer(0))
        mol.RemoveConformer(0)
        # Generate fresh conformer using ETKDGv3 (single-atom ions are placed
        # at the origin without embedding — see _embed_conformer_inplace).
        mol_h = Chem.AddHs(mol)
        _embed_conformer_inplace(mol_h)
        mol_h = Chem.RemoveHs(mol_h)

        # Build mask: exclude OXT
        in_crop_mask = np.array(
            [res_full.atom_name[i] != "OXT" for i in range(len(res_full))],
            dtype=bool)

        return mol_h, in_crop_mask
    except Exception as e:
        _logger.debug("Failed to build RDKit mol for %s: %s", ccd_code, e)
        return None, np.array([], dtype=bool)


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------


def _build_structure_from_polymers(polymers: list[dict], ) -> dict[str, Any]:
    """Build atom-level and token-level structure data from polymer list
    using Biotite CCD for accurate atom arrays.

    For protein inputs, each standard amino acid residue = one token.
    For RNA/DNA inputs, each standard nucleotide residue = one token.
    Atom names, elements, and counts come directly from the CCD.

    Dispatch decisions (citng context decisions):
      D-01: Nucleotide CCD path mirrors OSS (biotite struc_info.residue).
      D-02: Protein leaves OXT; RNA/DNA leave OP3/O3P.
      D-03: @lru_cache on all CCD lookup helpers.
      D-04: Unknown residues log a WARNING and fall back to type-specific placeholder.
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
    # mol_idx_counter: tracks the unique conformer instance ID for ref_space_uid.
    # OSS conformer.py:128-129 sets ref_space_uid = mol_idx (the index into
    # processed_ref_mol_list). One molecule per non-atomized residue or one
    # per atomized ligand chain. We track it explicitly per token so that
    # the ConformerFeatureGenerator can emit ref_space_uid by lookup.
    token_mol_idx: list[int] = []
    next_mol_idx = 0

    # Assign entity IDs: same "entity representation" → same entity, with
    # IDs assigned by ALPHABETICAL SORT ORDER of unique representations.
    # This mirrors OSS structure_with_ref_mols_from_query (query.py:552):
    #     all_entities = sorted(all_entities)
    #     entity_to_id = {e: i + 1 for i, e in enumerate(all_entities)}
    # Previous behavior assigned by first-appearance order, which broke L1
    # equivalence on multimers where the first-appearing sequence wasn't
    # the alphabetically-first one (Rule 1 fix in Plan 01-06 Task 1).
    #
    # Entity representation per polymer type (matches OSS query.py:544-551):
    #   - PROTEIN / RNA / DNA   : the sequence string
    #   - LIGAND_CCD            : the CCD code (stored in `sequence`)
    #   - LIGAND_SMILES         : the SMILES string (stored in `sequence`)
    def _entity_repr(poly: dict) -> str:
        return poly.get("sequence", "") or ""

    all_entities = sorted({_entity_repr(p) for p in polymers})
    entity_to_id: dict[str, int] = {
        e: i + 1
        for i, e in enumerate(all_entities)
    }

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

        entity_id = entity_to_id[_entity_repr(poly)]

        # Determine per-polymer-type residue lookup strategy (D-01 / D-02 / D-04)
        match polymer_type:
            case "protein":
                resname_1_to_3 = _PROTEIN_1TO3
                unk_res_3 = "UNK"
                _get_ccd_fn = _get_residue_from_ccd
                _build_mol_fn = _build_residue_rdkit_mol
            case "rna":
                resname_1_to_3 = RNA_RESTYPE_1TO3
                unk_res_3 = "N"  # D-04: RNA unknown placeholder
                _get_ccd_fn = _get_nucleotide_from_ccd
                _build_mol_fn = _build_nucleotide_rdkit_mol
            case "dna":
                resname_1_to_3 = DNA_RESTYPE_1TO3
                unk_res_3 = "DN"  # D-04: DNA unknown placeholder
                _get_ccd_fn = _get_nucleotide_from_ccd
                _build_mol_fn = _build_nucleotide_rdkit_mol
            case "ccd_ligand":
                # CCD ligand: the entire chain is a single atomized token.
                # Per 01-PATTERNS.md LIGAND_CCD dispatch: no leaving atoms
                # removed for ligands (OSS uses MoleculeType.LIGAND, which has
                # no leaving-atom list). Reuse the protein CCD helper which
                # only removes OXT — for a CCD ligand code (e.g. "ATP") there
                # is no OXT, so _get_residue_from_ccd is safe.
                ccd_code = sequence.strip()
                if not ccd_code:
                    raise ValueError(
                        f"LIGAND_CCD chain {chain_ids} has empty sequence/ccd_code"
                    )
                resname_3 = ccd_code  # e.g. "ATP", "ZN"

                # Get atom array (heavy atoms — keep ALL atoms including
                # OXT). OSS does NOT strip OXT for ligands: the leaving-atom
                # list for MoleculeType.LIGAND is empty (residues.py
                # MOLECULE_TYPE_TO_LEAVING_ATOMS). Some CCD ligands like SAH
                # include an OXT that must be retained — stripping it caused
                # the per-ligand-chain atom count to be off by one (Rule 1
                # fix in Plan 01-06 Task 1).
                try:
                    ccd_res = _get_residue_from_ccd_with_oxt(resname_3)
                    res_atom_names = list(ccd_res.atom_name)
                    res_elements = list(ccd_res.element)
                except Exception as e:
                    raise ValueError(
                        f"CCD lookup failed for ligand CCD code {ccd_code!r}: {e}"
                    ) from e

                # Build RDKit mol with ETKDGv3 conformer. The RDKit mol-
                # builder helper internally also retains OXT for conformer
                # topology — we pass an all-ones crop mask so no atoms are
                # excluded post-conformer-gen (ligands keep every atom).
                mol, _proto_crop_mask = _build_residue_rdkit_mol(resname_3)
                # Override crop mask to all-ones — keep every atom including
                # any OXT, matching OSS LIGAND_CCD behavior.
                in_crop_mask = np.ones(len(res_atom_names), dtype=bool)

                # Extract formal charges from RDKit mol
                charges = []
                if mol is not None:
                    for ai, atom in enumerate(mol.GetAtoms()):
                        if ai < len(in_crop_mask) and in_crop_mask[ai]:
                            charges.append(atom.GetFormalCharge())
                if len(charges) != len(res_atom_names):
                    charges = [0] * len(res_atom_names)

                # OSS atomizes ligands: each atom becomes its own token.
                # See openfold-3/openfold3/core/data/primitives/structure/
                # tokenization.py:142 `tokenize_atom_array`. The whole-ligand-
                # as-one-token shortcut produced an N-token vs N-atom-token
                # mismatch on multi-atom ligands like SAH (26 atoms) → 1 vs 26.
                # Rule 1 fix in Plan 01-06 Task 1.
                #
                # ref_space_uid: each ligand CHAIN is one conformer instance
                # → all atom-tokens of a chain share the same mol_idx
                # (matches OSS conformer.py:128-129 where ref_space_uid =
                # mol_idx and mol_idx enumerates processed_ref_mol_list which
                # has one entry per ligand chain).
                len(res_atom_names)
                for cid in chain_ids:
                    chain_mol_idx = next_mol_idx
                    next_mol_idx += 1
                    for ai_in_ligand, (aname, elem) in enumerate(
                            zip(res_atom_names, res_elements, strict=True)):
                        token_resnames.append(resname_3)
                        token_chain_ids.append(cid)
                        token_entity_ids.append(entity_id)
                        token_mol_types.append(mol_type)
                        # res_id is 1 for all atoms of a CCD ligand (single
                        # residue), matching OSS atom_array_from_ccd_code.
                        token_res_ids.append(1)
                        token_start_atoms.append(atom_idx)
                        atoms_per_token.append(1)
                        token_mol_idx.append(chain_mol_idx)
                        atom_names.append(aname)
                        atom_elements.append(elem)
                        atom_token_idx.append(token_idx)
                        atom_idx += 1

                        # Per-atom mini-mol entries so that ref_pos uses the
                        # full conformer coordinates of THIS atom (extracted
                        # via the per-atom crop mask). We attach the full
                        # ligand mol to each per-atom token and use a one-hot
                        # crop mask to select the right atom in
                        # ConformerFeatureGenerator.
                        per_atom_mask = np.zeros(len(in_crop_mask), dtype=bool)
                        per_atom_mask[ai_in_ligand] = bool(
                            in_crop_mask[ai_in_ligand])
                        residue_mols.append(mol)
                        residue_crop_masks.append(per_atom_mask)
                        residue_atom_charges.append(
                            [charges[ai_in_ligand]] if ai_in_ligand <
                            len(charges) else [0])
                        token_idx += 1

                # Ligand case handled — skip the inner residue loop below
                continue

            case "smiles_ligand":
                # SMILES ligand: one atomized token per chain.
                # Per 01-PATTERNS.md LIGAND_SMILES dispatch: RDKit MolFromSmiles
                # → AddHs → ETKDGv3 → RemoveHs → per-element 1-indexed atom names.
                smiles = sequence.strip()
                if not smiles:
                    raise ValueError(
                        f"LIGAND_SMILES chain {chain_ids} has empty SMILES string"
                    )

                mol_raw = Chem.MolFromSmiles(smiles)
                if mol_raw is None:
                    raise ValueError(
                        f"Failed to parse SMILES {smiles!r} for smiles_ligand "
                        f"chain {chain_ids}. Check that the SMILES string is valid."
                    )

                # Add ETKDGv3 conformer (T-03-01: RDKit None already handled above)
                try:
                    mol_embedded = _embed_smiles_mol(mol_raw)
                except ValueError as e:
                    raise ValueError(
                        f"SMILES conformer embedding failed for {smiles!r}: {e}"
                    ) from e

                # Build per-element 1-indexed atom names (C1, C2, O1, O2, ...)
                element_counts: dict[str, int] = {}
                smiles_atom_names: list[str] = []
                smiles_atom_elements: list[str] = []
                smiles_charges: list[int] = []
                for rdkit_atom in mol_embedded.GetAtoms():
                    symbol = rdkit_atom.GetSymbol()  # e.g. "C", "O", "N"
                    element_counts[symbol] = element_counts.get(symbol, 0) + 1
                    atom_label = f"{symbol}{element_counts[symbol]}"  # "C1", "O1"
                    smiles_atom_names.append(atom_label)
                    smiles_atom_elements.append(symbol)
                    smiles_charges.append(rdkit_atom.GetFormalCharge())

                resname_3 = "LIG"  # RESNAME_TO_IDX.get("LIG", UNK_IDX) → 20

                # OSS atomizes ligands: each atom becomes its own token.
                # Mirror the same pattern used for LIGAND_CCD above. Each
                # SMILES ligand chain is ONE conformer instance — all atom-
                # tokens of a chain share the same mol_idx (ref_space_uid).
                # Rule 1 fix in Plan 01-06 Task 1.
                n_smiles_atoms = len(smiles_atom_names)
                for cid in chain_ids:
                    chain_mol_idx = next_mol_idx
                    next_mol_idx += 1
                    for ai_in_ligand in range(n_smiles_atoms):
                        token_resnames.append(resname_3)
                        token_chain_ids.append(cid)
                        token_entity_ids.append(entity_id)
                        token_mol_types.append(mol_type)
                        token_res_ids.append(1)
                        token_start_atoms.append(atom_idx)
                        atoms_per_token.append(1)
                        token_mol_idx.append(chain_mol_idx)
                        atom_names.append(smiles_atom_names[ai_in_ligand])
                        atom_elements.append(
                            smiles_atom_elements[ai_in_ligand])
                        atom_token_idx.append(token_idx)
                        atom_idx += 1

                        # Per-atom mol entry: full SMILES mol + per-atom crop
                        # mask so ConformerFeatureGenerator extracts the
                        # right atom's coords.
                        per_atom_mask = np.zeros(n_smiles_atoms, dtype=bool)
                        per_atom_mask[ai_in_ligand] = True
                        residue_mols.append(mol_embedded)
                        residue_crop_masks.append(per_atom_mask)
                        residue_atom_charges.append(
                            [smiles_charges[ai_in_ligand]])
                        token_idx += 1

                # Ligand case handled — skip the inner residue loop below
                continue

            case _:
                raise ValueError(
                    f"Unsupported polymer_type for structure building: "
                    f"{polymer_type!r}. Supported types: "
                    f"protein, rna, dna, ccd_ligand, smiles_ligand")

        for cid in chain_ids:
            for res_idx, res_char in enumerate(sequence):
                # Resolve 3-letter residue code
                if res_char in resname_1_to_3:
                    resname_3 = resname_1_to_3[res_char]
                else:
                    # D-04: unknown residue → WARNING + placeholder
                    _logger.warning(
                        "Unknown %s residue %r at chain %s position %d; "
                        "using placeholder %r",
                        polymer_type,
                        res_char,
                        cid,
                        res_idx + 1,
                        unk_res_3,
                    )
                    resname_3 = unk_res_3

                # Get atom array from Biotite CCD (D-01, D-02)
                try:
                    ccd_res = _get_ccd_fn(resname_3)
                    res_atom_names = list(ccd_res.atom_name)
                    res_elements = list(ccd_res.element)
                except Exception as e:
                    # Fallback for CCD lookup failures
                    _logger.debug(
                        "CCD lookup failed for %s, using backbone fallback: %s",
                        resname_3,
                        e,
                    )
                    if polymer_type == "protein":
                        res_atom_names = ["N", "CA", "C", "O"]
                        res_elements = ["N", "C", "C", "O"]
                    else:
                        # Minimal nucleotide backbone: P, C4', C3'
                        res_atom_names = ["P", "C4'", "C3'"]
                        res_elements = ["P", "C", "C"]

                # Build RDKit mol for conformer generation (D-02)
                mol, in_crop_mask = _build_mol_fn(resname_3)
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
                # Each non-atomized residue is its own conformer instance.
                token_mol_idx.append(next_mol_idx)
                next_mol_idx += 1

                n_atoms_token = len(res_atom_names)
                atoms_per_token.append(n_atoms_token)

                for aname, elem in zip(res_atom_names,
                                       res_elements,
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
        # token_mol_idx: per-token conformer instance ID used as
        # ref_space_uid for each atom (OSS conformer.py:128-129).
        # Non-atomized residues bump per-residue; ligand chains share
        # one mol_idx across all atomized tokens of the chain.
        "token_mol_idx": token_mol_idx,
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
    """Renumber chain IDs to 1-based integers, sorted alphabetically.

    Mirrors OSS pipelines/featurization/structure.py::create_basic_features
    line 156, which uses ``np.unique(chain_ids_token, return_inverse=True)``
    — i.e. unique chain IDs are sorted alphabetically and renumbered starting
    at 1. Rule 1 fix in Plan 01-06 Task 1: previous behavior numbered chains
    by first-appearance order, which broke multimer L1 equivalence when
    chain_ids in the input JSON did not appear alphabetically (e.g.
    hemoglobin's [A, C, B, D]).
    """
    unique_sorted = sorted(set(chain_ids))
    chain_to_num: dict[str, int] = {
        cid: i + 1
        for i, cid in enumerate(unique_sorted)
    }
    return [chain_to_num[cid] for cid in chain_ids]


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

        # Per-chain template records + query sequences for the direct-CIF path
        # (protein only), keyed by the original chain_id string.
        templates_per_chain: dict[str, list] = {}
        template_query_seq: dict[str, str] = {}
        for poly in polymers:
            if poly.get("polymer_type", "protein") != "protein":
                continue
            templates = poly.get("templates")
            if not templates:
                continue
            chain_id = poly.get("chain_id")
            if isinstance(chain_id, list):
                cids = chain_id
            elif chain_id is not None:
                cids = [chain_id]
            else:
                cids = ["A"]
            for cid in cids:
                templates_per_chain[cid] = list(templates)
                template_query_seq[cid] = poly.get("sequence", "") or ""

        return {
            "structure": struct,
            "msa_per_chain": msa_per_chain,
            "paired_msa_per_chain": paired_msa_per_chain,
            "chain_sequences": chain_sequences,
            "templates_per_chain": templates_per_chain,
            "template_query_seq": template_query_seq,
        }
