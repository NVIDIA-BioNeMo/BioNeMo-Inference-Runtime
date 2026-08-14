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

"""Boltz2 constants and clean Python dataclass types (no NumPy structured dtypes)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# CHAINS
# -----------------------------------------------------------------------------

chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}
num_chain_types = len(chain_types)

# -----------------------------------------------------------------------------
# RESIDUES & TOKENS
# -----------------------------------------------------------------------------

canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]

tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

rna_letter_to_token = {"A": "A", "G": "G", "C": "C", "U": "U", "N": "N"}
dna_letter_to_token = {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}

# -----------------------------------------------------------------------------
# ATOMS
# -----------------------------------------------------------------------------

num_elements = 128

chirality_types = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER",
]
chirality_type_ids = {c: i for i, c in enumerate(chirality_types)}
unk_chirality_type = "CHI_OTHER"

ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "G": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "C": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "U": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
    ],
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
    "DA": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "DG": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "DC": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "DT": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
    ],
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"],
}

protein_backbone_atom_names = ["N", "CA", "C", "O"]
nucleic_backbone_atom_names = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'",
}

res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'",
}

res_to_center_atom_id = {res: ref_atoms[res].index(atom) for res, atom in res_to_center_atom.items()}
res_to_disto_atom_id = {res: ref_atoms[res].index(atom) for res, atom in res_to_disto_atom.items()}

# -----------------------------------------------------------------------------
# BONDS
# -----------------------------------------------------------------------------

bond_types = ["OTHER", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "COVALENT"]
bond_type_ids = {b: i for i, b in enumerate(bond_types)}
unk_bond_type = "OTHER"
num_bond_types = len(bond_types)

# -----------------------------------------------------------------------------
# CONTACTS
# -----------------------------------------------------------------------------

pocket_contact_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}
num_pocket_contact_info = len(pocket_contact_info)

contact_conditioning_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}

# -----------------------------------------------------------------------------
# BACKBONE ATOM INDICES (for atom_backbone_feat)
# -----------------------------------------------------------------------------

protein_backbone_atom_index = {name: i for i, name in enumerate(protein_backbone_atom_names)}
nucleic_backbone_atom_index = {name: i for i, name in enumerate(nucleic_backbone_atom_names)}

# -----------------------------------------------------------------------------
# MSA
# -----------------------------------------------------------------------------

# Match the upstream ``boltz predict`` runtime cap (``boltz.main.predict``) —
# the predict-path default is ``max_msa_seqs=8192``, NOT the 16384 from
# ``boltz.data.const.max_msa_seqs`` (which is the training/general-purpose
# cap). Byte-equivalent TRT featurization against predict-path references
# requires the same 8192 cap. Without this alignment, samples with deep MSAs
# (T1152 and smiles_demo at 15872 rows, T1047s1 at 8395) produce shape
# mismatches in 5 MSA-derived feature tensors.
max_msa_seqs = 8192
max_paired_seqs = 8192

# -----------------------------------------------------------------------------
# METHOD CONDITIONING
# -----------------------------------------------------------------------------

method_types_ids = {
    "md": 0,
    "x-ray diffraction": 1,
    "electron microscopy": 2,
    "solution nmr": 3,
    "solid-state nmr": 4,
    "neutron diffraction": 4,
    "electron crystallography": 4,
    "fiber diffraction": 4,
    "powder diffraction": 4,
    "infrared spectroscopy": 4,
    "fluorescence transfer": 4,
    "epr": 4,
    "theoretical model": 4,
    "solution scattering": 4,
    "other": 4,
    "afdb": 5,
    "boltz-1": 6,
    "future1": 7,
    "future2": 8,
    "future3": 9,
    "future4": 10,
    "future5": 11,
}
num_method_types = len(set(method_types_ids.values()))

# -----------------------------------------------------------------------------
# VDW RADII
# -----------------------------------------------------------------------------

# fmt: off
vdw_radii = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1,
    1.95, 1.8, 1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0,
    2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3, 2.15,
    2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0,
    2.7, 2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32,
    2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3,
    2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]
# fmt: on

# -----------------------------------------------------------------------------
# CLEAN PYTHON DATACLASS TYPES (no NumPy structured dtypes)
# -----------------------------------------------------------------------------


@dataclass
class Atom:
    """Single atom in a residue."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Atom:
        return cls(**d)


@dataclass
class Bond:
    """Bond between two atoms (global indices)."""

    chain_1: int
    chain_2: int
    res_1: int
    res_2: int
    atom_1: int
    atom_2: int
    type: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Bond:
        return cls(**d)


@dataclass
class Residue:
    """Residue in a chain."""

    name: str  # 3-letter CCD name, e.g. "ALA"
    res_type: int  # token_id
    res_idx: int
    atom_idx: int
    atom_num: int
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Residue:
        return cls(**d)


@dataclass
class Chain:
    """Chain in the structure."""

    name: str
    mol_type: int
    entity_id: int
    sym_id: int
    asym_id: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_num: int
    cyclic_period: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chain:
        return cls(**d)


@dataclass
class Token:
    """Token (one per residue for polymers) for featurization."""

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    res_name: str  # 3-letter CCD name for mol lookup
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: tuple[float, float, float]
    disto_coords: tuple[float, float, float]
    resolved_mask: bool
    disto_mask: bool
    modified: bool
    frame_rot: tuple[tuple[float, float, float], ...]  # 3x3 as 3 tuples
    frame_t: tuple[float, float, float]
    frame_mask: bool
    cyclic_period: int
    affinity_mask: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Token:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TokenBond:
    """Bond between two tokens (by token index). type is bond_type + 1."""

    token_1: int
    token_2: int
    type: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenBond:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# OSS-compatible ensemble: one row per conformer (atom_coord_idx, atom_num).
# Single conformer: [(0, n_atoms)] so coords[0:n_atoms] is the first conformer.
EnsembleDtype = np.dtype([("atom_coord_idx", np.int32), ("atom_num", np.int32)])


@dataclass
class Structure:
    """Full structure: atoms, bonds, residues, chains, coords, ensemble metadata."""

    atoms: list[Atom]
    bonds: list[Bond]
    residues: list[Residue]
    chains: list[Chain]
    coords: np.ndarray  # (n_atoms, 3) float32
    ensemble: np.ndarray  # (n_conformers,) dtype EnsembleDtype; OSS: ensemble[i]["atom_coord_idx"], ["atom_num"]
    mask: np.ndarray  # (n_chains,) bool
    bfactor: np.ndarray  # (n_atoms,) float32; from structure / OSS npz
    plddt: np.ndarray  # (n_atoms,) float32; from structure / OSS npz

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [a.to_dict() for a in self.atoms],
            "bonds": [b.to_dict() for b in self.bonds],
            "residues": [r.to_dict() for r in self.residues],
            "chains": [c.to_dict() for c in self.chains],
            "coords": self.coords.tolist(),
            "ensemble": self.ensemble.tolist(),
            "mask": self.mask.tolist(),
            "bfactor": self.bfactor.tolist(),
            "plddt": self.plddt.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Structure:

        def _np(val: Any, dtype: Any) -> np.ndarray:
            if isinstance(val, np.ndarray) and val.dtype != object:
                return val if val.dtype == dtype else val.astype(dtype)
            if isinstance(val, np.ndarray):
                val = val.tolist()
            return np.array(val, dtype=dtype)

        ens_raw = d["ensemble"]
        if isinstance(ens_raw, np.ndarray) and ens_raw.dtype == EnsembleDtype:
            ensemble = ens_raw
        elif hasattr(ens_raw, "__len__") and len(ens_raw) > 0:
            ensemble = np.array([tuple(e) for e in ens_raw], dtype=EnsembleDtype)
        else:
            ensemble = np.array([], dtype=EnsembleDtype)

        return cls(
            atoms=[Atom.from_dict(a) if isinstance(a, dict) else a for a in d["atoms"]],
            bonds=[Bond.from_dict(b) if isinstance(b, dict) else b for b in d["bonds"]],
            residues=[Residue.from_dict(r) if isinstance(r, dict) else r for r in d["residues"]],
            chains=[Chain.from_dict(c) if isinstance(c, dict) else c for c in d["chains"]],
            coords=_np(d["coords"], np.float32),
            ensemble=ensemble,
            mask=_np(d["mask"], bool),
            bfactor=_np(d["bfactor"], np.float32),
            plddt=_np(d["plddt"], np.float32),
        )
