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
"""OpenFold3 constants: residue types, element types, and feature dimensions."""

# ---------------------------------------------------------------------------
# Residue type mappings (32 classes for restype, 32 classes for MSA)
# Matches STANDARD_RESIDUES_WITH_GAP from OF3 residues.py
# ---------------------------------------------------------------------------

# 3-letter codes (used for restype encoding).
#
# Matches OSS ``STANDARD_RESIDUES_WITH_GAP_3`` at
# ``openfold3/core/data/resources/residues.py`` byte-for-byte: indices 25 and
# 30 are ``"N"`` (RNA any-nucleotide) and ``"DN"`` (DNA any-nucleotide). Note
# these intentionally differ from the ``canonical_name`` values on the
# corresponding ``ResTypes`` enum entries (``ResTypes.RX.canonical_name ==
# "RX"``, ``ResTypes.DX.canonical_name == "DX"``). The difference is handled
# on the writer side by ``_PDB_REMAP`` / ``_IHM_REMAP`` in
# ``data/writers/{pdb,cif}_writer.py``, which translate ``"RX" → "N"`` and
# ``"DX" → "DN"`` for emit. The current pipeline is protein-only, so the
# discrepancy is latent; if RNA/DNA ever flows through
# ``_build_structure_from_polymers``, either align ``ResTypes.{RX,DX}``
# canonical names with these strings or translate before ``RESNAME_TO_IDX``
# lookup.
RESTYPES_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",                               # index 20: unknown protein
    "A", "G", "C", "U", "N",            # indices 21-25: RNA
    "DA", "DG", "DC", "DT", "DN",       # indices 26-30: DNA
    "GAP",                                # index 31: gap
]

# Short codes (used for MSA and sequence encoding).
# Protein and RNA entries are single-letter; DNA entries use two-character
# codes (e.g. "DA", "DG", "DC", "DT", "DN").
RESTYPES_1 = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X",                                  # index 20: unknown protein
    "A", "G", "C", "U", "N",            # indices 21-25: RNA (same 1-letter)
    "DA", "DG", "DC", "DT", "DN",       # indices 26-30: DNA
    "-",                                  # index 31: gap
]

NUM_RESTYPE_CLASSES = 32
NUM_MSA_CLASSES = 32

# 3-letter code to restype index
RESNAME_TO_IDX = {name: i for i, name in enumerate(RESTYPES_3)}

# 1-letter protein code to restype index
_PROTEIN_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA_1_TO_IDX = {k: RESNAME_TO_IDX[v] for k, v in _PROTEIN_1TO3.items()}
AA_1_TO_IDX["X"] = 20  # unknown

# MSA character to index (protein only for now)
MSA_CHAR_TO_IDX = dict(AA_1_TO_IDX)
MSA_CHAR_TO_IDX["-"] = 31  # gap
MSA_CHAR_TO_IDX["."] = 31  # gap variant

# Gap index
GAP_IDX = 31
UNK_IDX = 20

# ---------------------------------------------------------------------------
# Element encoding (119 classes: periodic table elements 1-118 + unknown).
# One-hot channel dimension for ref_element is NUM_ELEMENT_CLASSES (= 119).
# Indices 0..117 are atomic numbers 1..118 (atomic_num - 1). Index 118
# (NUM_ELEMENT_CLASSES - 1) is reserved for the unknown class, though the
# current encoding path falls back to Carbon (index 5) for unmapped names.
# ---------------------------------------------------------------------------
NUM_ELEMENT_CLASSES = 119

# Common elements in amino acids
ELEMENT_TO_IDX = {
    "H": 0, "C": 5, "N": 6, "O": 7, "F": 8, "P": 14,
    "S": 15, "CL": 16, "SE": 33, "BR": 34, "I": 52,
}

# ---------------------------------------------------------------------------
# Atom name encoding (4 chars, 64 classes per char)
# ---------------------------------------------------------------------------
NUM_ATOM_NAME_CHARS = 4
NUM_CHAR_CLASSES = 64

# ---------------------------------------------------------------------------
# Template feature constants
# ---------------------------------------------------------------------------
TEMPLATE_DISTOGRAM_MIN_BIN = 3.25
TEMPLATE_DISTOGRAM_MAX_BIN = 50.75
TEMPLATE_DISTOGRAM_N_BINS = 39
DEFAULT_N_TEMPLATES = 4

# ---------------------------------------------------------------------------
# MSA constants
# ---------------------------------------------------------------------------
MAX_MSA_ROWS = 16384
MAX_MSA_ROWS_PAIRED = 8191

# ---------------------------------------------------------------------------
# Molecule type IDs (matching OF3 MoleculeType enum)
# ---------------------------------------------------------------------------
MOL_TYPE_PROTEIN = 0
MOL_TYPE_RNA = 1
MOL_TYPE_DNA = 2
MOL_TYPE_LIGAND = 3

POLYMER_TYPE_TO_MOL_TYPE = {
    "protein": MOL_TYPE_PROTEIN,
    "rna": MOL_TYPE_RNA,
    "dna": MOL_TYPE_DNA,
    "ccd_ligand": MOL_TYPE_LIGAND,
    "smiles_ligand": MOL_TYPE_LIGAND,
}

# ---------------------------------------------------------------------------
# Standard amino acid heavy atom names (backbone + side chain)
# Used to build atom arrays from sequence without external CCD lookups
# ---------------------------------------------------------------------------
PROTEIN_BACKBONE_ATOMS = ["N", "CA", "C", "O"]

# Standard amino acid heavy atoms in PDB order
AA_ATOMS = {
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
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
            "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2",
            "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

# Atom name → element symbol
ATOM_NAME_TO_ELEMENT = {
    "N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C",
    "CG": "C", "CG1": "C", "CG2": "C", "OG": "O", "OG1": "O",
    "SG": "S", "CD": "C", "CD1": "C", "CD2": "C", "ND1": "N",
    "ND2": "N", "OD1": "O", "OD2": "O", "SD": "S", "CE": "C",
    "CE1": "C", "CE2": "C", "CE3": "C", "NE": "N", "NE1": "N",
    "NE2": "N", "OE1": "O", "OE2": "O", "CH2": "C", "NH1": "N",
    "NH2": "N", "OH": "O", "CZ": "C", "CZ2": "C", "CZ3": "C",
    "NZ": "N", "OXT": "O",
}

# Element symbol → atomic number (1-indexed for periodic table)
ELEMENT_ATOMIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "CL": 17, "SE": 34, "BR": 35, "I": 53,
}
