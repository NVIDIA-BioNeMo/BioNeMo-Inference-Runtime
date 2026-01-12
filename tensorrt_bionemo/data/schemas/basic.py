# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional

import numpy as np


class EntityType:
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"
    POLYMER_HYBRID = "polymer_hybrid"
    WATER = "water"
    UNKNOWN = "unknown"
    MANUAL_GLYCAN = "manual_glycan"


@dataclass
class AtomType:
    name: str

    def __eq__(self, other):
        if not isinstance(other, AtomType):
            return False
        return self.name == other.name


class AtomTypes:
    N = AtomType(name="N")
    CA = AtomType(name="CA")
    C = AtomType(name="C")
    CB = AtomType(name="CB")
    O = AtomType(name="O")
    CG = AtomType(name="CG")
    CG1 = AtomType(name="CG1")
    CG2 = AtomType(name="CG2")
    OG = AtomType(name="OG")
    OG1 = AtomType(name="OG1")
    SG = AtomType(name="SG")
    CD = AtomType(name="CD")
    CD1 = AtomType(name="CD1")
    CD2 = AtomType(name="CD2")
    ND1 = AtomType(name="ND1")
    ND2 = AtomType(name="ND2")
    OD1 = AtomType(name="OD1")
    OD2 = AtomType(name="OD2")
    SD = AtomType(name="SD")
    CE = AtomType(name="CE")
    CE1 = AtomType(name="CE1")
    CE2 = AtomType(name="CE2")
    CE3 = AtomType(name="CE3")
    NE = AtomType(name="NE")
    NE1 = AtomType(name="NE1")
    NE2 = AtomType(name="NE2")
    OE1 = AtomType(name="OE1")
    OE2 = AtomType(name="OE2")
    CH2 = AtomType(name="CH2")
    NH1 = AtomType(name="NH1")
    NH2 = AtomType(name="NH2")
    OH = AtomType(name="OH")
    CZ = AtomType(name="CZ")
    CZ2 = AtomType(name="CZ2")
    CZ3 = AtomType(name="CZ3")
    NZ = AtomType(name="NZ")
    OXT = AtomType(name="OXT")

    _all_types = [
        N, CA, C, CB, O, CG, CG1, CG2, OG, OG1, SG, CD, CD1, CD2, ND1, ND2,
        OD1, OD2, SD, CE, CE1, CE2, CE3, NE, NE1, NE2, OE1, OE2, CH2, NH1, NH2,
        OH, CZ, CZ2, CZ3, NZ, OXT
    ]
    _by_name = {atom.name: atom for atom in _all_types}

    @staticmethod
    def from_string(s: str) -> AtomType:
        return AtomTypes._by_name.get(s, None)

    @staticmethod
    def num_types() -> int:
        return len(AtomTypes._by_name)

    @staticmethod
    def all_types() -> list[AtomType]:
        return AtomTypes._all_types


# Using dataclass instead of BaseModel to avoid validation overhead
@dataclass
class ResType:
    name: str
    canonical_name: str

    def __eq__(self, other):
        if not isinstance(other, ResType):
            return False
        return self.name == other.name

    def is_gap(self) -> bool:
        return self.name == "-"


class ResTypes:
    A = ResType(name="A", canonical_name="ALA")
    R = ResType(name="R", canonical_name="ARG")
    N = ResType(name="N", canonical_name="ASN")
    D = ResType(name="D", canonical_name="ASP")
    C = ResType(name="C", canonical_name="CYS")
    Q = ResType(name="Q", canonical_name="GLN")
    E = ResType(name="E", canonical_name="GLU")
    G = ResType(name="G", canonical_name="GLY")
    H = ResType(name="H", canonical_name="HIS")
    I = ResType(name="I", canonical_name="ILE")
    L = ResType(name="L", canonical_name="LEU")
    K = ResType(name="K", canonical_name="LYS")
    M = ResType(name="M", canonical_name="MET")
    F = ResType(name="F", canonical_name="PHE")
    P = ResType(name="P", canonical_name="PRO")
    S = ResType(name="S", canonical_name="SER")  # codespell:ignore
    T = ResType(name="T", canonical_name="THR")
    W = ResType(name="W", canonical_name="TRP")
    Y = ResType(name="Y", canonical_name="TYR")
    V = ResType(name="V", canonical_name="VAL")
    X = ResType(name="X", canonical_name="UNK")
    RA = ResType(name="RA", canonical_name="A")  # RNA: Adenine
    RC = ResType(name="RC", canonical_name="C")  # RNA: Cytosine
    RG = ResType(name="RG", canonical_name="G")  # RNA: Guanine
    RU = ResType(name="RU", canonical_name="U")  # RNA: Uracil
    RX = ResType(name="RX", canonical_name="RX")  # RNA: Unknown
    DA = ResType(name="DA", canonical_name="DA")  # DNA: Adenine
    DC = ResType(name="DC", canonical_name="DC")  # DNA: Cytosine
    DG = ResType(name="DG", canonical_name="DG")  # DNA: Guanine
    DT = ResType(name="DT", canonical_name="DT")  # DNA: Thymine
    DX = ResType(name="DX", canonical_name="DX")  # DNA: Unknown
    GAP = ResType(name="-", canonical_name="GAP")  # Gap
    PAD = ResType(name="<PAD>", canonical_name="PAD")  # Unknown

    _by_name = {
        res.name: res
        for res in [
            A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, RA, RC,
            RG, RU, RX, DA, DC, DG, DT, DX, GAP, PAD
        ]
    }
    _by_canonical_name = {
        res.canonical_name: res
        for res in [
            A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, RA, RC,
            RG, RU, RX, DA, DC, DG, DT, DX, GAP, PAD
        ]
    }

    @staticmethod
    def basic_20_residue_types() -> list[ResType]:
        return [
            ResTypes.A, ResTypes.R, ResTypes.N, ResTypes.D, ResTypes.C,
            ResTypes.Q, ResTypes.E, ResTypes.G, ResTypes.H, ResTypes.I,
            ResTypes.L, ResTypes.K, ResTypes.M, ResTypes.F, ResTypes.P,
            ResTypes.S, ResTypes.T, ResTypes.W, ResTypes.Y, ResTypes.V
        ]

    @staticmethod
    def rna_nucleotide_types() -> list[ResType]:
        return [ResType.RA, ResType.RC, ResType.RG, ResType.RU]

    @staticmethod
    def dna_nucleotide_types() -> list[ResType]:
        return [ResType.DA, ResType.DC, ResType.DG, ResType.DT]

    @staticmethod
    def from_string(s: str,
                    return_unknown: bool = False,
                    construct: bool = False) -> ResType:
        if construct:
            return ResType(name=s, canonical_name=s)
        ret = ResTypes._by_name.get(s, None)
        if ret is None:
            ret = ResTypes._by_canonical_name.get(s, None)
        if ret is None:
            if return_unknown:
                return ResTypes.X
        return ret

    @staticmethod
    def is_peptide(res: ResType) -> bool:
        if isinstance(res, str):
            res = ResTypes.from_string(res, construct=True)
        return res in ResTypes.basic_20_residue_types()

    @staticmethod
    def is_nucleotide(res: ResType) -> bool:
        if isinstance(res, str):
            res = ResTypes.from_string(res, construct=True)
        return res in ResTypes.rna_nucleotide_types(
        ) or res in ResTypes.dna_nucleotide_types()


class Sequence(dict):

    def __init__(self,
                 residues: Optional[str] = None,
                 description: Optional[str] = None):
        super().__init__(residues=residues, description=description)


class InputChain(dict):

    def __init__(self,
                 sequence: Sequence,
                 chain_id: Optional[str] = 'A',
                 entity_type: str = EntityType.PROTEIN):
        super().__init__(sequence=sequence,
                         chain_id=chain_id,
                         entity_type=entity_type)


class StructureMetadata(dict):

    def __init__(self,
                 resolution: float,
                 release_date: str,
                 method: str,
                 file_id: Optional[str] = "sample"):
        super().__init__(resolution=resolution,
                         release_date=release_date,
                         method=method,
                         file_id=file_id)


class InputRequest(dict):

    def __init__(self,
                 fasta_file: str,
                 a3m_files: dict[str, list[str]],
                 mmcif_files: Optional[dict[str, list[str]]] = None,
                 is_description_formatted: bool = False,
                 is_files: bool = True):
        super().__init__(fasta_file=fasta_file,
                         a3m_files=a3m_files,
                         mmcif_files=mmcif_files,
                         is_description_formatted=is_description_formatted,
                         is_files=is_files)


class FoldingOutput(dict):

    def __init__(self,
                 atom_positions: np.ndarray,
                 residue_types: np.ndarray,
                 atom_mask: np.ndarray,
                 residue_indices: np.ndarray,
                 b_factors: Optional[np.ndarray] = None,
                 chain_indices: Optional[np.ndarray] = None):
        """
        Args:
            atom_positions: (num_res, num_atom_type, 3)
                Cartesian coordinates of atoms in angstroms
            residue_types: (num_res)
                Amino-acid type for each residue represented as an integer between 0 and
                20, where 20 is 'X'.
            atom_mask: (num_res, num_atom_type)
                Binary float mask to indicate presence of a particular atom. 1.0 if an atom
                is present and 0.0 if not. This should be used for loss masking.
            residue_indices: (num_res)
                Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
            b_factors: (num_res, num_atom_type)
                B-factors, or temperature factors, of each residue (in sq. angstroms units),
                representing the displacement of the residue from its ground truth mean value.
            chain_indices: (num_res)
                Chain indices for multi-chain predictions.
        """
        super().__init__(atom_positions=atom_positions,
                         residue_types=residue_types,
                         atom_mask=atom_mask,
                         residue_indices=residue_indices,
                         b_factors=b_factors,
                         chain_indices=chain_indices)
