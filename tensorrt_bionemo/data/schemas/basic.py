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

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np


class PolymerType(str, Enum):
    """Polymer kind. Tells consumers how to interpret ``Polymer.sequence``.

    Sequence convention by type:

    - ``PROTEIN``: 1-letter amino-acid sequence (e.g. ``"ACDE"``).
    - ``RNA`` / ``DNA``: 1-letter nucleotide sequence (e.g. ``"AUGC"``).
    - ``CCD_LIGAND``: one CCD code or an underscore-joined list of CCD
      codes from the Chemical Component Dictionary (e.g. ``"ATP"`` or
      ``"ATP_FAD"`` for a multi-component ligand). Matches the
      Protenix lossless convention; upstream AF3/OF3 ``ccd_codes``
      lists round-trip through ``sequence.split("_")``.
    - ``SMILES_LIGAND``: a SMILES string for a custom small molecule
      (e.g. ``"CCO"`` for ethanol).
    """
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    CCD_LIGAND = "ccd_ligand"
    SMILES_LIGAND = "smiles_ligand"


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
        return [ResTypes.RA, ResTypes.RC, ResTypes.RG, ResTypes.RU]

    @staticmethod
    def dna_nucleotide_types() -> list[ResType]:
        return [ResTypes.DA, ResTypes.DC, ResTypes.DG, ResTypes.DT]

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


class MSARecord(dict):

    def __init__(
        self,
        content: Optional[str] = None,
        path: Optional[str] = None,
        format: str = "a3m",
    ):
        super().__init__(
            content=content,
            path=path,
            format=format,
        )

    def is_file(self) -> bool:
        return self["path"] is not None

    def get_content(self) -> str:
        if self["content"] is not None:
            return self["content"]
        if self["path"] is not None:
            with open(self["path"], "r") as f:
                self["content"] = f.read()
                return self["content"]
        raise ValueError("No content or file available")


class Template(dict):

    def __init__(self,
                 content: Optional[str] = None,
                 path: Optional[str] = None,
                 format: str = "cif"):
        super().__init__(content=content, path=path, format=format)

    def is_file(self) -> bool:
        return self["path"] is not None

    def get_content(self) -> str:
        if self["content"] is not None:
            return self["content"]
        if self["path"] is not None:
            with open(self["path"], "r") as f:
                self["content"] = f.read()
                return self["content"]
        raise ValueError("No content or file available")


class Polymer(dict):

    def __init__(self,
                 polymer_type: Union[PolymerType, str] = PolymerType.PROTEIN,
                 chain_id: Optional[Union[str, List[str]]] = None,
                 sequence: Optional[str] = None,
                 msas: Optional[List[MSARecord]] = None,
                 paired_msas: Optional[List[MSARecord]] = None,
                 templates: Optional[List[Template]] = None):
        if isinstance(polymer_type, str):
            polymer_type = PolymerType(polymer_type)

        self._validate_chain_id(chain_id)
        self._validate_polymer_fields(polymer_type, sequence, templates)

        super().__init__(polymer_type=polymer_type.value if isinstance(
            polymer_type, PolymerType) else polymer_type,
                         chain_id=chain_id,
                         sequence=sequence,
                         msas=msas,
                         paired_msas=paired_msas,
                         templates=templates)

    @staticmethod
    def _validate_chain_id(chain_id: Optional[Union[str, List[str]]]) -> None:
        if chain_id is None:
            return

        pattern = re.compile(r'^[A-Za-z0-9]{1,4}$')

        if isinstance(chain_id, str):
            if not pattern.match(chain_id):
                raise ValueError(
                    f"Chain ID '{chain_id}' is invalid. Must be 1-4 alphanumeric characters."
                )
        elif isinstance(chain_id, list):
            if len(chain_id) == 0:
                raise ValueError("Chain ID list cannot be empty.")
            for idx, cid in enumerate(chain_id):
                if not isinstance(cid, str):
                    raise ValueError(
                        f"Chain ID at index {idx} must be a string, got {type(cid).__name__}"
                    )
                if not pattern.match(cid):
                    raise ValueError(
                        f"Chain ID '{cid}' at index {idx} is invalid. Must be 1-4 alphanumeric characters."
                    )
        else:
            raise ValueError(
                f"Chain ID must be a string or list of strings, got {type(chain_id).__name__}"
            )

    _CCD_LIGAND_PATTERN = re.compile(r'^[A-Z0-9]{1,5}(?:_[A-Z0-9]{1,5})*$')

    @staticmethod
    def _validate_polymer_fields(polymer_type: PolymerType,
                                 sequence: Optional[str],
                                 templates: Optional[List]) -> None:
        if sequence is None:
            raise ValueError(f"{polymer_type.value} must have 'sequence'")

        if polymer_type == PolymerType.CCD_LIGAND:
            if not Polymer._CCD_LIGAND_PATTERN.match(sequence):
                raise ValueError(
                    f"ccd_ligand sequence {sequence!r} must be one CCD code "
                    f"or an underscore-joined list of CCD codes "
                    f"(uppercase A-Z0-9, each token 1-5 chars), "
                    f"e.g. 'ATP' or 'ATP_FAD'.")

        if templates is not None and len(templates) > 0:
            if polymer_type != PolymerType.PROTEIN:
                raise ValueError(
                    f"Templates are only allowed for protein molecules. "
                    f"Polymer type is '{polymer_type.value}' but templates were provided."
                )

    def get_chain_count(self) -> int:
        if self['chain_id'] is None:
            return 1
        if isinstance(self['chain_id'], list):
            return len(self['chain_id'])
        return 1


class InputRequest(dict):

    def __init__(
        self,
        input_id: Optional[str] = None,
        polymers: Optional[List[Polymer]] = None,
    ):
        super().__init__(
            input_id=input_id,
            polymers=polymers,
        )


class MSAParsed(dict):
    """Parsed A3M MSA file.

    Contains:
        sequences: aligned sequences (lowercase deletions removed)
        raw: original sequences with lowercase letters (deletion info)
        descriptions: sequence descriptions
        comments: optional list of comment lines (starting with '#') from the file
    """

    def __init__(self,
                 sequences: List[str],
                 raw: List[str],
                 descriptions: Optional[List[str]] = None,
                 comments: Optional[List[str]] = None):
        super().__init__(sequences=sequences,
                         raw=raw,
                         descriptions=descriptions,
                         comments=comments)

    @staticmethod
    def concat(msas: Optional[List['MSAParsed']]) -> Optional['MSAParsed']:
        # Validate for falsy input
        if not msas:
            return None

        # Return the sole element when only one MSA
        if len(msas) == 1:
            return msas[0]

        # Flatten all sequences
        sequences = [seq for msa in msas for seq in msa['sequences']]

        # Flatten all raw strings
        raw = [raw_seq for msa in msas for raw_seq in msa['raw']]

        # Concatenate descriptions - None only when no descriptions exist
        has_any_descriptions = any(msa['descriptions'] is not None
                                   for msa in msas)

        if has_any_descriptions:
            descriptions = []
            for msa in msas:
                if msa['descriptions'] is not None:
                    descriptions.extend(msa['descriptions'])
                else:
                    # Add empty strings as placeholders for MSAs without descriptions
                    descriptions.extend([''] * len(msa['sequences']))
        else:
            descriptions = None

        return MSAParsed(sequences=sequences,
                         raw=raw,
                         descriptions=descriptions)


class TemplateParsed(dict):

    def __init__(self, content: Optional[str] = None, format: str = "cif"):
        super().__init__(content=content, format=format)


class PolymerParsed(dict):

    def __init__(self,
                 polymer_type: Union[PolymerType, str] = PolymerType.PROTEIN,
                 chain_id: Optional[Union[str, List[str]]] = None,
                 sequence: Optional[str] = None,
                 msas: Optional[List[MSAParsed]] = None,
                 paired_msas: Optional[List[MSAParsed]] = None,
                 templates: Optional[List[TemplateParsed]] = None):
        super().__init__(polymer_type=polymer_type.value if isinstance(
            polymer_type, PolymerType) else polymer_type,
                         chain_id=chain_id,
                         sequence=sequence,
                         msas=msas,
                         paired_msas=paired_msas,
                         templates=templates)


class InputParsed(dict):

    def __init__(
        self,
        input_id: Optional[str] = None,
        polymers: Optional[List[PolymerParsed]] = None,
    ):
        super().__init__(input_id=input_id, polymers=polymers)


class FoldingOutput(dict):

    def __init__(
        self,
        atom_positions: np.ndarray,
        residue_types: np.ndarray,
        atom_mask: np.ndarray,
        residue_indices: np.ndarray,
        b_factors: Optional[np.ndarray] = None,
        chain_indices: Optional[np.ndarray] = None,
        plddt: Optional[np.ndarray] = None,
        ptm: Optional[float] = None,
        iptm: Optional[float] = None,
        pae: Optional[np.ndarray] = None,
        max_pae: Optional[float] = None,
    ):
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
            plddt: (num_res)
                Predicted Local Distance Difference Test score per residue, ranging
                from 0 to 100. Higher values indicate greater confidence in the
                predicted position of each residue. Scores above 90 are considered
                high confidence, 70-90 moderate, and below 70 low confidence.
            ptm: scalar
                Predicted Template Modeling (pTM) score between 0 and 1, estimating
                the overall global quality of the predicted structure. Higher values
                indicate better predicted alignment to the true structure.
            iptm: scalar
                Interface predicted Template Modeling (ipTM) score between 0 and 1,
                estimating the accuracy of predicted inter-chain interfaces in
                multimer predictions. Only meaningful for multi-chain structures.
            pae: (num_res, num_res)
                Predicted Aligned Error matrix in angstroms. Entry (i, j) represents
                the expected positional error at residue i when the predicted and
                true structures are aligned on residue j. Lower values indicate
                higher confidence in the relative positioning of residue pairs.
            max_pae: scalar
                Maximum possible Predicted Aligned Error value in angstroms, used
                for normalizing the PAE matrix.
        """
        super().__init__(atom_positions=atom_positions,
                         residue_types=residue_types,
                         atom_mask=atom_mask,
                         residue_indices=residue_indices,
                         b_factors=b_factors,
                         chain_indices=chain_indices,
                         plddt=plddt,
                         ptm=ptm,
                         iptm=iptm,
                         pae=pae,
                         max_pae=max_pae)

    def get_scores(self) -> dict:
        # Ensure all scores are json-able.
        ptm = None
        iptm = None
        max_pae = None
        plddt = None
        pae = None
        if self["plddt"] is not None:
            if isinstance(self["plddt"], np.ndarray):
                plddt = self["plddt"].tolist()
        if self["pae"] is not None:
            if isinstance(self["pae"], np.ndarray):
                pae = self["pae"].tolist()
        if self["ptm"] is not None and not np.isnan(self["ptm"]):
            ptm = float(self["ptm"])
        if self["iptm"] is not None and not np.isnan(self["iptm"]):
            iptm = float(self["iptm"])
        if self["max_pae"] is not None and not np.isnan(self["max_pae"]):
            max_pae = float(self["max_pae"])
        return {
            "plddt": plddt,
            "ptm": ptm,
            "iptm": iptm,
            "pae": pae,
            "max_pae": max_pae,
        }
