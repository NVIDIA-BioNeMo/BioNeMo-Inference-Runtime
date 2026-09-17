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

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from bionemo_ir.data.path import resolve_input_path

_logger = logging.getLogger(__name__)


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
    """Atom-name universe used by FoldingOutput / writers.

    Indices 0-36 cover the 37 AlphaFold-style protein atoms. Newer indices
    37+ cover nucleic-acid backbones, nucleobase atoms, and the common
    PDB/CCD ligand atom names (numeric suffixes on C/N/O/S/P, primed
    sugar names). Adding entries here is additive — the protein order
    stays fixed so existing models that consume the 37-slot layout keep
    their indices, and the new entries simply extend the universe for
    non-protein chains.
    """

    # ----- Protein (indices 0-36) — DO NOT REORDER ---------------------
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

    # ----- Nucleic-acid backbone (sugar + phosphate) ------------------
    P = AtomType(name="P")
    OP1 = AtomType(name="OP1")
    OP2 = AtomType(name="OP2")
    OP3 = AtomType(name="OP3")
    O5_prime = AtomType(name="O5'")
    C5_prime = AtomType(name="C5'")
    C4_prime = AtomType(name="C4'")
    O4_prime = AtomType(name="O4'")
    C3_prime = AtomType(name="C3'")
    O3_prime = AtomType(name="O3'")
    C2_prime = AtomType(name="C2'")
    O2_prime = AtomType(name="O2'")
    C1_prime = AtomType(name="C1'")
    S5_prime = AtomType(name="S5'")  # SAM/SAH-style methionine-adenosine link

    # ----- Nucleobase atoms (purines + pyrimidines + thymine methyl) --
    N1 = AtomType(name="N1")
    N2 = AtomType(name="N2")
    N3 = AtomType(name="N3")
    N4 = AtomType(name="N4")
    N6 = AtomType(name="N6")
    N7 = AtomType(name="N7")
    N9 = AtomType(name="N9")
    N11 = AtomType(name="N11")  # e.g. PRF (a CASP15 RNA-ligand cofactor)
    C2 = AtomType(name="C2")
    C4 = AtomType(name="C4")
    C5 = AtomType(name="C5")
    C6 = AtomType(name="C6")
    C7 = AtomType(name="C7")  # thymine methyl
    C8 = AtomType(name="C8")
    O2 = AtomType(name="O2")
    O4 = AtomType(name="O4")
    O6 = AtomType(name="O6")

    # ----- Common ligand atom names (carbohydrates, cofactors, etc.) --
    C1 = AtomType(name="C1")
    C3 = AtomType(name="C3")
    C9 = AtomType(name="C9")
    C10 = AtomType(name="C10")
    C11 = AtomType(name="C11")
    C12 = AtomType(name="C12")
    C13 = AtomType(name="C13")
    C14 = AtomType(name="C14")
    C15 = AtomType(name="C15")
    C16 = AtomType(name="C16")
    C17 = AtomType(name="C17")
    C18 = AtomType(name="C18")
    C19 = AtomType(name="C19")
    C20 = AtomType(name="C20")
    O1 = AtomType(name="O1")
    O3 = AtomType(name="O3")
    O5 = AtomType(name="O5")
    O7 = AtomType(name="O7")
    O8 = AtomType(name="O8")
    O9 = AtomType(name="O9")
    N5 = AtomType(name="N5")
    N8 = AtomType(name="N8")
    N10 = AtomType(name="N10")
    S1 = AtomType(name="S1")
    S2 = AtomType(name="S2")
    P1 = AtomType(name="P1")
    P2 = AtomType(name="P2")

    # ----- Nucleotide-triphosphate ligand atoms ----------------------
    PA = AtomType(name="PA")
    PB = AtomType(name="PB")
    PG = AtomType(name="PG")
    O1A = AtomType(name="O1A")
    O2A = AtomType(name="O2A")
    O3A = AtomType(name="O3A")
    O1B = AtomType(name="O1B")
    O2B = AtomType(name="O2B")
    O3B = AtomType(name="O3B")
    O1G = AtomType(name="O1G")
    O2G = AtomType(name="O2G")
    O3G = AtomType(name="O3G")

    _all_types = [
        # protein (indices 0-36)
        N,
        CA,
        C,
        CB,
        O,
        CG,
        CG1,
        CG2,
        OG,
        OG1,
        SG,
        CD,
        CD1,
        CD2,
        ND1,
        ND2,
        OD1,
        OD2,
        SD,
        CE,
        CE1,
        CE2,
        CE3,
        NE,
        NE1,
        NE2,
        OE1,
        OE2,
        CH2,
        NH1,
        NH2,
        OH,
        CZ,
        CZ2,
        CZ3,
        NZ,
        OXT,
        # nucleic backbone (37-50)
        P,
        OP1,
        OP2,
        OP3,
        O5_prime,
        C5_prime,
        C4_prime,
        O4_prime,
        C3_prime,
        O3_prime,
        C2_prime,
        O2_prime,
        C1_prime,
        S5_prime,
        # nucleobases (51-68)
        N1,
        N2,
        N3,
        N4,
        N6,
        N7,
        N9,
        N11,
        C2,
        C4,
        C5,
        C6,
        C7,
        C8,
        O2,
        O4,
        O6,
        # common ligand atoms (69-)
        C1,
        C3,
        C9,
        C10,
        C11,
        C12,
        C13,
        C14,
        C15,
        C16,
        C17,
        C18,
        C19,
        C20,
        O1,
        O3,
        O5,
        O7,
        O8,
        O9,
        N5,
        N8,
        N10,
        S1,
        S2,
        P1,
        P2,
        # nucleotide-triphosphate ligands
        PA,
        PB,
        PG,
        O1A,
        O2A,
        O3A,
        O1B,
        O2B,
        O3B,
        O1G,
        O2G,
        O3G,
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
            A,
            R,
            N,
            D,
            C,
            Q,
            E,
            G,
            H,
            I,
            L,
            K,
            M,
            F,
            P,
            S,
            T,
            W,
            Y,
            V,
            RA,
            RC,
            RG,
            RU,
            RX,
            DA,
            DC,
            DG,
            DT,
            DX,
            GAP,
            PAD,
        ]
    }
    _by_canonical_name = {
        res.canonical_name: res
        for res in [
            A,
            R,
            N,
            D,
            C,
            Q,
            E,
            G,
            H,
            I,
            L,
            K,
            M,
            F,
            P,
            S,
            T,
            W,
            Y,
            V,
            RA,
            RC,
            RG,
            RU,
            RX,
            DA,
            DC,
            DG,
            DT,
            DX,
            GAP,
            PAD,
        ]
    }

    @staticmethod
    def basic_20_residue_types() -> list[ResType]:
        return [
            ResTypes.A,
            ResTypes.R,
            ResTypes.N,
            ResTypes.D,
            ResTypes.C,
            ResTypes.Q,
            ResTypes.E,
            ResTypes.G,
            ResTypes.H,
            ResTypes.I,
            ResTypes.L,
            ResTypes.K,
            ResTypes.M,
            ResTypes.F,
            ResTypes.P,
            ResTypes.S,
            ResTypes.T,
            ResTypes.W,
            ResTypes.Y,
            ResTypes.V,
        ]

    @staticmethod
    def rna_nucleotide_types() -> list[ResType]:
        return [ResTypes.RA, ResTypes.RC, ResTypes.RG, ResTypes.RU]

    @staticmethod
    def dna_nucleotide_types() -> list[ResType]:
        return [ResTypes.DA, ResTypes.DC, ResTypes.DG, ResTypes.DT]

    @staticmethod
    def from_string(s: str, return_unknown: bool = False, construct: bool = False) -> ResType:
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
        return res in ResTypes.rna_nucleotide_types() or res in ResTypes.dna_nucleotide_types()


class MSARecord(dict):
    def __init__(
        self,
        content: str | None = None,
        path: str | None = None,
        format: str = "a3m",
    ):
        super().__init__(
            content=content,
            path=path,
            format=format,
        )

    def is_file(self) -> bool:
        return self["path"] is not None

    def get_content(self, allowed_root: str | Path | None = None) -> str:
        """Return inline content or read it from an optionally confined path."""
        if self["content"] is not None:
            return self["content"]
        if self["path"] is not None:
            path = resolve_input_path(self["path"], allowed_root)
            with path.open() as f:
                self["content"] = f.read()
                return self["content"]
        raise ValueError("No content or file available")


class Template(dict):
    def __init__(
        self,
        content: str | None = None,
        path: str | None = None,
        format: str = "cif",  # noqa: A002 — public field name mirrors the JSON schema "format" key; callers pass format=
        chain_id: str | None = None,
    ):
        # ``chain_id`` selects which chain of a multi-chain template CIF to use;
        # ``None`` = auto-select the best-aligning chain.
        super().__init__(content=content, path=path, format=format, chain_id=chain_id)

    def is_file(self) -> bool:
        return self["path"] is not None

    def get_content(self, allowed_root: str | Path | None = None) -> str:
        """Return inline content or read it from an optionally confined path."""
        if self["content"] is not None:
            return self["content"]
        if self["path"] is not None:
            path = resolve_input_path(self["path"], allowed_root)
            with path.open() as f:
                self["content"] = f.read()
                return self["content"]
        raise ValueError("No content or file available")


class Polymer(dict):
    def __init__(
        self,
        polymer_type: PolymerType | str = PolymerType.PROTEIN,
        chain_id: str | list[str] | None = None,
        sequence: str | None = None,
        msas: list[MSARecord] | None = None,
        paired_msas: list[MSARecord] | None = None,
        templates: list[Template] | None = None,
    ):
        if isinstance(polymer_type, str):
            polymer_type = PolymerType(polymer_type)

        self._validate_chain_id(chain_id)
        self._validate_polymer_fields(polymer_type, sequence, templates)
        sequence = self._normalize_polymer_sequence(polymer_type, chain_id, sequence)

        super().__init__(
            polymer_type=polymer_type.value if isinstance(polymer_type, PolymerType) else polymer_type,
            chain_id=chain_id,
            sequence=sequence,
            msas=msas,
            paired_msas=paired_msas,
            templates=templates,
        )

    @staticmethod
    def _validate_chain_id(chain_id: str | list[str] | None) -> None:
        if chain_id is None:
            return

        pattern = re.compile(r"^[A-Za-z0-9]{1,4}$")

        if isinstance(chain_id, str):
            if not pattern.match(chain_id):
                raise ValueError(f"Chain ID '{chain_id}' is invalid. Must be 1-4 alphanumeric characters.")
        elif isinstance(chain_id, list):
            if len(chain_id) == 0:
                raise ValueError("Chain ID list cannot be empty.")
            for idx, cid in enumerate(chain_id):
                if not isinstance(cid, str):
                    raise ValueError(f"Chain ID at index {idx} must be a string, got {type(cid).__name__}")
                if not pattern.match(cid):
                    raise ValueError(
                        f"Chain ID '{cid}' at index {idx} is invalid. Must be 1-4 alphanumeric characters."
                    )
        else:
            raise ValueError(f"Chain ID must be a string or list of strings, got {type(chain_id).__name__}")

    _CCD_LIGAND_PATTERN = re.compile(r"^[A-Z0-9]{1,5}(?:_[A-Z0-9]{1,5})*$")

    # (canonical, ambiguous, unknown) per sequence polymer type. Ambiguous
    # letters are rewritten to the unknown residue with a warning: no backend
    # models them, and OpenFold2 indexes its restype table directly, so passing
    # one through reaches feature generation as a KeyError.
    _SEQUENCE_ALPHABETS = {
        PolymerType.PROTEIN.value: ("ACDEFGHIKLMNPQRSTVWXY", "BJOUZ", "X"),
        PolymerType.RNA.value: ("ACGNU", "BDHKMRSTVWY", "N"),
        PolymerType.DNA.value: ("ACGNT", "BDHKMRSUVWY", "N"),
    }

    @staticmethod
    def _normalize_polymer_sequence(
        polymer_type: PolymerType, chain_id: str | list[str] | None, sequence: str | None
    ) -> str | None:
        """Normalise a sequence polymer and reject letters outside its alphabet.

        Only ``PROTEIN`` / ``RNA`` / ``DNA`` are touched. A SMILES string is
        case-sensitive (``C`` is aliphatic carbon, ``c`` aromatic) and a CCD code
        has its own pattern, so both are returned unchanged.

        Without this check a stray character reached the backends and was folded
        into the unknown residue -- silently in the Boltz-2 path, with a log line
        in the OpenFold3 one -- so malformed input produced a plausible-looking
        structure instead of an error.

        Raises:
            ValueError: A character outside the polymer's alphabet.
        """
        alphabets = Polymer._SEQUENCE_ALPHABETS.get(polymer_type.value)
        if alphabets is None or sequence is None:
            return sequence
        canonical, ambiguous, unknown = alphabets

        where = f" of chain {chain_id!r}" if chain_id is not None else ""
        residues: list[str] = []
        invalid: list[tuple[int, str]] = []
        lowered = False
        degraded: set[str] = set()

        for position, original in enumerate(sequence, start=1):
            # Upper-case one ASCII letter at a time. str.upper() over the whole
            # string folds 'ß' to 'SS' and 'ﬃ' to 'FFI', which would turn
            # rejected input into accepted residues and change the chain length.
            letter = original.upper() if "a" <= original <= "z" else original
            lowered = lowered or letter != original
            if letter in canonical:
                residues.append(letter)
            elif letter in ambiguous:
                degraded.add(letter)
                residues.append(unknown)
            else:
                invalid.append((position, original))

        if invalid:
            detail = ", ".join(f"{c!r} at position {i}" for i, c in invalid[:5])
            if len(invalid) > 5:
                detail += f", and {len(invalid) - 5} more"
            raise ValueError(
                f"Invalid {polymer_type.value} sequence{where}: {detail}. "
                f"A {polymer_type.value} sequence must use the one-letter residue codes "
                f"{canonical} (ambiguity codes {ambiguous} "
                f"are accepted and read as unknown)."
            )

        if lowered:
            _logger.warning(
                "Lower-case residues in the %s sequence%s were upper-cased; a primary "
                "sequence carries no alignment case convention.",
                polymer_type.value,
                where,
            )
        if degraded:
            _logger.warning(
                "Ambiguity codes %s in the %s sequence%s were replaced with %r.",
                ", ".join(repr(c) for c in sorted(degraded)),
                polymer_type.value,
                where,
                unknown,
            )

        return "".join(residues)

    @staticmethod
    def msa_row_alphabet(polymer_type: PolymerType) -> frozenset[str] | None:
        """Characters an A3M row may carry, or ``None`` for a non-sequence polymer.

        The residue codes of the primary-sequence contract, their lower-case
        forms, and the gap. Derived from the same table so the two paths cannot
        drift apart.
        """
        alphabets = Polymer._SEQUENCE_ALPHABETS.get(polymer_type.value)
        if alphabets is None:
            return None
        canonical, ambiguous, _ = alphabets
        letters = canonical + ambiguous
        return frozenset(letters + letters.lower() + "-")

    @staticmethod
    def validate_msa_row(
        polymer_type: PolymerType, chain_id: str | list[str] | None, row: int, sequence: str | None
    ) -> None:
        """Reject symbols an A3M row may not carry, without rewriting it.

        Case is load-bearing here and must survive untouched: an upper-case
        letter is an aligned residue, a lower-case one an insertion relative to
        the query that ``generate_deletion_matrix`` counts, and ``-`` an aligned
        gap. So this only rejects. Left unchecked, a stray symbol reaches the
        backends as an unknown residue under Boltz-2 and OpenFold3 and as a
        ``KeyError`` under OpenFold2 -- the split this validation exists to end.

        Raises:
            ValueError: A character outside the polymer's MSA alphabet.
        """
        allowed = Polymer.msa_row_alphabet(polymer_type)
        if allowed is None or not sequence:
            return
        invalid = [(position, c) for position, c in enumerate(sequence, start=1) if c not in allowed]
        if not invalid:
            return
        detail = ", ".join(f"{c!r} at position {position}" for position, c in invalid[:5])
        if len(invalid) > 5:
            detail += f", and {len(invalid) - 5} more"
        raise ValueError(
            f"Invalid {polymer_type.value} MSA row {row} of chain {chain_id!r}: {detail}. "
            f"An MSA row may carry the residue codes {''.join(sorted(allowed - {'-'}))}, "
            f"their lower-case forms for insertions, and '-' for a gap."
        )

    @staticmethod
    def _validate_polymer_fields(polymer_type: PolymerType, sequence: str | None, templates: list | None) -> None:
        if sequence is None:
            raise ValueError(f"{polymer_type.value} must have 'sequence'")

        # Guard every polymer type, not just the ones with an alphabet: a
        # non-string otherwise reaches re.match as a TypeError for a CCD code,
        # str.upper as an AttributeError for a sequence polymer, and nothing at
        # all for SMILES. Only ValueError carries the input id added downstream.
        if not isinstance(sequence, str):
            raise ValueError(f"{polymer_type.value} 'sequence' must be a string, got {type(sequence).__name__}")

        if polymer_type == PolymerType.CCD_LIGAND:
            if not Polymer._CCD_LIGAND_PATTERN.match(sequence):
                raise ValueError(
                    f"ccd_ligand sequence {sequence!r} must be one CCD code "
                    f"or an underscore-joined list of CCD codes "
                    f"(uppercase A-Z0-9, each token 1-5 chars), "
                    f"e.g. 'ATP' or 'ATP_FAD'."
                )

        if templates is not None and len(templates) > 0:
            if polymer_type != PolymerType.PROTEIN:
                raise ValueError(
                    f"Templates are only allowed for protein molecules. "
                    f"Polymer type is '{polymer_type.value}' but templates were provided."
                )


class InputRequest(dict):
    def __init__(
        self,
        input_id: str | None = None,
        polymers: list[Polymer] | None = None,
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

    def __init__(
        self,
        sequences: list[str],
        raw: list[str],
        descriptions: list[str] | None = None,
        comments: list[str] | None = None,
    ):
        super().__init__(sequences=sequences, raw=raw, descriptions=descriptions, comments=comments)

    @staticmethod
    def concat(msas: list["MSAParsed"] | None) -> Optional["MSAParsed"]:
        # Validate for falsy input
        if not msas:
            return None

        # Return the sole element when only one MSA
        if len(msas) == 1:
            return msas[0]

        # Flatten all sequences
        sequences = [seq for msa in msas for seq in msa["sequences"]]

        # Flatten all raw strings
        raw = [raw_seq for msa in msas for raw_seq in msa["raw"]]

        # Concatenate descriptions - None only when no descriptions exist
        has_any_descriptions = any(msa["descriptions"] is not None for msa in msas)

        if has_any_descriptions:
            descriptions = []
            for msa in msas:
                if msa["descriptions"] is not None:
                    descriptions.extend(msa["descriptions"])
                else:
                    # Add empty strings as placeholders for MSAs without descriptions
                    descriptions.extend([""] * len(msa["sequences"]))
        else:
            descriptions = None

        return MSAParsed(sequences=sequences, raw=raw, descriptions=descriptions)


class TemplateParsed(dict):
    def __init__(
        self,
        content: str | None = None,
        format: str = "cif",  # noqa: A002 — public field name mirrors the JSON schema "format" key; callers pass format=
        chain_id: str | None = None,
    ):
        super().__init__(content=content, format=format, chain_id=chain_id)


class PolymerParsed(dict):
    def __init__(
        self,
        polymer_type: PolymerType | str = PolymerType.PROTEIN,
        chain_id: str | list[str] | None = None,
        sequence: str | None = None,
        msas: list[MSAParsed] | None = None,
        paired_msas: list[MSAParsed] | None = None,
        templates: list[TemplateParsed] | None = None,
    ):
        super().__init__(
            polymer_type=polymer_type.value if isinstance(polymer_type, PolymerType) else polymer_type,
            chain_id=chain_id,
            sequence=sequence,
            msas=msas,
            paired_msas=paired_msas,
            templates=templates,
        )


class InputParsed(dict):
    def __init__(
        self,
        input_id: str | None = None,
        polymers: list[PolymerParsed] | None = None,
    ):
        super().__init__(input_id=input_id, polymers=polymers)


# Canonical mol-type encoding used by ``FoldingOutput.mol_types``. Producers
# whose internal encoding differs must remap to this convention in their
# postprocessor so downstream writers can rely on a single contract.
MOL_TYPE_PROTEIN = 0
MOL_TYPE_RNA = 1
MOL_TYPE_DNA = 2
MOL_TYPE_LIGAND = 3


class FoldingOutput(dict):
    def __init__(
        self,
        atom_positions: np.ndarray,
        residue_types: np.ndarray,
        atom_mask: np.ndarray,
        residue_indices: np.ndarray,
        b_factors: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,
        plddt: np.ndarray | None = None,
        ptm: float | None = None,
        iptm: float | None = None,
        pae: np.ndarray | None = None,
        max_pae: float | None = None,
        residue_names: list[str] | None = None,
        mol_types: np.ndarray | None = None,
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
            residue_names: (num_res,) list of strings, optional
                Per-residue CCD/PDB three-letter codes (``"ALA"``, ``"TYR"``,
                ``"SAH"``, ``"DA"``, ``"A"`` for RNA adenine, …). When
                provided, the CIF writer uses these for HETATM identity and
                Entity ``ChemComp`` construction instead of the protein
                heuristic. Producers that don't preserve CCD identity (e.g.
                AlphaFold2-style pipelines) leave this as ``None``.
            mol_types: (num_res,) integer ndarray, optional
                Per-residue molecule-type id (0=protein, 1=RNA, 2=DNA,
                3=ligand). Lets the writer classify chains explicitly rather
                than inferring from residue letters. When ``None``, writers
                fall back to the legacy classification heuristics.
        """
        super().__init__(
            atom_positions=atom_positions,
            residue_types=residue_types,
            atom_mask=atom_mask,
            residue_indices=residue_indices,
            b_factors=b_factors,
            chain_indices=chain_indices,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
            max_pae=max_pae,
            residue_names=residue_names,
            mol_types=mol_types,
        )

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
