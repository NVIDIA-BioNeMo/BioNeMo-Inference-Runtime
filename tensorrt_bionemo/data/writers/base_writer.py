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

from __future__ import annotations

from abc import ABC, abstractmethod

from tensorrt_bionemo.data.schemas.basic import (AtomType, AtomTypes,
                                                 FoldingOutput, ResType,
                                                 ResTypes)
from tensorrt_bionemo.logger import logger


# Shared helpers used by ``CIFWriter`` and ``PDBWriter`` to map residue
# names + classify chains consistently. Centralised here so the two writers
# can't drift apart on how they label RNA / DNA / non-polymer chains.
#
# Internal ResType ``name`` codes → CIF/IHM/PDB standard codes.
#
# - ``X``  → ``UNK``  (modelcif/ihm and PDB both use ``UNK`` for unknown
#   protein residues; Boltz2 also emits ``X`` for every per-atom-tokenised
#   ligand atom on a NONPOLYMER chain).
# - ``RA``/``RC``/``RG``/``RU`` → ``A``/``C``/``G``/``U`` (RNA: the bare
#   1-letter is what both ihm.RNAAlphabet and the PDB residue-name field
#   expect. Our internal ResTypes use the ``R`` prefix so the 1-letter
#   name doesn't collide with the 20-AA codes ``A`` and ``G``).
# - ``RX`` → ``N``, ``DX`` → ``DN`` (unknown nucleotide stand-ins).
# - DNA names (``DA``/``DC``/``DG``/``DT``) already match the CIF/PDB
#   convention and pass through unchanged.
_IHM_REMAP: dict[str, str] = {
    "X": "UNK",
    "RA": "A",
    "RC": "C",
    "RG": "G",
    "RU": "U",
    "RX": "N",
    "DX": "DN",
}

_RNA_RESNAMES: set[str] = {"RA", "RC", "RG", "RU", "RX"}
_DNA_RESNAMES: set[str] = {"DA", "DC", "DG", "DT", "DX"}
_PAD_RESNAMES: set[str] = {"-", "<PAD>"}

# Canonical mol-type ints in ``FoldingOutput`` (see
# ``tensorrt_bionemo/data/schemas/basic.py``): 0=protein, 1=RNA, 2=DNA,
# 3=ligand. Writers map these to their chain-classification labels.
_MOL_TYPE_TO_KIND: dict[int, str] = {
    0: "protein",
    1: "rna",
    2: "dna",
    3: "nonpoly",
}


def _classify_chain(restype_names: tuple[str, ...]) -> str:
    """Classify a chain by its residue-name set: ``protein|rna|dna|nonpoly``.

    Used by writers when the producer did not populate the explicit
    ``mol_types`` field on ``FoldingOutput``. A chain becomes:

    * ``rna`` if every non-pad residue is in ``_RNA_RESNAMES``,
    * ``dna`` if every non-pad residue is in ``_DNA_RESNAMES``,
    * ``nonpoly`` if every non-pad residue is ``X`` (Boltz2's NONPOLYMER
      tokenizer emits per-atom residues all of type ``X`` for ligand
      chains, so this heuristic catches them cleanly without changing
      the FoldingOutput schema),
    * ``protein`` otherwise (real protein chains contain at least one
      of the 20 amino acids beyond ``X``).
    """
    uniq = set(restype_names) - _PAD_RESNAMES
    if uniq and uniq.issubset(_RNA_RESNAMES):
        return "rna"
    if uniq and uniq.issubset(_DNA_RESNAMES):
        return "dna"
    if uniq and uniq.issubset({"X"}):
        return "nonpoly"
    return "protein"


class BaseWriter(ABC):
    """Writes a multi-chain protein structure to string and to file.

    Attributes:
        output_path (str): Path to output file created by write() method.
        res_type_mapping (dict[int, ResType]): Universe of existing residue types in the batch,
            represented as a map from integer to ResType
        atom_type_mapping (dict[int, AtomType]): Universe of existing atom types in the batch,
            represented as a map from integer to AtomType
        res_types (list[ResType]): Universe of existing residue types in the batch,
            represented as a 0-based list of ResType.  Assumes, res_type_mapping is 0-based.
        atom_types (list[AtomType]):  Universe of existing atom types in the batch,
            represented as a 0-based list of AtomType.  Assumes, atom_type_mapping is 0-based.
    """

    def __init__(
        self,
        res_type_mapping: dict[int, ResType],
        atom_type_mapping: dict[int, AtomType],
        output_path: str = "output.cif",
    ):
        """Initializes the BaseWriter with the given configuration..

        Args:
            res_type_mapping: Assigned to instance attribute.
            atom_type_mapping: Assigned to instance attribute.
            output_path: Assigned to instance attribute.
        Raises:
            None
        """
        logger.debug("BaseWriter.__init__() begin")

        if atom_type_mapping is None or res_type_mapping is None:
            raise ValueError("atom_type_mapping and res_type_mapping must be provided to dump PDB file")

        # process __init__ args
        self.output_path = output_path

        # Store the mappings as instance attributes
        self.res_type_mapping = res_type_mapping
        self.atom_type_mapping = atom_type_mapping

        # Build lists from mappings
        self.res_types = [
            y for _, y in sorted(self.res_type_mapping.items(),
                                 key=lambda pair: pair[0])
        ]
        self.atom_types = [
            y for _, y in sorted(self.atom_type_mapping.items(),
                                 key=lambda pair: pair[0])
        ]
        logger.debug("BaseWriter.__init__() end")

    def set_output_path(self, output_path: str):
        self.output_path = output_path

    @abstractmethod
    def write(self, folding_output: FoldingOutput) -> str:
        """Write the result of the network forward pass to file in the local
        environment.

        Args:
            folding_output: The result of the forward pass of a structure
                prediction network, e.g. OpenFold, or Boltz2
        """
