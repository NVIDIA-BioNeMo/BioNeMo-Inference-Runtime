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

"""Protenix v2 residue, atom and MSA vocabularies.

Values follow ByteDance Protenix v2.0.0 ``protenix/data/constants.py``
(Apache-2.0); they are derived from the shared OpenFold3 and Boltz tables.
"""

from string import ascii_uppercase

from bionemo_ir.pipeline.models.boltz2.const import ref_atoms
from bionemo_ir.pipeline.models.openfold3.const import RESTYPES_1, RESTYPES_3

RESTYPES = {name: index for index, name in enumerate(RESTYPES_3[:31])} | {"-": 31}
PROTEIN_CODES = dict(zip(RESTYPES_1[:21], RESTYPES_3[:21], strict=True))
RNA_CODES = {c: c for c in "AGCUNI"} | {"X": "N"}
DNA_CODES = {c: "D" + c for c in "AGCTNIU"} | {"X": "DN"}
MOL_TYPES = {"protein": 0, "dna": 1, "rna": 2, "ccd_ligand": 3, "smiles_ligand": 3}


def _dense_atoms(name: str) -> list[str]:
    """Release dense atom order: Boltz order plus terminal OXT or 5' OP3."""
    if name == "UNK":
        return ["N", "CA", "C", "O", "CB", "CG", "OXT"]
    if RESTYPES[name] >= RESTYPES["A"]:
        return ["OP3", *ref_atoms[name]]
    return [*ref_atoms[name], "OXT"]


RES_ATOMS_DICT = {name: {atom: i for i, atom in enumerate(_dense_atoms(name))} for name in RESTYPES_3[:31]}

MSA_PROTEIN_SEQ_TO_ID = {c: i for i, c in enumerate(RESTYPES_1[:21])} | {
    "B": RESTYPES["ASP"],
    "Z": RESTYPES["GLU"],
    "U": RESTYPES["CYS"],
    "J": RESTYPES["UNK"],
    "O": RESTYPES["UNK"],
    "-": 31,
}
MSA_RNA_SEQ_TO_ID = dict.fromkeys(ascii_uppercase, RESTYPES["N"]) | {c: RESTYPES[c] for c in "AGCU"} | {"-": 31}
MSA_DNA_SEQ_TO_ID = dict.fromkeys(ascii_uppercase, RESTYPES["DN"]) | {c: RESTYPES["D" + c] for c in "AGCT"} | {"-": 31}
