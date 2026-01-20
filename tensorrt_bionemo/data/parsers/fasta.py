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
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO, Union

from Bio import SeqIO

from tensorrt_bionemo.data.schemas import Polymer, PolymerType

_alphabetical_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SequenceParsed(dict):

    def __init__(self,
                 sequences: list[Polymer],
                 descriptions: Optional[list[str]] = None):
        super().__init__(sequences=sequences, descriptions=descriptions or [])


def parse_fasta_content(
    content: StringIO | TextIO,
    return_as_list: bool = False,
    is_description_formatted: bool = False
) -> Union[SequenceParsed, tuple[list[str], list[str]]]:
    if isinstance(content, str):
        content = StringIO(content)
    fasta_sequences = SeqIO.parse(content, "fasta")
    pattern = r'^([^|]+)\|([^|]+)\|([^|]+)$'
    sequences: list[Polymer] = []
    descriptions: list[str] = []

    if return_as_list:
        seqs = []
        desps = []
        for fasta in fasta_sequences:
            seqs.append(str(fasta.seq))
            desps.append(fasta.description)
        return seqs, desps

    for i, fasta in enumerate(fasta_sequences):
        desp = fasta.description
        seq = str(fasta.seq)

        match = None
        if is_description_formatted:
            match = re.match(pattern, desp)

        if match is not None:
            chain_id, entity_type_str, msa_id = match.groups()
            molecule_type = PolymerType(entity_type_str.lower())
            molecule = Polymer(polymer_type=molecule_type, chain_id=chain_id, sequence=seq)
        else:
            id_letter = _alphabetical_order[i % len(_alphabetical_order)]
            id_number = i // len(_alphabetical_order)
            if id_number == 0:
                chain_id = id_letter
            else:
                chain_id = f"{id_letter}{id_number}"

            molecule = Polymer(
                polymer_type=PolymerType.PROTEIN,
                chain_id=chain_id,
                sequence=seq
            )

        sequences.append(molecule)
        descriptions.append(desp)

    return SequenceParsed(sequences=sequences, descriptions=descriptions)


def read_fasta(file_path: str | Path,
               is_description_formatted: bool = False) -> SequenceParsed:
    with open(file_path) as source:
        return parse_fasta_content(
            source, is_description_formatted=is_description_formatted)
