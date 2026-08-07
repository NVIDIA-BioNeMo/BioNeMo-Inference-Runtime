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

from io import StringIO
from pathlib import Path
from typing import TextIO

from Bio import SeqIO

from tensorrt_bionemo.data.schemas import Polymer, PolymerType

_alphabetical_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _generate_chain_id(index: int) -> str:
    """Generate a chain ID following mmCIF-style conventions.

    This uses bijective base-26 encoding with letters A-Z, following the mmCIF format:
    - Indices 0-25: A-Z (single letter)
    - Indices 26-701: AA-ZZ (two letters)
    - Indices 702-18277: AAA-ZZZ (three letters)
    - Indices 18278-475253: AAAA-ZZZZ (four letters)

    This ensures chain IDs are always 1-4 alphanumeric characters as required
    by Polymer._validate_chain_id. Supports up to 475,254 sequences.

    Args:
        index: Zero-based sequence index

    Returns:
        A chain ID string of 1-4 letters (e.g., "A", "Z", "AA", "ZZ", "ZZZZ")

    Raises:
        ValueError: If index exceeds the maximum supported value (475,253)

    Examples:
        >>> _generate_chain_id(0)
        'A'
        >>> _generate_chain_id(25)
        'Z'
        >>> _generate_chain_id(26)
        'AA'
        >>> _generate_chain_id(27)
        'AB'
        >>> _generate_chain_id(701)
        'ZZ'
        >>> _generate_chain_id(702)
        'AAA'
    """
    # Calculate maximum index for 4-character chain IDs
    # 26 + 26^2 + 26^3 + 26^4 - 1 = 475,253
    max_index = 26 + 26**2 + 26**3 + 26**4 - 1

    if index > max_index:
        raise ValueError(
            f"Sequence index {index} exceeds maximum supported value ({max_index}). "
            f"Cannot generate valid chain ID following mmCIF conventions. "
            f"Chain IDs are limited to 4 characters by Polymer._validate_chain_id."
        )

    # Bijective base-26: convert index to chain ID
    # Adjust index by adding 1 to convert from 0-indexed to bijective base-26
    num = index + 1
    result = []

    while num > 0:
        # Adjust for bijective base-26 (no zero digit)
        num -= 1
        result.append(_alphabetical_order[num % 26])
        num //= 26

    return "".join(reversed(result))


class SequenceParsed(dict):
    def __init__(self, sequences: list[Polymer], descriptions: list[str] | None = None):
        super().__init__(sequences=sequences, descriptions=descriptions or [])


def parse_fasta_content(
    content: StringIO | TextIO, return_as_list: bool = False
) -> SequenceParsed | tuple[list[str], list[str]]:
    """Parse a FASTA file into a SequenceParsed object for protein entity"""
    if isinstance(content, str):
        content = StringIO(content)

    # Filter out comment lines starting with "#"
    filtered_lines = [line for line in content if not line.lstrip().startswith("#")]
    content = StringIO("".join(filtered_lines))

    fasta_sequences = SeqIO.parse(content, "fasta")
    sequences: list[Polymer] = []
    descriptions: list[str] = []

    if return_as_list:
        seqs = []
        desps = []
        for fasta in fasta_sequences:
            seqs.append(str(fasta.seq))
            desps.append(fasta.description)
        return seqs, desps

    seen_sequences = {}
    for i, fasta in enumerate(fasta_sequences):
        desp = fasta.description
        seq = str(fasta.seq)

        # Generate chain ID using base-36 encoding to ensure it stays within 4 characters
        try:
            chain_id = _generate_chain_id(i)
        except ValueError as e:
            raise ValueError(f"Cannot generate valid chain ID for sequence at index {i}. {str(e)}") from e

        if seq in seen_sequences:
            p = seen_sequences[seq]
            chain_ids = p["chain_id"]
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            chain_ids.append(chain_id)
            seen_sequences[seq]["chain_id"] = chain_ids
            continue

        molecule = Polymer(polymer_type=PolymerType.PROTEIN, chain_id=chain_id, sequence=seq)
        sequences.append(molecule)
        descriptions.append(desp)
        seen_sequences[seq] = molecule

    return SequenceParsed(sequences=sequences, descriptions=descriptions)


def read_fasta(file_path: str | Path, return_as_list: bool = False) -> SequenceParsed:
    with open(file_path) as source:
        return parse_fasta_content(source, return_as_list=return_as_list)
