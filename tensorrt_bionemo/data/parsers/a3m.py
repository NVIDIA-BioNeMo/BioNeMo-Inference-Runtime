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

import string
from io import StringIO
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from Bio import SeqIO

from tensorrt_bionemo.data.schemas.basic import MSAParsed
from tensorrt_bionemo.logger import logger


def generate_deletion_matrix(sequences: list[str]) -> torch.Tensor:
    deletion_matrix = []
    len_vec = None
    for i, msa_sequence in enumerate(sequences):
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        if len_vec is None:
            len_vec = len(deletion_vec)
        elif len(deletion_vec) != len_vec:
            logger.warning(f"Length of deletion vector is not consistent: {len(deletion_vec)} != {len_vec}, {i}")
        deletion_matrix.append(deletion_vec)
    ret = np.array(deletion_matrix)
    return ret


def parse_a3m_content(content: StringIO | TextIO, preserve_comments: bool = False) -> MSAParsed:
    """Parse A3M format MSA content.

    This parser filters out comment lines before parsing the FASTA-like content.
    Comment lines are identified as lines where lstrip().startswith("#") returns True.
    These lines are removed before calling Bio.SeqIO.parse to ensure compatibility
    with standard FASTA parsing.

    Args:
        content: A3M file content as StringIO or TextIO object
        preserve_comments: If True, collect and return comment lines in the result.
                          Default is False for backward compatibility.

    Returns:
        MSAParsed: Dictionary-like object containing:
            - sequences: aligned sequences (lowercase deletions removed)
            - raw: original sequences with lowercase letters (deletion info)
            - descriptions: sequence descriptions
            - comments: list of comment lines (only if preserve_comments=True)

    Note:
        Lines that match lstrip().startswith("#") are considered comments and
        are removed before FASTA parsing. If preserve_comments=True, these lines
        are collected (with leading/trailing whitespace stripped) and made available
        in the returned MSAParsed object.
    """
    if isinstance(content, str):
        content = StringIO(content)

    # Filter out comment lines starting with "#" and optionally collect them
    filtered_lines = []
    comment_lines = [] if preserve_comments else None

    for line in content:
        if line.lstrip().startswith("#"):
            if preserve_comments:
                comment_lines.append(line.strip())
        else:
            filtered_lines.append(line)

    content = StringIO("".join(filtered_lines))
    fasta_sequences = SeqIO.parse(content, "fasta")
    sequences = []
    descriptions = []
    for fasta in fasta_sequences:
        sequences.append(str(fasta.seq))
        descriptions.append(fasta.description)
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return MSAParsed(sequences=aligned_sequences, raw=sequences, descriptions=descriptions, comments=comment_lines)


def read_a3m(file_path: str | Path, preserve_comments: bool = False) -> MSAParsed:
    """Read and parse an A3M format MSA file.

    Args:
        file_path: Path to the A3M file
        preserve_comments: If True, collect and return comment lines in the result.
                          Default is False for backward compatibility.

    Returns:
        MSAParsed: Parsed MSA data with sequences, descriptions, and optionally comments.
    """
    with open(file_path) as f:
        return parse_a3m_content(f, preserve_comments=preserve_comments)


__all__ = ["MSAParsed", "generate_deletion_matrix", "parse_a3m_content", "read_a3m"]
