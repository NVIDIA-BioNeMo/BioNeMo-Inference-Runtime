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
from typing import List, TextIO, Union

import numpy as np
import torch
from Bio import SeqIO

from tensorrt_bionemo.data.schemas.basic import MSAParsed


def generate_deletion_matrix(sequences: list[str],
                             gpu_preferred: bool = False) -> torch.Tensor:
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)
    ret = np.array(deletion_matrix)
    return ret



def parse_a3m_content(content: Union[StringIO, TextIO]) -> MSAParsed:
    if isinstance(content, str):
        content = StringIO(content)
    fasta_sequences = SeqIO.parse(content, "fasta")
    sequences = []
    descriptions = []
    for fasta in fasta_sequences:
        sequences.append(str(fasta.seq))
        descriptions.append(fasta.description)
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return MSAParsed(sequences=aligned_sequences, raw=sequences, descriptions=descriptions)


def read_a3m(file_path: Union[str, Path]) -> MSAParsed:
    with open(file_path, "r") as f:
        return parse_a3m_content(f)


__all__ = ["MSAParsed", "generate_deletion_matrix", "parse_a3m_content", "read_a3m"]
