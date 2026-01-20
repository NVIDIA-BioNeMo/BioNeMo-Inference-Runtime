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
from typing import Optional, TextIO

import numpy as np
import torch

from tensorrt_bionemo.data.parsers.fasta import parse_fasta_content


class A3MParsed(dict):

    def __init__(self,
                 sequences: list[str],
                 raw: list[str],
                 descriptions: Optional[list[str]] = None):
        super().__init__(sequences=sequences,
                         raw=raw,
                         descriptions=descriptions)


def parse_a3m_content(content: StringIO | TextIO) -> A3MParsed:
    """
    Read an a3m file from a string or text stream and return a list of sequences.
    """

    sequences, descriptions = parse_fasta_content(content, return_as_list=True)
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    ret = A3MParsed(sequences=aligned_sequences,
                    raw=sequences,
                    descriptions=descriptions)
    return ret


def read_a3m(file_path: str | Path) -> A3MParsed:
    with open(file_path, "r") as f:
        return parse_a3m_content(f)
