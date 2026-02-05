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
from functools import lru_cache

import torch
import torch.nn.functional as F

from .schemas import AtomType, AtomTypes, ResType, ResTypes


@lru_cache
def get_all_residue_types(model: str,
                          include_gap: bool = True) -> list[ResType]:
    if "openfold2" in model or "alphafold2" in model:
        ret = ResTypes.basic_20_residue_types() + [ResTypes.X]
        if include_gap:
            ret.append(ResTypes.GAP)
        return ret
    elif "boltz" in model:
        return [ResTypes.PAD, ResTypes.GAP] + \
            ResTypes.basic_20_residue_types() + [ResTypes.X] + \
            ResTypes.rna_nucleotide_types() + [ResTypes.RX] + \
            ResTypes.dna_nucleotide_types() + [ResTypes.DX]
    else:
        raise ValueError(f"Invalid model: {model}")


@lru_cache
def get_all_atom_types(model: str) -> list[AtomType]:
    """ Return all atom types for a given model. """
    atom_types = AtomTypes.all_types()
    return atom_types


def sequence_to_onehot(sequence: str,
                       restype_to_idx: dict[str, int]) -> torch.IntTensor:
    """
    Maps the given sequence into a one-hot encoded matrix.
    """
    indices = []
    for residue in sequence:
        indices.append(restype_to_idx[residue])

    indices = torch.tensor(indices, dtype=torch.long)
    return F.one_hot(indices, num_classes=len(restype_to_idx))
