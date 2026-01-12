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

import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.data.parsers import InputParsed
from tensorrt_bionemo.data.utils import sequence_to_onehot
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase


class PrimaryContextGenerator(ContextGeneratorBase):

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        """
        Generate a structure context from an input parsed object.
        """
        parsed_primary = parsed['primary']
        chains = parsed_primary['chains']
        sequence = chains[0]['sequence']['residues']
        aatype = sequence_to_onehot(sequence, rc.restype_order_with_x)
        n_res = len(sequence)

        between_segment_residues = torch.zeros((n_res, ), dtype=torch.int32)
        residue_index = torch.arange(n_res, dtype=torch.int32)
        seq_length = torch.tensor([n_res] * n_res, dtype=torch.int32)

        return {
            'aatype': aatype,
            'between_segment_residues': between_segment_residues,
            'residue_index': residue_index,
            'seq_length': seq_length,
        }
