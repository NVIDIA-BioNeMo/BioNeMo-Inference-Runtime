# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.data.parsers import A3MParsed, InputParsed
from tensorrt_bionemo.data.parsers.a3m import generate_deletion_matrix
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase


class MSAContextGenerator(ContextGeneratorBase):

    def make_msa_features(
            self, parsed_msas: list[A3MParsed]) -> dict[str, torch.Tensor]:
        raw_sequences = []
        aligned_sequences = []
        for parsed_msa in parsed_msas:
            raw_sequences.extend(parsed_msa["raw"])
            aligned_sequences.extend(parsed_msa["sequences"])
        int_msa = []
        deletion_matrix = generate_deletion_matrix(raw_sequences)
        filted_deletion_matrix = []
        seen_sequences = set()

        for sequence_index, sequence in enumerate(aligned_sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])
            filted_deletion_matrix.append(deletion_matrix[sequence_index])

        num_res = len(parsed_msas[0]["sequences"][0])
        num_alignments = len(int_msa)
        features = {}
        filted_deletion_matrix = np.array(filted_deletion_matrix)
        features["deletion_matrix"] = torch.tensor(filted_deletion_matrix,
                                                   dtype=torch.float32)
        features["msa"] = torch.tensor(int_msa, dtype=torch.int32)
        features["num_alignments"] = torch.tensor([num_alignments] * num_res,
                                                  dtype=torch.int32)
        return features

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        parsed_msas = parsed['msa']
        parsed_primary = parsed['primary']

        if parsed_msas is None or len(parsed_msas) == 0:
            # Generate dummy MSA for the primary chain
            parsed_msas = {}
            dummy_parsed_msas = A3MParsed(
                sequences=[
                    parsed_primary['chains'][0]['sequence']['residues']
                ],
                descriptions=[
                    parsed_primary['chains'][0]['sequence']['description']
                ])
            parsed_msas[parsed_primary['chains'][0]['chain_id']] = [
                dummy_parsed_msas
            ]
        # TODO: Implement MSA context generation for multiple chains
        key_0 = list(parsed_msas.keys())[0]
        return self.make_msa_features(parsed_msas[key_0])
