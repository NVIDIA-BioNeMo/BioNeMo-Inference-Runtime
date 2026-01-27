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

import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.data.parsers import InputParsed
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase


class TemplateContextGenerator(ContextGeneratorBase):

    def empty_template_feats(self, n_res: int) -> dict[str, torch.Tensor]:
        return {
            "template_aatype":
            torch.zeros((0, n_res, len(rc.restypes_with_x_and_gap)),
                        dtype=torch.float32),
            "template_all_atom_mask":
            torch.zeros((0, n_res, rc.atom_type_num), dtype=torch.float32),
            "template_all_atom_positions":
            torch.zeros((0, n_res, rc.atom_type_num, 3), dtype=torch.float32),
            "template_sum_probs":
            torch.zeros((0, 1), dtype=torch.float32),
        }

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        polymer = parsed['polymers'][0]
        parsed_templates = polymer.get('templates')
        sequence = polymer['sequence']

        if parsed_templates is None or len(parsed_templates) == 0:
            n_res = len(sequence)
            tensors = self.empty_template_feats(n_res)
            return tensors

        raise NotImplementedError(
            "Template context generation for custom templates is not implemented yet."
        )
