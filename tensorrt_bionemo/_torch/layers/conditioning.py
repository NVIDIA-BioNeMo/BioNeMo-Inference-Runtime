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

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import FourierEmbedding
from tensorrt_bionemo.mapping import Mapping


class ContactConditioning(nn.Module):
    """ Boltz2 Contact Conditioning """

    def __init__(self,
                 token_z: int,
                 cutoff_min: float,
                 cutoff_max: float,
                 contact_conditioning_info: dict[str, int],
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z,
                                                  dtype=dtype,
                                                  mapping=mapping)
        self.encoder = Linear(token_z + len(contact_conditioning_info) - 1,
                              token_z,
                              dtype=dtype,
                              mapping=mapping,
                              tensor_parallel_mode=TensorParallelMode.COLUMN,
                              gather_output=True,
                              skip_create_weights=False)
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        self.contact_conditioning_info = contact_conditioning_info

    def forward(self, contact_conditioning: torch.Tensor,
                contact_threshold: torch.Tensor):
        assert self.contact_conditioning_info["UNSPECIFIED"] == 0
        assert self.contact_conditioning_info["UNSELECTED"] == 1
        contact_conditioning = contact_conditioning[:, :, :, 2:]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min)
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.flatten()).reshape(
                contact_threshold_normalized.shape + (-1, ))

        contact_conditioning = torch.cat(
            [
                contact_conditioning,
                contact_threshold_normalized.unsqueeze(-1),
                contact_threshold_fourier,
            ],
            dim=-1,
        )
        contact_conditioning = self.encoder(contact_conditioning)

        contact_conditioning = (
            contact_conditioning *
            (1 - contact_conditioning[:, :, :, 0:2].sum(dim=-1, keepdim=True)) +
            self.encoding_unspecified * contact_conditioning[:, :, :, 0:1] +
            self.encoding_unselected * contact_conditioning[:, :, :, 1:2])
        return contact_conditioning
