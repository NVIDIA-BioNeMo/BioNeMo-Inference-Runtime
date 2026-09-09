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
"""Protenix prediction heads."""

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.linear import Linear


class ProtenixDistogramHead(nn.Module):
    """Distogram head (AF3 Algorithm 1, line 17).

    Projects pair ``z`` to bins and symmetrizes *logits*
    (``W z + W zᵀ + 2 b``), matching OSS Protenix (not the shared
    input-symmetrizing ``DistogramModule``). Runs in fp32.
    """

    def __init__(
        self, c_z: int, no_bins: int = 64, dtype: torch.dtype = torch.float32, skip_create_weights: bool = False
    ) -> None:
        super().__init__()
        self.c_z = c_z
        self.no_bins = no_bins
        self.dtype = dtype
        self.linear = Linear(c_z, no_bins, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Symmetric distogram logits ``[*, N_token, N_token, no_bins]``.

        Args:
            z: ``[*, N_token, N_token, c_z]`` pair representation
        """
        z = z.to(self.dtype)
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
