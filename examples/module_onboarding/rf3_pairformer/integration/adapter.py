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
#
# Layout reference: RoseTTAFold3 (RosettaCommons/foundry), BSD-3-Clause.
# https://github.com/RosettaCommons/foundry/tree/production/models/rf3
# Only upstream parameter and module names are reproduced here.

"""
Adapter: runs a BioIR PairformerModule with the call signature of an RF3 PairformerBlock.

Source signature:   forward(S_I, Z_II) -> (S_I, Z_II), S_I [I, c_s], Z_II [I, I, c_z]
BioIR signature:   forward(s, z, mask, pair_mask, ...) -> (s, z), with a batch axis
"""

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule


class RF3PairformerAdapter(nn.Module):
    """Stands in for RF3's whole pairformer_stack, called like one PairformerBlock."""

    def __init__(self, bioir_module: PairformerModule):
        super().__init__()
        self.module = bioir_module
        self.dtype = bioir_module.config.torch_dtype

    def forward(self, S_I, Z_II):
        unbatched = Z_II.ndim == 3
        if unbatched:  # RF3 runs one structure without a batch axis
            S_I, Z_II = S_I[None], Z_II[None]

        # RF3 passes no padding mask: every token is valid.
        mask = torch.ones(S_I.shape[:-1], dtype=torch.float32, device=S_I.device)
        pair_mask = mask[..., :, None] * mask[..., None, :]

        # BioIR fixes its dtypes at construction, so keep the caller's autocast out.
        with torch.autocast(device_type=Z_II.device.type, enabled=False):
            s, z = self.module(S_I.to(self.dtype), Z_II.to(self.dtype), mask, pair_mask)
        s, z = s.to(S_I.dtype), z.to(Z_II.dtype)

        if unbatched:
            s, z = s[0], z[0]
        return s, z
