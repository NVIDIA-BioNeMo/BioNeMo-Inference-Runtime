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

"""
Drop-in adapter: wraps TRT-BNM PairformerModule to match BakerLab PairformerBlock forward signature.

Customer signature:  forward(S_I, Z_II, is_padding_I) -> (S_I, Z_II)
TRT-BNM signature:   forward(s, z, mask, pair_mask, ...) -> (s, z)
"""

import torch.nn as nn

from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule


class BakerLabPairformerAdapter(nn.Module):
    """Drop-in replacement for BakerLab's pairformer_stack (list of PairformerBlocks)."""

    def __init__(self, trtbnm_module: PairformerModule):
        super().__init__()
        self.module = trtbnm_module

    def forward(self, S_I, Z_II, is_padding_I):
        # Convert mask: customer is_padding_I (bool, True=pad) -> TRT-BNM mask (float, 1.0=valid)
        mask = (~is_padding_I).float()
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        s, z = self.module(S_I, Z_II, mask, pair_mask)
        return s, z
