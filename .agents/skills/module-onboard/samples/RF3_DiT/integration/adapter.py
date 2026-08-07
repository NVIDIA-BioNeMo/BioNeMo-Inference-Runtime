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
Drop-in adapter: wraps TRT-BNM DiffusionTransformerLayer to match
BakerLab RF3 DiffusionTransformerBlock forward signature.

Customer signature:  forward(A_I, S_I, Z_II, is_padding_I) -> A_I
TRT-BNM signature:   forward(a, s, bias, mask, ...) -> a
"""

import torch.nn as nn


class BakerLabDiTBlockAdapter(nn.Module):
    """Drop-in replacement for a single BakerLab DiffusionTransformerBlock.

    Customer forward: (A_I [B,D,I,C], S_I [B,D,I,C], Z_II [B,I,I,Cz], is_padding_I [B,I]) -> A_I [B,D,I,C]
    TRT-BNM forward:  (a [B,I,C], s [B,I,C], bias [B,I,I,Cz], mask [B,I]) -> a [B,I,C]

    The D (diffusion samples) dimension is flattened into B for TRT-BNM, then reshaped back.
    Z_II and is_padding_I have no D dim — they are broadcast across D.
    """

    def __init__(self, trtbnm_layer):
        super().__init__()
        self.layer = trtbnm_layer

    def forward(self, A_I, S_I, Z_II, is_padding_I):
        B, D = A_I.shape[0], A_I.shape[1]
        mask = (~is_padding_I).float()

        # Flatten D into B: [B, D, I, C] -> [B*D, I, C]
        a = A_I.reshape(B * D, *A_I.shape[2:])
        s = S_I.reshape(B * D, *S_I.shape[2:])
        # Expand Z_II and mask across D: [B, ...] -> [B*D, ...]
        z = Z_II[:, None].expand(-1, D, -1, -1, -1).reshape(B * D, *Z_II.shape[1:])
        m = mask[:, None].expand(-1, D, -1).reshape(B * D, mask.shape[-1])

        a = self.layer(a, s, z, mask=m)

        # Reshape back: [B*D, I, C] -> [B, D, I, C]
        return a.reshape(B, D, *a.shape[1:])


class BakerLabDiTStackAdapter(nn.Module):
    """Drop-in replacement for BakerLab's DiffusionTransformer (stack of blocks)."""

    def __init__(self, trtbnm_layers: nn.ModuleList):
        super().__init__()
        self.layers = trtbnm_layers

    def forward(self, A_I, S_I, Z_II, is_padding_I):
        B, D = A_I.shape[0], A_I.shape[1]
        mask = (~is_padding_I).float()

        a = A_I.reshape(B * D, *A_I.shape[2:])
        s = S_I.reshape(B * D, *S_I.shape[2:])
        z = Z_II[:, None].expand(-1, D, -1, -1, -1).reshape(B * D, *Z_II.shape[1:])
        m = mask[:, None].expand(-1, D, -1).reshape(B * D, mask.shape[-1])

        for layer in self.layers:
            a = layer(a, s, z, mask=m)

        return a.reshape(B, D, *a.shape[1:])
