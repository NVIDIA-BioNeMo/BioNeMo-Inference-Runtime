# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import torch

from .._common import VanillaAttentionMetadata, prep_qkv_for_vanilla
from ..interface import AttentionBackend, AttentionMetadata


class VanillaTriangleAttention(AttentionBackend[VanillaAttentionMetadata]):
    def __init__(self, layer_idx: int, num_heads: int, head_dim: int, num_kv_heads: int | None = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        assert self.num_heads == self.num_kv_heads, "num_heads must be equal to num_kv_heads"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of vanilla attention for triangle attention and pairwise attention.
        Args:
            q (torch.Tensor):
                Triangle attention:
                    query tensor, shape [*, I, J, H * D]
            k (torch.Tensor):
                Triangle attention:
                    key tensor, shape [*, I, J, H * D]
            v (torch.Tensor):
                Triangle attention:
                    value tensor, shape [*, I, J, H * D]
            biases (Optional[list[torch.Tensor]]): list of bias tensors
                - Triangle bias: [*, I, 1, 1, J], [*, H, J, J]
            metadata (Optional[AttentionMetadata]): attention metadata
        Forward pass for triangle attention
        Triangle bias has two terms:
            1. Mask over sequence length: [*, I, 1, 1, J]
            2. Bias for heads: [*, H, J, J]
        To avoid memory allocation, we use a vanilla implementation here.
        """
        mask = biases[0]

        q, k, v = prep_qkv_for_vanilla(q, k, v, self.num_heads, self.head_dim)
        a = torch.matmul(q, k)
        a /= math.sqrt(self.head_dim)

        a += mask
        if len(biases) > 1 and biases[1] is not None:
            bias = biases[1].unsqueeze(1)
            a += bias

        a = torch.nn.functional.softmax(a, dim=-1)

        a = torch.matmul(a, v)  # [*, I, H, J, D]
        attn_output = a.transpose(-3, -2)  # [*, I, J, H, D]
        return attn_output.contiguous()  # This ensure subsequence call on view will be successful
