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

import math
from typing import Optional

import torch
from einops import rearrange

from .interface import AttentionBackend, AttentionMetadata


class VanillaAttentionMetadata(AttentionMetadata):
    pass


class VanillaTriangleAttention(AttentionBackend[VanillaAttentionMetadata]):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: Optional[AttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of vanilla attention for triangle attention and pairwise attention.
        Args:
            q (torch.Tensor):
                Triangle attention:
                    query tensor, shape [B, I, J, H * D]
            k (torch.Tensor):
                Triangle attention:
                    key tensor, shape [B, I, J, H * D]
            v (torch.Tensor):
                Triangle attention:
                    value tensor, shape [B, I, J, H * D]
            biases (Optional[list[torch.Tensor]]): list of bias tensors
                - Triangle bias: [B, I, 1, 1, J], [B, H, J, J]
            metadata (Optional[AttentionMetadata]): attention metadata
        Forward pass for triangle attention
        Triangle bias has two terms:
            1. Mask over sequence length: [B, I, 1, 1, J]
            2. Bias for heads: [B, H, J, J]
        To avoid memory allocation, we use a vanilla implementation here.
        """
        mask = biases[0]
        bias = biases[1]
        bias = bias.unsqueeze(1)
        q = rearrange(q,
                      "b i j (h d) -> b i h j d",
                      h=self.num_heads,
                      d=self.head_dim)
        k = rearrange(k,
                      "b i j (h d) -> b i h d j",
                      h=self.num_heads,
                      d=self.head_dim)
        v = rearrange(v,
                      "b i j (h d) -> b i h j d",
                      h=self.num_heads,
                      d=self.head_dim)
        a = torch.matmul(q, k)
        a /= math.sqrt(self.head_dim)

        a += mask
        a += bias

        a = torch.nn.functional.softmax(a, dim=-1)

        a = torch.matmul(a, v)  # [B, I, H, J, D]
        attn_output = rearrange(a, "b i h j d -> b i j h d").contiguous()
        return attn_output


class VanillaPairwiseAttention(AttentionBackend[VanillaAttentionMetadata]):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: Optional[AttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of vanilla attention for triangle attention and pairwise attention.
        Args:
            q (torch.Tensor):
                Pairwise attention:
                    query tensor, shape [B, S_Q, H * D]
            k (torch.Tensor):
                Pairwise attention:
                    key tensor, shape [B, S_KV, H * D]
            v (torch.Tensor):
                Pairwise attention:
                    value tensor, shape [B, S_KV, H * D]
            biases (Optional[list[torch.Tensor]]): list of bias tensors
                - Pairwise biases: [B, 1, 1, S_KV], [B, H, S_Q, S_KV]
            metadata (Optional[AttentionMetadata]): attention metadata
        Forward pass for pairwise attention
        Pairwise bias has two terms:
            1. Mask over batch size: [B, 1, 1, s_kv]
            2. Bias with shape equal to the shape of QK^T: [B, h, s_q, s_kv]
        """
        q = rearrange(q,
                      "b j (h d) -> b h j d",
                      h=self.num_heads,
                      d=self.head_dim)
        k = rearrange(k,
                      "b j (h d) -> b h d j",
                      h=self.num_heads,
                      d=self.head_dim)
        v = rearrange(v,
                      "b j (h d) -> b h j d",
                      h=self.num_heads,
                      d=self.head_dim)

        a = torch.matmul(q, k)  # [B, H, s_q, s_kv]
        a /= math.sqrt(self.head_dim)

        if biases is not None:
            # Add mask bias
            if biases[0].ndim == 2:
                a += biases[0][:, None, None, :]
            else:
                a += biases[0]
            # Add pair bias
            a += biases[1]
        a = torch.nn.functional.softmax(a, dim=-1)

        a = torch.matmul(a, v)
        return a.transpose(1, 2).contiguous()
