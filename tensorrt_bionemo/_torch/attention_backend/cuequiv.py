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
from cuequivariance_torch.primitives.triangle import triangle_attention
from einops import rearrange

from .interface import AttentionBackend, AttentionMetadata


class CuEquivAttentionMetadata(AttentionMetadata):
    flip_mask: bool = False


class CuEquivAttention(AttentionBackend[CuEquivAttentionMetadata]):

    Metadata = CuEquivAttentionMetadata

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: Optional[AttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of trifast attention."""
        mask = biases[0]
        bias = biases[1]
        if metadata is None:
            metadata = CuEquivAttentionMetadata()
        assert q.ndim == k.ndim == v.ndim, "q, k, v must have the same number of dimensions"
        q.ndim
        if q.ndim == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        if mask.ndim == 4:
            mask = mask.unsqueeze(0)
        if bias.ndim == 3:
            bias = bias.unsqueeze(0)

        bs, i, j, hd = q.shape
        q = rearrange(q, "b i j (h d) -> b i h j d",
                      h=self.num_heads).contiguous()
        k = rearrange(k, "b i j (h d) -> b i h j d",
                      h=self.num_heads).contiguous()
        v = rearrange(v, "b i j (h d) -> b i h j d",
                      h=self.num_heads).contiguous()
        bias = rearrange(bias, "b h i j -> b () h i j").contiguous()
        mask = mask.bool().contiguous()
        if metadata.flip_mask:
            mask = ~mask

        sm_scale = self.head_dim**-0.5
        o = triangle_attention(q, k, v, bias, mask=mask, scale=sm_scale)
        o = rearrange(o, " b i h j d -> b i j h d").contiguous()
        return o
