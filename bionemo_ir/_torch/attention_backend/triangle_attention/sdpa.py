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


import torch
import torch.nn.functional as F

from .._common import SDPAAttentionMetadata, prep_qkv_for_sdpa
from ..interface import AttentionBackend, AttentionMetadata


class SDPATriangleAttention(AttentionBackend[SDPAAttentionMetadata]):
    """Triangle attention using PyTorch scaled-dot-product attention."""

    Metadata = SDPAAttentionMetadata

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
        """Run attention over the final sequence axis of ``[*, I, J, H*D]``.

        ``biases[0]`` is the additive row mask ``[*, I, 1, 1, J]``.
        ``biases[1]`` is the shared triangle bias ``[*, H, J, J]`` and is
        broadcast over ``I``.
        """
        q, k, v = prep_qkv_for_sdpa(q, k, v, self.num_heads, self.head_dim)

        attn_mask = None
        if biases is not None:
            attn_mask = biases[0].to(q)
            if len(biases) > 1 and biases[1] is not None:
                triangle_bias = biases[1]
                if triangle_bias.ndim == q.ndim - 1:
                    triangle_bias = triangle_bias.unsqueeze(-4)
                attn_mask = attn_mask + triangle_bias.to(q)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return output.transpose(-3, -2).contiguous()
