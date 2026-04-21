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
import torch.nn.functional as F
from einops import rearrange

from .interface import AttentionBackend, AttentionMetadata


class SDPAAttentionMetadata(AttentionMetadata):
    pass


def _prep_qkv(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, no_heads: int,
        head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape q/k/v from [*, seq, H*D] to [*, H, seq, D].

    Unlike the vanilla variant, k is NOT transposed here because
    ``scaled_dot_product_attention`` handles the transpose internally.
    """
    batch_dims = " ".join([f"b_{i}" for i in range(q.ndim - 2)])
    q = rearrange(q,
                  f"{batch_dims} j (h d) -> {batch_dims} h j d",
                  h=no_heads,
                  d=head_dim)
    k = rearrange(k,
                  f"{batch_dims} j (h d) -> {batch_dims} h j d",
                  h=no_heads,
                  d=head_dim)
    v = rearrange(v,
                  f"{batch_dims} j (h d) -> {batch_dims} h j d",
                  h=no_heads,
                  d=head_dim)
    return q, k, v


class SDPAPairwiseAttention(AttentionBackend[SDPAAttentionMetadata]):
    """Pairwise attention using ``torch.nn.functional.scaled_dot_product_attention``.

    Drop-in replacement for ``VanillaPairwiseAttention`` that delegates to
    PyTorch's SDPA dispatcher, which selects the best available kernel
    (memory-efficient, cuDNN, or math fallback).  Provides 1.8-2.7x speedup
    and up to 30% peak-memory reduction over the manual matmul path.
    """

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
        """Pairwise attention via ``F.scaled_dot_product_attention``.

        Args:
            q: query tensor, shape ``[*, S_Q, H * D]``.
            k: key tensor, shape ``[*, S_KV, H * D]``.
            v: value tensor, shape ``[*, S_KV, H * D]``.
            biases: list of bias tensors.
                - ``biases[0]``: mask bias ``[*, 1, 1, S_KV]`` (additive, large
                  negative for masked-out positions).
                - ``biases[1]``: pair bias ``[*, H, S_Q, S_KV]``.
            metadata: attention metadata (unused).

        Returns:
            Output tensor of shape ``[*, S_Q, H, D]``.
        """
        q, k, v = _prep_qkv(q, k, v, self.num_heads, self.head_dim)

        attn_mask = None
        if biases is not None:
            mask_bias = biases[0]
            if mask_bias.ndim == 2:
                mask_bias = mask_bias[:, None, None, :]
            attn_mask = mask_bias.to(q) + biases[1].to(q)

        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        a = a.transpose(-3, -2)  # [*, S_Q, H, D]
        return a.contiguous()
