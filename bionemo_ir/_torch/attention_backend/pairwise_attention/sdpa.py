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


def _empty_with_aligned_rows(like: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """An uninitialized ``shape`` view whose rows start 16-byte aligned.

    cuDNN takes a much slower kernel when the mask's row stride is not 16-byte
    aligned, as at any key count not divisible by 8 in bf16, so the buffer pads
    each row and the view keeps the keys.
    """
    keys = shape[-1]
    align = 16 // like.element_size()
    return like.new_empty(*shape[:-1], -(-keys // align) * align)[..., :keys]


def _sum_with_aligned_rows(terms: list[torch.Tensor]) -> torch.Tensor:
    """Sum broadcastable mask terms into a view whose rows start 16-byte aligned."""
    out = _empty_with_aligned_rows(terms[0], torch.broadcast_shapes(*(term.shape for term in terms)))
    out.copy_(terms[0])
    for term in terms[1:]:
        out.add_(term)
    return out


class SDPAPairwiseAttention(AttentionBackend[SDPAAttentionMetadata]):
    """Pairwise attention using ``torch.nn.functional.scaled_dot_product_attention``.

    Drop-in replacement for ``VanillaPairwiseAttention`` that delegates to
    PyTorch's SDPA dispatcher, which selects the best available kernel
    (memory-efficient, cuDNN, or math fallback).  Provides 1.8-2.7x speedup
    and up to 30% peak-memory reduction over the manual matmul path.
    """

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
        q, k, v = prep_qkv_for_sdpa(q, k, v, self.num_heads, self.head_dim)

        attn_mask = None
        if biases is not None:
            mask_bias = biases[0]
            if mask_bias.ndim == 2:
                mask_bias = mask_bias[:, None, None, :]
            attn_mask = mask_bias.to(q)
            if len(biases) > 1:
                attn_mask = _sum_with_aligned_rows([attn_mask, *(bias.to(q) for bias in biases[1:])])

        # SDPA's fused kernels take rank-4 operands; a higher rank silently
        # selects the math backend, which materializes FP32 logits. Fold the
        # leading axes, such as the atom path's [B, mult, K], into one.
        batch_dims = q.shape[:-3]
        flattened = len(batch_dims) > 1
        if flattened:
            q_ndim = q.ndim
            q, k, v = (tensor.reshape(-1, *tensor.shape[-3:]) for tensor in (q, k, v))
            if attn_mask is not None:
                # Expand the bias over every folded axis first. Pad it to q's
                # rank ahead of its head axis, since `expand` aligns trailing
                # axes and would pair its batch axis with a window axis.
                while attn_mask.ndim < q_ndim:
                    attn_mask = attn_mask.unsqueeze(-4)
                expanded = attn_mask.expand(*batch_dims, *attn_mask.shape[-3:])
                folded_shape = (-1, *expanded.shape[-3:])
                try:
                    attn_mask = expanded.view(folded_shape)
                except RuntimeError:
                    # The fold has to copy, so copy into aligned rows too.
                    attn_mask = _empty_with_aligned_rows(expanded, expanded.shape)
                    attn_mask.copy_(expanded)
                    attn_mask = attn_mask.view(folded_shape)

        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        a = a.transpose(-3, -2)  # [*, S_Q, H, D]
        if flattened:
            a = a.reshape(*batch_dims, *a.shape[-3:])
        return a.contiguous()
