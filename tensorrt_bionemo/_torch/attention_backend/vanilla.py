from typing import Optional

import torch

from .interface import (
    AttentionBackend,
    AttentionBiases,
    AttentionMetadata,
    PredefinedAttentionBiases,
)


class VanillaAttention(AttentionBackend[AttentionMetadata]):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"
        self.num_key_value_groups = 1

    def _single_request_forward(self, q, k, v, bias):
        """Forward pass for a single request"""
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=bias,
        )
        return attn_output

    def _slice_by_chunk(self, tensor: torch.Tensor, chunk_dim: int, offset: int,
                        chunk: int) -> torch.Tensor:
        """Slice the tensor by the given chunk size"""
        slice_obj = [slice(None)] * tensor.ndim
        slice_obj[chunk_dim] = slice(offset, offset + chunk)
        return tensor[tuple(slice_obj)]

    def _create_bias_term_with_chunk(
        self,
        chunk: int = None,
        chunk_dim: int = None,
        offset: int = None,
        biases: Optional[list[torch.Tensor]] = None,
        attention_biases: Optional[AttentionBiases] = PredefinedAttentionBiases.
        TRIANGLE,
    ) -> torch.Tensor:
        """Create a bias term with the given chunk size"""
        ret = None
        if biases is None:
            return ret

        if chunk is None:
            # Triangle bias has two terms:
            # 1. Bias over sequence length: [s, 1, 1, s]
            # 2. Bias for heads: [1, h, s, s]
            if attention_biases == PredefinedAttentionBiases.TRIANGLE:
                seq_len = biases[0].size(0)
                ret = biases[0] + biases[1].expand(seq_len, -1, -1, -1)
            # Pairwise bias has two terms:
            # 1. Bias over batch size: [B, 1, 1, s_kv]
            # 2. Bias with shape equal to the shape of QK^T: [B, h, s_q, s_kv]
            elif attention_biases == PredefinedAttentionBiases.PAIRWISE:
                ret = biases[0] + biases[1]
        else:
            if attention_biases == PredefinedAttentionBiases.TRIANGLE:
                ret = self._slice_by_chunk(biases[0], chunk_dim, offset,
                                           chunk) + biases[1].expand(
                                               chunk, -1, -1, -1)
            elif attention_biases == PredefinedAttentionBiases.PAIRWISE:
                ret = self._slice_by_chunk(
                    biases[0], chunk_dim, offset, chunk) + self._slice_by_chunk(
                        biases[1], chunk_dim, offset, chunk)
        return ret

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: Optional[AttentionMetadata] = None,
        attention_biases: Optional[AttentionBiases] = PredefinedAttentionBiases.
        TRIANGLE,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of vanilla attention for triangle attention and pairwise attention."""
        if metadata.chunk_size is not None:
            offset = 0
            attn_outputs = []
            chunk_dim = metadata.chunk_dim
            total_size = q.size(chunk_dim)
            while offset < total_size:
                chunk = min(metadata.chunk_size, total_size - offset)
                chunk_q = self._slice_by_chunk(q, chunk_dim, offset, chunk)
                chunk_k = self._slice_by_chunk(k, chunk_dim, offset, chunk)
                chunk_v = self._slice_by_chunk(v, chunk_dim, offset, chunk)
                bias = self._create_bias_term_with_chunk(
                    chunk, offset, biases, attention_biases)
                offset += chunk
                attn_output = self._single_request_forward(
                    chunk_q, chunk_k, chunk_v, bias)
                attn_outputs.append(attn_output)
            attn_output = torch.cat(attn_outputs, dim=0)
        else:
            bias = self._create_bias_term_with_chunk(None, 0, biases,
                                                     attention_biases)
            attn_output = self._single_request_forward(q, k, v, bias)
        return (attn_output.transpose(1, 2).contiguous().view(
            q.size(0), -1, self.num_heads * self.head_dim))
