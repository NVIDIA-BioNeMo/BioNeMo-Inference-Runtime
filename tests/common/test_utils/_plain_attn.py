import math
from typing import Optional

import torch


def _prep_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, no_heads: int,
              head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim == 3:
        q = q.view(q.size(0), -1, no_heads, head_dim)
    if k.ndim == 3:
        k = k.view(k.size(0), -1, no_heads, head_dim)
    if v.ndim == 3:
        v = v.view(v.size(0), -1, no_heads, head_dim)
    q = q.transpose(1, 2)  # [B, H, s_q, D]
    k = k.transpose(1, 2)  # [B, H, s_kv, D]
    v = v.transpose(1, 2)  # [B, H, s_kv, D]

    k = torch.permute(k, (0, 1, 3, 2))  # [B, H, D, s_kv]
    return q, k, v


def plain_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    no_heads: int,
    head_dim: int,
    biases: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Simple MHA for triangular attention"""
    q, k, v = _prep_qkv(q, k, v, no_heads, head_dim)

    a = torch.matmul(q, k)
    a /= math.sqrt(head_dim)

    if biases is not None:
        for b in biases:
            a += b

    a = torch.nn.functional.softmax(a, dim=-1)

    a = torch.matmul(a, v)
    a = a.transpose(1,
                    2).contiguous().view(a.size(0), -1,
                                         no_heads * head_dim)  # [B, s_q, H * D]
    return a


def plain_pairwise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    no_heads: int,
    head_dim: int,
    biases: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple cross-attention for pairwise attention"""
    q, k, v = _prep_qkv(q, k, v, no_heads, head_dim)
    a = torch.matmul(q, k)  # [B, H, s_q, s_kv]
    a /= math.sqrt(head_dim)
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
    a = a.transpose(1,
                    2).contiguous().view(a.size(0), -1,
                                         no_heads * head_dim)  # [B, s_q, H * D]
    return a
