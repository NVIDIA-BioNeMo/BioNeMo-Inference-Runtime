import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.hf.checkpoints import load_hf_weights


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


def plain_triangle_mha(
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
    a = a.transpose(1, 2).contiguous()
    return a


def plain_pairwise_mhca(
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
    a = a.transpose(1, 2).contiguous()
    return a


class RefTriangleAttention(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/v0.4.1/src/boltz/model/layers/triangular_attention/primitives.py#L310
    Testing purposes only
    """

    def __init__(self,
                 c_q: int,
                 c_k: int,
                 c_v: int,
                 c_hidden: int,
                 no_heads: int,
                 gating: bool = True):
        """
        Args:
            c_q (int): query dimension
            c_k (int): key dimension
            c_v (int): value dimension
            c_hidden (int): hidden dimension
            no_heads (int): number of attention heads
            gating (bool): if True, use gating
        """
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = nn.Linear(c_q,
                                  self.c_hidden * self.no_heads,
                                  bias=False)
        self.linear_k = nn.Linear(c_k,
                                  self.c_hidden * self.no_heads,
                                  bias=False)
        self.linear_v = nn.Linear(c_v,
                                  self.c_hidden * self.no_heads,
                                  bias=False)
        self.linear_o = nn.Linear(self.c_hidden * self.no_heads,
                                  c_q,
                                  bias=False)

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(c_q,
                                      self.c_hidden * self.no_heads,
                                      bias=False)
        self.sigmoid = nn.Sigmoid()

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-1",
                     triattn_layer_path:
                     str = "pairformer_module.layers.0.tri_att_start.mha",
                     no_heads: int = 4) -> 'RefTriangleAttention':
        state_dict = load_hf_weights(model)
        weights_path = [
            f"{triattn_layer_path}.linear_q.weight",
            f"{triattn_layer_path}.linear_k.weight",
            f"{triattn_layer_path}.linear_v.weight",
            f"{triattn_layer_path}.linear_o.weight",
        ]
        w_q = state_dict[weights_path[0]]
        c_q = c_k = c_v = w_q.shape[1]
        c_hidden = c_q // no_heads
        attn = cls(c_q, c_k, c_v, c_hidden, no_heads)
        layers = [attn.linear_q, attn.linear_k, attn.linear_v, attn.linear_o]
        if attn.gating:
            weights_path.append(f"{triattn_layer_path}.linear_g.weight")
            layers.append(attn.linear_g)
        for weights_path, layer in zip(weights_path, layers):
            layer.weight.data.copy_(state_dict[weights_path])
        return attn

    def forward(self,
                q_x: torch.Tensor,
                kv_x: torch.Tensor,
                biases: Optional[list[torch.Tensor]] = None):
        proj_q = self.linear_q(q_x)
        proj_k = self.linear_k(kv_x)
        proj_v = self.linear_v(kv_x)

        q = proj_q.view(proj_q.size(0), -1, self.no_heads, self.c_hidden)
        k = proj_k.view(proj_k.size(0), -1, self.no_heads, self.c_hidden)
        v = proj_v.view(proj_v.size(0), -1, self.no_heads, self.c_hidden)
        mha_o = plain_triangle_mha(q, k, v, self.no_heads, self.c_hidden,
                                   biases)
        if self.linear_g is not None:
            g = F.sigmoid(self.linear_g(q_x))
            g = g.view(g.size(0), -1, self.no_heads, self.c_hidden)
            o = mha_o * g
        o = o.view(o.size(0), -1, self.no_heads * self.c_hidden)
        o = self.linear_o(o)

        return o


class RefPairwiseAttention(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/v0.4.1/src/boltz/model/layers/attention.py#L8
    Testing purposes only
    """
