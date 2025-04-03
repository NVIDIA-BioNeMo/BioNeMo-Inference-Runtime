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
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.tri_att_start.mha",
            no_heads: int = 4,
            state_dict: Optional[dict] = None) -> 'RefTriangleAttention':
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)
        weights_path = [
            f"{layer_path}.linear_q.weight",
            f"{layer_path}.linear_k.weight",
            f"{layer_path}.linear_v.weight",
            f"{layer_path}.linear_o.weight",
        ]
        w_q = state_dict[weights_path[0]]
        c_q = c_k = c_v = w_q.shape[1]
        c_hidden = c_q // no_heads
        attn = cls(c_q, c_k, c_v, c_hidden, no_heads)
        layers = [attn.linear_q, attn.linear_k, attn.linear_v, attn.linear_o]
        if attn.gating:
            weights_path.append(f"{layer_path}.linear_g.weight")
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


class RefPairwiseSelfAttention(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/v0.4.1/src/boltz/model/layers/attention.py#L8
    Testing purposes only, without model cache for Pairformer module
    # TODO: Add a ref pairwise attention for diffusion modules (with model cache)
    """

    def __init__(self,
                 c_s: int,
                 c_z: int,
                 num_heads: int,
                 inf: float = 1e6,
                 initial_norm: bool = True) -> None:
        """
        Args:
            c_s (int):  The input sequence dimension.
            c_z (int): The input pairwise dimension.
            num_heads (int): number of attention heads
            inf (float): infinity value
        """
        super().__init__()
        assert c_s % num_heads == 0

        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.initial_norm = initial_norm

        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
        )
        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.attention",
            num_heads: int = 16,
            state_dict: Optional[dict] = None) -> 'RefPairwiseSelfAttention':
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.norm_s.weight", f"{layer_path}.norm_s.bias"),
            (f"{layer_path}.proj_q.weight", f"{layer_path}.proj_q.bias"),
            (f"{layer_path}.proj_k.weight", None),
            (f"{layer_path}.proj_v.weight", None),
            (f"{layer_path}.proj_g.weight", None),
            (f"{layer_path}.proj_z.0.weight", f"{layer_path}.proj_z.0.bias"),
            (f"{layer_path}.proj_z.1.weight", None),
            (f"{layer_path}.proj_o.weight", None),
        ]
        c_s = state_dict[weights_biases_path[0][0]].shape[0]
        c_z = state_dict[weights_biases_path[5][0]].shape[0]
        attn = cls(c_s, c_z, num_heads, initial_norm=True)
        layers = [
            attn.norm_s, attn.proj_q, attn.proj_k, attn.proj_v, attn.proj_g,
            attn.proj_z[0], attn.proj_z[1], attn.proj_o
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return attn

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                mask: torch.Tensor,
                multiplicity: int = 1) -> torch.Tensor:
        """
        Args:
            s (torch.Tensor): The input sequence (B, S, Ds).
            z (torch.Tensor): The input pairwise. (B, N, N, Dz)
            mask (torch.Tensor): The mask. (B, N)
            multiplicity (int): The multiplicity. The diffution batch size, default 1
        """
        B = s.size(0)
        if self.initial_norm:
            s = self.norm_s(s)
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(s).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(s).view(B, -1, self.num_heads, self.head_dim)
        z = self.proj_z(z)
        z = torch.moveaxis(z, 3, 1)  # [B, N, N, H] -> [B, H, N, N]
        g = self.proj_g(s).sigmoid()
        mask_bias = (1 - mask[:, None, None].float()) * -self.inf
        mhca_o = plain_pairwise_mhca(q, k, v, self.num_heads, self.head_dim,
                                     [mask_bias, z])

        o = mhca_o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
