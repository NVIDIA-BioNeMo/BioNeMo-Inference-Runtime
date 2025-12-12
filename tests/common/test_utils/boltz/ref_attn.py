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
from einops import rearrange

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo.hubs import load_weights


def _prep_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, no_heads: int,
              head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_dims = " ".join([f"b_{i}" for i in range(q.ndim - 2)])
    q = rearrange(q,
                  f"{batch_dims} j (h d) -> {batch_dims} h j d",
                  h=no_heads,
                  d=head_dim)
    k = rearrange(k,
                  f"{batch_dims} j (h d) -> {batch_dims} h d j",
                  h=no_heads,
                  d=head_dim)
    v = rearrange(v,
                  f"{batch_dims} j (h d) -> {batch_dims} h j d",
                  h=no_heads,
                  d=head_dim)
    return q, k, v


def plain_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    no_heads: int,
    head_dim: int,
    biases: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Simple MHA for triangular attention, pairwise attention, and global attention.
    Args:
        q (torch.Tensor): query tensor, shape [*, J, H * D]
        k (torch.Tensor): key tensor, shape [*, J, H * D]
        v (torch.Tensor): value tensor, shape [*, J, H * D]
        no_heads (int): number of attention heads
        head_dim (int): dimension of each head
        biases (Optional[list[torch.Tensor]]): list of biases with ability broadcast over the score matrix
        i.e.
           - Mask: [*, 1, 1, J]
           - Triangle bias: [B, 1, H, J, J]
    """
    q, k, v = _prep_qkv(q, k, v, no_heads, head_dim)

    a = torch.matmul(q, k)
    a /= math.sqrt(head_dim)  # [B, I, H, J, J]
    if biases is not None:
        for bias in biases:
            a += bias
    a = torch.nn.functional.softmax(a, dim=-1)

    a = torch.matmul(a, v)  # [B, I, H, J, D]
    batch_dims = " ".join([f"b_{i}" for i in range(a.ndim - 3)])
    a = rearrange(a, f"{batch_dims} h j d -> {batch_dims} j h d").contiguous()
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
                 gating: bool = True,
                 bias_flags: dict[str, bool] = {
                     "q": False,
                     "k": False,
                     "v": False,
                     "g": False,
                     "o": False
                 }):
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
        self.bias_flags = bias_flags

        self.linear_q = nn.Linear(c_q,
                                  self.c_hidden * self.no_heads,
                                  bias=bias_flags["q"])
        self.linear_k = nn.Linear(c_k,
                                  self.c_hidden * self.no_heads,
                                  bias=bias_flags["k"])
        self.linear_v = nn.Linear(c_v,
                                  self.c_hidden * self.no_heads,
                                  bias=bias_flags["v"])
        self.linear_o = nn.Linear(self.c_hidden * self.no_heads,
                                  c_q,
                                  bias=bias_flags["o"])

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(c_q,
                                      self.c_hidden * self.no_heads,
                                      bias=bias_flags["g"])
        self.sigmoid = nn.Sigmoid()

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.tri_att_start.mha",
            no_heads: int = 4,
            state_dict: Optional[dict] = None) -> 'RefTriangleAttention':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
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
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        mha_o = plain_mha(q.contiguous(), k.contiguous(), v.contiguous(),
                          self.no_heads, self.c_hidden, biases)
        o = mha_o
        if self.linear_g is not None:
            g = F.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, self.c_hidden))
            o = o * g
        o = o.view(o.shape[:-2] + (self.no_heads * self.c_hidden, ))
        o = self.linear_o(o)

        return o


class RefPairwiseSelfAttention(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/v0.4.1/src/boltz/model/layers/attention.py#L8
    Testing purposes only, without model cache for Pairformer module
    # TODO: Add a ref pairwise attention for diffusion modules (with model cache)
    """

    def __init__(
            self,
            c_s: int,
            c_z: int,
            num_heads: int,
            inf: float = 1e9,
            bias_flags: dict[str, bool] = {
                "q": True,
                "k": False,
                "v": False,
                "g": False,
                "z": False,
                "o": False
            },  # default for boltz
            compute_pair_bias: bool = True,
            transform_mask: bool = True,
            initial_norm: bool = True) -> None:
        """
        Args:
            c_s (int):  The input sequence dimension.
            c_z (int): The input pairwise dimension.
            num_heads (int): number of attention heads
            inf (float): infinity value
            q_bias (bool): if True, use bias in the q projection
            compute_pair_bias (bool): if True, compute the pair bias
            initial_norm (bool): if True, use layer norm
        """
        super().__init__()
        assert c_s % num_heads == 0

        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.initial_norm = initial_norm
        self.compute_pair_bias = compute_pair_bias
        self.transform_mask = transform_mask
        self.bias_flags = bias_flags

        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s, bias=bias_flags["q"])
        self.proj_k = nn.Linear(c_s, c_s, bias=bias_flags["k"])
        self.proj_v = nn.Linear(c_s, c_s, bias=bias_flags["v"])
        self.proj_g = nn.Linear(c_s, c_s, bias=bias_flags["g"])

        if self.compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=bias_flags["z"]),
            )
        self.proj_o = nn.Linear(c_s, c_s, bias=bias_flags["o"])

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.attention",
            state_dict: Optional[dict] = None) -> 'RefPairwiseSelfAttention':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
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
        compute_pair_bias = True
        c_s = state_dict[f"{layer_path}.proj_q.weight"].shape[0]
        if f"{layer_path}.proj_z.1.weight" in state_dict:
            c_z = state_dict[f"{layer_path}.proj_z.1.weight"].shape[1]
            num_heads = state_dict[f"{layer_path}.proj_z.1.weight"].shape[0]
        else:
            c_z = 0
            num_heads = 4
            compute_pair_bias = False

        if f"{layer_path}.norm_s.weight" in state_dict:
            attn = cls(c_s, c_z, num_heads, initial_norm=True)
            layers = [
                attn.norm_s, attn.proj_q, attn.proj_k, attn.proj_v, attn.proj_g,
                attn.proj_z[0], attn.proj_z[1], attn.proj_o
            ]
        else:
            attn = cls(c_s, c_z, num_heads, initial_norm=False)
            layers = [
                attn.proj_q, attn.proj_k, attn.proj_v, attn.proj_g,
                attn.proj_z[0], attn.proj_z[1], attn.proj_o
            ]
            weights_biases_path = weights_biases_path[1:]

        attn.compute_pair_bias = compute_pair_bias

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None and bias_path in state_dict:
                layer.bias.data.copy_(state_dict[bias_path])
            if weights_path in state_dict:
                layer.weight.data.copy_(state_dict[weights_path])

        return attn

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor,
            compute_pair_bias: bool = True,
            multiplicity: int = 1,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        """
        Args:
            s (torch.Tensor): The input sequence (B, I, Ds) or (B, J, I, Ds).
            z (torch.Tensor): The input pairwise. (B, I, I, Dz)
            mask (torch.Tensor): The mask. (B, J, I)
        """
        s.size(0)

        if self.initial_norm:
            s = self.norm_s(s)

        if not s.is_contiguous():
            s = s.contiguous()
        kv_in = s

        if attn_metadata is not None:
            # Get key-value from the query for sequence local atom attention
            query_to_keys = attn_metadata.query_to_keys
            # query_to_keys = lambda x: query_to_keys(x.view(bs, K * W, -1)).view(bs * K, H, -1)
            if query_to_keys is not None:
                kv_in = query_to_keys(s)
                mask = query_to_keys(mask.unsqueeze(-1)).squeeze(-1)

        q = self.proj_q(s)
        k = self.proj_k(kv_in)
        v = self.proj_v(kv_in)
        if compute_pair_bias and self.compute_pair_bias:
            z = self.proj_z(z)
            z = z.repeat_interleave(multiplicity, 0)
            if mask.ndim == 2:
                z = torch.moveaxis(z, 3, 1)  # [B, I, I, H] -> [B, H, N, N]
            if mask.ndim == 3:
                z = torch.moveaxis(z, 3, 1)  # [B, I, I, H] -> [B, H, N, N]
                z = z.unsqueeze(1)  # [B, I, I, H] -> [B, 1, H, I, I]
        g = self.proj_g(s).sigmoid()

        if self.transform_mask:
            mask_bias = (1 - mask[..., None, None, :].float()) * -self.inf
        else:
            mask_bias = mask.float()

        mha_o = plain_mha(q, k, v, self.num_heads, self.head_dim,
                          [mask_bias, z])
        batch_dims = mha_o.shape[:-2]
        o = mha_o.reshape(*batch_dims, self.num_heads * self.head_dim)

        g = self.proj_g(s).sigmoid()
        o = self.proj_o(g * o)
        return o
