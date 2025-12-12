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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from test_utils.boltz.ref_layers import \
    RefOuterProductMean as BoltzRefOuterProductMean
from test_utils.boltz.ref_layers import \
    RefTriangleAttentionNode as BoltzRefTriangleAttentionNode
from test_utils.boltz.ref_layers import \
    RefTriangleMultiplicationNode as BoltzRefTriangleMultiplicationNode
from test_utils.openfold.ref_attn import (RefGlobalAttention,
                                          RefPairwiseSelfAttention,
                                          RefTriangleAttention)

from tensorrt_bionemo.hubs import load_weights


class RefMSAAttention(nn.Module):

    def __init__(self,
                 c_in,
                 c_hidden,
                 no_heads,
                 pair_bias=False,
                 c_z=None,
                 inf=1e9,
                 using_tri_attn=True,
                 transpose_input=False,
                 eps=1e-5):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf
        self.eps = eps
        self.using_tri_attn = using_tri_attn
        self.transpose_input = transpose_input

        self.layer_norm_m = nn.LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = nn.LayerNorm(self.c_z, eps=eps)
            self.linear_z = nn.Linear(self.c_z, self.no_heads, bias=False)

        if not using_tri_attn:
            self.mha = RefPairwiseSelfAttention(
                c_s=self.c_in,
                c_z=None,
                num_heads=self.no_heads,
                inf=inf,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                },  # default for boltz
                compute_pair_bias=False,
                transform_mask=False,
                initial_norm=False)
        else:
            self.mha = RefTriangleAttention(c_q=c_in,
                                            c_k=c_in,
                                            c_v=c_in,
                                            c_hidden=c_hidden,
                                            no_heads=self.no_heads,
                                            bias_flags={
                                                "q": False,
                                                "k": False,
                                                "v": False,
                                                "g": True,
                                                "o": True
                                            })

    def forward(self,
                m: torch.Tensor,
                z: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            m (torch.Tensor): The input sequence (*, N_seq, N_res, C_m).
            z (torch.Tensor): The input pairwise. (*, N_res, N_res, C_z).
            mask (torch.Tensor): The mask. (*, N_seq, N_res).
        """
        if self.transpose_input:
            m = m.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(m.shape[:-3] + (n_seq, n_res), )
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        if self.layer_norm_z and self.linear_z:
            z = self.linear_z(self.layer_norm_z(z))
            if z.ndim == 4:
                z = torch.moveaxis(z, 3, 1)  # [B, N, N, H] -> [B, H, N, N]
            else:
                z = torch.moveaxis(z, 2, 0).unsqueeze(0)
        biases = [mask_bias]
        if z is not None:
            biases.append(z.unsqueeze(1))

        m = self.layer_norm_m(m)
        o = self.mha(q_x=m, kv_x=m, biases=biases)
        if self.transpose_input:
            o = o.transpose(-2, -3)
        return o

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.msa_att_row",
                     state_dict: dict = None,
                     using_tri_attn: bool = True):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        transpose_input = False
        if layer_path.endswith("msa_att_col"):
            layer_path = layer_path + "._msa_att"
            transpose_input = True

        weights_biases_path = [(f"{layer_path}.layer_norm_m.weight",
                                f"{layer_path}.layer_norm_m.bias")]

        pair_bias = False
        c_z = None
        if f"{layer_path}.layer_norm_z.weight" in state_dict:
            pair_bias = True
            c_z = state_dict[f"{layer_path}.linear_z.weight"].shape[1]
            num_heads = state_dict[f"{layer_path}.linear_z.weight"].shape[0]

            weights_biases_path.extend([(f"{layer_path}.layer_norm_z.weight",
                                         f"{layer_path}.layer_norm_z.bias"),
                                        (f"{layer_path}.linear_z.weight", None)
                                        ])
        else:
            num_heads = 8
        if using_tri_attn:
            mha = RefTriangleAttention.load_weights(
                model=model,
                layer_path=f"{layer_path}.mha",
                num_heads=num_heads)
            c_in = mha.c_q
            c_hidden = mha.c_hidden
            no_heads = mha.no_heads
        else:
            mha = RefPairwiseSelfAttention.load_weights(
                model=model,
                layer_path=f"{layer_path}.mha",
                num_heads=num_heads)
            c_in = mha.c_s
            c_hidden = mha.head_dim
            no_heads = mha.num_heads

        m = cls(c_in,
                c_hidden,
                no_heads,
                pair_bias=pair_bias,
                c_z=c_z,
                using_tri_attn=using_tri_attn,
                transpose_input=transpose_input)
        layers = [m.layer_norm_m]
        if pair_bias:
            layers.extend([m.layer_norm_z, m.linear_z])
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])

        setattr(m, "mha", mha)
        return m


class RefOuterProductMean(BoltzRefOuterProductMean):

    @classmethod
    def load_weights(
            cls,
            model: str = "openfold2_ptm_1",
            layer_path: str = "evoformer.blocks.0.core.outer_product_mean",
            state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.layer_norm.weight",
             f"{layer_path}.layer_norm.bias"),
            (f"{layer_path}.linear_1.weight", f"{layer_path}.linear_1.bias"),
            (f"{layer_path}.linear_2.weight", f"{layer_path}.linear_2.bias"),
            (f"{layer_path}.linear_out.weight",
             f"{layer_path}.linear_out.bias"),
        ]
        c_in = state_dict[f"{layer_path}.layer_norm.weight"].shape[0]
        c_hidden = state_dict[f"{layer_path}.linear_1.weight"].shape[0]
        c_out = state_dict[f"{layer_path}.linear_out.weight"].shape[0]

        m = cls(c_in=c_in,
                c_hidden=c_hidden,
                c_out=c_out,
                bias_flags={
                    "proj_a": True,
                    "proj_b": True,
                    "proj_o": True
                },
                norm_mask_by_eps=True,
                norm_before_output=False)
        layers = [
            m.norm,
            m.proj_a,
            m.proj_b,
            m.proj_o,
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefTriangleAttentionNode(BoltzRefTriangleAttentionNode):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.core.tri_att_start",
                     state_dict: dict = None):
        starting = True if "start" in layer_path else False
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.layer_norm.weight",
             f"{layer_path}.layer_norm.bias"),
            (f"{layer_path}.linear.weight", None),
        ]
        num_heads = state_dict[f"{layer_path}.linear.weight"].shape[0]
        mha = RefTriangleAttention.load_weights(model=model,
                                                layer_path=f"{layer_path}.mha",
                                                state_dict=state_dict,
                                                num_heads=num_heads)
        c_in = mha.c_q
        c_hidden = mha.c_hidden

        m = cls(c_in=c_in,
                c_hidden=c_hidden,
                num_heads=num_heads,
                mha_bias_flags=mha.bias_flags,
                starting=starting)
        layers = [m.layer_norm, m.linear]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        setattr(m, "mha", mha)
        return m


class RefTriangleMultiplicationNode(BoltzRefTriangleMultiplicationNode):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.core.tri_mul_out",
                     state_dict: dict = None):
        outgoing = True if "out" in layer_path else False
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.layer_norm_in.weight",
             f"{layer_path}.layer_norm_in.bias"),
            (f"{layer_path}.layer_norm_out.weight",
             f"{layer_path}.layer_norm_out.bias"),
            (f"{layer_path}.linear_z.weight", f"{layer_path}.linear_z.bias"),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
        ]
        p_in_0_weight = state_dict[f"{layer_path}.linear_a_p.weight"]
        p_in_0_bias = state_dict[f"{layer_path}.linear_a_p.bias"]
        p_in_1_weight = state_dict[f"{layer_path}.linear_b_p.weight"]
        p_in_1_bias = state_dict[f"{layer_path}.linear_b_p.bias"]
        g_in_0_weight = state_dict[f"{layer_path}.linear_a_g.weight"]
        g_in_0_bias = state_dict[f"{layer_path}.linear_a_g.bias"]
        g_in_1_weight = state_dict[f"{layer_path}.linear_b_g.weight"]
        g_in_1_bias = state_dict[f"{layer_path}.linear_b_g.bias"]

        p_in_weight = torch.cat([p_in_0_weight, p_in_1_weight], dim=0)
        p_in_bias = torch.cat([p_in_0_bias, p_in_1_bias], dim=0)
        g_in_weight = torch.cat([g_in_0_weight, g_in_1_weight], dim=0)
        g_in_bias = torch.cat([g_in_0_bias, g_in_1_bias], dim=0)

        dim = p_in_0_weight.shape[1]
        m = cls(dim=dim,
                outgoing=outgoing,
                bias_flags={
                    "p_in": True,
                    "g_in": True,
                    "p_out": True,
                    "g_out": True,
                })
        layers = [m.norm_in, m.norm_out, m.p_out, m.g_out]

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        m.p_in.weight.data.copy_(p_in_weight)
        m.g_in.weight.data.copy_(g_in_weight)
        m.p_in.bias.data.copy_(p_in_bias)
        m.g_in.bias.data.copy_(g_in_bias)
        return m


class RefPairTransition(nn.Module):

    def __init__(self, c_z, n):
        super().__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = nn.LayerNorm(self.c_z)
        self.linear_1 = nn.Linear(self.c_z, self.n * self.c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.n * self.c_z, c_z)

    @classmethod
    def load_weights(
            cls,
            model: str = "openfold2_ptm_1",
            layer_path: str = "evoformer.blocks.0.core.pair_transition",
            state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.layer_norm.weight",
             f"{layer_path}.layer_norm.bias"),
            (f"{layer_path}.linear_1.weight", f"{layer_path}.linear_1.bias"),
            (f"{layer_path}.linear_2.weight", f"{layer_path}.linear_2.bias"),
        ]
        c_z = state_dict[f"{layer_path}.layer_norm.weight"].shape[0]
        n = state_dict[f"{layer_path}.linear_1.weight"].shape[0] // c_z
        m = cls(c_z=c_z, n=n)
        layers = [m.layer_norm, m.linear_1, m.linear_2]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        z = z * mask
        return z


class RefMSATransition(nn.Module):

    def __init__(self, c_m, n):
        super().__init__()
        self.c_m = c_m
        self.n = n
        self.layer_norm = nn.LayerNorm(self.c_m)
        self.linear_1 = nn.Linear(self.c_m, self.n * self.c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.n * self.c_m, c_m)

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.core.msa_transition",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.layer_norm.weight",
             f"{layer_path}.layer_norm.bias"),
            (f"{layer_path}.linear_1.weight", f"{layer_path}.linear_1.bias"),
            (f"{layer_path}.linear_2.weight", f"{layer_path}.linear_2.bias"),
        ]
        c_m = state_dict[f"{layer_path}.layer_norm.weight"].shape[0]
        n = state_dict[f"{layer_path}.linear_1.weight"].shape[0] // c_m
        m = cls(c_m=c_m, n=n)
        layers = [m.layer_norm, m.linear_1, m.linear_2]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m


class RefEvoformerBlock(nn.Module):
    """ Reference: https://github.com/aqlaboratory/openfold/blob/main/openfold/model/evoformer.py#L100 """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_hidden_msa_att: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 transition_n: int,
                 no_column_attention: bool,
                 opm_first: bool,
                 inf: float = 1e9,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden_msa_att = c_hidden_msa_att
        self.c_hidden_opm = c_hidden_opm
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_msa = no_heads_msa
        self.no_heads_pair = no_heads_pair
        self.transition_n = transition_n
        self.no_column_attention = no_column_attention
        self.opm_first = opm_first
        self.inf = inf
        self.eps = eps

        self.msa_att_row = RefMSAAttention(c_in=c_m,
                                           c_hidden=c_hidden_msa_att,
                                           no_heads=no_heads_msa,
                                           pair_bias=True,
                                           c_z=c_z,
                                           inf=inf,
                                           using_tri_attn=True)
        self.msa_transition = RefMSATransition(c_m=c_m, n=transition_n)
        self.outer_product_mean = RefOuterProductMean(c_in=c_m,
                                                      c_hidden=c_hidden_opm,
                                                      c_out=c_z,
                                                      bias_flags={
                                                          "proj_a": True,
                                                          "proj_b": True,
                                                          "proj_o": True
                                                      },
                                                      norm_mask_by_eps=True,
                                                      norm_before_output=False)

        self.tri_mul_out = RefTriangleMultiplicationNode(dim=c_z,
                                                         outgoing=True,
                                                         bias_flags={
                                                             "p_out": True,
                                                             "g_out": True,
                                                             "p_in": True,
                                                             "g_in": True,
                                                         })
        self.tri_mul_in = RefTriangleMultiplicationNode(dim=c_z,
                                                        outgoing=False,
                                                        bias_flags={
                                                            "p_in": True,
                                                            "g_in": True,
                                                            "p_out": True,
                                                            "g_out": True,
                                                        })
        self.tri_attn_start = RefTriangleAttentionNode(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            num_heads=no_heads_pair,
            mha_bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "o": True,
            },
            inf=inf,
            starting=True)
        self.tri_attn_end = RefTriangleAttentionNode(c_in=c_z,
                                                     c_hidden=c_hidden_pair_att,
                                                     num_heads=no_heads_pair,
                                                     mha_bias_flags={
                                                         "q": False,
                                                         "k": False,
                                                         "v": False,
                                                         "g": True,
                                                         "o": True,
                                                     },
                                                     inf=inf,
                                                     starting=False)
        self.pair_transition = RefPairTransition(c_z=c_z, n=transition_n)
        if not self.no_column_attention:
            self.msa_att_col = RefMSAAttention(c_in=c_m,
                                               c_hidden=c_hidden_msa_att,
                                               no_heads=no_heads_msa,
                                               pair_bias=False,
                                               c_z=c_z,
                                               inf=inf,
                                               using_tri_attn=True,
                                               transpose_input=True)

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        msa_att_row = RefMSAAttention.load_weights(
            model=model,
            layer_path=f"{layer_path}.msa_att_row",
            state_dict=state_dict)
        msa_att_col = RefMSAAttention.load_weights(
            model=model,
            layer_path=f"{layer_path}.msa_att_col",
            state_dict=state_dict)
        msa_transition = RefMSATransition.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.msa_transition",
            state_dict=state_dict)
        outer_product_mean = RefOuterProductMean.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.outer_product_mean",
            state_dict=state_dict)
        tri_mul_out = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_mul_out",
            state_dict=state_dict)
        tri_mul_in = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_mul_in",
            state_dict=state_dict)
        tri_attn_start = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_att_start",
            state_dict=state_dict)
        tri_attn_end = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_att_end",
            state_dict=state_dict)
        pair_transition = RefPairTransition.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.pair_transition",
            state_dict=state_dict)
        m = cls(c_m=msa_att_row.c_in,
                c_z=outer_product_mean.c_out,
                c_hidden_msa_att=msa_att_row.c_hidden,
                c_hidden_opm=outer_product_mean.c_hidden,
                c_hidden_mul=tri_mul_out.dim,
                c_hidden_pair_att=tri_attn_start.c_hidden,
                no_heads_msa=msa_att_row.no_heads,
                no_heads_pair=tri_attn_start.num_heads,
                transition_n=msa_transition.n,
                no_column_attention=False,
                opm_first=False,
                inf=msa_att_row.inf,
                eps=msa_att_row.eps)
        setattr(m, "msa_att_row", msa_att_row)
        setattr(m, "msa_att_col", msa_att_col)
        setattr(m, "msa_transition", msa_transition)
        setattr(m, "outer_product_mean", outer_product_mean)
        setattr(m, "tri_mul_out", tri_mul_out)
        setattr(m, "tri_mul_in", tri_mul_in)
        setattr(m, "tri_attn_start", tri_attn_start)
        setattr(m, "tri_attn_end", tri_attn_end)
        setattr(m, "pair_transition", pair_transition)
        return m

    def _compute_opm(self, m: torch.Tensor, z: torch.Tensor,
                     msa_mask: torch.Tensor) -> torch.Tensor:
        opm = self.outer_product_mean(m, msa_mask)
        z = z + opm
        return m, z

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)
        m = m + self.msa_att_row(m, z=z, mask=msa_mask)
        if not self.no_column_attention:
            m = m + self.msa_att_col(m, z=None, mask=msa_mask)

        msa_trans_mask = msa_mask
        m = m + self.msa_transition(m, mask=msa_trans_mask)
        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)

        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_attn_start(z, pair_mask)
        z = z + self.tri_attn_end(z, pair_mask)

        pair_trans_mask = pair_mask
        z = z + self.pair_transition(z, pair_trans_mask)
        return m, z


class RefMSAColumnGlobalAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, inf=1e9, eps=1e-5):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = RefGlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "extra_msa_stack.blocks.0.msa_att_col",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        global_attention = RefGlobalAttention.load_weights(
            model=model,
            layer_path=f"{layer_path}.global_attention",
            state_dict=state_dict)
        m = cls(c_in=global_attention.c_in,
                c_hidden=global_attention.c_hidden,
                no_heads=global_attention.no_heads)
        setattr(m, "global_attention", global_attention)

        m.layer_norm_m.weight.data.copy_(
            state_dict[f"{layer_path}.layer_norm_m.weight"])
        m.layer_norm_m.bias.data.copy_(
            state_dict[f"{layer_path}.layer_norm_m.bias"])
        return m

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]

        if mask is None:
            # [*, N_seq, N_res]
            mask = torch.ones(
                m.shape[:-1],
                dtype=m.dtype,
                device=m.device,
            ).detach()

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        m = self.layer_norm_m(m)
        m = self.global_attention(m=m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m


class RefExtraMSABlock(RefEvoformerBlock):

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_hidden_msa_att: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 transition_n: int,
                 opm_first: bool,
                 inf: float = 1e9,
                 eps: float = 1e-5):
        super().__init__(c_m=c_m,
                         c_z=c_z,
                         c_hidden_msa_att=c_hidden_msa_att,
                         c_hidden_opm=c_hidden_opm,
                         c_hidden_mul=c_hidden_mul,
                         c_hidden_pair_att=c_hidden_pair_att,
                         no_heads_msa=no_heads_msa,
                         no_heads_pair=no_heads_pair,
                         transition_n=transition_n,
                         no_column_attention=True,
                         opm_first=opm_first,
                         inf=inf,
                         eps=eps)
        self.msa_att_col = RefMSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps)

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                pair_mask: torch.Tensor):
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)
        m = m + self.msa_att_row(m, z=z, mask=msa_mask)
        m = m + self.msa_att_col(m, mask=msa_mask)

        msa_trans_mask = msa_mask
        m = m + self.msa_transition(m, mask=msa_trans_mask)
        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)

        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_attn_start(z, pair_mask)
        z = z + self.tri_attn_end(z, pair_mask)

        pair_trans_mask = pair_mask
        z = z + self.pair_transition(z, pair_trans_mask)
        return m, z

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "extra_msa_stack.blocks.0",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        msa_att_row = RefMSAAttention.load_weights(
            model=model,
            layer_path=f"{layer_path}.msa_att_row",
            state_dict=state_dict)
        msa_att_col = RefMSAColumnGlobalAttention.load_weights(
            model=model,
            layer_path=f"{layer_path}.msa_att_col",
            state_dict=state_dict)
        msa_transition = RefMSATransition.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.msa_transition",
            state_dict=state_dict)
        outer_product_mean = RefOuterProductMean.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.outer_product_mean",
            state_dict=state_dict)
        tri_mul_out = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_mul_out",
            state_dict=state_dict)
        tri_mul_in = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_mul_in",
            state_dict=state_dict)
        tri_attn_start = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_att_start",
            state_dict=state_dict)
        tri_attn_end = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.tri_att_end",
            state_dict=state_dict)
        pair_transition = RefPairTransition.load_weights(
            model=model,
            layer_path=f"{layer_path}.core.pair_transition",
            state_dict=state_dict)
        m = cls(c_m=msa_att_row.c_in,
                c_z=outer_product_mean.c_out,
                c_hidden_msa_att=msa_att_row.c_hidden,
                c_hidden_opm=outer_product_mean.c_hidden,
                c_hidden_mul=tri_mul_out.dim,
                c_hidden_pair_att=tri_attn_start.c_hidden,
                no_heads_msa=msa_att_row.no_heads,
                no_heads_pair=tri_attn_start.num_heads,
                transition_n=msa_transition.n,
                opm_first=False,
                inf=msa_att_row.inf,
                eps=msa_att_row.eps)
        setattr(m, "msa_att_row", msa_att_row)
        setattr(m, "msa_att_col", msa_att_col)
        setattr(m, "msa_transition", msa_transition)
        setattr(m, "outer_product_mean", outer_product_mean)
        setattr(m, "tri_mul_out", tri_mul_out)
        setattr(m, "tri_mul_in", tri_mul_in)
        setattr(m, "tri_attn_start", tri_attn_start)
        setattr(m, "tri_attn_end", tri_attn_end)
        setattr(m, "pair_transition", pair_transition)
        return m


class RefDiffusionModule(nn.Module):

    def __init__(self, token_s: int, atom_s: int, atoms_per_window_queries: int,
                 atoms_per_window_keys: int, dim_fourier: int,
                 atom_encoder_depth: int, atom_encoder_heads: int,
                 token_transformer_depth: int, token_transformer_heads: int,
                 atom_decoder_depth: int, atom_decoder_heads: int,
                 conditioning_transition_layers: int):
        super().__init__()

        self.token_s = token_s
        self.atom_s = atom_s
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.dim_fourier = dim_fourier
        self.atom_encoder_depth = atom_encoder_depth
        self.atom_encoder_heads = atom_encoder_heads
        self.token_transformer_depth = token_transformer_depth
        self.token_transformer_heads = token_transformer_heads
        self.atom_decoder_depth = atom_decoder_depth
        self.atom_decoder_heads = atom_decoder_heads
        self.conditioning_transition_layers = conditioning_transition_layers

        self.single_conditioner = RefSingleConditioning(
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers)

        self.atom_attention_encoder = RefAtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys)

        self.atom_attention_decoder = RefAtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys)

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s),
            nn.Linear(2 * token_s, 2 * token_s, bias=False))

        self.token_transformer = BoltzRefDiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            heads=token_transformer_heads,
            num_blocks=token_transformer_depth,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

    @classmethod
    def load_weights(cls,
                     attn_window_queries: int = 32,
                     attn_window_keys: int = 128,
                     model: str = "boltz-2",
                     layer_path: str = "structure_module.score_model",
                     state_dict: Optional[dict] = None) -> 'RefDiffusionModule':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        single_conditioner = RefSingleConditioning.load_weights(
            model=model, layer_path=layer_path + ".single_conditioner")

        atom_attention_encoder = RefAtomAttentionEncoder.load_weights(
            model=model,
            layer_path=layer_path + ".atom_attention_encoder",
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys)

        atom_attention_decoder = RefAtomAttentionDecoder.load_weights(
            model=model,
            layer_path=layer_path + ".atom_attention_decoder",
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys)

        token_transformer = BoltzRefDiffusionTransformer.load_weights(
            model=model, layer_path=layer_path + ".token_transformer")

        s_to_a_linear_layer_norm_weight = state_dict[layer_path +
                                                     ".s_to_a_linear.0.weight"]
        s_to_a_linear_layer_norm_bias = state_dict[layer_path +
                                                   ".s_to_a_linear.0.bias"]

        s_to_a_linear_layer_linear_weight = state_dict[
            layer_path + ".s_to_a_linear.1.weight"]

        a_norm_weight = state_dict[layer_path + ".a_norm.weight"]
        a_norm_bias = state_dict[layer_path + ".a_norm.bias"]

        token_s = single_conditioner.token_s
        atom_s = atom_attention_encoder.atom_s

        atom_encoder_depth = atom_attention_encoder.atom_encoder.diffusion_transformer.num_blocks
        atom_encoder_heads = atom_attention_encoder.atom_encoder.diffusion_transformer.heads

        atom_decoder_depth = atom_attention_decoder.atom_decoder.diffusion_transformer.num_blocks
        atom_decoder_heads = atom_attention_decoder.atom_decoder.diffusion_transformer.heads

        token_transformer_depth = token_transformer.num_blocks
        token_transformer_heads = 16
        for i in range(token_transformer_depth):
            token_transformer.layers[i].pair_bias_attn.num_heads = 16
            token_transformer.layers[i].pair_bias_attn.head_dim = 48

        conditioning_transition_layers = len(single_conditioner.transitions)
        dim_fourier = single_conditioner.dim_fourier
        diffusion_module = cls(
            token_s=token_s,
            atom_s=atom_s,
            atoms_per_window_queries=attn_window_queries,
            atoms_per_window_keys=attn_window_keys,
            dim_fourier=dim_fourier,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            token_transformer_depth=token_transformer_depth,
            token_transformer_heads=token_transformer_heads,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            conditioning_transition_layers=conditioning_transition_layers)
        setattr(diffusion_module, "single_conditioner", single_conditioner)
        setattr(diffusion_module, "atom_attention_encoder",
                atom_attention_encoder)
        setattr(diffusion_module, "atom_attention_decoder",
                atom_attention_decoder)
        setattr(diffusion_module, "token_transformer", token_transformer)
        diffusion_module.s_to_a_linear[0].weight.data.copy_(
            s_to_a_linear_layer_norm_weight)
        diffusion_module.s_to_a_linear[0].bias.data.copy_(
            s_to_a_linear_layer_norm_bias)
        diffusion_module.s_to_a_linear[1].weight.data.copy_(
            s_to_a_linear_layer_linear_weight)
        diffusion_module.a_norm.weight.data.copy_(a_norm_weight)
        diffusion_module.a_norm.bias.data.copy_(a_norm_bias)
        return diffusion_module

    def forward(
            self,
            atom_to_token,
            atom_pad_mask,
            token_pad_mask,
            s_inputs,  # Float['b n ts']
            s_trunk,  # Float['b n ts']
            r_noisy,  # Float['bm m 3']
            times,  # Float['bm 1 1']
            diffusion_conditioning_q,
            diffusion_conditioning_c,
            diffusion_conditioning_atom_enc_bias,
            diffusion_conditioning_token_trans_bias,
            diffusion_conditioning_atom_dec_bias,
            multiplicity=1,
            attn_metadata=None):

        s, _ = self.single_conditioner(
            times,
            s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs.repeat_interleave(multiplicity, 0),
        )

        a, q_skip, c_skip = self.atom_attention_encoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            q=diffusion_conditioning_q,
            c=diffusion_conditioning_c,
            atom_enc_bias=diffusion_conditioning_atom_enc_bias,
            r=r_noisy,
            multiplicity=multiplicity,
            attn_metadata=attn_metadata)

        a = a + self.s_to_a_linear(s)

        mask = token_pad_mask.repeat_interleave(multiplicity, 0)
        a = self.token_transformer(a=a,
                                   mask=mask,
                                   s=s,
                                   z=diffusion_conditioning_token_trans_bias)

        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning_atom_dec_bias,
            multiplicity=multiplicity,
            attn_metadata=attn_metadata)
        return r_update


class RefInputEmbedder(nn.Module):

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = nn.Linear(self.no_bins, c_z)

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "input_embedder",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear_tf_z_i.weight",
             f"{layer_path}.linear_tf_z_i.bias"),
            (f"{layer_path}.linear_tf_z_j.weight",
             f"{layer_path}.linear_tf_z_j.bias"),
            (f"{layer_path}.linear_tf_m.weight",
             f"{layer_path}.linear_tf_m.bias"),
            (f"{layer_path}.linear_msa_m.weight",
             f"{layer_path}.linear_msa_m.bias"),
            (f"{layer_path}.linear_relpos.weight",
             f"{layer_path}.linear_relpos.bias"),
        ]
        c_z = state_dict[f"{layer_path}.linear_tf_z_i.weight"].shape[0]
        tf_dim = state_dict[f"{layer_path}.linear_tf_z_i.weight"].shape[1]
        c_m = state_dict[f"{layer_path}.linear_msa_m.weight"].shape[0]
        msa_dim = state_dict[f"{layer_path}.linear_msa_m.weight"].shape[1]
        relpos_k = 32

        m = cls(tf_dim=tf_dim,
                msa_dim=msa_dim,
                c_z=c_z,
                c_m=c_m,
                relpos_k=relpos_k)
        layers = [
            m.linear_tf_z_i, m.linear_tf_z_j, m.linear_tf_m, m.linear_msa_m,
            m.linear_relpos
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k,
                                  end=self.relpos_k + 1,
                                  device=d.device)
        reshaped_bins = boundaries.view(((1, ) * len(d.shape)) +
                                        (len(boundaries), ))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "target_feat":
                    Features of shape [*, N_res, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                "msa_feat":
                    Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = pair_emb + tf_emb_i[..., None, :]
        pair_emb = pair_emb + tf_emb_j[..., None, :, :]

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (self.linear_tf_m(tf).unsqueeze(-3).expand(
            ((-1, ) * len(tf.shape[:-2]) + (n_clust, -1, -1))))
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class RefInputEmbedderMultimer(nn.Module):

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_idx: int,
        use_chain_relative: bool,
        max_relative_chain: int,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_dim, c_m)

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if (self.use_chain_relative):
            self.no_bins = (2 * max_relative_idx + 2 + 1 +
                            2 * max_relative_chain + 2)
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = nn.Linear(self.no_bins, c_z)

    @classmethod
    def load_weights(cls,
                     model: str = "alphafold2_multimer_1",
                     layer_path: str = "input_embedder",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear_tf_z_i.weight",
             f"{layer_path}.linear_tf_z_i.bias"),
            (f"{layer_path}.linear_tf_z_j.weight",
             f"{layer_path}.linear_tf_z_j.bias"),
            (f"{layer_path}.linear_tf_m.weight",
             f"{layer_path}.linear_tf_m.bias"),
            (f"{layer_path}.linear_msa_m.weight",
             f"{layer_path}.linear_msa_m.bias"),
            (f"{layer_path}.linear_relpos.weight",
             f"{layer_path}.linear_relpos.bias"),
        ]
        c_z = state_dict[f"{layer_path}.linear_tf_z_i.weight"].shape[0]
        tf_dim = state_dict[f"{layer_path}.linear_tf_z_i.weight"].shape[1]
        c_m = state_dict[f"{layer_path}.linear_msa_m.weight"].shape[0]
        msa_dim = state_dict[f"{layer_path}.linear_msa_m.weight"].shape[1]
        max_relative_idx = 32
        use_chain_relative = True
        max_relative_chain = 2

        m = cls(tf_dim=tf_dim,
                msa_dim=msa_dim,
                c_z=c_z,
                c_m=c_m,
                max_relative_idx=max_relative_idx,
                use_chain_relative=use_chain_relative,
                max_relative_chain=max_relative_chain)
        layers = [
            m.linear_tf_z_i, m.linear_tf_z_j, m.linear_tf_m, m.linear_msa_m,
            m.linear_relpos
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def one_hot(self, x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
        reshaped_bins = v_bins.view(((1, ) * len(x.shape)) + (len(v_bins), ))
        diffs = x[..., None] - reshaped_bins
        am = torch.argmin(torch.abs(diffs), dim=-1)
        return F.one_hot(am, num_classes=len(v_bins)).float()

    def relpos(self, residue_index, asym_id, entity_id, sym_id):
        pos = residue_index
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(offset + self.max_relative_idx, 0,
                                     2 * self.max_relative_idx)

        rel_feats = []
        if (self.use_chain_relative):
            final_offset = torch.where(asym_id_same, clipped_offset,
                                       (2 * self.max_relative_idx + 1) *
                                       torch.ones_like(clipped_offset))
            boundaries = torch.arange(start=0,
                                      end=2 * self.max_relative_idx + 2,
                                      device=final_offset.device)
            rel_pos = self.one_hot(
                final_offset,
                boundaries,
            )

            rel_feats.append(rel_pos)

            entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
            rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain,
                0,
                2 * max_rel_chain,
            )

            final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                          (2 * max_rel_chain + 1) *
                                          torch.ones_like(clipped_rel_chain))

            boundaries = torch.arange(start=0,
                                      end=2 * max_rel_chain + 2,
                                      device=final_rel_chain.device)
            rel_chain = self.one_hot(
                final_rel_chain,
                boundaries,
            )

            rel_feats.append(rel_chain)
        else:
            boundaries = torch.arange(start=0,
                                      end=2 * self.max_relative_idx + 1,
                                      device=clipped_offset.device)
            rel_pos = self.one_hot(
                clipped_offset,
                boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats,
                             dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)

    def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor,
                msa_feat: torch.Tensor, asym_id: torch.Tensor,
                entity_id: torch.Tensor,
                sym_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(target_feat)
        tf_emb_j = self.linear_tf_z_j(target_feat)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(residue_index, asym_id, entity_id,
                                          sym_id)

        # [*, N_clust, N_res, c_m]
        n_clust = msa_feat.shape[-3]
        tf_m = (self.linear_tf_m(target_feat).unsqueeze(-3).expand(
            ((-1, ) * len(target_feat.shape[:-2]) + (n_clust, -1, -1))))
        msa_emb = self.linear_msa_m(msa_feat) + tf_m

        return msa_emb, pair_emb


class RefRecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = nn.Linear(self.no_bins, self.c_z)
        self.layer_norm_m = nn.LayerNorm(self.c_m)
        self.layer_norm_z = nn.LayerNorm(self.c_z)

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "recycling_embedder",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear.weight", f"{layer_path}.linear.bias"),
            (f"{layer_path}.layer_norm_m.weight",
             f"{layer_path}.layer_norm_m.bias"),
            (f"{layer_path}.layer_norm_z.weight",
             f"{layer_path}.layer_norm_z.bias"),
        ]
        c_m = state_dict[f"{layer_path}.layer_norm_m.weight"].shape[0]
        c_z = state_dict[f"{layer_path}.layer_norm_z.weight"].shape[0]
        min_bin = 3.25
        max_bin = 20.75
        no_bins = 15
        inf = 1e9

        m = cls(c_m=c_m,
                c_z=c_z,
                min_bin=min_bin,
                max_bin=max_bin,
                no_bins=no_bins,
                inf=inf)
        layers = [m.linear, m.layer_norm_m, m.layer_norm_z]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, m: torch.Tensor, z: torch.Tensor,
                x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
        )
        squared_bins = bins**2
        upper = torch.cat(
            [squared_bins[1:],
             squared_bins.new_tensor([self.inf])], dim=-1)
        d = torch.sum((x[..., None, :] - x[..., None, :, :])**2,
                      dim=-1,
                      keepdims=True)

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = z_update + d

        return m_update, z_update


class RefTemplateSingleEmbedder(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = nn.Linear(self.c_in, self.c_out)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.c_out, self.c_out)

    @classmethod
    def load_weights(
            cls,
            model: str = "openfold2_ptm_1",
            layer_path: str = "template_embedder.template_single_embedder",
            state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear_1.weight", f"{layer_path}.linear_1.bias"),
            (f"{layer_path}.linear_2.weight", f"{layer_path}.linear_2.bias"),
        ]
        c_in = state_dict[f"{layer_path}.linear_1.weight"].shape[1]
        c_out = state_dict[f"{layer_path}.linear_2.weight"].shape[0]

        m = cls(c_in=c_in, c_out=c_out)
        layers = [m.linear_1, m.linear_2]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class RefTemplatePairEmbedder(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
    ):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = nn.Linear(self.c_in, self.c_out)

    @classmethod
    def load_weights(
            cls,
            model: str = "openfold2_ptm_1",
            layer_path: str = "template_embedder.template_pair_embedder",
            state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear.weight", f"{layer_path}.linear.bias"),
        ]
        c_out, c_in = state_dict[f"{layer_path}.linear.weight"].shape
        m = cls(c_in=c_in, c_out=c_out)
        layers = [m.linear]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class RefTemplatePairStackBlock(nn.Module):

    def __init__(self,
                 c_t: int,
                 c_hidden_tri_att: int,
                 c_hidden_tri_mul: int,
                 no_heads: int,
                 pair_transition_n: int,
                 tri_mul_first: bool,
                 inf: float = 1e9):
        super().__init__()

        assert c_t == c_hidden_tri_mul, "c_t and c_hidden_tri_mul must be the same for Triangle Multiplication"
        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.tri_mul_first = tri_mul_first
        self.inf = inf

        self.tri_attn_start = RefTriangleAttentionNode(
            c_in=c_t,
            c_hidden=c_hidden_tri_att,
            num_heads=no_heads,
            mha_bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "o": True,
            },
            inf=inf,
            starting=True)
        self.tri_attn_end = RefTriangleAttentionNode(c_in=c_t,
                                                     c_hidden=c_hidden_tri_att,
                                                     num_heads=no_heads,
                                                     mha_bias_flags={
                                                         "q": False,
                                                         "k": False,
                                                         "v": False,
                                                         "g": True,
                                                         "o": True,
                                                     },
                                                     inf=inf,
                                                     starting=False)
        self.tri_mul_out = RefTriangleMultiplicationNode(dim=c_t,
                                                         outgoing=True,
                                                         bias_flags={
                                                             "p_out": True,
                                                             "g_out": True,
                                                             "p_in": True,
                                                             "g_in": True,
                                                         })
        self.tri_mul_in = RefTriangleMultiplicationNode(dim=c_t,
                                                        outgoing=False,
                                                        bias_flags={
                                                            "p_in": True,
                                                            "g_in": True,
                                                            "p_out": True,
                                                            "g_out": True,
                                                        })
        self.pair_transition = RefPairTransition(c_z=c_t, n=pair_transition_n)

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "template_pair_stack.blocks.0",
                     state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        tri_mul_out = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.tri_mul_out",
            state_dict=state_dict)
        tri_mul_in = RefTriangleMultiplicationNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.tri_mul_in",
            state_dict=state_dict)
        tri_attn_start = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.tri_att_start",
            state_dict=state_dict)
        tri_attn_end = RefTriangleAttentionNode.load_weights(
            model=model,
            layer_path=f"{layer_path}.tri_att_end",
            state_dict=state_dict)
        pair_transition = RefPairTransition.load_weights(
            model=model,
            layer_path=f"{layer_path}.pair_transition",
            state_dict=state_dict)

        c_t = c_hidden_tri_mul = tri_mul_out.dim
        c_hidden_tri_att = tri_attn_start.c_hidden
        no_heads = tri_attn_start.num_heads
        pair_transition_n = pair_transition.n

        if "multimer" in model:
            tri_mul_first = True
        else:
            tri_mul_first = False

        inf = 1e9
        m = cls(c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                tri_mul_first=tri_mul_first,
                inf=inf)

        setattr(m, "tri_mul_out", tri_mul_out)
        setattr(m, "tri_mul_in", tri_mul_in)
        setattr(m, "tri_attn_start", tri_attn_start)
        setattr(m, "tri_attn_end", tri_attn_end)
        setattr(m, "pair_transition", pair_transition)
        return m

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        single_templates = [t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)]
        single_templates_masks = [
            m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
        ]

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]

            if self.tri_mul_first:
                single = single + self.tri_mul_out(single, single_mask)
                single = single + self.tri_mul_in(single, single_mask)
                single = single + self.tri_attn_start(single, single_mask)
                single = single + self.tri_attn_end(single, single_mask)
            else:
                single = single + self.tri_attn_start(single, single_mask)
                single = single + self.tri_attn_end(single, single_mask)
                single = single + self.tri_mul_out(single, single_mask)
                single = single + self.tri_mul_in(single, single_mask)

            single = single + self.pair_transition(single, single_mask)

            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z


class RefTemplatePointwiseAttention(nn.Module):

    def __init__(self, c_t, c_z, c_hidden, no_heads, inf):
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.mha = RefTriangleAttention(c_q=c_z,
                                        c_k=c_t,
                                        c_v=c_t,
                                        c_hidden=c_hidden,
                                        no_heads=self.no_heads,
                                        bias_flags={
                                            "q": False,
                                            "k": False,
                                            "v": False,
                                            "g": False,
                                            "o": True
                                        })

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "template_pointwise_att",
                     state_dict: dict = None,
                     using_tri_attn: bool = True):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        num_heads = 4
        mha = RefTriangleAttention.load_weights(model=model,
                                                layer_path=f"{layer_path}.mha",
                                                num_heads=num_heads)
        c_z = mha.c_q
        c_t = mha.c_k
        c_hidden = mha.c_hidden
        no_heads = mha.no_heads
        inf = 1e9
        m = cls(c_t=c_t, c_z=c_z, c_hidden=c_hidden, no_heads=no_heads, inf=inf)
        setattr(m, "mha", mha)
        return m

    def forward(self,
                t: torch.Tensor,
                z: torch.Tensor,
                template_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        batch_dims = " ".join([f"b_{i}" for i in range(t.ndim - 4)])
        # t = permute_final_dims(t, (1, 2, 0, 3))
        t = rearrange(t, f"{batch_dims} t i j c -> {batch_dims} i j t c")

        # [*, N_res, N_res, 1, C_z]
        biases = [bias]

        z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z
