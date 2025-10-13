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
import torch
import torch.nn as nn
from test_utils.boltz.ref_attn import \
    RefPairwiseSelfAttention as BoltzRefPairwiseSelfAttention
from test_utils.boltz.ref_attn import \
    RefTriangleAttention as BoltzRefTriangleAttention

from tensorrt_bionemo.hubs import load_weights


class RefPairwiseSelfAttention(BoltzRefPairwiseSelfAttention):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.msa_att_row.mha",
                     state_dict: dict = None,
                     num_heads: int = 8):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear_q.weight", None),
            (f"{layer_path}.linear_k.weight", None),
            (f"{layer_path}.linear_v.weight", None),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
            (f"{layer_path}.linear_o.weight", f"{layer_path}.linear_o.bias"),
        ]
        c_q = c_k = c_v = state_dict[f"{layer_path}.linear_q.weight"].shape[1]

        m = cls(c_s=c_q,
                c_z=None,
                num_heads=num_heads,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                },
                compute_pair_bias=False,
                initial_norm=False,
                transform_mask=False)
        layers = [m.proj_q, m.proj_k, m.proj_v, m.proj_g, m.proj_o]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])

        return m


class RefTriangleAttention(BoltzRefTriangleAttention):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.msa_att_row.mha",
                     state_dict: dict = None,
                     num_heads: int = 8):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.linear_q.weight", None),
            (f"{layer_path}.linear_k.weight", None),
            (f"{layer_path}.linear_v.weight", None),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
            (f"{layer_path}.linear_o.weight", f"{layer_path}.linear_o.bias"),
        ]
        c_q = c_k = c_v = state_dict[f"{layer_path}.linear_q.weight"].shape[1]
        c_hidden = c_q // num_heads
        m = cls(c_q=c_q,
                c_k=c_k,
                c_v=c_v,
                c_hidden=c_hidden,
                no_heads=num_heads,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                })
        layers = [m.linear_q, m.linear_k, m.linear_v, m.linear_g, m.linear_o]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefGlobalAttention(nn.Module):

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 no_heads: int,
                 inf: float = 1e9,
                 eps: float = 1e-5):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = nn.Linear(c_in, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_in, c_hidden, bias=False)
        self.linear_v = nn.Linear(c_in, c_hidden, bias=False)
        self.linear_g = nn.Linear(c_in, c_hidden * no_heads)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_in)

        self.sigmoid = nn.Sigmoid()

    @classmethod
    def load_weights(
            cls,
            model: str = "openfold2_ptm_1",
            layer_path:
        str = "extra_msa_stack.blocks.0.msa_att_col.global_attention",
            state_dict: dict = None,
            num_heads: int = 8):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.linear_q.weight", None),
            (f"{layer_path}.linear_k.weight", None),
            (f"{layer_path}.linear_v.weight", None),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
            (f"{layer_path}.linear_o.weight", f"{layer_path}.linear_o.bias"),
        ]
        c_in = state_dict[f"{layer_path}.linear_q.weight"].shape[1]
        c_hidden = state_dict[f"{layer_path}.linear_k.weight"].shape[0]
        no_heads = state_dict[f"{layer_path}.linear_q.weight"].shape[
            0] // c_hidden
        m = cls(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads)
        layers = [m.linear_q, m.linear_k, m.linear_v, m.linear_g, m.linear_o]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1),
                      dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.c_hidden**(-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        # [*, N_res, H, N_seq]
        a = torch.matmul(
            q,
            k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
        )
        a += bias
        a = torch.nn.functional.softmax(a, dim=-1)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a,
            v,
        )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1, ))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m
