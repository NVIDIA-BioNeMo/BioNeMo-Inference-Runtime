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
from test_utils.boltz.ref_attn import (RefPairwiseSelfAttention,
                                       RefTriangleAttention)

from tensorrt_bionemo.hubs import load_weights


class RefTriangleMultiplicationNode(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_mult.py"""

    def __init__(
        self,
        dim: int = 128,
        outgoing: bool = True,
        bias_flags: dict[str, bool] = {
            "p_in": False,
            "g_in": False,
            "p_out": False,
            "g_out": False,
        }
    ) -> None:
        super().__init__()
        self.dim = dim
        self.outgoing = outgoing
        self.bias_flags = bias_flags
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=bias_flags["p_in"])
        self.g_in = nn.Linear(dim, 2 * dim, bias=bias_flags["g_in"])

        self.norm_out = nn.LayerNorm(dim, eps=1e-5)
        self.p_out = nn.Linear(dim, dim, bias=bias_flags["p_out"])
        self.g_out = nn.Linear(dim, dim, bias=bias_flags["g_out"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, D]
            mask: [*, I, J]
        Returns:
            [*, I, J, D]
        """
        if x.dtype != mask.dtype:
            x = x.to(mask.dtype)
        x = self.norm_in(x)
        x_in = x
        a, b = torch.chunk(self.p_in(x), 2, dim=-1)
        x = self.p_in(x) * self.g_in(x).sigmoid()
        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)
        # Triangular projection
        if x.dim() == 4:
            if self.outgoing:
                x = torch.einsum("bikd,bjkd->bijd", a, b)
            else:
                x = torch.einsum("bkid,bkjd->bijd", a, b)
        elif x.dim() == 3:
            if self.outgoing:
                x = torch.einsum("ikd,jkd->ijd", a, b)
            else:
                x = torch.einsum("kid,kjd->ijd", a, b)
        # Output gating
        self.norm_out = self.norm_out.float()  # cast to float
        self.p_out = self.p_out.float()  # cast to float
        self.g_out = self.g_out.float()  # cast to float
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in.float()).sigmoid()
        if x.dtype != mask.dtype:
            x = x.to(mask.dtype)
        return x

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.tri_mul_out",
            outgoing: bool = True,
            state_dict: Optional[dict] = None
    ) -> 'RefTriangleMultiplicationNode':
        if not outgoing:
            layer_path = layer_path.replace("tri_mul_out", "tri_mul_in")
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.norm_in.weight", f"{layer_path}.norm_in.bias"),
            (f"{layer_path}.p_in.weight", None),
            (f"{layer_path}.g_in.weight", None),
            (f"{layer_path}.norm_out.weight", f"{layer_path}.norm_out.bias"),
            (f"{layer_path}.p_out.weight", None),
            (f"{layer_path}.g_out.weight", None),
        ]
        w_q = state_dict[weights_biases_path[1][0]]
        dim = w_q.shape[0] // 2
        mul_node = cls(dim, outgoing=outgoing)
        layers = [
            mul_node.norm_in, mul_node.p_in, mul_node.g_in, mul_node.norm_out,
            mul_node.p_out, mul_node.g_out
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return mul_node


class RefTriangleAttentionNode(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_attention/attention.py """

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 num_heads: int,
                 starting: bool = True,
                 mha_bias_flags: dict[str, bool] = {
                     "q": False,
                     "k": False,
                     "v": False,
                     "g": False,
                     "o": False,
                 },
                 inf: float = 1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf
        self.mha_bias_flags = mha_bias_flags

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = nn.Linear(c_in, self.num_heads, bias=False)

        self.mha = RefTriangleAttention(self.c_in,
                                        self.c_in,
                                        self.c_in,
                                        self.c_hidden,
                                        self.num_heads,
                                        bias_flags=mha_bias_flags)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.tri_att_start",
            no_heads: int = 4,
            starting: bool = True,
            state_dict: Optional[dict] = None) -> 'RefTriangleAttentionNode':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_path = [
            f"{layer_path}.linear.weight",
            f"{layer_path}.layer_norm.weight",
        ]
        no_heads = state_dict[weights_path[0]].shape[0]
        c_in = state_dict[weights_path[0]].shape[1]
        mha = RefTriangleAttention.load_weights(state_dict=state_dict,
                                                layer_path=layer_path + ".mha",
                                                no_heads=no_heads)
        c_hidden = mha.c_hidden
        node = cls(c_in, c_hidden, no_heads, starting)
        # setattr(node, "mha", mha)
        node.mha = mha

        biases_path = [
            None,
            f"{layer_path}.layer_norm.bias",
        ]
        layers = [
            node.linear,
            node.layer_norm,
        ]
        for weights_path, bias_path, layer in zip(weights_path, biases_path,
                                                  layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return node

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if x.dtype != mask.dtype:
            x = x.to(mask.dtype)
        if not self.starting:
            x = x.transpose(-2, -3)
        # [B, I, J, C_in]

        x = self.layer_norm(x)
        # [B, H, I, J]
        lx = self.linear(x)
        if lx.dim() == 4:
            triangle_bias = torch.permute(
                lx, (0, 3, 1, 2))  # TA.permute_final_dims(lx, (2, 0, 1))
        elif lx.dim() == 3:
            triangle_bias = torch.permute(
                lx,
                (2, 0, 1))  # TA.permute_final_dims(lx, (2, 0, 1)), [*, H, I, J]

        mask_bias: Optional[torch.Tensor] = None
        if mask is not None:
            if not self.starting:
                mask = mask.transpose(-1, -2)
            # [*, I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
            biases = [mask_bias, triangle_bias]
        else:
            biases = [triangle_bias]
        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


class RefTransition(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/transition.py#L8 """

    def __init__(self,
                 dim: int = 128,
                 hidden: int = 512,
                 out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.out_dim = out_dim
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.transition_s",
            state_dict: Optional[dict] = None) -> 'RefTransition':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.norm.weight", f"{layer_path}.norm.bias"),
            (f"{layer_path}.fc1.weight", None),
            (f"{layer_path}.fc2.weight", None),
            (f"{layer_path}.fc3.weight", None),
        ]
        dim = state_dict[weights_biases_path[0][0]].shape[0]
        out_dim = state_dict[weights_biases_path[3][0]].shape[0]
        hidden = state_dict[weights_biases_path[1][0]].shape[0]
        m = cls(dim, hidden, out_dim)
        layers = [
            m.norm,
            m.fc1,
            m.fc2,
            m.fc3,
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)
        x = self.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class RefPairformerLayer(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/trunk.py#L535 """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.token_s = token_s
        self.num_heads = num_heads
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        if not self.no_update_s:
            self.attention = RefPairwiseSelfAttention(token_s, token_z,
                                                      num_heads)
        self.tri_mul_out = RefTriangleMultiplicationNode(token_z, outgoing=True)
        self.tri_mul_in = RefTriangleMultiplicationNode(token_z, outgoing=False)
        self.tri_attn_start = RefTriangleAttentionNode(token_z,
                                                       pairwise_head_width,
                                                       pairwise_num_heads,
                                                       inf=1e9,
                                                       starting=True)
        self.tri_attn_end = RefTriangleAttentionNode(token_z,
                                                     pairwise_head_width,
                                                     pairwise_num_heads,
                                                     inf=1e9,
                                                     starting=False)
        if not self.no_update_s:
            self.transition_s = RefTransition(token_s, token_s * 4)
        self.transition_z = RefTransition(token_z, token_z * 4)

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-1",
                     layer_path: str = "pairformer_module.layers.0",
                     state_dict: Optional[dict] = None) -> 'RefPairformerLayer':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        m = cls(128, 128)  # fake token_s and token_z
        submodules = [(
            RefPairwiseSelfAttention,
            layer_path + ".attention",
        ), (
            RefTriangleMultiplicationNode,
            layer_path + ".tri_mul_out",
        ), (
            RefTriangleMultiplicationNode,
            layer_path + ".tri_mul_in",
        ), (
            RefTriangleAttentionNode,
            layer_path + ".tri_att_start",
        ), (
            RefTriangleAttentionNode,
            layer_path + ".tri_att_end",
        ), (
            RefTransition,
            layer_path + ".transition_s",
        ), (
            RefTransition,
            layer_path + ".transition_z",
        )]
        for subm_cls, subm_path in submodules:
            subm = subm_cls.load_weights(state_dict=state_dict,
                                         layer_path=subm_path)
            if "tri_mul_out" in subm_path:
                subm.outgoing = True
                m.tri_mul_out = subm
            elif "tri_mul_in" in subm_path:
                subm.outgoing = False
                m.tri_mul_in = subm
            elif "tri_att_start" in subm_path:
                subm.starting = True
                m.tri_attn_start = subm
            elif "tri_att_end" in subm_path:
                subm.starting = False
                m.tri_attn_end = subm
            elif "transition_s" in subm_path:
                m.transition_s = subm
            elif "transition_z" in subm_path:
                m.transition_z = subm
            elif "attention" in subm_path:
                m.attention = subm
            base_path = subm_path.split(".")[-1]
            if base_path == "attention":
                m.token_z = subm.c_z
                m.token_s = subm.c_s
                m.num_heads = subm.num_heads
            if base_path == "tri_att_start":
                m.pairwise_num_heads = subm.num_heads
                m.pairwise_head_width = subm.c_hidden

        return m

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        if z.dtype != pair_mask.dtype:
            z = z.to(pair_mask.dtype)
        z = z + self.tri_attn_start(z, mask=pair_mask)
        z = z + self.tri_attn_end(z, mask=pair_mask)

        z = z + self.transition_z(z)

        # Compute sequence stack
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)

        return s, z


class RefAdaLN(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/transformers.py#L17 """

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.eps = eps

        self.a_norm = nn.LayerNorm(dim, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim, bias=True)
        self.s_bias = nn.Linear(dim_single_cond, dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = self.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path:
        str = "structure_module.score_model.token_transformer.layers.0.adaln",
            state_dict: Optional[dict] = None) -> 'RefAdaLN':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            # (f"{layer_path}.a_norm.weight", f"{layer_path}.a_norm.bias"),
            (f"{layer_path}.s_norm.weight", None),
            (f"{layer_path}.s_scale.weight", f"{layer_path}.s_scale.bias"),
            (f"{layer_path}.s_bias.weight", None),
        ]
        dim = state_dict[weights_biases_path[1][0]].shape[0]
        dim_single_cond = state_dict[weights_biases_path[1][0]].shape[1]
        m = cls(dim, dim_single_cond)
        layers = [
            # m.a_norm,
            m.s_norm,
            m.s_scale,
            m.s_bias,
        ]
        m.a_norm.weight.data.copy_(torch.ones(dim))

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefSwiGLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class RefConditionedTransitionBlock(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/transformers.py#L20 """

    def __init__(self,
                 dim_single: int,
                 dim_single_cond: int,
                 expansion_factor: int = 2) -> None:
        super().__init__()
        self.adaln = RefAdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = nn.Sequential(
            nn.Linear(dim_single, dim_inner * 2, bias=False),
            RefSwiGLU(),
        )
        self.a_to_b = nn.Linear(dim_single, dim_inner, bias=False)
        self.b_to_a = nn.Linear(dim_inner, dim_single, bias=False)

        self.output_projection = nn.Sequential(
            nn.Linear(dim_single_cond, dim_single, bias=True), nn.Sigmoid())

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)
        return a

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path:
        str = "structure_module.score_model.token_transformer.layers.0.transition",
            state_dict: Optional[dict] = None
    ) -> 'RefConditionedTransitionBlock':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        adaln = RefAdaLN.load_weights(state_dict=state_dict,
                                      layer_path=layer_path + ".adaln")
        m = cls(adaln.dim, adaln.dim_single_cond)
        setattr(m, "adaln", adaln)

        weights_biases_path = [
            (f"{layer_path}.swish_gate.0.weight", None),
            (f"{layer_path}.a_to_b.weight", None),
            (f"{layer_path}.b_to_a.weight", None),
            (f"{layer_path}.output_projection.0.weight",
             f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.swish_gate[0],
            m.a_to_b,
            m.b_to_a,
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefDiffusionTransformerLayer(nn.Module):

    def __init__(
        self,
        heads: int,
        dim: int = 384,
        dim_single_cond: Optional[int] = None,
        dim_pairwise: int = 128,
    ):
        super().__init__()
        dim_single_cond = dim_single_cond if dim_single_cond is not None else dim

        self.adaln = RefAdaLN(dim, dim_single_cond)

        self.pair_bias_attn = RefPairwiseSelfAttention(c_s=dim,
                                                       c_z=dim_pairwise,
                                                       num_heads=heads,
                                                       initial_norm=False)

        self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim),
                                               nn.Sigmoid())

        self.transition = RefConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        compute_pair_bias: bool = True,
        multiplicity: int = 1,
    ):
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b,
            z=bias,
            mask=mask,
            compute_pair_bias=compute_pair_bias,
        )
        b = self.output_projection(s) * b
        a = a + b
        a = a + self.transition(a, s)
        return a

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path:
        str = "structure_module.score_model.token_transformer.layers.0",
            state_dict: Optional[dict] = None
    ) -> 'RefDiffusionTransformerLayer':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        adaln = RefAdaLN.load_weights(state_dict=state_dict,
                                      layer_path=layer_path + ".adaln")
        pair_bias_attn = RefPairwiseSelfAttention.load_weights(
            state_dict=state_dict, layer_path=layer_path + ".pair_bias_attn")
        transition = RefConditionedTransitionBlock.load_weights(
            state_dict=state_dict, layer_path=layer_path + ".transition")
        m = cls(heads=pair_bias_attn.num_heads,
                dim=adaln.dim,
                dim_single_cond=adaln.dim_single_cond,
                dim_pairwise=pair_bias_attn.c_z)
        setattr(m, "adaln", adaln)
        setattr(m, "pair_bias_attn", pair_bias_attn)
        setattr(m, "transition", transition)

        weights_biases_path = [
            (f"{layer_path}.output_projection.0.weight",
             f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefPairformerNoSeqLayer(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/pairformer.py#L206 """

    def __init__(
        self,
        token_z: int,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads

        self.tri_mul_out = RefTriangleMultiplicationNode(token_z, outgoing=True)
        self.tri_mul_in = RefTriangleMultiplicationNode(token_z, outgoing=False)
        self.tri_attn_start = RefTriangleAttentionNode(token_z,
                                                       pairwise_head_width,
                                                       pairwise_num_heads,
                                                       inf=1e9,
                                                       starting=True)
        self.tri_attn_end = RefTriangleAttentionNode(token_z,
                                                     pairwise_head_width,
                                                     pairwise_num_heads,
                                                     inf=1e9,
                                                     starting=False)
        self.transition_z = RefTransition(token_z, token_z * 4)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-2-affinity",
            layer_path: str = "affinity_module1.pairformer_stack.layers.0",
            state_dict: Optional[dict] = None) -> 'RefPairformerNoSeqLayer':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        m = cls(128, 128)  # fake token_s and token_z
        submodules = [(
            RefTriangleMultiplicationNode,
            layer_path + ".tri_mul_out",
        ), (
            RefTriangleMultiplicationNode,
            layer_path + ".tri_mul_in",
        ), (
            RefTriangleAttentionNode,
            layer_path + ".tri_att_start",
        ), (
            RefTriangleAttentionNode,
            layer_path + ".tri_att_end",
        ), (
            RefTransition,
            layer_path + ".transition_z",
        )]
        for subm_cls, subm_path in submodules:
            subm = subm_cls.load_weights(state_dict=state_dict,
                                         layer_path=subm_path)
            if "tri_mul_out" in subm_path:
                subm.outgoing = True
                m.tri_mul_out = subm
            elif "tri_mul_in" in subm_path:
                subm.outgoing = False
                m.tri_mul_in = subm
            elif "tri_att_start" in subm_path:
                subm.starting = True
                m.tri_attn_start = subm
            elif "tri_att_end" in subm_path:
                subm.starting = False
                m.tri_attn_end = subm
            elif "transition_z" in subm_path:
                m.transition_z = subm
            base_path = subm_path.split(".")[-1]
            if base_path == "tri_att_start":
                m.pairwise_num_heads = subm.num_heads
                m.pairwise_head_width = subm.c_hidden

        return m

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        if z.dtype != pair_mask.dtype:
            z = z.to(pair_mask.dtype)
        z = z + self.tri_attn_start(z, mask=pair_mask)
        z = z + self.tri_attn_end(z, mask=pair_mask)

        z = z + self.transition_z(z)
        return z


class RefPairformerNoSeqModule(nn.Module):

    def __init__(self,
                 num_blocks: int = 8,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.layers = nn.ModuleList([
            RefPairformerNoSeqLayer(token_z, pairwise_head_width,
                                    pairwise_num_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, pair_mask)
        return z

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-2-affinity",
            layer_path: str = "affinity_module1.pairformer_stack",
            state_dict: Optional[dict] = None) -> 'RefPairformerNoSeqModule':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        all_keys = len([
            k for k in state_dict.keys()
            if k.startswith(f"{layer_path}.layers.")
        ])
        keys_layer_0 = [
            k for k in state_dict.keys()
            if k.startswith(f"{layer_path}.layers.0.")
        ]
        num_blocks = all_keys // len(keys_layer_0)
        m = cls(num_blocks=num_blocks,
                token_z=128)  # fake token_z and pairwise_head_width
        for i in range(m.num_blocks):
            layer_path_i = f"{layer_path}.layers.{i}"
            m.layers[i] = RefPairformerNoSeqLayer.load_weights(
                state_dict=state_dict, layer_path=layer_path_i)
        return m


class RefPairwiseConditioning(nn.Module):

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()
        self.token_z = token_z
        self.dim_token_rel_pos_feats = dim_token_rel_pos_feats
        self.num_transitions = num_transitions
        self.transition_expansion_factor = transition_expansion_factor

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            nn.Linear(token_z + dim_token_rel_pos_feats, token_z, bias=False),
        )

        transitions = nn.ModuleList([])
        for _ in range(num_transitions):
            transition = RefTransition(dim=token_z,
                                       hidden=transition_expansion_factor *
                                       token_z)
            transitions.append(transition)

        self.transitions = transitions

    def forward(
            self,
            z_trunk,  # Float['b n n tz'],
            token_rel_pos_feats,  # Float['b n n 3'],
    ):  # -> Float['b n n tz']:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)
        for transition in self.transitions:
            z = transition(z) + z

        return z

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-2-affinity",
            layer_path: str = "affinity_module1.pairwise_conditioner",
            state_dict: Optional[dict] = None) -> 'RefPairwiseConditioning':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.dim_pairwise_init_proj.0.weight",
             f"{layer_path}.dim_pairwise_init_proj.0.bias"),
            (f"{layer_path}.dim_pairwise_init_proj.1.weight", None),
        ]
        token_z = state_dict[
            f"{layer_path}.dim_pairwise_init_proj.1.weight"].shape[0]
        dim_token_rel_pos_feats = state_dict[
            f"{layer_path}.dim_pairwise_init_proj.1.weight"].shape[1] - token_z
        m = cls(token_z=token_z,
                dim_token_rel_pos_feats=dim_token_rel_pos_feats)
        layers = [
            m.dim_pairwise_init_proj[0],
            m.dim_pairwise_init_proj[1],
        ]

        for i in range(m.num_transitions):
            m.transitions[i] = RefTransition.load_weights(
                state_dict=state_dict,
                model=model,
                layer_path=f"{layer_path}.transitions.{i}")

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])

        return m


class RefAffinityHeadsTransformer(nn.Module):

    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks: int = None,
        num_heads: int = None,
        use_cross_transformer: bool = False,
        groups: dict = {},
    ):
        """ Reference affinity heads transformer: https://github.com/jwohlwend/boltz/blob/v2.1.1/src/boltz/model/modules/affinity.py """
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
        )

        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )

        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        z,
        cross_pair_mask,
        multiplicity=1,
    ):
        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7)
        g = self.affinity_out_mlp(g)

        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)

        affinity_logits_binary = self.to_affinity_logits_binary(
            affinity_pred_score).reshape(-1, 1)

        return affinity_pred_value, affinity_logits_binary

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-2-affinity",
            layer_path: str = "affinity_module1.affinity_heads",
            state_dict: Optional[dict] = None) -> 'RefAffinityHeadsTransformer':
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.affinity_out_mlp.0.weight",
             f"{layer_path}.affinity_out_mlp.0.bias"),
            (f"{layer_path}.affinity_out_mlp.2.weight",
             f"{layer_path}.affinity_out_mlp.2.bias"),
            (f"{layer_path}.to_affinity_pred_value.0.weight",
             f"{layer_path}.to_affinity_pred_value.0.bias"),
            (f"{layer_path}.to_affinity_pred_value.2.weight",
             f"{layer_path}.to_affinity_pred_value.2.bias"),
            (f"{layer_path}.to_affinity_pred_value.4.weight",
             f"{layer_path}.to_affinity_pred_value.4.bias"),
            (f"{layer_path}.to_affinity_pred_score.0.weight",
             f"{layer_path}.to_affinity_pred_score.0.bias"),
            (f"{layer_path}.to_affinity_pred_score.2.weight",
             f"{layer_path}.to_affinity_pred_score.2.bias"),
            (f"{layer_path}.to_affinity_pred_score.4.weight",
             f"{layer_path}.to_affinity_pred_score.4.bias"),
            (f"{layer_path}.to_affinity_logits_binary.weight",
             f"{layer_path}.to_affinity_logits_binary.bias"),
        ]
        token_z = state_dict[f"{layer_path}.affinity_out_mlp.0.weight"].shape[0]
        input_token_s = state_dict[
            f"{layer_path}.to_affinity_pred_value.0.weight"].shape[0]
        m = cls(token_z=token_z, input_token_s=input_token_s)
        layers = [
            m.affinity_out_mlp[0],
            m.affinity_out_mlp[2],
            m.to_affinity_pred_value[0],
            m.to_affinity_pred_value[2],
            m.to_affinity_pred_value[4],
            m.to_affinity_pred_score[0],
            m.to_affinity_pred_score[2],
            m.to_affinity_pred_score[4],
            m.to_affinity_logits_binary,
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefAffinityModule(nn.Module):
    """ Reference affinity module: https://github.com/jwohlwend/boltz/blob/v2.1.1/src/boltz/model/modules/affinity.py
    This affinity module does not support multiplicity > 1. And does not compute masks from feats.
    """

    def __init__(self,
                 token_s: int = 384,
                 token_z: int = 128,
                 num_dist_bins: int = 64,
                 pairformer_num_blocks: int = 8,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4):
        super().__init__()
        self.token_s = token_s
        self.token_z = token_z
        self.num_dist_bins = num_dist_bins
        self.pairformer_num_blocks = pairformer_num_blocks
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads

        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        self.s_to_z_prod_in1 = nn.Linear(token_s, token_z, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(token_s, token_z, bias=False)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = nn.Linear(token_z, token_z, bias=False)

        self.pairwise_conditioner = RefPairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )
        self.pairformer_stack = RefPairformerNoSeqModule(
            num_blocks=pairformer_num_blocks,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )
        self.affinity_heads = RefAffinityHeadsTransformer(
            token_z=token_z,
            input_token_s=token_s,
        )

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-2-affinity",
                     layer_path: str = "affinity_module1",
                     state_dict: Optional[dict] = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.dist_bin_pairwise_embed.weight", None),
            (f"{layer_path}.s_to_z_prod_in1.weight", None),
            (f"{layer_path}.s_to_z_prod_in2.weight", None),
            (f"{layer_path}.z_norm.weight", f"{layer_path}.z_norm.bias"),
            (f"{layer_path}.z_linear.weight", None),
        ]
        num_dist_bins = state_dict[
            f"{layer_path}.dist_bin_pairwise_embed.weight"].shape[0]
        token_z = state_dict[
            f"{layer_path}.dist_bin_pairwise_embed.weight"].shape[1]
        token_s = state_dict[f"{layer_path}.s_to_z_prod_in1.weight"].shape[1]

        m = cls(token_s=token_s, token_z=token_z, num_dist_bins=num_dist_bins)
        layers = [
            m.dist_bin_pairwise_embed,
            m.s_to_z_prod_in1,
            m.s_to_z_prod_in2,
            m.z_norm,
            m.z_linear,
        ]
        m.pairwise_conditioner = RefPairwiseConditioning.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.pairwise_conditioner")
        m.pairformer_stack = RefPairformerNoSeqModule.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.pairformer_stack")
        m.affinity_heads = RefAffinityHeadsTransformer.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.affinity_heads")

        m.pairformer_num_blocks = m.pairformer_stack.num_blocks
        m.pairwise_head_width = m.pairformer_stack.pairwise_head_width
        m.pairwise_num_heads = m.pairformer_stack.pairwise_num_heads

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self,
                s,
                z,
                distogram,
                cross_pair_mask_0,
                cross_pair_mask_1,
                multiplicity=1):
        # TODO: support multiplicity > 1
        z = self.z_linear(self.z_norm(z))

        z = (z + self.s_to_z_prod_in1(s)[:, :, None, :] +
             self.s_to_z_prod_in2(s)[:, None, :, :])
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z,
                                          token_rel_pos_feats=distogram)

        z = self.pairformer_stack(z, pair_mask=cross_pair_mask_0)

        affinity_pred_value, affinity_logits_binary = self.affinity_heads(
            z=z,
            cross_pair_mask=cross_pair_mask_1,
            multiplicity=multiplicity,
        )
        return affinity_pred_value, affinity_logits_binary


class RefPairWeightedAveraging(nn.Module):
    """ Reference pair weighted averaging: https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/model/layers/pair_averaging.py """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_h: int,
                 num_heads: int,
                 inf: float = 1e9):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-2",
            layer_path: str = "msa_module.layers.0.pair_weighted_averaging",
            state_dict: Optional[dict] = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.norm_m.weight", f"{layer_path}.norm_m.bias"),
            (f"{layer_path}.norm_z.weight", f"{layer_path}.norm_z.bias"),
            (f"{layer_path}.proj_m.weight", None),
            (f"{layer_path}.proj_g.weight", None),
            (f"{layer_path}.proj_z.weight", None),
            (f"{layer_path}.proj_o.weight", None),
        ]
        c_m = state_dict[f"{layer_path}.norm_m.weight"].shape[0]
        c_z = state_dict[f"{layer_path}.norm_z.weight"].shape[0]
        c_h_times_num_heads = state_dict[f"{layer_path}.proj_m.weight"].shape[0]
        num_heads = state_dict[f"{layer_path}.proj_z.weight"].shape[0]
        c_h = c_h_times_num_heads // num_heads
        m = cls(c_m=c_m, c_z=c_z, c_h=c_h, num_heads=num_heads)
        layers = [
            m.norm_m,
            m.norm_z,
            m.proj_m,
            m.proj_g,
            m.proj_z,
            m.proj_o,
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, m: torch.Tensor, z: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)

        # Project input tensors
        v = self.proj_m(m)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
        v = v.permute(0, 3, 1, 2, 4)

        # Compute weights
        b = self.proj_z(z)
        b = b.permute(0, 3, 1, 2)
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)
        # Compute gating
        g = self.proj_g(m)
        g = g.sigmoid()

        # Compute output
        o = torch.einsum("bhij,bhsjd->bhsid", w.to(v.dtype), v)
        o = o.permute(0, 2, 3, 1, 4)
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        o = self.proj_o(g * o)
        return o


class RefOuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        norm_mask_by_eps: bool = False,
        norm_before_output: bool = True,
        bias_flags: dict[str, bool] = {
            "proj_a": False,
            "proj_b": False,
            "proj_o": True
        }
    ) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.c_in = c_in
        self.norm_mask_by_eps = norm_mask_by_eps
        self.mask_eps = 1e-3
        self.norm_before_output = norm_before_output
        self.bias_flags = bias_flags

        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=bias_flags["proj_a"])
        self.proj_b = nn.Linear(c_in, c_hidden, bias=bias_flags["proj_b"])
        self.proj_o = nn.Linear(c_hidden * c_hidden,
                                c_out,
                                bias=bias_flags["proj_o"])

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-2",
                     layer_path: str = "msa_module.layers.0.outer_product_mean",
                     state_dict: Optional[dict] = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.norm.weight", f"{layer_path}.norm.bias"),
            (f"{layer_path}.proj_a.weight", None),
            (f"{layer_path}.proj_b.weight", None),
            (f"{layer_path}.proj_o.weight", f"{layer_path}.proj_o.bias"),
        ]
        c_in = state_dict[f"{layer_path}.norm.weight"].shape[0]
        c_hidden = state_dict[f"{layer_path}.proj_a.weight"].shape[0]
        c_out = state_dict[f"{layer_path}.proj_o.weight"].shape[0]
        m = cls(c_in=c_in,
                c_hidden=c_hidden,
                c_out=c_out,
                norm_before_output=True)
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

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(m)

        # Compute projections
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        mask = mask[:, :, None, :] * mask[:, :, :, None]
        if self.norm_mask_by_eps:
            # This for OF family models
            num_mask = mask.sum(1) + self.mask_eps
        else:
            # This for Boltz family models
            num_mask = mask.sum(1).clamp(min=1)
        z = torch.einsum("...sic,...sjd->...ijcd", a.float(), b.float())
        z = z.reshape(z.shape[:-2] + (-1, ))
        if self.norm_before_output:
            z = z / num_mask
        # Project to output
        z = self.proj_o(z.to(m))
        if not self.norm_before_output:
            z = z / num_mask
        return z


class RefMSALayer(nn.Module):

    def __init__(self,
                 msa_s: int,
                 token_z: int,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4) -> None:
        super().__init__()
        self.msa_s = msa_s
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads

        self.msa_transition = RefTransition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = RefPairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )
        self.pairformer_layer = RefPairformerNoSeqLayer(
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )
        self.outer_product_mean = RefOuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-2",
                     layer_path: str = "msa_module.layers.0",
                     state_dict: Optional[dict] = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        msa_transition = RefTransition.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.msa_transition")
        pair_weighted_averaging = RefPairWeightedAveraging.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.pair_weighted_averaging")
        pairformer_layer = RefPairformerNoSeqLayer.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.pairformer_layer")
        outer_product_mean = RefOuterProductMean.load_weights(
            state_dict=state_dict,
            model=model,
            layer_path=f"{layer_path}.outer_product_mean")

        m = cls(
            msa_s=msa_transition.dim,
            token_z=pair_weighted_averaging.c_z,
            pairwise_head_width=pairformer_layer.pairwise_head_width,
            pairwise_num_heads=pairformer_layer.pairwise_num_heads,
        )
        setattr(m, "msa_transition", msa_transition)
        setattr(m, "pair_weighted_averaging", pair_weighted_averaging)
        setattr(m, "pairformer_layer", pairformer_layer)
        setattr(m, "outer_product_mean", outer_product_mean)
        return m

    def forward(
        self,
        z: torch.Tensor,
        m: torch.Tensor,
        token_mask: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = m + self.pair_weighted_averaging(m, z, token_mask)
        m = m + self.msa_transition(m)

        z = z + self.outer_product_mean(m, msa_mask)
        # Compute pairwise stack
        z = self.pairformer_layer(z, token_mask)

        return z, m


class RefMSAModule(nn.Module):
    """ Reference MSA module: https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/model/modules/trunkv2.py """

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        token_s: int,
        msa_blocks: int,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        use_paired_feature: bool = True,
        num_tokens: int = 33,
        **kwargs,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.msa_s = msa_s
        self.token_z = token_z
        self.token_s = token_s
        self.msa_blocks = msa_blocks
        self.use_paired_feature = use_paired_feature
        self.num_tokens = num_tokens
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads

        self.s_proj = nn.Linear(token_s, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            num_tokens + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for i in range(msa_blocks):
            self.layers.append(
                RefMSALayer(
                    msa_s,
                    token_z,
                    pairwise_head_width,
                    pairwise_num_heads,
                ))

    @classmethod
    def load_weights(cls,
                     model: str = "boltz-2",
                     layer_path: str = "msa_module",
                     state_dict: Optional[dict] = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        s_proj_weights = state_dict[f"{layer_path}.s_proj.weight"]
        msa_proj_weights = state_dict[f"{layer_path}.msa_proj.weight"]

        token_s = s_proj_weights.shape[1]
        msa_s = s_proj_weights.shape[0]

        all_keys = len([
            k for k in state_dict.keys()
            if k.startswith(f"{layer_path}.layers.")
        ])
        keys_layer_0 = [
            k for k in state_dict.keys()
            if k.startswith(f"{layer_path}.layers.0.")
        ]
        msa_blocks = all_keys // len(keys_layer_0)
        msa_layers = []
        for i in range(msa_blocks):
            msa_layers.append(
                RefMSALayer.load_weights(state_dict=state_dict,
                                         model=model,
                                         layer_path=f"{layer_path}.layers.{i}"))
        token_z = msa_layers[0].token_z
        pairwise_head_width = msa_layers[0].pairwise_head_width
        pairwise_num_heads = msa_layers[0].pairwise_num_heads

        num_tokens = 33
        if msa_proj_weights.shape[1] > num_tokens + 2:
            use_paired_feature = True
        else:
            use_paired_feature = False

        m = cls(
            msa_s=msa_s,
            token_z=token_z,
            token_s=token_s,
            msa_blocks=msa_blocks,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            use_paired_feature=use_paired_feature,
            num_tokens=num_tokens,
        )
        m.s_proj.weight.data.copy_(s_proj_weights)
        m.msa_proj.weight.data.copy_(msa_proj_weights)
        m.layers = nn.ModuleList(msa_layers)
        return m

    def forward(self, z: torch.Tensor, emb: torch.Tensor, msa: torch.Tensor,
                has_deletion: torch.Tensor, deletion_value: torch.Tensor,
                msa_paired: torch.Tensor, msa_mask: torch.Tensor,
                token_pad_mask: torch.Tensor) -> torch.Tensor:

        msa = torch.nn.functional.one_hot(msa, num_classes=self.num_tokens)
        has_deletion = has_deletion.unsqueeze(-1)
        deletion_value = deletion_value.unsqueeze(-1)
        is_paired = msa_paired.unsqueeze(-1)
        token_mask = token_pad_mask.float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired],
                          dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        for i in range(self.msa_blocks):
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
            )
        return z
