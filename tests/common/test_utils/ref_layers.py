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
from test_utils.ref_attn import RefPairwiseSelfAttention, RefTriangleAttention

from tensorrt_bionemo.hf.checkpoints import load_hf_weights


class RefTriangleMultiplicationNode(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_mult.py"""

    def __init__(self, dim: int = 128, outgoing: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.outgoing = outgoing
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim, eps=1e-5)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def skip_cast(self):
        self.norm_out = self.norm_out.float()
        self.p_out = self.p_out.float()
        self.g_out = self.g_out.float()

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
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in.float()).sigmoid()

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
            state_dict = load_hf_weights(model, local_files_only=False)
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
                 inf: float = 1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = nn.Linear(c_in, self.num_heads, bias=False)

        self.mha = RefTriangleAttention(self.c_in, self.c_in, self.c_in,
                                        self.c_hidden, self.num_heads)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            layer_path: str = "pairformer_module.layers.0.tri_att_start",
            no_heads: int = 4,
            starting: bool = True,
            state_dict: Optional[dict] = None) -> 'RefTriangleAttentionNode':
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)
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
            state_dict = load_hf_weights(model, local_files_only=False)
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
            state_dict = load_hf_weights(model, local_files_only=False)
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
