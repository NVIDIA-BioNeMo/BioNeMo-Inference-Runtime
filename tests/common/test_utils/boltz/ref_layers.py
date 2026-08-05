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

# Upstream OpenFold/Boltz reference implementation, mirrored for parity tests.
# Kept in upstream style (star imports, forward refs), not held to these rules.
# ruff: noqa: B006, B007, B020, B905
import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils.boltz.ref_attn import RefPairwiseSelfAttention, RefTriangleAttention

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import create_indexing_matrix, query_to_keys
from tensorrt_bionemo.hubs import load_weights


class RefTriangleMultiplicationNode(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_mult.py"""

    def __init__(
        self,
        dim: int = 128,
        outgoing: bool = True,
        bias_flags: dict[str, bool] = {
            "p_in": False,
            "g_in": False,
            "p_out": False,
            "g_out": False,
        },
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
        state_dict: dict | None = None,
    ) -> "RefTriangleMultiplicationNode":
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
        layers = [mul_node.norm_in, mul_node.p_in, mul_node.g_in, mul_node.norm_out, mul_node.p_out, mul_node.g_out]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return mul_node


class RefTriangleAttentionNode(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_attention/attention.py"""

    def __init__(
        self,
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
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf
        self.mha_bias_flags = mha_bias_flags

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = nn.Linear(c_in, self.num_heads, bias=False)

        self.mha = RefTriangleAttention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.num_heads, bias_flags=mha_bias_flags
        )

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-1",
        layer_path: str = "pairformer_module.layers.0.tri_att_start",
        no_heads: int = 4,
        starting: bool = True,
        state_dict: dict | None = None,
    ) -> "RefTriangleAttentionNode":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_path = [
            f"{layer_path}.linear.weight",
            f"{layer_path}.layer_norm.weight",
        ]
        no_heads = state_dict[weights_path[0]].shape[0]
        c_in = state_dict[weights_path[0]].shape[1]
        mha = RefTriangleAttention.load_weights(
            state_dict=state_dict, layer_path=layer_path + ".mha", no_heads=no_heads
        )
        c_hidden = mha.c_hidden
        node = cls(c_in, c_hidden, no_heads, starting)
        node.mha = mha

        biases_path = [
            None,
            f"{layer_path}.layer_norm.bias",
        ]
        layers = [
            node.linear,
            node.layer_norm,
        ]
        for weights_path, bias_path, layer in zip(weights_path, biases_path, layers):
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
            # (B, I, J, H) -> (B, H, I, J); then unsqueeze a "starting row"
            # broadcast dim so the bias is shaped (B, 1, H, I, J) and
            # broadcasts properly against attention scores of shape
            # (B, I_start, H, J, J) for any batch size. For B=1 this is
            # numerically identical to the original (B, H, I, J) tensor
            # whose leading "1" used to be prepended implicitly.
            triangle_bias = torch.permute(lx, (0, 3, 1, 2)).unsqueeze(1)  # (B, 1, H, I, J)
        elif lx.dim() == 3:
            triangle_bias = torch.permute(lx, (2, 0, 1))  # TA.permute_final_dims(lx, (2, 0, 1)), [*, H, I, J]

        mask_bias: torch.Tensor | None = None
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
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/transition.py#L8"""

    def __init__(self, dim: int = 128, hidden: int = 512, out_dim: int | None = None) -> None:
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
        state_dict: dict | None = None,
    ) -> "RefTransition":
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
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
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
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/trunk.py#L535"""

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
            self.attention = RefPairwiseSelfAttention(token_s, token_z, num_heads)
        self.tri_mul_out = RefTriangleMultiplicationNode(token_z, outgoing=True)
        self.tri_mul_in = RefTriangleMultiplicationNode(token_z, outgoing=False)
        self.tri_attn_start = RefTriangleAttentionNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9, starting=True
        )
        self.tri_attn_end = RefTriangleAttentionNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9, starting=False
        )
        if not self.no_update_s:
            self.transition_s = RefTransition(token_s, token_s * 4)
        self.transition_z = RefTransition(token_z, token_z * 4)

    @classmethod
    def load_weights(
        cls, model: str = "boltz-1", layer_path: str = "pairformer_module.layers.0", state_dict: dict | None = None
    ) -> "RefPairformerLayer":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        m = cls(128, 128)  # fake token_s and token_z
        submodules = [
            (
                RefPairwiseSelfAttention,
                layer_path + ".attention",
            ),
            (
                RefTriangleMultiplicationNode,
                layer_path + ".tri_mul_out",
            ),
            (
                RefTriangleMultiplicationNode,
                layer_path + ".tri_mul_in",
            ),
            (
                RefTriangleAttentionNode,
                layer_path + ".tri_att_start",
            ),
            (
                RefTriangleAttentionNode,
                layer_path + ".tri_att_end",
            ),
            (
                RefTransition,
                layer_path + ".transition_s",
            ),
            (
                RefTransition,
                layer_path + ".transition_z",
            ),
        ]
        for subm_cls, subm_path in submodules:
            subm = subm_cls.load_weights(state_dict=state_dict, layer_path=subm_path)
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

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
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
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/transformers.py#L17"""

    def __init__(self, dim: int, dim_single_cond: int, eps: float = 1e-5) -> None:
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
        layer_path: str = "structure_module.score_model.token_transformer.layers.0.adaln",
        state_dict: dict | None = None,
    ) -> "RefAdaLN":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.s_norm.weight", None),
            (f"{layer_path}.s_scale.weight", f"{layer_path}.s_scale.bias"),
            (f"{layer_path}.s_bias.weight", None),
        ]
        dim = state_dict[weights_biases_path[1][0]].shape[0]
        dim_single_cond = state_dict[weights_biases_path[1][0]].shape[1]
        m = cls(dim, dim_single_cond)
        layers = [
            m.s_norm,
            m.s_scale,
            m.s_bias,
        ]
        m.a_norm.weight.data.copy_(torch.ones(dim))

        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class RefConditionedTransitionBlock(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/transformers.py#L20"""

    def __init__(self, dim_single: int, dim_single_cond: int, expansion_factor: int = 2) -> None:
        super().__init__()
        self.adaln = RefAdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = nn.Sequential(
            nn.Linear(dim_single, dim_inner * 2, bias=False),
            RefSwiGLU(),
        )
        self.a_to_b = nn.Linear(dim_single, dim_inner, bias=False)
        self.b_to_a = nn.Linear(dim_inner, dim_single, bias=False)

        self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim_single, bias=True), nn.Sigmoid())

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)
        return a

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-1",
        layer_path: str = "structure_module.score_model.token_transformer.layers.0.transition",
        state_dict: dict | None = None,
    ) -> "RefConditionedTransitionBlock":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        adaln = RefAdaLN.load_weights(state_dict=state_dict, layer_path=layer_path + ".adaln")
        m = cls(adaln.dim, adaln.dim_single_cond)
        m.adaln = adaln

        weights_biases_path = [
            (f"{layer_path}.swish_gate.0.weight", None),
            (f"{layer_path}.a_to_b.weight", None),
            (f"{layer_path}.b_to_a.weight", None),
            (f"{layer_path}.output_projection.0.weight", f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.swish_gate[0],
            m.a_to_b,
            m.b_to_a,
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefDiffusionTransformerLayer(nn.Module):
    def __init__(
        self,
        heads: int = 4,
        dim: int = 384,
        dim_single_cond: int | None = None,
        dim_pairwise: int = 128,
    ):
        super().__init__()
        dim_single_cond = dim_single_cond if dim_single_cond is not None else dim

        self.adaln = RefAdaLN(dim, dim_single_cond)

        # dim_pairwise == 0 → precomputed pair bias, no projection to build.
        self.pair_bias_attn = RefPairwiseSelfAttention(
            c_s=dim, c_z=dim_pairwise, num_heads=heads, compute_pair_bias=dim_pairwise > 0, initial_norm=False
        )

        self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim), nn.Sigmoid())

        self.transition = RefConditionedTransitionBlock(dim_single=dim, dim_single_cond=dim_single_cond)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        compute_pair_bias: bool = True,
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ):
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b,
            z=bias,
            mask=mask,
            compute_pair_bias=compute_pair_bias,
            attn_metadata=attn_metadata,
        )
        b = self.output_projection(s) * b
        a = a + b
        a = a + self.transition(a, s)
        return a

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-1",
        layer_path: str = "structure_module.score_model.token_transformer.layers.0",
        state_dict: dict | None = None,
    ) -> "RefDiffusionTransformerLayer":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        adaln = RefAdaLN.load_weights(state_dict=state_dict, layer_path=layer_path + ".adaln")
        pair_bias_attn = RefPairwiseSelfAttention.load_weights(
            state_dict=state_dict, layer_path=layer_path + ".pair_bias_attn"
        )
        transition = RefConditionedTransitionBlock.load_weights(
            state_dict=state_dict, layer_path=layer_path + ".transition"
        )
        m = cls(
            heads=pair_bias_attn.num_heads,
            dim=adaln.dim,
            dim_single_cond=adaln.dim_single_cond,
            dim_pairwise=pair_bias_attn.c_z,
        )
        m.adaln = adaln
        m.pair_bias_attn = pair_bias_attn
        m.transition = transition

        weights_biases_path = [
            (f"{layer_path}.output_projection.0.weight", f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefPairformerNoSeqLayer(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/pairformer.py#L206"""

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
        self.tri_attn_start = RefTriangleAttentionNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9, starting=True
        )
        self.tri_attn_end = RefTriangleAttentionNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9, starting=False
        )
        self.transition_z = RefTransition(token_z, token_z * 4)

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2-affinity",
        layer_path: str = "affinity_module1.pairformer_stack.layers.0",
        state_dict: dict | None = None,
    ) -> "RefPairformerNoSeqLayer":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        m = cls(128, 128)  # fake token_s and token_z
        submodules = [
            (
                RefTriangleMultiplicationNode,
                layer_path + ".tri_mul_out",
            ),
            (
                RefTriangleMultiplicationNode,
                layer_path + ".tri_mul_in",
            ),
            (
                RefTriangleAttentionNode,
                layer_path + ".tri_att_start",
            ),
            (
                RefTriangleAttentionNode,
                layer_path + ".tri_att_end",
            ),
            (
                RefTransition,
                layer_path + ".transition_z",
            ),
        ]
        for subm_cls, subm_path in submodules:
            subm = subm_cls.load_weights(state_dict=state_dict, layer_path=subm_path)
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
    def __init__(
        self,
        num_blocks: int = 8,
        token_z: int = 128,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.layers = nn.ModuleList(
            [RefPairformerNoSeqLayer(token_z, pairwise_head_width, pairwise_num_heads) for _ in range(num_blocks)]
        )

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, pair_mask)
        return z

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2-affinity",
        layer_path: str = "affinity_module1.pairformer_stack",
        state_dict: dict | None = None,
    ) -> "RefPairformerNoSeqModule":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        all_keys = len([k for k in state_dict.keys() if k.startswith(f"{layer_path}.layers.")])
        keys_layer_0 = [k for k in state_dict.keys() if k.startswith(f"{layer_path}.layers.0.")]
        num_blocks = all_keys // len(keys_layer_0)
        m = cls(num_blocks=num_blocks, token_z=128)  # fake token_z and pairwise_head_width
        for i in range(m.num_blocks):
            layer_path_i = f"{layer_path}.layers.{i}"
            m.layers[i] = RefPairformerNoSeqLayer.load_weights(state_dict=state_dict, layer_path=layer_path_i)
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
            transition = RefTransition(dim=token_z, hidden=transition_expansion_factor * token_z)
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
        state_dict: dict | None = None,
    ) -> "RefPairwiseConditioning":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.dim_pairwise_init_proj.0.weight", f"{layer_path}.dim_pairwise_init_proj.0.bias"),
            (f"{layer_path}.dim_pairwise_init_proj.1.weight", None),
        ]
        token_z = state_dict[f"{layer_path}.dim_pairwise_init_proj.1.weight"].shape[0]
        dim_token_rel_pos_feats = state_dict[f"{layer_path}.dim_pairwise_init_proj.1.weight"].shape[1] - token_z
        m = cls(token_z=token_z, dim_token_rel_pos_feats=dim_token_rel_pos_feats)
        layers = [
            m.dim_pairwise_init_proj[0],
            m.dim_pairwise_init_proj[1],
        ]

        for i in range(m.num_transitions):
            m.transitions[i] = RefTransition.load_weights(
                state_dict=state_dict, model=model, layer_path=f"{layer_path}.transitions.{i}"
            )

        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
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
        """Reference affinity heads transformer: https://github.com/jwohlwend/boltz/blob/v2.1.1/src/boltz/model/modules/affinity.py"""
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
        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7)
        g = self.affinity_out_mlp(g)

        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)

        affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score).reshape(-1, 1)

        return affinity_pred_value, affinity_logits_binary

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2-affinity",
        layer_path: str = "affinity_module1.affinity_heads",
        state_dict: dict | None = None,
    ) -> "RefAffinityHeadsTransformer":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.affinity_out_mlp.0.weight", f"{layer_path}.affinity_out_mlp.0.bias"),
            (f"{layer_path}.affinity_out_mlp.2.weight", f"{layer_path}.affinity_out_mlp.2.bias"),
            (f"{layer_path}.to_affinity_pred_value.0.weight", f"{layer_path}.to_affinity_pred_value.0.bias"),
            (f"{layer_path}.to_affinity_pred_value.2.weight", f"{layer_path}.to_affinity_pred_value.2.bias"),
            (f"{layer_path}.to_affinity_pred_value.4.weight", f"{layer_path}.to_affinity_pred_value.4.bias"),
            (f"{layer_path}.to_affinity_pred_score.0.weight", f"{layer_path}.to_affinity_pred_score.0.bias"),
            (f"{layer_path}.to_affinity_pred_score.2.weight", f"{layer_path}.to_affinity_pred_score.2.bias"),
            (f"{layer_path}.to_affinity_pred_score.4.weight", f"{layer_path}.to_affinity_pred_score.4.bias"),
            (f"{layer_path}.to_affinity_logits_binary.weight", f"{layer_path}.to_affinity_logits_binary.bias"),
        ]
        token_z = state_dict[f"{layer_path}.affinity_out_mlp.0.weight"].shape[0]
        input_token_s = state_dict[f"{layer_path}.to_affinity_pred_value.0.weight"].shape[0]
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
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefAffinityModule(nn.Module):
    """Reference affinity module: https://github.com/jwohlwend/boltz/blob/v2.1.1/src/boltz/model/modules/affinity.py
    This affinity module does not support multiplicity > 1. And does not compute masks from feats.
    """

    def __init__(
        self,
        token_s: int = 384,
        token_z: int = 128,
        num_dist_bins: int = 64,
        pairformer_num_blocks: int = 8,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ):
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
    def load_weights(
        cls, model: str = "boltz-2-affinity", layer_path: str = "affinity_module1", state_dict: dict | None = None
    ):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.dist_bin_pairwise_embed.weight", None),
            (f"{layer_path}.s_to_z_prod_in1.weight", None),
            (f"{layer_path}.s_to_z_prod_in2.weight", None),
            (f"{layer_path}.z_norm.weight", f"{layer_path}.z_norm.bias"),
            (f"{layer_path}.z_linear.weight", None),
        ]
        num_dist_bins = state_dict[f"{layer_path}.dist_bin_pairwise_embed.weight"].shape[0]
        token_z = state_dict[f"{layer_path}.dist_bin_pairwise_embed.weight"].shape[1]
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
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.pairwise_conditioner"
        )
        m.pairformer_stack = RefPairformerNoSeqModule.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.pairformer_stack"
        )
        m.affinity_heads = RefAffinityHeadsTransformer.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.affinity_heads"
        )

        m.pairformer_num_blocks = m.pairformer_stack.num_blocks
        m.pairwise_head_width = m.pairformer_stack.pairwise_head_width
        m.pairwise_num_heads = m.pairformer_stack.pairwise_num_heads

        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, s, z, distogram, cross_pair_mask_0, cross_pair_mask_1, multiplicity=1):
        # TODO: support multiplicity > 1
        z = self.z_linear(self.z_norm(z))

        z = z + self.s_to_z_prod_in1(s)[:, :, None, :] + self.s_to_z_prod_in2(s)[:, None, :, :]
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        z = self.pairformer_stack(z, pair_mask=cross_pair_mask_0)

        affinity_pred_value, affinity_logits_binary = self.affinity_heads(
            z=z,
            cross_pair_mask=cross_pair_mask_1,
            multiplicity=multiplicity,
        )
        return affinity_pred_value, affinity_logits_binary


class RefPairWeightedAveraging(nn.Module):
    """Reference pair weighted averaging: https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/model/layers/pair_averaging.py"""

    def __init__(self, c_m: int, c_z: int, c_h: int, num_heads: int, inf: float = 1e9):
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
        state_dict: dict | None = None,
    ):
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
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(self, m: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        bias_flags: dict[str, bool] = {"proj_a": False, "proj_b": False, "proj_o": True},
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
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out, bias=bias_flags["proj_o"])

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2",
        layer_path: str = "msa_module.layers.0.outer_product_mean",
        state_dict: dict | None = None,
    ):
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
        m = cls(c_in=c_in, c_hidden=c_hidden, c_out=c_out, norm_before_output=True)
        layers = [
            m.norm,
            m.proj_a,
            m.proj_b,
            m.proj_o,
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
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
        z = z.reshape(z.shape[:-2] + (-1,))
        if self.norm_before_output:
            z = z / num_mask
        # Project to output
        z = self.proj_o(z.to(m))
        if not self.norm_before_output:
            z = z / num_mask
        return z


class RefMSALayer(nn.Module):
    def __init__(self, msa_s: int, token_z: int, pairwise_head_width: int = 32, pairwise_num_heads: int = 4) -> None:
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
    def load_weights(
        cls, model: str = "boltz-2", layer_path: str = "msa_module.layers.0", state_dict: dict | None = None
    ):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        msa_transition = RefTransition.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.msa_transition"
        )
        pair_weighted_averaging = RefPairWeightedAveraging.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.pair_weighted_averaging"
        )
        pairformer_layer = RefPairformerNoSeqLayer.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.pairformer_layer"
        )
        outer_product_mean = RefOuterProductMean.load_weights(
            state_dict=state_dict, model=model, layer_path=f"{layer_path}.outer_product_mean"
        )

        m = cls(
            msa_s=msa_transition.dim,
            token_z=pair_weighted_averaging.c_z,
            pairwise_head_width=pairformer_layer.pairwise_head_width,
            pairwise_num_heads=pairformer_layer.pairwise_num_heads,
        )
        m.msa_transition = msa_transition
        m.pair_weighted_averaging = pair_weighted_averaging
        m.pairformer_layer = pairformer_layer
        m.outer_product_mean = outer_product_mean
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
    """Reference MSA module: https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/model/modules/trunkv2.py"""

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
                )
            )

    @classmethod
    def load_weights(cls, model: str = "boltz-2", layer_path: str = "msa_module", state_dict: dict | None = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        s_proj_weights = state_dict[f"{layer_path}.s_proj.weight"]
        msa_proj_weights = state_dict[f"{layer_path}.msa_proj.weight"]

        token_s = s_proj_weights.shape[1]
        msa_s = s_proj_weights.shape[0]

        all_keys = len([k for k in state_dict.keys() if k.startswith(f"{layer_path}.layers.")])
        keys_layer_0 = [k for k in state_dict.keys() if k.startswith(f"{layer_path}.layers.0.")]
        msa_blocks = all_keys // len(keys_layer_0)
        msa_layers = []
        for i in range(msa_blocks):
            msa_layers.append(
                RefMSALayer.load_weights(state_dict=state_dict, model=model, layer_path=f"{layer_path}.layers.{i}")
            )
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

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_paired: torch.Tensor,
        msa_mask: torch.Tensor,
        token_pad_mask: torch.Tensor,
    ) -> torch.Tensor:

        msa = torch.nn.functional.one_hot(msa, num_classes=self.num_tokens)
        has_deletion = has_deletion.unsqueeze(-1)
        deletion_value = deletion_value.unsqueeze(-1)
        is_paired = msa_paired.unsqueeze(-1)
        token_mask = token_pad_mask.float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
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


class RefAtomEmbedding(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/encodersv2.py#L245"""

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        structure_prediction=False,
        use_no_atom_char=False,
        use_atom_backbone_feat=False,
        use_residue_feats_atoms=False,
    ):
        """TODO: Implement for the structure prediction path"""
        super().__init__()
        self.version = "v2"
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.atom_feature_dim = atom_feature_dim
        self.structure_prediction = structure_prediction
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms

        self.embed_atom_features = nn.Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = nn.Linear(3, atom_z, bias=False)
        self.embed_atompair_ref_dist = nn.Linear(1, atom_z, bias=False)
        self.embed_atompair_mask = nn.Linear(1, atom_z, bias=False)

        self.structure_prediction = structure_prediction

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_s, atom_z, bias=False),
        )
        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_s, atom_z, bias=False),
        )

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
        )

    @classmethod
    def load_weights(
        cls, model: str = "boltz-2", layer_path: str = "input_embedder.atom_encoder", state_dict: dict | None = None
    ):
        """TODO: Implement for the structure prediction path"""
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        atom_feature_dim = state_dict[f"{layer_path}.embed_atom_features.weight"].shape[1]
        atom_s = state_dict[f"{layer_path}.embed_atom_features.weight"].shape[0]
        atom_z = state_dict[f"{layer_path}.embed_atompair_ref_pos.weight"].shape[0]
        token_s = None
        token_z = None
        atoms_per_window_queries = 32
        atoms_per_window_keys = 128

        weights_biases_path = [
            (f"{layer_path}.embed_atom_features.weight", f"{layer_path}.embed_atom_features.bias"),
            (f"{layer_path}.embed_atompair_ref_pos.weight", None),
            (f"{layer_path}.embed_atompair_ref_dist.weight", None),
            (f"{layer_path}.embed_atompair_mask.weight", None),
            (f"{layer_path}.c_to_p_trans_k.1.weight", None),
            (f"{layer_path}.c_to_p_trans_q.1.weight", None),
            (f"{layer_path}.p_mlp.1.weight", None),
            (f"{layer_path}.p_mlp.3.weight", None),
            (f"{layer_path}.p_mlp.5.weight", None),
        ]
        m = cls(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=False,
            use_atom_backbone_feat=False,
            use_residue_feats_atoms=False,
        )
        layers = [
            m.embed_atom_features,
            m.embed_atompair_ref_pos,
            m.embed_atompair_ref_dist,
            m.embed_atompair_mask,
            m.c_to_p_trans_k[1],
            m.c_to_p_trans_q[1],
            m.p_mlp[1],
            m.p_mlp[3],
            m.p_mlp[5],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m

    def forward(
        self,
        atom_to_token: torch.Tensor,
        ref_pos: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        ref_space_uid: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor | None = None,
        atom_backbone_feat: torch.Tensor | None = None,
        res_type: torch.Tensor | None = None,
        modified: torch.Tensor | None = None,
        mol_type: torch.Tensor | None = None,
    ):

        B, N, _ = ref_pos.shape
        atom_mask = atom_pad_mask.bool()  # Bool['b m'],

        atom_ref_pos = ref_pos  # Float['b m 3'],
        atom_uid = ref_space_uid  # Long['b m'],

        atom_feats = [
            atom_ref_pos,
            ref_charge.unsqueeze(-1),
            ref_element,
        ]
        if not self.use_no_atom_char:
            atom_feats.append(ref_atom_name_chars.reshape(B, N, 4 * 64))
        if self.use_atom_backbone_feat:
            atom_feats.append(atom_backbone_feat)
        if self.use_residue_feats_atoms:
            res_feats = torch.cat(
                [
                    res_type,
                    modified.unsqueeze(-1),
                    F.one_hot(mol_type, num_classes=4).float(),
                ],
                dim=-1,
            )
            atom_to_token = atom_to_token.float()
            atom_res_feats = torch.bmm(atom_to_token, res_feats)
            atom_feats.append(atom_res_feats)

        atom_feats = torch.cat(atom_feats, dim=-1)

        c = self.embed_atom_features(atom_feats)

        # note we are already creating the windows to make it more efficient
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = create_indexing_matrix(K, W, H, c.device)
        to_keys = partial(query_to_keys, keys_indexing_matrix=keys_indexing_matrix, W=W, H=H)

        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

        d = atom_ref_pos_keys - atom_ref_pos_queries  # Float['b k w h 3']
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)  # Float['b k w h 1']
        d_norm = 1 / (1 + d_norm)  # AF3 feeds in the reciprocal of the distance norm

        atom_mask_queries = atom_mask.view(B, K, W, 1)
        atom_mask_keys = to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
        atom_uid_queries = atom_uid.view(B, K, W, 1)
        atom_uid_keys = to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
        v = (
            (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys)).float().unsqueeze(-1)
        )  # Bool['b k w h 1']

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c
        p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)
        return q, c, p, to_keys


class BoltzRefDiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int = 3,
        heads: int = 4,
        dim: int = 384,
        dim_single_cond: int | None = None,
        dim_pairwise: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_blocks = num_blocks
        self.heads = heads
        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.dim_pairwise = dim_pairwise

        for i in range(self.num_blocks):
            layer = RefDiffusionTransformerLayer(
                heads=heads,
                dim=dim,
                dim_single_cond=dim_single_cond,
                dim_pairwise=dim_pairwise,
            )
            self.layers.append(layer)

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2",
        layer_path: str = "input_embedder.atom_attention_encoder",
        state_dict: dict | None = None,
    ) -> "BoltzRefDiffusionTransformer":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        search = layer_path + ".layers."

        num_blocks = 0
        for state_key in state_dict.keys():
            if search in state_key:
                layer_id = int(state_key.replace(search, "").split(".")[0])
                num_blocks = max(layer_id, num_blocks)

        num_blocks += 1
        layers = nn.ModuleList()
        for i in range(num_blocks):
            layer = RefDiffusionTransformerLayer.load_weights(model=model, layer_path=search + str(i))
            layers.append(layer)
        boltz_ref_diffusion_transformer = cls(
            num_blocks=num_blocks,
            heads=layers[0].pair_bias_attn.num_heads,
            dim=layers[0].adaln.dim,
            dim_single_cond=layers[0].adaln.dim_single_cond,
        )
        boltz_ref_diffusion_transformer.layers = layers
        return boltz_ref_diffusion_transformer

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor = None,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ):

        L = self.num_blocks
        N, M, D = z.shape[-3:]
        heads = D // L
        batch_dims = z.shape[:-3]
        z = z.view(*batch_dims, N, M, L, heads)  # [*, N, N, L, heads]
        z = torch.moveaxis(z, -1, -4)  # [*, heads, N, N, L]
        for i, layer in enumerate(self.layers):
            bias = z[..., i]
            a = layer(a=a, s=s, bias=bias, mask=mask, multiplicity=multiplicity, attn_metadata=attn_metadata)
        return a


class RefAtomTransformer(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/transformersv2.py#L211"""

    def __init__(self, attn_window_queries: int, attn_window_keys: int, diffusion_transformer: nn.Module = None):
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = diffusion_transformer

    @classmethod
    def load_weights(
        cls,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model.atom_attention_encoder",
        state_dict: dict | None = None,
    ):
        """TODO: Implement for the structure prediction path"""

        # Redundant instantiation is load-bearing: these tests seed once and
        # then draw inputs, so skipping this init shifts every later draw.
        diffusion_transformer = BoltzRefDiffusionTransformer().load_weights(
            model=model, layer_path=layer_path + ".diffusion_transformer"
        )
        atom_transformer = cls(
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            diffusion_transformer=diffusion_transformer,
        )
        return atom_transformer

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ):

        assert attn_metadata is not None, "Attention metadata is required for RefAtomTransformer"

        W = self.attn_window_queries
        H = self.attn_window_keys

        B, multiplicity, N, _ = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B, multiplicity, NW, W, -1))
        c = c.view((B, 1, NW, W, -1))  # expand dim 1 for broadcasting
        mask = mask.view(B, 1, NW, W)  # expand dim 1 for broadcasting

        # and repeat at the dim 1, this is different from the original implementation.
        bias = bias.view((B, 1, NW, W, H, -1))  # expand dim 1 for broadcasting
        a = self.diffusion_transformer(
            a=q, s=c, z=bias, mask=mask, multiplicity=multiplicity, attn_metadata=attn_metadata
        )

        a = a.view(B * multiplicity, N, -1)
        return a


class RefAtomAttentionEncoder(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/encodersv2.py#L414"""

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        diffusion_transformer_cls: Any = None,
        structure_prediction=True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_s = token_s
        self.atom_s = atom_s
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.structure_prediction = structure_prediction

        self.r_to_q_trans = nn.Linear(3, atom_s, bias=False)

        self.atom_encoder = RefAtomTransformer(
            attn_window_queries=atoms_per_window_queries, attn_window_keys=atoms_per_window_keys
        )

        self.atom_to_token_trans = nn.Sequential(
            nn.Linear(atom_s, 2 * token_s if structure_prediction else token_s, bias=False), nn.ReLU()
        )

    @classmethod
    def load_weights(
        cls,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        structure_prediction: bool = True,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model.atom_attention_encoder",
        state_dict: dict | None = None,
    ):
        """TODO: Implement for the structure prediction path"""

        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        atom_transformer = RefAtomTransformer.load_weights(
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            model=model,
            layer_path=layer_path + ".atom_encoder",
        )

        atom_to_token_trans_weights = state_dict[layer_path + ".atom_to_token_trans.0.weight"]
        if structure_prediction:
            token_s = atom_to_token_trans_weights.shape[0] // 2
        else:
            token_s = atom_to_token_trans_weights.shape[0]

        atom_s = atom_to_token_trans_weights.shape[1]

        atom_attention_encoder = cls(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=attn_window_queries,
            atoms_per_window_keys=attn_window_keys,
            structure_prediction=structure_prediction,
        )
        r_to_q_trans_weight = state_dict[layer_path + ".r_to_q_trans.weight"]
        atom_attention_encoder.r_to_q_trans.weight.data.copy_(r_to_q_trans_weight)
        atom_attention_encoder.atom_encoder = atom_transformer

        atom_attention_encoder.atom_to_token_trans[0].weight.data.copy_(atom_to_token_trans_weights)
        return atom_attention_encoder

    def forward(
        self,
        atom_to_token: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        atom_enc_bias: torch.Tensor,
        r=None,
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ):

        atom_mask = atom_pad_mask.bool()
        if self.structure_prediction:
            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

        q = q.unsqueeze(1)
        q = q.repeat_interleave(multiplicity, 1)  # [B, multiplicity, N_atoms, D]

        q = self.atom_encoder(
            q=q,
            c=c,
            mask=atom_mask,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            attn_metadata=attn_metadata,
        )

        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(q)
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        a = a.to(q)

        return a, q, c


class RefAtomAttentionDecoder(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/encodersv2.py#L495"""

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        diffusion_transformer_cls: Any = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.a_to_q_trans = nn.Linear(2 * token_s, atom_s, bias=False)
        self.token_s = token_s
        self.atom_s = atom_s
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.atom_decoder = RefAtomTransformer(
            attn_window_queries=atoms_per_window_queries, attn_window_keys=atoms_per_window_keys
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(nn.LayerNorm(atom_s), nn.Linear(atom_s, 3, bias=False))

    @classmethod
    def load_weights(
        cls,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model.atom_attention_decoder",
        state_dict: dict | None = None,
    ):
        """TODO: Implement for the structure prediction path"""

        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        atom_transformer = RefAtomTransformer.load_weights(
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            model=model,
            layer_path=layer_path + ".atom_decoder",
        )

        a_to_q_trans_weight = state_dict[layer_path + ".a_to_q_trans.weight"]
        atom_s = a_to_q_trans_weight.shape[0]
        token_s = a_to_q_trans_weight.shape[1] // 2

        atom_attention_decoder = cls(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=attn_window_queries,
            atoms_per_window_keys=attn_window_keys,
        )
        atom_attention_decoder.atom_decoder = atom_transformer

        atom_attention_decoder.a_to_q_trans.weight.data.copy_(a_to_q_trans_weight)

        atom_feat_to_atom_pos_update_norm_weight = state_dict[layer_path + ".atom_feat_to_atom_pos_update.0.weight"]
        atom_feat_to_atom_pos_update_norm_bias = state_dict[layer_path + ".atom_feat_to_atom_pos_update.0.bias"]
        atom_feat_to_atom_pos_update_linear_weight = state_dict[layer_path + ".atom_feat_to_atom_pos_update.1.weight"]

        atom_attention_decoder.atom_feat_to_atom_pos_update[0].weight.data.copy_(
            atom_feat_to_atom_pos_update_norm_weight
        )
        atom_attention_decoder.atom_feat_to_atom_pos_update[0].bias.data.copy_(atom_feat_to_atom_pos_update_norm_bias)
        atom_attention_decoder.atom_feat_to_atom_pos_update[1].weight.data.copy_(
            atom_feat_to_atom_pos_update_linear_weight
        )

        return atom_attention_decoder

    def forward(
        self,
        atom_to_token: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        atom_dec_bias: torch.Tensor,
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ):

        B = atom_to_token.shape[0]
        with torch.autocast("cuda", enabled=False):
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

            a_to_q = self.a_to_q_trans(a)
            a_to_q = torch.bmm(atom_to_token, a_to_q)
        q = q + a_to_q.to(q)
        atom_mask = atom_pad_mask.bool()
        N, H = q.shape[-2:]
        q = q.view(B, multiplicity, N, H)

        q = self.atom_decoder(
            q=q,
            c=c,
            mask=atom_mask,
            bias=atom_dec_bias,
            attn_metadata=attn_metadata,
        )
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update


class RefFourierEmbedding(nn.Module):
    """Fourier embedding layer."""

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Args:
            dim : int
                The dimension of the embeddings.
            dtype: torch.dtype
                The data type of the input features.
            mapping: Optional[Mapping]
                The mapping of the input features.
        """
        super().__init__()
        self.proj = nn.Linear(1, dim)

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model.single_conditioner.fourier_embed",
        state_dict: dict | None = None,
    ) -> "RefFourierEmbedding":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)
        proj_weight = state_dict[layer_path + ".proj.weight"]
        proj_bias = state_dict[layer_path + ".proj.bias"]
        fourier_embed = cls(dim=proj_weight.shape[0])
        fourier_embed.proj.weight.data.copy_(proj_weight)
        fourier_embed.proj.bias.data.copy_(proj_bias)
        return fourier_embed

    def forward(
        self,
        times,
    ):
        """
        Args:
            times: Input time tensor for diffusion process, shape (batch_size,)

        Returns:
            Fourier embedded time features with cosine encoding
        """
        times = times.unsqueeze(1)
        rand_proj = self.proj(times)
        return torch.cos(2 * math.pi * rand_proj)


class RefSingleConditioning(nn.Module):
    """Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/encodersv2.py#L123"""

    def __init__(
        self,
        token_s: int = 384,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
        disable_times: bool = False,
    ) -> None:
        super().__init__()
        self.disable_times = disable_times

        self.token_s = token_s
        self.dim_fourier = dim_fourier
        self.norm_single = nn.LayerNorm(2 * token_s, eps=eps)
        self.single_embed = nn.Linear(2 * token_s, 2 * token_s)
        if not self.disable_times:
            self.fourier_embed = RefFourierEmbedding(dim_fourier)
            self.norm_fourier = nn.LayerNorm(dim_fourier, eps=eps)
            self.fourier_to_single = nn.Linear(dim_fourier, 2 * token_s, bias=False)

        transitions = nn.ModuleList([])
        for _ in range(num_transitions):
            transition = RefTransition(dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s)
            transitions.append(transition)

        self.transitions = transitions

    @classmethod
    def load_weights(
        cls,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model.single_conditioner",
        state_dict: dict | None = None,
    ) -> "RefSingleConditioning":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        num_transitions = 0
        transition_layer_path = layer_path + ".transitions."
        for state_key in state_dict.keys():
            if transition_layer_path in state_key:
                num_layer = int(state_key.replace(transition_layer_path, "").split(".")[0])
                num_transitions = max(num_transitions, num_layer + 1)
        transitions = nn.ModuleList([])

        for i in range(num_transitions):
            transition = RefTransition.load_weights(
                model=model, layer_path=transition_layer_path + f"{i}", state_dict=state_dict
            )
            transitions.append(transition)

        fourier_embed = RefFourierEmbedding.load_weights(
            model=model, layer_path=layer_path + ".fourier_embed", state_dict=state_dict
        )
        token_s = state_dict["structure_module.score_model.single_conditioner.norm_single.weight"].shape[0] // 2
        norm_single_weight = state_dict["structure_module.score_model.single_conditioner.norm_single.weight"]
        norm_single_bias = state_dict["structure_module.score_model.single_conditioner.norm_single.bias"]
        norm_single = nn.LayerNorm(2 * token_s)
        norm_single.weight.data.copy_(norm_single_weight)
        norm_single.bias.data.copy_(norm_single_bias)

        dim_fourier = state_dict["structure_module.score_model.single_conditioner.norm_fourier.weight"].shape[0]
        norm_fourier_weight = state_dict["structure_module.score_model.single_conditioner.norm_fourier.weight"]
        norm_fourier_bias = state_dict["structure_module.score_model.single_conditioner.norm_fourier.bias"]
        norm_fourier = nn.LayerNorm(dim_fourier)
        norm_fourier.weight.data.copy_(norm_fourier_weight)
        norm_fourier.bias.data.copy_(norm_fourier_bias)

        fourier_to_single = nn.Linear(dim_fourier, 2 * token_s, bias=False)
        fourier_to_single.weight.data.copy_(
            state_dict["structure_module.score_model.single_conditioner.fourier_to_single.weight"]
        )

        single_embed_weight = state_dict["structure_module.score_model.single_conditioner.single_embed.weight"]
        single_embed_bias = state_dict["structure_module.score_model.single_conditioner.single_embed.bias"]
        single_embed = nn.Linear(2 * token_s, 2 * token_s)
        single_embed.weight.data.copy_(single_embed_weight)
        single_embed.bias.data.copy_(single_embed_bias)

        single_conditioner = cls(token_s=token_s, dim_fourier=dim_fourier, num_transitions=num_transitions)
        single_conditioner.norm_single = norm_single
        single_conditioner.single_embed = single_embed
        single_conditioner.fourier_embed = fourier_embed
        single_conditioner.norm_fourier = norm_fourier
        single_conditioner.fourier_to_single = fourier_to_single
        single_conditioner.transitions = transitions
        return single_conditioner

    def forward(
        self,
        times,
        s_trunk,
        s_inputs,
    ):
        """
        Args:
            times: [B]
            s_trunk: [B, N, token_s]
            s_inputs: [B, N, token_s]
        Returns:
            s: [B, N, 2*token_s]
            normed_fourier: [B, N, dim_fourier] (None if disable_times is True)
        """
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))
        normed_fourier = None

        if not self.disable_times:
            fourier_embed = self.fourier_embed(times)  # note: sigma rescaling done in diffusion module
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)
            s = fourier_to_single.unsqueeze(1) + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


class RefDiffusionModule(nn.Module):
    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        dim_fourier: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        token_transformer_depth: int,
        token_transformer_heads: int,
        atom_decoder_depth: int,
        atom_decoder_heads: int,
        conditioning_transition_layers: int,
    ):
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
            token_s=token_s, dim_fourier=dim_fourier, num_transitions=conditioning_transition_layers
        )

        self.atom_attention_encoder = RefAtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
        )

        self.atom_attention_decoder = RefAtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
        )

        self.s_to_a_linear = nn.Sequential(nn.LayerNorm(2 * token_s), nn.Linear(2 * token_s, 2 * token_s, bias=False))

        self.token_transformer = BoltzRefDiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            heads=token_transformer_heads,
            num_blocks=token_transformer_depth,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

    @classmethod
    def load_weights(
        cls,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        model: str = "boltz-2",
        layer_path: str = "structure_module.score_model",
        state_dict: dict | None = None,
    ) -> "RefDiffusionModule":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        single_conditioner = RefSingleConditioning.load_weights(
            model=model, layer_path=layer_path + ".single_conditioner"
        )

        atom_attention_encoder = RefAtomAttentionEncoder.load_weights(
            model=model,
            layer_path=layer_path + ".atom_attention_encoder",
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
        )

        atom_attention_decoder = RefAtomAttentionDecoder.load_weights(
            model=model,
            layer_path=layer_path + ".atom_attention_decoder",
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
        )

        token_transformer = BoltzRefDiffusionTransformer.load_weights(
            model=model, layer_path=layer_path + ".token_transformer"
        )

        s_to_a_linear_layer_norm_weight = state_dict[layer_path + ".s_to_a_linear.0.weight"]
        s_to_a_linear_layer_norm_bias = state_dict[layer_path + ".s_to_a_linear.0.bias"]

        s_to_a_linear_layer_linear_weight = state_dict[layer_path + ".s_to_a_linear.1.weight"]

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
            conditioning_transition_layers=conditioning_transition_layers,
        )
        diffusion_module.single_conditioner = single_conditioner
        diffusion_module.atom_attention_encoder = atom_attention_encoder
        diffusion_module.atom_attention_decoder = atom_attention_decoder
        diffusion_module.token_transformer = token_transformer
        diffusion_module.s_to_a_linear[0].weight.data.copy_(s_to_a_linear_layer_norm_weight)
        diffusion_module.s_to_a_linear[0].bias.data.copy_(s_to_a_linear_layer_norm_bias)
        diffusion_module.s_to_a_linear[1].weight.data.copy_(s_to_a_linear_layer_linear_weight)
        diffusion_module.a_norm.weight.data.copy_(a_norm_weight)
        diffusion_module.a_norm.bias.data.copy_(a_norm_bias)
        return diffusion_module

    def forward(
        self,
        atom_to_token,
        atom_pad_mask,
        token_pad_mask,
        s_inputs,
        s_trunk,
        r_noisy,
        times,
        diffusion_conditioning_q,
        diffusion_conditioning_c,
        diffusion_conditioning_atom_enc_bias,
        diffusion_conditioning_token_trans_bias,
        diffusion_conditioning_atom_dec_bias,
        multiplicity=1,
        attn_metadata=None,
    ):

        s, normed_fourier = self.single_conditioner(
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
            attn_metadata=attn_metadata,
        )

        a = a + self.s_to_a_linear(s)

        mask = token_pad_mask.repeat_interleave(multiplicity, 0)

        a = self.token_transformer(a=a, mask=mask, s=s, z=diffusion_conditioning_token_trans_bias)

        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning_atom_dec_bias,
            multiplicity=multiplicity,
            attn_metadata=attn_metadata,
        )

        return r_update


class RefTemplateV2Module(nn.Module):
    """Reference: boltz/model/modules/trunkv2.py::TemplateV2Module.

    The reference implementation deliberately mirrors the upstream module
    so test scripts can compare TRT-BNM's :class:`TemplateV2Module` against
    a plain PyTorch path. The only deviation from upstream is that the
    inner ``pairformer`` is built from :class:`RefPairformerNoSeqModule`.
    """

    def __init__(
        self,
        token_z: int = 128,
        template_dim: int = 64,
        template_blocks: int = 2,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        num_tokens: int = 33,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.template_dim = template_dim
        self.template_blocks = template_blocks
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.num_tokens = num_tokens

        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(num_tokens * 2 + num_bins + 5, template_dim, bias=False)
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        self.pairformer = RefPairformerNoSeqModule(
            num_blocks=template_blocks,
            token_z=template_dim,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )

    def forward(self, z: torch.Tensor, feats: dict[str, torch.Tensor], pair_mask: torch.Tensor) -> torch.Tensor:
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1).clamp(min=1)

        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]
        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        B, T = res_type.shape[:2]  # noqa: N806
        tmlp_pair_mask = (visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]).float()

        with torch.autocast(device_type="cuda", enabled=False):
            cb_dists = torch.cdist(cb_coords.float(), cb_coords.float())
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = F.one_hot(distogram, num_classes=self.num_bins).float()

            frame_rot_f = frame_rot.float().unsqueeze(2).transpose(-1, -2)
            frame_t_f = frame_t.float().unsqueeze(2).unsqueeze(-1)
            ca_coords_f = ca_coords.float().unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot_f, (ca_coords_f - frame_t_f))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

            a_tij = torch.cat([distogram, b_cb_mask.float(), unit_vector, b_frame_mask.float()], dim=-1)
            a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)

            res_type_f = res_type.float()
            res_i = res_type_f[:, :, :, None].expand(-1, -1, -1, res_type.size(2), -1)
            res_j = res_type_f[:, :, None, :].expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_i, res_j], dim=-1)
            # Upstream relies on an outer ``autocast(enabled=True)`` to recast
            # ``a_tij`` back to the model dtype on exit from the inner
            # ``autocast(enabled=False)`` block before this linear. Our tests
            # don't wrap the call in autocast, so cast explicitly to match the
            # projection's weight dtype.
            a_tij = self.a_proj(a_tij.to(self.a_proj.weight.dtype))

        N = z.shape[1]  # noqa: N806
        pair_mask_t = pair_mask[:, None].expand(-1, T, -1, -1).reshape(B * T, N, N)
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, N, N, self.template_dim)
        v = v + self.pairformer(v, pair_mask_t)
        v = self.v_norm(v)
        v = v.view(B, T, N, N, self.template_dim)

        # ``template_mask`` and ``num_templates`` are kept in fp32 to mirror
        # upstream. An outer ``autocast(enabled=True)`` would cast both back to
        # the model dtype; without autocast we cast explicitly so the
        # subsequent linear sees a dtype that matches its weight.
        template_mask = template_mask[:, :, None, None, None].to(v)
        num_templates = num_templates[:, None, None, None].to(v)
        u = (v * template_mask).sum(dim=1) / num_templates
        u = self.u_proj(self.relu(u))
        return u

    @classmethod
    def load_weights(
        cls, model: str = "boltz-2", layer_path: str = "template_module", state_dict: dict | None = None
    ) -> "RefTemplateV2Module":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        z_proj_w = state_dict[f"{layer_path}.z_proj.weight"]
        a_proj_w = state_dict[f"{layer_path}.a_proj.weight"]
        u_proj_w = state_dict[f"{layer_path}.u_proj.weight"]
        token_z = z_proj_w.shape[1]
        template_dim = z_proj_w.shape[0]

        pairformer_path = f"{layer_path}.pairformer"
        all_keys = len([k for k in state_dict.keys() if k.startswith(f"{pairformer_path}.layers.")])
        keys_layer_0 = [k for k in state_dict.keys() if k.startswith(f"{pairformer_path}.layers.0.")]
        num_blocks = all_keys // max(len(keys_layer_0), 1)

        m = cls(token_z=token_z, template_dim=template_dim, template_blocks=num_blocks)
        m.z_norm.weight.data.copy_(state_dict[f"{layer_path}.z_norm.weight"])
        m.z_norm.bias.data.copy_(state_dict[f"{layer_path}.z_norm.bias"])
        m.v_norm.weight.data.copy_(state_dict[f"{layer_path}.v_norm.weight"])
        m.v_norm.bias.data.copy_(state_dict[f"{layer_path}.v_norm.bias"])
        m.z_proj.weight.data.copy_(z_proj_w)
        m.a_proj.weight.data.copy_(a_proj_w)
        m.u_proj.weight.data.copy_(u_proj_w)
        m.pairformer = RefPairformerNoSeqModule.load_weights(state_dict=state_dict, layer_path=pairformer_path)
        return m
