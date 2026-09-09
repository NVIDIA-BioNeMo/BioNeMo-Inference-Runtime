# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Modified by NVIDIA Corporation and affiliates.

import torch
import torch.nn as nn
from einops import rearrange

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.layers.attention import CrossTriangleAttention
from bionemo_ir._torch.layers.transition import PairTransition
from bionemo_ir._torch.layers.transition import Transition as SwiGLUTransition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
)


class TemplatePairBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        tri_mul_first: bool,
        trimul_high_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        local_layer_idx: int = 0,
        eps: float = 1e-5,
        inf: float = 1e9,
        triangle_attn_backend: str = "VANILLA",
        skip_create_weights: bool = False,
        tri_attn_transposed_bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.tri_mul_first = tri_mul_first
        self.eps = eps
        self.inf = inf
        self.dtype = dtype

        transition_type = kwargs.get("transition_type", "relu")
        tri_mul_out_bias = (
            {"p_in": True, "g_in": True, "p_out": True, "g_out": True}
            if kwargs.get("tri_mul_out_bias", None) is None
            else kwargs.get("tri_mul_out_bias")
        )
        tri_mul_in_bias = (
            {"p_in": True, "g_in": True, "p_out": True, "g_out": True}
            if kwargs.get("tri_mul_in_bias", None) is None
            else kwargs.get("tri_mul_in_bias")
        )
        tri_attn_start_bias = (
            {"q": False, "k": False, "v": False, "g": True, "z": False, "o": True}
            if kwargs.get("tri_attn_start_bias", None) is None
            else kwargs.get("tri_attn_start_bias")
        )
        tri_attn_end_bias = (
            {"q": False, "k": False, "v": False, "g": True, "z": False, "o": True}
            if kwargs.get("tri_attn_end_bias", None) is None
            else kwargs.get("tri_attn_end_bias")
        )

        self.tri_mul_out = TriangleMultiplicationNode(
            layer_idx=local_layer_idx,
            dim=c_t,
            hidden_dim=c_hidden_tri_mul,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            bias_flags=tri_mul_out_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
        )

        self.tri_mul_in = TriangleMultiplicationNode(
            layer_idx=local_layer_idx,
            dim=c_t,
            hidden_dim=c_hidden_tri_mul,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            bias_flags=tri_mul_in_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
        )

        self.tri_attn_start = TriangleAttentionStartingNode(
            c_t,
            c_hidden_tri_att,
            no_heads,
            inf=inf,
            layer_idx=local_layer_idx,
            mha_bias_flags=tri_attn_start_bias,
            attn_backend=triangle_attn_backend,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            c_t,
            c_hidden_tri_att,
            no_heads,
            inf=inf,
            layer_idx=local_layer_idx,
            mha_bias_flags=tri_attn_end_bias,
            attn_backend=triangle_attn_backend,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            transposed_bias=tri_attn_transposed_bias,
        )

        if transition_type == "relu":
            self.pair_transition = PairTransition(c_z=c_t, n=pair_transition_n, dtype=dtype, eps=eps)
        elif transition_type == "swiglu":
            self.pair_transition = SwiGLUTransition(dim=c_t, hidden=c_t * pair_transition_n, dtype=dtype, eps=eps)
        else:
            raise ValueError(f"Transition type {transition_type} is not available")

    def trimul_update(self, single: torch.Tensor, single_mask: torch.Tensor) -> torch.Tensor:
        """
        Update the single template with the triangle multiplication
        """
        single = single + self.tri_mul_out(
            single,
            single_mask,
        )
        single = single + self.tri_mul_in(
            single,
            single_mask,
        )
        return single

    def triattn_update(
        self, single: torch.Tensor, single_mask: torch.Tensor, attn_metadata: AttentionMetadata | None = None
    ) -> torch.Tensor:
        """
        Update the single template with the triangle attention
        """
        single = single + self.tri_attn_start(single, single_mask, attn_metadata=attn_metadata)

        single = single + self.tri_attn_end(single, single_mask, attn_metadata=attn_metadata)

        return single

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor, attn_metadata: AttentionMetadata | None = None
    ) -> torch.Tensor:
        single_templates = [t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]

        for i in range(len(single_templates)):
            single = single_templates[i].to(self.dtype)
            single_mask = single_templates_masks[i].to(self.dtype)

            if self.tri_mul_first:
                # Check if single tensor contain multiple samples
                if single.ndim == 5:
                    single = single.flatten(0, 1)

                # Check if single mask tensor contain multiple samples
                if single_mask.ndim == 4:
                    single_mask = single_mask.flatten(0, 1)

                single = self.trimul_update(single, single_mask)
                single = self.triattn_update(single, single_mask, attn_metadata=attn_metadata)
            else:
                single = self.triattn_update(single, single_mask, attn_metadata=attn_metadata)
                single = self.trimul_update(single, single_mask)

            single = single + self.pair_transition(single, single_mask)

            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z


class TemplatePairStack(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_blocks: int,
        no_heads: int,
        pair_transition_n: int,
        tri_mul_first: bool = False,
        trimul_high_precision: bool = False,
        inf: float = 1e9,
        eps: float = 1e-5,
        triangle_attn_backend: str = "VANILLA",
        transition_type: str = "relu",
        tri_mul_out_bias: dict | None = None,
        tri_mul_in_bias: dict | None = None,
        tri_attn_start_bias: dict | None = None,
        tri_attn_end_bias: dict | None = None,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
        tri_attn_transposed_bias: bool = False,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        for layer_idx in range(no_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                tri_mul_first=tri_mul_first,
                trimul_high_precision=trimul_high_precision,
                inf=inf,
                eps=eps,
                local_layer_idx=layer_idx,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
                triangle_attn_backend=triangle_attn_backend,
                transition_type=transition_type,
                tri_mul_out_bias=tri_mul_out_bias,
                tri_mul_in_bias=tri_mul_in_bias,
                tri_attn_start_bias=tri_attn_start_bias,
                tri_attn_end_bias=tri_attn_end_bias,
                tri_attn_transposed_bias=tri_attn_transposed_bias,
            )
            self.blocks.append(block)

        self.layer_norm = nn.LayerNorm(c_t, dtype=dtype, eps=eps)

    def forward(self, t: torch.tensor, mask: torch.tensor, skip_template_pair_stack: bool = False) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        origin_dtype = t.dtype
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        if not skip_template_pair_stack:
            for block in self.blocks:
                t = block(z=t, mask=mask)
        t = self.layer_norm(t)
        return t.to(origin_dtype)


class TemplatePointwiseAttention(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
        eps: float = 1e-5,
        triangle_attn_backend: str = "VANILLA",
        chunk_size: int = 256,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.mha = CrossTriangleAttention(
            layer_idx=0,
            q_hidden_size=self.c_z,
            kv_hidden_size=self.c_t,
            head_dim=self.c_hidden,
            num_attention_heads=self.no_heads,
            num_key_value_heads=self.no_heads,
            gating=False,
            bias_flags={"q": False, "k": False, "v": False, "g": False, "z": False, "o": True},
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=triangle_attn_backend,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, template_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        TODO: Support the chunking or flash-attention here.
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        batch_dims = " ".join([f"b_{i}" for i in range(t.ndim - 4)])
        t = rearrange(t, f"{batch_dims} t i j c -> {batch_dims} i j t c")

        # [*, 1, 1, 1, N_temp]
        biases = [bias]
        if self.chunk_size > 1:
            seq_len = z.shape[1]
            niters = (seq_len + self.chunk_size - 1) // self.chunk_size
            outputs = []
            for i in range(niters):
                start = i * self.chunk_size
                end = start + self.chunk_size
                z_chunk = z[:, start:end:, :, :]
                t_chunk = t[:, start:end:, :, :]
                z_chunk = self.mha(q_x=z_chunk, kv_x=t_chunk, biases=biases)
                outputs.append(z_chunk)
            z = torch.cat(outputs, dim=1)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z
