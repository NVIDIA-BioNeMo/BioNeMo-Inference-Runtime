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
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .attention import SelfAttentionPairBias
from .transition import Transition
from .triangle_nodes import (TriangleAttentionEndingNode,
                             TriangleAttentionStartingNode,
                             TriangleMultiplicationNode,
                             TriangleMultiplicationNodeType)


class PairformerLayer(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 token_s: int,
                 token_z: int,
                 num_heads: int = 16,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 no_update_s: bool = False,
                 no_update_z: bool = False,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 max_transition_tp_size: bool = True,
                 max_attention_pairwise_tp_size: bool = True,
                 triangle_attn_node_chunk_size: int = 0,
                 mapping: Optional[Mapping] = None,
                 attn_backend: str = "VANILLA",
                 skip_create_weights: bool = False):
        super().__init__()
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.mapping = mapping or Mapping()

        if not self.no_update_s:
            self.attention = SelfAttentionPairBias(
                layer_idx=layer_idx,
                c_s=token_s,
                c_z=token_z,
                num_heads=num_heads,
                dtype=dtype,
                inf=inf,
                max_attention_pairwise_tp_size=max_attention_pairwise_tp_size,
                mapping=mapping,
                skip_create_weights=skip_create_weights,
                attn_backend=attn_backend,
            )
        self.tri_mul_out = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
        )
        self.tri_mul_in = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
        )
        self.tri_attn_start = TriangleAttentionStartingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=inf,
            layer_idx=layer_idx,
            dtype=dtype,
            chunk_size=triangle_attn_node_chunk_size,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=inf,
            layer_idx=layer_idx,
            dtype=dtype,
            chunk_size=triangle_attn_node_chunk_size,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )
        if not self.no_update_s:
            self.transition_s = Transition(
                token_s,
                token_s * 4,
                layer_idx=layer_idx,
                eps=eps,
                dtype=dtype,
                max_transition_tp_size=max_transition_tp_size,
                mapping=mapping,
                skip_create_weights=skip_create_weights,
            )
        self.transition_z = Transition(
            token_z,
            token_z * 4,
            layer_idx=layer_idx,
            eps=eps,
            dtype=dtype,
            max_transition_tp_size=max_transition_tp_size,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
        )

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor,
            pair_mask: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_attn_start(
            z,
            mask=pair_mask,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )

        z = z + self.tri_attn_end(
            z,
            mask=pair_mask,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )

        z = z + self.transition_z(z)

        if not self.no_update_s:
            s = s + self.attention(
                s.unsqueeze(0),
                z.unsqueeze(0),
                mask.unsqueeze(0),
                attn_metadata=attn_metadata,
                all_reduce_params=all_reduce_params).squeeze(0)
            s = s + self.transition_s(s)
        return s, z


class PairformerModule(nn.Module):

    def __init__(self, config: PairformerConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(config.num_blocks):
            self.layers.append(
                PairformerLayer(
                    layer_idx=i,
                    token_s=config.token_s,
                    token_z=config.token_z,
                    num_heads=config.num_heads,
                    pairwise_head_width=config.pairwise_head_width,
                    pairwise_num_heads=config.pairwise_num_heads,
                    no_update_s=config.no_update_s,
                    no_update_z=config.no_update_z,
                    dtype=config.torch_dtype,
                    eps=config.eps,
                    inf=config.inf,
                    max_transition_tp_size=config.max_transition_tp_size,
                    max_attention_pairwise_tp_size=config.
                    max_attention_pairwise_tp_size,
                    triangle_attn_node_chunk_size=config.
                    triangle_attn_node_chunk_size,
                    mapping=config.mapping,
                    skip_create_weights=config.skip_create_weights,
                    attn_backend=config.attn_backend,
                ))

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor,
            pair_mask: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, attn_metadata,
                         all_reduce_params)
        return s, z
