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

from tensorrt_llm.functional import AllReduceParams, Tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.network import Network

from tensorrt_bionemo.confs.modules.transformers import (PairformerBuildConfig,
                                                         PairformerConfig)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from ..module_utils import PretrainedModule
from .attention import AttentionParams, SelfAttentionPairBias
from .transition import Transition
from .triangle_nodes import (TriangleAttentionNode, TriangleAttentionNodeType,
                             TriangleMultiplicationNode,
                             TriangleMultiplicationNodeType)


class PairformerLayer(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 token_s: int,
                 token_z: int,
                 num_heads: int,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 no_update_s: bool = False,
                 no_update_z: bool = False,
                 chunk_size: int = 0,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 dtype: str = None,
                 max_transition_tp_size: bool = False,
                 max_attention_pairwise_tp_size: bool = False,
                 max_tri_mul_tp_size: bool = True,
                 mapping: Mapping = Mapping()):
        super().__init__()

        self.token_z = token_z
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z

        self.attention = None
        if not self.no_update_s:
            m = mapping
            if max_attention_pairwise_tp_size:
                m = create_max_tp_mapping(mapping, num_heads)
            self.attention = SelfAttentionPairBias(
                local_layer_idx=local_layer_idx,
                c_s=token_s,
                c_z=token_z,
                num_heads=num_heads,
                dtype=dtype,
                eps=eps,
                inf=inf,
                mapping=m)
        m = mapping
        if max_tri_mul_tp_size:
            m = create_max_tp_mapping(mapping, token_z)
        self.tri_mul_out = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=token_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            mapping=m)
        self.tri_mul_in = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=token_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            mapping=m)
        self.tri_attn_start = TriangleAttentionNode(
            local_layer_idx=local_layer_idx,
            c_in=token_z,
            c_hidden=pairwise_head_width,
            num_heads=pairwise_num_heads,
            node_type=TriangleAttentionNodeType.STARTING,
            dtype=dtype,
            eps=eps,
            inf=inf,
            chunk_size=chunk_size,
            mapping=mapping)
        self.tri_attn_end = TriangleAttentionNode(
            local_layer_idx=local_layer_idx,
            c_in=token_z,
            c_hidden=pairwise_head_width,
            num_heads=pairwise_num_heads,
            node_type=TriangleAttentionNodeType.ENDING,
            dtype=dtype,
            eps=eps,
            inf=inf,
            chunk_size=chunk_size,
            mapping=mapping)
        if not self.no_update_s:
            m = mapping
            if max_transition_tp_size:
                m = create_max_tp_mapping(mapping, token_s * 4)
            self.transition_s = Transition(local_layer_idx=local_layer_idx,
                                           dim=token_s,
                                           hidden=token_s * 4,
                                           eps=eps,
                                           mapping=m,
                                           dtype=dtype)
        m = mapping
        if max_transition_tp_size:
            m = create_max_tp_mapping(mapping, token_z * 4)
        self.transition_z = Transition(local_layer_idx=local_layer_idx,
                                       dim=token_z,
                                       hidden=token_z * 4,
                                       eps=eps,
                                       mapping=m,
                                       dtype=dtype)

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pairmask: Tensor,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        z = z + self.tri_mul_out(z, mask=pairmask)
        z = z + self.tri_mul_in(z, mask=pairmask)
        z = z + self.tri_attn_start(z,
                                    mask=pairmask,
                                    attention_params=attention_params,
                                    all_reduce_params=all_reduce_params)
        z = z + self.tri_attn_end(z,
                                  mask=pairmask,
                                  attention_params=attention_params,
                                  all_reduce_params=all_reduce_params)
        z = z + self.transition_z(z)
        if not self.no_update_s:
            s = s + self.attention(s,
                                   z,
                                   mask,
                                   attention_params=attention_params,
                                   all_reduce_params=all_reduce_params)
            s = s + self.transition_s(s)
        return s, z


class PairformerModule(PretrainedModule):
    config_class = PairformerConfig
    build_config_class = PairformerBuildConfig

    def __init__(self, config: PairformerConfig):
        super().__init__(config)

        self.layers = ModuleList([
            PairformerLayer(
                local_layer_idx=i,
                token_s=config.token_s,
                token_z=config.token_z,
                num_heads=config.num_heads,
                pairwise_head_width=config.pairwise_head_width,
                pairwise_num_heads=config.pairwise_num_heads,
                no_update_s=config.no_update_s,
                no_update_z=config.no_update_z,
                dtype=config.dtype,
                eps=config.norm_epsilon,
                inf=config.mask_inf,
                max_transition_tp_size=config.max_transition_tp_size,
                max_attention_pairwise_tp_size=config.
                max_attention_pairwise_tp_size,
                max_tri_mul_tp_size=config.max_tri_mul_tp_size,
                mapping=config.mapping) for i in range(config.num_blocks)
        ])

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, attention_params,
                         all_reduce_params)
        return s, z

    @staticmethod
    def weakly_typed(network: Network, dtype: str = None) -> Network:
        logger.info("Call weakly_typed on PairformerModule")
        return network
