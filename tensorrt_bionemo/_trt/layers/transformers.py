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

import tensorrt as trt
from tensorrt_llm.functional import AllReduceParams, Tensor, cast
from tensorrt_llm.layers.normalization import LayerNorm
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


class PairformerLayerV1(Module):

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
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 s_path_dtype: str = None,
                 attention_initial_norm: bool = True,
                 mapping: Mapping = Mapping(),
                 **kwargs):
        super().__init__()
        self.token_z = token_z
        self.token_s = token_s
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.support_batch = support_batch
        self.eps = eps
        self.inf = inf
        self.dtype = dtype
        if s_path_dtype is None:
            s_path_dtype = dtype

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
                dtype=s_path_dtype,
                eps=eps,
                inf=inf,
                initial_norm=attention_initial_norm,
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
            support_batch=support_batch,
            mapping=m)
        self.tri_mul_in = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=token_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            support_batch=support_batch,
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
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
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
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
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
                                           dtype=s_path_dtype)
        m = mapping
        if max_transition_tp_size:
            m = create_max_tp_mapping(mapping, token_z * 4)
        self.transition_z = Transition(local_layer_idx=local_layer_idx,
                                       dim=token_z,
                                       hidden=token_z * 4,
                                       eps=eps,
                                       mapping=m,
                                       dtype=dtype)

    def _transform_z(
            self,
            z: Tensor,
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
        return z

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pairmask: Tensor,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        original_dtype = z.dtype
        z = self._transform_z(z, pairmask, attention_params, all_reduce_params)
        if not self.no_update_s:
            if self.support_batch:
                s = s + self.attention(s,
                                       z,
                                       mask,
                                       attention_params=attention_params,
                                       all_reduce_params=all_reduce_params)
            else:
                s = s + self.attention(
                    s.unsqueeze(0),
                    z.unsqueeze(0),
                    mask.unsqueeze(0),
                    attention_params=attention_params,
                    all_reduce_params=all_reduce_params).squeeze(0, False)
            s = s + self.transition_s(s)
        if s.dtype != original_dtype:
            s = cast(s, original_dtype)
        if z.dtype != original_dtype:
            z = cast(z, original_dtype)
        return s, z


class PairformerLayerV2(PairformerLayerV1):
    config_class = PairformerConfig
    build_config_class = PairformerBuildConfig

    def __init__(self, post_layer_norm: bool = False, **kwargs):
        kwargs["s_path_dtype"] = "float32"
        super().__init__(**kwargs)
        self.post_layer_norm = post_layer_norm

        self.pre_norm_s = LayerNorm(normalized_shape=[self.token_s],
                                    eps=self.eps,
                                    dtype="float32",
                                    tp_size=1,
                                    tp_dim=0)

        self.post_norm_s = None
        if self.post_layer_norm:
            self.post_norm_s = LayerNorm(normalized_shape=[self.token_s],
                                         eps=self.eps,
                                         dtype="float32",
                                         tp_size=1,
                                         tp_dim=0)

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        z = self._transform_z(z, pair_mask, attention_params, all_reduce_params)
        original_dtype = s.dtype
        z = cast(z, "float32")
        s = cast(s, "float32")
        s_normed = self.pre_norm_s(s)
        if self.support_batch:
            s = s + self.attention(s_normed, z, mask, attention_params,
                                   all_reduce_params)
        else:
            s = s + self.attention(s_normed.unsqueeze(0), z.unsqueeze(0),
                                   mask.unsqueeze(0), attention_params,
                                   all_reduce_params).squeeze(0, False)
        s = s + self.transition_s(s)
        if self.post_layer_norm:
            s = self.post_norm_s(s)
        if s.dtype != original_dtype:
            s = cast(s, original_dtype)
        if z.dtype != original_dtype:
            z = cast(z, original_dtype)
        return s, z


class PairformerModule(PretrainedModule):
    config_class = PairformerConfig
    build_config_class = PairformerBuildConfig

    def __init__(self, config: PairformerConfig):
        super().__init__(config)
        layer_cls = PairformerLayerV1 if config.version == "v1" else PairformerLayerV2
        self.layers = ModuleList([
            layer_cls(local_layer_idx=i,
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
                      triangle_attn_backend=config.triangle_attn_backend,
                      support_batch=config.support_batch,
                      mapping=config.mapping,
                      post_layer_norm=config.post_layer_norm,
                      attention_initial_norm=config.attention_initial_norm)
            for i in range(config.num_blocks)
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
        for layer in network.get_layers():
            if "layer_norm_" in layer.name and "NORMALIZATION_0" in layer.name:
                layer.trt_layer.precision = trt.float32
            if "softmax" in layer.name and "SOFTMAX_0" in layer.name:
                layer.trt_layer.precision = trt.float32
        return network
