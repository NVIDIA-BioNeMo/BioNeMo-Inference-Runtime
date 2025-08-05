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
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation, cast,
                                     concat, shape, slice)
from tensorrt_llm.layers.linear import ColumnLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.network import Network

from tensorrt_bionemo._trt.functional import identity_sz
from tensorrt_bionemo.models.boltz1.configs import (PairformerBuildConfig, PairformerConfig,
                                       TokenTransformerBuildConfig,
                                       TokenTransformerConfig)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from ..module_utils import PretrainedModule
from .attention import AttentionParams, SelfAttentionPairBias
from .normalization import AdaLN
from .transition import ConditionedTransitionBlock, Transition
from .triangle_nodes import (TriangleAttentionNode, TriangleAttentionNodeType,
                             TriangleMultiplicationNode,
                             TriangleMultiplicationNodeType)


class PairformerLayerV1(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 token_s: int = 384,
                 token_z: int = 128,
                 num_heads: int = 16,
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
                 fallback_threshold: int = 0,
                 mapping: Mapping = Mapping(),
                 **kwargs):
        super().__init__()

        self.token_z = token_z
        self.token_s = token_s
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.support_batch = support_batch
        self.triangle_attn_backend = triangle_attn_backend
        self.fallback_threshold = fallback_threshold

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
                need_project_z=True,
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
            fallback_threshold=self.fallback_threshold,
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
            fallback_threshold=self.fallback_threshold,
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
            pair_mask: Tensor,
            attention_params: AttentionParams = None,
            all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        z = z + self.tri_attn_start(z,
                                    mask=pair_mask,
                                    attention_params=attention_params,
                                    all_reduce_params=all_reduce_params)
        z = z + self.tri_attn_end(z,
                                  mask=pair_mask,
                                  attention_params=attention_params,
                                  all_reduce_params=all_reduce_params)
        z = z + self.transition_z(z)
        return z

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        if not self.no_update_s:
            # Add identity to break Myelin fusion
            use_identity_plugin = self.triangle_attn_backend != "VANILLA"
            s, z = identity_sz(s, z, use_identity_plugin)
        original_dtype = z.dtype
        z = self._transform_z(z, pair_mask, attention_params, all_reduce_params)
        if not self.no_update_s:
            if self.support_batch:
                s = s + self.attention(s,
                                       z,
                                       mask,
                                       compute_pair_bias=True,
                                       attention_params=attention_params,
                                       all_reduce_params=all_reduce_params)
            else:
                s = s + self.attention(
                    s.unsqueeze(0),
                    z.unsqueeze(0),
                    mask.unsqueeze(0),
                    compute_pair_bias=True,
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
        # Add identity to break Myelin fusion
        use_identity_plugin = self.triangle_attn_backend != "VANILLA"
        s, z = identity_sz(s, z, use_identity_plugin)
        z = self._transform_z(z, pair_mask, attention_params, all_reduce_params)
        original_dtype = s.dtype
        z = cast(z, "float32")
        s = cast(s, "float32")
        s_normed = self.pre_norm_s(s)
        if self.support_batch:
            s = s + self.attention(s_normed,
                                   z,
                                   mask,
                                   compute_pair_bias=True,
                                   attention_params=attention_params,
                                   all_reduce_params=all_reduce_params)
        else:
            s = s + self.attention(s_normed.unsqueeze(0),
                                   z.unsqueeze(0),
                                   mask.unsqueeze(0),
                                   compute_pair_bias=True,
                                   attention_params=attention_params,
                                   all_reduce_params=all_reduce_params).squeeze(
                                       0, False)
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
        logger.info(
            f"Using triangle_attn_backend: {config.triangle_attn_backend}, cueq threshold: {config.triangle_attn_cueq_fallback_threshold}"
        )
        self.layers = ModuleList([
            layer_cls(
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
                triangle_attn_backend=config.triangle_attn_backend,
                support_batch=config.support_batch,
                mapping=config.mapping,
                fallback_threshold=config.triangle_attn_cueq_fallback_threshold,
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


class PairformerNoSeqLayer(PairformerLayerV1):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 triangle_attn_backend: str = 'VANILLA',
                 **kwargs):
        kwargs["no_update_s"] = True
        super().__init__(local_layer_idx=local_layer_idx,
                         token_z=token_z,
                         pairwise_head_width=pairwise_head_width,
                         pairwise_num_heads=pairwise_num_heads,
                         triangle_attn_backend=triangle_attn_backend,
                         **kwargs)

    def forward(self,
                z: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        _, update_z = super().forward(s=None,
                                      z=z,
                                      mask=None,
                                      pair_mask=pair_mask,
                                      attention_params=attention_params,
                                      all_reduce_params=all_reduce_params)
        return update_z


class PairformerNoSeqModule(Module):

    def __init__(self,
                 num_blocks: int = 8,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 dtype: str = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 triangle_attn_backend: str = 'VANILLA',
                 mapping: Mapping = Mapping(),
                 **kwargs):
        super().__init__()
        logger.info(f"Using triangle_attn_backend: {triangle_attn_backend}")
        self.layers = ModuleList([
            PairformerNoSeqLayer(local_layer_idx=i,
                                 token_z=token_z,
                                 pairwise_head_width=pairwise_head_width,
                                 pairwise_num_heads=pairwise_num_heads,
                                 dtype=dtype,
                                 eps=eps,
                                 inf=inf,
                                 triangle_attn_backend=triangle_attn_backend,
                                 mapping=mapping,
                                 **kwargs) for i in range(num_blocks)
        ])

    def forward(self,
                z: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        for layer in self.layers:
            z = layer(z, pair_mask, attention_params, all_reduce_params)
        return z


class DiffusionTransformerLayer(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 num_heads: int,
                 dim: int,
                 dim_single_cond: int,
                 dim_pairwise: int = 128,
                 dtype: str = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 attention_initial_norm: bool = False,
                 post_layer_norm: bool = False,
                 need_project_z: bool = True,
                 max_batch_size: int = 1,
                 mapping: Optional[Mapping] = None):
        super().__init__()
        self.num_heads = num_heads
        self.adaln = AdaLN(dim,
                           dim_single_cond,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping)

        self.pair_bias_attn = SelfAttentionPairBias(
            local_layer_idx=local_layer_idx,
            c_s=dim,
            c_z=dim_pairwise,
            num_heads=num_heads,
            dtype=dtype,
            eps=eps,
            inf=inf,
            initial_norm=attention_initial_norm,
            need_project_z=need_project_z,
            max_batch_size=max_batch_size,
            mapping=mapping)
        self.output_projection = ColumnLinear(
            dim_single_cond,
            dim,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
            is_qkv=False,
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim,
            dim_single_cond=dim_single_cond,
            expansion_factor=2,
            dtype=dtype,
            eps=eps,
            mapping=mapping)
        self.post_lnorm = None
        if post_layer_norm:
            self.post_lnorm = LayerNorm(normalized_shape=[dim],
                                        eps=eps,
                                        dtype=dtype,
                                        tp_size=1,
                                        tp_dim=0)

    def forward(self,
                a: Tensor,
                s: Tensor,
                bias: Tensor,
                mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        b = self.adaln(a, s)
        b = self.pair_bias_attn(s=b,
                                z=bias,
                                mask=mask,
                                compute_pair_bias=False,
                                attention_params=attention_params,
                                all_reduce_params=all_reduce_params)
        b = activation(self.output_projection(s),
                       trt.ActivationType.SIGMOID) * b  # TODO: fuse here
        a = a + b
        a = a + self.transition(a, s, all_reduce_params=all_reduce_params)
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


class TokenTransformer(PretrainedModule):
    config_class = TokenTransformerConfig
    build_config_class = TokenTransformerBuildConfig

    def __init__(self, config: TokenTransformerConfig):
        super().__init__(config)
        self.version = config.version
        logger.info(
            f"Using pairwise attention backend: {config.pairwise_attn_backend}")
        self.layers = ModuleList([
            DiffusionTransformerLayer(
                local_layer_idx=i,
                num_heads=config.num_heads,
                dim=config.dim,
                dim_single_cond=config.dim_single_cond,
                dtype=config.dtype,
                eps=config.norm_epsilon,
                inf=config.mask_inf,
                attention_initial_norm=config.attention_initial_norm,
                post_layer_norm=config.post_layer_norm,
                need_project_z=config.version == "v1",
                max_batch_size=config.max_batch_size,
                mapping=config.mapping) for i in range(config.num_blocks)
        ])

    def forward(self,
                a: Tensor,
                s: Tensor,
                z: Tensor,
                mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        """
        Token transformer for both v1 and v2
        Args:
            a: [B, S, dim]
            s: [B, S, dim_single_cond]
            z: [1, H, N, N, L] for v1, [1, N, N, H*L] for v2
            mask: [B, S]
            attention_params: AttentionParams
            all_reduce_params: AllReduceParams
        """
        if self.version == "v2":
            B = shape(z, 0)
            N = shape(z, 1)
            M = shape(z, 2)
            L = self.config.num_blocks
            D = self.config.num_heads
            z = z.view(concat([B, N, M, L, D]))
            z = z.permute([0, 4, 1, 2, 3])  # [B, H, N, N, L]
        else:
            B = shape(z, 0)
            N = shape(z, 2)
            M = shape(z, 3)
            L = self.config.num_blocks
            D = self.config.num_heads
        bias = z
        for i, layer in enumerate(self.layers):
            # Slice the bias term for each layer
            starts = concat([0, 0, 0, 0, i])
            ends = concat([B, D, N, M, 1])
            sub_bias = slice(bias, starts, ends).squeeze(-1, True)
            a = layer(a, s, sub_bias, mask, attention_params, all_reduce_params)
        return a
