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
from tensorrt_llm_lite.functional import (Tensor, activation, cast, concat,
                                          shape, slice)
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.layers.normalization import LayerNorm
from tensorrt_llm_lite.logger import logger
from tensorrt_llm_lite.module import Module, ModuleList
from tensorrt_llm_lite.network import Network

from tensorrt_bionemo._trt.functional import identity_sz
from tensorrt_bionemo.configs import (DiffusionTransformerBuildConfig,
                                      DiffusionTransformerConfig,
                                      EvoformerStackBuildConfig,
                                      EvoformerStackConfig,
                                      PairformerBuildConfig, PairformerConfig)

from ..module_utils import PretrainedModule
from .attention import AttentionParams, MSAAttention, SelfAttentionPairBias
from .normalization import AdaLN
from .outer_product_mean import OuterProductMean
from .transition import (ConditionedTransitionBlock, MSATransition,
                         PairTransition, Transition)
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
                 trimul_high_precision: bool = True,
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
        self.trimul_high_precision = trimul_high_precision
        self.eps = eps
        self.inf = inf
        self.dtype = dtype
        if s_path_dtype is None:
            s_path_dtype = dtype

        self.attention = None
        if not self.no_update_s:
            self.attention = SelfAttentionPairBias(
                local_layer_idx=local_layer_idx,
                c_s=token_s,
                c_z=token_z,
                num_heads=num_heads,
                dtype=s_path_dtype,
                eps=eps,
                inf=inf,
                initial_norm=attention_initial_norm,
                need_project_z=True)
        self.tri_mul_out = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=token_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            support_batch=support_batch,
            high_precision=trimul_high_precision)
        self.tri_mul_in = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=token_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            support_batch=support_batch,
            high_precision=trimul_high_precision)
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
            fallback_threshold=self.fallback_threshold)
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
            fallback_threshold=self.fallback_threshold)
        if not self.no_update_s:
            self.transition_s = Transition(local_layer_idx=local_layer_idx,
                                           dim=token_s,
                                           hidden=token_s * 4,
                                           eps=eps,
                                           dtype=s_path_dtype)
        self.transition_z = Transition(local_layer_idx=local_layer_idx,
                                       dim=token_z,
                                       hidden=token_z * 4,
                                       eps=eps,
                                       dtype=dtype)

    def _transform_z(self,
                     z: Tensor,
                     pair_mask: Tensor,
                     attention_params: AttentionParams = None) -> Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        z = z + self.tri_attn_start(
            z, mask=pair_mask, attention_params=attention_params)
        z = z + self.tri_attn_end(
            z, mask=pair_mask, attention_params=attention_params)
        z = z + self.transition_z(z)
        return z

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: AttentionParams = None) -> Tensor:
        if not self.no_update_s:
            # Add identity to break Myelin fusion
            use_identity_plugin = (
                self.triangle_attn_backend
                != "VANILLA")  # and self.trimul_high_precision, works for B200
            s, z = identity_sz(s, z, use_identity_plugin)
        original_dtype = z.dtype
        z = self._transform_z(z, pair_mask, attention_params)
        if not self.no_update_s:
            if self.support_batch:
                s = s + self.attention(s,
                                       z,
                                       mask,
                                       compute_pair_bias=True,
                                       attention_params=attention_params)
            else:
                s = s + self.attention(
                    s.unsqueeze(0),
                    z.unsqueeze(0),
                    mask.unsqueeze(0),
                    compute_pair_bias=True,
                    attention_params=attention_params).squeeze(0, False)
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
                                    dtype="float32")

        self.post_norm_s = None
        if self.post_layer_norm:
            self.post_norm_s = LayerNorm(normalized_shape=[self.token_s],
                                         eps=self.eps,
                                         dtype="float32")

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        # Add identity to break Myelin fusion
        use_identity_plugin = (
            self.triangle_attn_backend
            != "VANILLA")  # and self.trimul_high_precision, works for B200
        s, z = identity_sz(s, z, use_identity_plugin)
        z = self._transform_z(z, pair_mask, attention_params)
        original_dtype = s.dtype
        z = cast(z, "float32")
        s = cast(s, "float32")
        s_normed = self.pre_norm_s(s)
        if self.support_batch:
            s = s + self.attention(s_normed,
                                   z,
                                   mask,
                                   compute_pair_bias=True,
                                   attention_params=attention_params)
        else:
            s = s + self.attention(s_normed.unsqueeze(0),
                                   z.unsqueeze(0),
                                   mask.unsqueeze(0),
                                   compute_pair_bias=True,
                                   attention_params=attention_params).squeeze(
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
            f"Using triangle_attn_backend: {config.triangle_attention_backend}, trimul_high_precision: {config.trimul_high_precision}"
        )
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
                      triangle_attn_backend=config.triangle_attention_backend,
                      support_batch=config.support_batch,
                      fallback_threshold=config.
                      triangle_attn_cueq_fallback_threshold,
                      post_layer_norm=config.post_layer_norm,
                      attention_initial_norm=config.attention_initial_norm,
                      trimul_high_precision=config.trimul_high_precision)
            for i in range(config.num_blocks)
        ])

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, attention_params)
        return s, z

    @staticmethod
    def weakly_typed(network: Network, dtype: str = None) -> Network:
        logger.info("Call weakly_typed on PairformerModule")
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
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        _, update_z = super().forward(s=None,
                                      z=z,
                                      mask=None,
                                      pair_mask=pair_mask,
                                      attention_params=attention_params)
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
                                 **kwargs) for i in range(num_blocks)
        ])

    def forward(self,
                z: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        for layer in self.layers:
            z = layer(z, pair_mask, attention_params)
        return z


class DiffusionTransformerLayer(Module):

    def __init__(
            self,
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
            need_compute_pair_bias: bool = False,  # For OF3
            max_batch_size: int = 1,
            attn_bias_flags: dict[str, bool] = {
                "q": True,
                "k": False,
                "v": False,
                "g": False,
                "z": False,
                "norm_z": True,
                "o": False,
            },
            conditioned_transition_using_silu: bool = False,
            attn_output_gate: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.need_compute_pair_bias = need_compute_pair_bias
        self.attn_output_gate = attn_output_gate
        self.adaln = AdaLN(dim, dim_single_cond, eps=eps, dtype=dtype)

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
            bias_flags=attn_bias_flags)
        self.output_projection = None
        if self.attn_output_gate:
            self.output_projection = Linear(
                dim_single_cond,
                dim,
                dtype=dtype,
                is_qkv=False,
            )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim,
            dim_single_cond=dim_single_cond,
            expansion_factor=2,
            dtype=dtype,
            eps=eps,
            using_silu=conditioned_transition_using_silu)
        self.post_lnorm = None
        if post_layer_norm:
            self.post_lnorm = LayerNorm(normalized_shape=[dim],
                                        eps=eps,
                                        dtype=dtype)

    def forward(self,
                a: Tensor,
                s: Tensor,
                bias: Tensor,
                mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        b = self.adaln(a, s)
        b = self.pair_bias_attn(s=b,
                                z=bias,
                                mask=mask,
                                compute_pair_bias=self.need_compute_pair_bias,
                                attention_params=attention_params)
        if self.attn_output_gate:
            b = activation(self.output_projection(s),
                           trt.ActivationType.SIGMOID) * b  # TODO: fuse here
        a = a + b
        a = a + self.transition(a, s)
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


class TokenTransformer(PretrainedModule):
    config_class = DiffusionTransformerConfig
    build_config_class = DiffusionTransformerBuildConfig

    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__(config)
        self.version = config.version
        logger.info(
            f"Using pairwise attention backend: {config.pairwise_attn_backend}"
        )
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
                conditioned_transition_using_silu=getattr(
                    config, 'conditioned_transition_using_silu', False),
                attn_output_gate=getattr(config, 'attn_output_gate', True),
                attn_bias_flags={
                    "q": True,
                    "k": False,
                    "v": False,
                    "g": getattr(config, 'attn_gate_bias', False),
                    "z": False,
                    "norm_z": True,
                    "o": False,
                }) for i in range(config.num_blocks)
        ])

    def forward(self,
                a: Tensor,
                s: Tensor,
                z: Tensor,
                mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        """
        Token transformer for both v1 and v2
        Args:
            a: [B, S, dim]
            s: [B, S, dim_single_cond]
            z: [1, H, N, N, L] for v1, [1, N, N, H*L] for v2
            mask: [B, S]
            attention_params: AttentionParams
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
            a = layer(a, s, sub_bias, mask, attention_params)
        return a


class OpenFold3DiffusionTransformer(PretrainedModule):
    config_class = DiffusionTransformerConfig
    build_config_class = DiffusionTransformerBuildConfig

    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__(config)
        self.version = config.version
        logger.info(
            f"Using pairwise attention backend: {config.pairwise_attn_backend}"
        )
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
                need_project_z=True,
                need_compute_pair_bias=True,
                attn_bias_flags={
                    "q": True,
                    "k": False,
                    "v": False,
                    "g": False,
                    "z": False,
                    "norm_z": False,
                    "o": False,
                },
                max_batch_size=config.max_batch_size,
                conditioned_transition_using_silu=True)
            for i in range(config.num_blocks)
        ])

    def forward(self,
                a: Tensor,
                s: Tensor,
                z: Tensor,
                mask: Optional[Tensor] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        for layer in self.layers:
            a = layer(a, s, z, mask, attention_params)
        return a


class EvoformerBlock(Module):
    """ Implementation of the Evoformer block from the AlphaFold2 paper.
    https://github.com/aqlaboratory/openfold/blob/main/openfold/model/evoformer.py#L377
    """

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 c_m: int,
                 c_z: int,
                 c_hidden_msa_att: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 transition_n: int,
                 no_column_attention: bool = False,
                 opm_first: bool = False,
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 dtype: str = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 chunk_size: int = 0,
                 **kwargs):
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
        self.triangle_attn_backend = triangle_attn_backend
        self.support_batch = support_batch
        self.dtype = dtype
        self.eps = eps
        self.inf = inf

        self.msa_att_row = MSAAttention(
            local_layer_idx=local_layer_idx,
            c_in=c_m,
            num_heads=no_heads_msa,
            c_z=c_z,
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
            need_project_z=True,
            eps=eps,
            inf=inf,
            dtype=dtype)

        self.msa_transition = MSATransition(c_m=c_m,
                                            n=transition_n,
                                            dtype=dtype,
                                            eps=eps)

        self.outer_product_mean = OuterProductMean(
            c_in=c_m,
            c_hidden=c_hidden_opm,
            c_out=c_z,
            eps=eps,
            mask_eps=1e-3,
            norm_mask_by_eps=True,
            norm_before_output=False,
            cast_to_float_before_einsum=False,
            bias_flags={
                "proj_a": True,
                "proj_b": True,
                "proj_o": True
            },
            dtype=dtype)
        self.tri_mul_out = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=c_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            support_batch=support_batch,
            bias_flags={
                "p_in": True,
                "g_in": True,
                "p_out": True,
                "g_out": True
            },
            high_precision=False)
        self.tri_mul_in = TriangleMultiplicationNode(
            local_layer_idx=local_layer_idx,
            dim=c_z,
            dtype=dtype,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            support_batch=support_batch,
            bias_flags={
                "p_in": True,
                "g_in": True,
                "p_out": True,
                "g_out": True
            },
            high_precision=False)
        self.tri_attn_start = TriangleAttentionNode(
            local_layer_idx=local_layer_idx,
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            num_heads=no_heads_pair,
            node_type=TriangleAttentionNodeType.STARTING,
            dtype=dtype,
            eps=eps,
            inf=inf,
            chunk_size=chunk_size,
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
            mha_bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "z": False,
                "o": True
            })
        self.tri_attn_end = TriangleAttentionNode(
            local_layer_idx=local_layer_idx,
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            num_heads=no_heads_pair,
            node_type=TriangleAttentionNodeType.ENDING,
            dtype=dtype,
            eps=eps,
            inf=inf,
            chunk_size=chunk_size,
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
            mha_bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "z": False,
                "o": True
            })

        self.pair_transition = PairTransition(c_z=c_z,
                                              n=transition_n,
                                              dtype=dtype,
                                              eps=eps)

        if not self.no_column_attention:
            self.msa_att_col = MSAAttention(
                local_layer_idx=local_layer_idx,
                c_in=c_m,
                num_heads=no_heads_msa,
                c_z=None,
                triangle_attn_backend=triangle_attn_backend,
                support_batch=support_batch,
                need_project_z=False,
                transpose_input=True,
                eps=eps,
                inf=inf,
                dtype=dtype)

    def _compute_opm(
        self,
        m: Tensor,
        z: Tensor,
        msa_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        opm = self.outer_product_mean(m, mask=msa_mask)
        z = z + opm
        return m, z

    def forward(self,
                m: Optional[Tensor],
                z: Optional[Tensor],
                msa_mask: Tensor,
                pair_mask: Tensor,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
        """
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)

        m = m + self.msa_att_row(
            m, z, mask=msa_mask, attention_params=attention_params)

        if not self.no_column_attention:
            m = m + self.msa_att_col(
                m, z=None, mask=msa_mask, attention_params=attention_params)
        msa_trans_mask = msa_mask
        m = m + self.msa_transition(m, mask=msa_trans_mask)

        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)

        if self.opm_first:
            # Break Myelin fusion
            z, m = identity_sz(z, m, True)
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        z = z + self.tri_attn_start(
            z, mask=pair_mask, attention_params=attention_params)
        z = z + self.tri_attn_end(
            z, mask=pair_mask, attention_params=attention_params)
        pair_trans_mask = pair_mask
        z = z + self.pair_transition(z, mask=pair_trans_mask)

        return m, z


class EvoformerStack(PretrainedModule):
    config_class = EvoformerStackConfig
    build_config_class = EvoformerStackBuildConfig

    def __init__(self, config: EvoformerStackConfig):
        super().__init__(config)
        self.blocks = ModuleList([
            EvoformerBlock(
                local_layer_idx=i,
                c_m=config.c_m,
                c_z=config.c_z,
                c_hidden_msa_att=config.c_hidden_msa_att,
                c_hidden_opm=config.c_hidden_opm,
                c_hidden_mul=config.c_hidden_mul,
                c_hidden_pair_att=config.c_hidden_pair_att,
                no_heads_msa=config.no_heads_msa,
                no_heads_pair=config.no_heads_pair,
                transition_n=config.transition_n,
                no_column_attention=config.no_column_attention,
                opm_first=config.opm_first,
                triangle_attn_backend=config.triangle_attention_backend,
                support_batch=config.support_batch,
                dtype=config.dtype,
                eps=config.norm_epsilon,
                inf=config.mask_inf,
                chunk_size=config.chunk_size) for i in range(config.no_blocks)
        ])
        self.linear = Linear(
            config.c_m,
            config.c_s,
            bias=True,
            dtype=config.dtype,
            is_qkv=False,
        )

    def forward(
        self,
        m: Optional[Tensor],
        z: Optional[Tensor],
        msa_mask: Tensor,
        pair_mask: Tensor,
        attention_params: Optional[AttentionParams] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            m, z = block(m, z, msa_mask, pair_mask, attention_params)

        starts = concat([0, 0, 0, 0])
        ends = concat([shape(m, 0), 1, shape(m, 2), shape(m, 3)])
        s = slice(m, starts, ends).squeeze(1, True)
        s = self.linear(s)

        return m, z, s
