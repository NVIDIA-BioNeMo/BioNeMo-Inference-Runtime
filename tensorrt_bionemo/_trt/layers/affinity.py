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
from typing import Optional, Tuple

import tensorrt as trt
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation,
                                     allgather, constant_to_tensor_, flatten,
                                     split, sum)
from tensorrt_llm.layers.embedding import Embedding
from tensorrt_llm.layers.linear import ColumnLinear, Linear, RowLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo.configs import (AffinityModuleBuildConfig,
                                      AffinityModuleConfig)
from tensorrt_bionemo.mapping import Mapping

from ..module_utils import PretrainedModule
from .attention import AttentionParams
from .transformers import PairformerNoSeqModule
from .transition import PairwiseConditioning


class AffinityHeadsTransformer(Module):

    def __init__(self,
                 token_z: int,
                 token_s: int,
                 dtype: str = None,
                 eps: float = 1e-5,
                 mapping: Optional[Mapping] = None):
        super().__init__()
        self.token_z = token_z
        self.token_s = token_s
        self.dtype = dtype
        self.eps = eps
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group

        self.affinity_out_mlp_linear_0 = ColumnLinear(token_z,
                                                      token_z,
                                                      bias=True,
                                                      dtype=dtype,
                                                      tp_group=self.tp_group,
                                                      tp_size=self.tp_size,
                                                      gather_output=False)

        self.affinity_out_mlp_linear_1 = RowLinear(token_z,
                                                   token_s,
                                                   bias=True,
                                                   dtype=dtype,
                                                   tp_group=self.tp_group,
                                                   tp_size=self.tp_size)

        self.to_affinity_pred_value_0 = ColumnLinear(token_s,
                                                     token_s,
                                                     bias=True,
                                                     dtype=dtype,
                                                     tp_group=self.tp_group,
                                                     tp_size=self.tp_size,
                                                     gather_output=False)

        self.to_affinity_pred_value_1 = RowLinear(token_s,
                                                  token_s,
                                                  bias=True,
                                                  dtype=dtype,
                                                  tp_group=self.tp_group,
                                                  tp_size=self.tp_size)
        self.to_affinity_pred_value_2 = Linear(token_s,
                                               1,
                                               bias=True,
                                               dtype=dtype)

        self.to_affinity_pred_score_0 = ColumnLinear(token_s,
                                                     token_s,
                                                     bias=True,
                                                     dtype=dtype,
                                                     tp_group=self.tp_group,
                                                     tp_size=self.tp_size,
                                                     gather_output=False)

        self.to_affinity_pred_score_1 = RowLinear(token_s,
                                                  token_s,
                                                  bias=True,
                                                  dtype=dtype,
                                                  tp_group=self.tp_group,
                                                  tp_size=self.tp_size)
        self.to_affinity_pred_score_2 = Linear(token_s,
                                               1,
                                               bias=True,
                                               dtype=dtype)

        self.to_affinity_logits_binary = Linear(1, 1, bias=True, dtype=dtype)

    def forward(self,
                z: Tensor,
                cross_pair_mask: Tensor,
                multiplicity: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (B, I, token_s)
            cross_pair_mask: (B, num_dist_bins, num_dist_bins, 1)
            multiplicity(int): default 1, unsupported for now (TODO: support multiplicity > 1)

        Returns:
            pred_value: (batch_size, 1)
            logits_binary: (batch_size, 1)
        """
        assert multiplicity == 1, "Multiplicity > 1 is not supported yet"

        a = sum(z * cross_pair_mask, dim=(1, 2))
        eps_const = constant_to_tensor_(1e-7, dtype=z.dtype, to_array=False)
        b = sum(cross_pair_mask, dim=(1, 2)) + eps_const
        g = a / b

        g = self.affinity_out_mlp_linear_0(g)
        g = activation(g, act_type=trt.ActivationType.RELU)
        g = self.affinity_out_mlp_linear_1(g)
        g = activation(g, act_type=trt.ActivationType.RELU)

        pred_value = self.to_affinity_pred_value_0(g)  # column linear
        pred_value = activation(pred_value, act_type=trt.ActivationType.RELU)
        pred_value = self.to_affinity_pred_value_1(pred_value)  # row linear
        pred_value = activation(pred_value, act_type=trt.ActivationType.RELU)
        pred_value = self.to_affinity_pred_value_2(pred_value)

        pred_score = self.to_affinity_pred_score_0(g)  # column linear
        pred_score = activation(pred_score, act_type=trt.ActivationType.RELU)
        pred_score = self.to_affinity_pred_score_1(pred_score)  # row linear
        pred_score = activation(pred_score, act_type=trt.ActivationType.RELU)
        pred_score = self.to_affinity_pred_score_2(pred_score)

        pred_value = flatten(pred_value).unsqueeze(1)
        pred_score = flatten(pred_score).unsqueeze(1)
        logits_binary = self.to_affinity_logits_binary(pred_score)

        return pred_value, logits_binary


class AffinityModule(PretrainedModule):
    """ This module is used in Boltz-2 """
    config_class = AffinityModuleConfig
    build_config_class = AffinityModuleBuildConfig

    def __init__(self, config: AffinityModuleConfig):
        super().__init__(config)
        self.mapping = self.config.mapping
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_group = self.mapping.tp_group

        num_dist_bins = self.config.num_dist_bins
        token_z = self.config.token_z
        token_s = self.config.token_s
        self.dtype = self.config.dtype
        eps = self.config.norm_epsilon
        inf = self.config.mask_inf
        pairformer_num_blocks = self.config.pairformer_num_blocks
        pairwise_head_width = self.config.pairwise_head_width
        pairwise_num_heads = self.config.pairwise_num_heads

        self.dist_bin_pairwise_embed = Embedding(num_embeddings=num_dist_bins,
                                                 embedding_dim=token_z,
                                                 dtype=self.dtype,
                                                 tp_group=self.tp_group,
                                                 tp_size=self.tp_size)

        self.fused_s_to_z = ColumnLinear(token_s,
                                         token_z * 2,
                                         bias=False,
                                         dtype=self.dtype,
                                         tp_group=self.tp_group,
                                         tp_size=self.tp_size,
                                         gather_output=False)

        self.z_norm = LayerNorm(normalized_shape=[token_z],
                                eps=eps,
                                dtype=self.dtype)
        self.z_linear = ColumnLinear(token_z,
                                     token_z,
                                     bias=False,
                                     dtype=self.dtype,
                                     tp_group=self.tp_group,
                                     tp_size=self.tp_size,
                                     gather_output=False)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
            eps=eps,
            dtype=self.dtype,
            mapping=self.mapping)

        self.pairformer_stack = PairformerNoSeqModule(
            num_blocks=pairformer_num_blocks,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            triangle_attn_backend=config.triangle_attention_backend,
            dtype=self.dtype,
            eps=eps,
            inf=inf,
            mapping=self.mapping)
        self.token_z = token_z // self.tp_size
        self.affinity_heads = AffinityHeadsTransformer(token_z=token_z,
                                                       token_s=token_s,
                                                       dtype=self.dtype,
                                                       eps=eps,
                                                       mapping=self.mapping)

    def forward(self,
                s: Tensor,
                z: Tensor,
                distogram: Tensor,
                cross_pair_mask_0: Tensor,
                cross_pair_mask_1: Tensor,
                multiplicity: int = 1,
                attention_params: Optional[AttentionParams] = None,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        """
        Args:
            s: (B, I, token_s)
            z: (B, I, I, token_z)
            distogram: (B, num_dist_bins, num_dist_bins)
            cross_pair_mask_0: (B, num_dist_bins, num_dist_bins)
            cross_pair_mask_1: (B, num_dist_bins, num_dist_bins, 1)
            multiplicity(int): default 1, unsupported for now (TODO: support multiplicity > 1)
        """
        assert multiplicity == 1, "Multiplicity > 1 is not supported yet"
        z = self.z_norm(z)
        z = self.z_linear(z)

        fused_s_to_z = self.fused_s_to_z(s)
        in1, in2 = split(fused_s_to_z, [self.token_z, self.token_z], dim=-1)
        z = z + in1.unsqueeze(2) + in2.unsqueeze(1)
        embed_distogram = self.dist_bin_pairwise_embed(distogram)
        if self.tp_size > 1:
            z = allgather(z, self.tp_group, gather_dim=-1)
        z = z + self.pairwise_conditioner(z_trunk=z,
                                          token_rel_pos_feats=embed_distogram)

        z = self.pairformer_stack(z,
                                  pair_mask=cross_pair_mask_0,
                                  attention_params=attention_params,
                                  all_reduce_params=all_reduce_params)
        pred_value, logits_binary = self.affinity_heads(
            z, cross_pair_mask_1, multiplicity=multiplicity)

        return pred_value, logits_binary
