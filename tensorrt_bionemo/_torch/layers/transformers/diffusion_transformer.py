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
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.attention import AttentionPairBias
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo._torch.layers.transition import ConditionedTransitionBlock
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.config import PretrainedModuleConfig
from tensorrt_bionemo.mapping import Mapping


class DiffusionTransformerLayer(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 dim: int = 384,
                 dim_single_cond: Optional[int] = None,
                 dim_pairwise: int = 128,
                 bias_proj: bool = False,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 attention_initial_norm: bool = False,
                 post_layer_norm: bool = False,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 conditioned_transition_using_silu: bool = False):
        super().__init__()
        self.adaln = AdaLN(dim,
                           dim_single_cond,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping,
                           skip_create_weights=skip_create_weights)

        self.pair_bias_attn = AttentionPairBias(
            layer_idx=layer_idx,
            c_s=dim,
            c_z=dim_pairwise,
            num_heads=num_heads,
            initial_norm=attention_initial_norm,
            bias_proj=bias_proj,
            eps=eps,
            inf=inf,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights)

        self.output_projection = Linear(
            dim_single_cond,
            dim,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)
        self.transition = ConditionedTransitionBlock(
            dim_single=dim,
            dim_single_cond=dim_single_cond,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            using_silu=conditioned_transition_using_silu)
        self.post_lnorm = None
        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim, dtype=dtype, eps=eps)

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                bias: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> torch.Tensor:
        """ First version of DiffusionTransformerLayer, does not support multiplicity > 1 and atom encoder, decoder"""

        b = self.adaln(a, s)
        b = self.pair_bias_attn(s=b,
                                z=bias,
                                mask=mask,
                                attn_metadata=attn_metadata,
                                all_reduce_params=all_reduce_params)
        b = F.sigmoid(self.output_projection(s)) * b
        a = a + b
        a = a + self.transition(a, s, all_reduce_params=all_reduce_params)
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


class BoltzDiffusionTransformer(nn.Module):

    def __init__(self, config: PretrainedModuleConfig):
        """
        Args:
            config: tensorrt_bionemo.models.boltz1.configs.DiffusionTransformerConfig
                The configuration of the token transformer module.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.version = config.version
        self.num_blocks = config.num_blocks
        for i in range(config.num_blocks):
            self.layers.append(
                DiffusionTransformerLayer(
                    layer_idx=i,
                    num_heads=config.num_heads,
                    dim=config.dim,
                    dim_single_cond=config.dim_single_cond,
                    dim_pairwise=config.dim_pairwise,
                    post_layer_norm=config.post_layer_norm,
                    bias_proj=False,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    attention_initial_norm=config.attention_initial_norm,
                    mapping=config.mapping,
                    skip_create_weights=config.skip_create_weights,
                ))

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self,
                a: torch.Tensor = None,
                s: torch.Tensor = None,
                z: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> torch.Tensor:
        L = self.num_blocks
        # Transformer z -> [*, heads, N, N, L]
        N, M, D = z.shape[-3:]
        heads = D // L
        batch_dims = z.shape[:-3]
        z = z.view(*batch_dims, N, M, L, heads)  # [*, N, N, L, heads]
        z = torch.moveaxis(z, -1, -4)  # [*, heads, N, N, L]

        for i, layer in enumerate(self.layers):
            bias = z[..., i]
            a = layer(a, s, bias, mask, attn_metadata, all_reduce_params)
        return a


class OpenFold3DiffusionTransformer(nn.Module):

    def __init__(self, config: PretrainedModuleConfig):
        """
        Args:
            config: tensorrt_bionemo.models.boltz1.configs.DiffusionTransformerConfig
                The configuration of the token transformer module.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.version = config.version
        self.num_blocks = config.num_blocks
        for i in range(config.num_blocks):
            layer = DiffusionTransformerLayer(
                layer_idx=i,
                num_heads=config.num_heads,
                dim=config.dim,
                dim_single_cond=config.dim_single_cond,
                dim_pairwise=config.dim_pairwise,
                post_layer_norm=config.post_layer_norm,
                bias_proj=True,
                dtype=config.torch_dtype,
                eps=config.norm_epsilon,
                inf=config.mask_inf,
                attention_initial_norm=config.attention_initial_norm,
                mapping=config.mapping,
                skip_create_weights=config.skip_create_weights,
                conditioned_transition_using_silu=True,
            )
            dim = layer.pair_bias_attn.proj_z[0].weight.shape
            eps = layer.pair_bias_attn.proj_z[0].eps
            new_layer = nn.LayerNorm(dim, bias=False, eps=eps)
            layer.pair_bias_attn.proj_z[0] = new_layer
            self.layers.append(layer)

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self,
                a: torch.Tensor = None,
                s: torch.Tensor = None,
                z: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> torch.Tensor:

        for layer in self.layers:
            a = layer(a, s, z, mask, attn_metadata, all_reduce_params)
        return a
