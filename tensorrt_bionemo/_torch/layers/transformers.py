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
from tensorrt_llm.llmapi.utils import print_colored_debug

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.configs import (PairformerConfig,
                                                    TokenTransformerConfig)

from ..attention_backend import AttentionMetadata
from .attention import SelfAttentionPairBias, SelfAttentionPairBiasWithCache
from .linear import Linear, TensorParallelMode
from .normalization import AdaLN
from .transition import ConditionedTransitionBlock, Transition
from .triangle_nodes import (TriangleAttentionEndingNode,
                             TriangleAttentionStartingNode,
                             TriangleMultiplicationNode,
                             TriangleMultiplicationNodeType)


class PairformerLayerV1(nn.Module):

    def __init__(self,
                 layer_idx: int = 0,
                 token_s: int = 384,
                 token_z: int = 128,
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
                 max_tri_mul_tp_size: bool = True,
                 triangle_attn_node_chunk_size: int = 0,
                 mapping: Optional[Mapping] = None,
                 triangle_attn_backend: str = "VANILLA",
                 pairwise_attn_backend: str = "VANILLA",
                 skip_create_weights: bool = False,
                 attention_initial_norm: bool = False,
                 s_path_dtype: torch.dtype = None,
                 **kwargs):
        super().__init__()
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.token_s = token_s
        self.token_z = token_z
        self.mapping = mapping or Mapping()

        if s_path_dtype is None:
            s_path_dtype = dtype

        if not self.no_update_s:
            self.attention = SelfAttentionPairBias(
                layer_idx=layer_idx,
                c_s=token_s,
                c_z=token_z,
                num_heads=num_heads,
                dtype=s_path_dtype,
                bias_proj=True,
                eps=eps,
                inf=inf,
                max_attention_pairwise_tp_size=max_attention_pairwise_tp_size,
                mapping=mapping,
                skip_create_weights=skip_create_weights,
                attn_backend=pairwise_attn_backend,
                initial_norm=attention_initial_norm,
            )
        self.tri_mul_out = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            max_tri_mul_tp_size=max_tri_mul_tp_size,
        )
        self.tri_mul_in = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            max_tri_mul_tp_size=max_tri_mul_tp_size,
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
            attn_backend=triangle_attn_backend,
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
            attn_backend=triangle_attn_backend,
        )
        if not self.no_update_s:
            self.transition_s = Transition(
                token_s,
                token_s * 4,
                layer_idx=layer_idx,
                eps=eps,
                dtype=s_path_dtype,
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

    def _transform_z(
            self,
            z: torch.Tensor,
            pair_mask: torch.Tensor,
            attn_metadatas: Optional[dict[str, AttentionMetadata]] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        if z.dtype != pair_mask.dtype:
            z = z.to(pair_mask.dtype)
        z = z + self.tri_attn_start(
            z,
            mask=pair_mask,
            attn_metadata=attn_metadatas.get("triangle_attn"),
            all_reduce_params=all_reduce_params,
        )

        z = z + self.tri_attn_end(
            z,
            mask=pair_mask,
            attn_metadata=attn_metadatas.get("triangle_attn"),
            all_reduce_params=all_reduce_params,
        )

        z = z + self.transition_z(z)
        return z

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                mask: torch.Tensor,
                pair_mask: torch.Tensor,
                attn_metadatas: Optional[dict[str, AttentionMetadata]] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._transform_z(z, pair_mask, attn_metadatas, all_reduce_params)
        if not self.no_update_s:
            s = s + self.attention(
                s,
                z,
                mask,
                attn_metadata=attn_metadatas.get("pairwise_attn"),
                all_reduce_params=all_reduce_params)
            s = s + self.transition_s(s)
        return s, z


class PairformerNoSeqLayer(PairformerLayerV1):

    def __init__(self,
                 *,
                 layer_idx: int,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 **kwargs):
        kwargs["no_update_s"] = True
        super().__init__(layer_idx=layer_idx,
                         token_z=token_z,
                         pairwise_head_width=pairwise_head_width,
                         pairwise_num_heads=pairwise_num_heads,
                         **kwargs)

    def forward(self,
                z: torch.Tensor,
                pair_mask: torch.Tensor,
                attn_metadatas: Optional[dict[str, AttentionMetadata]] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> torch.Tensor:
        _, update_z = super().forward(s=None,
                                      z=z,
                                      mask=None,
                                      pair_mask=pair_mask,
                                      attn_metadatas=attn_metadatas,
                                      all_reduce_params=all_reduce_params)
        return update_z


class PairformerNoSeqModule(nn.Module):

    def __init__(self,
                 num_blocks: int = 8,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            PairformerNoSeqLayer(layer_idx=i,
                                 token_z=token_z,
                                 pairwise_head_width=pairwise_head_width,
                                 pairwise_num_heads=pairwise_num_heads,
                                 **kwargs) for i in range(num_blocks)
        ])

    def forward(self,
                z: torch.Tensor,
                pair_mask: torch.Tensor,
                attn_metadatas: Optional[dict[str, AttentionMetadata]] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, pair_mask, attn_metadatas, all_reduce_params)
        return z


class PairformerLayerV2(PairformerLayerV1):

    def __init__(self, post_layer_norm: bool = False, **kwargs):
        kwargs["s_path_dtype"] = torch.float32
        super().__init__(**kwargs)
        self.post_layer_norm = post_layer_norm
        self.pre_norm_s = nn.LayerNorm(self.token_s, dtype=torch.float32)
        self.post_norm_s = None
        if self.post_layer_norm:
            self.post_norm_s = nn.LayerNorm(self.token_s, dtype=torch.float32)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: Optional[dict[str, AttentionMetadata]] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._transform_z(z, pair_mask, attn_metadatas, all_reduce_params)
        original_dtype = s.dtype

        # v2 use float precision on the computing of s
        z = z.float()
        s = s.float()
        s_normed = self.pre_norm_s(s)
        s = s + self.attention(
            s_normed,
            z,
            mask,
            attn_metadata=attn_metadatas.get("pairwise_attn"),
            all_reduce_params=all_reduce_params)
        s = s + self.transition_s(s)
        if self.post_layer_norm:
            s = self.post_norm_s(s)
        s = s.to(original_dtype)
        z = z.to(original_dtype)
        return s, z


class PairformerModule(nn.Module):

    def __init__(self, config: PairformerConfig):
        super().__init__()
        self.config = config
        layer_cls = PairformerLayerV1 if config.version == "v1" else PairformerLayerV2
        self.layers = nn.ModuleList()
        for i in range(config.num_blocks):
            self.layers.append(
                layer_cls(
                    layer_idx=i,
                    token_s=config.token_s,
                    token_z=config.token_z,
                    num_heads=config.num_heads,
                    pairwise_head_width=config.pairwise_head_width,
                    pairwise_num_heads=config.pairwise_num_heads,
                    no_update_s=config.no_update_s,
                    no_update_z=config.no_update_z,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    max_transition_tp_size=config.max_transition_tp_size,
                    max_attention_pairwise_tp_size=config.
                    max_attention_pairwise_tp_size,
                    triangle_attn_node_chunk_size=config.
                    triangle_attn_node_chunk_size,
                    max_tri_mul_tp_size=config.max_tri_mul_tp_size,
                    mapping=config.mapping,
                    skip_create_weights=config.skip_create_weights,
                    triangle_attn_backend=config.triangle_attn_backend,
                    pairwise_attn_backend=config.pairwise_attn_backend,
                    post_layer_norm=config.post_layer_norm,
                    attention_initial_norm=config.attention_initial_norm,
                    s_path_dtype=config.s_path_dtype,
                ))

    def load_weights(self, weights: dict):
        loaded_weight = set()

        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                print_colored_debug(f"loading for: {name}")
                try:
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=weights[name])
                    else:
                        print_colored_debug(f" use copy_ to load {name}")
                        module_weights = weights[name][0]
                        for n, p in module._parameters.items():
                            if p is not None:
                                weight = module_weights[n][:]
                                if p.dtype != weight.dtype:
                                    weight = weight.to(p.dtype)
                                p.data.copy_(weight)

                except Exception as e:
                    raise e
            loaded_weight.add(name)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                mask: torch.Tensor,
                pair_mask: torch.Tensor,
                attn_metadatas: Optional[dict[str, AttentionMetadata]] = dict(),
                all_reduce_params: Optional[AllReduceParams] = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, attn_metadatas,
                         all_reduce_params)
        return s, z


class DiffusionTransformerLayer(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 dim: int = 384,
                 dim_single_cond: Optional[int] = None,
                 dim_pairwise: int = 128,
                 with_pair_bias_cache: bool = False,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 attention_initial_norm: bool = False,
                 post_layer_norm: bool = False,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        self.adaln = AdaLN(dim,
                           dim_single_cond,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping,
                           skip_create_weights=skip_create_weights)
        if with_pair_bias_cache:
            attn_cls = SelfAttentionPairBiasWithCache
        else:
            attn_cls = SelfAttentionPairBias
        self.pair_bias_attn = attn_cls(layer_idx=layer_idx,
                                       c_s=dim,
                                       c_z=dim_pairwise,
                                       num_heads=num_heads,
                                       initial_norm=attention_initial_norm,
                                       eps=eps,
                                       inf=inf,
                                       dtype=dtype,
                                       mapping=mapping,
                                       skip_create_weights=skip_create_weights)

        self.with_pair_bias_cache = with_pair_bias_cache

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
            skip_create_weights=skip_create_weights)
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
        if self.with_pair_bias_cache:
            b = self.pair_bias_attn(s=b,
                                    z=bias,
                                    mask=mask,
                                    attn_metadata=attn_metadata,
                                    all_reduce_params=all_reduce_params)
        else:
            b = self.pair_bias_attn(s=b,
                                    z=bias,
                                    mask=mask,
                                    compute_pair_bias=False,
                                    attn_metadata=attn_metadata,
                                    all_reduce_params=all_reduce_params)
        b = F.sigmoid(self.output_projection(s)) * b
        a = a + b
        a = a + self.transition(a, s, all_reduce_params=all_reduce_params)
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


class TokenTransformer(nn.Module):

    def __init__(self, config: TokenTransformerConfig):
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
                    with_pair_bias_cache=config.with_pair_bias_cache,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    attention_initial_norm=config.attention_initial_norm,
                    mapping=config.mapping,
                    skip_create_weights=config.skip_create_weights,
                ))

    def load_weights(self, weights: dict):
        loaded_weight = set()

        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                print_colored_debug(f"loading for: {name}")
                try:
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=weights[name])
                    else:
                        print_colored_debug(f" use copy_ to load {name}")
                        module_weights = weights[name][0]
                        for n, p in module._parameters.items():
                            if p is not None:
                                weight = module_weights[n][:]
                                if p.dtype != weight.dtype:
                                    weight = weight.to(p.dtype)
                                p.data.copy_(weight)

                except Exception as e:
                    raise e
            loaded_weight.add(name)
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
        if self.version == "v2":
            B, N, M, D = z.shape
            L = self.num_blocks
            z = z.view(B, N, M, L, D // L)
            z = z.permute(0, 4, 1, 2, 3)  # [B, H, N, N, L]
        bias = z
        for i, layer in enumerate(self.layers):
            if self.version == "v2":
                bias = z[..., i]
            a = layer(a, s, bias, mask, attn_metadata, all_reduce_params)
        return a
