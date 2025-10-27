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
from tensorrt_llm.llmapi.utils import print_colored_debug

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.outer_product_mean import OuterProductMean
from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerNoSeqLayer
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo.config import PretrainedModuleConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.const import POCKET_CONTACT_INFO


class MSALayer(nn.Module):

    def __init__(self,
                 msa_s: int,
                 token_z: int,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 layer_idx: int = 0,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 triangle_attn_backend: str = "VANILLA",
                 mapping: Optional[Mapping] = None) -> None:
        super().__init__()
        self.msa_s = msa_s
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.eps = eps
        self.inf = inf
        self.dtype = dtype

        self.mapping = mapping or Mapping()

        self.msa_transition = Transition(
            dim=msa_s,
            hidden=msa_s * 4,
            layer_idx=layer_idx,
            eps=eps,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            mapping=mapping)

        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            mapping=mapping)

        self.pairformer_layer = PairformerNoSeqLayer(
            layer_idx=layer_idx,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            triangle_attn_backend=triangle_attn_backend,
            mapping=mapping)
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
            eps=eps,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            mapping=mapping)

    def forward(
        self,
        z: torch.Tensor,
        m: torch.Tensor,
        token_mask: torch.Tensor,
        msa_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z(Tensor): The input tensor of shape (B, N, N, token_z).
            m(Tensor): The input tensor of shape (B, S, N, msa_s).
            token_mask(Tensor): The mask tensor of shape (B, N, N).
            msa_mask(Tensor): The mask tensor of shape (B, S, N).
        Returns:
            Tuple[Tensor, Tensor]: The output tensor of shape (B, N, N, token_z), (B, S, N, msa_s).
        """
        m = m + self.pair_weighted_averaging(m, z, token_mask,
                                             all_reduce_params)
        m = m + self.msa_transition(m, all_reduce_params)
        z = z + self.outer_product_mean(m, msa_mask, all_reduce_params)

        # Compute pairwise stack
        z = self.pairformer_layer(
            z,
            token_mask,
            attn_metadatas={"triangle_attn": attn_metadata},
            all_reduce_params=all_reduce_params)
        return z, m


class MSAModule(nn.Module):

    def __init__(self, config: PretrainedModuleConfig) -> None:
        """
        Boltz MSAModule
        TODO: add support for subsampling, chunking
        """
        super().__init__()

        self.mapping = config.mapping or Mapping()
        self.msa_s = config.msa_s
        self.token_z = config.token_z
        self.token_s = config.token_s
        self.msa_blocks = config.msa_blocks
        self.num_tokens = config.num_tokens
        self.pairwise_head_width = config.pairwise_head_width
        self.pairwise_num_heads = config.pairwise_num_heads
        self.use_paired_feature = config.use_paired_feature
        self.dtype = config.torch_dtype
        self.version = config.version

        if config.version == "v1":
            s_input_dim = self.token_s + 2 * self.num_tokens + 1 + len(
                POCKET_CONTACT_INFO)
        else:
            s_input_dim = self.token_s
        self.s_proj = Linear(s_input_dim,
                             self.msa_s,
                             bias=False,
                             dtype=self.dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=config.skip_create_weights)
        self.msa_proj = Linear(self.num_tokens + 2 +
                               int(self.use_paired_feature),
                               self.msa_s,
                               bias=False,
                               dtype=self.dtype,
                               mapping=self.mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=config.skip_create_weights)

        self.layers = nn.ModuleList()
        for layer_idx in range(self.msa_blocks):
            self.layers.append(
                MSALayer(msa_s=self.msa_s,
                         token_z=self.token_z,
                         layer_idx=layer_idx,
                         pairwise_head_width=self.pairwise_head_width,
                         pairwise_num_heads=self.pairwise_num_heads,
                         eps=config.norm_epsilon,
                         inf=config.mask_inf,
                         dtype=self.dtype,
                         skip_create_weights=config.skip_create_weights,
                         triangle_attn_backend=config.triangle_attn_backend,
                         mapping=self.mapping))

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
                    print(name)
                    raise e
            loaded_weight.add(name)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

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
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            z(Tensor): The input tensor of shape (B, N, N, token_z).
            emb(Tensor): The input tensor of shape (B, N, token_s).
            msa(Tensor): The input tensor of shape (B, N_msa, N).
            has_deletion(Tensor): The input tensor of shape (B, N_msa, N).
            deletion_value(Tensor): The input tensor of shape (B, N_msa, N).
            msa_paired(Tensor): The input tensor of shape (B, N_msa, N).
            msa_mask(Tensor): The input tensor of shape (B, N_msa, N).
            token_pad_mask(Tensor): The input tensor of shape (B, N).
            attn_metadata(Optional[AttentionMetadata]): The attention metadata.
            all_reduce_params(Optional[AllReduceParams]): The all reduce parameters.
        Returns:
            Tensor: The output tensor of shape (B, N, N, token_z).
        """
        if self.version == "v2":
            msa = torch.nn.functional.one_hot(msa, num_classes=self.num_tokens)
        msa = msa.to(self.dtype)
        has_deletion = has_deletion.unsqueeze(-1)
        deletion_value = deletion_value.unsqueeze(-1)
        is_paired = msa_paired.unsqueeze(-1)
        token_mask = token_pad_mask.to(self.dtype)
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired],
                          dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        for i in range(self.msa_blocks):
            z, m = self.layers[i](z, m, token_mask, msa_mask, attn_metadata,
                                  all_reduce_params)
        return z
