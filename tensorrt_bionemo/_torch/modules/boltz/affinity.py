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

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.distributed import (
    AllReduceParams, get_default_tp_group_coordinator)
from tensorrt_bionemo._torch.layers.conditioning import PairwiseConditioning
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerNoSeqModule
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping


def get_best_coords(coords: torch.Tensor, iptm: torch.Tensor) -> torch.Tensor:
    """
    Get the best coordinates from the predicted coordinates.
    Args:
        coords: (B, I, 3)
        iptm: (B, I)
    Returns:
        best_coords: (B, 3)
    """
    argsort = torch.argsort(iptm, descending=True, dim=1)
    best_idx = argsort[:, 0]
    best_coords = coords[torch.arange(len(best_idx)), best_idx]

    return best_coords


def create_cross_pair_mask(
        token_pad_mask: torch.Tensor,
        mol_type: torch.Tensor,
        affinity_token_mask: torch.Tensor,
        include_mask_for_head: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a cross-pair mask for affinity prediction.
    """
    rec_mask = ((mol_type == 0))
    rec_mask = rec_mask * token_pad_mask
    lig_mask = (affinity_token_mask.to(torch.bool)) * token_pad_mask
    lig_mask = lig_mask * token_pad_mask
    cross_pair_mask = (lig_mask[:, :, None] * rec_mask[:, None, :] +
                       rec_mask[:, :, None] * lig_mask[:, None, :] +
                       lig_mask[:, :, None] * lig_mask[:, None, :])
    mask_for_head = None
    if include_mask_for_head:
        mask_for_head = cross_pair_mask.unsqueeze(-1) * (
            1 - torch.eye(lig_mask.shape[1],
                          device=lig_mask.device).unsqueeze(-1).unsqueeze(0))
    return cross_pair_mask, mask_for_head


def compute_distogram(x_pred: torch.Tensor,
                      boundaries: torch.Tensor,
                      token_to_rep_atom: torch.Tensor,
                      multiplicity: int = 1,
                      dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """
    Compute the distogram from the predicted atom coordinates.
    Args:
        x_pred: (B, mult, N, 3)
        boundaries: (num_dist_bins - 1,)
        token_to_rep_atom: (B, I, n_atoms)
        multiplicity: int
    Returns:
        distogram: (B, num_dist_bins, num_dist_bins)
    """
    if len(x_pred.shape) == 4:
        B, mult, N, _ = x_pred.shape
        x_pred = x_pred.reshape(B * mult, N, -1)
    else:
        BM, N, _ = x_pred.shape
        B = BM // multiplicity
        mult = multiplicity
    x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
    d = torch.cdist(x_pred_repr, x_pred_repr)
    distogram = (d.unsqueeze(-1) > boundaries).sum(dim=-1).to(dtype)
    return distogram


class AffinityHeadsTransformer(nn.Module):

    def __init__(self,
                 token_z: int,
                 token_s: int,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        mapping = mapping or Mapping()
        self.token_z = token_z
        self.token_s = token_s
        self.dtype = dtype
        self.eps = eps
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group

        self.affinity_out_mlp_linear_0 = Linear(
            token_z,
            token_z,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights)

        self.affinity_out_mlp_linear_1 = Linear(
            token_z,
            token_s,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            skip_create_weights=skip_create_weights)

        self.to_affinity_pred_value_0 = Linear(
            token_s,
            token_s,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights)

        self.to_affinity_pred_value_1 = Linear(
            token_s,
            token_s,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            skip_create_weights=skip_create_weights)
        self.to_affinity_pred_value_2 = nn.Linear(token_s,
                                                  1,
                                                  bias=True,
                                                  dtype=dtype)

        self.to_affinity_pred_score_0 = Linear(
            token_s,
            token_s,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights)

        self.to_affinity_pred_score_1 = Linear(
            token_s,
            token_s,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            skip_create_weights=skip_create_weights)
        self.to_affinity_pred_score_2 = nn.Linear(token_s,
                                                  1,
                                                  bias=True,
                                                  dtype=dtype)

        self.to_affinity_logits_binary = nn.Linear(1,
                                                   1,
                                                   bias=True,
                                                   dtype=dtype)

    def forward(self, z: torch.Tensor,
                cross_pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, I, token_s)
            cross_pair_mask: (B, num_dist_bins, num_dist_bins, 1)

        Returns:
            pred_value: (batch_size, 1)
            logits_binary: (batch_size, 1)
            affinity_embedding: (batch_size, token_s) pooled representation
                after ``affinity_out_mlp`` (matches OSS boltz
                ``AffinityHeadsTransformer``).
        """
        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7)
        g = self.affinity_out_mlp_linear_0(g)
        g = F.relu(g)
        g = self.affinity_out_mlp_linear_1(g)
        affinity_embedding = F.relu(g)

        pred_value = self.to_affinity_pred_value_0(affinity_embedding)
        pred_value = F.relu(pred_value)
        pred_value = self.to_affinity_pred_value_1(pred_value)
        pred_value = F.relu(pred_value)
        pred_value = self.to_affinity_pred_value_2(pred_value)

        pred_score = self.to_affinity_pred_score_0(affinity_embedding)
        pred_score = F.relu(pred_score)
        pred_score = self.to_affinity_pred_score_1(pred_score)
        pred_score = F.relu(pred_score)
        pred_score = self.to_affinity_pred_score_2(pred_score)

        logits_binary = self.to_affinity_logits_binary(pred_score)

        return pred_value, logits_binary, affinity_embedding


class AffinityModule(nn.Module):

    def __init__(self, config: BaseConfig):
        """ Boltz-2 Affinity Module
        Args:
            config: tensorrt_bionemo.models.boltz2.configs.AffinityModuleConfig
                The configuration of the affinity module.
        """
        super().__init__()
        self.config = config
        self.mapping = config.mapping
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_group = self.mapping.tp_group

        skip_create_weights = False
        self.dist_bin_pairwise_embed = nn.Embedding(
            num_embeddings=config.num_dist_bins,
            embedding_dim=config.token_z,
            dtype=config.torch_dtype)

        self.fused_s_to_z = Linear(
            config.token_s,
            config.token_z * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR))
        self.z_norm = nn.LayerNorm(config.token_z,
                                   eps=config.norm_epsilon,
                                   dtype=config.torch_dtype)
        self.z_linear = Linear(config.token_z,
                               config.token_z,
                               bias=False,
                               dtype=config.torch_dtype,
                               mapping=config.mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=False,
                               skip_create_weights=skip_create_weights)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=config.token_z,
            dim_token_rel_pos_feats=config.token_z,
            num_transitions=2,
            eps=config.norm_epsilon,
            dtype=config.torch_dtype,
            mapping=config.mapping)

        # Affinity ``cross_pair_mask`` is bipartite (receptor rows have
        # interior 1's in the ligand-column range, not a left-aligned
        # prefix), so the trimul x_x dual GEMM must avoid the CuTeDSL LM
        # kernel -- it masks via a per-row prefix count and would
        # silently produce wrong outputs. ``pair_mask_left_aligned=False``
        # routes the dispatcher to cuEquiv / CUTLASS instead, which
        # consume the full ``mask`` tensor and handle arbitrary masks.
        self.pairformer_stack = PairformerNoSeqModule(
            num_blocks=config.pairformer_num_blocks,
            token_z=config.token_z,
            pairwise_head_width=config.pairwise_head_width,
            pairwise_num_heads=config.pairwise_num_heads,
            dtype=config.torch_dtype,
            eps=config.norm_epsilon,
            inf=config.mask_inf,
            mapping=config.mapping,
            triangle_attn_backend=config.triangle_attention_backend,
            pair_mask_left_aligned=False)

        self.affinity_heads = AffinityHeadsTransformer(
            token_z=config.token_z,
            token_s=config.token_s,
            dtype=config.torch_dtype,
            eps=config.norm_epsilon,
            mapping=config.mapping,
            skip_create_weights=skip_create_weights)

        self.token_z = config.token_z // self.tp_size

        self.tp_group_comm = None
        if self.tp_size > 1:
            self.tp_group_comm = get_default_tp_group_coordinator()
            assert self.tp_group_comm is not None, "Failed to get the default TP group coordinator"

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        distogram: torch.Tensor,
        cross_pair_mask_0: torch.Tensor,
        cross_pair_mask_1: torch.Tensor,
        attn_metadatas: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s: (B, I, token_s)
            z: (B, I, I, token_z)
            distogram: (B, num_dist_bins, num_dist_bins)
            cross_pair_mask_0: (B, num_dist_bins, num_dist_bins)
            cross_pair_mask_1: (B, num_dist_bins, num_dist_bins, 1)

        Returns:
            pred_value: (B, 1)
            logits_binary: (B, 1)
            affinity_embedding: (B, token_s)
        """
        assert len(s.shape) == 3, "s must be (B, I, token_s)"
        assert len(z.shape) == 4, "z must be (B, I, I, token_z)"

        z = self.z_norm(z)
        z = self.z_linear(z)
        fused_s_to_z = self.fused_s_to_z(s)
        in1, in2 = fused_s_to_z.split([self.token_z, self.token_z], dim=-1)
        z = z + in1.unsqueeze(2) + in2.unsqueeze(1)

        embed_distogram = self.dist_bin_pairwise_embed(distogram)

        if self.tp_size > 1:
            z = self.tp_group_comm.all_gather(z, dim=-1)
        z = z + self.pairwise_conditioner(z_trunk=z,
                                          token_rel_pos_feats=embed_distogram)
        z = self.pairformer_stack(z,
                                  pair_mask=cross_pair_mask_0,
                                  attn_metadatas=attn_metadatas,
                                  all_reduce_params=all_reduce_params)
        pred_value, logits_binary, affinity_embedding = self.affinity_heads(
            z, cross_pair_mask_1)

        return pred_value, logits_binary, affinity_embedding
