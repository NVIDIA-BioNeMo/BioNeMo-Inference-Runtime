# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.utils import (
    PrecomputedPairMasks, precompute_pair_masks)
from tensorrt_bionemo._torch.auto_chunk import (CHUNK_REGISTRY, MSA_TRANSITION,
                                                PAIR_TRANSITION)
from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging
from tensorrt_bionemo._torch.layers.transformers.evoformer import \
    EvoformerBlock
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig


class MSAModuleBlock(EvoformerBlock):

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
                 opm_first: bool,
                 support_batch: bool = True,
                 triangle_attn_backend: str = 'VANILLA',
                 trimul_high_precision: bool = False,
                 msa_att_row_chunk_size: Optional[int] = None,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 skip_create_weights: bool = False,
                 outer_product_mean_bias: Optional[dict] = None,
                 tri_mul_out_bias: Optional[dict] = None,
                 tri_mul_in_bias: Optional[dict] = None,
                 tri_attn_start_bias: Optional[dict] = None,
                 tri_attn_end_bias: Optional[dict] = None,
                 last_block: bool = False,
                 **kwargs):
        super().__init__(
            local_layer_idx=local_layer_idx,
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            opm_first=opm_first,
            support_batch=support_batch,
            no_column_attention=True,
            triangle_attn_backend=triangle_attn_backend,
            trimul_high_precision=trimul_high_precision,
            dtype=dtype,
            eps=eps,
            inf=inf,
            skip_create_weights=skip_create_weights,
            outer_product_mean_bias=outer_product_mean_bias,
            tri_mul_out_bias=tri_mul_out_bias,
            tri_mul_in_bias=tri_mul_in_bias,
            tri_attn_start_bias=tri_attn_start_bias,
            tri_attn_end_bias=tri_attn_end_bias,
        )

        if not last_block:
            if hasattr(self, 'msa_att_row'):
                del self.msa_att_row

            if hasattr(self, 'msa_transition'):
                del self.msa_transition

            self.msa_att_row = PairWeightedAveraging(
                c_m=c_m,
                c_z=c_z,
                c_h=c_hidden_msa_att,
                num_heads=no_heads_msa,
                inf=inf,
                eps=eps,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
            # SwiGLU Transition — chunk along MSA rows (dim S of [B,S,N,C_m]),
            # matching OSS MSAStack.msa_chunk_size=2048 so deep MSAs fit.
            self.msa_transition = Transition(
                dim=c_m,
                hidden=c_m * transition_n,
                layer_idx=local_layer_idx,
                eps=eps,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
                auto_chunk_policy=CHUNK_REGISTRY.get(MSA_TRANSITION),
            )
        else:
            self.msa_att_row = None
            self.msa_transition = None

        # Row-chunk the [B, N, N, c_z * transition_n] SwiGLU hidden activation on
        # the first pair axis (numerically identical, position-wise op). Uses the
        # same registry policy as the trunk Pairformer's ``transition_z``; the dense
        # fast path is kept below the memory-scaled auto-chunk threshold.
        self.pair_transition = Transition(
            dim=c_z,
            hidden=c_z * transition_n,
            layer_idx=local_layer_idx,
            eps=eps,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            auto_chunk_policy=CHUNK_REGISTRY.get(PAIR_TRANSITION),
        )

        self.msa_att_row_chunk_size = msa_att_row_chunk_size

    def _compute_opm(
            self, m: torch.Tensor, z: torch.Tensor,
            msa_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        opm = self.outer_product_mean(m, mask=msa_mask)
        z = z + opm
        return m, z

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        precomputed_masks: Optional[PrecomputedPairMasks] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            precomputed_masks:
                Optional precomputed mask biases for triangle attention.
        """
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)

        if self.msa_att_row is not None:
            m = m + self.msa_att_row(m, z, pair_mask)

        msa_trans_mask = msa_mask

        if self.msa_transition is not None:
            m = m + self.msa_transition(m, mask=msa_trans_mask.unsqueeze(-1))

        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        if precomputed_masks is not None:
            z = z + self.tri_attn_start(z,
                                        mask_bias=precomputed_masks.mask_bias,
                                        attn_metadata=attn_metadata)
            z = z + self.tri_attn_end(
                z,
                mask_bias=precomputed_masks.mask_bias_transposed,
                attn_metadata=attn_metadata)
        else:
            z = z + self.tri_attn_start(
                z, mask=pair_mask, attn_metadata=attn_metadata)
            z = z + self.tri_attn_end(
                z, mask=pair_mask, attn_metadata=attn_metadata)

        pair_trans_mask = pair_mask
        z = z + self.pair_transition(z, mask=pair_trans_mask.unsqueeze(-1))

        return m, z


class MSAModuleStack(nn.Module):

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        self.num_blocks = config.no_blocks
        for i in range(self.num_blocks):
            self.blocks.append(
                MSAModuleBlock(
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
                    opm_first=config.opm_first,
                    support_batch=config.support_batch,
                    triangle_attn_backend=config.triangle_attention_backend,
                    trimul_high_precision=config.trimul_high_precision,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    skip_create_weights=config.skip_create_weights,
                    msa_att_row_chunk_size=config.msa_att_row_chunk_size,
                    outer_product_mean_bias={
                        "proj_a": False,
                        "proj_b": False,
                        "proj_o": True
                    },
                    tri_mul_out_bias={
                        "p_in": False,
                        "g_in": False,
                        "p_out": False,
                        "g_out": False
                    },
                    tri_mul_in_bias={
                        "p_in": False,
                        "g_in": False,
                        "p_out": False,
                        "g_out": False
                    },
                    tri_attn_start_bias={
                        "q": False,
                        "k": False,
                        "v": False,
                        "g": False,
                        "z": False,
                        "o": False
                    },
                    tri_attn_end_bias={
                        "q": False,
                        "k": False,
                        "v": False,
                        "g": False,
                        "z": False,
                        "o": False
                    },
                    last_block=True if i == self.num_blocks - 1 else False))

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        precomputed_masks: Optional[PrecomputedPairMasks] = None
    ) -> torch.Tensor:
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
            attn_metadata:
                Attention metadata
            precomputed_masks:
                Optional precomputed triangle-attention mask bias. When given
                (e.g. precomputed once by a recycling trunk), it is reused
                instead of recomputing from ``pair_mask`` on every call.
        """
        # Expand the batch dimensions if needed
        n_dims = m.ndim
        if n_dims == 3:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)
            msa_mask = msa_mask.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0)

        m = m.to(dtype=self.config.torch_dtype)
        z = z.to(dtype=self.config.torch_dtype)
        msa_mask = msa_mask.to(dtype=self.config.torch_dtype)
        pair_mask = pair_mask.to(dtype=self.config.torch_dtype)

        precomputed = precomputed_masks if precomputed_masks is not None else \
            precompute_pair_masks(
                self.blocks[0].triangle_attn_backend,
                pair_mask,
                inf=self.blocks[0].inf,
                dtype=self.blocks[0].dtype,
            )

        for block in self.blocks:
            m, z = block(m,
                         z,
                         msa_mask,
                         pair_mask,
                         attn_metadata,
                         precomputed_masks=precomputed)
        if n_dims == 3:
            m = m.squeeze(0)
            z = z.squeeze(0)
        return z
