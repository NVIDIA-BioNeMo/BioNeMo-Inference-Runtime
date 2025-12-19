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

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.layers.attention import MSAColumnGlobalAttention
from tensorrt_bionemo._torch.layers.transformers.evoformer import \
    EvoformerBlock
from tensorrt_bionemo._torch.layers.transformers.evoformer import \
    EvoformerStack as _EvoformerStack
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping


class EvoformerStack(_EvoformerStack):
    """ Overriding the forward method to support batch dimension. OF2 may pass tensors without batch dimension. """

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_dims = m.ndim
        if n_dims == 3:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)
            msa_mask = msa_mask.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0)

        m, z, s = super().forward(m, z, msa_mask, pair_mask, attn_metadata,
                                  all_reduce_params)
        if n_dims == 3:
            m = m.squeeze(0)
            z = z.squeeze(0)
            s = s.squeeze(0)
        return m, z, s


class ExtraMSABlock(EvoformerBlock):

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
                 opm_chunk_size: Optional[int] = None,
                 opm_mask_chunk_size: Optional[int] = None,
                 dtype: torch.dtype = None,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None,
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
            opm_chunk_size=opm_chunk_size,
            opm_mask_chunk_size=opm_mask_chunk_size,
            dtype=dtype,
            eps=eps,
            inf=inf,
            skip_create_weights=skip_create_weights,
            mapping=mapping,
        )
        self.msa_att_col = MSAColumnGlobalAttention(
            local_layer_idx=local_layer_idx,
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            attn_bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "o": True,
            },
            eps=eps,
            inf=inf,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights)

    def _compute_opm(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opm = self.outer_product_mean(m,
                                      mask=msa_mask,
                                      all_reduce_params=all_reduce_params)
        z = z + opm
        return m, z

    def forward(
            self,
            m: torch.Tensor,
            z: torch.Tensor,
            msa_mask: torch.Tensor,
            pair_mask: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
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
        """
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask, all_reduce_params)
        m = m + self.msa_att_row(m,
                                 z,
                                 mask=msa_mask,
                                 attn_metadata=attn_metadata,
                                 all_reduce_params=all_reduce_params)
        m = m + self.msa_att_col(
            m, mask=msa_mask, all_reduce_params=all_reduce_params)
        msa_trans_mask = msa_mask
        m = m + self.msa_transition(m, mask=msa_trans_mask)

        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask, all_reduce_params)
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_attn_start(z,
                                    mask=pair_mask,
                                    attn_metadata=attn_metadata,
                                    all_reduce_params=all_reduce_params)
        z = z + self.tri_attn_end(z,
                                  mask=pair_mask,
                                  attn_metadata=attn_metadata,
                                  all_reduce_params=all_reduce_params)
        pair_trans_mask = pair_mask
        z = z + self.pair_transition(z, mask=pair_trans_mask)

        return m, z


class ExtraMSAStack(nn.Module):

    def __init__(self, config: BaseConfig) -> None:
        """
        OpenFold2 ExtraMSAModule
        TODO: add support for subsampling, chunking
        Args:
            config: tensorrt_bionemo.models.openfold2.configs.ExtraMSAModuleConfig
                The configuration of the extra msa stack module.
        """
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        self.num_blocks = config.no_blocks
        for i in range(self.num_blocks):
            self.blocks.append(
                ExtraMSABlock(
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
                    opm_chunk_size=config.opm_chunk_size,
                    opm_mask_chunk_size=config.opm_mask_chunk_size,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    skip_create_weights=config.skip_create_weights,
                    mapping=config.mapping,
                ))

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
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
        all_reduce_params: Optional[AllReduceParams] = None
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
            attn_metadata:
                Attention metadata
            all_reduce_params:
                AllReduce parameters
        """
        # Expand the batch dimensions if needed
        n_dims = m.ndim
        if n_dims == 3:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)
            msa_mask = msa_mask.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0)

        for block in self.blocks:
            m, z = block(m, z, msa_mask, pair_mask, attn_metadata,
                         all_reduce_params)
        if n_dims == 3:
            m = m.squeeze(0)
            z = z.squeeze(0)
        return z
