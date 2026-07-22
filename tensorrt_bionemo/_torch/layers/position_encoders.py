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

from math import pi
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo.mapping import Mapping


class FourierEmbedding(nn.Module):
    """Fourier embedding layer."""

    def __init__(self,
                 dim,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None):
        """Initialize the Fourier Embeddings.

        Args:
            dim : int
                The dimension of the embeddings.
            dtype: torch.dtype
                The data type of the input features.
            mapping: Optional[Mapping]
                The mapping of the input features.
        """
        super().__init__()
        self.proj = Linear(1,
                           dim,
                           bias=True,
                           dtype=dtype,
                           mapping=mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=True,
                           skip_create_weights=False)

    def forward(
        self,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            times : torch.Tensor
                Shape [B, multiplicity]. The times of the input features.
        Returns:
            torch.Tensor
                Shape [B, multiplicity, dim_fourier]. The Fourier embedded times.
        """
        times = times.unsqueeze(-1)
        rand_proj = self.proj(times)
        return rand_proj.mul_(2 * pi).cos_()


class RelativePositionEncoder(nn.Module):
    """AF3 relative-position encoder.

    Accepts raw token features or a precomputed ``relp``. The Protenix variant
    uses ``fix_sym_check=True`` and ``cyclic_pos_enc=False``.
    """

    def __init__(
            self,
            token_z: int,
            r_max: int = 32,
            s_max: int = 2,
            fix_sym_check: bool = False,
            cyclic_pos_enc: bool = True,
            period_broadcast: bool = True,  # Set False for Boltz2
            dtype: torch.dtype = torch.float32,
            mapping: Optional[Mapping] = None,
            skip_create_weights: bool = False):
        """Initialize the relative position encoder.

        Args:
            token_z: int
                The pair representation dimension.
            r_max: int, optional
                The maximum index distance, by default 32.
            s_max: int, optional
                The maximum chain distance, by default 2.
            fix_sym_check: bool
                Whether to fix the sym_check.
            cyclic_pos_enc: bool
                Whether to use cyclic positional encoding.
            period_broadcast: bool
                Whether to broadcast the period.
            dtype: torch.dtype
                The data type of the input features.
            mapping: Optional[Mapping]
                The mapping of the input features.
            skip_create_weights: bool
                Whether to skip creating weights.
        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear = Linear(4 * (r_max + 1) + 2 * (s_max + 1) + 1,
                             token_z,
                             bias=False,
                             dtype=dtype,
                             mapping=mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=skip_create_weights)
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc
        self.period_broadcast = period_broadcast
        self.mapping = mapping or Mapping()

    def _relp_buckets(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        entity_id: torch.Tensor,
        token_index: torch.Tensor,
        sym_id: torch.Tensor,
        cyclic_period: Optional[torch.Tensor] = None,
    ):
        """Bucket pairwise residue, token, and symmetry-chain offsets.

        Args:
            asym_id / residue_index / entity_id / token_index / sym_id:
                integer token features, each ``[B, N_token]``.
            cyclic_period: optional ``[B, N_token]`` cyclic period.

        Returns:
            ``(d_residue, d_token, d_chain, b_same_entity)``, each
            ``[B, N, N]``.
        """
        b_same_chain = torch.eq(asym_id[:, :, None], asym_id[:, None, :])
        b_same_residue = torch.eq(residue_index[:, :, None],
                                  residue_index[:, None, :])
        b_same_entity = torch.eq(entity_id[:, :, None], entity_id[:, None, :])
        d_residue = (residue_index[:, :, None] - residue_index[:, None, :])
        if (self.cyclic_pos_enc and cyclic_period is not None
                and torch.any(cyclic_period > 0)):
            period = torch.where(
                cyclic_period > 0,
                cyclic_period,
                torch.zeros_like(cyclic_period) + 10000,
            )
            if self.period_broadcast:
                period = period.unsqueeze(1)
            d_residue = (d_residue -
                         period * torch.round(d_residue / period)).long()

        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )

        d_residue = torch.where(
            b_same_chain, d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1)

        d_token = torch.clip(
            token_index[:, :, None] - token_index[:, None, :] + self.r_max,
            0,
            2 * self.r_max,
        )

        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )

        d_chain = torch.clip(
            sym_id[:, :, None] - sym_id[:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        b_same_chain = (~b_same_entity) if self.fix_sym_check else b_same_chain
        d_chain = torch.where(b_same_chain,
                              torch.zeros_like(d_chain) + 2 * self.s_max + 1,
                              d_chain)
        return d_residue, d_token, d_chain, b_same_entity

    def generate_relp(
            self,
            asym_id: torch.Tensor,
            residue_index: torch.Tensor,
            entity_id: torch.Tensor,
            token_index: torch.Tensor,
            sym_id: torch.Tensor,
            cyclic_period: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Materialize the relative-position one-hot feature.

        Args:
            asym_id / residue_index / entity_id / token_index / sym_id:
                integer token features, each ``[B, N_token]``.
            cyclic_period: optional ``[B, N_token]`` cyclic period.

        Returns:
            ``relp`` ``[B, N_token, N_token, 4 * r_max + 2 * s_max + 7]`` float.
        """
        d_residue, d_token, d_chain, b_same_entity = self._relp_buckets(
            asym_id, residue_index, entity_id, token_index, sym_id,
            cyclic_period)
        n_pos = 2 * self.r_max + 2
        n_chain = 2 * self.s_max + 2
        a_rel_pos = F.one_hot(d_residue, n_pos)
        a_rel_token = F.one_hot(d_token, n_pos)
        a_rel_chain = F.one_hot(d_chain, n_chain)
        return torch.cat(
            [
                a_rel_pos.float(),
                a_rel_token.float(),
                b_same_entity.unsqueeze(-1).float(),
                a_rel_chain.float(),
            ],
            dim=-1,
        )

    def forward(self,
                asym_id: Optional[torch.Tensor] = None,
                residue_index: Optional[torch.Tensor] = None,
                entity_id: Optional[torch.Tensor] = None,
                cyclic_period: Optional[torch.Tensor] = None,
                token_index: Optional[torch.Tensor] = None,
                sym_id: Optional[torch.Tensor] = None,
                relp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project raw or precomputed relative-position features.

        Args:
            asym_id / residue_index / entity_id / cyclic_period / token_index /
                sym_id: raw token features used when ``relp`` is omitted.
            relp: optional precomputed ``[B, N, N, 4 * r_max + 2 * s_max + 7]``
                feature; when supplied, raw features are ignored.

        Returns:
            ``[B, N_token, N_token, token_z]`` pair contribution.
        """
        if relp is not None:
            return self.linear(relp.to(self.linear.weight.dtype))

        d_residue, d_token, d_chain, b_same_entity = self._relp_buckets(
            asym_id, residue_index, entity_id, token_index, sym_id,
            cyclic_period)
        n_pos = 2 * self.r_max + 2
        n_chain = 2 * self.s_max + 2

        # one_hot(index) @ W is an embedding lookup; accumulate weight slices
        # without materializing the one-hots or their concatenation:
        # [ d_residue (n_pos) | d_token (n_pos) | b_same_entity (1) | d_chain (n_chain) ].
        if self.mapping.tp_size == 1:
            wt = self.linear.weight.t()
            p = F.embedding(d_residue, wt[0:n_pos])
            p += F.embedding(d_token, wt[n_pos:2 * n_pos])
            p += b_same_entity[..., None].to(p.dtype) * wt[2 * n_pos]
            p += F.embedding(d_chain,
                             wt[2 * n_pos + 1:2 * n_pos + 1 + n_chain])
            return p

        # Column-sharded Linear requires the concatenated input.
        a_rel_pos = F.one_hot(d_residue, n_pos)
        a_rel_token = F.one_hot(d_token, n_pos)
        a_rel_chain = F.one_hot(d_chain, n_chain)
        return self.linear(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            ))
