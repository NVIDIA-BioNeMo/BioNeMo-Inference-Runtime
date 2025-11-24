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

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import FourierEmbedding
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo.mapping import Mapping


class ContactConditioning(nn.Module):
    """ Boltz2 Contact Conditioning """

    def __init__(self,
                 token_z: int,
                 cutoff_min: float,
                 cutoff_max: float,
                 contact_conditioning_info: dict[str, int],
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z,
                                                  dtype=dtype,
                                                  mapping=mapping)
        self.encoder = Linear(token_z + len(contact_conditioning_info) - 1,
                              token_z,
                              dtype=dtype,
                              mapping=mapping,
                              tensor_parallel_mode=TensorParallelMode.COLUMN,
                              gather_output=True,
                              skip_create_weights=skip_create_weights)
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        self.contact_conditioning_info = contact_conditioning_info

    def load_weights(self, weights: dict = None):
        """
        Args:
            weights: The weights of the model. State dict of the original model.
        """
        self.fourier_embedding.proj.load_weights(weights["fourier_embedding"])
        self.encoder.load_weights(weights["encoder"])

    def forward(self, contact_conditioning: torch.Tensor,
                contact_threshold: torch.Tensor):
        assert self.contact_conditioning_info["UNSPECIFIED"] == 0
        assert self.contact_conditioning_info["UNSELECTED"] == 1
        final_contact_conditioning = contact_conditioning[:, :, :, 2:]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min)
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.flatten()).reshape(
                contact_threshold_normalized.shape + (-1, ))

        final_contact_conditioning = torch.cat(
            [
                final_contact_conditioning,
                contact_threshold_normalized.unsqueeze(-1),
                contact_threshold_fourier,
            ],
            dim=-1,
        )
        final_contact_conditioning = self.encoder(final_contact_conditioning)

        final_contact_conditioning = (
            final_contact_conditioning *
            (1 - contact_conditioning[:, :, :, 0:2].sum(dim=-1, keepdim=True)) +
            self.encoding_unspecified * contact_conditioning[:, :, :, 0:1] +
            self.encoding_unselected * contact_conditioning[:, :, :, 1:2])
        return final_contact_conditioning


class PairwiseConditioning(nn.Module):

    def __init__(self,
                 token_z: int,
                 dim_token_rel_pos_feats: int,
                 num_transitions: int = 2,
                 transition_expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        mapping = mapping or Mapping()
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group
        self.dtype = dtype
        self.token_z = token_z
        self.dim_token_rel_pos_feats = dim_token_rel_pos_feats
        self.num_transitions = num_transitions

        self.init_proj_norm = nn.LayerNorm(token_z + dim_token_rel_pos_feats,
                                           eps=eps,
                                           dtype=dtype)

        self.init_proj_linear = Linear(
            token_z + dim_token_rel_pos_feats,
            token_z,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        transitions = []
        for i in range(num_transitions):
            transitions.append(
                Transition(dim=token_z,
                           hidden=token_z * transition_expansion_factor,
                           out_dim=token_z,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping,
                           layer_idx=i,
                           skip_create_weights=skip_create_weights))
        self.transitions = nn.ModuleList(transitions)

    def forward(
            self,
            z_trunk: torch.Tensor,
            token_rel_pos_feats: torch.Tensor,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.init_proj_norm(z)
        z = self.init_proj_linear(z)
        for transition in self.transitions:
            z = transition(z, all_reduce_params=all_reduce_params) + z
        return z


class SingleConditioning(nn.Module):
    """Boltz single conditioning layer."""

    def __init__(self,
                 token_s: int = 384,
                 dim_fourier: int = 256,
                 num_transitions: int = 2,
                 transition_expansion_factor: int = 2,
                 additional_input_dim: int = 0,
                 eps: float = 1e-20,
                 disable_times: bool = False,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.disable_times = disable_times
        input_dim = 2 * token_s + additional_input_dim

        self.norm_single = nn.LayerNorm(input_dim, dtype=dtype, eps=eps)
        self.single_embed = Linear(
            input_dim,
            2 * token_s,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier,
                                                  dtype=torch.float32,
                                                  mapping=mapping)
            self.norm_fourier = nn.LayerNorm(dim_fourier,
                                             dtype=torch.float32,
                                             eps=eps)
            self.fourier_to_single = Linear(
                dim_fourier,
                2 * token_s,
                bias=False,
                dtype=torch.float32,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)

        transitions = nn.ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(dim=2 * token_s,
                                    hidden=transition_expansion_factor * 2 *
                                    token_s,
                                    dtype=dtype,
                                    mapping=mapping,
                                    skip_create_weights=skip_create_weights)
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        times: torch.Tensor,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            times: [B, multiplicity] or [B]
            s_trunk: [B, N, token_s]
            s_inputs: [B, N, token_s]
        Returns:
            s: [B, multiplicity, N, 2*token_s]
            normed_fourier: [B, multiplicity, N, dim_fourier]
        """
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))

        if times.ndim == 1:
            times = times.unsqueeze(-1)  # multiplicity = 1

        if not self.disable_times:
            # note: sigma rescaling done in diffusion module
            fourier_embed = self.fourier_embed(times)
            # [B, multiplicity, dim_fourier]
            normed_fourier = self.norm_fourier(fourier_embed)
            # [B, multiplicity, 2*token_s]
            fourier_to_single = self.fourier_to_single(normed_fourier)
            s = fourier_to_single.unsqueeze(2).to(s) + s.unsqueeze(1)

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier if not self.disable_times else None
