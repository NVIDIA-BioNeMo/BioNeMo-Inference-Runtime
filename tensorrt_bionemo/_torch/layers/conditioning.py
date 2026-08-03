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

from tensorrt_bionemo._torch.auto_chunk import (CHUNK_REGISTRY,
                                                DIFFUSION_PAIR_TRANSITION)
from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.layers.position_encoders import FourierEmbedding
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo._torch.modules.openfold3.utils.relpos import \
    relpos_complex


class ContactConditioning(nn.Module):
    """ Boltz2 Contact Conditioning """

    def __init__(self,
                 token_z: int,
                 cutoff_min: float,
                 cutoff_max: float,
                 contact_conditioning_info: dict[str, int],
                 dtype: torch.dtype = torch.float32,
                 skip_create_weights: bool = False):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z, dtype=dtype)
        self.encoder = Linear(token_z + len(contact_conditioning_info) - 1,
                              token_z,
                              dtype=dtype,
                              skip_create_weights=skip_create_weights)
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        self.contact_conditioning_info = contact_conditioning_info

    def forward(self, contact_conditioning: torch.Tensor,
                contact_threshold: torch.Tensor):
        assert self.contact_conditioning_info["UNSPECIFIED"] == 0
        assert self.contact_conditioning_info["UNSELECTED"] == 1
        final_contact_conditioning = contact_conditioning[:, :, :, 2:]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min
                                        ) / (self.cutoff_max - self.cutoff_min)
        # Inline the Fourier features into the cat instead of binding them to a variable, so the
        # [N,N,token_z] fp32 fourier tensor (~15 GB at N~5k) is freed right after the cat rather
        # than kept live through the encoder + masking below.
        final_contact_conditioning = self.encoder(
            torch.cat(
                [
                    final_contact_conditioning,
                    contact_threshold_normalized.unsqueeze(-1),
                    self.fourier_embedding(
                        contact_threshold_normalized.flatten()).reshape(
                            contact_threshold_normalized.shape + (-1, )),
                ],
                dim=-1,
            ))

        # Fold the unspecified/unselected masking IN PLACE (the encoder
        # output is freshly owned), avoiding 3-4 separate
        # [N,N,token_z] temporaries for the multiply and the two adds.
        mask = 1 - contact_conditioning[:, :, :, 0:2].sum(dim=-1, keepdim=True)
        final_contact_conditioning *= mask
        final_contact_conditioning += (self.encoding_unspecified *
                                       contact_conditioning[:, :, :, 0:1])
        final_contact_conditioning += (self.encoding_unselected *
                                       contact_conditioning[:, :, :, 1:2])
        return final_contact_conditioning


class PairwiseConditioning(nn.Module):

    def __init__(self,
                 token_z: int,
                 dim_token_rel_pos_feats: int,
                 num_transitions: int = 2,
                 transition_expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False):
        super().__init__()
        self.dtype = dtype
        self.token_z = token_z
        self.dim_token_rel_pos_feats = dim_token_rel_pos_feats
        self.num_transitions = num_transitions

        self.init_proj_norm = nn.LayerNorm(token_z + dim_token_rel_pos_feats,
                                           eps=eps,
                                           dtype=dtype)

        self.init_proj_linear = Linear(token_z + dim_token_rel_pos_feats,
                                       token_z,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip_create_weights)

        transitions = []
        for i in range(num_transitions):
            transitions.append(
                Transition(dim=token_z,
                           hidden=token_z * transition_expansion_factor,
                           out_dim=token_z,
                           eps=eps,
                           dtype=dtype,
                           layer_idx=i,
                           skip_create_weights=skip_create_weights))
        self.transitions = nn.ModuleList(transitions)

    def forward(self, z_trunk: torch.Tensor,
                token_rel_pos_feats: torch.Tensor) -> torch.Tensor:
        # Cast inputs to the conditioner dtype so the [N, N, *] init-proj + FFN run at that
        # precision even when the trunk feeds a higher-precision (e.g. fp32) pair rep.
        if self.dtype is not None:
            z_trunk = z_trunk.to(self.dtype)
            token_rel_pos_feats = token_rel_pos_feats.to(self.dtype)
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.init_proj_norm(z)
        z = self.init_proj_linear(z)
        for transition in self.transitions:
            z = transition(z) + z
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
                 skip_create_weights: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.disable_times = disable_times
        input_dim = 2 * token_s + additional_input_dim

        self.norm_single = nn.LayerNorm(input_dim, dtype=dtype, eps=eps)
        self.single_embed = Linear(input_dim,
                                   2 * token_s,
                                   dtype=dtype,
                                   skip_create_weights=skip_create_weights)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier,
                                                  dtype=torch.float32)
            self.norm_fourier = nn.LayerNorm(dim_fourier,
                                             dtype=torch.float32,
                                             eps=eps)
            self.fourier_to_single = Linear(
                dim_fourier,
                2 * token_s,
                bias=False,
                dtype=torch.float32,
                skip_create_weights=skip_create_weights)

        transitions = nn.ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(dim=2 * token_s,
                                    hidden=transition_expansion_factor * 2 *
                                    token_s,
                                    dtype=dtype,
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


class DiffusionConditioning(nn.Module):
    """
    Implements AF3 Algorithm 21 — Diffusion conditioning for AlphaFold3.

    Prepares the single and pair conditioning representations consumed by the
    diffusion module. This includes:
      - Fourier embedding of the noise level (sigma),
      - Fusing per-token input features and trunk single representations,
      - Encoding relative position, relative token index, relative chain, and
        same-entity features into the pair representation.
    """

    def __init__(self,
                 c_s_input: int,
                 c_s: int,
                 c_z: int,
                 c_fourier_emb: int,
                 max_relative_idx: int,
                 max_relative_chain: int,
                 sigma_data: float,
                 eps: float = 1e-5,
                 dtype: torch.dtype = torch.float32,
                 skip_create_weights: bool = False):
        """
        Args:
            c_s_input:
                Per token input representation channel dimension
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_fourier_emb:
                Fourier embedding channel dimension
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
            sigma_data:
                Constant determined by data variance
        """
        super().__init__()

        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.c_fourier_emb = c_fourier_emb
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain
        self.sigma_data = sigma_data
        self.dtype = dtype

        num_rel_pos_bins = 2 * max_relative_idx + 2
        num_rel_token_bins = 2 * max_relative_idx + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (num_rel_pos_bins + num_rel_token_bins +
                           num_rel_chain_bins + num_same_entity_features)

        self.layer_norm_z = nn.LayerNorm(num_relpos_dims + self.c_z,
                                         bias=False,
                                         dtype=dtype,
                                         eps=eps)
        self.linear_z = Linear(num_relpos_dims + self.c_z,
                               self.c_z,
                               bias=False,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)

        # Only transition_z is row-chunkable; transition_s uses dim 1 for samples.
        self.transition_z = nn.ModuleList([
            Transition(dim=self.c_z,
                       hidden=self.c_z * 2,
                       eps=eps,
                       dtype=dtype,
                       skip_create_weights=skip_create_weights,
                       auto_chunk_policy=CHUNK_REGISTRY.get(
                           DIFFUSION_PAIR_TRANSITION)) for _ in range(2)
        ])

        self.layer_norm_s = nn.LayerNorm(self.c_s + self.c_s_input,
                                         bias=False,
                                         dtype=dtype,
                                         eps=eps)
        self.linear_s = Linear(self.c_s + self.c_s_input,
                               self.c_s,
                               bias=False,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)

        self.fourier_emb = FourierEmbedding(c_fourier_emb, dtype=dtype)

        self.layer_norm_n = nn.LayerNorm(self.c_fourier_emb,
                                         bias=False,
                                         dtype=dtype,
                                         eps=eps)
        self.linear_n = Linear(self.c_fourier_emb,
                               self.c_s,
                               bias=False,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)

        self.transition_s = nn.ModuleList([
            Transition(dim=self.c_s,
                       hidden=self.c_s * 2,
                       eps=eps,
                       dtype=dtype,
                       skip_create_weights=skip_create_weights)
            for _ in range(2)
        ])

    def _embed_trunk_inputs(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pair conditioning
        relpos_zij = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=zij_trunk.dtype)

        zij = torch.cat([zij_trunk, relpos_zij], dim=-1)

        zij = self.linear_z(self.layer_norm_z(zij))

        # Single conditioning
        si = torch.cat([si_trunk, si_input], dim=-1)
        si = self.linear_s(self.layer_norm_s(si))

        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n)

        si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)

        return si, zij

    def _forward(
            self, si: torch.Tensor, zij: torch.Tensor,
            token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)

        # Pair conditioning
        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_token_mask.unsqueeze(-1))

        # Single conditioning
        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask.unsqueeze(-1))

        return si, zij

    def forward(
            self,
            batch: dict,
            t: torch.Tensor,
            si_input: torch.Tensor,
            si_trunk: torch.Tensor,
            zij_trunk: torch.Tensor,
            use_conditioning: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Feature dictionary
            t:
                [*] Noise level at a diffusion timestep
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Pair representation
        Returns:
            si:
                [*, N_token, c_s] Conditioned single representation
            zij:
                [*, N_token, N_token, c_z] Conditioned pair representation
        """
        token_mask = batch["token_mask"]
        if not use_conditioning:
            si_trunk = si_trunk.zero_()
            zij_trunk = zij_trunk.zero_()

        si, zij = self._embed_trunk_inputs(batch=batch,
                                           t=t,
                                           si_input=si_input,
                                           si_trunk=si_trunk,
                                           zij_trunk=zij_trunk)

        si, zij = self._forward(si=si, zij=zij, token_mask=token_mask)

        return si, zij
