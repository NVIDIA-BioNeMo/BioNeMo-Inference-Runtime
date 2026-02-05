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

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.modules.openfold3.sequence_local_atom_attention import \
    AtomAttentionEncoder
from tensorrt_bionemo._torch.modules.openfold3.utils.relpos import \
    relpos_complex
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs.base import BaseConfig


class InputEmbedderAllAtom(nn.Module):
    """
    Embeds a subset of the input features.

    AF3 Algorithm 1 lines 1-5. Includes Algorithms 2 (InputFeatureEmbedder)
    and 3 (RelativePositionEncoding).
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.max_relative_idx = config.max_relative_idx
        self.max_relative_chain = config.max_relative_chain
        self.dtype = config.torch_dtype
        self.mapping = config.mapping
        self.skip_create_weights = config.skip_create_weights

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom_ref_element=config.c_atom_ref_element,
            c_atom_ref_name_chars=config.c_atom_ref_name_chars,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            atom_transformer_config=config.atom_transformer_config,
            n_query=config.n_query,
            n_key=config.n_key,
            c_s=config.c_s,
            c_z=config.c_z,
            inf=config.mask_inf,
            eps=config.norm_epsilon,
            add_noisy_pos=config.add_noisy_pos,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights)

        self.linear_s = Linear(config.c_s_input,
                               config.c_s,
                               bias=False,
                               dtype=self.dtype,
                               mapping=self.mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=self.skip_create_weights)

        self.linear_z_i = Linear(
            config.c_s_input,
            config.c_z,
            bias=False,
            dtype=self.dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.skip_create_weights)

        self.linear_z_j = Linear(
            config.c_s_input,
            config.c_z,
            bias=False,
            dtype=self.dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.skip_create_weights)

        num_rel_pos_bins = 2 * self.max_relative_idx + 2
        num_rel_token_bins = 2 * self.max_relative_idx + 2
        num_rel_chain_bins = 2 * self.max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (num_rel_pos_bins + num_rel_token_bins +
                           num_rel_chain_bins + num_same_entity_features)

        self.linear_relpos = Linear(
            num_relpos_dims,
            config.c_z,
            bias=False,
            dtype=self.dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.skip_create_weights)

        # Expecting binary feature "token_bonds" of shape [*, N_token, N_token, 1]
        self.linear_token_bonds = Linear(
            1,
            config.c_z,
            bias=False,
            dtype=self.dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.skip_create_weights)

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        batch: dict,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """
        #TODO: Check if we need to cast the dtype to float32 here (if accuracy is not affected during inference)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            a, _, _, _ = self.atom_attn_enc(batch=batch,
                                            atom_mask=batch["atom_mask"],
                                            attn_metadata=attn_metadata)

        a = a.to(dtype=self.linear_s.weight.dtype)

        # [*, N_token, C_s_input]
        s_input = torch.cat(
            [
                a,
                batch["restype"],
                batch["profile"],
                batch["deletion_mean"].unsqueeze(-1),
            ],
            dim=-1,
        )

        # [*, N_token, C_s]
        s = self.linear_s(s_input)

        s_input_emb_i = self.linear_z_i(s_input)
        s_input_emb_j = self.linear_z_j(s_input)
        token_bonds_emb = self.linear_token_bonds(
            batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype))

        # [*, N_token, N_token, C_z]
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]

        relpos_feats = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=z.dtype)
        relpos_emb = self.linear_relpos(relpos_feats)
        z = z + relpos_emb

        z = z + token_bonds_emb

        return s_input, s, z
