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
from typing import Any, Optional

import torch
import torch.nn as nn
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo.config import PretrainedModuleConfig
from tensorrt_bionemo.mapping import Mapping


class AtomTransformer(nn.Module):

    def __init__(self,
                 attn_window_queries: int = None,
                 attn_window_keys: int = None,
                 diffusion_transformer_config: PretrainedModuleConfig = None,
                 diffusion_transformer_cls: Any = None):
        """
        Args:
            attn_window_queries: int
                The number of atoms per window for queries.
            attn_window_keys: int
                The number of atoms per window for keys.
            diffusion_transformer_config: PretrainedModuleConfig
                The configuration for the DiffusionTransformer.
            diffusion_transformer_cls: nn.Module
                The implementation class of the DiffusionTransformer.
        """
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer: nn.Module = diffusion_transformer_cls(
            diffusion_transformer_config)

    def load_weights(self, weights: dict):
        self.diffusion_transformer.load_weights(
            weights["diffusion_transformer"])

    def forward(
            self,
            q: torch.Tensor,
            c: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            q: torch.Tensor
                The query tensor. Shape [B, multiplicity, N, D]
            c: torch.Tensor
                The condition tensor. Shape [B, N, D]
            bias: torch.Tensor
                The bias tensor. Shape [B, N, H, D]
            mask: torch.Tensor
                The mask tensor. Shape [B, N]
        Returns:
            torch.Tensor
            The output tensor. Shape [B, multiplicity, N, D]
        Usage: Example for Boltz1x AtomTransformer:
        >>> W = 32; H = 128;
        >>> batch_size = 2; seq_len = 928; multiplicity = 1;
        >>> K = seq_len // W
        >>> keys_indexing_matrix = create_indexing_matrix(K, W, H, device=torch.device("cuda"))
        >>> query_to_keys = partial(
        >>>     query_to_keys, keys_indexing_matrix=keys_indexing_matrix, W=W, H=H
        >>> )
        >>> atom_transformer = AtomTransformer(
        >>>     attn_window_queries=32,
        >>>     attn_window_keys=128,
        >>>     diffusion_transformer_config=config,
        >>>     diffusion_transformer_cls=BoltzDiffusionTransformer
        >>> )
        >>> q = torch.randn(batch_size, multiplicity, seq_len, 128, device=torch.device("cuda"))
        >>> c = torch.randn(batch_size, seq_len, 128, device=torch.device("cuda"))
        >>> bias = torch.randn(batch_size, seq_len, 128, 16, device=torch.device("cuda"))
        >>> mask = torch.randint(0, 2, (batch_size, seq_len), device=torch.device("cuda"), dtype=torch.float32)
        >>> attn_metadata = AttentionMetadata(
        >>>     query_to_keys=query_to_keys,
        >>>     bias_cache={},
        >>> )
        >>> output = atom_transformer(q, c, bias, mask, attn_metadata)
        """
        assert attn_metadata is not None, "Attention metadata is required for Boltz1AtomTransformer"
        W = self.attn_window_queries
        H = self.attn_window_keys

        B, multiplicity, N, _ = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B, multiplicity, NW, W, -1))
        c = c.view((B, 1, NW, W, -1))  # expand dim 1 for broadcasting
        mask = mask.view(B, 1, NW, W)  # expand dim 1 for broadcasting

        # and repeat at the dim 1, this is different from the original implementation.
        bias = bias.view((B, 1, NW, W, H, -1))  # expand dim 1 for broadcasting

        a = self.diffusion_transformer(
            a=q,
            s=c,
            z=bias,
            mask=mask,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )
        a = a.view((B, multiplicity, N, -1))
        return a


class AtomAttentionEncoder(nn.Module):

    def __init__(self,
                 token_s: int,
                 atoms_per_window_queries: int,
                 atoms_per_window_keys: int,
                 diffusion_transformer_config: PretrainedModuleConfig,
                 diffusion_transformer_cls: Any = None,
                 structure_prediction=True,
                 dtype: Optional[torch.dtype] = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        """
        Args:
            token_s: int
                The token single representation dimension.
            atoms_per_window_queries: int
                The number of atoms per window for queries.
            atoms_per_window_keys: int
                The number of atoms per window for keys.
            diffusion_transformer_config: PretrainedModuleConfig
                The configuration for the DiffusionTransformer.
            diffusion_transformer_cls: nn.Module
                The implementation class of the DiffusionTransformer.
        TODO: Implement for the structure prediction.
        """
        super().__init__()
        self.atom_encoder = AtomTransformer(
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            diffusion_transformer_config=diffusion_transformer_config,
            diffusion_transformer_cls=diffusion_transformer_cls,
        )
        self.structure_prediction = structure_prediction
        # Fill config values if not provided.
        atom_s = diffusion_transformer_config.dim
        dtype = dtype or diffusion_transformer_config.dtype
        mapping = mapping or diffusion_transformer_config.mapping
        skip_create_weights = skip_create_weights or diffusion_transformer_config.skip_create_weights

        self.atom_to_token_trans = nn.Sequential(
            Linear(
                atom_s,
                token_s,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
            nn.ReLU(),
        )

    def load_weights(self, weights: dict):
        self.atom_to_token_trans[0].load_weights(
            weights["atom_to_token_trans.0"])
        self.atom_encoder.load_weights(weights["atom_encoder"])

    def forward(
            self,
            atom_to_token: torch.Tensor,
            atom_pad_mask: torch.Tensor,
            q: torch.Tensor,
            c: torch.Tensor,
            bias: torch.Tensor,
            multiplicity=1,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token: torch.Tensor
                The atom to token mapping. Shape [B, N_atoms, N_res]
            atom_pad_mask: torch.Tensor
                The atom pad mask. Shape [B, N_atoms]
            q: torch.Tensor
                The query tensor. Shape [B, N_atoms, D]
            c: torch.Tensor
                The condition tensor. Shape [B, N_atoms, D]
            bias: torch.Tensor
                The bias tensor. Shape [B, N_atoms, H, D]
            multiplicity: int
                The multiplicity that used for the structure prediction.
        TODO: Implement for the structure prediction.
        """
        assert attn_metadata is not None, "Attention metadata is required for AtomAttentionEncoder"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for AtomAttentionEncoder"

        atom_mask = atom_pad_mask.bool()

        q = q.unsqueeze(1)
        q = q.repeat_interleave(multiplicity,
                                1)  # [B, multiplicity, N_atoms, D]

        q = self.atom_encoder(q=q,
                              c=c,
                              bias=bias,
                              mask=atom_mask,
                              attn_metadata=attn_metadata,
                              all_reduce_params=all_reduce_params)

        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(
                q).float()  # [B, multiplicity, N_atoms, D]
            atom_to_token = atom_to_token.float()
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
            atom_to_token_mean = atom_to_token_mean.unsqueeze(1)
            atom_to_token_mean = atom_to_token_mean.repeat_interleave(
                multiplicity, 1)  # [B, multiplicity, N_atoms, N_res]

            # a = torch.bmm(atom_to_token_mean.transpose(-2, -1), q_to_a) # [B, multiplicity, N_res, D]
            a = torch.einsum("bijd,bijk->bikd", q_to_a,
                             atom_to_token_mean)  # [B, multiplicity, N_res, D]

        a = a.to(q)
        return a, q, c
