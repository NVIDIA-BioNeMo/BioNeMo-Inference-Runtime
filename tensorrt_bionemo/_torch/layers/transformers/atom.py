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
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping


class AtomTransformer(nn.Module):

    def __init__(self,
                 attn_window_queries: int = None,
                 attn_window_keys: int = None,
                 diffusion_transformer_config: BaseConfig = None,
                 diffusion_transformer_cls: Any = None):
        """
        Args:
            attn_window_queries: int
                The number of atoms per window for queries.
            attn_window_keys: int
                The number of atoms per window for keys.
            diffusion_transformer_config: BaseConfig
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
        >>> batch_size = 2; n_atoms = 928; multiplicity = 1;
        >>> K = n_atoms // W
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
        >>> q = torch.randn(batch_size, multiplicity, n_atoms, 128, device=torch.device("cuda"))
        >>> c = torch.randn(batch_size, n_atoms, 128, device=torch.device("cuda"))
        >>> bias = torch.randn(batch_size, n_atoms, 128, 16, device=torch.device("cuda"))
        >>> mask = torch.randint(0, 2, (batch_size, n_atoms), device=torch.device("cuda"), dtype=torch.float32)
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

        # q: [B, multiplicity, NW, W, D]
        # c: [B, 1, NW, W, D]
        # bias: [B, 1, NW, W, H, D]
        # mask: [B, 1, NW, W]
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
                 atom_s: int,
                 token_s: int,
                 atoms_per_window_queries: int,
                 atoms_per_window_keys: int,
                 diffusion_transformer_config: BaseConfig = None,
                 diffusion_transformer_cls: Any = None,
                 structure_prediction=True,
                 version: str = "v1",
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
            diffusion_transformer_config: BaseConfig
                The configuration for the DiffusionTransformer.
            diffusion_transformer_cls: nn.Module
                The implementation class of the DiffusionTransformer.
        """
        super().__init__()
        self.structure_prediction = structure_prediction
        atom_s = atom_s or diffusion_transformer_config.dim
        dtype = dtype or diffusion_transformer_config.dtype
        mapping = mapping or diffusion_transformer_config.mapping
        skip_create_weights = skip_create_weights or diffusion_transformer_config.skip_create_weights
        dtype = str_dtype_to_torch(dtype) if isinstance(dtype, str) else dtype
        self.version = version

        if self.structure_prediction:
            self.r_to_q_trans = Linear(
                10 if self.version == "v1" else 3,  # v1 uses 10, v2 uses 3
                atom_s,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)

        self.atom_encoder = AtomTransformer(
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            diffusion_transformer_config=diffusion_transformer_config,
            diffusion_transformer_cls=diffusion_transformer_cls,
        )
        self.structure_prediction = structure_prediction
        # Fill config values if not provided.
        self.atom_to_token_trans = nn.Sequential(
            Linear(
                atom_s,
                2 * token_s if structure_prediction else token_s,
                bias=False,
                # dtype=dtype,
                dtype=torch.float32,
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
        if hasattr(self, "r_to_q_trans"):
            self.r_to_q_trans.load_weights(weights["r_to_q_trans"])

    def forward(
        self,
        atom_to_token: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor,
        r: torch.Tensor = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            r: torch.Tensor
                The residue coordinates. Shape [B, multiplicity, N_atoms, 3]
        Returns:
            a: torch.Tensor
                The atom feature tensor. Shape [B, multiplicity, N_res, 2 * token_s]
            q: torch.Tensor
                The query tensor. Shape [B, multiplicity, N_atoms, atom_s]
            c: torch.Tensor
                The condition tensor. Shape [B, 1, N_atoms, D]
        """
        if self.structure_prediction:
            B, multiplicity, N, _ = r.shape
            # Sanity check the shape of the r tensor, and attention metadata.
            assert r.ndim == 4, "r must be 4D, shape: (B, multiplicity, N_atoms, 3)"
        else:
            multiplicity = 1
        assert attn_metadata is not None, "Attention metadata is required for AtomAttentionEncoder"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for AtomAttentionEncoder"

        atom_mask = atom_pad_mask.bool()
        q = q.unsqueeze(1)

        if self.structure_prediction:
            # r_to_q: [B, multiplicity, N_atoms, atom_s]
            r_input = r
            if self.version == "v1":
                # See: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/encoders.py#L512C13-L515C14
                r_input = torch.cat(
                    [r, torch.zeros((B, multiplicity, N, 7)).to(r)],
                    dim=-1,
                )
            r_to_q = self.r_to_q_trans(r_input)
            # q: [B, multiplicity, N_atoms, atom_s]
            q = q + r_to_q

        q = self.atom_encoder(q=q,
                              c=c,
                              bias=bias,
                              mask=atom_mask,
                              attn_metadata=attn_metadata,
                              all_reduce_params=all_reduce_params)

        with torch.autocast("cuda", enabled=False):
            # [B, multiplicity, N_atoms, 2 * token_s]
            q_to_a = self.atom_to_token_trans(q.float())
            atom_to_token_mean = atom_to_token.float() / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
            atom_to_token_mean = atom_to_token_mean.unsqueeze(1)
            atom_to_token_mean = atom_to_token_mean.repeat_interleave(
                multiplicity, 1)  # [B, multiplicity, N_atoms, N_res]

            # a = torch.bmm(atom_to_token_mean.transpose(-2, -1), q_to_a) # [B, multiplicity, N_res, D]
            a = torch.einsum("bijd,bijk->bikd", q_to_a,
                             atom_to_token_mean)  # [B, multiplicity, N_res, D]

        a = a.to(q)
        return a, q, c


class AtomAttentionDecoder(nn.Module):

    def __init__(self,
                 token_s: int,
                 atom_s: int,
                 atoms_per_window_queries: int,
                 atoms_per_window_keys: int,
                 diffusion_transformer_config: BaseConfig,
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
            diffusion_transformer_config: BaseConfig
                The configuration for the DiffusionTransformer.
            diffusion_transformer_cls: nn.Module
                The implementation class of the DiffusionTransformer.
        """
        super().__init__()

        dtype = dtype or diffusion_transformer_config.torch_dtype
        mapping = mapping or diffusion_transformer_config.mapping
        skip_create_weights = skip_create_weights or diffusion_transformer_config.skip_create_weights

        self.token_s = token_s
        self.atom_s = atom_s
        self.a_to_q_trans = Linear(
            token_s * 2,
            atom_s,
            bias=False,
            # dtype=dtype,
            dtype=torch.float32,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self.atom_decoder = AtomTransformer(
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            diffusion_transformer_config=diffusion_transformer_config,
            diffusion_transformer_cls=diffusion_transformer_cls,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(atom_s,
                         dtype=dtype,
                         eps=diffusion_transformer_config.norm_epsilon),
            Linear(atom_s,
                   3,
                   bias=False,
                   dtype=dtype,
                   mapping=mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=skip_create_weights))

    def load_weights(self, weights: dict):
        self.a_to_q_trans.load_weights(weights["a_to_q_trans"])
        self.atom_decoder.load_weights(weights["atom_decoder"])

        self.atom_feat_to_atom_pos_update[0].weight.data.copy_(
            weights["atom_feat_to_atom_pos_update.0"][0]["weight"])
        self.atom_feat_to_atom_pos_update[0].bias.data.copy_(
            weights["atom_feat_to_atom_pos_update.0"][0]["bias"])

        self.atom_feat_to_atom_pos_update[1].load_weights(
            weights["atom_feat_to_atom_pos_update.1"])

    def forward(
            self,
            atom_to_token: torch.Tensor,
            atom_pad_mask: torch.Tensor,
            a: torch.Tensor,
            q: torch.Tensor,
            c: torch.Tensor,
            bias: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token: torch.Tensor
                The atom to token mapping. Shape [B, N_atoms, N_res]
            atom_pad_mask: torch.Tensor
                The atom pad mask. Shape [B, N_atoms]
            a: torch.Tensor
                The atom feature tensor. Shape [B, multiplicity, N_res, 2 * token_s]
            q: torch.Tensor
                The query tensor. Shape [B, N_atoms, atom_s], or [B, multiplicity, N_atoms, atom_s]
            c: torch.Tensor
                The condition tensor. Shape [B, 1, N_atoms, D]
            bias: torch.Tensor
                The bias tensor. Shape [B, N_atoms, H, D]
            multiplicity: int
                The multiplicity that used for the structure prediction.
        """
        assert a.ndim == 4, "a must be 4D, shape: (B, multiplicity, N_res, 2 * token_s)"
        assert attn_metadata is not None, "Attention metadata is required for AtomAttentionDecoder"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for AtomAttentionDecoder"
        _, multiplicity, _, _ = a.shape

        with torch.autocast("cuda", enabled=False):
            atom_to_token = atom_to_token.unsqueeze(1)
            # [B, multiplicity, N_atoms, N_res]
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 1)
            # [B, multiplicity, N_res, 2*token_s]
            a_to_q = self.a_to_q_trans(a.float())

            # [B, multiplicity, N_atoms, 2*token_s]
            # a_to_q = torch.bmm(atom_to_token, a_to_q)
            a_to_q = torch.einsum("bikj,bijd->bikd", atom_to_token.float(),
                                  a_to_q)

        # Auto broadcast the q and a_to_q
        if q.ndim == 3:
            # Make the multiplicity dimension
            q = q.unsqueeze(1)
        q = q + a_to_q.to(q)
        atom_mask = atom_pad_mask.bool()

        # [B, multiplicity, N_atoms, atom_s]
        q = self.atom_decoder(q=q,
                              c=c,
                              bias=bias,
                              mask=atom_mask,
                              attn_metadata=attn_metadata,
                              all_reduce_params=all_reduce_params)

        # [B, multiplicity, N_atoms, 3]
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
