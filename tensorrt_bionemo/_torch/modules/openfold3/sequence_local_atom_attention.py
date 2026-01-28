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

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    OpenFold3DiffusionTransformer as DiffusionTransformer
from tensorrt_bionemo._torch.modules.openfold3.utils.atomize_utils import (
    aggregate_atom_feat_to_tokens, broadcast_token_feat_to_atoms)
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping

TensorDict = dict[str, torch.Tensor]


def pad_to_multiple_and_divide(tensor: torch.Tensor,
                               multiple: int,
                               dim: int = 1):
    """
    Pad a tensor to a multiple of a given value along a given dimension.
    Args:
        tensor: The tensor to pad.
        multiple: The multiple to pad to.
        dim: The dimension to pad along.
        Example: Tensor with shape (1, 601, 128) dim = 1, multiple = 32 -> (1, 608, 128) -> (1, 19, 32, 128)
    Returns:
        The padded and divided tensor.
    """
    current_size = tensor.shape[dim]
    pad_size = multiple - (tensor.shape[dim] % multiple)
    extend_size = tensor.shape[dim] + pad_size
    pad = [0, 0] * (tensor.dim() - dim - 1) + [0, pad_size]
    tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = extend_size // multiple
    tensor_shape.insert(dim + 1, multiple)
    tensor = tensor.reshape(tensor_shape)
    return tensor, current_size


def convert_pair_atom_to_blocks(
        zij_trunk: torch.Tensor, atom_to_token_index: torch.Tensor,
        atom_mask: torch.Tensor, n_query: int, n_key: int,
        attn_metadata: AttentionMetadata) -> torch.Tensor:
    """
    Args:
        zij_trunk:
            [*, M, N_token, N_token, c_z] Trunk pair representation
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        atom_to_token_index:
            [*, M, N_atom] Atom to token index
        atom_mask:
            [*, M, N_atom] Atom mask
    Returns:
        zij_trunk:
            [*, M, N_blocks, N_query, N_key, c_atom_pair] Trunk pair representation
    """
    batch_dims = zij_trunk.shape[:-3]
    n_atom = atom_to_token_index.shape[-1]
    num_blocks = (n_atom + n_query - 1) // n_query
    atom_to_token_index_q, _ = pad_to_multiple_and_divide(
        atom_to_token_index,
        multiple=n_query,
        dim=atom_to_token_index.ndim - 1)

    zij_trunk = zij_trunk.unsqueeze(-4).expand(
        (*batch_dims, num_blocks, *zij_trunk.shape[-3:]))

    zij_trunk = torch.gather(
        zij_trunk,
        dim=-3,
        index=atom_to_token_index_q[..., None, None].expand(
            (*batch_dims, num_blocks, n_query, *zij_trunk.shape[-2:])).long(),
    )
    atom_to_token_index_k = attn_metadata.query_to_keys(
        atom_to_token_index_q.unsqueeze(-1)).squeeze(-1)
    zij_trunk = torch.gather(
        zij_trunk,
        dim=-2,
        index=atom_to_token_index_k[..., None, :, None].expand(
            (*batch_dims, num_blocks, n_query, n_key,
             zij_trunk.shape[-1])).long(),
    )

    atom_mask, _ = pad_to_multiple_and_divide(atom_mask.unsqueeze(-1),
                                              multiple=n_query,
                                              dim=atom_mask.ndim - 1)

    atom_pair_mask = atom_mask * attn_metadata.query_to_keys(
        atom_mask).squeeze(-1).unsqueeze(-2)
    zij_trunk = zij_trunk * atom_pair_mask.unsqueeze(-1)
    return zij_trunk


class RefAtomFeatureEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 1 - 6).
    """

    def __init__(
        self,
        c_atom_ref_element,
        c_atom_ref_name_chars,
        c_atom: int,
        c_atom_pair: int,
        dtype: torch.dtype = torch.float32,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_atom_ref:
                Dict of reference atom channel dimensions per feature
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
        """
        super().__init__()
        # Ref conformer feats
        self.dtype = dtype
        self.mapping = mapping

        self.linear_merge_ref_features = Linear(
            3 + 1 + 1 + c_atom_ref_element + c_atom_ref_name_chars,
            c_atom,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM))

        self.linear_ref_offset = Linear(
            3,
            c_atom_pair,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self.linear_inv_sq_dists = Linear(
            1,
            c_atom_pair,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self.linear_valid_mask = Linear(
            1,
            c_atom_pair,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

    def forward(
        self,
        batch: TensorDict,
        n_query: int,
        attn_metadata: AttentionMetadata = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "ref_pos": [*, N_atom, 3] atom positions in the
                        reference conformer
                    - "ref_mask": [*, N_atom] atom mask for the reference conformer
                    - "ref_element": [*, N_atom, 128] one-hot encoding of atomic number
                        in the reference conformer
                    - "ref_charge": [*, N_atom] atom charge in the reference conformer
                    - "ref_atom_name_chars": [*, N_atom, 4, 64] one-hot encoding of
                        unicode integers representing unique atom names in the
                        reference conformer
                    - "ref_space_uid": [*, n_atom,] numerical encoding of the chain id
                        and residue index in the reference conformer
            n_query:
                Number of queries (block height)
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
        """
        dtype = batch["ref_pos"].dtype

        # Embed atom features
        # [*, N_atom, c_atom]
        #TODO: Checking TP here. We can do gather later for more efficient.

        cl = torch.cat([
            batch["ref_pos"],
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1)),
            batch["ref_mask"].unsqueeze(-1).to(dtype=dtype),
            batch["ref_element"].to(dtype=dtype),
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype)
        ],
                       dim=-1)

        cl = self.linear_merge_ref_features(cl)

        # Embed offsets
        # Convert all atom rep to block format ahead of time due to
        # reduce memory cost
        # dl, dm: [*, N_blocks, N_query, 3], [*, N_blocks, N_key, 3]
        # vl, vm: [*, N_blocks, N_query, 1], [*, N_blocks, N_key, 1]
        # atom_mask: [*, N_blocks, N_query, N_key]

        d_l, _ = pad_to_multiple_and_divide(batch["ref_pos"],
                                            multiple=n_query,
                                            dim=batch["ref_pos"].ndim - 2)

        d_m = attn_metadata.query_to_keys(d_l)
        if batch["ref_pos"].ndim == 2:
            d_m = d_m.squeeze(1)

        atom_mask, _ = pad_to_multiple_and_divide(
            batch["atom_mask"].unsqueeze(-1),
            multiple=n_query,
            dim=batch["atom_mask"].ndim - 1)

        if batch["atom_mask"].ndim == 2:
            atom_mask = atom_mask * attn_metadata.query_to_keys(
                atom_mask).squeeze(1).squeeze(-1).unsqueeze(-2)
        else:
            atom_mask = atom_mask * attn_metadata.query_to_keys(
                atom_mask).squeeze(-1).unsqueeze(-2)

        v_l, _ = pad_to_multiple_and_divide(
            batch["ref_space_uid"].unsqueeze(-1),
            multiple=n_query,
            dim=batch["ref_space_uid"].ndim - 1)

        v_m = attn_metadata.query_to_keys(v_l)
        if batch["ref_space_uid"].ndim == 2:
            v_m = v_m.squeeze(1)

        # dlm: [*, N_blocks, N_query, N_key, 3]
        # vlm: [*, N_blocks, N_query, N_key, 1]
        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype) * atom_mask.unsqueeze(-1)

        plm = self.linear_ref_offset(dlm) * vlm

        # Embed pairwise inverse squared distances
        # [*, N_blocks, N_query, N_key, c_atom_pair]
        inv_sq_dists = 1.0 / (1 + torch.sum(dlm**2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 8 - 12).
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atom_pair: int,
        dtype: torch.dtype = torch.float32,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
        eps: float = 1e-5,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
        """
        super().__init__()
        self.layer_norm_s = nn.LayerNorm(c_s, bias=False, dtype=dtype, eps=eps)
        self.linear_s = Linear(c_s,
                               c_atom,
                               bias=False,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)
        self.layer_norm_z = nn.LayerNorm(c_z, bias=False, dtype=dtype, eps=eps)
        self.linear_z = Linear(c_z,
                               c_atom_pair,
                               bias=False,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)
        self.linear_r = Linear(3,
                               c_atom,
                               bias=False,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

    def forward(
        self,
        batch: TensorDict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
        n_query: int,
        n_key: int,
        attn_metadata: AttentionMetadata = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
            si_trunk:
                [*, N_token, c_s] Trunk single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Trunk pair representation
            rl:
                [*, N_atom, 3] Noisy atom positions
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            attn_metadata:
                Attention metadata (for key and query to tokens conversion)
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning with trunk single
                    representation embedded
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning with
                    trunk pair representation embedded
            ql:
                [*, N_atom, c_atom] Atom single representation with noisy coordinate
                    projection
        """

        # Broadcast trunk single representation into atom single conditioning
        # [*, N_atom, c_atom]
        si_trunk = self.linear_s(self.layer_norm_s(si_trunk))
        si_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=si_trunk,
            token_dim=-2,
        )
        cl = cl + si_trunk

        # Broadcast trunk pair representation into atom pair conditioning
        zij_trunk = self.linear_z(self.layer_norm_z(zij_trunk))
        zij_trunk = self.convert_pair_atom_to_blocks(
            zij_trunk=zij_trunk,
            atom_to_token_index=batch["atom_to_token_index"],
            atom_mask=batch["atom_mask"],
            n_query=n_query,
            n_key=n_key,
            attn_metadata=attn_metadata)

        plm = plm + zij_trunk

        # Add noisy coordinate projection
        # [*, N_atom, c_atom]
        ql = cl + self.linear_r(rl)

        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """
    Implements AF3 Algorithm 5.
    """

    def __init__(self,
                 c_atom_ref_element: int = 119,
                 c_atom_ref_name_chars: int = 256,
                 c_atom: int = 128,
                 c_atom_pair: int = 384,
                 c_token: int = 384,
                 n_query: int = 32,
                 n_key: int = 128,
                 c_s: int | None = None,
                 c_z: int | None = None,
                 atom_transformer_config: BaseConfig = None,
                 inf: float = 1e9,
                 eps: float = 1e-5,
                 add_noisy_pos: bool = False,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        """
        Args:
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            add_noisy_pos:
                Whether to add noisy positions and trunk embeddings
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            c_s:
                Single representation channel dimension (optional)
            c_z:
                Pair representation channel dimension (optional)
        """
        super().__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.inf = inf
        self.dtype = dtype
        self.mapping = mapping
        self.add_noisy_pos = add_noisy_pos
        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref_element=c_atom_ref_element,
            c_atom_ref_name_chars=c_atom_ref_name_chars,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair)

        if self.add_noisy_pos:
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                c_atom_pair=c_atom_pair,
                dtype=dtype,
                mapping=mapping,
                skip_create_weights=skip_create_weights,
                eps=eps,
            )

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom,
                               c_atom_pair,
                               bias=False,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)
        self.linear_m = Linear(c_atom,
                               c_atom_pair,
                               bias=False,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atom_pair,
                   c_atom_pair,
                   bias=False,
                   dtype=dtype,
                   mapping=mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=skip_create_weights),
            nn.ReLU(),
            Linear(c_atom_pair,
                   c_atom_pair,
                   bias=False,
                   dtype=dtype,
                   mapping=mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=skip_create_weights),
            nn.ReLU(),
            Linear(c_atom_pair,
                   c_atom_pair,
                   bias=False,
                   dtype=dtype,
                   mapping=mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=skip_create_weights),
        )

        self.atom_transformer = DiffusionTransformer(
            config=atom_transformer_config)

        self.c_token = c_token
        self.linear_q = nn.Sequential(
            Linear(c_atom,
                   c_token,
                   bias=False,
                   dtype=dtype,
                   mapping=mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=skip_create_weights), nn.ReLU())

    def get_atom_reps(
        self,
        batch: TensorDict,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk:
                [*, N_atom, c_s] Trunk single representation (optional)
            zij_trunk:
                [*, N_atom, N_atom, c_z] Trunk pair representation (optional)
        Returns:
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format ahead of time due to reduce memory cost
        """
        # Embed reference atom features
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_blocks, N_query, N_key, c_atom_pair]

        cl, plm = self.ref_atom_feature_embedder(batch=batch,
                                                 n_query=self.n_query,
                                                 attn_metadata=attn_metadata)

        if self.add_noisy_pos and rl is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch,
                cl=cl,
                plm=plm,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                rl=rl,
                n_query=self.n_query,
                n_key=self.n_key,
            )
        else:
            # Initialize atom single representation when trunk / noisy position
            # inputs are not present
            # [*, N_atom, c_atom]
            ql = cl.clone()

        # Add the combined single conditioning to the pair rep (line 13 - 14)

        cl_l, _ = pad_to_multiple_and_divide(cl,
                                             multiple=self.n_query,
                                             dim=cl.ndim - 2)

        cl_m = attn_metadata.query_to_keys(cl_l)
        if cl.ndim == 3:
            cl_m = cl_m.squeeze(1)

        atom_mask, _ = pad_to_multiple_and_divide(
            batch["atom_mask"].unsqueeze(-1),
            multiple=self.n_query,
            dim=batch["atom_mask"].ndim - 1)

        if batch["atom_mask"].ndim == 2:
            atom_mask = atom_mask * attn_metadata.query_to_keys(
                atom_mask).squeeze(1).squeeze(-1).unsqueeze(-2)
        else:
            atom_mask = atom_mask * attn_metadata.query_to_keys(
                atom_mask).squeeze(-1).unsqueeze(-2)

        # Note to devs: in previous checkpoints before v13, linear_l and linear_m
        #  were reversed. Changed it for consistent naming.
        cl_lm = (self.linear_l(self.relu(cl_l.unsqueeze(-2))) + self.linear_m(
            self.relu(cl_m.unsqueeze(-3)))) * atom_mask.unsqueeze(-1)

        # [*, N_blocks, N_query, N_key, c_atom_pair]
        plm = plm + cl_lm

        plm = plm + self.pair_mlp(plm)

        plm = plm * atom_mask.unsqueeze(-1)

        return ql, cl, plm

    def forward(
        self,
        batch: TensorDict,
        atom_mask: torch.Tensor,
        attn_metadata: AttentionMetadata,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "ref_pos": [*, N_atom, 3] atom positions in the
                        reference conformer
                    - "ref_mask": [*, N_atom] atom mask for the reference conformer
                    - "ref_element": [*, N_atom, 128] one-hot encoding of atomic number
                        in the reference conformer
                    - "ref_charge": [*, N_atom] atom charge in the reference conformer
                    - "ref_atom_name_chars": [*, N_atom, 4, 64] one-hot encoding of
                        unicode integers representing unique atom names in the
                        reference conformer
                    - "ref_space_uid": [*, n_atom,] numerical encoding of the chain id
                        and residue index in the reference conformer
                    - "token_mask": [*, N_token] token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk:
                [*, N_atom, c_s] Trunk single representation (optional)
            zij_trunk:
                [*, N_atom, N_atom, c_z] Trunk pair representation (optional)
        Returns:
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format ahead of time due to reduce memory cost
        """
        ql, cl, plm = self.get_atom_reps(
            batch=batch,
            rl=rl,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            attn_metadata=attn_metadata,
        )
        # Cross attention transformer (line 15)
        # [*, N_blocks, N_query, c_atom]
        # batch_size = ql.shape[0]
        # current_n_atom = ql.shape[1]

        is_contained_diffusion_chanel = False
        if ql.ndim == 4:
            is_contained_diffusion_chanel = True

        ql, current_size = pad_to_multiple_and_divide(ql,
                                                      multiple=self.n_query,
                                                      dim=ql.ndim - 2)

        if is_contained_diffusion_chanel == False:
            ql = ql.unsqueeze(1)

        cl, _ = pad_to_multiple_and_divide(cl,
                                           multiple=self.n_query,
                                           dim=cl.ndim - 2)
        if is_contained_diffusion_chanel == False:
            cl = cl.unsqueeze(1)

        atom_mask, _ = pad_to_multiple_and_divide(atom_mask,
                                                  multiple=self.n_query,
                                                  dim=atom_mask.ndim - 1)

        ql = self.atom_transformer(a=ql,
                                   s=cl,
                                   z=plm,
                                   mask=atom_mask,
                                   attn_metadata=attn_metadata)

        if is_contained_diffusion_chanel == False:
            ql = ql.flatten(1, 3)[:, :current_size, :]
            cl = cl.flatten(1, 3)[:, :current_size, :]
            atom_mask = atom_mask.flatten(1, 2)[:, :current_size]

        else:
            ql = ql.flatten(2, 3)[:, :, :current_size, :]
            cl = cl.flatten(2, 3)[:, :, :current_size, :]
            atom_mask = atom_mask.flatten(2, 3)[:, :, :current_size]

        ql = ql * atom_mask.unsqueeze(-1)

        atom_feat = self.linear_q(ql)

        ai = aggregate_atom_feat_to_tokens(
            token_mask=batch["token_mask"],
            atom_to_token_index=batch["atom_to_token_index"],
            atom_mask=atom_mask,
            atom_feat=atom_feat,
            atom_dim=-2,
            aggregate_fn="mean",
        )

        return ai, ql, cl, plm
