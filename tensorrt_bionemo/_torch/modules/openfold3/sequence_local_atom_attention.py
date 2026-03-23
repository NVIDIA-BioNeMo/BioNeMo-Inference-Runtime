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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    pad_to_multiple_and_divide, compute_block_indices, fix_boundary_blocks)
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

def convert_pair_atom_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    """
    TRT-compatible equivalent of convert_pair_rep_to_blocks.

    Args:
        batch: dict with:
            "atom_mask":           [B, N_atom] or [B, S, N_atom]  1=valid, 0=padding
            "atom_to_token_index": [B, N_atom] or [B, S, N_atom]  long, atom→token map
        zij_trunk:    [B, N_token, N_token, C] or [B, S, N_token, N_token, C]
        n_query:      query window size (must be even; n_key % (n_query//2) == 0)
        n_key:        key window size
        attn_metadata: AttentionMetadata with query_to_keys callable

    Returns:
        plm: [B, K, n_query, n_key, C]      (no sample dim)
          or [B, S, K, n_query, n_key, C]   (with sample dim)
    """
    atom_mask     = batch["atom_mask"]            # [B, N_atom] or [B, S, N_atom]
    atom_to_token = batch["atom_to_token_index"]  # [B, N_atom] or [B, S, N_atom]

    has_sample_dim = atom_mask.ndim == 3

    if has_sample_dim:
        B, S, N_atom = atom_mask.shape
        atom_mask     = atom_mask.reshape(B * S, N_atom)
        atom_to_token = atom_to_token.reshape(B * S, N_atom)
        zij_trunk     = zij_trunk.reshape(B * S, *zij_trunk.shape[2:])  # [BS, N_tok, N_tok, C]
    else:
        B = atom_mask.shape[0]

    BS, N_atom = atom_mask.shape
    K = math.ceil(N_atom / n_query)
    device = zij_trunk.device

    # ── Q token indices: pad and block ────────────────────────────────────────
    q_token_blocked, _ = pad_to_multiple_and_divide(atom_to_token.float(), multiple=n_query, dim=1)
    q_token_blocked = q_token_blocked[:, :K]          # [BS, K, n_query]

    atom_mask_blocked, _ = pad_to_multiple_and_divide(atom_mask, multiple=n_query, dim=1)
    atom_mask_blocked = atom_mask_blocked[:, :K]      # [BS, K, n_query]

    # ── K token indices via query_to_keys ──────────────────────────────────────
    k_token_float = attn_metadata.query_to_keys(
        q_token_blocked.unsqueeze(-1)               # [BS, K, n_query, 1]
    ).squeeze(1).squeeze(-1)                         # [BS, K, n_key]

    # flat views needed for boundary-fix and unfold
    mask_flat  = atom_mask_blocked.reshape(BS, K * n_query)       # [BS, N_padded]
    q_tok_flat = q_token_blocked.reshape(BS, K * n_query)         # [BS, N_padded]

    # ── per-block shift info ───────────────────────────────────────────────────
    total_shift, is_edge, n_atom_true = compute_block_indices(mask_flat, K, n_query, n_key)

    # ── atom_mask_k via unfold ─────────────────────────────────────────────────
    left_pad    = n_key // 2 - n_query // 2
    mask_padded = F.pad(mask_flat, (left_pad, n_key), value=0.0)
    atom_mask_k = mask_padded.unfold(-1, n_key, n_query)[:, :K]  # [BS, K, n_key]

    # ── fix edge blocks (k_token + mask) in one combined gather pass ───────────
    source   = torch.stack([q_tok_flat, mask_flat], dim=-1)        # [BS, N_padded, 2]
    trt_comb = torch.stack([k_token_float, atom_mask_k], dim=-1)   # [BS, K, n_key, 2]

    fixed = fix_boundary_blocks(trt_comb, source, n_atom_true, total_shift, is_edge)

    k_token_idx = fixed[..., 0].long()  # [BS, K, n_key]
    atom_mask_k = fixed[..., 1]         # [BS, K, n_key]

    q_token_idx = q_token_blocked.long()  # [BS, K, n_query]

    # ── 2D gather from zij_trunk [BS, N_tok, N_tok, C] ────────────────────────
    batch_idx = torch.arange(BS, device=device).view(BS, 1, 1, 1)
    plm = zij_trunk[
        batch_idx,
        q_token_idx.unsqueeze(-1),   # [BS, K, n_query, 1]
        k_token_idx.unsqueeze(-2),   # [BS, K, 1, n_key]
    ]  # [BS, K, n_query, n_key, C]

    # ── apply atom pair mask ───────────────────────────────────────────────────
    atom_pair_mask = atom_mask_blocked.unsqueeze(-1) * atom_mask_k.unsqueeze(-2)
    plm = plm * atom_pair_mask.unsqueeze(-1).to(dtype=plm.dtype)

    if has_sample_dim:
        C = plm.shape[-1]
        plm = plm.reshape(B, S, K, n_query, n_key, C)

    return plm

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
            c_atom_ref_element:
                Reference atom element channel dimension
            c_atom_ref_name_chars:
                Reference atom name characters channel dimension
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

        self.linear_ref_pair_features = Linear(
            3 + 1 + 1,
            c_atom_pair,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM))

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
        ], dim=-1)
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
            batch["atom_mask"].unsqueeze(-1), multiple=n_query, dim=batch["atom_mask"].ndim - 1)
        if batch["atom_mask"].ndim == 2:
            atom_mask = atom_mask * attn_metadata.query_to_keys(atom_mask).squeeze(
                1).squeeze(-1).unsqueeze(-2)
        else:
            atom_mask = atom_mask * attn_metadata.query_to_keys(atom_mask).squeeze(-1).unsqueeze(-2)

        v_l, _ = pad_to_multiple_and_divide(
            batch["ref_space_uid"].unsqueeze(-1), multiple=n_query, dim=batch["ref_space_uid"].ndim - 1)

        v_m = attn_metadata.query_to_keys(v_l)
        if batch["ref_space_uid"].ndim == 2:
            v_m = v_m.squeeze(1)

        # dlm: [*, N_blocks, N_query, N_key, 3]
        # vlm: [*, N_blocks, N_query, N_key, 1]
        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype) * atom_mask.unsqueeze(-1)

        # Embed pairwise inverse squared distances
        # [*, N_blocks, N_query, N_key, c_atom_pair]
        inv_sq_dists = 1.0 / (1 + torch.sum(dlm**2, dim=-1, keepdim=True))
        ref_pair_input = torch.cat([dlm, inv_sq_dists, vlm], dim=-1)
        plm = self.linear_ref_pair_features(ref_pair_input) * vlm

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
        self.dtype = dtype
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
        zij_trunk = convert_pair_atom_to_blocks(batch=batch,
                                                zij_trunk=zij_trunk,
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
            c_atom_ref_element:
                Reference atom element channel dimension
            c_atom_ref_name_chars:
                Reference atom name characters channel dimension
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            c_s:
                Single representation channel dimension (optional)
            c_z:
                Pair representation channel dimension (optional)
            atom_transformer_config:
                Configuration for the atom transformer
            inf:
                Large number used for attention masking
            eps:
                Small value for numerical stability in layer norms
            add_noisy_pos:
                Whether to add noisy positions and trunk embeddings
        """
        super().__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.inf = inf
        self.dtype = dtype
        self.mapping = mapping
        self.add_noisy_pos = add_noisy_pos
        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            dtype=dtype,
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
                eps=eps
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
                attn_metadata=attn_metadata
            )
        else:
            # Initialize atom single representation when trunk / noisy position
            # inputs are not present
            # [*, N_atom, c_atom]
            ql = cl.clone()

        # Add the combined single conditioning to the pair rep (line 13 - 14)

        cl_l, _ = pad_to_multiple_and_divide(cl, multiple=self.n_query, dim=cl.ndim - 2)

        cl_m = attn_metadata.query_to_keys(cl_l)
        if cl.ndim == 3:
            cl_m = cl_m.squeeze(1)

        atom_mask, _ = pad_to_multiple_and_divide(
            batch["atom_mask"].unsqueeze(-1), multiple=self.n_query, dim=batch["atom_mask"].ndim - 1)
        
        if batch["atom_mask"].ndim == 2:
            atom_mask = atom_mask * attn_metadata.query_to_keys(atom_mask).squeeze(
                1).squeeze(-1).unsqueeze(-2)
        else:
            atom_mask = atom_mask * attn_metadata.query_to_keys(atom_mask).squeeze(-1).unsqueeze(-2)

        # # Note to devs: in previous checkpoints before v13, linear_l and linear_m
        # #  were reversed. Changed it for consistent naming.

        cl_lm = (
            self.linear_l(self.relu(cl_l.unsqueeze(-2)))
            + self.linear_m(self.relu(cl_m.unsqueeze(-3)))
        ) * atom_mask.unsqueeze(-1)
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

        is_contained_diffusion_channel = False
        if ql.ndim == 4:
            is_contained_diffusion_channel = True

        ql, current_size = pad_to_multiple_and_divide(ql,
                                                      multiple=self.n_query,
                                                      dim=ql.ndim - 2)

        if is_contained_diffusion_channel == False:
            ql = ql.unsqueeze(1)

        cl, _ = pad_to_multiple_and_divide(cl,
                                           multiple=self.n_query,
                                           dim=cl.ndim - 2)
        if is_contained_diffusion_channel == False:
            cl = cl.unsqueeze(1)

        atom_mask, _ = pad_to_multiple_and_divide(atom_mask,
                                                  multiple=self.n_query,
                                                  dim=atom_mask.ndim - 1)
        ql = self.atom_transformer(a=ql,
                                   s=cl,
                                   z=plm,
                                   mask=atom_mask,
                                   attn_metadata=attn_metadata)
        if is_contained_diffusion_channel == False:
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

class AtomAttentionDecoder(nn.Module):
    """
    Implements AF3 Algorithm 6.
    """

    def __init__(self,
                 c_atom: int = 128,
                 c_atom_pair: int = 384,
                 c_token: int = 384,
                 c_hidden: int = 32,
                 n_query: int = 32,
                 n_key: int = 128,
                 atom_attn_decoder_config: BaseConfig = None,
                 inf: float = 1e9,
                 eps: float = 1e-5,
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
            c_hidden:
                Hidden channel dimension
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            atom_attn_decoder_config:
                Configuration for the atom attention decoder transformer
            inf:
                Large number used for attention masking
            eps:
                Small value for numerical stability in layer norms
        """
        super().__init__()

        self.inf = inf
        self.dtype = dtype
        self.mapping = mapping
        self.skip_create_weights = skip_create_weights
        self.n_query = n_query
        self.n_key = n_key

        self.linear_q_in = Linear(
            c_token,
            c_atom,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)
        self.atom_transformer = DiffusionTransformer(
            config=atom_attn_decoder_config)

        self.layer_norm = nn.LayerNorm(c_atom,
                                       bias=False,
                                       dtype=self.dtype,
                                       eps=eps)
        self.linear_q_out = Linear(
            c_atom,
            3,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

    def forward(
        self,
        batch: TensorDict,
        atom_mask: torch.Tensor,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format in AtomAttentionEncoder
        Returns:
            rl_update:
                [*, N_atom, 3] Atom position updates
        """
        # Broadcast per-token activations to atoms
        # [*, N_atom, c_atom]
        ql = ql + broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=self.linear_q_in(ai),
            token_dim=-2,
        )

        # Atom transformer
        # [*, N_atom, c_atom]

        ql, current_size = pad_to_multiple_and_divide(ql,
                                                multiple=self.n_query,
                                                dim=ql.ndim - 2)

        cl, _ = pad_to_multiple_and_divide(cl, multiple=self.n_query, dim=cl.ndim - 2)
        atom_mask, _ = pad_to_multiple_and_divide(atom_mask, multiple=self.n_query, dim=atom_mask.ndim - 1)

        ql = self.atom_transformer(
            a=ql,
            s=cl,
            z=plm,
            mask=atom_mask,
            attn_metadata=attn_metadata,
        )
        ql = ql.flatten(2, 3)[:, :, :current_size, :]

        # Compute updates for atom positions
        # [*, N_atom, 3]
        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update
