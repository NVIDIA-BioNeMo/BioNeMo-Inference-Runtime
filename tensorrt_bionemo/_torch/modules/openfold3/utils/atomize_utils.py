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

from typing import Literal

import torch

def broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    token_feat: torch.Tensor,
    token_dim: int | None = -1,
    max_num_atoms_per_token: int | None = None,
):
    """
    Broadcast token-level features to atom-level features.

    Args:
        token_mask:
            [*, N_token] Token mask
        num_atoms_per_token:
            [*, N_token] Number of atoms per token
        token_feat:
            [*, N_token] Token-level feature
        token_dim:
            Token dimension
        max_num_atoms_per_token:
            Maximum number of atoms per tokenx
    Returns:
        atom_feat:
            [*, N_atom] Broadcasted atom-level feature (if max_num_atoms_per_token
            is provided, the output would be [*, N_token * max_num_atoms_per_token])
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = token_feat.shape[:token_dim]
    feat_dims = token_feat.shape[token_dim:][1:]

    # Apply token mask
    num_atoms_per_token = num_atoms_per_token * token_mask.int()
    token_feat = token_feat * token_mask.reshape(
        (*batch_dims, n_token, *((1,) * len(feat_dims)))
    )

    # Pad atoms at token level
    if max_num_atoms_per_token is not None:
        num_atoms_per_token = torch.stack(
            [num_atoms_per_token, max_num_atoms_per_token - num_atoms_per_token], dim=-1
        ).reshape((*batch_dims, 2 * n_token))
        token_feat = torch.stack(
            [token_feat, torch.zeros_like(token_feat)], dim=token_dim
        ).reshape((*batch_dims, 2 * n_token, *feat_dims))

    # Pad token features
    # Flatten batch and token dimensions
    max_num_atoms = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int()
    padded_token_feat = torch.concat(
        [
            token_feat,
            torch.zeros(
                (*feat_batch_dims, 1, *feat_dims),
                dtype=token_feat.dtype,
                device=token_feat.device,
            ),
        ],
        dim=token_dim,
    ).reshape(-1, *feat_dims)

    # Pad number of atoms per token
    # Flatten batch and token dimensions
    padded_num_atoms_per_token = torch.concat(
        [
            num_atoms_per_token,
            max_num_atoms - torch.sum(num_atoms_per_token, dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    if batch_dims != feat_batch_dims:
        batch_n_repeat = feat_batch_dims[-1]
        padded_num_atoms_per_token = padded_num_atoms_per_token.repeat(
            *((1,) * len(batch_dims[:-1]) + (batch_n_repeat,) + (1,))
        )
    padded_num_atoms_per_token = padded_num_atoms_per_token.reshape(-1).int()

    # Create atom-level features
    atom_feat = torch.repeat_interleave(
        input=padded_token_feat, repeats=padded_num_atoms_per_token, dim=0
    )

    # Unflatten batch and token dimensions
    atom_feat = atom_feat.reshape((*feat_batch_dims, max_num_atoms, *feat_dims))

    return atom_feat


def aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    atom_dim: int | None = -1,
    aggregate_fn: Literal["mean", "sum"] = "mean",
    eps: float = 1e-9,
):
    """
    Aggregate atom-level features to token-level features with mean or sum aggregation.

    Args:
        token_mask:
            [*, N_token] Token mask
        atom_to_token_index:
            [*, N_atom] Mapping from atom to its token index
        atom_mask:
            [*, N_atom] Atom mask
        atom_feat:
            [*, N_atom, *feat_dims] Atom-level features
        atom_dim:
            Atom dimension
        aggregate_fn:
            Function to aggregate atom features into tokens. Possible values are
            "mean" and "sum", where mean is the default.
        eps:
            Small float for numerical stability
    Returns:
        token_feat:
            [*, N_token, *feat_dims] Token-level features
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = atom_feat.shape[:atom_dim]
    feat_dims = atom_feat.shape[atom_dim:][1:]
    atom_feat = atom_feat * atom_mask.reshape(atom_mask.shape + (1,) * len(feat_dims))

    # Mask out atoms that are not part of the structure
    # Padding value must be greater than the largest index so that it
    # is properly excluded from the aggregation
    atom_to_token_index = (
        atom_to_token_index * atom_mask.int()
        + n_token * torch.ones_like(atom_to_token_index) * (1 - atom_mask.int())
    )

    # Prepare atom to token index for aggregation
    # Check for broadcasting and repeat accordingly
    if batch_dims == feat_batch_dims:
        repeated_atom_to_token_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape + (1,) * len(feat_dims)
        ).repeat(*((1,) * (len(batch_dims) + 1) + feat_dims))
    else:
        batch_n_repeat = feat_batch_dims[-1]
        repeated_atom_to_token_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape + (1,) * len(feat_dims)
        ).repeat(*((1,) * (len(batch_dims) - 1) + (batch_n_repeat,) + (1,) + feat_dims))

    if aggregate_fn not in ["mean", "sum"]:
        raise ValueError(f"Invalid aggregation function: {aggregate_fn}")

    # Compute summed token-level feature
    token_feat = torch.zeros(
        (*feat_batch_dims, n_token + 1, *feat_dims),
        device=atom_feat.device,
        dtype=atom_feat.dtype,
    ).scatter_add_(
        index=repeated_atom_to_token_index.long(), src=atom_feat, dim=atom_dim
    )
    token_feat = token_feat.reshape((*feat_batch_dims, n_token + 1, -1))[
        ..., :n_token, :
    ].reshape((*feat_batch_dims, n_token, *feat_dims))

    # Compute mean token-level feature
    if aggregate_fn == "mean":
        # Compute number of atoms (non-masked) per token
        token_num_atoms = torch.zeros(
            (*batch_dims, n_token + 1), device=atom_feat.device, dtype=atom_feat.dtype
        ).scatter_add_(
            index=atom_to_token_index.long(),
            src=atom_mask.to(dtype=atom_feat.dtype),
            dim=-1,
        )[..., :n_token]

        token_feat = token_feat / (
            token_num_atoms.reshape(token_num_atoms.shape + (1,) * len(feat_dims)) + eps
        )

    return token_feat
