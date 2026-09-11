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

import math
from typing import Literal

import torch

from bionemo_ir._torch.layers.sequence_local_atom import (
    aggregate_atom_features_to_tokens as _aggregate_atom_features_to_tokens,
)
from bionemo_ir._torch.layers.sequence_local_atom import (
    broadcast_token_features_to_atoms as _broadcast_token_features_to_atoms,
)
from bionemo_ir._torch.layers.sequence_local_atom import (
    compute_atom_broadcast_index as _compute_atom_broadcast_index,
)
from bionemo_ir._torch.layers.sequence_local_atom import (
    select_atoms_from_padded_tokens as _select_atoms_from_padded_tokens,
)
from bionemo_ir._torch.modules.openfold3.utils.residues import STANDARD_PROTEIN_RESIDUES_ORDER
from bionemo_ir._torch.modules.openfold3.utils.token_atom_constants import (
    TOKEN_TYPES_WITH_GAP,
    atom_name_to_index_by_restype,
)
from bionemo_ir._torch.utils.common import _deterministic_algorithms as _shared_deterministic_algorithms


def _deterministic_algorithms():
    """Compatibility export for existing OpenFold3 determinism tests."""
    return _shared_deterministic_algorithms()


def compute_atom_broadcast_index(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
) -> torch.Tensor:
    """Precompute the token->atom expansion index used by
    :func:`broadcast_token_feat_to_atoms` (its ``max_num_atoms_per_token=None``
    path, with ``feat_batch_dims == batch_dims``).

    The broadcast packs ragged per-token atoms via
    ``torch.repeat_interleave(repeats=<tensor>)``, whose output length is
    data-dependent and so forces a device->host sync — illegal during CUDA-graph
    capture. This returns the equivalent flat gather index so the broadcast can
    instead run a static ``index_select`` inside a captured graph. Because it
    uses ``repeat_interleave``/``max`` itself, it must be called **eagerly**
    (outside capture); the result is constant for a request, so a captured graph
    that consumes it replays correctly.

    Returns a 1-D int index of length ``prod(batch_dims) * max_num_atoms`` whose
    values point into the flattened ``[*, n_token + 1, ...]`` padded token
    features (the final row per batch element is the zero padding row).
    """
    return _compute_atom_broadcast_index(token_mask, num_atoms_per_token)


def broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    token_feat: torch.Tensor,
    token_dim: int | None = -1,
    max_num_atoms_per_token: int | None = None,
    expand_index: torch.Tensor | None = None,
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
            Maximum number of atoms per token
    Returns:
        atom_feat:
            [*, N_atom] Broadcasted atom-level feature (if max_num_atoms_per_token
            is provided, the output would be [*, N_token * max_num_atoms_per_token])
    """
    return _broadcast_token_features_to_atoms(
        token_mask,
        num_atoms_per_token,
        token_feat,
        token_dim=token_dim,
        max_num_atoms_per_token=max_num_atoms_per_token,
        expand_index=expand_index,
    )


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

    The atom->token scatter is accumulated under a deterministic-algorithms context (see
    ``_deterministic_algorithms``), so the result is bit-identical run-to-run;
    the output is cast back to the input dtype before returning. This matters
    because the aggregation runs on every diffusion rollout step, where CUDA
    ``scatter_add_``'s non-deterministic atomic-add order would otherwise
    compound into divergent structures.

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
    return _aggregate_atom_features_to_tokens(
        token_mask,
        atom_to_token_index,
        atom_mask,
        atom_feat,
        atom_dim=atom_dim,
        aggregate_fn=aggregate_fn,
        eps=eps,
    )


def max_atom_per_token_masked_select(
    atom_feat: torch.Tensor,
    max_atom_per_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Select atoms from features padded to max atoms per token.

    Args:
        atom_feat
            [*, N_token * max_atoms_per_token, c_out] Atom features padded to
            max atoms per token
        max_atom_per_token_mask:
            [*, N_token * max_atoms_per_token] Mask denoting valid atoms
    Returns:
        atom_feat:
            [*, N_atom, c_out] Selected valid atom features
    """
    return _select_atoms_from_padded_tokens(atom_feat, max_atom_per_token_mask)


def get_token_representative_atoms(batch: dict, x: torch.Tensor, atom_mask: torch.Tensor):
    """
    Extract representative atoms per token, which returns
        -   Cb for standard amino acid residues (Ca for glycines)
        -   C4 for purines
        -   C2 for pyrimidines
        -   the first and only atom for modified amino acid or nucleotide residues and
            all ligands (which are tokenized per-atom)

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
    Returns:
        rep_x:
            [*, N_token, 3] Representative atom positions
        rep_atom_mask:
            [*, N_token] Representative atom mask
    """
    # Create masks for standard amino acid residues
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_glycine = is_standard_protein * batch["restype"][..., STANDARD_PROTEIN_RESIDUES_ORDER["G"]]

    # Create masks for purines and pyrimadines
    is_standard_dna = batch["is_dna"] * (1 - batch["is_atomized"])
    is_standard_rna = batch["is_rna"] * (1 - batch["is_atomized"])
    is_standard_purine = is_standard_dna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DA")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DG")]
    ) + is_standard_rna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("A")] + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("G")]
    )
    is_standard_pyrimidine = is_standard_dna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DC")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DT")]
    ) + is_standard_rna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("C")] + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("U")]
    )

    # Get index of representative atoms
    restype = batch["restype"].float()
    start_atom_index = batch["start_atom_index"].long()
    cb_atom_index_offset, cb_atom_mask = get_token_atom_index_offset(atom_name="CB", restype=restype)
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(atom_name="CA", restype=restype)
    c4_atom_index_offset, c4_atom_mask = get_token_atom_index_offset(atom_name="C4", restype=restype)
    c2_atom_index_offset, c2_atom_mask = get_token_atom_index_offset(atom_name="C2", restype=restype)
    rep_index = (
        ((start_atom_index + cb_atom_index_offset) * is_standard_protein * (1 - is_standard_glycine))
        + (start_atom_index + ca_atom_index_offset) * is_standard_glycine
        + (start_atom_index + c4_atom_index_offset) * is_standard_purine
        + (start_atom_index + c2_atom_index_offset) * is_standard_pyrimidine
        + start_atom_index * batch["is_atomized"]
    )
    token_atom_mask = (
        cb_atom_mask * is_standard_protein * (1 - is_standard_glycine)
        + ca_atom_mask * is_standard_glycine
        + c4_atom_mask * is_standard_purine
        + c2_atom_mask * is_standard_pyrimidine
        + batch["is_atomized"]
    )

    # Get coordinates of representative atoms
    # [*, N_token, 3]
    rep_x = torch.gather(
        x,
        dim=-2,
        index=rep_index.unsqueeze(-1).expand(*(x.shape[:-2] + (rep_index.shape[-1], 3))).long(),
    )

    # Get representative atom mask
    # [*, N_token]
    rep_atom_mask = (
        torch.gather(
            atom_mask,
            dim=-1,
            index=rep_index.expand(*(atom_mask.shape[:-1] + (rep_index.shape[-1],))).long(),
        )
        * batch["token_mask"]
    ) * token_atom_mask

    return rep_x, rep_atom_mask


def get_token_atom_index_offset(atom_name: str, restype: torch.Tensor):
    """
    Get index of a given atom (within its residue) in each residues.

    Args:
        atom_name:
            Atom name to get indices
        restype:
            [*, N_token, 32] One-hot residue types. Must be float dtype.
    Returns:
        token_atom_index_offset:
            [*, N_token] Atom indices (within their residues) of the given atom name
        token_atom_mask:
            [*, N_token] Atom mask to indicate missing atoms
    """
    token_atom_index_offset = torch.einsum(
        "...k,k->...",
        restype,
        torch.tensor(
            atom_name_to_index_by_restype[atom_name]["index"],
            device=restype.device,
        ).float(),
    ).long()
    token_atom_mask = torch.einsum(
        "...k,k->...",
        restype,
        torch.tensor(
            atom_name_to_index_by_restype[atom_name]["mask"],
            device=restype.device,
        ).float(),
    ).long()
    return token_atom_index_offset, token_atom_mask


def get_token_frame_atoms(
    batch: dict,
    x: torch.Tensor,
    atom_mask: torch.Tensor,
    angle_threshold: float = 25.0,
    eps: float = 1e-8,
    inf: float = 1e9,
):
    """
    Extract frame atoms per token, which returns
        -   (N, Ca, C) for standard amino acid residues
        -   (C3', C1', C4') for standard nucleotide residues
        -   closest neighbors for atomized tokens (modified residues and ligands),
            subject to additional angle and chain constraints from Subsection 4.3.2

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        angle_threshold:
            Angle threshold imposed on frame atom selections for atomized tokens
        eps:
            Small constant for numerical stability
        inf:
            Large constant for numerical stability
    Returns:
        phi:
            ([*, N_token, 3], [*, N_token, 3], [*, N_token, 3])
            Tuple of three frame atoms
        valid_frame_mask:
            [*, N_token] Mask denoting valid frames
    """
    # Create pairwise atom mask
    pair_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # Update pairwise atom mask
    # Restrict to atoms within the same chain
    atom_asym_id = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["asym_id"],
    )
    atom_asym_id_mask = atom_asym_id[..., None] == atom_asym_id[..., None, :]
    pair_mask = pair_mask * atom_asym_id_mask

    # Compute distance matrix
    # [*, N_atom, N_atom]
    d = torch.sum(eps + (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5
    d = d * pair_mask + inf * (1 - pair_mask)

    # Find indices of two closest atoms for start atoms
    # [*, N_token]
    start_atom_index = batch["start_atom_index"].long()
    start_atom_index = start_atom_index.expand(*x.shape[:-2], start_atom_index.shape[-1])
    _, closest_atom_index = torch.topk(d, k=3, dim=-1, largest=False)
    a_index = torch.gather(closest_atom_index[..., 1], dim=-1, index=start_atom_index)
    c_index = torch.gather(closest_atom_index[..., 2], dim=-1, index=start_atom_index)

    # Construct indices of atoms used for frame construction
    # [*, N_token]
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (1 - batch["is_atomized"])

    restype = batch["restype"].float()
    n_atom_index_offset, n_atom_mask = get_token_atom_index_offset(atom_name="N", restype=restype)
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(atom_name="CA", restype=restype)
    c_atom_index_offset, c_atom_mask = get_token_atom_index_offset(atom_name="C", restype=restype)
    c3p_atom_index_offset, c3p_atom_mask = get_token_atom_index_offset(atom_name="C3'", restype=restype)
    c1p_atom_index_offset, c1p_atom_mask = get_token_atom_index_offset(atom_name="C1'", restype=restype)
    c4p_atom_index_offset, c4p_atom_mask = get_token_atom_index_offset(atom_name="C4'", restype=restype)
    frame_atoms = {
        "a": {
            "index": (
                a_index * batch["is_atomized"]
                + (start_atom_index + n_atom_index_offset) * is_standard_protein
                + (start_atom_index + c3p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"] + n_atom_mask * is_standard_protein + c3p_atom_mask * is_standard_nucleotide
            ),
        },
        "b": {
            "index": (
                start_atom_index * batch["is_atomized"]
                + (start_atom_index + ca_atom_index_offset) * is_standard_protein
                + (start_atom_index + c1p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"] + ca_atom_mask * is_standard_protein + c1p_atom_mask * is_standard_nucleotide
            ),
        },
        "c": {
            "index": (
                c_index * batch["is_atomized"]
                + (start_atom_index + c_atom_index_offset) * is_standard_protein
                + (start_atom_index + c4p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"] + c_atom_mask * is_standard_protein + c4p_atom_mask * is_standard_nucleotide
            ),
        },
    }

    # Extract coordinates
    for key in frame_atoms:
        frame_atoms[key].update(
            {
                "atom_positions": torch.gather(
                    x,
                    dim=-2,
                    index=frame_atoms[key]["index"]
                    .unsqueeze(-1)
                    .expand(*(x.shape[:-2] + (frame_atoms[key]["index"].shape[-1], 3)))
                    .long(),
                ),
                "asym_id": torch.gather(
                    atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]),
                    dim=-1,
                    index=frame_atoms[key]["index"].long(),
                ),
                "atom_mask": torch.gather(
                    atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]),
                    dim=-1,
                    index=frame_atoms[key]["index"].long(),
                )
                * batch["token_mask"]
                * frame_atoms[key]["token_atom_mask"],
            }
        )

    # Compute cosine of angles
    u = frame_atoms["a"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    v = frame_atoms["c"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    uv = torch.einsum("...i,...i->...", u, v)
    u_norm = (eps + torch.sum(u**2, dim=-1)) ** 0.5
    v_norm = (eps + torch.sum(v**2, dim=-1)) ** 0.5
    cos_angle = uv / (u_norm * v_norm)

    # Compute valid frame mask from angle constraints
    # (for ligand and non-standard residues)
    cos_angle_min_bound = math.cos((180 - angle_threshold) * math.pi / 180)
    cos_angle_max_bound = math.cos(angle_threshold * math.pi / 180)
    valid_frame_mask_angle = (cos_angle < cos_angle_max_bound) * (cos_angle > cos_angle_min_bound)
    valid_frame_mask_angle = (
        valid_frame_mask_angle * batch["is_atomized"]
        + torch.ones_like(valid_frame_mask_angle) * (1 - batch["is_atomized"])
    ) * batch["token_mask"]

    # Compute valid frame mask from atom mask constraints
    valid_frame_mask_atom = (
        frame_atoms["a"]["atom_mask"] * frame_atoms["b"]["atom_mask"] * frame_atoms["c"]["atom_mask"]
    )

    # Compute valid frame mask from chain constraints
    valid_frame_mask_asym_id = (frame_atoms["a"]["asym_id"] == frame_atoms["b"]["asym_id"]) * (
        frame_atoms["b"]["asym_id"] == frame_atoms["c"]["asym_id"]
    )

    # Compute final valid frame mask
    valid_frame_mask = valid_frame_mask_angle * valid_frame_mask_atom * valid_frame_mask_asym_id
    phi = (
        frame_atoms["a"]["atom_positions"],
        frame_atoms["b"]["atom_positions"],
        frame_atoms["c"]["atom_positions"],
    )

    return phi, valid_frame_mask
