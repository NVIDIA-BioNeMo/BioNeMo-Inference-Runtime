# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import torch
from torch import nn

from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, CONTACT_PROB, ChunkPolicy, chunk_apply
from tensorrt_bionemo.pipeline.models.boltz2.const import chain_type_ids

# Number of leading (nearest) distogram bins whose probability mass counts as a token-pair contact.
NUM_CONTACT_BINS = 20


def repeat_with_multiplicity(tensor: torch.Tensor, multiplicity: int) -> torch.Tensor:
    """Repeat a tensor with multiplicity."""
    return tensor.unsqueeze(1).repeat_interleave(multiplicity, 1)


def compute_distogram(
    x_pred: torch.Tensor,
    boundaries: torch.Tensor,
    token_to_rep_atom: torch.Tensor,
    multiplicity: int = 1,
    dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the distogram from the predicted atom coordinates.
    Args:
        x_pred: (B, mult, N_atoms, 3)
        boundaries: (num_dist_bins - 1,)
        token_to_rep_atom: (B, N_tokens, N_atoms)
        multiplicity: int
    Returns:
        d: (B, mult, N_tokens, N_tokens) distances between the tokens' representative atoms
        distogram: (B, mult, N_tokens, N_tokens) index of the bin each distance falls in
    """
    if len(x_pred.shape) == 4:
        B, mult, N, _ = x_pred.shape
    else:
        BM, N, _ = x_pred.shape
        B = BM // multiplicity
        mult = multiplicity
        x_pred = x_pred.view(B, mult, N, -1)
    x_pred_repr = torch.einsum("bij,bmjk->bmik", token_to_rep_atom.float(), x_pred)
    d = torch.cdist(x_pred_repr, x_pred_repr)  # [B, mult, N_tokens, N_tokens]
    distogram = (d.unsqueeze(-1) > boundaries).sum(dim=-1).to(dtype).long()  # [B, mult, N_tokens, N_tokens]
    return d, distogram


def _contact_prob_dense(logits: torch.Tensor, contacts: torch.Tensor) -> torch.Tensor:
    """Contact probability for one row-slice of ``logits`` (see :func:`compute_contact_prob`)."""
    return (torch.softmax(logits.float(), dim=-1) * contacts).sum(-1)


def compute_contact_prob(
    logits: torch.Tensor,
    num_contact_bins: int = NUM_CONTACT_BINS,
    policy: ChunkPolicy | None = None,
) -> torch.Tensor:
    """Reduce predicted distogram logits to the probability mass within the contact cutoff.

    This is the only form in which the confidence heads read the predicted distogram, so call it at
    the producer and never hold the ``[B, N, N, num_bins]`` logits. Row-chunked because ``softmax``
    is independent per pair row, so a dense call would transiently allocate what this saves.

    Args:
        logits: (B, N_tokens, N_tokens, num_bins) predicted distogram logits.
        num_contact_bins: number of leading (nearest) distance bins that count as a contact.
        policy: row-chunking policy; defaults to the ``contact_prob`` registry entry.

    Returns:
        (B, N_tokens, N_tokens) fp32 probability that a token pair is in contact.
    """
    if policy is None:
        policy = CHUNK_REGISTRY.get(CONTACT_PROB)
    # Full-width 0/1 mask rather than a slice, so the sum over the bin dim stays bit-identical to the
    # heads' original mask-then-sum.
    contacts = logits.new_zeros(logits.shape[-1], dtype=torch.float32)
    contacts[:num_contact_bins] = 1.0
    return chunk_apply(_contact_prob_dense, logits, policy=policy, contacts=contacts)


def compute_aggregated_metric(logits: torch.Tensor, end: float = 1.0) -> torch.Tensor:
    """Compute the metric from the logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits of the metric
    end : float
        Max value of the metric, by default 1.0

    Returns
    -------
    Tensor
        The metric value

    """
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(start=0.5 * bin_width, end=end, step=bin_width, device=logits.device)
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def tm_function(d, Nres):
    """Compute the rescaling function for pTM.

    Parameters
    ----------
    d : torch.Tensor
        The input
    Nres : torch.Tensor
        The number of residues

    Returns
    -------
    Tensor
        Output of the function

    """
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(
    logits: torch.Tensor, x_preds: torch.Tensor, feats: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, dict[int, torch.Tensor]]]:
    """Compute pTM and ipTM scores.

    Each score is reduced over both token dims (a masked mean over one, then a max over the other),
    so all four are per-sample scalars.

    Args
        logits : torch.Tensor
            pae logits. Shape, [B, mult, N_tokens, N_tokens, num_dist_bins].
        x_preds : torch.Tensor
            The predicted coordinates. Shape, [B, mult, N_atoms, 3].
        feats : Dict[str, torch.Tensor]
            The input features.

    Returns:
        pTM score: torch.Tensor
            pTM score. Shape, [B, mult].
        ipTM score: torch.Tensor
            ipTM score. Shape, [B, mult].
        ligand ipTM score: torch.Tensor
            ligand ipTM score. Shape, [B, mult].
        protein ipTM score: torch.Tensor
            protein ipTM score. Shape, [B, mult].
        pair chain ipTM score: dict[int, dict[int, torch.Tensor]]
            Per chain pair, keyed asym_id -> asym_id. Each entry has shape [B, mult].

    """
    B, multiplicity, _, _ = x_preds.shape
    # Compute mask for collinear and overlapping tokens
    # [B, mult, N_tokens]
    _, mask_collinear_pred = compute_frame_pred(x_preds, feats["frames_idx"], feats)
    maski = mask_collinear_pred.unsqueeze(-1)
    mask_pad = repeat_with_multiplicity(feats["token_pad_mask"], multiplicity)
    N_res = mask_pad.sum(dim=-1, keepdim=True)

    mask_pad_l = mask_pad.unsqueeze(-2)
    mask_pad_r = mask_pad.unsqueeze(-1)

    asym_id = repeat_with_multiplicity(feats["asym_id"], multiplicity)
    asym_id_l = asym_id.unsqueeze(-2)
    asym_id_r = asym_id.unsqueeze(-1)

    # [B, mult, N_tokens, N_tokens]
    pair_mask_ptm = maski * mask_pad_l * mask_pad_r
    # [B, mult, N_tokens, N_tokens]
    pair_mask_iptm = pair_mask_ptm * (asym_id_l != asym_id_r)

    # Extract pae values
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    # [1, 1, num_dist_bins]
    pae_value = torch.arange(start=0.5 * bin_width, end=end, step=bin_width, device=logits.device)[None, None, :]

    # compute pTM and ipTM
    tm_value = tm_function(pae_value, N_res).unsqueeze(-2).unsqueeze(-2)
    probs = nn.functional.softmax(logits, dim=-1)

    # shape (B, mult, N, N)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )
    ptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_ptm, dim=-1) / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
        dim=-1,
    ).values
    iptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_iptm, dim=-1) / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
        dim=-1,
    ).values

    # compute ligand and protein ipTM
    token_type = feats["mol_type"]
    token_type = repeat_with_multiplicity(token_type, multiplicity)
    is_ligand_token = (token_type == chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == chain_type_ids["PROTEIN"]).float()

    # [B, mult, N_tokens, N_tokens]
    ligand_iptm_mask = pair_mask_iptm * (
        (is_ligand_token.unsqueeze(-2) * is_protein_token.unsqueeze(-1))
        + (is_protein_token.unsqueeze(-2) * is_ligand_token.unsqueeze(-1))
    )

    # [B, mult, N_tokens, N_tokens]
    protein_ipmt_mask = pair_mask_iptm * (is_protein_token.unsqueeze(-2) * is_protein_token.unsqueeze(-1))

    ligand_iptm = torch.max(
        torch.sum(tm_expected_value * ligand_iptm_mask, dim=-1) / (torch.sum(ligand_iptm_mask, dim=-1) + 1e-5),
        dim=-1,
    ).values
    protein_iptm = torch.max(
        torch.sum(tm_expected_value * protein_ipmt_mask, dim=-1) / (torch.sum(protein_ipmt_mask, dim=-1) + 1e-5),
        dim=-1,
    ).values

    # Compute pair chain ipTM
    chain_pair_iptm = {}
    asym_ids_list = torch.unique(asym_id).tolist()
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = pair_mask_ptm * (asym_id_l == idx1) * (asym_id_r == idx2)

            chain_iptm[idx2] = torch.max(
                torch.sum(tm_expected_value * mask_pair_chain, dim=-1) / (torch.sum(mask_pair_chain, dim=-1) + 1e-5),
                dim=-1,
            ).values
        chain_pair_iptm[idx1] = chain_iptm

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm


def compute_collinear_mask(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute the mask for collinear or overlapping atoms.

    Args:
        v1: torch.Tensor
            The first vector. Shape, [B, mult, N, 3]
        v2: torch.Tensor
            The second vector. Shape, [B, mult, N, 3]
    Returns:
        mask: torch.Tensor
            The mask for collinear or overlapping atoms. Shape, [B, mult, N, N]
    """
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords: torch.Tensor,
    frames_idx_true: torch.Tensor,
    feats: dict[str, torch.Tensor],
    resolved_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        pred_atom_coords: torch.Tensor
            The predicted atom coordinates. Shape, [B, mult, N_tokens, 3]
        frames_idx_true: torch.Tensor
            The true frames indices. Shape, [B, mult, N_tokens, 3]
        feats: dict[str, torch.Tensor]
            The input features. Shape, [B, mult, N_tokens, ...]
        resolved_mask: Optional[torch.Tensor]
            The resolved mask. Shape, [B, mult, N_tokens, N_atoms]
    Returns:
        frames_idx_pred: torch.Tensor
            The predicted frames indices. Shape, [B, mult, N_tokens, 3]
        mask_collinear_pred: torch.Tensor
            The mask for collinear or overlapping atoms. Shape, [B, mult, N_tokens]
    """
    # extract necessary features
    asym_id_token = feats["asym_id"]
    asym_id_atom = torch.bmm(feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()).squeeze(-1)
    B, multiplicity, N, _ = pred_atom_coords.shape
    frames_idx_pred = repeat_with_multiplicity(frames_idx_true, multiplicity)

    # Iterate through the batch and update the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        # pred_atom_coord: (mult, N, 3)
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if feats["mol_type"][i, token_idx] != chain_type_ids["NONPOLYMER"] or num_atoms < 3:
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5

            # Sort the atoms by distance
            resolved_pair = 1 - (
                feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
            ).to(torch.float32)
            resolved_pair[resolved_pair == 1] = torch.inf
            indices = torch.sort(dist_mat + resolved_pair, axis=2).indices

            # Compute the frames
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            token_idx += num_tokens
            atom_idx += num_atoms

    # Expand the frames with the multiplicity
    frames_expanded = pred_atom_coords[
        torch.arange(0, B, 1)[:, None, None, None].to(frames_idx_pred.device),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(frames_idx_pred.device),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinear or overlapping atoms in the frame
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B, multiplicity, -1)

    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


def concat_out_dicts(out_dicts: dict[str, Any]) -> dict[str, Any]:
    """Concatenate the output dictionaries.

    Args:
        out_dicts: dict[str, Any]
            The output dictionaries.
    Returns:
        out_dict: dict[str, Any]
            The concatenated output dictionary.
    """
    out_dict = {}
    for key in out_dicts[0]:
        if key != "pair_chains_iptm":
            out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=1)
        else:
            pair_chains_iptm = {}
            for chain_idx1 in out_dicts[0][key]:
                chains_iptm = {}
                for chain_idx2 in out_dicts[0][key][chain_idx1]:
                    chains_iptm[chain_idx2] = torch.cat(
                        [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                        dim=1,
                    )
                pair_chains_iptm[chain_idx1] = chains_iptm
            out_dict[key] = pair_chains_iptm
    return out_dict
