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
"""Tensor-only insertion assignment for unindexed RFProteína motifs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.optimize import linear_sum_assignment

from .atom_utils import condition_pair, prepare_coordinate_condition, require_shape
from .contracts import RFProteinaInsertionOutput
from .groups import CONDITION_COORDINATE_GROUP
from .representation_builder import (
    ATOM37_SIZE,
    CONDITION_COORDINATE,
    CONDITION_COORDINATE_MASK,
    CONDITION_INTERNAL_COORDINATE,
    CONDITION_INTERNAL_COORDINATE_MASK,
    CONDITION_SEQUENCE,
    CONDITION_SEQUENCE_MASK,
    UNSPECIFIED_RESIDUE_TOKEN,
)

CONDITION_INDEX_MASK = "mask_index_1_residue"
UNASSIGNED_TOKEN_INDEX = -1


def _resolve_indexed_tokens(
    index_mask: torch.Tensor,
    atom37_mask: torch.Tensor | None,
    token_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Reduce token- or atom37-level index conditioning to tokens."""
    if index_mask.shape == token_shape:
        return index_mask.to(device=device, dtype=torch.bool)
    if index_mask.shape != (*token_shape, ATOM37_SIZE):
        raise ValueError(f"index_mask must have shape [B, N] or [B, N, 37], got {tuple(index_mask.shape)}")
    if atom37_mask is None:
        raise ValueError("atom37_mask is required with an atom37 index_mask")
    if atom37_mask.shape != index_mask.shape:
        raise ValueError(f"atom37_mask must have shape {tuple(index_mask.shape)}, got {tuple(atom37_mask.shape)}")
    return (index_mask.to(device=device, dtype=torch.bool) & atom37_mask.to(device=device, dtype=torch.bool)).any(
        dim=-1
    )


def assign_unindexed_tokens(
    insertion_logits: torch.Tensor,
    padding_mask: torch.Tensor,
    index_mask: torch.Tensor,
    chain_iid: torch.Tensor,
    atom37_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assign each unindexed token to a distinct same-chain generated token.

    The returned ``[B, N]`` tensor stores the assigned generated tensor index
    for each unindexed query token. Indexed, padded, and unmatched tokens carry
    :data:`UNASSIGNED_TOKEN_INDEX`. The assignment is discrete and detached
    from autograd. An atom37 ``index_mask`` requires ``atom37_mask`` so its
    default-filled nonexistent atom slots do not make unindexed tokens appear
    indexed.
    """
    if insertion_logits.ndim != 3:
        raise ValueError(f"insertion_logits must have shape [B, N, N], got {tuple(insertion_logits.shape)}")
    batch, tokens, columns = insertion_logits.shape
    if columns != tokens:
        raise ValueError(f"insertion_logits must be square in token dimensions, got {tuple(insertion_logits.shape)}")
    token_shape = (batch, tokens)
    if padding_mask.shape != token_shape:
        raise ValueError(f"padding_mask must have shape {token_shape}, got {tuple(padding_mask.shape)}")
    if chain_iid.shape != token_shape:
        raise ValueError(f"chain_iid must have shape {token_shape}, got {tuple(chain_iid.shape)}")

    device = insertion_logits.device
    valid = padding_mask.to(device=device, dtype=torch.bool)
    is_indexed = _resolve_indexed_tokens(index_mask, atom37_mask, token_shape, device)
    chain_iid = chain_iid.to(device=device)
    assignment = torch.full(token_shape, UNASSIGNED_TOKEN_INDEX, dtype=torch.long, device=device)

    for batch_index in range(batch):
        generated = valid[batch_index] & is_indexed[batch_index]
        unindexed = valid[batch_index] & ~is_indexed[batch_index]
        if not torch.any(unindexed):
            continue

        unindexed_chains = torch.unique(chain_iid[batch_index][unindexed]).tolist()
        for chain in unindexed_chains:
            query_indices = torch.nonzero(
                unindexed & (chain_iid[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            candidate_indices = torch.nonzero(
                generated & (chain_iid[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if candidate_indices.numel() == 0:
                continue

            score = insertion_logits[batch_index].index_select(0, query_indices).index_select(1, candidate_indices)
            rows, columns = linear_sum_assignment(score.detach().float().cpu().numpy(), maximize=True)
            rows = torch.as_tensor(rows, device=device)
            columns = torch.as_tensor(columns, device=device)
            assignment[batch_index, query_indices[rows]] = candidate_indices[columns]

    return assignment


def _reverse_assignment(
    assignment: torch.Tensor,
    valid: torch.Tensor,
    is_indexed: torch.Tensor,
    chain_iid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate source-to-target assignments and return their sparse indices."""
    token_shape = valid.shape
    if assignment.shape != token_shape:
        raise ValueError(f"assignment must have shape {tuple(token_shape)}, got {tuple(assignment.shape)}")
    assignment = assignment.to(device=valid.device, dtype=torch.long)
    mapped = assignment >= 0
    batch_indices, source_indices = torch.nonzero(mapped, as_tuple=True)
    target_indices = assignment[batch_indices, source_indices]
    if target_indices.numel() and torch.any(target_indices >= token_shape[1]):
        raise ValueError("assignment contains an out-of-range target index")

    generated = valid & is_indexed
    unindexed = valid & ~is_indexed
    if source_indices.numel():
        if torch.any(~unindexed[batch_indices, source_indices]):
            raise ValueError("only valid unindexed tokens may carry assignments")
        if torch.any(~generated[batch_indices, target_indices]):
            raise ValueError("assignments must target valid indexed generated tokens")
        if torch.any(chain_iid[batch_indices, source_indices] != chain_iid[batch_indices, target_indices]):
            raise ValueError("assignments must stay within each chain")
        for batch_index in range(token_shape[0]):
            batch_targets = target_indices[batch_indices == batch_index]
            if torch.unique(batch_targets).numel() != batch_targets.numel():
                raise ValueError("multiple unindexed tokens target the same generated token")

    return assignment, batch_indices, source_indices, target_indices


def _project_atom_condition(
    static_features: Mapping[str, torch.Tensor],
    value_key: str,
    mask_key: str,
    generated_token_mask: torch.Tensor,
    batch_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    coordinate_reference: torch.Tensor,
    *,
    finite_marks_conditioned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project direct and assigned atom conditions onto generated tokens."""
    values, mask = condition_pair(
        static_features,
        value_key,
        mask_key,
        finite_marks_conditioned=finite_marks_conditioned,
    )
    if values is None:
        return (
            torch.zeros_like(coordinate_reference),
            torch.zeros(coordinate_reference.shape[:-1], dtype=torch.bool, device=coordinate_reference.device),
        )

    token_shape = generated_token_mask.shape
    expected_value_shape = (*token_shape, ATOM37_SIZE, 3)
    expected_mask_shape = (*token_shape, ATOM37_SIZE)
    require_shape(values, expected_value_shape, value_key)
    require_shape(mask, expected_mask_shape, mask_key)

    atom37_mask = static_features["atom37_mask"]
    require_shape(atom37_mask, expected_mask_shape, "atom37_mask")
    values, mask = prepare_coordinate_condition(
        values,
        mask,
        coordinate_reference,
        intersect_mask=atom37_mask,
        value_name=value_key,
        mask_name=mask_key,
    )

    projected_mask = mask & generated_token_mask.unsqueeze(-1)
    projected_values = torch.where(projected_mask.unsqueeze(-1), values, torch.zeros_like(values))
    if source_indices.numel():
        source_mask = mask[batch_indices, source_indices]
        source_values = values[batch_indices, source_indices]
        current_values = projected_values[batch_indices, target_indices]
        current_mask = projected_mask[batch_indices, target_indices]
        projected_values[batch_indices, target_indices] = torch.where(
            source_mask.unsqueeze(-1), source_values, current_values
        )
        projected_mask[batch_indices, target_indices] = current_mask | source_mask

    return projected_values, projected_mask


def _project_sequence_condition(
    static_features: Mapping[str, torch.Tensor],
    generated_token_mask: torch.Tensor,
    batch_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    residue_types: torch.Tensor,
    condition_sequence_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply direct and assigned sequence conditions to generated tokens."""
    values, mask = condition_pair(
        static_features,
        CONDITION_SEQUENCE,
        CONDITION_SEQUENCE_MASK,
        unspecified_token=UNSPECIFIED_RESIDUE_TOKEN,
    )
    if values is None:
        return residue_types.clone(), torch.zeros_like(generated_token_mask)

    token_shape = generated_token_mask.shape
    if values.shape != token_shape:
        raise ValueError(f"{CONDITION_SEQUENCE} must have shape {token_shape}, got {tuple(values.shape)}")
    if mask.shape != token_shape:
        raise ValueError(f"{CONDITION_SEQUENCE_MASK} must have shape {token_shape}, got {tuple(mask.shape)}")

    device = residue_types.device
    values = values.to(device=device, dtype=residue_types.dtype) + condition_sequence_offset
    mask = mask.to(device=device, dtype=torch.bool)
    projected_mask = mask & generated_token_mask
    output = torch.where(projected_mask, values, residue_types)
    if source_indices.numel():
        source_mask = mask[batch_indices, source_indices]
        source_values = values[batch_indices, source_indices]
        current_values = output[batch_indices, target_indices]
        current_mask = projected_mask[batch_indices, target_indices]
        output[batch_indices, target_indices] = torch.where(source_mask, source_values, current_values)
        projected_mask[batch_indices, target_indices] = current_mask | source_mask
    return output, projected_mask


def _project_internal_groups(
    static_features: Mapping[str, torch.Tensor],
    internal_mask: torch.Tensor,
    generated_token_mask: torch.Tensor,
    batch_indices: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Project coordinate-group IDs alongside internal-coordinate references."""
    if not torch.any(internal_mask):
        return torch.full_like(internal_mask, -1, dtype=torch.long, device=internal_mask.device)
    if CONDITION_COORDINATE_GROUP not in static_features:
        raise ValueError(f"{CONDITION_COORDINATE_GROUP} is required for internal-coordinate insertion")
    groups = static_features[CONDITION_COORDINATE_GROUP]
    if groups.shape != internal_mask.shape:
        raise ValueError(
            f"{CONDITION_COORDINATE_GROUP} must have shape {tuple(internal_mask.shape)}, got {tuple(groups.shape)}"
        )
    groups = groups.to(device=internal_mask.device, dtype=torch.long)
    source_mask = static_features[CONDITION_INTERNAL_COORDINATE_MASK].to(device=internal_mask.device, dtype=torch.bool)
    source_mask &= static_features["atom37_mask"].to(device=internal_mask.device, dtype=torch.bool)
    projected = torch.full_like(groups, -1)
    direct_mask = source_mask & generated_token_mask.unsqueeze(-1)
    projected[direct_mask] = groups[direct_mask]
    if source_indices.numel():
        mapped_source_mask = source_mask[batch_indices, source_indices]
        source_groups = groups[batch_indices, source_indices]
        current_groups = projected[batch_indices, target_indices]
        projected[batch_indices, target_indices] = torch.where(mapped_source_mask, source_groups, current_groups)
    if torch.any(projected[internal_mask] < 1):
        raise ValueError("internal-coordinate atoms must carry a positive coordinate group")
    return projected


def _superimpose_reference(mobile: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Rigidly superimpose reference points onto generated points."""
    output_dtype = mobile.dtype
    mobile = mobile.float()
    fixed = fixed.float()
    mobile_center = mobile.mean(dim=0, keepdim=True)
    fixed_center = fixed.mean(dim=0, keepdim=True)
    mobile_centered = mobile - mobile_center
    fixed_centered = fixed - fixed_center
    covariance = mobile_centered.transpose(0, 1) @ fixed_centered
    left, _, right_t = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=mobile.device, dtype=mobile.dtype)
    correction[-1, -1] = torch.where(
        torch.linalg.det(left @ right_t) < 0,
        correction.new_tensor(-1),
        correction.new_tensor(1),
    )
    rotation = left @ correction @ right_t
    return ((mobile_centered @ rotation) + fixed_center).to(dtype=output_dtype)


def _align_internal_groups(
    reference: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor,
    generated_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Align each internal-coordinate group to its generated target atoms."""
    aligned = reference.clone()
    for batch_index in range(reference.shape[0]):
        group_ids = torch.unique(groups[batch_index][mask[batch_index]]).tolist()
        for group_id in group_ids:
            group_mask = mask[batch_index] & (groups[batch_index] == group_id)
            mobile = reference[batch_index][group_mask]
            fixed = generated_coordinates[batch_index][group_mask]
            if not torch.isfinite(mobile).all() or not torch.isfinite(fixed).all():
                raise ValueError("internal-coordinate alignment requires finite points")
            aligned[batch_index][group_mask] = _superimpose_reference(mobile, fixed)
    return aligned


@torch.no_grad()
def apply_insertion_assignment(
    coordinates: torch.Tensor,
    residue_types: torch.Tensor,
    assignment: torch.Tensor,
    static_features: Mapping[str, torch.Tensor],
    condition_sequence_offset: int = 0,
) -> RFProteinaInsertionOutput:
    """Materialize indexed and assigned motif conditions as a tensor view.

    Raw predictions remain unchanged. The returned coordinates mirror OSS
    ``coords-with-insertion`` semantics: absolute references are copied in
    directly, internal-coordinate groups are rigidly aligned to generated
    targets, and unconditioned atoms retain model coordinates.
    ``condition_sequence_offset`` converts conditioned sequence IDs into the
    decoder output space (``-1`` for the legacy 20-class decoder and ``0`` for
    a unified decoder).
    """
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (ATOM37_SIZE, 3):
        raise ValueError(f"coordinates must have shape [B, N, 37, 3], got {tuple(coordinates.shape)}")
    token_shape = coordinates.shape[:2]
    if residue_types.shape != token_shape:
        raise ValueError(f"residue_types must have shape {token_shape}, got {tuple(residue_types.shape)}")
    required = ("padding_mask", CONDITION_INDEX_MASK, "atom37_mask", "chain_iid")
    missing = [key for key in required if key not in static_features]
    if missing:
        raise ValueError(f"static_features are missing insertion keys: {missing}")

    device = coordinates.device
    valid = static_features["padding_mask"].to(device=device, dtype=torch.bool)
    if valid.shape != token_shape:
        raise ValueError(f"padding_mask must have shape {token_shape}, got {tuple(valid.shape)}")
    atom37_mask = static_features["atom37_mask"]
    expected_atom37_shape = (*token_shape, ATOM37_SIZE)
    if atom37_mask.shape != expected_atom37_shape:
        raise ValueError(f"atom37_mask must have shape {expected_atom37_shape}, got {tuple(atom37_mask.shape)}")
    is_indexed = _resolve_indexed_tokens(
        static_features[CONDITION_INDEX_MASK],
        atom37_mask,
        token_shape,
        device,
    )
    chain_iid = static_features["chain_iid"].to(device=device)
    if chain_iid.shape != token_shape:
        raise ValueError(f"chain_iid must have shape {token_shape}, got {tuple(chain_iid.shape)}")
    assignment, batch_indices, source_indices, target_indices = _reverse_assignment(
        assignment, valid, is_indexed, chain_iid
    )
    generated_token_mask = valid & is_indexed

    source_token_index = torch.full(token_shape, UNASSIGNED_TOKEN_INDEX, dtype=torch.long, device=device)
    source_token_index[batch_indices, target_indices] = source_indices
    target_mask = source_token_index >= 0

    absolute_reference, absolute_mask = _project_atom_condition(
        static_features,
        CONDITION_COORDINATE,
        CONDITION_COORDINATE_MASK,
        generated_token_mask,
        batch_indices,
        source_indices,
        target_indices,
        coordinates,
        finite_marks_conditioned=True,
    )
    internal_reference, internal_mask = _project_atom_condition(
        static_features,
        CONDITION_INTERNAL_COORDINATE,
        CONDITION_INTERNAL_COORDINATE_MASK,
        generated_token_mask,
        batch_indices,
        source_indices,
        target_indices,
        coordinates,
    )

    coordinates_with_insertion = torch.where(absolute_mask.unsqueeze(-1), absolute_reference, coordinates)
    if torch.any(internal_mask):
        internal_groups = _project_internal_groups(
            static_features,
            internal_mask,
            generated_token_mask,
            batch_indices,
            source_indices,
            target_indices,
        )
        aligned_internal = _align_internal_groups(internal_reference, internal_mask, internal_groups, coordinates)
        coordinates_with_insertion = torch.where(
            internal_mask.unsqueeze(-1), aligned_internal, coordinates_with_insertion
        )

    residue_types_with_insertion, sequence_mask = _project_sequence_condition(
        static_features,
        generated_token_mask,
        batch_indices,
        source_indices,
        target_indices,
        residue_types,
        condition_sequence_offset,
    )
    return RFProteinaInsertionOutput(
        assignment=assignment,
        source_token_index=source_token_index,
        generated_token_mask=generated_token_mask,
        target_mask=target_mask,
        coordinates=coordinates_with_insertion,
        coordinate_mask=absolute_mask | internal_mask,
        residue_types=residue_types_with_insertion,
        sequence_mask=sequence_mask,
    )
