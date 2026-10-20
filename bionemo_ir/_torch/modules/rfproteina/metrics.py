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
"""Local tensor metrics for RFProteína search scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from bionemo_ir._torch.modules.rfproteina.atom_utils import NANOMETER_TO_ANGSTROM, finite_atom_mask
from bionemo_ir._torch.modules.rfproteina.atomized import ATOMIZED_TOKEN_TYPE
from bionemo_ir._torch.modules.rfproteina.contracts import RFProteinaPredictionOutput
from bionemo_ir._torch.modules.rfproteina.representation_builder import (
    CONDITION_COORDINATE,
    CONDITION_COORDINATE_MASK,
    CONDITION_SEQUENCE,
    CONDITION_SEQUENCE_MASK,
)

CLASH_METRIC_KEY = "clash_metrics.frac_atoms_clashing_1p5A"
MOTIF_RMSD_METRIC_KEY = "design_motif_allatom_rmsd.rmsd"
SEQUENCE_ADHERENCE_METRIC_KEY = "sequence_adherence.sequence_adherence"
LIGAND_CONTACT_METRIC_KEY = "ligand_contacts.frac_atoms_contacting_4p5A"

_CLASH_THRESHOLD_A = 1.5
_CONTACT_THRESHOLD_A = 4.5
_PROTEIN_TOKEN_TYPE = 0


def _kabsch_rmsd(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Return RMSD in the same units as ``predicted`` / ``target``, ``[M, 3]``."""
    count = predicted.shape[0]
    if count == 0:
        return float("nan")
    if count < 3:
        return float((predicted - target).square().sum(dim=-1).mean().sqrt().item())
    centered_pred = predicted - predicted.mean(dim=0)
    centered_target = target - target.mean(dim=0)
    covariance = centered_pred.transpose(0, 1) @ centered_target
    u_factor, _, vt_factor = torch.linalg.svd(covariance)
    rotation = vt_factor.transpose(0, 1) @ u_factor.transpose(0, 1)
    if torch.det(rotation) < 0:
        vt_factor = vt_factor.clone()
        vt_factor[-1] *= -1
        rotation = vt_factor.transpose(0, 1) @ u_factor.transpose(0, 1)
    aligned = centered_pred @ rotation.transpose(0, 1)
    return float((aligned - centered_target).square().sum(dim=-1).mean().sqrt().item())


def _clash_fraction(
    coordinates_a: torch.Tensor,
    atom_mask: torch.Tensor,
    token_index: torch.Tensor,
    chain_id: torch.Tensor,
    conditioned: torch.Tensor,
    threshold: float = _CLASH_THRESHOLD_A,
) -> float:
    """Fraction of occupied atoms that participate in a steric clash."""
    occupied = atom_mask.to(torch.bool)
    if int(occupied.sum()) < 2:
        return 0.0
    xyz = coordinates_a[occupied]
    tokens = token_index[occupied]
    chains = chain_id[occupied]
    fixed = conditioned[occupied]
    count = xyz.shape[0]
    delta = xyz.unsqueeze(0) - xyz.unsqueeze(1)
    dist = delta.square().sum(dim=-1).sqrt()
    idx_i, idx_j = torch.triu_indices(count, count, offset=1, device=xyz.device)
    pair_dist = dist[idx_i, idx_j]
    same_token = tokens[idx_i] == tokens[idx_j]
    adjacent = ((tokens[idx_i] - tokens[idx_j]).abs() == 1) & (chains[idx_i] == chains[idx_j])
    both_fixed = fixed[idx_i] & fixed[idx_j]
    clashing = (pair_dist < threshold) & ~same_token & ~adjacent & ~both_fixed
    flagged = torch.zeros(count, dtype=torch.bool, device=xyz.device)
    flagged[idx_i[clashing]] = True
    flagged[idx_j[clashing]] = True
    return float(flagged.float().mean().item())


def _ligand_contact_fraction(
    coordinates_a: torch.Tensor,
    atom_mask: torch.Tensor,
    token_type: torch.Tensor,
    padding_mask: torch.Tensor,
    threshold: float = _CONTACT_THRESHOLD_A,
) -> float | None:
    """Fraction of ligand atoms within ``threshold`` Å of a protein atom."""
    ligand_tokens = padding_mask & (token_type == ATOMIZED_TOKEN_TYPE)
    protein_tokens = padding_mask & (token_type == _PROTEIN_TOKEN_TYPE)
    ligand_atoms = ligand_tokens.unsqueeze(-1) & atom_mask
    protein_atoms = protein_tokens.unsqueeze(-1) & atom_mask
    if not bool(ligand_atoms.any()) or not bool(protein_atoms.any()):
        return None
    ligand_xyz = coordinates_a[ligand_atoms]
    protein_xyz = coordinates_a[protein_atoms]
    dist = (ligand_xyz.unsqueeze(1) - protein_xyz.unsqueeze(0)).square().sum(dim=-1).sqrt()
    contacting = dist.min(dim=1).values < threshold
    return float(contacting.float().mean().item())


def evaluate_local_metrics(
    prediction: RFProteinaPredictionOutput,
    static_features: Mapping[str, torch.Tensor],
    batch_index: int,
) -> dict[str, float]:
    """Compute inexpensive tensor metrics for one decoded row.

    Coordinates are converted from nanometres to ångströms. Condition tensors
    keep their upstream units (ångströms for coordinates, unified residue IDs
    for sequence). Missing optional conditions omit their metric keys.
    """
    padding = static_features["padding_mask"][batch_index].to(torch.bool)
    coordinates_nm = prediction.final_coordinates[batch_index]
    atom_mask = prediction.final_atom_mask[batch_index].to(torch.bool) & padding.unsqueeze(-1)
    coordinates_a = coordinates_nm * NANOMETER_TO_ANGSTROM
    tokens, atoms = atom_mask.shape
    token_index = torch.arange(tokens, device=atom_mask.device).unsqueeze(-1).expand(tokens, atoms)
    if "chain_iid" in static_features:
        chain_id = static_features["chain_iid"][batch_index].unsqueeze(-1).expand(tokens, atoms)
    else:
        chain_id = torch.zeros(tokens, atoms, device=atom_mask.device, dtype=torch.long)

    conditioned = torch.zeros_like(atom_mask)
    if CONDITION_COORDINATE_MASK in static_features:
        conditioned = static_features[CONDITION_COORDINATE_MASK][batch_index].to(torch.bool) & atom_mask
    elif CONDITION_COORDINATE in static_features:
        conditioned = finite_atom_mask(static_features[CONDITION_COORDINATE][batch_index]) & atom_mask

    metrics: dict[str, float] = {
        CLASH_METRIC_KEY: _clash_fraction(coordinates_a, atom_mask, token_index, chain_id, conditioned),
    }

    if CONDITION_COORDINATE in static_features:
        if CONDITION_COORDINATE_MASK in static_features:
            motif_mask = static_features[CONDITION_COORDINATE_MASK][batch_index].to(torch.bool) & atom_mask
        else:
            motif_mask = finite_atom_mask(static_features[CONDITION_COORDINATE][batch_index]) & atom_mask
        if bool(motif_mask.any()):
            target = static_features[CONDITION_COORDINATE][batch_index].to(coordinates_a.dtype)
            metrics[MOTIF_RMSD_METRIC_KEY] = _kabsch_rmsd(coordinates_a[motif_mask], target[motif_mask])

    if CONDITION_SEQUENCE in static_features and CONDITION_SEQUENCE_MASK in static_features:
        seq_mask = static_features[CONDITION_SEQUENCE_MASK][batch_index].to(torch.bool) & padding
        if bool(seq_mask.any()):
            predicted_types = prediction.final_residue_types[batch_index]
            target_types = static_features[CONDITION_SEQUENCE][batch_index].to(predicted_types.dtype)
            metrics[SEQUENCE_ADHERENCE_METRIC_KEY] = float(
                (predicted_types[seq_mask] == target_types[seq_mask]).to(torch.float32).mean().item()
            )

    token_type = static_features.get("token_type")
    if token_type is not None:
        contact = _ligand_contact_fraction(
            coordinates_a,
            atom_mask,
            token_type[batch_index],
            padding,
        )
        if contact is not None:
            metrics[LIGAND_CONTACT_METRIC_KEY] = contact
    return metrics


class RFProteinaLocalMetricEvaluator:
    """Default search evaluator: clash, motif RMSD, sequence, ligand contacts."""

    def __call__(
        self,
        prediction: RFProteinaPredictionOutput,
        static_features: Mapping[str, torch.Tensor],
    ) -> Sequence[Mapping[str, float]]:
        """Evaluate every row of a decoded batch."""
        batch = prediction.final_coordinates.shape[0]
        return [evaluate_local_metrics(prediction, static_features, index) for index in range(batch)]
