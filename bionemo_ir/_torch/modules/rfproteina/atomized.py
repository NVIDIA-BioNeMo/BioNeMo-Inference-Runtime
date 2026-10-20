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
"""Materialize RFProteína atomized-token outputs without Atomworks."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .atom_utils import condition_pair, optional_feature, require_shape, restore_conditioned_coordinates
from .contracts import RFProteinaAtomizedOutput
from .representation_builder import ATOM37_SIZE, CONDITION_COORDINATE, CONDITION_COORDINATE_MASK

ATOMIZED_TOKEN_TYPE = 3
ATOMIZED_RESIDUE_TYPE = 32
ATOMIZED_COORDINATE_SLOT = 1


@torch.no_grad()
def materialize_atomized_output(
    bb_ca: torch.Tensor,
    static_features: Mapping[str, torch.Tensor],
) -> RFProteinaAtomizedOutput | None:
    """Build a ligand-safe output view from the sampled product-space state.

    Returns ``None`` when the batch has no valid ``token_type == 3`` tokens.
    Absolute coordinate conditions are expressed in ångströms by the feature
    contract and converted to the model's nanometre output units.
    """
    if bb_ca.ndim != 3 or bb_ca.shape[-1] != 3:
        raise ValueError(f"bb_ca must have shape [B, N, 3], got {tuple(bb_ca.shape)}")
    token_shape = bb_ca.shape[:2]
    required = ("padding_mask", "token_type")
    missing = [key for key in required if key not in static_features]
    if missing:
        return None

    device = bb_ca.device
    padding_mask = static_features["padding_mask"]
    token_type = static_features["token_type"]
    require_shape(padding_mask, token_shape, "padding_mask")
    require_shape(token_type, token_shape, "token_type")
    padding_mask = padding_mask.to(device=device, dtype=torch.bool)
    token_type = token_type.to(device=device, dtype=torch.long)
    token_mask = padding_mask & (token_type == ATOMIZED_TOKEN_TYPE)
    if not torch.any(token_mask):
        return None

    required = ("atom37_mask", "residue_type")
    missing = [key for key in required if key not in static_features]
    if missing:
        raise ValueError(f"atomized tokens require static features: {missing}")
    atom37_mask = static_features["atom37_mask"]
    expected_atom_shape = (*token_shape, ATOM37_SIZE)
    require_shape(atom37_mask, expected_atom_shape, "atom37_mask")
    atom_mask = atom37_mask.to(device=device, dtype=torch.bool) & token_mask.unsqueeze(-1)
    occupied_atoms = atom_mask.sum(dim=-1)
    if torch.any(occupied_atoms[token_mask] != 1):
        raise ValueError("each atomized token must occupy exactly one atom37 slot")
    if torch.any(~atom_mask[..., ATOMIZED_COORDINATE_SLOT][token_mask]):
        raise ValueError("atomized token coordinates must occupy atom37 slot 1")

    residue_types = static_features["residue_type"]
    require_shape(residue_types, token_shape, "residue_type")
    residue_types = residue_types.to(device=device, dtype=torch.long)
    if torch.any(residue_types[token_mask] != ATOMIZED_RESIDUE_TYPE):
        bad_values = torch.unique(residue_types[token_mask]).detach().cpu().tolist()
        raise ValueError(f"atomized tokens must use unified residue_type 32; got {bad_values}")

    expanded_bb_ca = bb_ca.unsqueeze(-2).expand(-1, -1, ATOM37_SIZE, -1)
    generated_coordinates = torch.where(atom_mask.unsqueeze(-1), expanded_bb_ca, torch.zeros_like(expanded_bb_ca))
    coordinates = generated_coordinates
    coordinate_condition_mask = torch.zeros_like(atom_mask)
    coordinate, coordinate_mask = condition_pair(
        static_features,
        CONDITION_COORDINATE,
        CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )
    if coordinate is not None:
        coordinates, coordinate_condition_mask = restore_conditioned_coordinates(
            generated_coordinates,
            coordinate,
            coordinate_mask,
            intersect_mask=atom_mask,
            value_name=CONDITION_COORDINATE,
            mask_name=CONDITION_COORDINATE_MASK,
        )

    return RFProteinaAtomizedOutput(
        token_mask=token_mask,
        generated_coordinates=generated_coordinates,
        coordinates=coordinates,
        atom_mask=atom_mask,
        coordinate_condition_mask=coordinate_condition_mask,
        residue_types=residue_types,
        token_type=token_type,
        element=optional_feature(static_features, "element", token_shape, device),
        charge=optional_feature(static_features, "charge", token_shape, device),
        chirality=optional_feature(static_features, "chirality", token_shape, device),
        molecule_iid=optional_feature(static_features, "molecule_iid", token_shape, device),
        chain_iid=optional_feature(static_features, "chain_iid", token_shape, device),
        bond_mask=optional_feature(static_features, "bond_mask", (*token_shape, token_shape[-1]), device),
        bond_order=optional_feature(static_features, "bond_order", (*token_shape, token_shape[-1]), device),
        is_aromatic=optional_feature(static_features, "is_aromatic", (*token_shape, token_shape[-1]), device),
        token_bond_path_length=optional_feature(
            static_features,
            "token_bond_path_length",
            (*token_shape, token_shape[-1]),
            device,
        ),
    )
