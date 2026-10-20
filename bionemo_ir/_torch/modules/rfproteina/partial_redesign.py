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
"""Canonical model output for RFProteína existing-structure redesign."""

from __future__ import annotations

from collections.abc import Mapping

import torch

# isort: off
from .atom_utils import condition_pair, require_shape, restore_conditioned_coordinates
from .contracts import RFProteinaCleanStateEncoding, RFProteinaPartialRedesignOutput, RFProteinaPartialSamplingOutput
from .representation_builder import ATOM37_SIZE, CONDITION_COORDINATE, CONDITION_COORDINATE_MASK

# isort: on
__all__ = [
    "RFProteinaCleanStateEncoding",
    "RFProteinaPartialRedesignOutput",
    "materialize_partial_redesign_output",
]


@torch.no_grad()
def materialize_partial_redesign_output(
    decoded_coordinates: torch.Tensor,
    static_features: Mapping[str, torch.Tensor],
    partial_sampling: RFProteinaPartialSamplingOutput,
) -> RFProteinaPartialRedesignOutput:
    """Restore only fixed coordinates; generated sequence remains untouched."""
    if decoded_coordinates.ndim != 4:
        raise ValueError(f"decoded_coordinates must have rank 4, got {tuple(decoded_coordinates.shape)}")
    token_shape = decoded_coordinates.shape[:2]
    expected_atom_shape = (*token_shape, ATOM37_SIZE)
    require_shape(decoded_coordinates, (*expected_atom_shape, 3), "decoded_coordinates")
    conditioned, condition_mask = condition_pair(
        static_features,
        CONDITION_COORDINATE,
        CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )

    fixed_mask = torch.zeros(expected_atom_shape, device=decoded_coordinates.device, dtype=torch.bool)
    coordinates = decoded_coordinates
    if conditioned is not None:
        eligible_mask = torch.ones_like(fixed_mask)
        atom_mask = static_features.get("atom37_mask")
        if atom_mask is not None:
            require_shape(atom_mask, expected_atom_shape, "atom37_mask")
            eligible_mask &= atom_mask.to(device=decoded_coordinates.device, dtype=torch.bool)
        padding_mask = static_features.get("padding_mask")
        if padding_mask is not None:
            require_shape(padding_mask, token_shape, "padding_mask")
            eligible_mask &= padding_mask.to(device=decoded_coordinates.device, dtype=torch.bool).unsqueeze(-1)
        coordinates, fixed_mask = restore_conditioned_coordinates(
            decoded_coordinates,
            conditioned,
            condition_mask,
            intersect_mask=eligible_mask,
            value_name=CONDITION_COORDINATE,
            mask_name=CONDITION_COORDINATE_MASK,
        )

    corruption = partial_sampling.corruption
    return RFProteinaPartialRedesignOutput(
        start_times=dict(partial_sampling.start_times),
        num_steps=partial_sampling.num_steps,
        corruption=corruption,
        raw_coordinates=decoded_coordinates,
        coordinates=coordinates,
        fixed_coordinate_mask=fixed_mask,
    )
