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
"""Token-level groupings behind RFProteína's per-group stochastic centering.

The ``per_group`` centering scope translates every token into a group id and
gives each group its own random offset, so that independently placed chains or
coordinate groups do not all move together. Extraction lives here rather than in
the sampler because the group tensors are static features, and the sampler
should not need to know how a batch spells them.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

#: Atom-level coordinate-group condition; a token belongs to the highest group
#: any of its atoms carries.
CONDITION_COORDINATE_GROUP = "condition_CoordinateGroup_1_atom"

#: Group tensors the ``per_group`` scope may be keyed on.
GROUP_KEYS = frozenset({"chain_iid", "pn_unit_iid", "molecule_iid", "coord_group"})


def extract_groups(
    static_features: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    """Collect the token-level group tensors a batch happens to carry.

    Args:
        static_features: Batch of per-token features.

    Returns:
        Every name in :data:`GROUP_KEYS` mapped to its ``[batch, tokens]``
        tensor, or ``None`` when the batch does not carry it. Absent keys stay
        in the mapping so a scope that needs one can report it by name.
    """
    coordinate_group = static_features.get(CONDITION_COORDINATE_GROUP)
    if coordinate_group is not None:
        # [batch, tokens, 37] atom-level -> token-level.
        coordinate_group = coordinate_group.long().max(dim=-1).values

    groups: dict[str, torch.Tensor | None] = {"coord_group": coordinate_group}
    for key in ("chain_iid", "pn_unit_iid", "molecule_iid"):
        value = static_features.get(key)
        groups[key] = None if value is None else value.long()
    return groups


def compose_group_ids(
    groups: Mapping[str, torch.Tensor | None] | None,
    keys: tuple[str, ...],
) -> torch.Tensor:
    """Map each unique combination of ``keys`` to one integer id per token.

    Ids are assigned over the whole batch, so the same combination in two batch
    elements shares an id. That is what makes the per-group offsets line up
    across a batch of replicates.

    Args:
        groups: Group tensors as returned by :func:`extract_groups`.
        keys: Group names to combine, each of which must be present.

    Returns:
        ``[batch, tokens]`` integer group ids.

    Raises:
        ValueError: If ``keys`` is empty or names a group the batch lacks.
    """
    if not keys:
        raise ValueError("compose_group_ids needs at least one group key")
    groups = groups or {}
    tensors = []
    for key in keys:
        value = groups.get(key)
        if value is None:
            raise ValueError(f"per-group stochastic centering requested {key!r}, which this batch does not provide")
        tensors.append(value.long())
    stacked = torch.stack(tensors, dim=-1)
    _, inverse = torch.unique(stacked.reshape(-1, stacked.shape[-1]), dim=0, return_inverse=True)
    return inverse.reshape(stacked.shape[:-1])
