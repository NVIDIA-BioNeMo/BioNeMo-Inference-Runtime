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

"""RFProteína coordinate units, condition handling, and tensor validation."""

from __future__ import annotations

import torch

ANGSTROM_TO_NANOMETER = 0.1


def angstrom_to_nanometer(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert a coordinate tensor from ångströms to nanometres.

    The operation preserves the tensor's shape, dtype, and device.

    Args:
        coordinates: Coordinates in ångströms.

    Returns:
        Coordinates in nanometres.

    Shape:
        Accepts any shape. Structure-model coordinates are typically
        ``[..., N_atom, 3]`` or ``[B, N_token, N_atom, 3]``.

    Used by:
        RFProteina condition preparation and representation construction.
    """
    return coordinates * ANGSTROM_TO_NANOMETER


def finite_atom_mask(coordinates: torch.Tensor) -> torch.Tensor:
    """Mark atoms that carry any finite coordinate component.

    Args:
        coordinates: Coordinate tensor whose last dimension is xyz.

    Returns:
        Boolean mask with the xyz dimension removed.

    Shape:
        ``[..., 3]`` becomes ``[...]``. RFProteina atom37 coordinates are
        ``[B, N, 37, 3]``, so the mask is ``[B, N, 37]``.
    """
    return torch.isfinite(coordinates).any(dim=-1)
