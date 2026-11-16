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

import torch

from bionemo_ir.data.schemas import residue_vocabulary


def get_residue_atom_mask(residue_type: torch.Tensor, n_residue_types: int, valid: torch.Tensor) -> torch.Tensor:
    """Build encoder test inputs in the checkpoint's residue vocabulary."""
    occupancy = residue_vocabulary(147).atom37_occupancy()
    occupancy = occupancy[1:21] if n_residue_types == 20 else occupancy[:n_residue_types]
    table = torch.zeros(n_residue_types, 37, dtype=torch.bool)
    for index, atoms in enumerate(occupancy):
        table[index, list(atoms)] = True
    return table.to(residue_type.device)[residue_type.clamp_min(0)] & valid.unsqueeze(-1)
