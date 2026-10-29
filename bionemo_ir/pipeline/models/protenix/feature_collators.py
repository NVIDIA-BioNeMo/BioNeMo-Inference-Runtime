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

"""Batch Protenix features and attach writer metadata."""

import numpy as np
import torch

from bionemo_ir.pipeline.base import FeatureCollatorBase

from .const import MOL_TYPES
from .feature_generators import structure_context


class ProtenixFeatureCollator(FeatureCollatorBase):
    """Batch model tensors and keep per-token writer identity."""

    def __call__(self, features: dict, context: dict) -> dict:
        output = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        structure = structure_context(context)
        atoms = structure["atoms"]
        centers = [t["center"] for t in structure["tokens"]]
        # The fixed restype vocabulary cannot name arbitrary CCD codes.
        output["token_resnames"] = atoms.res_name[centers].tolist()
        # A declared CCD ligand stays a non-polymer even when it is tokenized as a residue (TYR, 2MG).
        output["token_mol_types"] = np.where(
            atoms.hetero[centers], MOL_TYPES["ccd_ligand"], atoms.mol_type[centers]
        ).tolist()
        output["chain_ids"] = [c["chain_id"] for c in structure["chains"]]
        return output
