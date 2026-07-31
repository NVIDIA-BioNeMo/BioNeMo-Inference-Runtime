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

import os

import numpy as np

from tensorrt_bionemo.data.schemas.basic import FoldingOutput

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sample_folding_output() -> FoldingOutput:
    sample = np.load(os.path.join(_CURRENT_DIR, "sample_folding_output.npy"),
                     allow_pickle=True).item()

    return FoldingOutput(atom_positions=sample["atom_positions"],
                         residue_types=sample["residue_types"],
                         atom_mask=sample["atom_mask"],
                         residue_indices=sample["residue_indices"],
                         b_factors=sample["b_factors"],
                         chain_indices=sample["chain_indices"])
