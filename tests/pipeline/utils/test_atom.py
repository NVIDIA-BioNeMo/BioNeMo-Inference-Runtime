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

import numpy as np
import torch

from bionemo_ir.pipeline.models.openfold3.common import encode_atom_name_chars as encode_atom_name_chars_openfold3
from bionemo_ir.pipeline.utils.atom import (
    ATOM_NAME_TO_FOLDING_INDEX,
    NUM_FOLDING_ATOM_TYPES,
    decode_atom_name_chars,
    encode_atom_name_chars,
    encode_atom_name_chars_one_hot,
    scatter_flat_atoms_to_folding_layout,
)


def test_atom_name_codec_is_shared_with_openfold3():
    assert encode_atom_name_chars("CA") == [35, 33, 0, 0]
    assert encode_atom_name_chars_openfold3("CA") == encode_atom_name_chars("CA")

    names = ["CA", "C4'", "ZN", "PAD"]
    encoded = encode_atom_name_chars_one_hot(names)
    decoded = decode_atom_name_chars(
        encoded,
        torch.tensor([True, True, True, False]),
    )
    assert decoded == ["CA", "C4'", "ZN", ""]


def test_empty_atom_name_batch_preserves_codec_rank():
    encoded = encode_atom_name_chars_one_hot([])
    assert encoded.shape == (0, 4, 64)
    assert encoded.dtype == torch.int32


def test_scatter_flat_atoms_to_folding_layout():
    positions = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    ownership = np.array([0, 1, 2])
    names = ["CA", "N", "UNKNOWN"]
    active = np.array([True, True, True])

    output, mask = scatter_flat_atoms_to_folding_layout(
        positions,
        ownership,
        names,
        active,
        num_tokens=2,
    )

    assert output.shape == (2, NUM_FOLDING_ATOM_TYPES, 3)
    assert mask.shape == (2, NUM_FOLDING_ATOM_TYPES)
    assert np.array_equal(
        output[0, ATOM_NAME_TO_FOLDING_INDEX["CA"]],
        positions[0],
    )
    assert np.array_equal(
        output[1, ATOM_NAME_TO_FOLDING_INDEX["N"]],
        positions[1],
    )
    assert mask.sum() == 2
