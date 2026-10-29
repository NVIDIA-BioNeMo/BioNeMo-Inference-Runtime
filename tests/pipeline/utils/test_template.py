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
import pytest

from bionemo_ir.pipeline.utils.template import masked_distogram


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_masked_distogram_boundaries(dtype, empty):
    coords = np.array([[[0, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0], [np.nan, 0, 0]]], dtype=dtype)
    mask = np.array([[1, 1, 1, 1, 0]], dtype=np.float32)
    if empty:
        mask.fill(0)
    lower = np.array([0, 1, 4], dtype=dtype)
    destination = np.full((3, 5, 5, 3), -1, dtype=np.float32)
    output = destination[1:2]
    actual = masked_distogram(coords, mask, lower, 100, output=output)
    squared = ((coords[:, :, None] - coords[:, None]) ** 2).sum(-1)
    upper = np.array([1, 4, 100], dtype=dtype)
    expected = ((squared[..., None] > lower) & (squared[..., None] < upper)).astype(np.float32)
    expected *= mask[:, :, None, None] * mask[:, None, :, None]
    assert actual is output
    np.testing.assert_array_equal(actual, expected)
    assert np.all(destination[[0, 2]] == -1)


def test_masked_distogram_chain_restrictions():
    coords = np.array([[[0, 0, 0], [2, 0, 0], [3, 0, 0]]], dtype=np.float32)
    pair_mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    result = masked_distogram(coords, np.ones((1, 3)), np.array([0, 2, 8], dtype=np.float32), 100, pair_mask)
    assert result[0, 0, 1, 1] == 1
    assert result[0, 0, 2].sum() == 0
    assert result.sum() == 2
