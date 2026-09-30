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
"""Template distogram matches the dense reference for every input dtype."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3.common import create_template_distogram


def _reference(coords, pb_mask, pair_mask, min_bin, max_bin, n_bins, inf_value) -> torch.Tensor:
    coords = np.asarray(coords)
    distogram = np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1, keepdims=True)
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)], axis=-1)
    binned = torch.tensor(((distogram > lower) * (distogram < upper)).astype(distogram.dtype), dtype=torch.float32)
    pair = (pb_mask[..., None] * pb_mask[..., None, :])[..., None]
    return binned * pair * pair_mask


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16, np.int64])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.float64])
def test_matches_dense_reference(dtype: type, mask_dtype: torch.dtype) -> None:
    rng = np.random.default_rng(0)
    coords = (rng.normal(size=(2, 40, 3)) * rng.choice([0.1, 1.0, 10.0], size=(2, 40, 3))).astype(dtype)
    pb_mask = torch.from_numpy(rng.random((2, 40)) > 0.2).to(mask_dtype)
    chains = rng.integers(0, 3, 40)
    pair_mask = torch.from_numpy(chains[:, None] == chains[None, :])[None, ..., None].to(mask_dtype)
    args = (coords, pb_mask, pair_mask, 3.25, 50.75, 39, 1e8)
    got, expected = create_template_distogram(*args), _reference(*args)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert torch.equal(got, expected)
