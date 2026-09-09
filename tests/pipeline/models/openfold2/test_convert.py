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

"""Checkpoint-layout handling in the OpenFold2 weight converter."""

import pytest
import torch

from bionemo_ir.models.openfold2.convert import get_point_projection_torch_weights

PREFIX = "structure_module.ipa.linear_q_points"


@pytest.fixture
def projection() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(144, 384), torch.randn(144)


def _assert_projection(converted: list[dict], weight: torch.Tensor, bias: torch.Tensor) -> None:
    assert len(converted) == 1
    assert torch.equal(converted[0]["weight"], weight)
    assert torch.equal(converted[0]["bias"], bias)


def test_nested_layout(projection):
    """AlphaFold2-converted checkpoints nest the projection under ``.linear``."""
    weight, bias = projection
    state_dict = {f"{PREFIX}.linear.weight": weight, f"{PREFIX}.linear.bias": bias}

    _assert_projection(get_point_projection_torch_weights(state_dict, PREFIX), weight, bias)


def test_flat_layout(projection):
    """OpenFold-trained checkpoints store the same tensors one level up."""
    weight, bias = projection
    state_dict = {f"{PREFIX}.weight": weight, f"{PREFIX}.bias": bias}

    _assert_projection(get_point_projection_torch_weights(state_dict, PREFIX), weight, bias)


def test_missing_projection_raises(projection):
    state_dict = {"structure_module.ipa.linear_q.weight": projection[0]}

    with pytest.raises(KeyError):
        get_point_projection_torch_weights(state_dict, PREFIX)
