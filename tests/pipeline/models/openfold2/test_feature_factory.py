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

import random

import numpy as np
import torch

from tensorrt_bionemo.pipeline.models.openfold2.feature_factory import pre_init


def test_pre_init_is_independent_of_python_worker_rng_state():
    original_python_state = random.getstate()
    try:
        random.seed(1)
        first_python_state = random.getstate()
        np.random.seed(2)
        torch.manual_seed(3)
        first = pre_init({"random_seed": 20260720})
        first_numpy = np.random.random()
        first_torch = torch.rand(1)

        assert random.getstate() == first_python_state

        random.seed(99)
        second_python_state = random.getstate()
        np.random.seed(98)
        torch.manual_seed(97)
        second = pre_init({"random_seed": 20260720})
        second_numpy = np.random.random()
        second_torch = torch.rand(1)

        assert random.getstate() == second_python_state
        assert first["ensemble_seed"] == second["ensemble_seed"]
        assert first_numpy == second_numpy
        assert torch.equal(first_torch, second_torch)
    finally:
        random.setstate(original_python_state)


def test_pre_init_generated_seed_preserves_python_worker_rng_state():
    original_python_state = random.getstate()
    try:
        random.seed(7)
        expected_python_state = random.getstate()

        result = pre_init({"random_seed": None})

        assert random.getstate() == expected_python_state
        assert 0 <= result["ensemble_seed"] <= torch.iinfo(torch.int32).max
    finally:
        random.setstate(original_python_state)
