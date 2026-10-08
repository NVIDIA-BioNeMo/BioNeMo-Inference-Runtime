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
import pytest
import torch

from bionemo_ir.pipeline.models.openfold2.feature_factory import pre_init


def test_pre_init_is_independent_of_python_worker_rng_state():
    original_python_state = random.getstate()
    try:
        random.seed(1)
        first_python_state = random.getstate()
        np.random.seed(2)
        torch.manual_seed(3)
        first = pre_init({"random_seed": 20260720})
        first_numpy = first["_rng"].numpy.random()
        first_torch = torch.rand(1, generator=first["_rng"].torch)

        assert random.getstate() == first_python_state

        random.seed(99)
        second_python_state = random.getstate()
        np.random.seed(98)
        torch.manual_seed(97)
        second = pre_init({"random_seed": 20260720})
        second_numpy = second["_rng"].numpy.random()
        second_torch = torch.rand(1, generator=second["_rng"].torch)

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


@pytest.mark.parametrize("multimer", [False, True])
@pytest.mark.parametrize("steps", [None, 1, 3, 30])
def test_feature_repeats_match_forward_limit(
    multimer: bool, steps: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bionemo_ir.models.openfold2.config import OpenFold2Config, OpenFold2MultimerConfig
    from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, _build_feature_generator_stage

    original = OpenFold2MultimerConfig() if multimer else OpenFold2Config()
    monkeypatch.setattr(EngineProcessorConfig, "get_model_pretrained_config", lambda self: original)
    config = EngineProcessorConfig(
        model_source="alphafold2_multimer_1" if multimer else "alphafold2_1",
        executor_backend=None,
        runtime_args={"recycling_steps": steps},
    )
    stage = _build_feature_generator_stage(config, {"batch_size": 1, "concurrency": None, "runtime_env": None})
    expected = original.max_recycling_iters + 1
    if steps is not None:
        expected = min(expected, steps)
    assert stage.fn_constructor_kwargs["feature_collators"][0].n_iter == expected
    assert original.max_recycling_iters == (20 if multimer else 3)


@pytest.mark.parametrize("steps", [0, -1])
def test_feature_repeats_reject_nonpositive_limit(steps: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from bionemo_ir.models.openfold2.config import OpenFold2Config
    from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, _build_feature_generator_stage

    monkeypatch.setattr(EngineProcessorConfig, "get_model_pretrained_config", lambda self: OpenFold2Config())
    config = EngineProcessorConfig(model_source="alphafold2_1", runtime_args={"recycling_steps": steps})
    with pytest.raises(ValueError, match="recycling_steps must be positive"):
        _build_feature_generator_stage(config, {})


@pytest.mark.parametrize("multimer", [False, True])
@pytest.mark.parametrize("steps", [True, False, 1.5, 3.0, "3", float("nan"), float("inf")])
def test_feature_repeats_reject_nonintegers(multimer: bool, steps: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from bionemo_ir.models.openfold2.config import OpenFold2Config, OpenFold2MultimerConfig
    from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, _build_feature_generator_stage

    original = OpenFold2MultimerConfig() if multimer else OpenFold2Config()
    monkeypatch.setattr(EngineProcessorConfig, "get_model_pretrained_config", lambda self: original)
    config = EngineProcessorConfig(
        model_source="alphafold2_multimer_1" if multimer else "alphafold2_1",
        runtime_args={"recycling_steps": steps},
    )
    with pytest.raises(ValueError, match="recycling_steps must be an integer"):
        _build_feature_generator_stage(config, {})
    assert original.max_recycling_iters == (20 if multimer else 3)
