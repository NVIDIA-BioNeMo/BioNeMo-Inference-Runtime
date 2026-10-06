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
"""Pipeline confidence policy preserves caller configs and model-specific contracts."""

import pytest

from bionemo_ir.models.openfold3.config import OpenFold3Config
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig
from bionemo_ir.registry import ModelRegistry


def test_opt_out_preserves_engine_kwargs() -> None:
    config = EngineProcessorConfig(model_source="openfold3", executor_backend=None, compact_confidence=False)
    assert config.compact_confidence is False
    assert config.get_model_pretrained_config().auxiliary_heads_config.compact_output is False
    assert "config" not in config.folding_engine_kwargs()


def test_opt_in_enables_compact_output_on_a_copy_for_the_engine() -> None:
    user_config = OpenFold3Config()
    config = EngineProcessorConfig(
        model_source="openfold3",
        executor_backend=None,
        compact_confidence=True,
        engine_kwargs={"config": user_config, "profile_inference": True},
    )
    engine_kwargs = config.folding_engine_kwargs()
    assert engine_kwargs["profile_inference"] is True
    assert engine_kwargs["config"].auxiliary_heads_config.compact_output is True
    assert engine_kwargs["config"] is not user_config
    assert user_config.auxiliary_heads_config.compact_output is False
    assert config.get_model_pretrained_config().auxiliary_heads_config.compact_output is True


def test_opt_in_works_from_the_pretrained_config() -> None:
    config = EngineProcessorConfig(model_source="openfold3", executor_backend=None, compact_confidence=True)
    assert config.folding_engine_kwargs()["config"].auxiliary_heads_config.compact_output is True


@pytest.mark.parametrize("model_source", ModelRegistry.get_models())
def test_default_supports_every_model(model_source: str) -> None:
    factory = ModelRegistry.get_factory(model_source)
    original = factory.get_model_class().get_pretrained_config(model_source)
    config = EngineProcessorConfig(model_source=model_source, engine_kwargs={"config": original})
    actual = config.folding_engine_kwargs()["config"]
    assert actual is not original
    expected_head = original
    actual_head = actual
    for name in factory.confidence_config_path:
        expected_head = getattr(expected_head, name)
        actual_head = getattr(actual_head, name)
    assert expected_head.compact_output is False
    assert actual_head.compact_output is True
    assert config.engine_kwargs["config"] is original


def test_batched_of3_preserves_raw_config() -> None:
    original = OpenFold3Config()
    original.auxiliary_heads_config.memory_efficient_mode = False
    config = EngineProcessorConfig(model_source="openfold3", engine_kwargs={"config": original})
    assert config.folding_engine_kwargs()["config"] is original
    assert original.auxiliary_heads_config.compact_output is False
