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


def test_protenix_merges_release_runtime_defaults():
    from bionemo_ir.pipeline.processor.engine_proc import _resolve_runtime_args

    config = EngineProcessorConfig(
        model_source="protenix-v2", executor_backend=None, runtime_args={"diffusion_samples": 2}
    )
    _resolve_runtime_args(config)
    from bionemo_ir.pipeline.models.protenix.postprocessor import INPUT_FEATURES

    assert config.runtime_args == {
        "recycling_steps": 9,
        "num_sampling_steps": 200,
        "diffusion_samples": 2,
        "consume_input_features": True,
        "keep_input_features": INPUT_FEATURES,
    }
    assert config.folding_engine_kwargs()["config"].compact_output is True


@pytest.mark.parametrize(
    ("runtime_args", "message"),
    [
        ({"return_full_data": False}, "requires return_full_data"),
        ({"keep_input_features": ()}, "must keep the Protenix postprocessor inputs"),
    ],
)
def test_protenix_rejects_runtime_options_before_loading_metadata(
    monkeypatch: pytest.MonkeyPatch, runtime_args: dict, message: str
) -> None:
    from bionemo_ir.pipeline.processor import engine_proc

    def unexpected_metadata_load(config: EngineProcessorConfig) -> None:
        pytest.fail("Runtime validation must precede metadata loading")

    monkeypatch.setattr(engine_proc, "_resolve_metadata", unexpected_metadata_load)
    config = EngineProcessorConfig(model_source="protenix-v2", executor_backend=None, runtime_args=runtime_args)
    with pytest.raises(ValueError, match=message):
        engine_proc.build_processor(config)


@pytest.mark.parametrize("model_source", ModelRegistry.get_models())
def test_runtime_resolution_preserves_caller_arguments(model_source: str) -> None:
    from bionemo_ir.pipeline.processor.engine_proc import _resolve_runtime_args

    supplied = {"diffusion_samples": 2, "compact_output": False}
    config = EngineProcessorConfig(model_source=model_source, runtime_args=supplied, compact_confidence=False)
    _resolve_runtime_args(config)
    assert config.runtime_args == {**ModelRegistry.get_factory(model_source).get_default_runtime_args(), **supplied}
    assert supplied == {"diffusion_samples": 2, "compact_output": False}
