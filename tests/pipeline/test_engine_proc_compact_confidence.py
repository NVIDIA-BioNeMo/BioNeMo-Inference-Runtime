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
"""The pipeline's opt-in compact confidence setting reaches the OpenFold3 model config and nothing else."""

import pytest

from bionemo_ir.models.openfold3.config import OpenFold3Config
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig


def test_default_leaves_the_raw_contract_and_engine_kwargs_alone() -> None:
    config = EngineProcessorConfig(model_source="openfold3", executor_backend=None)
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


def test_opt_in_rejects_models_without_compact_outputs() -> None:
    config = EngineProcessorConfig(model_source="boltz-2", executor_backend=None, compact_confidence=True)
    with pytest.raises(ValueError, match="compact confidence"):
        config.folding_engine_kwargs()
