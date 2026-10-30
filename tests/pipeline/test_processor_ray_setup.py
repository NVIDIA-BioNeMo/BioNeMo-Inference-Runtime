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

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bionemo_ir.pipeline.processor import engine_proc
from bionemo_ir.pipeline.processor.utils import normalize_cpu_stage_concurrency


@pytest.fixture
def processor_config(monkeypatch: pytest.MonkeyPatch) -> engine_proc.EngineProcessorConfig:
    tokenizer = SimpleNamespace(context_generator_specs={}, transform_specs=[], context_merger_func=None)
    feature_factory = SimpleNamespace(
        pre_init=None, feature_generator_specs=[], feature_collator_specs=[], features_merger_func=None
    )
    monkeypatch.setattr(engine_proc, "get_tokenizer", lambda model_source: tokenizer)
    monkeypatch.setattr(engine_proc, "get_feature_factory", lambda model_source: feature_factory)
    monkeypatch.setattr(engine_proc, "get_all_residue_types", lambda model_source: [])
    monkeypatch.setattr(engine_proc, "get_all_atom_types", lambda model_source: [])
    monkeypatch.setattr(engine_proc, "get_available_gpu_count", lambda: 1)
    monkeypatch.setattr(engine_proc.EngineProcessorConfig, "get_model_pretrained_config", lambda config: None)
    monkeypatch.setattr(engine_proc.EngineProcessorConfig, "folding_engine_kwargs", lambda config: {})
    stage_settings = {
        "compute": (2, 3),
        "batch_size": 4,
        "runtime_env": {"env_vars": {"STAGE_TEST": "1"}},
        "num_cpus": 2,
        "memory": 1024,
        "compute_by_rows": False,
        "drop_keys": ["stage_test"],
    }
    return engine_proc.EngineProcessorConfig(
        model_source="stage-test",
        metadata={},
        parser_stage=stage_settings,
        tokenizer_stage=stage_settings,
        feature_generator_stage=stage_settings,
        writer_stage=stage_settings,
    )


def test_processor_import_avoids_ray() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from bionemo_ir.pipeline.processor import engine_proc; "
            "assert not any(name == 'ray' or name.startswith('ray.') for name in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_serial_stages_skip_ray_kwargs(
    processor_config: engine_proc.EngineProcessorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setitem(sys.modules, "ray.data", None)
    stages = engine_proc._build_stages(processor_config, {})
    assert len(stages) == 5
    for stage in [stages[0], stages[1], stages[2], stages[4]]:
        assert stage.map_batches_kwargs == {}
        assert stage.compute_by_rows is False
        assert stage.drop_keys == ["stage_test"]
    assert stages[3].map_batches_kwargs["num_gpus"] == 1


def test_ray_stages_preserve_resources(processor_config: engine_proc.EngineProcessorConfig) -> None:
    from ray.data import ActorPoolStrategy

    processor_config.executor_backend = "ray"
    stages = engine_proc._build_stages(processor_config, {})
    for stage in [stages[0], stages[1], stages[2], stages[4]]:
        kwargs = stage.map_batches_kwargs
        assert isinstance(kwargs["compute"], ActorPoolStrategy)
        assert kwargs["compute"].min_size == 2
        assert kwargs["compute"].max_size == 3
        assert kwargs["zero_copy_batch"] is True
        assert kwargs["batch_size"] == 4
        assert kwargs["num_cpus"] == 2
        assert kwargs["memory"] == 1024
        assert kwargs["runtime_env"] == {"env_vars": {"STAGE_TEST": "1"}}
    assert stages[3].map_batches_kwargs["num_gpus"] == 1
    assert isinstance(stages[3].map_batches_kwargs["compute"], ActorPoolStrategy)


@pytest.mark.parametrize("num_cpus", [None, 0.0, 0.5, 2.0])
def test_ray_engine_stage_preserves_cpu_reservation(
    processor_config: engine_proc.EngineProcessorConfig, num_cpus: float | None
) -> None:
    processor_config.executor_backend = "ray"
    processor_config.engine_stage = {"compute": 1, "num_cpus": num_cpus}
    stage = engine_proc._build_folding_engine_stage(processor_config, {})
    kwargs = stage.get_dataset_map_batches_kwargs(batch_size=1)
    if num_cpus is None:
        assert "num_cpus" not in stage.map_batches_kwargs
        assert "num_cpus" not in kwargs
    else:
        assert stage.map_batches_kwargs["num_cpus"] == num_cpus
        assert kwargs["num_cpus"] == num_cpus


@pytest.mark.parametrize("engine_stage_num_cpus, expected_num_cpus", [(None, 4), (2, 2)])
def test_default_replica_config_engine_cpu_reservation(
    processor_config: engine_proc.EngineProcessorConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    engine_stage_num_cpus: int | None,
    expected_num_cpus: int,
) -> None:
    monkeypatch.setattr(engine_proc, "get_available_gpu_count", lambda: 8)
    config = engine_proc.EngineProcessorConfig.create_default_replica_mode_config(
        model_source=processor_config.model_source,
        output_dir=str(tmp_path),
        engine_stage_num_cpus=engine_stage_num_cpus,
    )
    assert config.engine_stage.num_cpus == expected_num_cpus
    assert config.engine_stage.compute == 8
    stage = engine_proc._build_folding_engine_stage(config, {})
    kwargs = stage.get_dataset_map_batches_kwargs(batch_size=1)
    assert kwargs["num_cpus"] == expected_num_cpus
    assert kwargs["compute"].min_size == kwargs["compute"].max_size == 8


def test_ray_engine_stage_explicit_cpu_override_is_preserved(
    processor_config: engine_proc.EngineProcessorConfig,
) -> None:
    processor_config.executor_backend = "ray"
    processor_config.engine_stage = {"compute": 1, "num_cpus": 2.0}
    stage = engine_proc._build_folding_engine_stage(processor_config, {})
    stage.map_batches_kwargs["num_cpus"] = 0.5
    assert stage.get_dataset_map_batches_kwargs(batch_size=1)["num_cpus"] == 0.5
    assert processor_config.engine_stage["num_cpus"] == 2.0


def test_serial_engine_stage_skips_cpu_reservation(
    processor_config: engine_proc.EngineProcessorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    processor_config.engine_stage = {"num_cpus": 2.0}
    stage = engine_proc._build_folding_engine_stage(processor_config, {})
    assert "num_cpus" not in stage.map_batches_kwargs


@pytest.mark.parametrize("concurrency, expected", [(None, (1, 1)), (4, (1, 4)), ((2, 5), (2, 5))])
def test_ray_concurrency_ranges(concurrency: int | tuple[int, int] | None, expected: tuple[int, int]) -> None:
    strategy = normalize_cpu_stage_concurrency(concurrency)
    assert (strategy.min_size, strategy.max_size) == expected
