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
import asyncio
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np
import pytest
import torch

from bionemo_ir._torch.modules.openfold3.embedders import MSAModuleEmbedder
from bionemo_ir.pipeline.base import ContextGeneratorBase, FeatureGeneratorBase, default_context_and_feature_merger
from bionemo_ir.pipeline.models.boltz1.feature_factory import pre_init as boltz1_init
from bionemo_ir.pipeline.models.boltz2.feature_factory import pre_init as boltz2_init
from bionemo_ir.pipeline.models.boltz2.structure import _build_smiles_mol
from bionemo_ir.pipeline.models.openfold2.feature_factory import pre_init as of2_init
from bionemo_ir.pipeline.models.openfold3.feature_factory import pre_init as of3_init
from bionemo_ir.pipeline.processor.base import ProcessorConfig, SerialProcessor
from bionemo_ir.pipeline.processor.pipelined import PipelinedSerialProcessor
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF
from bionemo_ir.pipeline.stages.feature_generator_stage import FeatureGeneratorStage, FeatureGeneratorUDF
from bionemo_ir.pipeline.stages.tokenizer_stage import TokenizerUDF
from bionemo_ir.pipeline.utils._rng import _feature_rng, _python_rng, _RequestRNG, _torch_generator

FACTORIES = [of2_init, of3_init, boltz1_init, boltz2_init]


class _DrawContext(ContextGeneratorBase):
    def __call__(self) -> dict[str, torch.Tensor]:
        return {"draw": torch.randn(4, generator=_torch_generator())}


class _DrawFeature(FeatureGeneratorBase):
    def __init__(self, prepared: Event | None = None) -> None:
        super().__init__()
        self.prepared = prepared
        self.calls = 0

    def __call__(self, batch: dict[str, torch.Tensor], context: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.calls += 1
        draw = torch.randn(4, generator=_torch_generator())
        if self.calls >= 2 and self.prepared is not None:
            self.prepared.set()
        return {"draw": draw}


class _DrawEngine(StatefulStageUDF):
    def __init__(self, *args: Any, prepared: Event, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prepared = prepared

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        assert self.prepared.wait(timeout=10)
        return {**row, "inference_draw": torch.randn(4)}


@pytest.mark.parametrize("pre_init", FACTORIES)
def test_factories_preserve_rng(pre_init: Callable) -> None:
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    context = pre_init({"random_seed": 17})
    with _feature_rng(context["_rng"]):
        _python_rng().randint(0, 100)
        torch.randn(10, generator=_torch_generator())
        context["_rng"].numpy.random(10)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all(), strict=True))


@pytest.mark.parametrize("pre_init", FACTORIES)
@pytest.mark.parametrize("stage", ["tokenizer", "feature"])
def test_stages_bind_rng(pre_init: Callable, stage: str) -> None:
    kwargs = {
        "compute_by_rows": True,
        "drop_keys": [],
        "expected_input_keys": [],
        "update_row": False,
        "pre_init": pre_init,
    }
    udf = (
        TokenizerUDF(**kwargs, context_generators={"draw": _DrawContext()})
        if stage == "tokenizer"
        else FeatureGeneratorUDF(
            **kwargs, feature_generators=[_DrawFeature()], features_merger_func=default_context_and_feature_merger
        )
    )
    result = asyncio.run(udf.udf_for_item({"random_seed": 17}))
    first = result["draw"]
    if stage == "feature" and pre_init is not of2_init:
        assert result["__sampling_seed"] == 17
    asyncio.run(udf.udf_for_item({"random_seed": 29}))
    again = asyncio.run(udf.udf_for_item({"random_seed": 17}))["draw"]
    expected = torch.randn(4, generator=pre_init({"random_seed": 17})["_rng"].torch)
    assert torch.equal(first, expected) and torch.equal(again, expected)
    assert _torch_generator() is None


@pytest.mark.parametrize("pre_init", FACTORIES)
def test_overlap_preserves_rng(pre_init: Callable) -> None:
    def stages() -> list[StatefulStage]:
        prepared = Event()
        return [
            FeatureGeneratorStage(
                fn_constructor_kwargs={
                    "pre_init": pre_init,
                    "feature_generators": [_DrawFeature(prepared)],
                    "features_merger_func": default_context_and_feature_merger,
                }
            ),
            StatefulStage(
                fn=_DrawEngine, fn_constructor_kwargs={"prepared": prepared}, map_batches_kwargs={"num_gpus": 1}
            ),
        ]

    config = ProcessorConfig(model_source="test")
    records = [{"__record_id": str(i), "random_seed": i + 17} for i in range(4)]
    engine_state = torch.get_rng_state()
    expected = SerialProcessor(config, stages())(records)
    final_state = torch.get_rng_state()
    torch.set_rng_state(engine_state)
    with PipelinedSerialProcessor(config, stages(), prefetch=2) as processor:
        actual = processor(records)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a["__data__"]["draw"], b["__data__"]["draw"])
        assert torch.equal(a["__data__"]["inference_draw"], b["__data__"]["inference_draw"])
    assert torch.equal(torch.get_rng_state(), final_state)


def test_rng_error_cleanup() -> None:
    outer, inner = _RequestRNG(17), _RequestRNG(29)
    with _feature_rng(outer):
        with pytest.raises(RuntimeError), _feature_rng(inner):
            raise RuntimeError("feature failure")
        assert _torch_generator() is outer.torch
    assert _torch_generator() is None
    assert _python_rng() is random


def test_concurrent_rng_isolation() -> None:
    ready = Event()

    def draw(seed: int) -> torch.Tensor:
        with _feature_rng(_RequestRNG(seed)):
            ready.wait(timeout=10)
            return torch.randn(100, generator=_torch_generator())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(draw, seed) for seed in (17, 29)]
        ready.set()
        outputs = [future.result() for future in futures]
    for seed, actual in zip((17, 29), outputs, strict=True):
        expected = torch.randn(100, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_msa_rng_isolation(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class _MsaProbe:
        forward = MSAModuleEmbedder.forward
        _subsample_all_msa = staticmethod(MSAModuleEmbedder._subsample_all_msa)
        subsample_main_msa = False
        subsample_all_msa = True
        min_subsampled_all_msa = 2
        max_subsampled_all_msa = 5

        def linear_m(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def linear_s_input(self, value: torch.Tensor) -> torch.Tensor:
            return value

    batch = {
        "msa": torch.arange(6 * 3 * 4, device=device).view(6, 3, 4).float(),
        "has_deletion": torch.zeros(6, 3, device=device),
        "deletion_value": torch.zeros(6, 3, device=device),
        "msa_mask": torch.ones(6, 3, device=device),
    }
    s_input = torch.zeros(3, 6, device=device)
    state = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    probe = _MsaProbe()
    first = probe.forward(batch, s_input, generator=torch.Generator(device=device).manual_seed(17))
    probe.forward(batch, s_input, generator=torch.Generator(device=device).manual_seed(29))
    again = probe.forward(batch, s_input, generator=torch.Generator(device=device).manual_seed(17))
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    actual_state = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(state, actual_state)


@pytest.mark.parametrize("pre_init", [boltz1_init, boltz2_init])
def test_smiles_rng_isolation(pre_init: Callable) -> None:
    state = random.getstate()

    def positions(seed: int) -> np.ndarray:
        context = pre_init({"random_seed": seed})
        with _feature_rng(context["_rng"]):
            return _build_smiles_mol("CCCOCC", "LIG").GetConformer().GetPositions()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, other, again = executor.map(positions, [17, 29, 17])
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
    assert random.getstate() == state
