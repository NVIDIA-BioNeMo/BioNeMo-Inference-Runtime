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
"""Replay-copy skipping for stable keyword inputs of CUDAGraphOptimizationTracker."""

import pytest
import torch
import torch.nn as nn

import bionemo_ir._torch.graph_optimization.cuda_graph.runtime as trk
from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
    InputKeyMethod,
    InputRoutingConfig,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.graph_optimization.decorator import support_graph_optimization
from bionemo_ir._torch.graph_optimization.tensor_copy_utils import _tensor_leaves

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS_TO_CAPTURE = 4  # warmup thresholds (1, 3) + capture on the next call


class _Conditioned(nn.Module):
    def forward(self, x: torch.Tensor, cond: dict[str, torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
        return (x + cond["bias"]) * cond["gain"] + scale


def _make_tracker() -> CUDAGraphOptimizationTracker:
    config = CUDAGraphOptimizationConfig(input_routing_config=InputRoutingConfig(stable_input_kwargs=["cond"]))
    return CUDAGraphOptimizationTracker(config, inner_module=_Conditioned().cuda()).eval()


def _inputs(seed: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(4, 8, device="cuda", generator=generator)
    cond = {
        "bias": torch.randn(4, 8, device="cuda", generator=generator),
        "gain": torch.randn(4, 8, device="cuda", generator=generator),
    }
    return x, cond, torch.randn(4, 8, device="cuda", generator=generator)


def _reference(x: torch.Tensor, cond: dict[str, torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    return (x + cond["bias"]) * cond["gain"] + scale


@pytest.fixture
def copied(monkeypatch) -> list[int]:
    """Record the ids of the tensors each replay copies into static buffers."""
    ids: list[int] = []
    original = trk._copy_tensors_into

    def recording(dest, src):
        ids.extend(id(leaf) for leaf in _tensor_leaves(src))
        return original(dest, src)

    monkeypatch.setattr(trk, "_copy_tensors_into", recording)
    return ids


def _capture(tracker, x, cond, scale) -> None:
    for _ in range(NUM_CALLS_TO_CAPTURE):
        tracker(x, cond=cond, scale=scale)
    state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, cond=cond, scale=scale)]
    assert state.preparation_state is CUDAGraphPreparationState.GRAPH_CAPTURED


def test_unchanged_stable_kwarg_skips_replay_copy(copied):
    tracker = _make_tracker()
    x, cond, scale = _inputs(0)
    with torch.no_grad():
        _capture(tracker, x, cond, scale)
        copied.clear()
        for step in range(3):
            x_step = x + step
            torch.testing.assert_close(tracker(x_step, cond=cond, scale=scale), _reference(x_step, cond, scale))

    stable_ids = {id(t) for t in cond.values()}
    assert not stable_ids & set(copied)
    assert id(scale) in copied


def test_new_stable_tensors_are_copied(copied):
    tracker = _make_tracker()
    x, cond, scale = _inputs(0)
    _, new_cond, _ = _inputs(1)
    with torch.no_grad():
        _capture(tracker, x, cond, scale)
        copied.clear()
        torch.testing.assert_close(tracker(x, cond=new_cond, scale=scale), _reference(x, new_cond, scale))
    assert {id(t) for t in new_cond.values()} <= set(copied)


def test_in_place_update_of_a_stable_tensor_is_copied():
    tracker = _make_tracker()
    x, cond, scale = _inputs(0)
    with torch.no_grad():
        _capture(tracker, x, cond, scale)
        cond["bias"].add_(1.0)
        torch.testing.assert_close(tracker(x, cond=cond, scale=scale), _reference(x, cond, scale))


def test_inference_mode_skips_by_identity(copied):
    tracker = _make_tracker()
    with torch.inference_mode():
        x, cond, scale = _inputs(0)
        _, new_cond, _ = _inputs(1)
        _capture(tracker, x, cond, scale)
        copied.clear()
        torch.testing.assert_close(tracker(x, cond=cond, scale=scale), _reference(x, cond, scale))
        assert not {id(t) for t in cond.values()} & set(copied)
        torch.testing.assert_close(tracker(x, cond=new_cond, scale=scale), _reference(x, new_cond, scale))
        assert {id(t) for t in new_cond.values()} <= set(copied)


def test_release_forgets_stable_sources():
    tracker = _make_tracker()
    x, cond, scale = _inputs(0)
    with torch.no_grad():
        _capture(tracker, x, cond, scale)
    state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, cond=cond, scale=scale)]
    assert state.stable_sources
    state.release()
    assert state.stable_sources == {}


@support_graph_optimization(
    named_dims=(),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    input_key_method=InputKeyMethod.EXACT,
    stable_kwargs=("cond",),
)
class _DeclaresStableCondition(_Conditioned):
    pass


def test_config_without_routing_inherits_declared_stable_kwargs(copied):
    tracker = CUDAGraphOptimizationTracker(
        CUDAGraphOptimizationConfig(), inner_module=_DeclaresStableCondition().cuda()
    ).eval()
    x, cond, scale = _inputs(0)
    with torch.no_grad():
        _capture(tracker, x, cond, scale)
        copied.clear()
        torch.testing.assert_close(tracker(x, cond=cond, scale=scale), _reference(x, cond, scale))
    assert not {id(t) for t in cond.values()} & set(copied)


def test_decorator_rejects_an_unknown_stable_kwarg():
    with pytest.raises(ValueError, match="stable kwarg 'missing'"):

        @support_graph_optimization(
            named_dims=(),
            graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            input_key_method=InputKeyMethod.EXACT,
            stable_kwargs=("missing",),
        )
        class _Module(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
