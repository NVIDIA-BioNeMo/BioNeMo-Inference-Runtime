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
"""Static buffers of state keyword inputs, which CUDAGraphOptimizationTracker hands back uncloned."""

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, InputRoutingConfig
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS = 7  # warmup thresholds (1, 3), the capturing call, then replays


class _Recycle(nn.Module):
    """Overwrites ``state`` with the next call's and returns it, as a recycling trunk step does with its pair."""

    def forward(self, x: torch.Tensor, state: torch.Tensor, init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = x * state.sum(-1, keepdim=True)
        return out, torch.add(init, out.tanh(), out=state)


def _make_tracker(state_kwargs: tuple[str, ...]) -> CUDAGraphOptimizationTracker:
    routing = InputRoutingConfig(state_input_kwargs=list(state_kwargs))
    config = CUDAGraphOptimizationConfig(input_routing_config=routing)
    return CUDAGraphOptimizationTracker(config, inner_module=_Recycle().cuda()).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 8, device="cuda", generator=generator)
    init = torch.randn(4, 8, device="cuda", generator=generator)
    return x, init


def _run(tracker: nn.Module, x: torch.Tensor, init: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each call's ``(out, state)``, each call taking the state the last one returned, the first a copy of init."""
    state, calls = init.clone(), []
    for _ in range(NUM_CALLS):
        out, state = tracker(x, state=state, init=init)
        calls.append((out.clone(), state))
    return calls


def test_state_buffer_goes_back_uncloned_and_is_not_copied_in(copied):
    tracker = _make_tracker(("state",))
    x, init = _inputs()
    with torch.inference_mode():
        calls = _run(tracker, x, init)
        reference = [(out, state.clone()) for out, state in _run(_Recycle().cuda(), x, init)]
    graph_state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, state=init, init=init)]
    assert graph_state.preparation_state is CUDAGraphPreparationState.GRAPH_CAPTURED
    buffer = graph_state.static_input_kwargs["state"]

    replayed = calls[3:]  # the capturing call and the replays
    assert all(state is buffer for _, state in replayed)
    # Only the capturing call copied a live state in; the replays took the buffer back as it was.
    assert copied.count(id(buffer)) == 0
    (out, _), (expected_out, expected_state) = calls[-1], reference[-1]
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(buffer, expected_state)


def test_a_new_state_tensor_is_copied_in(copied):
    tracker = _make_tracker(("state",))
    x, init = _inputs()
    with torch.inference_mode():
        _run(tracker, x, init)
        fresh = init.clone()
        copied.clear()
        out, state = tracker(x, state=fresh, init=init)
        expected_out, expected_state = _Recycle()(x, init.clone(), init)
    assert id(fresh) in copied
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(state, expected_state)


def test_without_the_field_the_state_comes_back_cloned():
    tracker = _make_tracker(())
    x, init = _inputs()
    with torch.inference_mode():
        calls = _run(tracker, x, init)
    graph_state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, state=init, init=init)]
    assert all(state is not graph_state.static_input_kwargs["state"] for _, state in calls[3:])


def test_a_state_kwarg_cannot_also_be_stable():
    """The module overwrites a state kwarg's buffer, which a stable kwarg's skipped copy would leave in place."""
    with pytest.raises(ValueError, match=r"state kwargs cannot also be stable kwargs: \['state'\]"):
        InputRoutingConfig(state_input_kwargs=["state"], stable_input_kwargs=["state", "init"])


def test_a_verified_capture_reads_the_borrowed_state_its_caller_passed():
    """Verification replays the graph, which overwrites a borrowed state kwarg's buffer, the caller's own tensor.

    The capturing call must still compute from the value its caller passed, as every other call does.
    """
    routing = InputRoutingConfig(state_input_kwargs=["state"], borrowed_input_kwargs=["state"])
    config = CUDAGraphOptimizationConfig(input_routing_config=routing, verify_capture=True)
    tracker = CUDAGraphOptimizationTracker(config, inner_module=_Recycle().cuda()).eval()
    x, init = _inputs()
    buffer = torch.empty_like(init)
    with torch.inference_mode():
        for call in range(NUM_CALLS):
            buffer.copy_(init * (call + 1))
            expected_out, expected_state = _Recycle()(x, buffer.clone(), init)
            out, state = tracker(x, state=buffer, init=init)
            assert torch.equal(out, expected_out) and torch.equal(state, expected_state), call
    assert tracker.execution_counts["capture"] == 1
