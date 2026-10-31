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
"""Borrowed keyword inputs, whose caller-owned tensor CUDAGraphOptimizationTracker reads in place of a copy."""

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, InputRoutingConfig
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS = 6  # warmup thresholds (1, 3), the capturing call, then replays


class _Scale(nn.Module):
    def forward(self, x: torch.Tensor, init: torch.Tensor) -> torch.Tensor:
        return x * init + 1


def _make_tracker(borrowed: tuple[str, ...]) -> CUDAGraphOptimizationTracker:
    config = CUDAGraphOptimizationConfig(input_routing_config=InputRoutingConfig(borrowed_input_kwargs=list(borrowed)))
    return CUDAGraphOptimizationTracker(config, inner_module=_Scale().cuda()).eval()


def test_a_borrowed_buffer_is_read_where_it_lies(copied):
    """The tensor passed at capture becomes the static buffer: values written into it between calls reach the
    replays, and passing it costs no copy."""
    tracker = _make_tracker(("init",))
    x = torch.randn(4, 8, device="cuda")
    init = torch.zeros(4, 8, device="cuda")
    with torch.inference_mode():
        for step in range(NUM_CALLS):
            init.fill_(step)
            out = tracker(x, init=init)
            torch.testing.assert_close(out, x * step + 1)
    state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, init=init)]
    assert state.preparation_state is CUDAGraphPreparationState.GRAPH_CAPTURED
    assert state.static_input_kwargs["init"] is init
    assert id(init) not in copied


def test_another_tensor_is_copied_into_the_borrowed_buffer(copied):
    tracker = _make_tracker(("init",))
    x = torch.randn(4, 8, device="cuda")
    init = torch.ones(4, 8, device="cuda")
    with torch.inference_mode():
        for _ in range(NUM_CALLS):
            tracker(x, init=init)
        other = torch.full_like(init, 3.0)
        out = tracker(x, init=other)
    assert id(other) in copied
    torch.testing.assert_close(out, x * 3 + 1)
    assert torch.equal(init, other), "the replay reads the borrowed buffer, so the new value lands there"


def test_without_the_field_the_input_is_copied(copied):
    tracker = _make_tracker(())
    x = torch.randn(4, 8, device="cuda")
    init = torch.ones(4, 8, device="cuda")
    with torch.inference_mode():
        for _ in range(NUM_CALLS):
            tracker(x, init=init)
    state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x, init=init)]
    assert state.static_input_kwargs["init"] is not init
    assert id(init) in copied


def test_a_tensor_under_two_borrowed_kwargs_gets_a_buffer_for_each(copied):
    """One buffer cannot be two inputs, so a tensor passed under two borrowed names is copied for each."""
    tracker = _make_tracker(("x", "init"))
    shared = torch.full((4, 8), 2.0, device="cuda")
    with torch.inference_mode():
        for _ in range(NUM_CALLS):
            out = tracker(x=shared, init=shared)
    torch.testing.assert_close(out, shared * shared + 1)
    state = tracker.graph_state_by_key[tracker.input_key_for_this_call(x=shared, init=shared)]
    assert state.preparation_state is CUDAGraphPreparationState.GRAPH_CAPTURED
    buffers = state.static_input_kwargs["x"], state.static_input_kwargs["init"]
    assert buffers[0] is not buffers[1] and all(buffer is not shared for buffer in buffers)
    assert id(shared) in copied


def test_a_borrowed_kwarg_cannot_also_be_stable():
    """A stable kwarg's copy is skipped while its tensor is unchanged, which misses writes into a borrowed buffer."""
    with pytest.raises(ValueError, match=r"borrowed kwargs cannot also be stable kwargs: \['init'\]"):
        InputRoutingConfig(borrowed_input_kwargs=["init", "x"], stable_input_kwargs=["init"])
