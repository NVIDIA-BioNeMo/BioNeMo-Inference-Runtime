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
"""Capture/replay + fallback behavior of CUDAGraphOptimizationTracker.

Covers requirement (4): revert to eager on (3.1/4.1.1) memory-gate refusal,
(4.1.2) capture failure, and (4.1.3) replay failure — plus the grad/training
guard — and asserts the captured path matches eager.
"""
import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.graph_optimization import memory as gc_mem
import tensorrt_bionemo._torch.graph_optimization.graph_optimization_tracker as trk
from tensorrt_bionemo._torch.graph_optimization.config_schema import (
    CUDAGraphOptimizationConfig)
from tensorrt_bionemo._torch.graph_optimization.graph_optimization_tracker import (
    CUDAGraphOptimizationTracker, CUDAGraphPreparationState)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS_TO_CAPTURE = 4  # warmup thresholds (1, 3) + capture on the next call


def _make():
    """Return (tracker over a fresh net, an independent eager clone)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).cuda().eval()
    raw = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    raw.load_state_dict(net.state_dict())
    raw = raw.cuda().eval()
    tracker = CUDAGraphOptimizationTracker(
        CUDAGraphOptimizationConfig(), inner_module=net).eval()
    return tracker, raw


def _key(tracker, x):
    return tracker.input_key_for_this_call(x)


def test_capture_then_replay_matches_eager():
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        ref = raw(x)
        for _ in range(6):
            out = m(x)
    state = m.graph_state_by_key[_key(m, x)]
    # After a successful capture the key advances through GRAPH_CAPTURED to its
    # resting state GRAPH_VERIFIED (GRAPH_CAPTURED is only a transient value set
    # mid-capture, before the verify step).
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert not state.fallback_to_eager
    assert state.working_set_bytes > 0
    assert torch.allclose(out, ref, atol=1e-5)


def test_grad_or_training_falls_back_to_eager():
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    # grad enabled (no no_grad) -> strategy forces eager; no state is created
    for _ in range(6):
        out = m(x)
    assert len(m.graph_state_by_key) == 0
    assert torch.allclose(out, raw(x), atol=1e-5)


def test_memory_gate_refusal_reverts_to_eager(monkeypatch):
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    monkeypatch.setattr(
        trk, "check_capacity_for_capture",
        lambda *a, **k: gc_mem.MemoryCheck(False, "forced-low-mem", 1 << 40, 1 << 20))
    with torch.no_grad():
        ref = raw(x)
        for _ in range(6):
            out = m(x)
    state = m.graph_state_by_key[_key(m, x)]
    assert state.fallback_to_eager
    assert state.graph is None
    assert torch.allclose(out, ref, atol=1e-5)


def test_capture_failure_reverts_to_eager(monkeypatch):
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")

    class _BoomGraph:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("forced capture failure")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **k: _BoomGraph())
    with torch.no_grad():
        ref = raw(x)
        for _ in range(6):
            out = m(x)
    state = m.graph_state_by_key[_key(m, x)]
    assert state.fallback_to_eager
    assert state.graph is None
    assert torch.allclose(out, ref, atol=1e-5)


def test_replay_failure_reverts_to_eager():
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        ref = raw(x)
        for _ in range(NUM_CALLS_TO_CAPTURE):
            m(x)
    state = m.graph_state_by_key[_key(m, x)]
    assert state.graph is not None  # captured

    def _boom():
        raise RuntimeError("forced replay failure")

    state.graph.replay = _boom
    with torch.no_grad():
        out = m(x)
    assert state.fallback_to_eager
    assert torch.allclose(out, ref, atol=1e-5)


class _MutatesScratchBuffers(nn.Module):
    """Mimics the Boltz-2 / OpenFold3 score model: a shared ``buffers`` scratch
    dict is pre-filled by the caller (an "encoder") and then reused by this
    module under the *same key* at a *different shape* (``ensure_buffer`` key
    reuse). The output is a deterministic fn of ``x`` only."""

    def forward(self, x, buffers=None):
        if buffers is not None:
            buffers["pw"] = torch.zeros(x.shape[0], 768, device=x.device,
                                        dtype=x.dtype)
        return x * 2.0 + 1.0


def test_input_key_ignores_scratch_buffers():
    # The scratch ``buffers`` dict must not contribute to the input key: calls
    # that differ only in their (side-effect) scratch shape share one graph.
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")
    k_small = m.input_key_for_this_call(
        x, buffers={"pw": torch.zeros(2, 128, device="cuda")})
    k_large = m.input_key_for_this_call(
        x, buffers={"pw": torch.zeros(2, 768, device="cuda")})
    k_none = m.input_key_for_this_call(x)
    assert k_small == k_large == k_none


def test_shared_scratch_buffers_capture_then_replay_matches_eager():
    # Regression: a side-effect-populated ``buffers`` scratch dict (whose entries
    # the captured module reshapes vs. what the caller put there) must not be
    # treated as a graph input. Before the fix this crashed in the per-replay
    # static-buffer copy ("size of tensor a (768) must match tensor b (128)").
    net = _MutatesScratchBuffers().cuda().eval()
    m = CUDAGraphOptimizationTracker(
        CUDAGraphOptimizationConfig(verify_capture=True),
        inner_module=net).eval()
    x = torch.randn(2, 8, device="cuda")
    ref = x * 2.0 + 1.0
    with torch.no_grad():
        for _ in range(6):
            # Fresh, "encoder"-shaped scratch each call (a different shape than
            # the wrapped module writes), as in the real diffusion loop.
            out = m(x, buffers={"pw": torch.zeros(2, 128, device="cuda")})
    state = m.graph_state_by_key[_key(m, x)]
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert not state.fallback_to_eager
    assert torch.allclose(out, ref, atol=1e-5)
