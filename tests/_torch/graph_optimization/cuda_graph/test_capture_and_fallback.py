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

Covers reverting to eager on memory-gate refusal, capture failure, and
replay failure — plus the grad/training guard — and asserts the captured
path matches eager.
"""

import pytest
import torch
import torch.nn as nn

import tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime as trk
from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputKeyMethod,
    InputRoutingConfigFactory,
    NamedDimTies,
)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph import memory as gc_mem
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS_TO_CAPTURE = 4  # warmup thresholds (1, 3) + capture on the next call


def _make():
    """Return (tracker over a fresh net, an independent eager clone)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).cuda().eval()
    raw = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    raw.load_state_dict(net.state_dict())
    raw = raw.cuda().eval()
    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=net).eval()
    return tracker, raw


def _make_bucketed_tracker():
    factory = InputRoutingConfigFactory()
    factory.set_named_dim_ties([NamedDimTies(name="tokens", input_dims=(("input", (1,)),), output_dims=((0, (1,)),))])
    factory.set_input_acceptance_dim("tokens", 8)
    factory.set_padded_dim("tokens", 4, 8, 1, multiple_of=1)
    config = CUDAGraphOptimizationConfig(
        input_key_method=InputKeyMethod.BUCKETED_SHAPES,
        input_routing_config=factory.export_config(),
    )
    return CUDAGraphOptimizationTracker(config, inner_module=nn.ReLU().cuda()).eval()


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
    assert _key(m, x) not in m.fallback_to_eager_by_key
    assert state.working_set_bytes > 0
    assert torch.allclose(out, ref, atol=1e-5)


def test_exact_state_reuses_cached_shape_routing(monkeypatch):
    tracker, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        expected = raw(x)
        tracker(x)

    def unexpected_routing(*_args, **_kwargs):
        pytest.fail("established exact key recomputed shape routing")

    monkeypatch.setattr(tracker, "_extract_tensor_container_shape_maps", unexpected_routing)
    monkeypatch.setattr(tracker, "validate_input_ties", unexpected_routing)
    monkeypatch.setattr(tracker, "input_accepted", unexpected_routing)

    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            output = tracker(x)

    state = tracker.graph_state_by_key[_key(tracker, x)]
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert torch.allclose(output, expected, atol=1e-5)


def test_bucketed_state_reuses_cached_shape_routing(monkeypatch):
    tracker = _make_bucketed_tracker()
    x = torch.randn(1, 5, 4, device="cuda")
    with torch.no_grad():
        expected = torch.relu(x)
        tracker(x)

    def unexpected_routing(*_args, **_kwargs):
        pytest.fail("established bucket recomputed shape routing")

    monkeypatch.setattr(tracker, "_extract_tensor_container_shape_maps", unexpected_routing)
    monkeypatch.setattr(tracker, "validate_input_ties", unexpected_routing)
    monkeypatch.setattr(tracker, "input_accepted", unexpected_routing)

    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            output = tracker(x)

    assert len(tracker.graph_state_by_key) == 1
    state = next(iter(tracker.graph_state_by_key.values()))
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert torch.equal(output, expected)


def test_bucketed_shape_cache_is_evicted_with_graph_state():
    tracker = _make_bucketed_tracker()
    x_large = torch.randn(1, 5, 4, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            tracker(x_large)

    ((old_graph_key, old_state),) = tracker.graph_state_by_key.items()
    old_input_key = tracker.input_key_for_this_call(x_large)
    assert old_state.cached_input_key == old_input_key
    del old_state

    x_small = torch.randn(1, 3, 4, device="cuda")
    with torch.no_grad():
        tracker(x_small)

    assert old_graph_key not in tracker.graph_state_by_key
    assert all(state.cached_input_key != old_input_key for state in tracker.graph_state_by_key.values())


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
def test_bucketed_forward_avoids_host_device_synchronization():
    tracker = _make_bucketed_tracker()
    x = torch.randn(1, 5, 4, device="cuda")

    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            tracker(x)

    x_replay = torch.randn(1, 6, 4, device="cuda")
    previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.no_grad():
            output = tracker(x_replay)
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_debug_mode)

    assert len(tracker.graph_state_by_key) == 1
    state = next(iter(tracker.graph_state_by_key.values()))
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert output.shape == x_replay.shape


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
        trk, "check_capacity_for_capture", lambda *a, **k: gc_mem.MemoryCheck(False, "forced-low-mem", 1 << 40, 1 << 20)
    )
    with torch.no_grad():
        ref = raw(x)
        # The NUM_CALLS_TO_CAPTURE-th call reaches the capture step, where the
        # memory gate refuses; the extra calls verify permanence (the key stays
        # eager and never re-warms).
        for _ in range(NUM_CALLS_TO_CAPTURE + 3):
            out = m(x)
    # On refusal the key is permanently reverted to eager: recorded on the
    # tracker's ``fallback_to_eager_by_key`` and its would-be graph state evicted
    # (buffers freed), so the key is flagged eager and absent from the state
    # cache; the call still returns the correct eager result.
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert _key(m, x) not in m.graph_state_by_key
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
        # The NUM_CALLS_TO_CAPTURE-th call reaches the capture step (which
        # raises); the extra calls verify permanence (no re-warm).
        for _ in range(NUM_CALLS_TO_CAPTURE + 3):
            out = m(x)
    # Capture raises, so the key is permanently reverted to eager: recorded on
    # the tracker's ``fallback_to_eager_by_key`` and its graph state evicted
    # (graph + buffers freed); the call falls back to the correct eager result.
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert _key(m, x) not in m.graph_state_by_key
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
        # A further call must stay eager (the key does not re-warm).
        out = m(x)
    # The failing replay permanently reverts the key to eager: recorded on the
    # tracker's ``fallback_to_eager_by_key`` and its state evicted (captured
    # graph freed); calls fall back to the correct eager result.
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert _key(m, x) not in m.graph_state_by_key
    assert torch.allclose(out, ref, atol=1e-5)


class _MutatesScratchBuffers(nn.Module):
    """Mimics the Boltz-2 / OpenFold3 score model: a shared ``buffers`` scratch
    dict is pre-filled by the caller (an "encoder") and then reused by this
    module under the *same key* at a *different shape* (``ensure_buffer`` key
    reuse). The output is a deterministic fn of ``x`` only."""

    def forward(self, x, buffers=None):
        if buffers is not None:
            buffers["pw"] = torch.zeros(x.shape[0], 768, device=x.device, dtype=x.dtype)
        return x * 2.0 + 1.0


def test_input_key_ignores_scratch_buffers():
    # The scratch ``buffers`` dict must not contribute to the input key: calls
    # that differ only in their (side-effect) scratch shape share one graph.
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")
    k_small = m.input_key_for_this_call(x, buffers={"pw": torch.zeros(2, 128, device="cuda")})
    k_large = m.input_key_for_this_call(x, buffers={"pw": torch.zeros(2, 768, device="cuda")})
    k_none = m.input_key_for_this_call(x)
    assert k_small == k_large == k_none


def test_shared_scratch_buffers_capture_then_replay_matches_eager():
    # Regression: a side-effect-populated ``buffers`` scratch dict (whose entries
    # the captured module reshapes vs. what the caller put there) must not be
    # treated as a graph input. Before the fix this crashed in the per-replay
    # static-buffer copy ("size of tensor a (768) must match tensor b (128)").
    net = _MutatesScratchBuffers().cuda().eval()
    m = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(verify_capture=True), inner_module=net).eval()
    x = torch.randn(2, 8, device="cuda")
    ref = x * 2.0 + 1.0
    with torch.no_grad():
        for _ in range(6):
            # Fresh, "encoder"-shaped scratch each call (a different shape than
            # the wrapped module writes), as in the real diffusion loop.
            out = m(x, buffers={"pw": torch.zeros(2, 128, device="cuda")})
    state = m.graph_state_by_key[_key(m, x)]
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert _key(m, x) not in m.fallback_to_eager_by_key
    assert torch.allclose(out, ref, atol=1e-5)
