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

import os
import re
from contextlib import AbstractContextManager
from typing import Literal

import pytest
import torch
import torch.nn as nn

import bionemo_ir._torch.graph_optimization.cuda_graph.runtime as trk
from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputKeyMethod,
    InputRoutingConfig,
    InputRoutingConfigFactory,
    NamedDimTies,
)
from bionemo_ir._torch.graph_optimization.cuda_graph import memory as gc_mem
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
    CUDAGraphState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")

NUM_CALLS_TO_CAPTURE = 4  # warmup thresholds (1, 3) + capture on the next call


class _InplaceRegion(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.add_(2.0)


class _RandomRegion(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.rand_like(value)


def _early_config(**overrides: object) -> CUDAGraphOptimizationConfig:
    """Capture on the first call after one warmup, keeping four graphs, unless ``overrides`` say otherwise."""
    settings: dict[str, object] = {
        "num_calls_for_kernel_compilation": 1,
        "num_calls_for_memory_allocator": 1,
        "capture_on_first_call": True,
        "num_graphs_max_for_this_module": 4,
    }
    return CUDAGraphOptimizationConfig(**(settings | overrides))


# A capacity check that refuses: far more needed than free.
_REFUSED_CAPACITY = gc_mem.MemoryCheck(False, "forced-low-mem", 1 << 40, 1 << 20)


@pytest.mark.parametrize("verify", [False, True])
def test_first_capture_restores_mutated_static_inputs(verify: bool) -> None:
    tracker = CUDAGraphOptimizationTracker(_early_config(verify_capture=verify), _InplaceRegion().cuda()).eval()
    value = torch.randn(2, 8, device="cuda")
    original = value.clone()
    with torch.inference_mode():
        first = tracker(value)
        second = tracker(value + 3.0)
    torch.testing.assert_close(first, original + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(second, (original + 3.0) + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert tracker.execution_counts["capture"] == 1
    assert tracker.execution_counts["replay"] == 1
    assert not tracker.fallback_to_eager_by_key


def test_first_capture_preparation_preserves_rng() -> None:
    raw = _RandomRegion().cuda().eval()
    tracker = CUDAGraphOptimizationTracker(_early_config(verify_capture=True), raw).eval()
    value = torch.zeros(2, 8, device="cuda")
    with torch.inference_mode():
        torch.manual_seed(42)
        expected = raw(value)
        expected_rng = torch.cuda.get_rng_state()
        torch.manual_seed(42)
        actual = tracker(value)
        actual_rng = torch.cuda.get_rng_state()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, expected_rng)
    assert tracker.execution_counts["capture"] == 1
    assert not tracker.fallback_to_eager_by_key


class _MutatingRandomRegion(nn.Module):
    """Mutate a positional input and a state kwarg, returning both as aliases."""

    def forward(self, value: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value.add_(torch.rand_like(value))
        return value, state.add_(value)


@pytest.mark.parametrize("warmups", [1, 3])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("failure", ["capacity", "capture", "verification"])
def test_first_capture_failure_runs_eager_on_caller_inputs(
    monkeypatch: pytest.MonkeyPatch,
    warmups: int,
    borrowed: bool,
    failure: Literal["capacity", "capture", "verification"],
) -> None:
    """A refused first capture preserves eager mutations, aliases and RNG offsets.

    Args:
        monkeypatch: Patches the selected capture failure.
        warmups: Number of warmup calls before capture.
        borrowed: Whether the state kwarg uses the caller's buffer for capture.
        failure: Stage at which the first capture is refused.
    """
    routing = InputRoutingConfig(state_input_kwargs=["state"], borrowed_input_kwargs=["state"] if borrowed else [])
    config = _early_config(input_routing_config=routing, verify_capture=True, num_calls_for_memory_allocator=warmups)
    region = _MutatingRandomRegion().cuda().eval()
    tracker = CUDAGraphOptimizationTracker(config, region).eval()

    def no_capacity(*args: object, **kwargs: object) -> gc_mem.MemoryCheck:
        return _REFUSED_CAPACITY

    def no_capture(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
        raise RuntimeError("forced capture failure")

    def mismatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("forced verification mismatch")

    if failure == "capacity":
        monkeypatch.setattr(trk, "check_capacity_for_capture", no_capacity)
    elif failure == "capture":
        monkeypatch.setattr(trk, "_capture_without_barrier", no_capture)
    else:
        monkeypatch.setattr(trk, "_assert_equal_but_distinct", mismatch)

    value = torch.ones(2, 8, device="cuda")
    state = torch.zeros_like(value)
    expected_value, expected_state = value.clone(), state.clone()
    key = tracker.input_key_for_this_call(value, state=state)
    with torch.inference_mode():
        torch.manual_seed(42)
        region(expected_value, expected_state)
        expected_cpu_rng = torch.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state()
        torch.manual_seed(42)
        first_value, first_state = tracker(value, state=state)
        assert first_value is value and first_state is state
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)

        # The next call must stay eager and use the next random values exactly once.
        region(expected_value, expected_state)
        next_cuda_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(expected_cuda_rng)
        next_value, next_state = tracker(value, state=state)
        assert next_value is value and next_state is state
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        assert torch.equal(torch.cuda.get_rng_state(), next_cuda_rng)

    assert tracker.fallback_to_eager_by_key[key] is True
    assert key not in tracker.graph_state_by_key
    assert tracker.execution_counts["fallback"] == 1
    assert not tracker.allocator_warmed_up


class _CountingRegion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("runs", torch.zeros((), dtype=torch.int64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.runs.add_(1)
        return value * 2.0


@pytest.mark.parametrize("warmups", [1, 3])
def test_first_capture_call_returns_its_warmup_without_replaying(warmups: int) -> None:
    config = _early_config(num_calls_for_memory_allocator=warmups)
    region = _CountingRegion().cuda()
    tracker = CUDAGraphOptimizationTracker(config, region).eval()
    value = torch.randn(2, 8, device="cuda")
    with torch.inference_mode():
        first = tracker(value)
        assert region.runs.item() == warmups
        second = tracker(value + 1.0)
        assert region.runs.item() == warmups + 1
    torch.testing.assert_close(first, value * 2.0, rtol=0, atol=0)
    torch.testing.assert_close(second, (value + 1.0) * 2.0, rtol=0, atol=0)
    assert tracker.execution_counts["capture"] == 1
    assert tracker.execution_counts["replay"] == 1


# More allocator warmups than kernel-compilation ones, so a call's run count shows which kind it ran.
_ALLOCATOR_WARMUPS = 3


def _one_graph_config(**overrides: object) -> CUDAGraphOptimizationConfig:
    """Capture after the allocator warmups and keep one resident graph, like a region that does not repeat."""
    return _early_config(
        num_calls_for_memory_allocator=_ALLOCATOR_WARMUPS, num_graphs_max_for_this_module=1, **overrides
    )


def test_later_signature_skips_the_allocator_warmups() -> None:
    region = _CountingRegion().cuda()
    tracker = CUDAGraphOptimizationTracker(_one_graph_config(), region).eval()
    small = torch.randn(2, 8, device="cuda")
    large = torch.randn(4, 8, device="cuda")
    with torch.inference_mode():
        tracker(small)
        assert region.runs.item() == _ALLOCATOR_WARMUPS
        first_large = tracker(large)  # One kernel-compilation warmup, whose result the call returns.
        assert region.runs.item() == _ALLOCATOR_WARMUPS + 1
        replayed_large = tracker(large + 1.0)
        assert region.runs.item() == _ALLOCATOR_WARMUPS + 2
        tracker.reset()
        tracker(small)
        assert region.runs.item() == 2 * _ALLOCATOR_WARMUPS + 2
    torch.testing.assert_close(first_large, large * 2.0, rtol=0, atol=0)
    torch.testing.assert_close(replayed_large, (large + 1.0) * 2.0, rtol=0, atol=0)
    assert tracker.execution_counts["capture"] == 3
    assert tracker.execution_counts["eviction"] == 1
    assert not tracker.fallback_to_eager_by_key


def test_later_signature_preserves_rng() -> None:
    raw = _RandomRegion().cuda().eval()
    tracker = CUDAGraphOptimizationTracker(_one_graph_config(verify_capture=True), raw).eval()
    with torch.inference_mode():
        tracker(torch.zeros(2, 8, device="cuda"))
        value = torch.zeros(4, 8, device="cuda")
        torch.manual_seed(42)
        expected = raw(value)
        expected_rng = torch.cuda.get_rng_state()
        torch.manual_seed(42)
        actual = tracker(value)
        actual_rng = torch.cuda.get_rng_state()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, expected_rng)
    assert tracker.execution_counts["capture"] == 2
    assert not tracker.fallback_to_eager_by_key


def test_moving_an_empty_tracker_restores_the_allocator_warmups() -> None:
    region = _CountingRegion().cuda()
    tracker = CUDAGraphOptimizationTracker(_one_graph_config(), region).eval()
    with torch.inference_mode():
        tracker(torch.randn(2, 8, device="cuda"))
        tracker._evict_all_keys()
        tracker.cuda()
        tracker(torch.randn(4, 8, device="cuda"))
    assert region.runs.item() == 2 * _ALLOCATOR_WARMUPS


class _BufferRegion(nn.Module):
    """Writes its result into a registered buffer and returns that buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("out", torch.zeros(2, 8))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.out.copy_(value * 2.0)
        return self.out


@pytest.mark.parametrize("warmups", [1, 3])
@pytest.mark.parametrize("verify", [False, True])
def test_first_result_survives_later_replays(verify: bool, warmups: int) -> None:
    config = _early_config(verify_capture=verify, num_calls_for_memory_allocator=warmups)
    region = _BufferRegion().cuda()
    tracker = CUDAGraphOptimizationTracker(config, region).eval()
    value = torch.randn(2, 8, device="cuda")
    with torch.inference_mode():
        first = tracker(value)
        second = tracker(value + 10.0)
        third = tracker(value - 10.0)
    assert first.data_ptr() != region.out.data_ptr()
    torch.testing.assert_close(first, value * 2.0, rtol=0, atol=0)
    torch.testing.assert_close(second, (value + 10.0) * 2.0, rtol=0, atol=0)
    torch.testing.assert_close(third, (value - 10.0) * 2.0, rtol=0, atol=0)
    assert tracker.execution_counts["capture"] == 1
    assert tracker.execution_counts["replay"] == 2
    assert not tracker.fallback_to_eager_by_key


def test_precision_contexts_have_independent_graph_keys() -> None:
    tracker = CUDAGraphOptimizationTracker(_early_config(), nn.Linear(8, 8).cuda()).eval()
    value = torch.randn(2, 8, device="cuda")
    with torch.inference_mode():
        ordinary = tracker(value)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reduced = tracker(value)
            torch.testing.assert_close(reduced, tracker.inner_module(value), rtol=0, atol=0)
        torch.testing.assert_close(tracker(value), ordinary, rtol=0, atol=0)
    assert ordinary.dtype == torch.float32
    assert reduced.dtype == torch.bfloat16
    assert tracker.execution_counts["capture"] == 2


def test_forcing_cubins_keys_another_graph(monkeypatch) -> None:
    """Ops choose their CuTeDSL source or CUBIN path per call, so a graph captured under one choice
    never replays under the other."""
    tracker = CUDAGraphOptimizationTracker(_early_config(), nn.Linear(8, 8).cuda()).eval()
    value = torch.randn(2, 8, device="cuda")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    source_key = tracker.input_key_for_this_call(value)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert tracker.input_key_for_this_call(value) != source_key
    with torch.inference_mode():
        tracker(value)
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "0")
        tracker(value)
    assert tracker.execution_counts["capture"] == 2


def test_graph_budget_evicts_old_states_but_keeps_one_graph() -> None:
    tracker = CUDAGraphOptimizationTracker(_early_config(graph_cache_budget_bytes=1), nn.ReLU().cuda()).eval()
    with torch.inference_mode():
        first = tracker(torch.randn(2, 8, device="cuda"))
        held = first.clone()
        tracker(torch.randn(3, 8, device="cuda"))
    torch.testing.assert_close(first, held, rtol=0, atol=0)
    assert len(tracker.graph_state_by_key) == 1
    assert tracker.execution_counts["capture"] == 2
    assert tracker.execution_counts["eviction"] == 1


def test_eviction_waits_for_the_evicted_replay_not_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evicting a state waits on its own last replay and leaves other queued work running."""
    tracker = CUDAGraphOptimizationTracker(_early_config(graph_cache_budget_bytes=1), nn.ReLU().cuda()).eval()
    with torch.inference_mode():
        tracker(torch.randn(2, 8, device="cuda"))
        old = torch.randn(2, 8, device="cuda")
        replayed = tracker(old)
    (state,) = tracker.graph_state_by_key.values()
    assert state.last_use is not None
    last_use = state.last_use
    waits: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: waits.append("device"))
    real_event_sync = torch.cuda.Event.synchronize

    def event_sync(event: torch.cuda.Event) -> None:
        waits.append("replay")
        real_event_sync(event)

    monkeypatch.setattr(torch.cuda.Event, "synchronize", event_sync)
    with torch.inference_mode():
        new = tracker(torch.randn(3, 8, device="cuda"))
    assert waits == ["replay"]
    assert last_use.query()
    assert tracker.execution_counts["eviction"] == 1
    torch.testing.assert_close(replayed, old.relu(), rtol=0, atol=0)
    assert new.shape == (3, 8)


# GPU clock cycles a test sleeps on the caller's stream, so the work queued behind the sleep is still pending when the
# host acts next.
_CALLER_DELAY_CYCLES = 200_000_000


@pytest.mark.parametrize(("output_step", "calls_before_eviction"), [("_caller_output", 2), ("_first_call_output", 1)])
def test_eviction_waits_for_the_output_copy(
    monkeypatch: pytest.MonkeyPatch, output_step: str, calls_before_eviction: int
) -> None:
    """An evicted state's buffers outlive the copy of the last call's output, however long it queues."""
    tracker = CUDAGraphOptimizationTracker(_early_config(graph_cache_budget_bytes=1), nn.ReLU().cuda()).eval()
    copied: list[torch.cuda.Event] = []
    real_step = getattr(CUDAGraphOptimizationTracker, output_step)

    def slow_step(self: CUDAGraphOptimizationTracker, *args: object) -> object:
        torch.cuda._sleep(_CALLER_DELAY_CYCLES)
        result = real_step(self, *args)
        copied.append(torch.cuda.Event())
        copied[-1].record()
        return result

    monkeypatch.setattr(CUDAGraphOptimizationTracker, output_step, slow_step)
    copied_at_release: list[bool] = []
    real_release = CUDAGraphState.release

    def release(state: CUDAGraphState) -> None:
        real_release(state)
        copied_at_release.append(copied[-1].query())

    monkeypatch.setattr(CUDAGraphState, "release", release)
    x = torch.randn(2, 8, device="cuda")
    with torch.inference_mode():
        for _ in range(calls_before_eviction):
            out = tracker(x)
        tracker(torch.randn(3, 8, device="cuda"))
    assert tracker.execution_counts["eviction"] == 1
    assert copied_at_release[0]
    torch.testing.assert_close(out, x.relu(), rtol=0, atol=0)


# Blocks the reuse test takes from the warmup stream's pool before it stops looking for the freed output's block.
_REUSE_PROBES_MAX = 64


@pytest.mark.parametrize("capture_on_first_call", [True, False])
def test_freed_first_output_keeps_its_memory_for_the_callers_reads(capture_on_first_call: bool) -> None:
    """The first call's output, freed while a read of it is queued, keeps its memory until that read has run.

    The output comes from the warmup stream's pool, and a later ``torch.cuda.Stream()`` can wrap the same CUDA
    stream. Here a second stream object on that CUDA stream takes blocks of the output's size and fills them.
    """
    tracker = CUDAGraphOptimizationTracker(
        _early_config(capture_on_first_call=capture_on_first_call), _CountingRegion().cuda()
    ).eval()
    value = torch.ones(1 << 20, device="cuda")
    with torch.inference_mode():
        output = tracker(value)
        warmup_stream = next(iter(tracker.graph_state_by_key.values())).warmup_stream
        torch.cuda._sleep(_CALLER_DELAY_CYCLES)
        seen = output * 1.0  # The caller's read, queued behind the sleep.
        freed = output.data_ptr()
        del output
        with torch.cuda.stream(torch.cuda.ExternalStream(warmup_stream.cuda_stream)):
            probes = [torch.empty_like(value)]
            while probes[-1].data_ptr() != freed and len(probes) < _REUSE_PROBES_MAX:
                probes.append(torch.empty_like(value))
            for probe in probes:
                probe.fill_(7.0)
        torch.cuda.synchronize()
    assert all(probe.data_ptr() != freed for probe in probes)
    torch.testing.assert_close(seen, value * 2.0, rtol=0, atol=0)


def _make():
    """Return (tracker over a fresh net, an independent eager clone)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).cuda().eval()
    raw = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    raw.load_state_dict(net.state_dict())
    raw = raw.cuda().eval()
    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(verify_capture=True), inner_module=net).eval()
    return tracker, raw


def _make_bucketed_tracker():
    factory = InputRoutingConfigFactory()
    factory.set_named_dim_ties([NamedDimTies(name="tokens", input_dims=(("input", (1,)),), output_dims=((0, (1,)),))])
    factory.set_input_acceptance_dim("tokens", 8)
    factory.set_padded_dim("tokens", 4, 8, 1, multiple_of=1)
    config = CUDAGraphOptimizationConfig(
        input_key_method=InputKeyMethod.BUCKETED_SHAPES,
        input_routing_config=factory.export_config(),
        verify_capture=True,
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
    # Verification is enabled: the eager/replay output check must have passed.
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
    monkeypatch.setattr(trk, "check_capacity_for_capture", lambda *a, **k: _REFUSED_CAPACITY)
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


def test_memory_gate_refuses_after_one_release_when_still_short(monkeypatch: pytest.MonkeyPatch) -> None:
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    events: list[str] = []

    def check(*_args: object, **_kwargs: object) -> gc_mem.MemoryCheck:
        events.append("check")
        return gc_mem.MemoryCheck(False, "insufficient GPU memory", 1 << 30, 1 << 20)

    monkeypatch.setattr(trk, "check_capacity_for_capture", check)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    with torch.no_grad():
        ref = raw(x)
        for _ in range(NUM_CALLS_TO_CAPTURE + 2):
            out = m(x)

    # One release and one recheck, then the key stays eager without further checks.
    assert events == ["check", "empty_cache", "check"]
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert torch.allclose(out, ref, atol=1e-5)


def test_memory_gate_refuses_host_shortage_without_releasing_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    events: list[str] = []

    def check(*_args: object, **_kwargs: object) -> gc_mem.MemoryCheck:
        events.append("check")
        return gc_mem.MemoryCheck(False, "insufficient host memory", 1 << 20, 1 << 30)

    monkeypatch.setattr(trk, "check_capacity_for_capture", check)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    with torch.no_grad():
        ref = raw(x)
        for _ in range(NUM_CALLS_TO_CAPTURE):
            out = m(x)

    assert events == ["check"]
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert torch.allclose(out, ref, atol=1e-5)


_AMPLE_FREE_BYTES = 40 << 30  # Far more than the test region's capture needs.
_STARVED_FREE_BYTES = 1 << 20  # Below the capture margin, so no capture fits.


def _fake_device_memory(
    monkeypatch: pytest.MonkeyPatch, free_bytes: int, free_after_release: int | None = None
) -> list[str]:
    """Fake free device memory and return the ordered cache releases and captures."""
    events: list[str] = []
    free = [free_bytes]

    def mem_get_info(device: torch.device | int | None = None) -> tuple[int, int]:
        return free[0], free[0]  # The capacity check reads only the free bytes.

    def empty_cache() -> None:
        events.append("release")
        if free_after_release is not None:
            free[0] = free_after_release

    real_capture = trk._capture_without_barrier

    def capture(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> AbstractContextManager[None]:
        events.append("capture")
        return real_capture(graph, stream)

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(trk, "_capture_without_barrier", capture)
    return events


def _run_to_capture(m: CUDAGraphOptimizationTracker, raw: nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        ref = raw(x)
        for _ in range(NUM_CALLS_TO_CAPTURE + 1):
            out = m(x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_capture_that_fits_keeps_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that fits in free memory leaves cached allocations in place.

    Args:
        monkeypatch: Patches device memory availability and records capture.
    """
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    events = _fake_device_memory(monkeypatch, free_bytes=_AMPLE_FREE_BYTES)
    _run_to_capture(m, raw, x)
    assert events == ["capture"]
    assert m.execution_counts["fallback"] == 0
    assert m.execution_counts["cache_release"] == 0


def test_capture_crowded_out_by_the_cache_releases_it_and_captures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that fits after releasing cached allocations still succeeds.

    Args:
        monkeypatch: Patches memory availability before and after release.
    """
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    events = _fake_device_memory(monkeypatch, free_bytes=_STARVED_FREE_BYTES, free_after_release=_AMPLE_FREE_BYTES)
    _run_to_capture(m, raw, x)
    assert events == ["release", "capture"]
    assert m.execution_counts["fallback"] == 0
    assert m.execution_counts["cache_release"] == 1
    assert m.graph_state_by_key[_key(m, x)].preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED


def test_cache_release_failure_reverts_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed cache release reverts the key to eager instead of raising.

    Args:
        monkeypatch: Starves device memory and makes the release raise.
    """
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")
    events = _fake_device_memory(monkeypatch, free_bytes=_STARVED_FREE_BYTES)

    def refuse_release() -> None:
        events.append("release")
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(torch.cuda, "empty_cache", refuse_release)
    _run_to_capture(m, raw, x)
    assert events == ["release"]
    assert m.execution_counts["cache_release"] == 0
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert _key(m, x) not in m.graph_state_by_key


def test_capture_failure_reverts_to_eager(monkeypatch):
    m, raw = _make()
    x = torch.randn(2, 8, device="cuda")

    def boom(*_args: object) -> None:
        raise RuntimeError("forced capture failure")

    monkeypatch.setattr(trk, "_capture_without_barrier", boom)
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


class _SyncsDuringCapture(nn.Module):
    """Runs eagerly, but synchronizing invalidates a stream capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.synchronize()
        return x * 2.0 + 1.0


def test_real_capture_error_reverts_to_eager():
    # torch appends its device-side-assertion hint to every CUDA error, so a
    # recoverable capture error must not be mistaken for a sticky fault.
    m = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=_SyncsDuringCapture()).eval()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE + 2):
            out = m(x)
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is True
    assert _key(m, x) not in m.graph_state_by_key
    assert torch.allclose(out, x * 2.0 + 1.0)


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


def test_release_frees_graph_and_buffers_and_is_idempotent():
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            m(x)
    state = m.graph_state_by_key[_key(m, x)]
    assert state.graph is not None

    state.release()
    # ``graph`` must still resolve to None rather than raise: several call sites
    # read it after a failed capture or replay.
    assert state.graph is None
    assert state.static_output is None
    assert state.static_input_arg is None
    assert state.warmup_stream is None
    assert state.preparation_state == CUDAGraphPreparationState.WARMUP
    state.release()


def test_release_cleans_up_after_a_warmup_that_raised():
    class _Raises(nn.Module):
        def forward(self, x):
            return (x * 2).sum() + _boom()

    def _boom():
        raise RuntimeError("warmup blew up")

    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=_Raises().cuda()).eval()
    x = torch.randn(8, 8, device="cuda")
    with torch.no_grad(), pytest.raises(RuntimeError, match="warmup blew up"):
        tracker(x)

    state = next(iter(tracker.graph_state_by_key.values()))
    assert state.graph is None
    assert state.static_output is None
    assert state.warmup_stream is not None
    assert state.static_unadjusted_input_tensor_shapes is not None

    state.release()
    assert state.warmup_stream is None
    assert state.static_unadjusted_input_tensor_shapes is None
    state.release()


def test_release_skips_a_state_that_holds_nothing():
    # The synchronize is a device-wide barrier, so an untouched state must not pay it.
    state = CUDAGraphState()
    state.release()
    assert state.graph is None
    assert state.warmup_stream is None


def test_lru_overflow_releases_the_evicted_graph():
    # With one graph per module, each new shape evicts the previous one. The
    # eviction must tear the graph down, not leave it to an arbitrary later
    # garbage collection.
    config = CUDAGraphOptimizationConfig(num_graphs_max_for_this_module=1)
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).cuda().eval()
    tracker = CUDAGraphOptimizationTracker(config, inner_module=net).eval()

    first = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            tracker(first)
    evicted = tracker.graph_state_by_key[_key(tracker, first)]
    assert evicted.graph is not None

    second = torch.randn(3, 8, device="cuda")
    with torch.no_grad():
        tracker(second)

    assert _key(tracker, first) not in tracker.graph_state_by_key
    assert evicted.graph is None
    assert evicted.static_output is None


def _cublas_workspace_bytes() -> int:
    """Size of torch's cached cuBLAS workspace (``parseChosenWorkspaceSize`` in CublasHandlePool.cpp)."""
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config:
        return sum(int(size) * 1024 * int(count) for size, count in re.findall(r":([0-9]+):([0-9]+)", config))
    if torch.cuda.get_device_capability()[0] in (9, 10, 12):
        return 4096 * 8 * 1024
    return 4096 * 1024 * 2 + 16 * 1024 * 8


def test_eviction_leaves_the_cublas_workspace_of_a_graph_on_the_same_stream(monkeypatch):
    # Graph states draw streams from torch's recycled pool, and destroying a graph frees
    # the cuBLAS workspace torch caches for its stream. A graph captured on the same
    # stream with that cached workspace replayed into the freed block: a fault once
    # released, otherwise silent writes into whatever took it over.
    stream = torch.cuda.Stream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: stream)
    # A long reduction makes cuBLAS split K through its workspace.
    linear = nn.Linear(65536, 64, bias=False).cuda()
    tracker = CUDAGraphOptimizationTracker(_early_config(), inner_module=linear).eval()
    first = torch.randn(64, 65536, device="cuda")
    second = torch.randn(65, 65536, device="cuda")
    with torch.inference_mode():
        expected = linear(second)
        tracker(first)
        (first_key,) = tracker.graph_state_by_key.keys()
        tracker(second)
        tracker._evict_key(first_key)
        # Take over the block the eviction freed, if any, with a sentinel.
        with torch.cuda.stream(stream):
            sentinel = torch.full((_cublas_workspace_bytes() // 4,), 7.0, device="cuda")
        # Replay runs on the current stream: order it after the fill, which could mask its writes.
        torch.cuda.current_stream().wait_stream(stream)
        replay = tracker(second)
    torch.cuda.synchronize()

    assert tracker.execution_counts["replay"] == 1
    assert torch.equal(sentinel, torch.full_like(sentinel, 7.0))
    torch.testing.assert_close(replay, expected)


def test_evict_key_tolerates_an_already_evicted_key():
    # Overlapping failure paths must not turn one fault into a KeyError.
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            m(x)
    key = _key(m, x)
    m._revert_to_eager(key)
    m._revert_to_eager(key)
    assert key not in m.graph_state_by_key


def test_device_fatal_capture_error_propagates(monkeypatch):
    # A poisoned CUDA context cannot be recovered by falling back to eager, and
    # swallowing it reports the fault at an unrelated call site later.
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")

    def illegal_access(*_args: object) -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(trk, "_capture_without_barrier", illegal_access)
    with torch.no_grad(), pytest.raises(RuntimeError, match="illegal memory access"):
        for _ in range(NUM_CALLS_TO_CAPTURE):
            m(x)
    assert m.fallback_to_eager_by_key.get(_key(m, x)) is None


def test_device_fatal_replay_error_propagates():
    m, _ = _make()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            m(x)
    state = m.graph_state_by_key[_key(m, x)]

    def _illegal_access():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    state.graph.replay = _illegal_access
    with torch.no_grad(), pytest.raises(RuntimeError, match="illegal memory access"):
        m(x)


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
