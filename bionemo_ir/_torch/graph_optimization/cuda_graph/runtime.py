# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-key CUDA-graph warmup, capture, verification, and replay.

States are LRU-cached. Memory, capture, verification, or replay failures
permanently revert the affected key to eager execution.
"""

import enum
import gc
import re
import weakref
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from lru import LRU
from torch import Tensor
from torch.cuda.streams import Stream

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, InputKeyMethod
from bionemo_ir._torch.graph_optimization.cuda_graph.memory import (
    check_capacity_for_capture,
    container_device,
    tensor_bytes,
)
from bionemo_ir._torch.graph_optimization.tensor_copy_utils import (
    _assert_equal_but_distinct,
    _clone_tensors,
    _copy_kept_leaves_into,
    _copy_tensors_into,
    _delete_tensors_in_container,
    _tensor_leaves,
)
from bionemo_ir._torch.graph_optimization.tracker import (
    GraphOptimizationTracker,
    TensorContainerHostShapes,
    TensorContainerShapes,
)
from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache
from bionemo_ir.logger import logger
from bionemo_ir.utils import is_device_fatal

_EAGER: ContextVar[bool] = ContextVar("bioir_eager_graphs", default=False)
_SIGNATURES_MAX_FOR_RECAPTURE_CHECK = 64
_LIST_INDEX_SUFFIX = re.compile(r"(\.\d+)+$")


def graph_eligible(training: bool) -> bool:
    """Whether calls may take the CUDA-graph path: not under gradients, training, :func:`eager_graphs` or a capture."""
    return not (torch.is_grad_enabled() or training or _EAGER.get() or torch.cuda.is_current_stream_capturing())


@contextmanager
def eager_graphs(enabled: bool = True) -> Iterator[None]:
    """Run graph regions and trackers called in this scope eagerly.

    An enclosing graph uses it while it prepares and captures, so nested graphs run inside it. Models use it
    for calls a graph would not pay off for, such as a confidence head that runs once.
    """
    token = _EAGER.set(_EAGER.get() or enabled)
    try:
        yield
    finally:
        _EAGER.reset(token)


@contextmanager
def graph_owned_cublas_workspace() -> Iterator[None]:
    """Make a capture allocate its cuBLAS workspace in its own pool, out of reach of later work.

    PyTorch caches one cuBLAS workspace per (handle, stream) and frees a stream's entry
    when any graph captured on that stream is destroyed. A capture that reuses a cached
    workspace keeps its address, and graph states draw their streams from torch's recycled
    pool, so evicting one graph freed the workspace another live graph still replayed
    into: an illegal address once the allocator returned it, or silent writes into
    whatever reused it. With the cache empty, the capture allocates a fresh workspace in
    its private pool; emptying it again afterwards keeps that workspace from being handed
    to eager calls or later captures. torch's cudagraph trees do the same around capture.
    """
    torch._C._cuda_clearCublasWorkspaces()
    try:
        yield
    finally:
        torch._C._cuda_clearCublasWorkspaces()


@contextmanager
def _capture_without_barrier(graph: torch.cuda.CUDAGraph, stream: Stream) -> Iterator[None]:
    """Capture on ``stream`` as ``torch.cuda.graph`` does, without the work it does on entry.

    ``torch.cuda.graph`` synchronizes the device and empties the device and
    pinned-host caching allocators, and collects garbage when configured to.
    Capture records launches without running them, so work already queued,
    such as the warmup whose result the call returns, need not finish first.
    The private pool allocates fresh segments either way, and the capacity
    check reads driver-free memory, which excludes cached blocks; flushing the
    cache would only make later eager allocations pay ``cudaMalloc`` again.
    """
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="global")
        try:
            yield
        finally:
            graph.capture_end()


class CUDAGraphPreparationState(enum.Enum):
    """Per-key warmup-to-replay state."""

    WARMUP = 0
    WARMUP_KERNELS_COMPILED = 1
    WARMUP_MEMORY_ALLOCATOR_READY = 2
    GRAPH_CAPTURED = 3
    GRAPH_VERIFIED = 4


class CUDAGraphState:
    """Per-key state, static buffers, stream, and captured graph."""

    def __init__(self) -> None:
        self.preparation_state: CUDAGraphPreparationState = CUDAGraphPreparationState.WARMUP
        self.num_prev_calls_by_input_key: int = 0
        self.warmup_stream: Stream | None = None
        self.static_input_arg: tuple[Tensor, ...] | None = None
        self.static_input_kwargs: dict[str, Tensor] | None = None
        self.static_output: Tensor | tuple[Tensor, ...] | None = None
        self.static_unadjusted_input_tensor_shapes: dict[str, Tensor] | None = None
        self.cached_input_key: str | None = None
        self.cached_input_tensor_shapes_host: TensorContainerHostShapes | None = None
        self.static_unadjusted_output_tensor_shapes: dict[str, Tensor] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        # Static buffers pinned by this key's graph.
        self.working_set_bytes: int = 0
        # Peak warmup activations later retained by the graph mempool.
        self.warmup_peak_activation_bytes: int = 0
        # Per stable kwarg, the tensors (and their versions) its static buffer last received.
        self.stable_sources: dict[str, list[tuple[weakref.ref, int | None]]] = {}
        # Recorded once a call's output is out of the static buffers, so teardown waits for this graph's work only.
        self.last_use: torch.cuda.Event | None = None

    def mark_used(self) -> None:
        """Record, on the current stream, the end of the last work queued that reads this state's buffers."""
        if self.last_use is None:
            self.last_use = torch.cuda.Event()
        self.last_use.record()

    def release(self) -> None:
        """Wait for this graph's last use, then drop this key's buffers and graph. Idempotent.

        Leaving teardown to refcounting fires whenever the last reference happens
        to drop, possibly mid-forward or during a later capture, and destroys the
        graph without draining the replays still in flight.

        The last use covers a call's replay and the copy of its output out of the
        static buffers, so it suffices whichever stream allocated them and however
        the allocator hands their memory on. A device-wide sync would also wait for
        unrelated queued work, such as the warmup of the key whose capture evicts
        this one.
        """
        if not any(
            resource is not None
            for resource in (
                self.graph,
                self.warmup_stream,
                self.static_input_arg,
                self.static_input_kwargs,
                self.static_output,
                self.static_unadjusted_input_tensor_shapes,
                self.static_unadjusted_output_tensor_shapes,
            )
        ):
            return
        if self.last_use is not None:
            self.last_use.synchronize()
            self.last_use = None

        # Static outputs live in the graph's private pool, so they go before it does.
        _delete_tensors_in_container(self.static_input_arg)
        _delete_tensors_in_container(self.static_input_kwargs)
        _delete_tensors_in_container(self.static_output)
        _delete_tensors_in_container(self.static_unadjusted_input_tensor_shapes)
        _delete_tensors_in_container(self.static_unadjusted_output_tensor_shapes)
        self.static_input_arg = None
        self.static_input_kwargs = None
        self.static_output = None
        self.static_unadjusted_input_tensor_shapes = None
        self.static_unadjusted_output_tensor_shapes = None
        self.stable_sources = {}

        # Assign rather than ``del``: later reads must see None, not AttributeError.
        self.graph = None
        self.warmup_stream = None
        self.preparation_state = CUDAGraphPreparationState.WARMUP

    def __del__(self) -> None:
        """Release resources without raising during partial init or shutdown."""
        try:
            self.release()
        except Exception:
            pass


def _leaf_sources(value: Any) -> list[tuple[weakref.ref, int | None]]:
    """Record the tensor objects copied from ``value`` and their version counters."""
    return [(weakref.ref(leaf), None if leaf.is_inference() else leaf._version) for leaf in _tensor_leaves(value)]


def _unchanged_since_copy(sources: list[tuple[weakref.ref, int | None]] | None, value: Any) -> bool:
    """Whether ``value``'s tensor leaves are the objects ``sources`` recorded, unmodified since.

    Identity, not address, decides: freed storage can return at the same address
    under a new tensor, but never as the same object. Inference tensors keep no
    version counter, so identity alone decides for them.
    """
    if sources is None:
        return False
    leaves = _tensor_leaves(value)
    return len(leaves) == len(sources) and all(
        ref() is leaf and (version is None or leaf._version == version)
        for (ref, version), leaf in zip(sources, leaves, strict=True)
    )


def _rng_devices(device: torch.device | None) -> list[int]:
    """The CUDA devices whose RNG state ``torch.random.fork_rng`` saves for work on ``device``."""
    if device is None:
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _record_caller_stream(value: Any) -> None:
    """Mark each CUDA tensor in ``value`` as in use on its device's current stream, the caller's.

    A tensor that a warmup stream allocated goes back to that stream's pool when
    the caller frees it, and PyTorch hands the same CUDA stream to later
    ``torch.cuda.Stream()`` callers. With the mark, the allocator reuses the
    memory only after the work that the caller's stream queued before the release.
    """
    for leaf in _tensor_leaves(value):
        if leaf.is_cuda:
            leaf.record_stream(torch.cuda.current_stream(leaf.device))


def cudagraph_delete_callback(_input_key: str, value: CUDAGraphState, counts: Counter[str] | None = None) -> None:
    """Release an LRU-evicted graph state.

    The LRU drops its reference after this returns, so the state is collected either
    way; releasing here makes the teardown ordered and prompt.
    """
    value.release()
    if counts is not None:
        counts["eviction"] += 1


def _reset_before_load(tracker: "CUDAGraphOptimizationTracker", *args: object) -> None:
    """Release captured storage before state-dict loading can replace it."""
    tracker.forget_captures()


class CUDAGraphOptimizationTracker(GraphOptimizationTracker):
    """Drive per-key CUDA-graph warmup, capture, and replay."""

    def __init__(self, config: CUDAGraphOptimizationConfig, inner_module: nn.Module | None = None) -> None:
        """Initialize the tracker and per-key LRU cache.

        Args:
            config: CUDA-graph configuration.
            inner_module: Wrapped eager module.

        Raises:
            ValueError: If the config type is invalid.
        """
        super().__init__(config, inner_module=inner_module)
        if not isinstance(self.graph_optimization_config, CUDAGraphOptimizationConfig):
            raise ValueError(f"Expected CUDAGraphOptimizationConfig, got {type(self.graph_optimization_config)}")

        self.execution_counts: Counter[str] = Counter()
        self.graph_state_by_key = LRU(
            size=self.graph_optimization_config.num_graphs_max_for_this_module,
            callback=partial(cudagraph_delete_callback, counts=self.execution_counts),
        )

        # Tracker-level flags survive state eviction, keeping failed keys eager.
        self.fallback_to_eager_by_key: dict[str, bool] = {}
        # Non-tensor metadata last seen per tensor signature, to name values that recapture a same-shape call.
        self.nontensor_metadata_by_signature = LRU(_SIGNATURES_MAX_FOR_RECAPTURE_CHECK)
        self._reported_recapture_paths: set[tuple[str, ...]] = set()
        # Set by the first capture kept since construction, reset, move or reload. With capture on the first call,
        # later signatures then skip the allocator warmups.
        self.allocator_warmed_up = False
        # A function rather than a bound method keeps the tracker out of a reference cycle, so a dropped
        # tracker frees its graphs at once, not in a garbage collection that may run inside another capture.
        self.register_load_state_dict_pre_hook(_reset_before_load)

    def input_key_for_this_call(self, *args, **kwargs) -> str:
        """Keep graphs separate across precision, deterministic and CuTeDSL source-or-CUBIN contexts."""
        context = (
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.are_deterministic_algorithms_enabled(),
            CuteKernelCache.force_cubin(),
        )
        return f"{super().input_key_for_this_call(*args, **kwargs)};context={context}"

    def __del__(self) -> None:
        """Drop cached states without raising during shutdown."""
        try:
            for state in self.graph_state_by_key.values():
                state.release()
            self.graph_state_by_key.clear()
            del self.graph_state_by_key
        except Exception:
            pass

    def reset(self) -> None:
        """Clear graph states and permanent-eager flags.

        Collection plus CUDA synchronization returns graph pools to the
        allocator deterministically.
        """
        graph_state_by_key = getattr(self, "graph_state_by_key", None)
        if graph_state_by_key is not None:
            for state in graph_state_by_key.values():
                state.release()
            graph_state_by_key.clear()
        fallback_to_eager_by_key = getattr(self, "fallback_to_eager_by_key", None)
        if fallback_to_eager_by_key is not None:
            fallback_to_eager_by_key.clear()
        self.allocator_warmed_up = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def forget_captures(self) -> None:
        """Drop every graph and the warm-allocator mark before parameter storage changes."""
        if self.graph_state_by_key:
            self.reset()
        self.allocator_warmed_up = False

    def update_graph_state_by_key(self, *args, **kwargs) -> tuple[str, int]:
        """Create or advance this call's graph state.

        Permanently eager keys return without recreating state.

        Returns:
            Input key and its prior-call count.
        """
        key = self.input_key_for_this_call(*args, **kwargs)
        return key, self._advance_graph_state_by_key(key)

    def _advance_graph_state_by_key(self, key: str) -> int:
        """Create or advance a state for an already-derived key."""
        # Failed keys stay eager even after their state is evicted.
        if self.fallback_to_eager_by_key.get(key, False):
            return 0
        if key not in self.graph_state_by_key:
            self.graph_state_by_key[key] = CUDAGraphState()
        else:
            self.graph_state_by_key[key].num_prev_calls_by_input_key += 1

        state = self.graph_state_by_key[key]
        if state.preparation_state == CUDAGraphPreparationState.WARMUP:
            if state.num_prev_calls_by_input_key >= self.graph_optimization_config.num_calls_for_kernel_compilation:
                state.preparation_state = CUDAGraphPreparationState.WARMUP_KERNELS_COMPILED

        if state.preparation_state == CUDAGraphPreparationState.WARMUP_KERNELS_COMPILED:
            if state.num_prev_calls_by_input_key >= self.graph_optimization_config.num_calls_for_memory_allocator:
                state.preparation_state = CUDAGraphPreparationState.WARMUP_MEMORY_ALLOCATOR_READY

        return state.num_prev_calls_by_input_key

    def _find_cached_graph_state(self, input_key: str) -> tuple[str, CUDAGraphState] | None:
        """Find the graph state that cached an unadjusted input key."""
        if input_key in self.graph_state_by_key:
            state = self.graph_state_by_key[input_key]
            if state.cached_input_key == input_key:
                return input_key, state
        graph_key = next(
            (graph_key for graph_key, state in self.graph_state_by_key.items() if state.cached_input_key == input_key),
            None,
        )
        if graph_key is None:
            return None
        return graph_key, self.graph_state_by_key[graph_key]

    def forward_udf(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        """Eager execution of the wrapped module (the fallback target)."""
        return self.inner_module(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        """Execute through warmup, capture, replay, or eager fallback."""
        if not graph_eligible(self.training):
            return self.inner_module(*args, **kwargs)

        key_method = self.graph_optimization_config.input_key_method
        if key_method not in (InputKeyMethod.EXACT, InputKeyMethod.BUCKETED_SHAPES):
            raise ValueError(f"Unsupported input key method: {key_method}")

        raw_input_key = self.input_key_for_this_call(*args, **kwargs)
        if key_method == InputKeyMethod.EXACT and self.fallback_to_eager_by_key.get(raw_input_key, False):
            return self.inner_module(*args, **kwargs)

        cached = self._find_cached_graph_state(raw_input_key)
        if cached is not None:
            input_key, state = cached
            input_tensor_shapes_host = state.cached_input_tensor_shapes_host
            if input_tensor_shapes_host is None:
                raise RuntimeError("Cached input shapes were not initialized")

            if key_method == InputKeyMethod.EXACT:
                adjusted_args = args
                adjusted_kwargs = kwargs
            else:
                adjusted_args, adjusted_kwargs = self.pad_input(
                    args, kwargs, input_tensor_shapes=input_tensor_shapes_host
                )

            if state.preparation_state not in (
                CUDAGraphPreparationState.GRAPH_CAPTURED,
                CUDAGraphPreparationState.GRAPH_VERIFIED,
            ):
                self._advance_graph_state_by_key(input_key)
        else:
            host_shapes = self._extract_host_shapes(args, kwargs)
            self.validate_input_ties(host_shapes)
            if not self.input_accepted(host_shapes):
                self.execution_counts["out_of_range"] += 1
                return self.inner_module(*args, **kwargs)
            input_tensor_shapes_device, input_tensor_shapes_host = self._extract_tensor_container_shape_maps(
                args, kwargs
            )
            self.validate_input_ties(input_tensor_shapes_host)
            if not self.input_accepted(input_tensor_shapes_host):
                return self.inner_module(*args, **kwargs)
            self._check_metadata_recapture(args, kwargs)

            if key_method == InputKeyMethod.EXACT:
                adjusted_args = args
                adjusted_kwargs = kwargs
                input_key = raw_input_key
                self._advance_graph_state_by_key(input_key)
            else:
                adjusted_args, adjusted_kwargs = self.pad_input(
                    args, kwargs, input_tensor_shapes=input_tensor_shapes_host
                )
                input_key, _ = self.update_graph_state_by_key(*adjusted_args, **adjusted_kwargs)
                if self.fallback_to_eager_by_key.get(input_key, False):
                    return self.inner_module(*args, **kwargs)

            state = self.graph_state_by_key[input_key]
            if state.static_unadjusted_input_tensor_shapes is None:
                state.static_unadjusted_input_tensor_shapes = input_tensor_shapes_device
            elif key_method == InputKeyMethod.BUCKETED_SHAPES:
                _copy_tensors_into(
                    dest=state.static_unadjusted_input_tensor_shapes,
                    src=input_tensor_shapes_device,
                )
            state.cached_input_key = raw_input_key
            state.cached_input_tensor_shapes_host = input_tensor_shapes_host

        ps = state.preparation_state
        if ps in (CUDAGraphPreparationState.WARMUP, CUDAGraphPreparationState.WARMUP_KERNELS_COMPILED):
            if self.config.capture_on_first_call:
                adjusted_output = self._first_capture_call(adjusted_args, adjusted_kwargs, input_key)
            else:
                adjusted_output = self._warmup_call(adjusted_args, adjusted_kwargs, input_key)
                _record_caller_stream(adjusted_output)

        elif ps == CUDAGraphPreparationState.WARMUP_MEMORY_ALLOCATOR_READY:
            adjusted_output = self._capture_call(adjusted_args, adjusted_kwargs, input_key)

        elif ps in (CUDAGraphPreparationState.GRAPH_CAPTURED, CUDAGraphPreparationState.GRAPH_VERIFIED):
            adjusted_output = self._replay_call(adjusted_args, adjusted_kwargs, input_key)

        else:
            raise ValueError(f"Invalid state {ps} for key {input_key}")

        if key_method == InputKeyMethod.EXACT:
            output = adjusted_output
        else:
            # Wrapping a lone tensor avoids iterating away its leading axis.
            if isinstance(adjusted_output, tuple):
                output = self.unpad_output(adjusted_output, input_tensor_shapes_host)
            else:
                output = self.unpad_output((adjusted_output,), input_tensor_shapes_host)

        return output

    def _check_metadata_recapture(self, args: tuple, kwargs: dict) -> None:
        """Count, and report once per path set, a new key whose tensors match an earlier one.

        Such a key differs only in non-tensor values, usually request metadata a caller passed in a
        container; each new value then evicts, warms up and recaptures a graph of the same shape.
        ``InputRoutingConfig.feature_kwargs`` keeps those values out.
        """
        tensor_parts: list[str] = []
        nontensor_leaves: set[tuple[str, str]] = set()

        def on_leaf(path: str, leaf: Any) -> None:
            is_tensor, part = self._leaf_metadata(path, leaf)
            if is_tensor:
                tensor_parts.append(part)
            else:
                nontensor_leaves.add((path, part))

        self._walk_call(args, kwargs, on_leaf)
        signature = self.SEP_FOR_ARGS.join(tensor_parts)
        current = frozenset(nontensor_leaves)
        prior = self.nontensor_metadata_by_signature.get(signature)
        self.nontensor_metadata_by_signature[signature] = current
        if prior is None or prior == current:
            return
        # Only paths are reported, never values. A top-level scalar such as a conditioning flag selects a graph on
        # purpose; only values inside a container are suspect.
        changed = {path for path, _ in prior ^ current if self.SEP_BW_ARG_AND_DIM in path}
        paths = tuple(sorted({_LIST_INDEX_SUFFIX.sub("", path) for path in changed}))
        if not paths:
            return
        self.execution_counts["metadata_recapture"] += 1
        if paths in self._reported_recapture_paths:
            return
        self._reported_recapture_paths.add(paths)
        logger.warning(
            f"{type(self).__name__}: same-shape call recaptures on non-tensor inputs {list(paths)}; "
            "list the read keys in feature_kwargs"
        )

    def _first_capture_call(self, args: tuple, kwargs: dict, input_key: str) -> Tensor | tuple[Tensor, ...]:
        """Warm up and capture a new signature and return its last warmup's result, or run eagerly if capture fails.

        Warmups run on copies of the inputs, so the caller's tensors stay
        unmutated, as under a replay. Warmups before the last do not consume the
        prediction's RNG. The last one does, so its output is this call's result
        and the capture that follows need not replay it: the same kernels run on
        equal values with equal RNG offsets, so for deterministic kernels the
        result equals a replay's. If capture fails, the RNG goes back and the
        region runs eagerly on the caller's inputs, which it may mutate.
        """
        device = container_device((args, kwargs))
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device is not None else None
        self._run_isolated_warmups(args, kwargs, input_key, device)
        state = self.graph_state_by_key[input_key]
        # The last warmup and the copy of its result stay queued on the warmup stream, so the capture records while
        # they run; the input copies live until the caller's stream has joined that stream.
        warmup_inputs = self._warmup_copies(args, kwargs)
        output = self._warmup_call(*warmup_inputs, input_key, join=False)
        with torch.cuda.stream(state.warmup_stream):
            # The region may return storage it owns, such as a buffer, that later replays overwrite.
            output = _clone_tensors(output)
        self._advance_graph_state_by_key(input_key)
        # A refused capture evicts the state and its stream, so read the stream to join first.
        warmup_stream = state.warmup_stream
        captured = self._capture(args, kwargs, input_key)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        # The result's copy came from the warmup stream's pool, and from here on the caller's stream reads it.
        _record_caller_stream(output)
        del warmup_inputs
        if captured:
            output = self._first_call_output(state, output)
            state.mark_used()
            return output

        del output
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        return self.inner_module(*args, **kwargs)

    def _run_isolated_warmups(self, args: tuple, kwargs: dict, input_key: str, device: torch.device | None) -> None:
        """Run a new signature's warmups before the last on input copies, without consuming the prediction's RNG.

        Only the tracker's first capture runs the allocator warmups.
        """
        cfg = self.graph_optimization_config
        warmups = cfg.num_calls_for_kernel_compilation
        if not self.allocator_warmed_up:
            warmups = max(warmups, cfg.num_calls_for_memory_allocator)
        state = self.graph_state_by_key[input_key]
        if state.num_prev_calls_by_input_key + 1 < warmups:
            with torch.random.fork_rng(devices=_rng_devices(device)):
                while state.num_prev_calls_by_input_key + 1 < warmups:
                    self._warmup_call(*self._warmup_copies(args, kwargs), input_key)
                    self._advance_graph_state_by_key(input_key)

    def _warmup_copies(self, args: tuple, kwargs: dict) -> tuple[Any, dict]:
        """Copies of a call's inputs for a warmup; stable kwargs, which the module never writes, stay shared."""
        stable = self._stable_input_kwargs()
        keep = tuple(_tensor_leaves([kwargs[name] for name in stable if name in kwargs]))
        return _clone_tensors(args), _clone_tensors(kwargs, keep)

    def _reserve_graph_budget(self, input_key: str) -> None:
        """Evict older states until this signature fits the estimated budget."""
        budget = self.config.graph_cache_budget_bytes
        if budget is None:
            return
        while len(self.graph_state_by_key) > 1:
            resident = sum(
                state.working_set_bytes + state.warmup_peak_activation_bytes
                for state in self.graph_state_by_key.values()
            )
            if resident <= budget:
                return
            oldest = next(key for key in reversed(list(self.graph_state_by_key.keys())) if key != input_key)
            self._evict_key(oldest)

    def _warmup_call(
        self,
        args: Tensor | tuple | Any,
        kwargs: dict,
        input_key: str,
        join: bool = True,
    ) -> Tensor | tuple[Tensor, ...]:
        """Warm up on a side stream and initialize the static output.

        Without ``join``, the current stream does not wait for the warmup: the
        caller joins later and keeps ``args`` and ``kwargs`` alive until then.
        """
        state = self.graph_state_by_key[input_key]
        self.execution_counts["warmup"] += 1

        if state.warmup_stream is None:
            state.warmup_stream = torch.cuda.Stream()

        # Capture retains warmup activations beyond static buffer bytes. Keep the
        # highest per-device peak, including possible autotuning workspace.
        device = container_device((args, kwargs))
        if device is not None:
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(state.warmup_stream):
            adjusted_output = self._f_capture(
                args,
                kwargs,
                input_tensor_shapes=state.static_unadjusted_input_tensor_shapes,
            )
            if device is not None:
                peak_activation = torch.cuda.max_memory_allocated(device) - allocated_before
                state.warmup_peak_activation_bytes = max(state.warmup_peak_activation_bytes, peak_activation)
            if state.static_output is None:
                state.static_output = _clone_tensors(adjusted_output)

        if join:
            torch.cuda.current_stream().wait_stream(state.warmup_stream)

        return adjusted_output

    def _capture_call(self, args: tuple, kwargs: dict, input_key: str) -> Tensor | tuple[Tensor, ...]:
        """Capture, then replay this call.

        Replays copy every input except stable kwargs whose tensors are the same,
        unmodified objects as last time.
        """
        if not self._capture(args, kwargs, input_key):
            return self.inner_module(*args, **kwargs)
        # Capture may mutate its static inputs. Refreshing before the first replay,
        # just as on every later call, applies an in-place block once.
        return self._replay_for_caller(args, kwargs, input_key, self.graph_state_by_key[input_key])

    def _capture(self, args: tuple, kwargs: dict, input_key: str) -> bool:
        """Check capacity, capture, and optionally verify.

        Any failure permanently reverts this key to eager.

        Returns:
            Whether the key holds a usable graph.
        """
        # The warmup stream's dependency on the current stream preserves input
        # ordering; queued warmups keep running while the host records the graph.
        state = self.graph_state_by_key[input_key]
        if state.warmup_stream is None:
            state.warmup_stream = torch.cuda.Stream()

        state.warmup_stream.wait_stream(torch.cuda.current_stream())

        # Capture requires fixed-address inputs; replay refreshes their values.
        if state.static_input_arg is None:
            state.static_input_arg = _clone_tensors(args)
        if state.static_input_kwargs is None:
            state.static_input_kwargs = _clone_tensors(kwargs, self._borrowed_tensors(kwargs))
            state.stable_sources = {
                name: _leaf_sources(kwargs[name]) for name in self._stable_input_kwargs() if name in kwargs
            }

        state.working_set_bytes = (
            tensor_bytes(state.static_input_arg)
            + tensor_bytes(state.static_input_kwargs)
            + tensor_bytes(state.static_unadjusted_input_tensor_shapes)
            + tensor_bytes(state.static_output)
        )
        self._reserve_graph_budget(input_key)
        # Include warmup activations retained by the graph mempool.
        capture_bytes = state.working_set_bytes + state.warmup_peak_activation_bytes
        static_inputs = (state.static_input_arg, state.static_input_kwargs)
        check = check_capacity_for_capture(working_set_bytes=capture_bytes, input_container=static_inputs)
        if check.gpu_short:
            # Free memory excludes the allocator's unused cache, which capture's
            # private pool cannot reuse. Releasing it waits for the device, so
            # it happens only when the capture would not fit otherwise.
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - e.g. another thread's capture refuses cudaFree -> eager
                self._revert_unless_fatal(input_key, "cache release", exc)
                return False
            self.execution_counts["cache_release"] += 1
            check = check_capacity_for_capture(working_set_bytes=capture_bytes, input_container=static_inputs)
        if not check.ok:
            logger.info(f"{type(self).__name__}: skipping capture, revert to eager ({check.reason})")
            self._revert_to_eager(input_key)
            return False

        # Capture on the warmup stream to preserve allocator stream ownership.
        graph = torch.cuda.CUDAGraph()
        try:
            with graph_owned_cublas_workspace(), _capture_without_barrier(graph, state.warmup_stream):
                state.static_output = self._f_capture(
                    state.static_input_arg,
                    state.static_input_kwargs,
                    input_tensor_shapes=state.static_unadjusted_input_tensor_shapes,
                )

        except Exception as exc:  # noqa: BLE001 - any capture failure -> eager
            self._revert_unless_fatal(input_key, "capture", exc)
            return False

        state.graph = graph
        self.execution_counts["capture"] += 1
        state.preparation_state = CUDAGraphPreparationState.GRAPH_CAPTURED

        if self.config.verify_capture:
            if not self._verify_capture(args, kwargs, input_key):
                return False
            state.preparation_state = CUDAGraphPreparationState.GRAPH_VERIFIED
        self.allocator_warmed_up = True
        return True

    def _revert_unless_fatal(self, input_key: str, step: str, exc: Exception) -> None:
        """Re-raise a device-fatal error from ``step``; otherwise log it and permanently revert the key to eager.

        A sticky CUDA error is not a capture problem: eager execution would hit
        the same poisoned context and report the fault somewhere unrelated.
        """
        if is_device_fatal(exc):
            raise exc
        logger.info(f"{type(self).__name__}: {step} failed, revert to eager ({exc})")
        self._revert_to_eager(input_key)

    def _verify_capture(self, args: tuple, kwargs: dict, input_key: str) -> bool:
        """Compare capture output with eager and revert on mismatch.

        The verifying replay writes the static buffers, and a borrowed kwarg's buffer is the caller's own tensor,
        so its value is put back for the call's own replay or eager fallback.

        Returns:
            Whether outputs match.
        """
        state = self.graph_state_by_key[input_key]
        devices = _rng_devices(container_device((args, kwargs)))
        with torch.random.fork_rng(devices=devices):
            eager_out = self._f_capture(
                _clone_tensors(args),
                _clone_tensors(kwargs),
                input_tensor_shapes=state.static_unadjusted_input_tensor_shapes,
            )
        borrowed = self._borrowed_tensors(kwargs)
        saved = _clone_tensors(borrowed)
        with torch.random.fork_rng(devices=devices):
            self._refresh_inputs(args, kwargs, state)
            state.graph.replay()
            replay_out = _clone_tensors(state.static_output)
        _copy_tensors_into(borrowed, saved)
        state.mark_used()
        try:
            _assert_equal_but_distinct(eager_out, replay_out)
            return True
        except AssertionError as exc:
            logger.info(f"{type(self).__name__}: capture verification failed, revert to eager ({exc})")
            self._revert_to_eager(input_key)
            return False

    def _replay_call(self, args: Tensor | tuple, kwargs: dict, input_key: str) -> Tensor | tuple[Tensor, ...]:
        self.execution_counts["replay"] += 1
        return self._replay_for_caller(args, kwargs, input_key, self.graph_state_by_key[input_key])

    def _replay_for_caller(
        self, args: tuple, kwargs: dict, input_key: str, state: CUDAGraphState
    ) -> Tensor | tuple[Tensor, ...]:
        """Refresh the static inputs, replay, and return the caller's output, or run eagerly if the replay fails.

        Only real inputs are copied: the captured graph owns workspace kwargs, and
        an unchanged stable kwarg's buffer already holds its values.
        """
        self._refresh_inputs(args, kwargs, state)
        if not self._replay(input_key, state):
            return self.inner_module(*args, **kwargs)
        output = self._caller_output(state)
        state.mark_used()
        return output

    def _caller_output(self, state: CUDAGraphState) -> Tensor | tuple[Tensor, ...]:
        """Clone the static output for the caller, except a state kwarg's static buffer the module returned.

        That buffer goes back as itself: the caller hands it to the next call, which then copies nothing in.
        """
        return _clone_tensors(state.static_output, self._state_buffers(state))

    def _first_call_output(self, state: CUDAGraphState, output: Any) -> Any:
        """The first call's warm-up result, with its state outputs moved into their static buffers.

        A replay would have left them there, so the caller gets those buffers back, as from every replay.
        """
        keep = self._state_buffers(state)
        return _copy_kept_leaves_into(state.static_output, output, keep) if keep else output

    def _state_buffers(self, state: CUDAGraphState) -> tuple[Tensor, ...]:
        """The static buffers of the state kwargs, which outputs hand back uncloned."""
        state_kwargs = self._state_input_kwargs()
        if not state_kwargs:
            return ()
        static_kwargs = self._graph_input_kwargs(state.static_input_kwargs)
        return tuple(static_kwargs[name] for name in state_kwargs if isinstance(static_kwargs.get(name), Tensor))

    def _refresh_inputs(self, args: tuple, kwargs: dict, state: CUDAGraphState) -> None:
        """Restore dynamic inputs before replay, including a capturing call."""
        _copy_tensors_into(dest=state.static_input_arg, src=args)
        static_kwargs = self._graph_input_kwargs(state.static_input_kwargs)
        live_kwargs = self._graph_input_kwargs(kwargs)
        assert static_kwargs.keys() == live_kwargs.keys(), (
            "structure mismatch: static buffer and live input have different keys"
        )
        stable = self._stable_input_kwargs()
        for name, value in live_kwargs.items():
            # A state kwarg's static buffer, passed back, already holds this call's input.
            if value is static_kwargs[name]:
                continue
            if name in stable and _unchanged_since_copy(state.stable_sources.get(name), value):
                continue
            _copy_tensors_into(dest=static_kwargs[name], src=value)
            if name in stable:
                state.stable_sources[name] = _leaf_sources(value)

    def _state_input_kwargs(self) -> tuple[str, ...]:
        """Return the kwargs whose static buffers the module updates in place and returns uncloned."""
        routing = self.graph_optimization_config.input_routing_config
        return tuple(routing.state_input_kwargs) if routing is not None else ()

    def _borrowed_tensors(self, kwargs: dict) -> tuple[Tensor, ...]:
        """The borrowed kwargs' tensors that become their own static buffers at capture.

        A tensor the call also passes under another name stays out: one buffer cannot be two inputs, so each
        name gets its own copy.
        """
        routing = self.graph_optimization_config.input_routing_config
        names = routing.borrowed_input_kwargs if routing is not None else ()
        leaves = [leaf for value in kwargs.values() for leaf in _tensor_leaves(value)]
        borrowed = (kwargs[name] for name in names if isinstance(kwargs.get(name), Tensor))
        return tuple(tensor for tensor in borrowed if sum(leaf is tensor for leaf in leaves) == 1)

    def _evict_key(self, key: str) -> None:
        """Free and remove a key's cached graph state.

        Explicit LRU deletion does not invoke the overflow callback, so the state
        is released directly. Tolerates an already-evicted key so that overlapping
        failure paths cannot turn one fault into a ``KeyError``.
        """
        state = self.graph_state_by_key.get(key)
        if state is None:
            return
        state.release()
        del self.graph_state_by_key[key]
        self.execution_counts["eviction"] += 1

    def _evict_all_keys(self) -> int:
        """Release cached storage without resetting permanent eager decisions."""
        keys = list(self.graph_state_by_key.keys())
        for key in keys:
            self._evict_key(key)
        return len(keys)

    def _revert_to_eager(self, key: str) -> None:
        """Permanently mark a key eager and free its graph state.

        The tracker-level flag survives state eviction.
        """
        self.fallback_to_eager_by_key[key] = True
        self.execution_counts["fallback"] += 1
        self._evict_key(key)

    def _replay(self, input_key: str, state: CUDAGraphState) -> bool:
        try:
            state.graph.replay()
        except Exception as exc:  # noqa: BLE001 - any replay failure -> evict
            # Same reasoning as capture: a poisoned context cannot be recovered by
            # evicting the graph and retrying eagerly.
            if is_device_fatal(exc):
                raise
            logger.info(f"{type(self).__name__}: replay failed, evicting cached graph for key {input_key!r} ({exc})")
            self._revert_to_eager(input_key)
            return False

        return True

    def _f_capture(
        self,
        args: tuple,
        kwargs: dict,
        input_tensor_shapes: TensorContainerShapes | None = None,
    ) -> Tensor | tuple[Tensor]:
        """Run the wrapped module for warmup, capture, or verification.

        Bucketed inputs and outputs remain padded until ``forward`` unpads them.
        ``input_tensor_shapes`` carries live device dimensions for specialized trackers.
        """
        with (
            eager_graphs(),
            torch.amp.autocast(
                "cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                dtype=torch.get_autocast_dtype("cuda"),
                cache_enabled=False,
            ),
        ):
            return self.inner_module(*args, **kwargs)

    def post_load_weights(self) -> None:
        """Invalidate graphs whose captured parameter storage has been reloaded."""
        self.reset()

    def _apply(self, fn: Callable, recurse: bool = True) -> nn.Module:
        self.forget_captures()
        return super()._apply(fn, recurse=recurse)
