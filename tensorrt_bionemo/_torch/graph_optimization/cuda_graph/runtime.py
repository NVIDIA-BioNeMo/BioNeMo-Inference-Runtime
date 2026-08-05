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
"""Per-input-key CUDA-graph warmup/capture/replay tracker.

A module is graph-optimized by wrapping it in a :class:`CUDAGraphOptimizationTracker`
(an ``nn.Module`` that holds the eager module as its ``inner_module``); the
wrapper is installed by ``OptimizedModuleSetterMixin.optimize`` for modules whose
config requests the TORCH CUDA-graph backend. On each call the tracker derives an
*input key* from the input tensor shapes and drives a small per-key state machine
(:class:`CUDAGraphPreparationState`): a few warmup calls on a side stream, then a
one-time capture of a ``torch.cuda.CUDAGraph``, an optional post-capture
verification against eager (when ``verify_capture`` is set), then cheap replays
for every subsequent call with that key. Per-key bookkeeping lives in
:class:`CUDAGraphState`; the keys are held in an LRU cache so memory is bounded.
Any of memory-gate refusal, capture failure, capture-verification failure, or
replay failure reverts that key permanently to eager.

The shared input-key derivation and shape-bucket padding live on the
:class:`GraphOptimizationTracker` base in
:mod:`tensorrt_bionemo._torch.graph_optimization.tracker`.
"""

import enum
import gc
from typing import Any

import torch
import torch.nn as nn
from lru import LRU
from torch import Tensor
from torch.cuda.streams import Stream

from tensorrt_bionemo._torch.graph_optimization.config import CUDAGraphOptimizationConfig, InputKeyMethod
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.memory import (
    check_capacity_for_capture,
    container_device,
    tensor_bytes,
)
from tensorrt_bionemo._torch.graph_optimization.tensor_copy_utils import (
    _assert_equal_but_distinct,
    _clone_tensors,
    _copy_tensors_into,
    _delete_tensors_in_container,
)
from tensorrt_bionemo._torch.graph_optimization.tracker import GraphOptimizationTracker, TensorContainerShapes
from tensorrt_bionemo.logger import logger


class CUDAGraphPreparationState(enum.Enum):
    """State-machine position for a single input key, in capture order.

    A key advances WARMUP -> WARMUP_KERNELS_COMPILED -> WARMUP_MEMORY_ALLOCATOR_READY
    as its call count crosses the config thresholds, then GRAPH_CAPTURED once its
    graph has been captured, then GRAPH_VERIFIED (its terminal steady state).
    Calls in either captured state are served by replay.

    Members:
        WARMUP: Initial state; module runs eagerly on a side stream while kernels
            autotune/compile.
        WARMUP_KERNELS_COMPILED: Kernels compiled; continue warming so the caching
            allocator's pools stabilize.
        WARMUP_MEMORY_ALLOCATOR_READY: Allocator primed; the next call captures the
            graph.
        GRAPH_CAPTURED: Graph just captured; the capture call optionally verifies
            it against eager (when ``verify_capture`` is set) before advancing.
        GRAPH_VERIFIED: Terminal steady state after capture (and optional
            verification); every call copies inputs into the static buffers and
            replays the graph.
    """

    WARMUP = 0
    WARMUP_KERNELS_COMPILED = 1
    WARMUP_MEMORY_ALLOCATOR_READY = 2
    GRAPH_CAPTURED = 3
    GRAPH_VERIFIED = 4


class CUDAGraphState:
    """Per-input-key state for the warmup/capture/replay lifecycle.

    One instance exists per distinct input key. It holds the state-machine
    position, the call counter that drives transitions, the side stream used for
    warmup/capture, the static input/output buffers (fixed-address clones the
    captured graph reads from and writes to), and the captured graph itself.
    """

    def __init__(self) -> None:
        self.preparation_state: CUDAGraphPreparationState = CUDAGraphPreparationState.WARMUP
        self.num_prev_calls_by_input_key: int = 0
        self.warmup_stream: Stream | None = None
        self.static_input_arg: tuple[Tensor, ...] | None = None
        self.static_input_kwargs: dict[str, Tensor] | None = None
        self.static_output: Tensor | tuple[Tensor, ...] | None = None
        self.static_unadjusted_input_tensor_shapes: dict[str, Tensor] | None = None
        self.static_unadjusted_output_tensor_shapes: dict[str, Tensor] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        # Bytes of the static input/output buffers pinned for this key's graph;
        # populated at capture time and used by the memory gate.
        self.working_set_bytes: int = 0
        # Peak *additional* GPU bytes the eager forward allocates during warmup
        # (its intermediate-activation working set), measured via the caching
        # allocator. The capture holds this resident in the graph's mempool, so
        # the memory gate adds it to ``working_set_bytes``. 0 until measured.
        self.warmup_peak_activation_bytes: int = 0

    def __del__(self) -> None:
        """Release the per-key buffers and graph when this state is dropped.

        Called on LRU eviction (via ``cudagraph_delete_callback``) or normal GC.
        The static input/output containers are freed with
        ``_delete_tensors_in_container`` so their tensors can be reclaimed.

        Guarded so it can never raise: ``__del__`` may run during interpreter
        shutdown, when module globals (e.g. ``torch`` inside
        ``_delete_tensors_in_container``) are already torn down — there is
        nothing left to reclaim then, so any such error is ignored. (A raising
        ``__del__`` only prints a noisy "Exception ignored in" traceback.) The
        ``getattr`` defaults also make this safe if ``__init__`` raised before
        the buffers were assigned.
        """
        try:
            _delete_tensors_in_container(getattr(self, "static_input_arg", None))
            _delete_tensors_in_container(getattr(self, "static_input_kwargs", None))
            _delete_tensors_in_container(getattr(self, "static_output", None))
            _delete_tensors_in_container(getattr(self, "static_unadjusted_input_tensor_shapes", None))
            _delete_tensors_in_container(getattr(self, "static_unadjusted_output_tensor_shapes", None))
        except Exception:
            pass


def cudagraph_delete_callback(_input_key: str, value: CUDAGraphState) -> None:
    """LRU eviction callback: drop the evicted :class:`CUDAGraphState`.

    The ``(key, value)`` signature is dictated by ``lru.LRU``'s callback API; the
    key is unused. Deleting the last reference triggers ``CUDAGraphState.__del__``,
    which frees that key's static buffers and captured graph.
    """
    del value


class CUDAGraphOptimizationTracker(GraphOptimizationTracker):
    """Wraps a module to drive per-input-key CUDA-graph warmup/capture/replay.

    For each input key (derived from input shapes) this tracks:
        - the number of prior calls seen with that key
        - the warmup/capture state-machine position for that key
        - the static input/output buffers and captured graph for that key
    """

    def __init__(self, config: CUDAGraphOptimizationConfig, inner_module: nn.Module | None = None) -> None:
        """Validate the config and create the LRU cache of per-key graph state.

        Args:
            config: Must be a :class:`CUDAGraphOptimizationConfig`.
            inner_module: The eager module to graph-optimize / fall back to.

        Raises:
            ValueError: If ``config`` is not a ``CUDAGraphOptimizationConfig``.
        """
        super().__init__(config, inner_module=inner_module)
        if not isinstance(self.graph_optimization_config, CUDAGraphOptimizationConfig):
            raise ValueError(f"Expected CUDAGraphOptimizationConfig, got {type(self.graph_optimization_config)}")

        self.graph_state_by_key = LRU(
            size=self.graph_optimization_config.num_graphs_max_for_this_module, callback=cudagraph_delete_callback
        )

        # Per-key permanent-eager flags, keyed by input key. Kept on the tracker
        # (not on the per-key CUDAGraphState) so a key stays eager across calls
        # even after its state is evicted to free the captured graph and static
        # buffers. Checked in ``forward`` before any state is (re)created; set by
        # ``_revert_to_eager`` on memory-gate refusal, capture failure, capture-
        # verification failure, or replay failure.
        self.fallback_to_eager_by_key: dict[str, bool] = {}

    def __del__(self) -> None:
        """Drop every cached :class:`CUDAGraphState` (and thus its buffers/graph).

        Guarded so it never raises during GC / interpreter shutdown (``__init__``
        may have raised before ``graph_state_by_key`` was set, and torn-down
        globals can make teardown error). Dropping the LRU releases the states,
        whose own guarded ``__del__`` reclaims the buffers.
        """
        try:
            self.graph_state_by_key.clear()
            del self.graph_state_by_key
        except Exception:
            pass

    def reset(self) -> None:
        """Clear all per-key state and release captured-graph resources.

        Empties the graph-state cache — dropping every :class:`CUDAGraphState`,
        whose guarded ``__del__`` frees that key's captured graph and static
        buffers — and clears the permanent-eager flags, so every key re-warms
        from scratch on its next call. The tracker stays usable afterwards
        (unlike :meth:`__del__`, which also drops the cache object).

        Overrides the :class:`~tensorrt_bionemo.runtime.backend.BackendBase`
        ``reset`` hook. Idempotent and safe to call at any time — including before
        ``__init__`` finished or after a previous ``reset`` — since missing/empty
        state is treated as already-clear. Forces a GC pass (a captured state can
        be reachable only through ``nn.Module`` reference cycles) and, on CUDA, a
        synchronize + ``empty_cache`` so the freed graph mempool is returned to
        the allocator deterministically.
        """
        graph_state_by_key = getattr(self, "graph_state_by_key", None)
        if graph_state_by_key is not None:
            graph_state_by_key.clear()
        fallback_to_eager_by_key = getattr(self, "fallback_to_eager_by_key", None)
        if fallback_to_eager_by_key is not None:
            fallback_to_eager_by_key.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def update_graph_state_by_key(self, *args, **kwargs) -> tuple[str, int]:
        """Look up (or create) this call's state and advance its state machine.

        Computes the input key. A key already recorded in
        ``fallback_to_eager_by_key`` (permanently reverted to eager, its state
        evicted) is returned immediately without recreating or advancing any
        state. Otherwise creates a fresh :class:`CUDAGraphState` on first sight
        (else increments its call counter), then advances the preparation state
        past the warmup thresholds in the config.

        Returns:
            ``(key, num_prev_calls_by_input_key)`` for the resolved key (the
            count is ``0`` for an eager-reverted key, which has no live state).
        """
        key = self.input_key_for_this_call(*args, **kwargs)
        # A key that permanently reverted to eager was evicted; do not recreate
        # or advance its state — ``forward`` serves it eagerly.
        if self.fallback_to_eager_by_key.get(key, False):
            return key, 0
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

        return key, state.num_prev_calls_by_input_key

    # ------------------------------------------------------------------
    # The tracker is a standalone nn.Module that wraps ``inner_module``.
    # ``forward`` drives the warmup -> capture -> verify -> replay state machine,
    # calling the wrapped ``inner_module`` for the real computation and reverting
    # permanently to eager for a key on any of: autograd/training, memory-gate
    # refusal, capture failure, capture-verification failure, or replay failure.
    # ------------------------------------------------------------------
    def forward_udf(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        """Eager execution of the wrapped module (the fallback target)."""
        return self.inner_module(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        """Graph-compile per input key, reverting to eager when unsafe.

        Args:
            args (tuple):  Positional args.  If positional args, then (,).
            kwargs (dict): Keyword args.  If keyword args, then {}.

        Returns:
            Tensor | tuple[Tensor, ...]: The module output, shape-matching the
                eager result (bucketed inputs are padded for capture and the
                output is truncated back to the live shape).
        """
        # Early eager exit under autograd or training
        #   - run eager and do not create/advance any per-key state.
        if torch.is_grad_enabled():
            return self.inner_module(*args, **kwargs)

        # Early eager exit if input does not satisfy input-acceptance-rule
        input_tensor_shapes: TensorContainerShapes = self._extract_tensor_container_shapes(args, kwargs)
        # Host-side copy for the control flow that reads dim lengths as Python
        # ints (input acceptance here, output un-padding below): those
        # ``int(shape[...])`` reads would each force a D2H sync on the CUDA
        # ``input_tensor_shapes``. The CUDA copy is kept for the capture/replay
        # path, where the captured graph reads the live shape on-device.
        input_tensor_shapes_cpu: TensorContainerShapes = GraphOptimizationTracker._tensor_container_shapes_to_cpu(
            input_tensor_shapes
        )
        # The first representative call validates that every configured
        # input tie resolves to a real tensor axis (raises otherwise). One-shot.
        self.validate_input_ties(input_tensor_shapes_cpu)
        if not self.input_accepted(input_tensor_shapes_cpu):
            return self.inner_module(*args, **kwargs)

        # -----------------------------------------------------------
        # adjust input to target tensor shapes
        # -----------------------------------------------------------
        if self.graph_optimization_config.input_key_method == InputKeyMethod.EXACT:
            adjusted_args = args  # assign same address
            adjusted_kwargs = kwargs  # assign same address

        elif self.graph_optimization_config.input_key_method == InputKeyMethod.BUCKETED_SHAPES:
            adjusted_args, adjusted_kwargs = self.pad_input(args, kwargs, input_tensor_shapes=input_tensor_shapes_cpu)

        # -----------------------------------------------------------
        # compute key with adjusted input
        # -----------------------------------------------------------
        input_key, _ = self.update_graph_state_by_key(*adjusted_args, **adjusted_kwargs)
        # Permanent eager: a key that failed the memory gate, capture,
        # verification, or replay on an earlier call was evicted and recorded
        # here. Run eager without touching per-key state
        # (update_graph_state_by_key does not recreate it for such keys), so
        # the key never re-warms (see _revert_to_eager).
        if self.fallback_to_eager_by_key.get(input_key, False):
            return self.inner_module(*args, **kwargs)
        state = self.graph_state_by_key[input_key]

        # -----------------------------------------------------------
        # select call methods
        # ----------------------------------------------------------
        ps = state.preparation_state
        if ps in (CUDAGraphPreparationState.WARMUP, CUDAGraphPreparationState.WARMUP_KERNELS_COMPILED):
            adjusted_output = self._warmup_call(  # adjusted_output: Tensor | tuple[Tensor]
                adjusted_args, adjusted_kwargs, input_key, input_tensor_shapes
            )

        elif ps == CUDAGraphPreparationState.WARMUP_MEMORY_ALLOCATOR_READY:
            adjusted_output = self._capture_call(adjusted_args, adjusted_kwargs, input_key, input_tensor_shapes)

        elif ps in (CUDAGraphPreparationState.GRAPH_CAPTURED, CUDAGraphPreparationState.GRAPH_VERIFIED):
            adjusted_output = self._replay_call(adjusted_args, adjusted_kwargs, input_key, input_tensor_shapes)

        else:
            raise ValueError(f"Invalid state {ps} for key {input_key}")

        # ------------------------------------------------
        # adjust output to orig tensor shapes
        # -----------------------------------------------
        if self.graph_optimization_config.input_key_method == InputKeyMethod.EXACT:
            output = adjusted_output  # assign same address
        elif self.graph_optimization_config.input_key_method == InputKeyMethod.BUCKETED_SHAPES:
            # Splat a tuple output into unpad_output's positional args; pass a
            # single-tensor output as one arg. Splatting a bare tensor would
            # iterate it over dim 0, silently dropping its leading
            # (batch/samples) axis.
            if isinstance(adjusted_output, tuple):
                output = self.unpad_output(adjusted_output, input_tensor_shapes_cpu)
            else:
                output = self.unpad_output((adjusted_output,), input_tensor_shapes_cpu)

        return output

    def _warmup_call(
        self, args: Tensor | tuple | Any, kwargs: dict, input_key: str, input_tensor_shapes: TensorContainerShapes
    ) -> Tensor | tuple[Tensor, ...]:
        """Eager run on a side stream; clone the fixed-address static output
        buffer once (the static input buffers are cloned later at capture time).

            args: adjusted or not adjusted based on caller
        """
        state = self.graph_state_by_key[input_key]

        if state.warmup_stream is None:
            state.warmup_stream = torch.cuda.Stream()

        # Measure the peak *additional* GPU allocation across the eager forward —
        # the intermediate-activation working set the capture will later hold
        # resident in its mempool, which the input/output buffer bytes alone do
        # not capture. Kept as the max over warmup calls (conservative; the first
        # call may also include one-off kernel-autotuning workspace). Device is
        # taken from the inputs so the stat is read on the right GPU.
        device = container_device((args, kwargs))
        if device is not None:
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(state.warmup_stream):
            adjusted_output = self._f_capture(args, kwargs, input_tensor_shapes=input_tensor_shapes)

        torch.cuda.current_stream().wait_stream(state.warmup_stream)

        if device is not None:
            peak_activation = torch.cuda.max_memory_allocated(device) - allocated_before
            state.warmup_peak_activation_bytes = max(state.warmup_peak_activation_bytes, peak_activation)

        # Create and populate static output buffers once
        #   - state.static_output can be a Tensor or a tuple[Tuple]
        if state.static_output is None:
            state.static_output = _clone_tensors(adjusted_output)

        return adjusted_output

    def _capture_call(
        self, args: tuple, kwargs: dict, input_key: str, input_tensor_shapes: TensorContainerShapes
    ) -> Tensor | tuple[Tensor, ...]:
        """Gate on memory, capture the graph, then replay once for this call.

        Copy live inputs into the static buffers.

        Benchmarks show that it is an order of magnitude faster to copy the
        live inputs into the static buffers, than to check if arg has different
        values than static_input_arg and static_input_kwargs, and only copy if
        they differ. The latter is a deep recursive check that is expensive
        for large nested structures.


        At this call
            (1) kernels are compiled
            (2) the caching allocator is primed
            (3) the static output buffer has been created and populated during
                warmup (the static input buffers are created here, below)

        Any refusal/failure reverts this key permanently to eager.

        Arguments:
            args: adjusted or not adjusted based on caller

        """
        # Finish warmup work before capturing. Make the warmup stream wait on the
        # current stream's queued work (which produced this call's inputs), then
        # block the host on the warmup stream only. That transitively waits on the
        # current stream too (via the wait_stream dependency), so it gives the
        # same ordering guarantee as a device-wide ``torch.cuda.synchronize()``
        # without stalling unrelated streams / other GPUs before capture.
        state = self.graph_state_by_key[input_key]
        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        state.warmup_stream.synchronize()

        # Create the fixed-address static input buffers once (clones of this
        # key's adjusted/padded inputs). The captured graph reads from these,
        # and every subsequent replay copies the live inputs into them (see
        # _replay_call). Without this the capture runs on ``None`` inputs and
        # produces an empty graph, permanently reverting the key to eager.
        if state.static_input_arg is None:
            state.static_input_arg = _clone_tensors(args)
        if state.static_input_kwargs is None:
            state.static_input_kwargs = _clone_tensors(kwargs)
        if state.static_unadjusted_input_tensor_shapes is None:
            state.static_unadjusted_input_tensor_shapes = _clone_tensors(input_tensor_shapes)

        # Check that the static input/output buffers fit within the memory gate
        state.working_set_bytes = (
            tensor_bytes(state.static_input_arg)
            + tensor_bytes(state.static_input_kwargs)
            + tensor_bytes(state.static_unadjusted_input_tensor_shapes)
            + tensor_bytes(state.static_output)
        )
        # Gate on the static-buffer working set plus the activation working set
        # measured during warmup (0 if unmeasured, e.g. no CUDA input), so the
        # check reflects the intermediate activations the capture holds resident,
        # not just the input/output buffers.
        check = check_capacity_for_capture(
            working_set_bytes=(state.working_set_bytes + state.warmup_peak_activation_bytes),
            input_container=(state.static_input_arg, state.static_input_kwargs),
        )
        if not check.ok:
            logger.info(f"{type(self).__name__}: skipping capture, revert to eager ({check.reason})")
            self._revert_to_eager(input_key)
            return self.inner_module(*args, **kwargs)

        # --------------------------------------------------------------------
        # capture the graph on the warmup stream
        #   - so the caching allocator's stream-aware bookkeeping stays consistent.
        # --------------------------------------------------------------------
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=state.warmup_stream):
                state.static_output = self._f_capture(
                    state.static_input_arg,
                    state.static_input_kwargs,
                    input_tensor_shapes=state.static_unadjusted_input_tensor_shapes,
                )

        except Exception as exc:  # noqa: BLE001 - any capture failure -> eager
            logger.info(f"{type(self).__name__}: capture failed, revert to eager ({exc})")
            self._revert_to_eager(input_key)
            return self.inner_module(*args, **kwargs)

        state.graph = graph
        state.preparation_state = CUDAGraphPreparationState.GRAPH_CAPTURED

        if self.config.verify_capture:
            if not self._verify_capture(args, kwargs, input_key, input_tensor_shapes):
                return self.inner_module(*args, **kwargs)
        state.preparation_state = CUDAGraphPreparationState.GRAPH_VERIFIED

        # --------------------------------------------------------------------
        # replay the graph on the real stream
        # --------------------------------------------------------------------
        replay_ok: bool = self._replay(input_key, state)
        if not replay_ok:
            return self.inner_module(*args, **kwargs)

        # -----------------------------------------------------------
        # copy result from static buffers
        # -----------------------------------------------------------
        output = _clone_tensors(state.static_output)

        return output

    def _verify_capture(
        self, args: tuple, kwargs: dict, input_key: str, input_tensor_shapes: TensorContainerShapes
    ) -> bool:
        """Check that the captured graph produced the same output as eager.

        On mismatch, catches the ``AssertionError``, reverts this key
        permanently to eager, and drops the captured graph.

        Returns:
            ``True`` if the replay output matches eager, ``False`` otherwise.
        """
        state = self.graph_state_by_key[input_key]
        state.graph.replay()  # same data as used for capture

        eager_out = self._f_capture(args, kwargs, input_tensor_shapes=input_tensor_shapes)
        replay_out = _clone_tensors(state.static_output)
        try:
            _assert_equal_but_distinct(eager_out, replay_out)
            return True
        except AssertionError as exc:
            logger.info(f"{type(self).__name__}: capture verification failed, revert to eager ({exc})")
            self._revert_to_eager(input_key)
            return False

    def _replay_call(
        self, args: Tensor | tuple, kwargs: dict, input_key: str, input_tensor_shapes: TensorContainerShapes = None
    ) -> Tensor | tuple[Tensor, ...]:

        # Copy only the *real* inputs; graph-internal scratch kwargs (e.g.
        # ``buffers``) are left to the captured graph (see
        # GRAPH_INTERNAL_WORKSPACE_KWARGS). Guard the copy: if the live inputs
        # no longer line up with the captured static buffers (an un-keyed
        # shape change, or a scratch container the key didn't capture),
        # revert this key permanently to eager instead of crashing here.

        state = self.graph_state_by_key[input_key]

        _copy_tensors_into(dest=state.static_input_arg, src=args)
        _copy_tensors_into(
            dest=self._graph_input_kwargs(state.static_input_kwargs), src=self._graph_input_kwargs(kwargs)
        )
        _copy_tensors_into(dest=state.static_unadjusted_input_tensor_shapes, src=input_tensor_shapes)

        replay_ok: bool = self._replay(input_key, state)
        if not replay_ok:
            return self.inner_module(*args, **kwargs)

        # -----------------------------------------------------------
        # copy result from static buffers
        # -----------------------------------------------------------
        adjusted_output = _clone_tensors(state.static_output)
        return adjusted_output

    def _evict_key(self, key: str) -> None:
        """Drop ``key``'s cached graph state: free its captured graph and remove
        the entry from ``graph_state_by_key``.

        Deletes the captured graph immediately, then removes the ``key -> state``
        mapping. That drops the cache's (long-lived) reference to the
        ``CUDAGraphState``; once the in-flight call-stack references unwind, the
        state is reclaimed and its ``__del__`` frees the static buffers. Note
        ``lru.LRU`` fires ``cudagraph_delete_callback`` only on size-overflow
        eviction, not on explicit removal — so this frees the graph itself rather
        than relying on that callback.
        """
        state = self.graph_state_by_key[key]
        if state.graph is not None:
            del state.graph
        del self.graph_state_by_key[key]

    def _revert_to_eager(self, key: str) -> None:
        """Permanently revert ``key`` to eager and free its cached graph state.

        Records ``key`` in ``fallback_to_eager_by_key`` so every future call for
        it runs the eager module (``forward`` checks this before (re)creating any
        state, so the key never re-warms), then evicts its ``CUDAGraphState`` via
        :meth:`_evict_key` to release the captured graph and static buffers. The
        flag lives on the tracker rather than the per-key state so it survives
        that eviction — a permanent eager fallback that still frees GPU memory.
        """
        self.fallback_to_eager_by_key[key] = True
        self._evict_key(key)

    def _replay(self, input_key: str, state: CUDAGraphState) -> bool:
        try:
            state.graph.replay()
        except Exception as exc:  # noqa: BLE001 - any replay failure -> evict
            logger.info(f"{type(self).__name__}: replay failed, evicting cached graph for key {input_key!r} ({exc})")
            self._revert_to_eager(input_key)
            return False

        return True

    def _f_capture(
        self,
        args: tuple,
        kwargs: dict,
        input_tensor_shapes: TensorContainerShapes = None,
    ) -> Tensor | tuple[Tensor]:
        """Run the wrapped module for warmup / capture / verification.

        The padding of ``args`` / ``kwargs`` (and hence of the returned output)
        depends on the configured ``input_key_method``:

        * ``InputKeyMethod.EXACT``: inputs are the live tensors and the output is
          at the live shape — nothing is padded.
        * ``InputKeyMethod.BUCKETED_SHAPES``: inputs have already been padded to
          their bucket shapes (by ``pad_input`` in ``forward``), so the returned
          output is likewise at the padded (bucket) shape — the caller truncates
          it back to the live shape via ``unpad_output``.

        ``input_tensor_shapes`` carries the *unpadded* (live) per-tensor dim
        lengths, available here so a module that needs the true live shapes can be
        forwarded them; the base implementation does not need them and passes only
        the (possibly padded) inputs.
        """
        return self.inner_module(*args, **kwargs)
