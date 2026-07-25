# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-input-key CUDA-graph warmup/capture/replay machinery.

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
"""
import abc
import enum
from typing import Any

import torch
import torch.nn as nn
from lru import LRU
from torch import Tensor
from torch.cuda.streams import Stream

from tensorrt_bionemo._torch.graph_optimization.config_schema import (
    BaseGraphOptimizationConfig, CUDAGraphOptimizationConfig, InputKeyMethod)
from tensorrt_bionemo._torch.graph_optimization.memory import (
    check_capacity_for_capture, tensor_bytes)
from tensorrt_bionemo._torch.graph_optimization.tensor_copy_utils import (
    _assert_equal_but_distinct, _clone_tensors, _copy_tensors_into,
    _delete_tensors_in_container)
from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.runtime.backend import BackendBase


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
        self.graph: torch.cuda.CUDAGraph | None = None
        # Set True once this key has permanently reverted to eager (memory-gate
        # refusal, capture failure, or replay failure); thereafter every call
        # for this key runs the eager module and never touches the graph.
        self.fallback_to_eager: bool = False
        # Bytes of the static input/output buffers pinned for this key's graph;
        # populated at capture time and used by the memory gate.
        self.working_set_bytes: int = 0
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
        except Exception:
            pass


class GraphOptimizationTracker(BackendBase):
    """Backend that wraps a module and keys graph state on a call's inputs.

    Inherits from :class:`tensorrt_bionemo.runtime.backend.BackendBase` (req 2.1)
    so the eager-fallback plumbing (``set_fallback_module`` / ``_fallback_module``)
    and the backend config surface are in place for memory management and the
    revert-to-eager strategy. Provides input-key derivation shared by concrete
    trackers: a key is the join of a shape-encoded subkey per input tensor,
    walked recursively through nested dicts/lists/tuples. The separators below
    build that string; concrete subclasses implement the state-management policy.

    Class attributes:
        SEP_FOR_ARGS: Separator joining the per-tensor subkeys into one key.
        SEP_BW_NAME_AND_SHAPE: Separator between a tensor's path and its shape.
        SEP_FOR_DIMS: Separator between the dims within a tensor's shape.
        GRAPH_INTERNAL_WORKSPACE_KWARGS: Names of keyword arguments that are
            graph-*internal scratch* rather than real inputs (see below).
    """
    SEP_FOR_ARGS = "|"
    SEP_BW_NAME_AND_SHAPE = "+"
    SEP_FOR_DIMS = "-"

    # Keyword arguments that carry a graph-internal *workspace*, not stable inputs.
    #
    # The Boltz-2 / OpenFold3 score model threads a single ``buffers``
    # (``PreallocatedBuffers``) dict through the atom encoder -> token
    # transformer -> atom decoder. ``ensure_buffer`` lazily (re)allocates entries
    # in it *as a side effect* of each forward, and the encoder and token
    # transformer reuse the same buffer names (e.g. ``pw_attn_output``) at
    # different shapes. So at the wrapped module's entry the dict holds the
    # caller's (encoder-shaped) scratch, but the captured forward mutates it to
    # the module's own (token-shaped) scratch.
    #
    # That makes ``buffers`` unusable under the "clone the kwargs at capture,
    # copy the live kwargs into the static buffers on every replay" contract:
    # the live dict never matches the captured one (different shapes, or
    # different keys), which previously crashed in ``_copy_tensors_into``.
    #
    # These kwargs are therefore excluded from the input key *and* from the
    # per-replay static-buffer copy: the captured graph allocates and reuses this
    # scratch internally (in the graph mempool), so there is nothing to key on or
    # copy in. They are still passed through verbatim during warmup/capture so
    # the wrapped module runs exactly as it does eagerly.
    GRAPH_INTERNAL_WORKSPACE_KWARGS: frozenset = frozenset({"buffers"})

    def _graph_input_kwargs(self, kwargs: dict) -> dict:
        """Return ``kwargs`` without the graph-internal workspace entries.

        Used for both input-key derivation and the per-replay copy so that
        side-effect-populated workspace dicts (see
        :attr:`GRAPH_INTERNAL_WORKSPACE_KWARGS`) are never treated as graph inputs.
        """
        if not self.GRAPH_INTERNAL_WORKSPACE_KWARGS:
            return kwargs
        return {
            k: v
            for k, v in kwargs.items()
            if k not in self.GRAPH_INTERNAL_WORKSPACE_KWARGS
        }

    def __init__(self,
                 graph_optimization_config: BaseGraphOptimizationConfig,
                 inner_module: nn.Module | None = None) -> None:
        """Store the optimization config and the wrapped (inner) module.

        Args:
            graph_optimization_config: The graph-optimization configuration. Also used as this
                backend's ``config`` (it is a ``BaseConfig`` subclass), so the
                wrapper exposes ``config.backend`` etc.
            inner_module: The eager module being graph-optimized. This is the
                module the tracker warms up, captures, and falls back to.
        """
        super().__init__(graph_optimization_config)
        self.inner_module = inner_module

    @property
    def graph_optimization_config(self) -> BaseGraphOptimizationConfig:
        return self._config
    
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped eager ``inner_module``.

        The wrapper replaces the eager module in the model tree, so callers that
        read the original module's attributes (e.g. ``dtype``, ``version``) must
        still find them. ``nn.Module.__getattr__`` is tried first (parameters /
        buffers / submodules, including ``inner_module`` itself); only genuinely
        missing names fall through to the wrapped module.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            inner = modules.get("inner_module") if modules else None
            if inner is not None:
                return getattr(inner, name)
            raise

    def input_key_for_this_call(self, *args, **kwargs) -> str:
        """Return the graph-cache key for this call's positional/keyword inputs.

        Joins a shape-encoded subkey for every input tensor (per
        ``input_key_method``). With ``InputKeyMethod.EXACT`` the key encodes each
        tensor's exact shape, so calls with identical shapes share one captured
        graph.

        Raises:
            ValueError: If the configured ``input_key_method`` is unsupported.
        """
        if self.graph_optimization_config.input_key_method == InputKeyMethod.EXACT:
            subkeys: list[str] = []
            self._collect_subkeys(args, "", subkeys)
            self._collect_subkeys(self._graph_input_kwargs(kwargs), "", subkeys)
            key = self.SEP_FOR_ARGS.join(subkeys)
        else:
            raise ValueError(f"Input key method {self.graph_optimization_config.input_key_method} not supported")
        return key
    
    def _collect_subkeys(self, value: Any, path: str, subkeys: list[str]) -> None:
        """Recursively walk ``value``, appending a shape-encoded subkey to
        ``subkeys`` for every tensor found.

        ``path`` is the dotted name of the current position (e.g.
        ``feature_dict.ref_pos`` or ``coords.0``) so tensors at different
        nesting locations produce distinct keys.
        """
        if isinstance(value, torch.Tensor):
            subkeys.append(
                self.SEP_BW_NAME_AND_SHAPE.join(
                    [path, self.SEP_FOR_DIMS.join([str(dim) for dim in value.shape])]
                )
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}.{k}" if path else str(k)
                self._collect_subkeys(v, child, subkeys)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{path}.{i}" if path else str(i)
                self._collect_subkeys(v, child, subkeys)
        # Non-tensor leaves (None, scalars, callables, ...) are ignored.
    
    @abc.abstractmethod
    def update_graph_state_by_key(self, *args, **kwargs) -> tuple[str, int]:
        """Advance and return this call's per-key state. Implemented by subclasses."""
        ...


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
        - the compilation mode used for that input key
        - the number of prior calls seen with that key
        - the warmup/capture state-machine position for that key
        - the static input/output buffers and captured graph for that key
    """
    def __init__(self,
                 config: CUDAGraphOptimizationConfig,
                 inner_module: nn.Module | None = None) -> None:
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
            size=self.graph_optimization_config.num_graphs_max_for_this_module,
            callback=cudagraph_delete_callback)

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

    def update_graph_state_by_key(self, *args, **kwargs) -> tuple[str, int]:
        """Look up (or create) this call's state and advance its state machine.

        Computes the input key, creates a fresh :class:`CUDAGraphState` on first
        sight (else increments its call counter), then advances the preparation
        state past the warmup thresholds in the config.

        Returns:
            ``(key, num_prev_calls_by_input_key)`` for the resolved key.
        """
        key = self.input_key_for_this_call(*args, **kwargs)
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
        """Graph-compile per input key, reverting to eager when unsafe."""
        # CUDA graphs capture an inference forward.  Under autograd, run eager.
        # Do not do eager for self.training=True.
        if torch.is_grad_enabled():
            return self.inner_module(*args, **kwargs)

        key, _ = self.update_graph_state_by_key(*args, **kwargs)
        state = self.graph_state_by_key[key]
        if state.fallback_to_eager:
            return self.inner_module(*args, **kwargs)

        ps = state.preparation_state
        if ps in (CUDAGraphPreparationState.WARMUP,
                  CUDAGraphPreparationState.WARMUP_KERNELS_COMPILED):
            return self._warmup_call(args, kwargs, state)
        elif ps == CUDAGraphPreparationState.WARMUP_MEMORY_ALLOCATOR_READY:
            return self._capture_call(args, kwargs, state)
        elif ps in (CUDAGraphPreparationState.GRAPH_CAPTURED,
                    CUDAGraphPreparationState.GRAPH_VERIFIED):
            return self._replay_call(args, kwargs, state)
        raise ValueError(f"Invalid state {ps} for key {key}")

    def _warmup_call(self, 
                     args: tuple, 
                     kwargs: dict, 
                     state: CUDAGraphState) -> Tensor | tuple[Tensor, ...]:
        """Eager run on a side stream; clone fixed-address static buffers once."""
        if state.warmup_stream is None:
            state.warmup_stream = torch.cuda.Stream()

        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(state.warmup_stream):
            output = self.inner_module(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(state.warmup_stream)

        # Create and populate static output buffers once
        if state.static_output is None:
            state.static_output = _clone_tensors(output)
        return output

    def _capture_call(self, 
                      args: tuple, 
                      kwargs: dict, 
                      state: CUDAGraphState) -> Tensor | tuple[Tensor, ...]:
        """Gate on memory, capture the graph, then replay once for this call.
        
        At this call 
            (1) kernels are compiled
            (2) the caching allocator is primed
            (3) the static input/output buffers have been created and populated

        Any refusal/failure reverts this key permanently to eager.
        """
        # Finish warmup work
        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        
        # Create and populate static input buffers once
        # - static_input must have correct shape, etc, before capture
        if state.static_input_arg is None:
            state.static_input_arg = _clone_tensors(args)
        if state.static_input_kwargs is None:
            state.static_input_kwargs = _clone_tensors(kwargs)
        
        # Check that the static input/output buffers fit within the memory gate    
        state.working_set_bytes = (tensor_bytes(state.static_input_arg)
                                   + tensor_bytes(state.static_input_kwargs)
                                   + tensor_bytes(state.static_output))
        check = check_capacity_for_capture(state.working_set_bytes)
        if not check.ok:
            logger.info(
                f"{type(self).__name__}: skipping capture, revert to eager "
                f"({check.reason})")
            state.fallback_to_eager = True
            return self.inner_module(*args, **kwargs)


        # --------------------------------------------------------------------
        # capture the graph on the warmup stream
        #   - so the caching allocator's stream-aware bookkeeping stays consistent.
        # --------------------------------------------------------------------
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=state.warmup_stream):
                state.static_output = self.inner_module(
                    *state.static_input_arg, **state.static_input_kwargs)
        except Exception as exc:  # noqa: BLE001 - any capture failure -> eager
            logger.info(
                f"{type(self).__name__}: capture failed, revert to eager ({exc})")
            state.fallback_to_eager = True
            del state.graph
            state.graph = None
            return self.inner_module(*args, **kwargs)

        state.graph = graph
        state.preparation_state = CUDAGraphPreparationState.GRAPH_CAPTURED

        if self.config.verify_capture:
            self._verify_capture(args, kwargs, state)
        state.preparation_state = CUDAGraphPreparationState.GRAPH_VERIFIED
            
        # --------------------------------------------------------------------
        # replay the graph on the real stream
        # --------------------------------------------------------------------
        return self._replay(args, kwargs, state, do_copy_input_tensors=False)

    def _verify_capture(self,
                        args: tuple,
                        kwargs: dict,
                        state: CUDAGraphState) -> Tensor | tuple[Tensor, ...] | None:
        """Check that the captured graph produced the same output as eager.
        Raises:
            AssertionError: If the outputs differ.
        """
        state.graph.replay() # same data as used for capture
        eager_out = self.inner_module(*args, **kwargs)
        replay_out = _clone_tensors(state.static_output)
        try:
            _assert_equal_but_distinct(eager_out, replay_out)
        except AssertionError as exc:
            logger.info(
                f"{type(self).__name__}: capture verification failed, revert to eager ({exc})")
            state.fallback_to_eager = True
            del state.graph
            state.graph = None
            return self.inner_module(*args, **kwargs)
        
    def _replay_call(self, args: tuple, kwargs: dict, state: CUDAGraphState) -> Tensor | tuple[Tensor, ...]:
        return self._replay(args, kwargs, state, do_copy_input_tensors=True)
    
    def _replay(self, 
                args: tuple, 
                kwargs: dict, 
                state: CUDAGraphState, 
                do_copy_input_tensors: bool = True) -> Tensor | tuple[Tensor, ...]:
        """Copy live inputs into the static buffers and replay the graph.
        
        Benchmarks show that it is an order of magnitude faster to copy the 
        live inputs into the static buffers, than to check if arg has different 
        values than static_input_arg and static_input_kwargs, and only copy if 
        they differ. The latter is a deep recursive check that is expensive 
        for large nested structures.
        """
        if do_copy_input_tensors:
            # Copy only the *real* inputs; graph-internal scratch kwargs (e.g.
            # ``buffers``) are left to the captured graph (see
            # GRAPH_INTERNAL_WORKSPACE_KWARGS). Guard the copy: if the live inputs
            # no longer line up with the captured static buffers (an un-keyed
            # shape change, or a scratch container the key didn't capture),
            # revert this key permanently to eager instead of crashing here.
            try:
                _copy_tensors_into(dest=state.static_input_arg, src=args)
                _copy_tensors_into(
                    dest=self._graph_input_kwargs(state.static_input_kwargs),
                    src=self._graph_input_kwargs(kwargs))
            except (RuntimeError, AssertionError) as exc:
                logger.info(
                    f"{type(self).__name__}: live inputs do not match the "
                    f"captured static buffers, revert to eager ({exc})")
                state.fallback_to_eager = True
                return self.inner_module(*args, **kwargs)
        try:
            state.graph.replay()
        except Exception as exc:  # noqa: BLE001 - any replay failure -> eager
            logger.info(
                f"{type(self).__name__}: replay failed, revert to eager ({exc})")
            state.fallback_to_eager = True
            del state.graph
            state.graph = None
            return self.inner_module(*args, **kwargs)
        return _clone_tensors(state.static_output)