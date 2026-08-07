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
from tensorrt_bionemo._torch.graph_optimization.tracker import (
    GraphOptimizationTracker,
    TensorContainerHostShapes,
    TensorContainerShapes,
)
from tensorrt_bionemo.logger import logger


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

    def __del__(self) -> None:
        """Release buffers without raising during partial init or shutdown."""
        try:
            _delete_tensors_in_container(getattr(self, "static_input_arg", None))
            _delete_tensors_in_container(getattr(self, "static_input_kwargs", None))
            _delete_tensors_in_container(getattr(self, "static_output", None))
            _delete_tensors_in_container(getattr(self, "static_unadjusted_input_tensor_shapes", None))
            _delete_tensors_in_container(getattr(self, "static_unadjusted_output_tensor_shapes", None))
        except Exception:
            pass


def cudagraph_delete_callback(_input_key: str, value: CUDAGraphState) -> None:
    """Drop an LRU-evicted graph state and its resources."""
    del value


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

        self.graph_state_by_key = LRU(
            size=self.graph_optimization_config.num_graphs_max_for_this_module, callback=cudagraph_delete_callback
        )

        # Tracker-level flags survive state eviction, keeping failed keys eager.
        self.fallback_to_eager_by_key: dict[str, bool] = {}

    def __del__(self) -> None:
        """Drop cached states without raising during shutdown."""
        try:
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
            graph_state_by_key.clear()
        fallback_to_eager_by_key = getattr(self, "fallback_to_eager_by_key", None)
        if fallback_to_eager_by_key is not None:
            fallback_to_eager_by_key.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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
        if torch.is_grad_enabled():
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
            input_tensor_shapes_device, input_tensor_shapes_host = self._extract_tensor_container_shape_maps(
                args, kwargs
            )
            self.validate_input_ties(input_tensor_shapes_host)
            if not self.input_accepted(input_tensor_shapes_host):
                return self.inner_module(*args, **kwargs)

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
            adjusted_output = self._warmup_call(adjusted_args, adjusted_kwargs, input_key)

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

    def _warmup_call(
        self,
        args: Tensor | tuple | Any,
        kwargs: dict,
        input_key: str,
    ) -> Tensor | tuple[Tensor, ...]:
        """Warm up on a side stream and initialize the static output."""
        state = self.graph_state_by_key[input_key]

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

        torch.cuda.current_stream().wait_stream(state.warmup_stream)

        if device is not None:
            peak_activation = torch.cuda.max_memory_allocated(device) - allocated_before
            state.warmup_peak_activation_bytes = max(state.warmup_peak_activation_bytes, peak_activation)

        if state.static_output is None:
            state.static_output = _clone_tensors(adjusted_output)

        return adjusted_output

    def _capture_call(self, args: tuple, kwargs: dict, input_key: str) -> Tensor | tuple[Tensor, ...]:
        """Check capacity, capture, optionally verify, and replay.

        Inputs are always copied; recursively checking for changes costs more.
        Any failure permanently reverts this key to eager.
        """
        # Synchronize only the warmup stream. Its dependency on the current
        # stream preserves input ordering without a device-wide barrier.
        state = self.graph_state_by_key[input_key]
        if state.warmup_stream is None:
            state.warmup_stream = torch.cuda.Stream()

        state.warmup_stream.wait_stream(torch.cuda.current_stream())
        state.warmup_stream.synchronize()

        # Capture requires fixed-address inputs; replay refreshes their values.
        if state.static_input_arg is None:
            state.static_input_arg = _clone_tensors(args)
        if state.static_input_kwargs is None:
            state.static_input_kwargs = _clone_tensors(kwargs)

        state.working_set_bytes = (
            tensor_bytes(state.static_input_arg)
            + tensor_bytes(state.static_input_kwargs)
            + tensor_bytes(state.static_unadjusted_input_tensor_shapes)
            + tensor_bytes(state.static_output)
        )
        # Include warmup activations retained by the graph mempool.
        check = check_capacity_for_capture(
            working_set_bytes=(state.working_set_bytes + state.warmup_peak_activation_bytes),
            input_container=(state.static_input_arg, state.static_input_kwargs),
        )
        if not check.ok:
            logger.info(f"{type(self).__name__}: skipping capture, revert to eager ({check.reason})")
            self._revert_to_eager(input_key)
            return self.inner_module(*args, **kwargs)

        # Capture on the warmup stream to preserve allocator stream ownership.
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
            if not self._verify_capture(args, kwargs, input_key):
                return self.inner_module(*args, **kwargs)
        state.preparation_state = CUDAGraphPreparationState.GRAPH_VERIFIED

        replay_ok: bool = self._replay(input_key, state)
        if not replay_ok:
            return self.inner_module(*args, **kwargs)

        output = _clone_tensors(state.static_output)

        return output

    def _verify_capture(self, args: tuple, kwargs: dict, input_key: str) -> bool:
        """Compare capture output with eager and revert on mismatch.

        Returns:
            Whether outputs match.
        """
        state = self.graph_state_by_key[input_key]
        state.graph.replay()

        eager_out = self._f_capture(
            args,
            kwargs,
            input_tensor_shapes=state.static_unadjusted_input_tensor_shapes,
        )
        replay_out = _clone_tensors(state.static_output)
        try:
            _assert_equal_but_distinct(eager_out, replay_out)
            return True
        except AssertionError as exc:
            logger.info(f"{type(self).__name__}: capture verification failed, revert to eager ({exc})")
            self._revert_to_eager(input_key)
            return False

    def _replay_call(self, args: Tensor | tuple, kwargs: dict, input_key: str) -> Tensor | tuple[Tensor, ...]:
        # Copy real inputs only; the captured graph owns workspace kwargs.
        state = self.graph_state_by_key[input_key]

        _copy_tensors_into(dest=state.static_input_arg, src=args)
        _copy_tensors_into(
            dest=self._graph_input_kwargs(state.static_input_kwargs), src=self._graph_input_kwargs(kwargs)
        )

        replay_ok: bool = self._replay(input_key, state)
        if not replay_ok:
            return self.inner_module(*args, **kwargs)

        adjusted_output = _clone_tensors(state.static_output)
        return adjusted_output

    def _evict_key(self, key: str) -> None:
        """Free and remove a key's cached graph state.

        Explicit LRU deletion does not invoke the overflow callback, so the
        graph is deleted directly.
        """
        state = self.graph_state_by_key[key]
        if state.graph is not None:
            del state.graph
        del self.graph_state_by_key[key]

    def _revert_to_eager(self, key: str) -> None:
        """Permanently mark a key eager and free its graph state.

        The tracker-level flag survives state eviction.
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
        input_tensor_shapes: TensorContainerShapes | None = None,
    ) -> Tensor | tuple[Tensor]:
        """Run the wrapped module for warmup, capture, or verification.

        Bucketed inputs and outputs remain padded until ``forward`` unpads them.
        ``input_tensor_shapes`` carries live device dimensions for specialized trackers.
        """
        return self.inner_module(*args, **kwargs)
