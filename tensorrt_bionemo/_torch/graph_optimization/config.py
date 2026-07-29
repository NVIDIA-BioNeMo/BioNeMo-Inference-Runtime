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
import enum
from typing import Dict, List, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from tensorrt_bionemo.configs.base import BaseConfig


class SpacingMethod(str, enum.Enum):
    """How the bucket boundary lengths for a padded dimension are spaced between
    ``dim_len_min`` and ``dim_len_max``.

    Members:
        LINEAR: Evenly spaced lengths (``np.linspace``); the default.
        EXPONENTIAL: Geometrically spaced lengths (``np.geomspace``), giving
            finer buckets at small lengths and coarser ones at large lengths.
    """
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


# padded_dims value:
#   (dim_len_min, dim_len_max, num_intervals, spacing_method, dim_len_values)
PaddedDimSpec = Tuple[int, int, int, SpacingMethod, Tuple[int, ...]]
# (input_tensor_name, input_dim_idx, dim_name)
InputDimAssignment = Tuple[str, int, str]
# (output_tensor_index, output_dim_idx, dim_name)
OutputDimAssignment = Tuple[int, int, str]


class InputRoutingConfig(BaseModel):
    """Serializable snapshot of the bucketing rules collected by a :class:`InputRoutingConfigFactory`.

    Attributes:
        padded_dims: Maps a named padded dimension to its bucketing spec
            ``(dim_len_min, dim_len_max, num_intervals, spacing_method,
            dim_len_values)``, where ``dim_len_values`` are the precomputed
            integer bucket-boundary lengths.
        input_dims_with_assigned_padded_dim: ``(input_tensor_name,
            input_dim_idx, dim_name)`` records tying an input tensor axis to a
            named padded dimension.
        output_dims_with_assigned_padded_dim: ``(output_tensor_index,
            output_dim_idx, dim_name)`` records tying an output tensor axis to a
            named padded dimension.
        input_acceptance_dims: Maps a named dimension to the inclusive
            upper-bound length accepted for it (an input axis tied to that
            dimension is in range when it does not exceed this length, and out
            of range only when strictly greater). Carried as plain data so the
            acceptance predicate can be rebuilt from a deserialized snapshot —
            no un-picklable closure ever crosses a boundary.
        input_dims_with_assigned_input_acceptance_dim: ``(input_tensor_name,
            input_dim_idx, dim_name)`` records tying an input tensor axis to a
            named acceptance dimension; the tracker reads these to enforce the
            ``input_acceptance_dims`` limit on each call.
    """

    padded_dims: Dict[str, PaddedDimSpec] = Field(default_factory=dict)
    input_dims_with_assigned_padded_dim: List[InputDimAssignment] = Field(
        default_factory=list)
    output_dims_with_assigned_padded_dim: List[OutputDimAssignment] = Field(
        default_factory=list)
    input_acceptance_dims: Dict[str, int] = Field(default_factory=dict)
    input_dims_with_assigned_input_acceptance_dim: List[InputDimAssignment] = Field(
        default_factory=list)

    class Config:
        extra = "allow"


class InputRoutingConfigFactory:
    """Collects shape-bucketing rules for a module's ``forward`` inputs/outputs.

    Bucketing is organized around **named** padded dimensions (5.3.0): a padded
    dimension is declared once with :meth:`set_padded_dim` (giving it a
    name and a set of bucket-boundary lengths), then attached to any number of
    input/output tensor axes with :meth:`input_dim_is_padded` /
    :meth:`output_dim_is_padded`. Axes sharing a name are padded to
    the same bucket length in lockstep.

    Call :meth:`export_config` to snapshot the collected rules into an
    immutable, serializable :class:`InputRoutingConfig`.
    """

    def __init__(self) -> None:
        # dim_name -> dim_len_max (inclusive upper bound accepted for this dim)
        self.input_acceptance_dims: Dict[str, int] = {}
        # (input_tensor_name, input_dim_idx) -> dim_name (acceptance-bounded axis)
        self.input_dims_with_assigned_input_acceptance_dim: Dict[Tuple[str, int], str] = {}

        # dim_name -> (min, max, num_intervals, spacing_method, dim_len_values)
        self.padded_dims: Dict[str, PaddedDimSpec] = {}

        # (input_tensor_name, input_dim_idx) -> dim_name
        self.input_dims_with_assigned_padded_dim: Dict[Tuple[str, int], str] = {}

        # (output_tensor_index, output_dim_idx) -> dim_name
        self.output_dims_with_assigned_padded_dim: Dict[Tuple[int, int], str] = {}


    def set_input_acceptance_dim(self, dim_name: str, dim_len_max: int) -> None:
        """Declare the largest input length accepted for a named dimension.

        Records into ``input_acceptance_dims`` that inputs whose ``dim_name``
        axis is strictly longer than ``dim_len_max`` fall outside this module's
        captured-graph coverage. Keyed by ``dim_name``, so a later call for the
        same name overrides the earlier limit.

        Args:
            dim_name: Name of the dimension the rule applies to.
            dim_len_max: Inclusive upper bound on the accepted length; an axis is
                accepted when it does not exceed this, rejected only when
                strictly greater.
        """
        self.input_acceptance_dims[dim_name] = dim_len_max

    def input_dim_is_acceptance(
        self,
        input_tensor_name: str,
        input_dim_idx: int,
        input_acceptance_dim: str,
    ) -> None:
        """Attach a named input-acceptance dimension to an input tensor axis.

        Records into ``input_dims_with_assigned_input_acceptance_dim`` that the
        ``input_dim_idx`` axis of the ``input_tensor_name`` input is bounded by
        the ``input_acceptance_dim`` acceptance rule: at call time that axis must
        not exceed ``input_acceptance_dims[input_acceptance_dim]`` for the call to
        be accepted.

        Args:
            input_tensor_name: Walk path of the input tensor: ``"arg{i}"`` for
                positional arg ``i`` (e.g. ``"arg0"``) or the keyword parameter
                name for a keyword arg.
            input_dim_idx: Axis of the input tensor. May be negative, counting
                from the end (``-1`` is the last dimension).
            input_acceptance_dim: Name of a dimension previously declared with
                :meth:`set_input_acceptance_dim`.

        Raises:
            ValueError: If ``input_acceptance_dim`` has no configured acceptance
                rule.
        """
        if input_acceptance_dim not in self.input_acceptance_dims:
            raise ValueError(
                f"input-acceptance dim {input_acceptance_dim!r} is not "
                "configured; call set_input_acceptance_dim before assigning it "
                f"(known: {sorted(self.input_acceptance_dims)})")
        self.input_dims_with_assigned_input_acceptance_dim[
            (input_tensor_name, input_dim_idx)] = input_acceptance_dim

    def set_padded_dim(
        self,
        dim_name: str,
        dim_len_min: int,
        dim_len_max: int,
        num_intervals: int,
        multiple_of: int = 128,
        spacing_method: Union[SpacingMethod, str] = SpacingMethod.LINEAR,
    ) -> None:
        """Declare a named padded dimension and its bucket-boundary lengths.

        Args:
            dim_name: Name identifying this padded dimension (the ``padded_dims``
                key). Assignments reference the dimension by this name.
            dim_len_min: Smallest bucketed length (inclusive).
            dim_len_max: Largest bucketed length (inclusive).
            num_intervals: Number of intervals dividing
                ``[dim_len_min, dim_len_max]``; yields ``num_intervals + 1``
                bucket-boundary lengths (both endpoints inclusive).
            multiple_of: Alignment granularity. Each computed boundary length is
                snapped up (via :meth:`snap_to`) to the lowest multiple of this
                value that is >= it, so padded shapes land on hardware-friendly
                tile sizes. Defaults to 128.
            spacing_method: A :class:`SpacingMethod` (or its string value) giving
                how the boundary lengths are spaced. Defaults to
                :attr:`SpacingMethod.LINEAR`.

        Raises:
            ValueError: If the length range is invalid, ``num_intervals`` is not
                positive, ``spacing_method`` is unrecognized, or an exponential
                spacing is requested with ``dim_len_min < 1``.
        """
        if dim_len_min < 0 or dim_len_max < dim_len_min:
            raise ValueError(
                "Require 0 <= dim_len_min <= dim_len_max, got "
                f"dim_len_min={dim_len_min}, dim_len_max={dim_len_max}")
        if num_intervals < 1:
            raise ValueError(
                f"num_intervals must be >= 1, got {num_intervals}")
        try:
            spacing_method = SpacingMethod(spacing_method)
        except ValueError:
            raise ValueError(
                f"spacing_method must be one of {[m.value for m in SpacingMethod]}, "
                f"got {spacing_method!r}")
        if spacing_method == SpacingMethod.EXPONENTIAL and dim_len_min < 1:
            raise ValueError(
                "exponential spacing requires dim_len_min >= 1, got "
                f"{dim_len_min}")

        dim_len_values = InputRoutingConfigFactory.compute_dim_len_values(
            dim_len_min, dim_len_max, num_intervals, spacing_method)
        dim_len_values = InputRoutingConfigFactory.snap_to(
            dim_len_values, multiple_of)
        self.padded_dims[dim_name] = (
            dim_len_min, dim_len_max, num_intervals, spacing_method,
            dim_len_values)

    @staticmethod
    def snap_to(
        dim_len_values: Tuple[int, ...],
        multiple_of: int,
    ) -> Tuple[int, ...]:
        """Round each length up to the lowest multiple of ``multiple_of``.

        Every entry of ``dim_len_values`` is increased (never decreased) to the
        smallest multiple of ``multiple_of`` that is equal to or greater than
        it; a value already a multiple of ``multiple_of`` is left unchanged.
        Snapping bucket-boundary lengths to a fixed granularity (e.g. 128)
        aligns padded shapes to hardware-friendly tile sizes.

        Args:
            dim_len_values: Bucket-boundary lengths to align, in order.
            multiple_of: Positive alignment granularity to snap up to.

        Returns:
            The aligned lengths, in the same order as ``dim_len_values``.

        Raises:
            ValueError: If ``multiple_of`` is not positive.
        """
        if multiple_of < 1:
            raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
        return tuple(
            -(-int(v) // multiple_of) * multiple_of for v in dim_len_values)

    @staticmethod
    def compute_dim_len_values(
        dim_len_min: int,
        dim_len_max: int,
        num_intervals: int,
        spacing_method: SpacingMethod,
    ) -> Tuple[int, ...]:
        """Return the integer bucket-boundary lengths for a padded dimension.

        ``num_intervals`` is the number of intervals dividing
        ``[dim_len_min, dim_len_max]``, so ``num_intervals + 1`` boundary lengths
        are returned (both endpoints inclusive), spaced linearly (``np.linspace``)
        or geometrically (``np.geomspace``). Values are rounded to ``int`` since
        tensor dimension lengths are integers.
        """
        if spacing_method == SpacingMethod.LINEAR:
            values = np.linspace(dim_len_min, dim_len_max, num=num_intervals+1)
        elif spacing_method == SpacingMethod.EXPONENTIAL:
            values = np.geomspace(dim_len_min, dim_len_max, num=num_intervals+1)
        else:
            raise ValueError(
                f"spacing_method must be one of {[m.value for m in SpacingMethod]}, "
                f"got {spacing_method!r}")
        return tuple(int(round(v)) for v in values)

    def input_dim_is_padded(
        self,
        input_tensor_name: str,
        input_dim_idx: int,
        padded_dim_name: str,
    ) -> None:
        """Attach a named padded dimension to an input tensor axis.

        Records into ``input_dims_with_assigned_padded_dim`` that the
        ``input_dim_idx`` axis of the ``input_tensor_name`` input is padded/
        bucketed according to the ``padded_dim_name`` padded dimension.

        Args:
            input_tensor_name: Walk path of the input tensor: ``"arg{i}"`` for
                positional arg ``i`` (e.g. ``"arg0"``) or the keyword parameter
                name for a keyword arg.
            input_dim_idx: Axis of the input tensor. May be negative, counting
                from the end (``-1`` is the last dimension).
            padded_dim_name: Name of a dimension previously declared with
                :meth:`set_padded_dim`.

        Raises:
            ValueError: If ``padded_dim_name`` was not configured.
        """
        self._check_dim_name(padded_dim_name)
        self.input_dims_with_assigned_padded_dim[
            (input_tensor_name, input_dim_idx)] = padded_dim_name

    def output_dim_is_padded(
        self,
        output_tensor_index: int,
        output_dim_idx: int,
        padded_dim_name: str,
    ) -> None:
        """Attach a named padded dimension to an output tensor axis.

        Records into ``output_dims_with_assigned_padded_dim`` that the
        ``output_dim_idx`` axis of output tensor ``output_tensor_index`` is
        padded/bucketed according to the ``padded_dim_name`` padded dimension.

        Args:
            output_tensor_index: Position of the tensor in the ``forward``
                output.
            output_dim_idx: Axis of the output tensor. May be negative, counting
                from the end (``-1`` is the last dimension).
            padded_dim_name: Name of a dimension previously declared with
                :meth:`set_padded_dim`.

        Raises:
            ValueError: If ``padded_dim_name`` was not configured or
                ``output_tensor_index`` is negative.
        """
        self._check_dim_name(padded_dim_name)
        if output_tensor_index < 0:
            raise ValueError(
                f"output_tensor_index must be >= 0, got {output_tensor_index}")
        self.output_dims_with_assigned_padded_dim[
            (output_tensor_index, output_dim_idx)] = padded_dim_name

    def _check_dim_name(self, dim_name: str) -> None:
        if dim_name not in self.padded_dims:
            raise ValueError(
                f"padded dim {dim_name!r} is not configured; call "
                "set_padded_dim before assigning it "
                f"(known: {sorted(self.padded_dims)})")

    def export_config(self) -> InputRoutingConfig:
        """Snapshot the collected rules into a :class:`InputRoutingConfig`."""
        input_assignments: List[InputDimAssignment] = [
            (input_tensor_name, input_dim_idx, dim_name)
            for (input_tensor_name, input_dim_idx), dim_name
            in self.input_dims_with_assigned_padded_dim.items()
        ]
        output_assignments: List[OutputDimAssignment] = [
            (output_tensor_index, output_dim_idx, dim_name)
            for (output_tensor_index, output_dim_idx), dim_name
            in self.output_dims_with_assigned_padded_dim.items()
        ]
        acceptance_input_assignments: List[InputDimAssignment] = [
            (input_tensor_name, input_dim_idx, dim_name)
            for (input_tensor_name, input_dim_idx), dim_name
            in self.input_dims_with_assigned_input_acceptance_dim.items()
        ]
        return InputRoutingConfig(
            padded_dims=dict(self.padded_dims),
            input_dims_with_assigned_padded_dim=input_assignments,
            output_dims_with_assigned_padded_dim=output_assignments,
            input_acceptance_dims=dict(self.input_acceptance_dims),
            input_dims_with_assigned_input_acceptance_dim=acceptance_input_assignments,
        )


class GraphOptimizationMode(enum.Enum):
    """Which graph-optimization backend to apply when wrapping a module.

    Members:
        NO_OPTIMIZATION: Leave the module untouched
        CUDA_GRAPH_VIA_TORCH: Wrap the module to drive per-input-key CUDA-graph
            warmup, capture, and replay via ``torch.cuda.graph``.
    """
    NO_OPTIMIZATION = "no_optimization"
    CUDA_GRAPH_VIA_TORCH = "cuda_graph_via_torch"
    # ToDo: TORCH_COMPILE = "torch_compile"


class InputKeyMethod(enum.Enum):
    """How to derive the per-call cache key that selects a captured graph.

    Each distinct key maps to its own ``CUDAGraphState`` (and thus its own
    captured graph / static buffers) in the tracker's LRU cache.

    Members:
        EXACT: Key on the exact shapes of the input tensors, so a separate graph
            is captured for every distinct input-shape signature.
        BUCKETED_SHAPES: Key on bucketed (padded) input shapes, so one captured
            graph serves a range of live shapes; inputs are padded up to their
            bucket length for capture/replay and outputs truncated back.
    """
    EXACT = "exact"
    BUCKETED_SHAPES = "bucketed_shapes"


class GraphOptimizationConfig(BaseConfig):
    """Base config describing how a module should be graph-optimized.

    Rooted in :class:`tensorrt_bionemo.configs.BaseConfig` so it composes with the
    rest of the config system (a module's ``config.graph_optimization_config`` holds
    one of these, and the wrapper carries it as its own ``config``). Concrete
    subclasses (e.g. :class:`CUDAGraphOptimizationConfig`) add framework-specific
    options.

    Attributes:
        graph_optimization_mode: Backend to apply; ``NO_OPTIMIZATION`` leaves
            the module eager. See :class:`GraphOptimizationMode`.
        input_key_method: How a call's input is reduced to a graph-cache key.
            See :class:`InputKeyMethod`.
        input_routing_config: Optional :class:`InputRoutingConfig` supplying the
            input-acceptance limits and shape-bucket (padding) rules; ``None``
            disables both (all inputs accepted, no padding).
        num_graphs_max_for_this_module: Capacity of the per-module LRU cache of
            captured graphs; the least-recently-used graph is evicted (and its
            buffers freed) once this many distinct input keys are live.
    """
    graph_optimization_mode: GraphOptimizationMode = GraphOptimizationMode.NO_OPTIMIZATION
    input_key_method: InputKeyMethod = InputKeyMethod.EXACT
    input_routing_config: InputRoutingConfig = None
    num_graphs_max_for_this_module: int = 1


class CUDAGraphOptimizationConfig(GraphOptimizationConfig):
    """Config for the ``CUDA_GRAPH_VIA_TORCH`` path.

    Adds the warmup-schedule thresholds the tracker uses to advance an input
    key's state machine (warmup -> kernels compiled -> allocator ready ->
    captured) before capturing its CUDA graph, plus an optional post-capture
    correctness check.

    Attributes:
        num_calls_for_kernel_compilation: Number of prior calls for a key
            before kernels are considered compiled/autotuned.
        num_calls_for_memory_allocator: Number of prior calls for a key
            before the caching allocator is considered primed and the graph is
            captured on the next call.
        verify_capture: When True, immediately after capturing a key's graph the
            tracker replays it and compares the result against a fresh eager run;
            a mismatch reverts that key permanently to eager. Adds one extra eager
            forward at capture time, so it is off by default.
    """
    num_calls_for_kernel_compilation: int = 1
    num_calls_for_memory_allocator: int = 3
    verify_capture: bool = False
