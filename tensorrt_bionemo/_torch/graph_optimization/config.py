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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


@dataclass(frozen=True)
class NamedDimTies:
    """Signature-derived tie points for one named dimension of a module.

    Records which ``forward`` input/output tensor axes carry ``name`` (e.g.
    ``num_tokens``). These are properties of the module's ``forward`` signature,
    not of any particular deployment, so they belong on the class.

    Attributes:
        name: The named dimension (the ``dim_name`` passed to the factory).
        input_dims: Ordered ``(tensor_name, axes)`` pairs. ``tensor_name`` is the
            ``forward`` parameter name (or ``"arg{i}"`` for a positional-only
            call before ``Signature.bind`` normalization); ``axes`` are the dim
            indices on that tensor that carry ``name``. Order is preserved so the
            exported config is stable and deterministic.
        output_dims: Ordered ``(output_tensor_index, axes)`` pairs tying output
            tensor axes to ``name`` (used only when bucketing/padding is active).
    """
    name: str
    input_dims: Tuple[Tuple[str, Tuple[int, ...]], ...]
    output_dims: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class PaddedDimSpec:
    name: str
    dim_len_min: int
    dim_len_max: int
    num_intervals: int
    multiple_of: int = 128
    spacing_method: str = "linear"


@dataclass(frozen=True)
class InputAcceptanceDimSpec:
    name: str
    dim_len_max: int


class InputRoutingConfig(BaseModel):
    """Serializable snapshot of the bucketing rules collected by a :class:`InputRoutingConfigFactory`.

    Attributes:
        named_dim_ties: One :class:`NamedDimTies` per named dimension, recording
            which ``forward`` input/output tensor axes carry that dimension. The
            ``config`` module expands these (together with ``padded_dims`` /
            ``input_acceptance_dims``) into the per-axis assignments the tracker
            consumes.
        padded_dims: One :class:`PaddedDimSpec` per named padded dimension,
            giving its bucketing spec (length range, interval count, alignment,
            spacing).
        input_acceptance_dims: One :class:`InputAcceptanceDimSpec` per named
            dimension, giving the inclusive upper-bound length accepted for it (an
            input axis tied to that dimension is in range when it does not exceed
            this length, out of range only when strictly greater). Carried as
            plain data so the acceptance predicate can be rebuilt from a
            deserialized snapshot — no un-picklable closure ever crosses a
            boundary.
        static_args: Names of ``forward`` arguments the user has declared do not
            change from one call to the next for a given input key (so they need
            not participate in that key).
        internal_workspace_kwargs: Names of ``forward`` keyword arguments that
            carry graph-internal scratch (e.g. preallocated ``buffers``) rather
            than stable inputs; excluded from input-key derivation and from the
            per-replay static-buffer copy.
    """

    named_dim_ties: List[NamedDimTies] = Field(default_factory=list)
    padded_dims: List[PaddedDimSpec] = Field(default_factory=list)
    input_acceptance_dims: List[InputAcceptanceDimSpec] = Field(
        default_factory=list)
    static_args: List[str] = Field(default_factory=list)
    internal_workspace_kwargs: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"


class InputRoutingConfigFactory:
    """Collects shape-bucketing rules for a module's ``forward`` inputs/outputs.

    Routing is organized around **named** dimensions: the tie points (which
    ``forward`` input/output axes carry each named dim) are declared as
    :class:`NamedDimTies` via :meth:`set_named_dim_ties`, and what to do with a
    dim is declared separately — a bucket (padding) spec via
    :meth:`set_padded_dim` and/or an acceptance limit via
    :meth:`set_input_acceptance_dim`. Axes sharing a name are padded to the same
    bucket length in lockstep.

    Call :meth:`export_config` to snapshot the collected rules into an
    immutable, serializable :class:`InputRoutingConfig`.
    """

    def __init__(self) -> None:
        # Tie points: which forward input/output axes carry each named dim.
        self.named_dim_ties: List[NamedDimTies] = []
        # Per-dim declarative bucket (padding) specs, keyed by ``.name``.
        self.padded_dims: List[PaddedDimSpec] = []
        # Per-dim acceptance-limit specs, keyed by ``.name``.
        self.input_acceptance_dims: List[InputAcceptanceDimSpec] = []
        # Names of forward args declared static (order-preserving, deduped).
        self.static_args: List[str] = []
        # Names of forward kwargs carrying graph-internal workspaces.
        self.internal_workspace_kwargs: List[str] = []

    @staticmethod
    def _upsert_by_name(items: list, item) -> None:
        """Replace an existing entry with the same ``.name``, else append.

        Preserves first-seen order while letting a later declaration for a
        dim override the earlier one."""
        for i, existing in enumerate(items):
            if existing.name == item.name:
                items[i] = item
                return
        items.append(item)

    def set_named_dim_ties(self, named_dim_ties: Sequence[NamedDimTies]) -> None:
        """Declare the signature-derived tie points for one or more named dims.

        Each :class:`NamedDimTies` records which ``forward`` input/output axes
        carry its dimension. A tie for a name already present replaces it.
        """
        for tie in named_dim_ties:
            self._upsert_by_name(self.named_dim_ties, tie)


    def set_static_args(self, arg_names: Sequence[str]) -> None:
        """Declare ``forward`` arguments that are static for a given input key.

        Appends each name in order, skipping duplicates, into ``static_args``.
        """
        for name in arg_names:
            if name not in self.static_args:
                self.static_args.append(name)

    def set_internal_workspace_kwargs(self, kwarg_names: Sequence[str]) -> None:
        """Declare ``forward`` keyword args that carry graph-internal scratch.

        Appends each name in order, skipping duplicates, into
        ``internal_workspace_kwargs``.
        """
        for name in kwarg_names:
            if name not in self.internal_workspace_kwargs:
                self.internal_workspace_kwargs.append(name)

    def set_input_acceptance_dim(self, dim_name: str, dim_len_max: int) -> None:
        """Declare the largest input length accepted for a named dimension.

        Records an :class:`InputAcceptanceDimSpec` that inputs whose ``dim_name``
        axis is strictly longer than ``dim_len_max`` fall outside this module's
        captured-graph coverage. Keyed by ``dim_name``, so a later call for the
        same name overrides the earlier limit. The axes the limit applies to are
        those the dim's :class:`NamedDimTies` ties it to.

        Args:
            dim_name: Name of the dimension the rule applies to.
            dim_len_max: Inclusive upper bound on the accepted length; an axis is
                accepted when it does not exceed this, rejected only when
                strictly greater.
        """
        self._upsert_by_name(
            self.input_acceptance_dims,
            InputAcceptanceDimSpec(name=dim_name, dim_len_max=dim_len_max))

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

        self._upsert_by_name(
            self.padded_dims,
            PaddedDimSpec(
                name=dim_name,
                dim_len_min=dim_len_min,
                dim_len_max=dim_len_max,
                num_intervals=num_intervals,
                multiple_of=multiple_of,
                spacing_method=spacing_method.value,
            ))

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
            ValueError: If ``multiple_of`` is not positive, or if any entry of
                ``dim_len_values`` is negative (a length can't be negative, and
                the ceil-to-multiple arithmetic is only meaningful for
                non-negative values).
        """
        if multiple_of < 1:
            raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
        if any(v < 0 for v in dim_len_values):
            raise ValueError(
                "dim_len_values must be non-negative, got "
                f"{tuple(dim_len_values)}")
        return tuple(
            InputRoutingConfigFactory.ceil_div(v, multiple_of) * multiple_of for v in dim_len_values)

    @staticmethod
    def ceil_div(k: int, divisor: int)-> int:
        return (k + divisor - 1) // divisor

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

    def export_config(self) -> InputRoutingConfig:
        """Snapshot the collected rules into a :class:`InputRoutingConfig`.

        Raises:
            ValueError: if a named dimension's acceptance maximum
                exceeds its largest capture bucket — an accepted input could
                then not be padded to any bucket.
        """
        padded_by_name = {spec.name: spec for spec in self.padded_dims}
        for acc in self.input_acceptance_dims:
            spec = padded_by_name.get(acc.name)
            if spec is None:
                continue
            largest_bucket = max(_bucket_lengths(spec))
            if acc.dim_len_max > largest_bucket:
                raise ValueError(
                    f"acceptance maximum {acc.dim_len_max} for dim "
                    f"{acc.name!r} exceeds its largest capture bucket "
                    f"{largest_bucket}; an accepted input could not be padded "
                    "to a bucket")
        return InputRoutingConfig(
            named_dim_ties=list(self.named_dim_ties),
            padded_dims=list(self.padded_dims),
            input_acceptance_dims=list(self.input_acceptance_dims),
            static_args=list(self.static_args),
            internal_workspace_kwargs=list(self.internal_workspace_kwargs),
        )


def effective_ranges(config: InputRoutingConfig) -> Dict[str, dict]:
    """Report the effective live range and capture bucket per named dim.

    For each named dimension the config references (via acceptance and/or
    padding), returns ``{dim_name: {"live_max", "covering_bucket",
    "largest_bucket"}}`` where:

    * ``live_max`` is the largest live length the module handles — the
      acceptance maximum if one is set, else the largest capture bucket.
    * ``covering_bucket`` is the smallest capture bucket that covers
      ``live_max`` (``None`` when the dim has no buckets).
    * ``largest_bucket`` is the largest capture bucket (``None`` when unbucketed).

    Purely informational; :meth:`InputRoutingConfigFactory.export_config` is what
    rejects an acceptance maximum above the largest bucket.
    """
    acceptance_by_name = {
        spec.name: spec.dim_len_max for spec in config.input_acceptance_dims}
    padded_by_name = {spec.name: spec for spec in config.padded_dims}

    ranges: Dict[str, dict] = {}
    for dim_name in sorted(set(acceptance_by_name) | set(padded_by_name)):
        acceptance_max = acceptance_by_name.get(dim_name)
        spec = padded_by_name.get(dim_name)
        bucket_lengths = _bucket_lengths(spec) if spec is not None else None
        largest_bucket = max(bucket_lengths) if bucket_lengths else None
        live_max = acceptance_max if acceptance_max is not None else largest_bucket
        covering_bucket = None
        if bucket_lengths is not None and live_max is not None:
            covering_bucket = min(
                (b for b in bucket_lengths if b >= live_max), default=None)
        ranges[dim_name] = {
            "live_max": live_max,
            "covering_bucket": covering_bucket,
            "largest_bucket": largest_bucket,
        }
    return ranges


def _bucket_lengths(spec: PaddedDimSpec) -> Tuple[int, ...]:
    """Recompute the aligned bucket-boundary lengths for a declarative
    :class:`PaddedDimSpec` (which stores the spec, not the derived lengths)."""
    values = InputRoutingConfigFactory.compute_dim_len_values(
        spec.dim_len_min, spec.dim_len_max, spec.num_intervals,
        SpacingMethod(spec.spacing_method))
    return InputRoutingConfigFactory.snap_to(values, spec.multiple_of)


# ---------------------------------------------------------------------------
# Consumer-facing views: expand the named-dim tie points + per-dim specs into
# the flat ``(tensor_name, dim_idx, dim_name)`` assignment tuples the tracker
# iterates. A dim's tie axes serve both acceptance and padding; whether a dim is
# accepted and/or padded is decided by its presence in ``input_acceptance_dims``
# / ``padded_dims``.
# ---------------------------------------------------------------------------
def input_acceptance_assignments(
        config: InputRoutingConfig) -> List[Tuple[str, int, str]]:
    """``[(tensor_name, dim_idx, dim_name)]`` for every input axis tied to a dim
    that carries an acceptance limit."""
    accepted = {spec.name for spec in config.input_acceptance_dims}
    return [(tensor_name, axis, tie.name)
            for tie in config.named_dim_ties if tie.name in accepted
            for tensor_name, axes in tie.input_dims for axis in axes]


def input_padded_assignments(
        config: InputRoutingConfig) -> List[Tuple[str, int, str]]:
    """``[(tensor_name, dim_idx, dim_name)]`` for every input axis tied to a dim
    that carries a bucket (padding) spec."""
    padded = {spec.name for spec in config.padded_dims}
    return [(tensor_name, axis, tie.name)
            for tie in config.named_dim_ties if tie.name in padded
            for tensor_name, axes in tie.input_dims for axis in axes]


def output_padded_assignments(
        config: InputRoutingConfig) -> List[Tuple[int, int, str]]:
    """``[(output_tensor_index, dim_idx, dim_name)]`` for every output axis tied
    to a dim that carries a bucket (padding) spec."""
    padded = {spec.name for spec in config.padded_dims}
    return [(out_idx, axis, tie.name)
            for tie in config.named_dim_ties if tie.name in padded
            for out_idx, axes in tie.output_dims for axis in axes]


def acceptance_max_by_name(config: InputRoutingConfig) -> Dict[str, int]:
    """``{dim_name: dim_len_max}`` acceptance limits keyed by name."""
    return {spec.name: spec.dim_len_max for spec in config.input_acceptance_dims}


def bucket_lengths_by_name(
        config: InputRoutingConfig) -> Dict[str, Tuple[int, ...]]:
    """``{dim_name: (aligned bucket-boundary lengths)}`` for each padded dim."""
    return {spec.name: _bucket_lengths(spec) for spec in config.padded_dims}


class GraphOptimizationMode(enum.Enum):
    """Which graph-optimization backend to apply when wrapping a module.

    Members:
        NO_OPTIMIZATION: Leave the module untouched
        CUDA_GRAPH_VIA_TORCH: Wrap the module to drive per-input-key CUDA-graph
            warmup, capture, and replay via ``torch.cuda.graph``.
    """
    NO_OPTIMIZATION = "no_optimization"
    CUDA_GRAPH_VIA_TORCH = "cuda_graph_via_torch"
    # Note: ``torch.compile`` is not offered as a mode here — the modules in
    # this repo are optimized via explicit CUDA-graph capture instead.


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
    input_routing_config: Optional[InputRoutingConfig] = None
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


