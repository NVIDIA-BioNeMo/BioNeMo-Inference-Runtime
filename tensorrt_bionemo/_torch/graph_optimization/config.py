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
import enum
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from tensorrt_bionemo.configs.base import BaseConfig


class SpacingMethod(enum.StrEnum):
    """Bucket-boundary spacing method."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class NamedDimTies:
    """Input and output axes tied to one named dimension.

    Attributes:
        name: Dimension name.
        input_dims: Ordered ``(parameter, axes)`` pairs.
        output_dims: Ordered ``(output index, axes)`` pairs.
    """

    name: str
    input_dims: tuple[tuple[str, tuple[int, ...]], ...]
    output_dims: tuple[tuple[int, tuple[int, ...]], ...] = ()


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
    """Serializable input-routing and bucketing rules.

    Internal workspace kwargs are excluded from graph keys and replay copies.
    """

    named_dim_ties: list[NamedDimTies] = Field(default_factory=list)
    padded_dims: list[PaddedDimSpec] = Field(default_factory=list)
    input_acceptance_dims: list[InputAcceptanceDimSpec] = Field(default_factory=list)
    static_args: list[str] = Field(default_factory=list)
    internal_workspace_kwargs: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"


class InputRoutingConfigFactory:
    """Build serializable routing rules around named dimensions."""

    def __init__(self) -> None:
        self.named_dim_ties: list[NamedDimTies] = []
        self.padded_dims: list[PaddedDimSpec] = []
        self.input_acceptance_dims: list[InputAcceptanceDimSpec] = []
        self.static_args: list[str] = []
        self.internal_workspace_kwargs: list[str] = []

    @staticmethod
    def _upsert_by_name(items: list, item) -> None:
        """Replace a same-name entry or append, preserving order."""
        for i, existing in enumerate(items):
            if existing.name == item.name:
                items[i] = item
                return
        items.append(item)

    def set_named_dim_ties(self, named_dim_ties: Sequence[NamedDimTies]) -> None:
        """Set input/output axis ties, replacing duplicate names."""
        for tie in named_dim_ties:
            self._upsert_by_name(self.named_dim_ties, tie)

    def set_static_args(self, arg_names: Sequence[str]) -> None:
        """Add deduplicated static ``forward`` arguments."""
        for name in arg_names:
            if name not in self.static_args:
                self.static_args.append(name)

    def set_internal_workspace_kwargs(self, kwarg_names: Sequence[str]) -> None:
        """Add deduplicated graph-owned workspace kwargs."""
        for name in kwarg_names:
            if name not in self.internal_workspace_kwargs:
                self.internal_workspace_kwargs.append(name)

    def set_input_acceptance_dim(self, dim_name: str, dim_len_max: int) -> None:
        """Set an inclusive input-length limit for a named dimension.

        Args:
            dim_name: Dimension name.
            dim_len_max: Maximum accepted length.
        """
        self._upsert_by_name(self.input_acceptance_dims, InputAcceptanceDimSpec(name=dim_name, dim_len_max=dim_len_max))

    def set_padded_dim(
        self,
        dim_name: str,
        dim_len_min: int,
        dim_len_max: int,
        num_intervals: int,
        multiple_of: int = 128,
        spacing_method: SpacingMethod | str = SpacingMethod.LINEAR,
    ) -> None:
        """Configure bucket boundaries for a padded dimension.

        Args:
            dim_name: Dimension name.
            dim_len_min: Minimum bucket length.
            dim_len_max: Maximum bucket length.
            num_intervals: Number of intervals between endpoints.
            multiple_of: Bucket alignment.
            spacing_method: Linear or exponential spacing.

        Raises:
            ValueError: If any bucket parameter is invalid.
        """
        if dim_len_min < 0 or dim_len_max < dim_len_min:
            raise ValueError(
                f"Require 0 <= dim_len_min <= dim_len_max, got dim_len_min={dim_len_min}, dim_len_max={dim_len_max}"
            )
        if num_intervals < 1:
            raise ValueError(f"num_intervals must be >= 1, got {num_intervals}")
        try:
            spacing_method = SpacingMethod(spacing_method)
        except ValueError:
            # The custom error already lists valid values.
            raise ValueError(
                f"spacing_method must be one of {[m.value for m in SpacingMethod]}, got {spacing_method!r}"
            ) from None
        if spacing_method == SpacingMethod.EXPONENTIAL and dim_len_min < 1:
            raise ValueError(f"exponential spacing requires dim_len_min >= 1, got {dim_len_min}")

        self._upsert_by_name(
            self.padded_dims,
            PaddedDimSpec(
                name=dim_name,
                dim_len_min=dim_len_min,
                dim_len_max=dim_len_max,
                num_intervals=num_intervals,
                multiple_of=multiple_of,
                spacing_method=spacing_method.value,
            ),
        )

    @staticmethod
    def snap_to(
        dim_len_values: tuple[int, ...],
        multiple_of: int,
    ) -> tuple[int, ...]:
        """Round lengths up to an alignment.

        Args:
            dim_len_values: Lengths to align.
            multiple_of: Positive alignment.

        Returns:
            Aligned lengths in input order.

        Raises:
            ValueError: If the alignment or a length is invalid.
        """
        if multiple_of < 1:
            raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
        if any(v < 0 for v in dim_len_values):
            raise ValueError(f"dim_len_values must be non-negative, got {tuple(dim_len_values)}")
        return tuple(InputRoutingConfigFactory.ceil_div(v, multiple_of) * multiple_of for v in dim_len_values)

    @staticmethod
    def ceil_div(k: int, divisor: int) -> int:
        return (k + divisor - 1) // divisor

    @staticmethod
    def compute_dim_len_values(
        dim_len_min: int,
        dim_len_max: int,
        num_intervals: int,
        spacing_method: SpacingMethod,
    ) -> tuple[int, ...]:
        """Compute rounded bucket boundaries, including both endpoints."""
        if spacing_method == SpacingMethod.LINEAR:
            values = np.linspace(dim_len_min, dim_len_max, num=num_intervals + 1)
        elif spacing_method == SpacingMethod.EXPONENTIAL:
            values = np.geomspace(dim_len_min, dim_len_max, num=num_intervals + 1)
        else:
            raise ValueError(
                f"spacing_method must be one of {[m.value for m in SpacingMethod]}, got {spacing_method!r}"
            )
        return tuple(int(round(v)) for v in values)

    def export_config(self) -> InputRoutingConfig:
        """Export validated routing rules.

        Raises:
            ValueError: If an accepted length exceeds all capture buckets.
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
                    "to a bucket"
                )
        return InputRoutingConfig(
            named_dim_ties=list(self.named_dim_ties),
            padded_dims=list(self.padded_dims),
            input_acceptance_dims=list(self.input_acceptance_dims),
            static_args=list(self.static_args),
            internal_workspace_kwargs=list(self.internal_workspace_kwargs),
        )


def effective_ranges(config: InputRoutingConfig) -> dict[str, dict]:
    """Return live maximum and covering/largest buckets per named dimension."""
    acceptance_by_name = {spec.name: spec.dim_len_max for spec in config.input_acceptance_dims}
    padded_by_name = {spec.name: spec for spec in config.padded_dims}

    ranges: dict[str, dict] = {}
    for dim_name in sorted(set(acceptance_by_name) | set(padded_by_name)):
        acceptance_max = acceptance_by_name.get(dim_name)
        spec = padded_by_name.get(dim_name)
        bucket_lengths = _bucket_lengths(spec) if spec is not None else None
        largest_bucket = max(bucket_lengths) if bucket_lengths else None
        live_max = acceptance_max if acceptance_max is not None else largest_bucket
        covering_bucket = None
        if bucket_lengths is not None and live_max is not None:
            covering_bucket = min((b for b in bucket_lengths if b >= live_max), default=None)
        ranges[dim_name] = {
            "live_max": live_max,
            "covering_bucket": covering_bucket,
            "largest_bucket": largest_bucket,
        }
    return ranges


def _bucket_lengths(spec: PaddedDimSpec) -> tuple[int, ...]:
    """Compute aligned boundaries from a declarative bucket spec."""
    values = InputRoutingConfigFactory.compute_dim_len_values(
        spec.dim_len_min, spec.dim_len_max, spec.num_intervals, SpacingMethod(spec.spacing_method)
    )
    return InputRoutingConfigFactory.snap_to(values, spec.multiple_of)


# Expand named ties into the flat assignments consumed by the tracker.
def input_acceptance_assignments(config: InputRoutingConfig) -> list[tuple[str, int, str]]:
    """Return input axes with acceptance limits."""
    accepted = {spec.name for spec in config.input_acceptance_dims}
    return [
        (tensor_name, axis, tie.name)
        for tie in config.named_dim_ties
        if tie.name in accepted
        for tensor_name, axes in tie.input_dims
        for axis in axes
    ]


def input_padded_assignments(config: InputRoutingConfig) -> list[tuple[str, int, str]]:
    """Return input axes with padding specs."""
    padded = {spec.name for spec in config.padded_dims}
    return [
        (tensor_name, axis, tie.name)
        for tie in config.named_dim_ties
        if tie.name in padded
        for tensor_name, axes in tie.input_dims
        for axis in axes
    ]


def output_padded_assignments(config: InputRoutingConfig) -> list[tuple[int, int, str]]:
    """Return output axes with padding specs."""
    padded = {spec.name for spec in config.padded_dims}
    return [
        (out_idx, axis, tie.name)
        for tie in config.named_dim_ties
        if tie.name in padded
        for out_idx, axes in tie.output_dims
        for axis in axes
    ]


def acceptance_max_by_name(config: InputRoutingConfig) -> dict[str, int]:
    """Return acceptance limits by dimension name."""
    return {spec.name: spec.dim_len_max for spec in config.input_acceptance_dims}


def bucket_lengths_by_name(config: InputRoutingConfig) -> dict[str, tuple[int, ...]]:
    """Return aligned bucket boundaries by dimension name."""
    return {spec.name: _bucket_lengths(spec) for spec in config.padded_dims}


class GraphOptimizationMode(enum.Enum):
    """Module graph-optimization backend."""

    NO_OPTIMIZATION = "no_optimization"
    CUDA_GRAPH_VIA_TORCH = "cuda_graph_via_torch"
    # This API supports explicit CUDA graphs, not ``torch.compile``.


class InputKeyMethod(enum.Enum):
    """Graph-cache key strategy: exact or bucketed input shapes."""

    EXACT = "exact"
    BUCKETED_SHAPES = "bucketed_shapes"


class GraphOptimizationConfig(BaseConfig):
    """Backend, keying, routing, and graph-cache settings."""

    graph_optimization_mode: GraphOptimizationMode = GraphOptimizationMode.NO_OPTIMIZATION
    input_key_method: InputKeyMethod = InputKeyMethod.EXACT
    input_routing_config: InputRoutingConfig | None = None
    num_graphs_max_for_this_module: int = 1


class CUDAGraphOptimizationConfig(GraphOptimizationConfig):
    """CUDA-graph warmup thresholds and optional capture verification."""

    num_calls_for_kernel_compilation: int = 1
    num_calls_for_memory_allocator: int = 3
    verify_capture: bool = False
