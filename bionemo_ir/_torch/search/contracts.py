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
"""Metric evaluators and weighted objectives for search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Key under which the aggregated scalar reward is published into metrics.
SEARCH_REWARD_KEY = "search.reward"

_OVERRIDABLE_FIELDS = frozenset({"weights", "clip_values", "reward_metrics_tags"})


@dataclass(frozen=True)
class MetricClip:
    """Bounds applied to a metric value before its search weight."""

    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("clip must set min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"clip has min={self.min} > max={self.max}")

    def apply(self, value: float) -> float:
        """Clip ``value`` to the configured bounds."""
        clipped = float(value)
        if self.min is not None:
            clipped = max(clipped, self.min)
        if self.max is not None:
            clipped = min(clipped, self.max)
        return clipped


@dataclass(frozen=True)
class MetricEvaluationContext:
    """Identity and policy attached to one metric-evaluation batch.

    Attributes:
        checkpoint: Global rollout step being scored, if applicable.
        tags: Metric-selection tags from the active weighted objective.
        request_ids: Stable row IDs aligned with the decoded prediction batch.
    """

    checkpoint: int | None = None
    tags: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()


class MetricEvaluator(Protocol):
    """Score one decoded batch into per-row metric dictionaries.

    A mapping return is allowed when the batch size is 1. Callers may inject
    an OSS metric manager behind this protocol; BioIR core stays tensor-only.
    """

    def __call__(
        self,
        prediction: object,
        static_features: Mapping[str, object],
    ) -> Mapping[str, float] | Sequence[Mapping[str, float]]:
        """Return metrics for each row of ``prediction``."""


@runtime_checkable
class ContextualMetricEvaluator(Protocol):
    """Metric evaluator that consumes checkpoint identity and metric tags."""

    def evaluate(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Mapping[str, float] | Sequence[Mapping[str, float]]:
        """Return metrics for each row under ``context``."""


MetricEvaluatorLike = MetricEvaluator | ContextualMetricEvaluator


def evaluate_metrics(
    evaluator: MetricEvaluatorLike,
    prediction: object,
    static_features: Mapping[str, object],
    context: MetricEvaluationContext,
) -> Mapping[str, float] | Sequence[Mapping[str, float]]:
    """Call a contextual evaluator when supported, otherwise preserve the legacy call."""
    if isinstance(evaluator, ContextualMetricEvaluator):
        return evaluator.evaluate(prediction, static_features, context)
    return evaluator(prediction, static_features)


class CompositeMetricEvaluator:
    """Merge metrics from an ordered collection of evaluators.

    Every evaluator receives the same prediction, features, and context.
    Results must contain the same number of rows and disjoint metric keys.
    """

    def __init__(self, evaluators: Sequence[MetricEvaluatorLike]) -> None:
        if not evaluators:
            raise ValueError("evaluators must contain at least one metric evaluator")
        self.evaluators = tuple(evaluators)

    def evaluate(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Sequence[Mapping[str, float]]:
        """Evaluate and merge metrics row by row."""
        merged_rows: list[dict[str, float]] | None = None
        for evaluator_index, evaluator in enumerate(self.evaluators):
            result = evaluate_metrics(evaluator, prediction, static_features, context)
            rows = [dict(result)] if isinstance(result, Mapping) else [dict(row) for row in result]
            if merged_rows is None:
                merged_rows = [{} for _ in rows]
            elif len(rows) != len(merged_rows):
                raise ValueError(
                    f"metric evaluator {evaluator_index} returned {len(rows)} rows; expected {len(merged_rows)}"
                )
            for row_index, row in enumerate(rows):
                overlap = sorted(set(merged_rows[row_index]) & set(row))
                if overlap:
                    raise ValueError(
                        f"metric evaluator {evaluator_index} collides with earlier evaluators "
                        f"on row {row_index}: {overlap}"
                    )
                merged_rows[row_index].update(row)
        return merged_rows if merged_rows is not None else []


@dataclass(frozen=True)
class WeightedObjective:
    """Additive ``weight * clip(metric)`` objective used to rank candidates.

    A negative weight minimizes that metric. A weighted key that is missing or
    non-finite invalidates the candidate (``-inf``). Zero weights are skipped
    so a key can be disabled without removing it. ``checkpoint_overrides``
    replace named fields wholesale at a scoring step; omitted fields inherit.
    """

    weights: dict[str, float]
    clips: dict[str, MetricClip] = field(default_factory=dict)
    reward_metrics_tags: tuple[str, ...] = ()
    checkpoint_overrides: dict[int, WeightedObjective] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(key for key in self.clips if key not in self.weights)
        if unknown:
            raise ValueError(
                f"clip_values keys {unknown} have no entry in weights, so clipping them would have no effect"
            )

    def at_step(self, step: int) -> WeightedObjective:
        """Return the objective used when scoring checkpoint ``step``."""
        return self.checkpoint_overrides.get(int(step), self)

    def validate_checkpoint_steps(self, scoring_steps: Sequence[int]) -> None:
        """Reject overrides that target a step the search never scores."""
        unknown = sorted(set(self.checkpoint_overrides) - set(scoring_steps))
        if unknown:
            raise ValueError(
                f"checkpoint_overrides targets step(s) {unknown}, which the search never scores at; "
                f"overrides must key on one of {sorted(scoring_steps)}"
            )

    def score(self, metrics: Mapping[str, float]) -> float:
        """Aggregate ``metrics`` into a scalar reward, or ``-inf`` if invalid."""
        return score_metrics(metrics, self)


def score_metrics(metrics: Mapping[str, float], objective: WeightedObjective) -> float:
    """Weight clipped metric values. Missing or non-finite keys score ``-inf``."""
    reward = 0.0
    for key, weight in objective.weights.items():
        if not weight:
            continue
        if key not in metrics:
            return float("-inf")
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            return float("-inf")
        if not math.isfinite(value):
            return float("-inf")
        clip = objective.clips.get(key)
        if clip is not None:
            value = clip.apply(value)
        reward += weight * value
    if not math.isfinite(reward):
        return float("-inf")
    return float(reward)


def _as_clip(bounds: MetricClip | Mapping[str, float | None]) -> MetricClip:
    if isinstance(bounds, MetricClip):
        return bounds
    return MetricClip(min=bounds.get("min"), max=bounds.get("max"))


def build_weighted_objective(
    weights: Mapping[str, float],
    clip_values: Mapping[str, MetricClip | Mapping[str, float | None]] | None = None,
    reward_metrics_tags: Sequence[str] | None = None,
    checkpoint_overrides: Mapping[int, Mapping[str, object]] | None = None,
) -> WeightedObjective:
    """Build a :class:`WeightedObjective`, resolving per-checkpoint variants."""
    if not isinstance(weights, Mapping):
        raise ValueError(f"weights must map a metric key to a number; got {weights!r}")
    resolved_weights = dict(weights)
    null_weights = sorted(key for key, weight in resolved_weights.items() if weight is None)
    if null_weights:
        raise ValueError(f"weights {null_weights} are null; give a number (0.0 disables a term)")

    clips = {key: _as_clip(bounds) for key, bounds in (clip_values or {}).items()}
    tags = tuple(reward_metrics_tags or ())
    variants: dict[int, WeightedObjective] = {}
    for step, spec in (checkpoint_overrides or {}).items():
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"checkpoint_overrides keys must be integer steps; got {step!r}")
        if not isinstance(spec, Mapping):
            raise ValueError(f"checkpoint_overrides[{step}] must be a mapping; got {spec!r}")
        unknown = sorted(set(spec) - _OVERRIDABLE_FIELDS)
        if unknown:
            raise ValueError(
                f"checkpoint_overrides[{step}] has unsupported field(s) {unknown}; "
                f"supported fields: {sorted(_OVERRIDABLE_FIELDS)}"
            )
        variant_weights = spec.get("weights", resolved_weights)
        variant_clips = spec.get("clip_values", clips)
        variant_tags = spec.get("reward_metrics_tags", tags)
        if not isinstance(variant_weights, Mapping):
            raise ValueError(f"checkpoint_overrides[{step}].weights must be a mapping")
        resolved_clips = (
            {key: _as_clip(bounds) for key, bounds in variant_clips.items()}
            if isinstance(variant_clips, Mapping)
            else clips
        )
        variants[int(step)] = WeightedObjective(
            weights=dict(variant_weights),
            clips=resolved_clips,
            reward_metrics_tags=tuple(variant_tags or ()),
        )
    return WeightedObjective(
        weights=resolved_weights,
        clips=clips,
        reward_metrics_tags=tags,
        checkpoint_overrides=variants,
    )
