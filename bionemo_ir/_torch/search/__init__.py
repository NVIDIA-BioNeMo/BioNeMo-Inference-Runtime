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
"""Reusable metric-weighted search contracts for iterative samplers."""

from .beam import select_beam_indices, validate_step_checkpoints
from .contracts import (
    SEARCH_REWARD_KEY,
    CompositeMetricEvaluator,
    ContextualMetricEvaluator,
    MetricClip,
    MetricEvaluationContext,
    MetricEvaluator,
    MetricEvaluatorLike,
    WeightedObjective,
    build_weighted_objective,
    evaluate_metrics,
    score_metrics,
)
from .rng import derive_search_seed

__all__ = [
    "SEARCH_REWARD_KEY",
    "CompositeMetricEvaluator",
    "ContextualMetricEvaluator",
    "MetricClip",
    "MetricEvaluationContext",
    "MetricEvaluator",
    "MetricEvaluatorLike",
    "WeightedObjective",
    "build_weighted_objective",
    "derive_search_seed",
    "evaluate_metrics",
    "score_metrics",
    "select_beam_indices",
    "validate_step_checkpoints",
]
