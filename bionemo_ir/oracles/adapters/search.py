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
"""Adapt custom oracle providers to metric-weighted search."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from bionemo_ir._torch.search import MetricEvaluationContext
from bionemo_ir.data.schemas import OracleRequest, OracleResponse, OracleStatus
from bionemo_ir.oracles.providers import OracleProvider

OracleRequestItem = OracleRequest | OracleResponse


class OracleRequestAdapter(Protocol):
    """Build one oracle request or terminal response per prediction row."""

    def batch_size(self, prediction: object) -> int:
        """Return the decoded prediction batch size."""

    def build_requests(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Sequence[OracleRequestItem]:
        """Build shape-aligned requests, using a response for unsupported rows."""


class OracleMetricEvaluator:
    """Adapt custom-oracle responses to contextual search metrics.

    Tag conditions are checked before building requests or calling the
    provider. They consume ``reward_metrics_tags`` from each checkpoint's
    weighted objective.
    """

    def __init__(
        self,
        provider: OracleProvider,
        request_adapter: OracleRequestAdapter,
        *,
        required_tags_any: Collection[str] = (),
        required_tags_all: Collection[str] = (),
        prohibited_tags: Collection[str] = (),
    ) -> None:
        self.provider = provider
        self.request_adapter = request_adapter
        self.required_tags_any = frozenset(required_tags_any)
        self.required_tags_all = frozenset(required_tags_all)
        self.prohibited_tags = frozenset(prohibited_tags)

    def _applies(self, tags: Collection[str]) -> bool:
        tag_set = frozenset(tags)
        if self.required_tags_any and self.required_tags_any.isdisjoint(tag_set):
            return False
        if not self.required_tags_all.issubset(tag_set):
            return False
        return self.prohibited_tags.isdisjoint(tag_set)

    def evaluate(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Sequence[Mapping[str, float]]:
        """Conditionally invoke the oracle provider and return its metrics."""
        batch_size = self.request_adapter.batch_size(prediction)
        rows: list[dict[str, float]] = [{} for _ in range(batch_size)]
        if not self._applies(context.tags):
            return rows

        items = tuple(self.request_adapter.build_requests(prediction, static_features, context))
        if len(items) != batch_size:
            raise ValueError(f"oracle request adapter returned {len(items)} rows for a batch of {batch_size}")
        item_ids = [item.request_id for item in items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("oracle request adapter must produce unique request IDs")

        pending = [item for item in items if isinstance(item, OracleRequest)]
        evaluated = self.provider.evaluate(pending)
        evaluated_by_id = {response.request_id: response for response in evaluated}
        for row_index, item in enumerate(items):
            response = item if isinstance(item, OracleResponse) else evaluated_by_id[item.request_id]
            if response.status is not OracleStatus.OK:
                continue
            rows[row_index].update(response.metrics)
        return rows


__all__ = ["OracleMetricEvaluator", "OracleRequestAdapter", "OracleRequestItem"]
