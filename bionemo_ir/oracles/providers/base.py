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
"""Base class for user-supplied structure-oracle providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType

from bionemo_ir.data.schemas import OracleRequest, OracleResponse


class OracleProvider(ABC):
    """Lifecycle and batch contract shared by all oracle providers.

    Subclasses implement :meth:`run`. Consumers call :meth:`evaluate`, which
    validates response IDs and restores request order. Each request must
    produce an explicit response, including failures and unsupported inputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable provider name for diagnostics and cache keys."""

    def setup(self) -> None:
        """Allocate optional provider resources."""
        return None

    def teardown(self) -> None:
        """Release optional provider resources."""
        return None

    @abstractmethod
    def run(self, requests: Sequence[OracleRequest]) -> Sequence[OracleResponse]:
        """Evaluate a batch of requests.

        Args:
            requests: Canonical oracle requests with unique request IDs.

        Returns:
            Exactly one response per request. Response order is unrestricted.
        """

    def evaluate(self, requests: Sequence[OracleRequest]) -> tuple[OracleResponse, ...]:
        """Run and validate a request batch, returning responses in request order."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("oracle provider name must be a non-empty string")
        ordered_requests = tuple(requests)
        if not ordered_requests:
            return ()
        request_ids = [request.request_id for request in ordered_requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("oracle request IDs must be unique within a batch")

        responses = tuple(self.run(ordered_requests))
        for response in responses:
            if not isinstance(response, OracleResponse):
                raise TypeError(f"oracle provider must return OracleResponse, got {type(response).__name__}")
        response_ids = [response.request_id for response in responses]
        if len(set(response_ids)) != len(response_ids):
            raise ValueError("oracle response IDs must be unique within a batch")
        missing = sorted(set(request_ids) - set(response_ids))
        unexpected = sorted(set(response_ids) - set(request_ids))
        if missing or unexpected:
            raise ValueError(f"oracle responses do not match requests: missing={missing}, unexpected={unexpected}")

        by_id = {response.request_id: response for response in responses}
        return tuple(by_id[request_id] for request_id in request_ids)

    def __call__(self, requests: Sequence[OracleRequest]) -> tuple[OracleResponse, ...]:
        """Alias :meth:`evaluate` for provider composition."""
        return self.evaluate(requests)

    def __enter__(self) -> OracleProvider:
        """Set up resources for a context-managed provider."""
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Tear down resources after a context-managed provider."""
        del exc_type, exc_value, traceback
        self.teardown()


__all__ = ["OracleProvider"]
