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
"""Provider adapter for a user-supplied Python callable."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bionemo_ir.data.schemas import OracleRequest, OracleResponse
from bionemo_ir.oracles.providers.base import OracleProvider

OracleCallable = Callable[[Sequence[OracleRequest]], Sequence[OracleResponse]]


class CallableOracleProvider(OracleProvider):
    """Run a user callable behind the validated provider base contract."""

    def __init__(self, name: str, function: OracleCallable) -> None:
        if not name:
            raise ValueError("oracle provider name must not be empty")
        self._name = name
        self._function = function

    @property
    def name(self) -> str:
        """Return the configured provider name."""
        return self._name

    def run(self, requests: Sequence[OracleRequest]) -> Sequence[OracleResponse]:
        """Delegate a request batch to the user callable."""
        return self._function(requests)


__all__ = ["CallableOracleProvider", "OracleCallable"]
