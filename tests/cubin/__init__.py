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

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

import pytest


def require_builder_sources(implementation_paths: Iterable[str]) -> None:
    """Skip source-dependent builder checks when private kernels were stripped."""
    modules = {path.rpartition(".")[0] for path in implementation_paths}
    try:
        missing = sorted(module for module in modules if importlib.util.find_spec(module) is None)
    except ImportError:
        missing = sorted(modules)
    if missing:
        pytest.skip(f"private CuTeDSL builder sources are unavailable: {', '.join(missing)}", allow_module_level=True)
