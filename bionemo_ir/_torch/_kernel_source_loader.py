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
"""Import a kernel family's optional ``_source`` adapter, if present.

Source-free distributions omit these adapters, so every caller must tolerate
:class:`ImportError` and fall back to the packaged CUBIN path.

The module name is assembled at run time rather than written as a literal
``from ._source import ...``, so that a missing adapter surfaces as a
recoverable error at kernel-resolution time rather than an import error at
package-import time.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["load_source_module"]

_SOURCE_SUFFIX = "_" + "source"


def load_source_module(package: str) -> ModuleType:
    """Return ``<package>._source``, raising :class:`ImportError` if stripped."""
    return importlib.import_module(f"{package}.{_SOURCE_SUFFIX}")
