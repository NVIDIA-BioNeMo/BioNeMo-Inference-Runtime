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

import importlib
import warnings
from pathlib import Path

import pytest

_PRIVATE_CUTEDSL_SOURCE_DIR = Path(__file__).resolve().parents[1] / "bionemo_ir" / "dsl_kernels" / "cute"
_CUTEDSL_LIBRARY_MODULE = "bionemo_ir.libs._cutedsl_kernels"
_CUTEDSL_BUILD_COMMAND = "pip install --no-build-isolation -v -e '.[dev]'"


def require_public_cutedsl_library() -> None:
    """Require the CUBIN launcher extension in a source-free checkout."""
    if _PRIVATE_CUTEDSL_SOURCE_DIR.is_dir():
        return

    try:
        importlib.import_module(_CUTEDSL_LIBRARY_MODULE)
    except (ImportError, OSError) as error:
        message = (
            "This is a source-free public checkout, so pytest requires "
            f"{_CUTEDSL_LIBRARY_MODULE}. Build the shared library first with:\n"
            f"  {_CUTEDSL_BUILD_COMMAND}\n"
            "Ensure BIOIR_BUILD_CUTEDSL_KERNELS is not set to 0."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise pytest.UsageError(message) from error
