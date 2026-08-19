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
_CUTEDSL_LIBRARY_DIR = Path(__file__).resolve().parents[1] / "bionemo_ir" / "libs"
_CUTEDSL_LIBRARY_MODULE = "bionemo_ir.libs._cutedsl_kernels"
_CUTEDSL_BUILD_COMMAND = "pip install --no-build-isolation -v -e '.[dev]'"


def _not_built_message() -> str:
    return (
        "This is a source-free public checkout, so pytest requires "
        f"{_CUTEDSL_LIBRARY_MODULE}. Build the shared library first with:\n"
        f"  {_CUTEDSL_BUILD_COMMAND}\n"
        "Ensure BIOIR_BUILD_CUTEDSL_KERNELS is not set to 0."
    )


def _will_not_load_message(built: Path, error: BaseException) -> str:
    # The driver is the common cause but not the only one: a wrong interpreter
    # ABI or a missing runtime library fails the same dlopen. Name the driver
    # only when the loader named it, so the remedy matches the error.
    if "libcuda.so.1" in str(error):
        cause = (
            "It links libcuda.so.1, which the NVIDIA container toolkit injects at "
            "`docker run --gpus`. A container started without GPUs, or a host with no "
            "driver, cannot import it -- rebuilding will not help."
        )
    else:
        cause = (
            "The extension is present, so the build is not what failed. The loader "
            "error above names what is missing; rebuild only if it names something "
            "the build produces."
        )
    return f"{_CUTEDSL_LIBRARY_MODULE} is built ({built.name}) but will not load:\n  {error}\n{cause}"


def require_public_cutedsl_library() -> None:
    """Require the CUBIN launcher extension in a source-free checkout."""
    if any(path.is_file() and path.name != "__init__.py" for path in _PRIVATE_CUTEDSL_SOURCE_DIR.glob("*.py")):
        return

    try:
        importlib.import_module(_CUTEDSL_LIBRARY_MODULE)
    except (ImportError, OSError) as error:
        # A failed dlopen surfaces as ImportError too, so the exception type does
        # not separate "never built" from "built, but the driver is absent".
        # The extension on disk does: if it is there, no rebuild will fix this.
        built = next(iter(sorted(_CUTEDSL_LIBRARY_DIR.glob("_cutedsl_kernels*.so"))), None)
        message = _not_built_message() if built is None else _will_not_load_message(built, error)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise pytest.UsageError(message) from error
