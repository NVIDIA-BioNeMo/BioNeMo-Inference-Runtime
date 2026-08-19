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
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from torch import nn


def setattr_safe(module: nn.Module, attr_name: str, attr_value: Any):
    if hasattr(module, attr_name):
        setattr(module, attr_name, attr_value)
    else:
        raise AttributeError(f"module {module} does not have attribute {attr_name}")


def path_for_package_in_repo(package: ModuleType) -> Path:
    return Path(package.__file__).parent


def require_vendored_submodule(path: Path) -> str:
    """Return *path* as a sys.path entry, skipping the caller if it is empty.

    The reference implementations under ``3rdparty/`` are optional: a checkout
    that never initialised them keeps the directories and leaves them empty, so
    importing from one fails while pytest is still collecting. A collection
    error stops the module outright and is reported as an error rather than a
    skip, which is neither what docs/dev.md promises for an absent input nor
    something the affected tests can do anything about.

    Skipping at module level instead means the guard serves every entry point at
    once -- pytest, scripts/run_tests.sh, or a bare pytest invocation -- rather
    than each maintaining its own list of files to ignore.
    """
    if not path.is_dir() or next(path.iterdir(), None) is None:
        pytest.skip(
            f"the {path.name} reference implementation is not checked out; "
            "initialise it with `make -C docker submodules`",
            allow_module_level=True,
        )
    return str(path)
