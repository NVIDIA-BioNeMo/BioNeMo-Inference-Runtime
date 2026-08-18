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
"""A stripped build must say so, and must not claim a broken adapter is stripped.

Synthetic packages throughout: a real family's adapter is present in an internal
checkout and absent in a public one, so neither case is assertable from a family
name in both builds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bionemo_ir._torch._kernel_source_loader import (
    _SOURCE_SUFFIX,
    KernelSourceUnavailable,
    load_source_module,
)

_ADAPTER = f"{_SOURCE_SUFFIX}.py"


@pytest.fixture
def importable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Make packages built under ``tmp_path`` importable for one test."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def build(name: str, adapter: str | None) -> str:
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        if adapter is not None:
            (package / _ADAPTER).write_text(adapter)
        return name

    yield build

    for name in [module for module in sys.modules if module.startswith("bioir_fake_")]:
        del sys.modules[name]


def test_missing_adapter_names_the_build(importable) -> None:
    package = importable("bioir_fake_stripped", None)

    with pytest.raises(KernelSourceUnavailable, match="not available in this build"):
        load_source_module(package)


def test_the_typed_error_still_reaches_an_ImportError_handler(importable) -> None:
    """Every CUBIN fallback in the tree catches ImportError, not this class."""
    package = importable("bioir_fake_fallback", None)

    assert issubclass(KernelSourceUnavailable, ImportError)
    try:
        load_source_module(package)
    except ImportError as error:
        assert isinstance(error, KernelSourceUnavailable)
    else:
        pytest.fail("a stripped adapter must raise")


def test_a_broken_adapter_is_not_reported_as_stripped(importable) -> None:
    """Mislabelling this sends a reader hunting for a strip that never happened."""
    package = importable("bioir_fake_broken", "import bioir_no_such_dependency\n")

    with pytest.raises(ModuleNotFoundError) as raised:
        load_source_module(package)
    assert not isinstance(raised.value, KernelSourceUnavailable)
    assert raised.value.name == "bioir_no_such_dependency"


def test_a_present_adapter_is_returned(importable) -> None:
    package = importable("bioir_fake_present", "MARKER = 'adapter'\n")

    assert load_source_module(package).MARKER == "adapter"
