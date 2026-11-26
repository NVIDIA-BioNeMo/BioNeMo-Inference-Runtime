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
import sys
from importlib import import_module
from pathlib import Path

import pytest
from check_public_api import _top_level_names

_GETATTR = (
    "\n\ndef __getattr__(name):\n"
    "    module_name = _EXPORTS.get(name)\n"
    "    if module_name is None:\n"
    "        raise AttributeError(name)\n"
    "    return getattr(import_module(module_name, __name__), name)\n"
)


def _lazy_package(root: Path, exports: str, modeling: str = "class Model:\n    pass\n") -> Path:
    """Write a lazy package whose ``__getattr__`` resolves names through ``_EXPORTS``."""
    package = root / "pkg"
    package.mkdir()
    (package / "modeling.py").write_text(modeling)
    init = package / "__init__.py"
    init.write_text(f'from importlib import import_module\n\n__all__ = ["Model"]\n_EXPORTS = {exports}\n{_GETATTR}')
    return init


def test_lazy_exports_count_as_reexports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lazy module exports an ``__all__`` name that ``_EXPORTS`` maps to a submodule defining it."""
    source = _lazy_package(tmp_path, '{"Model": ".modeling"}')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pkg", raising=False)
    monkeypatch.delitem(sys.modules, "pkg.modeling", raising=False)

    assert import_module("pkg").Model.__name__ == "Model"
    assert "Model" in _top_level_names(source)


@pytest.mark.parametrize(
    ("exports", "modeling"),
    [
        pytest.param("{}", "class Model:\n    pass\n", id="not-in-exports"),
        pytest.param('{"Model": ".missing"}', "class Model:\n    pass\n", id="missing-submodule"),
        pytest.param('{"Model": ".modeling"}', "class Other:\n    pass\n", id="not-in-submodule"),
    ],
)
def test_unresolved_lazy_exports_are_rejected(tmp_path: Path, exports: str, modeling: str) -> None:
    """A lazy ``__all__`` name that ``__getattr__`` cannot resolve is not an export."""
    source = _lazy_package(tmp_path, exports, modeling)

    assert "Model" not in _top_level_names(source)


def test_getattr_without_exports_is_not_an_export(tmp_path: Path) -> None:
    """A ``__getattr__`` that always raises does not export ``__all__``."""
    source = tmp_path / "lazy.py"
    source.write_text('__all__ = ["Model"]\n\n\ndef __getattr__(name):\n    raise AttributeError(name)\n')

    assert "Model" not in _top_level_names(source)


def test_all_without_getattr_is_not_an_export(tmp_path: Path) -> None:
    """``__all__`` alone does not define a name."""
    source = tmp_path / "eager.py"
    source.write_text('__all__ = ["Model"]\n')

    assert "Model" not in _top_level_names(source)
