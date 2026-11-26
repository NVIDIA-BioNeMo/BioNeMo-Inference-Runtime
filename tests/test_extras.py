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
"""The ``all`` extra: the friendly import error and the lists that must agree."""

import ast
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

import bionemo_ir
from bionemo_ir import _extras
from bionemo_ir._extras import INSTALL_HINT, PIPELINE_MODULES, pipeline_extra

_REPO = Path(bionemo_ir.__file__).resolve().parent.parent
_PACKAGE = _REPO / "bionemo_ir"

# The distribution in the `all` extra that installs each import name.
_DISTRIBUTION = {
    "Bio": "biopython",
    "biotite": "biotite",
    "gemmi": "gemmi",
    "ihm": "modelcif",  # modelcif depends on ihm.
    "kalign": "kalign-python",
    "modelcif": "modelcif",
    "pyarrow": "ray",  # ray[data] installs pyarrow.
    "ray": "ray",
    "rdkit": "rdkit",
    "scipy": "scipy",
}


def _missing(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named '{name}'", name=name)


def test_real_import_of_an_absent_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real find_spec, with a package name that cannot exist.
    absent = "bioir_test_absent_pipeline_package"
    monkeypatch.setattr(_extras, "PIPELINE_MODULES", PIPELINE_MODULES | {absent})
    with pytest.raises(ModuleNotFoundError) as caught, pipeline_extra():
        __import__(f"{absent}.sub")
    assert INSTALL_HINT in str(caught.value)
    assert caught.value.name == absent


def test_stub_module_without_spec_keeps_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_spec raises ValueError for a sys.modules entry whose __spec__ is None.
    stub = "bioir_test_stub_pipeline_package"
    monkeypatch.setattr(_extras, "PIPELINE_MODULES", PIPELINE_MODULES | {stub})
    monkeypatch.setitem(sys.modules, stub, types.ModuleType(stub))
    original = _missing(f"{stub}.sub")
    with pytest.raises(ModuleNotFoundError) as caught, pipeline_extra():
        raise original
    assert caught.value is original


def test_absent_pipeline_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_extras, "_absent", lambda package: True)
    original = _missing("rdkit.Chem")
    with pytest.raises(ModuleNotFoundError) as caught, pipeline_extra():
        raise original
    assert INSTALL_HINT in str(caught.value)
    assert "'rdkit' package is not installed" in str(caught.value)
    assert caught.value.name == "rdkit.Chem"
    assert caught.value.__cause__ is original


def test_installed_package_with_missing_submodule_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # Installing the extra would not bring back a submodule of an installed package.
    monkeypatch.setattr(_extras, "_absent", lambda package: False)
    original = _missing("ray.removed_module")
    with pytest.raises(ModuleNotFoundError) as caught, pipeline_extra():
        raise original
    assert caught.value is original


def test_other_import_errors_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_extras, "_absent", lambda package: True)
    original = _missing("not_a_pipeline_package")
    with pytest.raises(ModuleNotFoundError) as caught, pipeline_extra():
        raise original
    assert caught.value is original

    with pytest.raises(ImportError) as caught_import, pipeline_extra():
        raise ImportError("cannot import name 'x' from 'rdkit'", name="rdkit")
    assert INSTALL_HINT not in str(caught_import.value)


def test_pipeline_modules_come_from_the_all_extra() -> None:
    assert set(_DISTRIBUTION) == PIPELINE_MODULES
    project = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]

    def names(requirements: list[str]) -> set[str]:
        return {re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", entry)[0].lower() for entry in requirements}

    extra = names(project["optional-dependencies"]["all"])
    base = names(project["dependencies"])
    assert set(_DISTRIBUTION.values()) <= extra, set(_DISTRIBUTION.values()) - extra
    assert not set(_DISTRIBUTION.values()) & base


def _wrapped_roots(path: Path) -> set[str]:
    """Top-level names imported inside ``with pipeline_extra():`` blocks of one file."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, "id", None) == "pipeline_extra"
            for item in node.items
        ):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                roots.update(alias.name.partition(".")[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and not child.level and child.module:
                roots.add(child.module.partition(".")[0])
    return roots


def _imported_roots(path: Path) -> set[str]:
    """Top-level names one file imports anywhere, at module level or deferred."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_wrapped_imports_are_pipeline_modules() -> None:
    wrapped: dict[str, set[str]] = {}
    for path in _PACKAGE.rglob("*.py"):
        for root in _wrapped_roots(path) - {"bionemo_ir"}:
            wrapped.setdefault(root, set()).add(str(path.relative_to(_REPO)))
    unknown = {root: sorted(files) for root, files in wrapped.items() if root not in PIPELINE_MODULES}
    assert not unknown, f"wrapped imports that PIPELINE_MODULES does not list: {unknown}"


def test_every_pipeline_module_is_still_imported() -> None:
    # A stale entry would keep a hint for a package the code no longer uses.
    imported = set().union(*(_imported_roots(path) for path in _PACKAGE.rglob("*.py")))
    stale = PIPELINE_MODULES - imported
    assert not stale, f"PIPELINE_MODULES lists names bionemo_ir never imports: {sorted(stale)}"
