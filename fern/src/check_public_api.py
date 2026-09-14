#!/usr/bin/env python3
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

"""Validate BioIR imports in the public Python API documentation."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from common import REPO_ROOT, Finding, report

PYTHON_FENCE_RE = re.compile(r"^```python\s*\n(?P<body>.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _module_path(repo_root: Path, module: str) -> Path | None:
    """Return the source path that defines an imported module."""
    relative = Path(*module.split("."))
    module_file = repo_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = repo_root / relative / "__init__.py"
    return package_file if package_file.is_file() else None


def _top_level_names(path: Path) -> set[str]:
    """Return names defined or imported at the top level of a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
    return names


def _python_blocks(text: str) -> list[tuple[int, str]]:
    """Return the source line and body of each Python code block."""
    return [
        (text.count("\n", 0, match.start("body")) + 1, match.group("body")) for match in PYTHON_FENCE_RE.finditer(text)
    ]


def check(repo_root: Path | None = None) -> list[Finding]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    api_page = repo_root / "docs" / "ref" / "api.md"
    findings: list[Finding] = []
    documented: set[str] = set()
    if not api_page.is_file():
        return [Finding(api_page.relative_to(repo_root), 0, "missing public API documentation")]

    text = api_page.read_text(encoding="utf-8")
    relative = api_page.relative_to(repo_root)
    for line, body in _python_blocks(text):
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            findings.append(Finding(relative, line + (exc.lineno or 1) - 1, f"invalid Python example: {exc.msg}"))
            continue

        for node in ast.walk(tree):
            imports: list[tuple[str, str | None]] = []
            if isinstance(node, ast.Import):
                imports = [(alias.name, None) for alias in node.names if alias.name.startswith("bionemo_ir")]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("bionemo_ir")
            ):
                imports = [(node.module, alias.name) for alias in node.names]

            for module, symbol in imports:
                label = f"{module}.{symbol}" if symbol else module
                documented.add(label)
                source = _module_path(repo_root, module)
                if source is None:
                    findings.append(Finding(relative, line, f"module does not exist: {module}"))
                elif symbol == "*":
                    findings.append(Finding(relative, line, f"wildcard import is not allowed: {module}"))
                elif symbol and symbol not in _top_level_names(source):
                    findings.append(
                        Finding(
                            relative,
                            line,
                            f"{symbol} is not defined or re-exported by {source.relative_to(repo_root)}",
                        )
                    )

    if not documented:
        findings.append(Finding(relative, 0, "no BioIR imports found in Python examples"))
    print(f"Checked {len(documented)} documented BioIR imports.")
    return findings


def main() -> int:
    """Validate the published public API index."""
    return report(check())


if __name__ == "__main__":
    raise SystemExit(main())
