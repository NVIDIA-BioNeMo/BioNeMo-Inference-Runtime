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

"""Generate the API reference: one virtual page per ``bionemo_ir`` module.

Runs inside the MkDocs build via mkdocs-gen-files (see ``mkdocs.yml``); the
pages never touch the checkout. Griffe parses the sources statically, so the
GPU stack that ``bionemo_ir`` imports at runtime is not needed to build docs.

Modules under namespace directories (no ``__init__.py``, e.g.
``dsl_kernels/cute/``) are skipped: griffe's static resolver cannot attach
them to their parent package. They are listed in a build warning.
"""

import logging
import sys
from pathlib import Path

import mkdocs_gen_files

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "bionemo_ir"

log = logging.getLogger("mkdocs.plugins.gen_files")

# Griffe falls back to sys.path for module resolution; make sure the repo
# root is on it no matter where the mkdocs command is invoked from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _under_regular_package(path: Path) -> bool:
    parent = path.parent
    while parent != REPO_ROOT:
        if not (parent / "__init__.py").exists():
            return False
        parent = parent.parent
    return True


nav = mkdocs_gen_files.Nav()
skipped: list[str] = []

for path in sorted((REPO_ROOT / PACKAGE).rglob("*.py")):
    if not _under_regular_package(path):
        skipped.append(path.relative_to(REPO_ROOT).as_posix())
        continue
    module_path = path.relative_to(REPO_ROOT).with_suffix("")
    doc_path = module_path.with_suffix(".md")
    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    if not parts:
        continue

    full_doc_path = Path("reference", doc_path)
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")
        if is_package:
            # Re-exported members are documented on their defining module's
            # page; rendering them here too would duplicate anchors.
            fd.write("    options:\n      show_submodules: true\n      members: []\n")

if skipped:
    log.warning("skipping modules under namespace directories (no __init__.py):\n  " + "\n  ".join(skipped))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
