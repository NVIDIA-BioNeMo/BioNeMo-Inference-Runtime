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

"""Import the repo's ``docs/`` tree into the site as virtual pages.

``docs/`` is written for GitHub rendering and must stay untouched, so the
import happens at build time (mkdocs-gen-files). Files are copied verbatim
except for link rewriting: links that escape ``docs/`` (to source files,
examples, or the repo README) would 404 on the rendered site, so they are
rewritten to absolute GitHub URLs. Links between imported files are left
alone — the tree structure is preserved under ``development/``, so they keep
resolving inside the site.
"""

import re
from pathlib import Path

import mkdocs_gen_files

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
TARGET_DIR = "development"
GITHUB_REPO = "https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime"

# docs/nv/ holds NVIDIA-internal release/process docs; keep them off the
# public site. Links from imported files into it are rewritten to GitHub.
EXCLUDED_TOP_DIRS = {"nv"}

_DIR_TITLES = {"ref": "Reference"}
_INLINE_LINK = re.compile(r"\]\(([^)]+)\)")
_REF_DEF = re.compile(r"^(\s*\[[^\]]+\]:\s*)(\S+)", re.MULTILINE)
_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _github_url(repo_rel: str) -> str:
    kind = "blob" if "." in repo_rel.rsplit("/", 1)[-1] else "tree"
    return f"{GITHUB_REPO}/{kind}/main/{repo_rel}"


def _rewrite_target(target: str, source: Path) -> str:
    if "://" in target or target.startswith(("#", "mailto:")):
        return target
    path, _, anchor = target.partition("#")
    if not path:
        return target
    resolved = (source.parent / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        return target
    if resolved.is_relative_to(DOCS_DIR):
        if resolved.relative_to(DOCS_DIR).parts[0] not in EXCLUDED_TOP_DIRS:
            # README.md is imported as index.md; keep in-tree links resolving.
            if resolved.name == "README.md":
                suffix = f"#{anchor}" if anchor else ""
                return path[: -len("README.md")] + "index.md" + suffix
            return target
    suffix = f"#{anchor}" if anchor else ""
    return _github_url(resolved.relative_to(REPO_ROOT).as_posix()) + suffix


def _rewrite_inline(match: re.Match[str], source: Path) -> str:
    parts = match.group(1).split(None, 1)
    target = _rewrite_target(parts[0], source)
    return "](" + target + (" " + parts[1] if len(parts) > 1 else "") + ")"


def _title_of(content: str, fallback: str) -> str:
    match = _HEADING.search(content)
    return match.group(1).strip() if match else fallback


def main() -> None:
    nav = mkdocs_gen_files.Nav()
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(DOCS_DIR)
        if rel.parts[0] in EXCLUDED_TOP_DIRS:
            continue
        dest = rel.with_name("index.md") if rel.name == "README.md" else rel
        content = path.read_text(encoding="utf-8")
        content = _INLINE_LINK.sub(lambda m, p=path: _rewrite_inline(m, p), content)
        content = _REF_DEF.sub(lambda m, p=path: m.group(1) + _rewrite_target(m.group(2), p), content)
        with mkdocs_gen_files.open(Path(TARGET_DIR, dest), "w") as fd:
            fd.write(content)
        title = "Overview" if rel == Path("README.md") else _title_of(content, rel.stem)
        key = tuple(_DIR_TITLES.get(d, d) for d in dest.parent.parts if d != ".") + (title,)
        nav[key] = dest.as_posix()
    with mkdocs_gen_files.open(Path(TARGET_DIR, "SUMMARY.md"), "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


main()
