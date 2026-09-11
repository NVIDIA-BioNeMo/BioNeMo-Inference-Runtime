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

"""Enforce math conventions that render on both GitHub and the Fern site.

Canonical Markdown is read in two renderers: GitHub's math pipeline, which
rejects a set of macros (e.g. \\operatorname) with "The following macros are
not allowed", and Fern, which renders $...$ and $$...$$ but shows a fenced
```math block as literal code. These rules keep one syntax working on both:

- display math is written as $$ ... $$, never as a ```math fence;
- macros GitHub refuses are banned outright;
- a $$ block contains no blank lines and is always closed, both of which
  GitHub otherwise renders as literal text.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

from common import REPO_ROOT, Finding, iter_source_lines, report

# Directories mirroring the rumdl surface in .gitlab/ci/pages.yml: every
# Markdown file GitHub renders, whether or not Fern publishes it.
DOCSET_DIRS = ("docs", ".agents")
DOCSET_SUFFIXES = {".md", ".mdx"}
# Generated Fern snapshots mirror the canonical sources; linting them would
# double-report every finding.
EXCLUDED_PARTS = {".build"}

MATH_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*math\b", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
DISPLAY_MATH_RE = re.compile(r"\$\$")

# Macros measured to fail GitHub's math renderer. GitHub runs an unpublished
# macro allowlist, and one disallowed macro fails the whole block with
# "The following macros are not allowed: ...". The list is therefore
# empirical: grow it from observed failures, not from guesses about what a
# sanitizer would reject.
BLOCKED_MACROS = {
    # Confirmed failing in this repository's rendered docs (sampling.md);
    # \mathrm{...} renders the same upright text.
    "operatorname": r"use \mathrm{...} instead",
    # Blocked by GitHub after CSS-injection reports through \unicode.
    "unicode": None,
}
BLOCKED_MACRO_RE = re.compile(
    r"\\(?P<name>" + "|".join(sorted(BLOCKED_MACROS, key=len, reverse=True)) + r")(?![a-zA-Z])"
)


def _docset_paths(repo_root: Path) -> list[Path]:
    """Collect the Markdown files GitHub renders: the rumdl lint surface."""
    paths = [p for p in repo_root.iterdir() if p.is_file() and p.suffix.lower() in DOCSET_SUFFIXES]
    for name in DOCSET_DIRS:
        root = repo_root / name
        if root.is_dir():
            paths.extend(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in DOCSET_SUFFIXES and not EXCLUDED_PARTS & set(p.parts)
            )
    return sorted(paths)


def _lint_math(relative: Path, text: str) -> list[Finding]:
    """Check math conventions in one Markdown document."""
    findings: list[Finding] = []
    math_line = 0
    for number, raw, in_fence in iter_source_lines(text):
        if in_fence:
            if MATH_FENCE_RE.match(raw):
                findings.append(
                    Finding(
                        relative,
                        number,
                        "math fence renders as a code block on the Fern site; use $$ ... $$ display math",
                    )
                )
            continue
        line = INLINE_CODE_RE.sub("", raw)
        if math_line and not line.strip():
            findings.append(Finding(relative, number, f"blank line inside the $$ block opened on line {math_line}"))
        for match in BLOCKED_MACRO_RE.finditer(line):
            name = match.group("name")
            hint = BLOCKED_MACROS[name]
            message = f"macro is not allowed by GitHub's math renderer: \\{name}"
            if hint:
                message = f"{message}; {hint}"
            findings.append(Finding(relative, number, message))
        if len(DISPLAY_MATH_RE.findall(line)) % 2:
            math_line = 0 if math_line else number
    if math_line:
        findings.append(Finding(relative, math_line, "unclosed $$ display-math block"))
    return findings


def check_paths(paths: Iterable[Path], repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Check math conventions in the given Markdown files."""
    findings: list[Finding] = []
    for path in paths:
        path = Path(path)
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            relative = path
        findings.extend(_lint_math(relative, path.read_text(encoding="utf-8")))
    return findings


def check(repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Check math conventions across the linted documentation set."""
    paths = _docset_paths(repo_root.resolve())
    findings = check_paths(paths, repo_root.resolve())
    print(f"Checked math conventions in {len(paths)} Markdown files.")
    return findings


def main() -> int:
    """Lint all documentation Markdown, or only the paths given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="specific files to check (defaults to the docset)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    paths = args.paths or _docset_paths(repo_root)
    return report(check_paths(paths, repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
