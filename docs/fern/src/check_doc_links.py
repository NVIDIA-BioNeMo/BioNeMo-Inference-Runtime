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

"""Validate canonical documentation link policy and Fern publication coverage.

Markdown link correctness belongs to rumdl, which resolves relative targets
(MD057) and heading anchors (MD051). This module checks only what publication
adds: navigation coverage, images that survive the snapshot copy, and links
that would mint a URL for a path the public subset never ships.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from common import (
    ALLOWED_INTERNAL_GITHUB_LINKS,
    ASSETS_DIR,
    DOCS_ROOT,
    EXTERNAL_RE,
    INTERNAL_GITHUB_RE,
    PAGE_ROUTES,
    Finding,
    image_targets,
    is_asset,
    iter_source_lines,
    link_targets,
    report,
    unpublished_github_path,
    unpublished_relative,
)


@dataclass(slots=True)
class _Section:
    indent: int
    slug: str
    skips_slug: bool = False


def _slug(value: str) -> str:
    """Convert an unquoted Fern navigation label to its default slug."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip("'\"").lower()).strip("-")


def _navigation_routes(index_yml: Path) -> dict[Path, str]:
    """Resolve page paths and routes declared by the Fern navigation file."""
    section_pattern = re.compile(r"^(?P<indent>\s*)-\s+section:\s*(?P<value>.+?)\s*$")
    page_pattern = re.compile(r"^(?P<indent>\s*)-\s+page:\s*(?P<value>.+?)\s*$")
    property_pattern = re.compile(r"^(?P<indent>\s*)(?P<name>path|slug|skip-slug):\s*(?P<value>\S+)\s*$")
    pages: dict[Path, tuple[str, tuple[_Section, ...]]] = {}
    section_stack: list[_Section] = []
    page_indent = -1
    page_slug = ""
    page_path: str | None = None

    def trim_sections(indent: int) -> None:
        """Restore the section scope for an indentation level."""
        while section_stack and section_stack[-1].indent >= indent:
            section_stack.pop()

    def finish_page() -> None:
        """Add the current page after all of its properties have been read."""
        nonlocal page_indent, page_slug, page_path
        if page_indent < 0:
            return
        if page_path is None:
            raise ValueError(f"navigation page is missing a path in {index_yml}")
        resolved = (index_yml.parent / page_path.strip("'\"")).resolve()
        if resolved in pages:
            raise ValueError(f"duplicate navigation page path in {index_yml}: {page_path}")
        pages[resolved] = (page_slug, tuple(section_stack))
        page_indent = -1
        page_slug = ""
        page_path = None

    for line in index_yml.read_text(encoding="utf-8").splitlines():
        section_match = section_pattern.match(line)
        page_match = page_pattern.match(line)
        property_match = property_pattern.match(line)
        indent = len(line) - len(line.lstrip())

        if page_indent >= 0 and line.strip() and indent <= page_indent:
            finish_page()

        if section_match:
            finish_page()
            section_indent = len(section_match.group("indent"))
            trim_sections(section_indent)
            section_stack.append(_Section(section_indent, _slug(section_match.group("value"))))
            continue
        if page_match:
            finish_page()
            page_indent = len(page_match.group("indent"))
            trim_sections(page_indent)
            page_slug = _slug(page_match.group("value"))
            continue
        if property_match is None:
            continue

        name = property_match.group("name")
        value = property_match.group("value")
        property_indent = len(property_match.group("indent"))
        if page_indent >= 0 and property_indent > page_indent:
            if name == "path":
                page_path = value
            elif name == "slug":
                page_slug = value.strip("'\"")
        else:
            trim_sections(property_indent)
            if section_stack and property_indent > section_stack[-1].indent:
                section = section_stack[-1]
                if name == "slug":
                    section.slug = value.strip("'\"")
                elif name == "skip-slug":
                    section.skips_slug = value.lower() == "true"

    finish_page()
    return {
        path: "/".join(
            segment
            for segment in [
                *(section.slug for section in sections if section.slug and not section.skips_slug),
                page_slug,
            ]
            if segment
        )
        for path, (page_slug, sections) in pages.items()
    }


def check(docs_root: Path, fern_dir: Path) -> list[Finding]:
    index_yml = fern_dir / "index.yml"
    if not index_yml.is_file():
        raise ValueError(f"missing Fern navigation: {index_yml}")

    expected = {(docs_root / relative).resolve() for relative in PAGE_ROUTES}
    navigation = _navigation_routes(index_yml)
    findings: list[Finding] = []
    for path in sorted(expected - navigation.keys()):
        findings.append(Finding(index_yml.relative_to(docs_root), 0, f"public page is missing from navigation: {path}"))
    for path in sorted(navigation.keys() - expected):
        findings.append(Finding(index_yml.relative_to(docs_root), 0, f"unexpected navigation page: {path}"))
    for path in sorted(expected & navigation.keys()):
        relative = path.relative_to(docs_root.resolve())
        configured = PAGE_ROUTES[relative]
        if navigation[path] != configured:
            findings.append(
                Finding(
                    index_yml.relative_to(docs_root),
                    0,
                    f"navigation route for {relative} is {navigation[path]!r}, but link rewriting uses {configured!r}",
                )
            )

    page_text = {path: path.read_text(encoding="utf-8") for path in expected if path.is_file()}
    for missing in sorted(expected - page_text.keys()):
        findings.append(Finding(missing, 0, "public page does not exist"))

    repository_root = docs_root.parent
    for path, text in page_text.items():
        relative = path.relative_to(docs_root)
        for line, content, in_fence in iter_source_lines(text):
            if in_fence:
                continue
            for target in image_targets(content):
                # Only assets survive the snapshot copy; an image anywhere else
                # is rewritten to a GitHub blob URL that no browser can render.
                if EXTERNAL_RE.match(target) or target.startswith("/"):
                    continue
                image = (path.parent / target.partition("#")[0].partition("?")[0]).resolve()
                if not is_asset(image, docs_root):
                    findings.append(Finding(relative, line, f"image must live in docs/{ASSETS_DIR}: {target}"))
            for target in link_targets(content):
                withheld = unpublished_github_path(target)
                if withheld:
                    findings.append(
                        Finding(relative, line, f"unpublished path cannot be linked from public docs: {target}")
                    )
                    continue
                if INTERNAL_GITHUB_RE.match(target) and target not in ALLOWED_INTERNAL_GITHUB_LINKS:
                    findings.append(Finding(relative, line, f"repository link must stay relative: {target}"))
                    continue
                if EXTERNAL_RE.match(target) or target.startswith("?"):
                    continue
                if target.startswith("/"):
                    findings.append(Finding(relative, line, f"site-root link breaks repository navigation: {target}"))
                    continue

                # rumdl owns Markdown link correctness: MD057 resolves relative
                # targets and MD051 resolves heading anchors, in this file and
                # across files. What is left here is publication policy.
                page = target.partition("#")[0].partition("?")[0]
                if not page:
                    continue
                resolved = (path.parent / page).resolve()
                withheld = unpublished_relative(resolved, repository_root)
                if withheld:
                    findings.append(
                        Finding(relative, line, f"unpublished path cannot be linked from public docs: {target}")
                    )

    print(f"Checked {len(page_text)} public pages and {len(navigation)} Fern navigation entries.")
    return findings


def main() -> int:
    """Run canonical documentation link validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=DOCS_ROOT)
    parser.add_argument("--fern-dir", type=Path)
    args = parser.parse_args()
    docs_root = args.docs_root.expanduser().resolve()
    fern_dir = (args.fern_dir or docs_root / "fern").expanduser().resolve()
    try:
        findings = check(docs_root, fern_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return report(findings)


if __name__ == "__main__":
    raise SystemExit(main())
