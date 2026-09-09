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

"""Compose BioIR's generated Fern development documentation.

Canonical Markdown stays at its repository-native paths and keeps relative
links that work in IDEs and Git forges. This module copies only the public
documentation into a generated Fern tree and converts links in that copy to
Fern routes or GitHub URLs.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from common import (
    ASSETS_DIR,
    EXTERNAL_RE,
    GITHUB_REPOSITORY,
    LATEST_VERSION_SLUG,
    PAGE_ROUTES,
    SITE_PREFIX,
    is_asset,
    iter_source_lines,
    rewrite_link_line,
    source_paths,
    unpublished_relative,
)


def _replace_snapshot(source_docs: Path, destination: Path) -> None:
    """Replace a generated snapshot with the public source documents."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative in PAGE_ROUTES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_docs / relative, target)
    assets = source_docs / ASSETS_DIR
    if assets.is_dir():
        shutil.copytree(assets, destination / ASSETS_DIR)


def _iter_content_files(root: Path) -> Iterable[Path]:
    """Yield generated Markdown and MDX files in deterministic order."""
    yield from sorted((*root.rglob("*.md"), *root.rglob("*.mdx")))


def _route_map(source_docs: Path, version: str) -> dict[Path, str]:
    """Map canonical page paths to versioned Fern routes."""
    return {
        (source_docs / relative).resolve(): f"{SITE_PREFIX}/{version}/{route}"
        for relative, route in PAGE_ROUTES.items()
    }


def _github_target(resolved: Path, repository_root: Path, ref: str, original: str) -> str:
    """Return a ref-pinned GitHub URL for a non-published repository target."""
    relative = resolved.relative_to(repository_root.resolve()).as_posix()
    kind = "tree" if original.endswith("/") or resolved.is_dir() else "blob"
    return f"https://{GITHUB_REPOSITORY}/{kind}/{ref}/{relative}"


def _rewrite_target(
    target: str,
    source_path: Path,
    source_docs: Path,
    routes: dict[Path, str],
    ref: str,
) -> str:
    """Rewrite one canonical relative link for a generated Fern snapshot."""
    wrapped = target.startswith("<") and target.endswith(">")
    raw = target[1:-1] if wrapped else target
    if not raw or raw.startswith(("#", "?", "/")) or EXTERNAL_RE.match(raw):
        return target

    path_and_query, separator, fragment = raw.partition("#")
    path_text, query_separator, query = path_and_query.partition("?")
    if not path_text:
        return target

    resolved = (source_path.parent / path_text).resolve()
    if is_asset(resolved, source_docs):
        return target
    rewritten = routes.get(resolved)
    if rewritten is None:
        repository_root = source_docs.parent
        if unpublished_relative(resolved, repository_root):
            return target
        try:
            rewritten = _github_target(resolved, repository_root, ref, path_text)
        except ValueError:
            return target

    if query_separator:
        rewritten = f"{rewritten}?{query}"
    if separator:
        rewritten = f"{rewritten}#{fragment}"
    return f"<{rewritten}>" if wrapped else rewritten


def _rewrite_line(
    line: str,
    source_path: Path,
    source_docs: Path,
    routes: dict[Path, str],
    ref: str,
) -> str:
    """Rewrite supported link syntaxes on one non-code Markdown line."""

    def replace(match: re.Match[str]) -> str:
        """Replace the target captured by a supported link expression."""
        target = _rewrite_target(match.group("target"), source_path, source_docs, routes, ref)
        return f"{match.group('prefix')}{target}{match.groupdict().get('suffix', '')}"

    return rewrite_link_line(line, replace)


def _rewrite_snapshot_links(snapshot: Path, source_docs: Path, version: str, ref: str) -> None:
    """Rewrite links in a generated snapshot without touching canonical files."""
    routes = _route_map(source_docs, version)
    for generated in _iter_content_files(snapshot):
        source_path = source_docs / generated.relative_to(snapshot)
        lines: list[str] = []
        for _, line, in_fence in iter_source_lines(generated.read_text(encoding="utf-8"), keep_eol=True):
            if in_fence:
                lines.append(line)
            else:
                lines.append(_rewrite_line(line, source_path, source_docs, routes, ref))
        generated.write_text("".join(lines), encoding="utf-8")


def _version_navigation(source: Path, destination: Path, source_docs: Path, snapshot_name: str) -> None:
    """Write version navigation that points into a generated snapshot."""

    def replace(match: re.Match[str]) -> str:
        """Replace a source navigation path with its snapshot-relative path."""
        value = match.group("value").strip("'\"")
        resolved = (source.parent / value).resolve()
        relative = resolved.relative_to(source_docs.resolve()).as_posix()
        return f"{match.group('prefix')}../{snapshot_name}/{relative}"

    pattern = re.compile(r"(?P<prefix>^\s*path:\s*)(?P<value>\S+)", re.MULTILINE)
    text = pattern.sub(replace, source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _versions_block(lines: list[str]) -> tuple[int, int]:
    """Return the start and end line indexes of the top-level versions block."""
    start = next((index for index, line in enumerate(lines) if line.rstrip("\n") == "versions:"), -1)
    if start < 0:
        raise ValueError("docs.yml has no top-level versions block")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            end = index
            break
    return start, end


def _write_development_version(path: Path, source_text: str) -> None:
    """Point the generated Fern configuration at its development navigation."""
    text = source_text
    lines = text.splitlines(keepends=True)
    start, end = _versions_block(lines)
    versions = [
        "versions:\n",
        f"  - display-name: {LATEST_VERSION_SLUG}\n",
        "    path: ./versions/dev.yml\n",
        f"    slug: {LATEST_VERSION_SLUG}\n",
        "    availability: beta\n",
        "\n",
    ]
    path.write_text("".join(lines[:start] + versions + lines[end:]), encoding="utf-8")


def sync_development(source_root: Path, site_root: Path) -> None:
    """Sync public development documentation into a generated site checkout."""
    source_docs, source_fern = source_paths(source_root)
    destination_fern = site_root / "docs" / "fern"
    destination_fern.mkdir(parents=True, exist_ok=True)

    snapshot = destination_fern / "pages-dev"
    _replace_snapshot(source_docs, snapshot)
    _rewrite_snapshot_links(snapshot, source_docs, LATEST_VERSION_SLUG, "main")
    _version_navigation(
        source_fern / "index.yml", destination_fern / "versions" / "dev.yml", source_docs, snapshot.name
    )
    shutil.copy2(source_fern / "fern.config.json", destination_fern / "fern.config.json")

    destination_docs = destination_fern / "docs.yml"
    _write_development_version(destination_docs, (source_fern / "docs.yml").read_text(encoding="utf-8"))

    source_index = destination_fern / "index.yml"
    if source_index.exists():
        source_index.unlink()
