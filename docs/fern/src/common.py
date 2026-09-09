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

"""Shared constants and Markdown helpers for documentation checks."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
FERN_ROOT = SRC_ROOT.parent
DOCS_ROOT = FERN_ROOT.parent
REPO_ROOT = DOCS_ROOT.parent
DEFAULT_SITE_ROOT = DOCS_ROOT / ".build" / "site"
SITE_PREFIX = "/bionemo/inference-runtime"
LATEST_VERSION_SLUG = "latest"
GITHUB_REPOSITORY = "github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime"
# Shared media for every canonical page. The generated Fern tree mirrors the
# docs directory, so this directory is copied verbatim and links into it are
# left alone: one relative path works in a repository checkout and on the site.
ASSETS_DIR = Path("assets")
PAGE_ROUTES = {
    Path("coding.md"): "references/coding",
    Path("CODE_OF_CONDUCT.md"): "community/code-of-conduct",
    Path("contributing.md"): "community/contributing",
    Path("dev.md"): "references/dev",
    Path("fern/pages/overview.mdx"): "overview",
    Path("install.md"): "install",
    Path("quickstart.md"): "quickstart",
    Path("ray.md"): "references/ray",
    Path("ref/api.md"): "references/api",
    Path("ref/architecture.md"): "references/architecture",
    Path("ref/benchmark.md"): "references/benchmark",
    Path("ref/config.md"): "references/config",
    Path("ref/docker-images.md"): "references/docker-images",
    Path("ref/gpu-stack.md"): "references/gpu-stack",
    Path("ref/model-weights.md"): "references/model-weights",
    Path("ref/support-matrix.md"): "references/support-matrix",
    Path("ref/system-information.md"): "references/system-information",
    Path("SECURITY.md"): "community/security",
}
# Trees omitted from the public GitHub subset. Canonical pages must not link
# into them: generated Fern snapshots would otherwise mint GitHub URLs for
# files that are not published.
UNPUBLISHED_PATH_PREFIXES = (
    ("docs", "nv"),
    (".gitlab",),
    # The other agent playbooks publish; hiding-kernel names private kernels.
    (".agents", "skills", "hiding-kernel"),
    ("tests", "internal"),
    ("cpp", "tools"),
)
UNPUBLISHED_FILES = frozenset({".gitlab-ci.yml"})
LINK_RE = re.compile(
    r"(?P<prefix>\[(?:[^]\[]|\[[^]]*])*]\(\s*)"
    r"(?P<target><[^>]+>|[^)\s]+)"
    r"(?P<suffix>\s*(?:\"[^\"]*\")?\))"
)
IMAGE_RE = re.compile(r"!\[(?:[^]\[]|\[[^]]*])*]\(\s*(?P<target><[^>]+>|[^)\s]+)")
SRC_RE = re.compile(r'\bsrc="(?P<target>[^"]+)"')
# Footnote definitions contain prose after the colon, not a link target.
REFERENCE_RE = re.compile(r"^(?P<prefix>\s*\[(?!\^)[^]]+]:\s*)(?P<target>\S+)")
HREF_RE = re.compile(r'(?P<prefix>\bhref=")(?P<target>[^"]+)(?P<suffix>")')
FENCE_RE = re.compile(r"\s*(`{3,}|~{3,})")
EXTERNAL_RE = re.compile(r"^(?:https?:|mailto:|tel:|ftp:)")
INTERNAL_GITHUB_RE = re.compile(rf"^https://{re.escape(GITHUB_REPOSITORY)}/(?:blob|tree)/main/")
ALLOWED_INTERNAL_GITHUB_LINKS = frozenset(
    {
        f"https://{GITHUB_REPOSITORY}/blob/main/.agents/skills/module-onboard/SKILL.md",
    }
)
GITHUB_CONTENT_RE = re.compile(rf"^https://{re.escape(GITHUB_REPOSITORY)}/(?:blob|tree)/[^/]+/(?P<relative>[^?#]+)")


class Finding:
    """Represent one documentation validation failure."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        """Render the finding in compiler-style location format."""
        location = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"{location}: {self.message}"


def unpublished_posix(relative: Path) -> str | None:
    posix = relative.as_posix()
    if posix in UNPUBLISHED_FILES:
        return posix
    parts = relative.parts
    if any(parts[: len(prefix)] == prefix for prefix in UNPUBLISHED_PATH_PREFIXES):
        return posix
    return None


def unpublished_relative(resolved: Path, repository_root: Path) -> str | None:
    try:
        relative = resolved.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return None
    return unpublished_posix(relative)


def unpublished_github_path(target: str) -> str | None:
    match = GITHUB_CONTENT_RE.match(target)
    if match is None:
        return None
    return unpublished_posix(Path(match.group("relative")))


def is_asset(resolved: Path, docs_root: Path) -> bool:
    """Report whether a resolved path lives in the shared assets directory."""
    try:
        relative = resolved.resolve().relative_to(docs_root.resolve())
    except ValueError:
        return False
    return relative.parts[: len(ASSETS_DIR.parts)] == ASSETS_DIR.parts


def image_targets(line: str) -> list[str]:
    """Collect Markdown image and HTML media targets from one non-code line."""
    targets = [match.group("target").strip("<>") for match in IMAGE_RE.finditer(line)]
    targets.extend(match.group("target") for match in SRC_RE.finditer(line))
    return targets


def source_paths(root: Path) -> tuple[Path, Path]:
    docs = root / "docs"
    fern = docs / "fern"
    required = (
        fern / "docs.yml",
        fern / "index.yml",
        fern / "fern.config.json",
        *[docs / relative for relative in PAGE_ROUTES],
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"documentation source is incomplete: {', '.join(missing)}")
    return docs, fern


def iter_source_lines(text: str, *, keep_eol: bool = False) -> Iterable[tuple[int, str, bool]]:
    fence: str | None = None
    for number, line in enumerate(text.splitlines(keepends=keep_eol), 1):
        marker = FENCE_RE.match(line)
        if marker:
            character = marker.group(1)[0]
            fence = None if fence == character else character if fence is None else fence
            yield number, line, True
            continue
        yield number, line, fence is not None


def link_targets(line: str) -> list[str]:
    targets = [match.group("target").strip("<>") for match in LINK_RE.finditer(line)]
    reference = REFERENCE_RE.match(line)
    if reference:
        targets.append(reference.group("target").strip("<>"))
    targets.extend(match.group("target") for match in HREF_RE.finditer(line))
    return targets


def rewrite_link_line(line: str, replace_target: Callable[[re.Match[str]], str]) -> str:
    rewritten = LINK_RE.sub(replace_target, line)
    rewritten = HREF_RE.sub(replace_target, rewritten)
    return REFERENCE_RE.sub(replace_target, rewritten)


def report(findings: list[Finding]) -> int:
    for finding in findings:
        print(f"error: {finding}", file=sys.stderr)
    return 1 if findings else 0
