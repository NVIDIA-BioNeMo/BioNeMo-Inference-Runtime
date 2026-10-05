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

"""Compose BioIR's generated Fern documentation from final release tags.

Canonical Markdown stays at its repository-native paths and keeps relative
links that work in IDEs and Git forges. This module copies only the public
documentation into a generated Fern tree and converts links in that copy to
Fern routes or GitHub URLs.
"""

from __future__ import annotations

import posixpath
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from common import (
    ASSETS_DIR,
    EXTERNAL_RE,
    GITHUB_REPOSITORY,
    PAGE_ROUTES,
    SITE_PREFIX,
    is_asset,
    iter_source_lines,
    rewrite_link_line,
    source_paths,
    unpublished_relative,
)
from release_notes import (
    INDEX_NAME,
    NOTES_DIR,
    Note,
    final_release_tags,
    load_notes,
    render_page,
    tag_date,
    version_key,
)

RELEASE_NOTES_NAV_MARKER = "      # bioir:release-note-pages"
PREVIEW_SLUG = "preview"
# The newest final release keeps one stable URL; older releases use their version.
LATEST_SLUG = "latest"


def _replace_snapshot(source_docs: Path, destination: Path, pages: Iterable[Path] = PAGE_ROUTES) -> None:
    """Replace a generated snapshot with the public source documents."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative in pages:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_docs / relative, target)
    assets = source_docs / ASSETS_DIR
    if assets.is_dir():
        shutil.copytree(assets, destination / ASSETS_DIR)


def _iter_content_files(root: Path) -> Iterable[Path]:
    """Yield generated Markdown and MDX files in deterministic order."""
    yield from sorted((*root.rglob("*.md"), *root.rglob("*.mdx")))


def _route_map(source_docs: Path, version: str, page_routes: dict[Path, str] = PAGE_ROUTES) -> dict[Path, str]:
    """Map canonical page paths to versioned Fern routes."""
    return {
        (source_docs / relative).resolve(): f"{SITE_PREFIX}/{version}/{route}"
        for relative, route in page_routes.items()
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


def _rewrite_snapshot_links(
    snapshot: Path, source_docs: Path, version: str, ref: str, page_routes: dict[Path, str] = PAGE_ROUTES
) -> None:
    """Rewrite links in a generated snapshot without touching canonical files."""
    routes = _route_map(source_docs, version, page_routes)
    for generated in _iter_content_files(snapshot):
        source_path = source_docs / generated.relative_to(snapshot)
        lines: list[str] = []
        for _, line, in_fence in iter_source_lines(generated.read_text(encoding="utf-8"), keep_eol=True):
            if in_fence:
                lines.append(line)
            else:
                lines.append(_rewrite_line(line, source_path, source_docs, routes, ref))
        generated.write_text("".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class Version:
    """One entry in the generated Fern version selector."""

    slug: str
    label: str
    availability: str


def _version_navigation(
    source: Path,
    destination: Path,
    source_docs: Path,
    snapshot_name: str,
    note_versions: list[str],
    *,
    require_notes_marker: bool = True,
) -> None:
    """Write version navigation that points into a generated snapshot."""

    def replace(match: re.Match[str]) -> str:
        """Replace a source navigation path with its snapshot-relative path."""
        value = match.group("value").strip("'\"")
        resolved = (source.parent / value).resolve()
        relative = resolved.relative_to(source_docs.resolve()).as_posix()
        return f"{match.group('prefix')}../{snapshot_name}/{relative}"

    pattern = re.compile(r"(?P<prefix>^\s*path:\s*)(?P<value>\S+)", re.MULTILINE)
    text = pattern.sub(replace, source.read_text(encoding="utf-8"))
    marker_count = text.count(RELEASE_NOTES_NAV_MARKER)
    if marker_count != 1 and (require_notes_marker or marker_count):
        raise ValueError("Fern navigation needs one release-note page marker")
    note_pages = "".join(
        f"      - page: {version}\n        path: ../{snapshot_name}/{NOTES_DIR}/{version}.md\n        slug: {version}\n"
        for version in note_versions
    )
    if marker_count:
        text = text.replace(RELEASE_NOTES_NAV_MARKER + "\n", note_pages)
    else:
        # Tags cut before release notes existed get the section appended.
        text += (
            "\n  - section: Release Notes\n"
            f"    slug: {NOTES_DIR}\n"
            "    contents:\n"
            "      - page: All releases\n"
            f"        path: ../{snapshot_name}/{NOTES_DIR}/{INDEX_NAME}\n"
            "        slug: overview\n" + note_pages
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _write_release_notes(
    snapshot: Path, notes: dict[str, Note], dates: dict[str, str | None], slug: str, source_index: Path
) -> None:
    """Add one page per release note and a landing page that lists them."""
    directory = snapshot / NOTES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / INDEX_NAME
    shutil.copy2(source_index, index)
    with index.open("a", encoding="utf-8") as output:
        output.write("\n")
        for version in notes:
            released = dates.get(version)
            suffix = f" — released {released}" if released else " — unreleased"
            output.write(f"- [{version}]({SITE_PREFIX}/{slug}/{NOTES_DIR}/{version}){suffix}\n")
    for version, note in notes.items():
        (directory / f"{version}.md").write_text(render_page(note, dates.get(version)), encoding="utf-8")


def _versions_block(lines: list[str]) -> tuple[int, int]:
    """Return the start and end line indexes of the top-level versions block."""
    start = next((index for index, line in enumerate(lines) if line.startswith("versions:")), -1)
    if start < 0:
        raise ValueError("docs.yml has no top-level versions block")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            end = index
            break
    return start, end


def _write_versions(path: Path, source_text: str, versions: list[Version]) -> None:
    """Replace the source versions block; Fern serves the first entry by default."""
    lines = source_text.splitlines(keepends=True)
    start, end = _versions_block(lines)
    block = ["versions:\n"]
    for version in versions:
        block.extend(
            (
                f"  - display-name: {version.label}\n",
                f"    path: ./versions/{version.slug}.yml\n",
                f"    slug: {version.slug}\n",
                f"    availability: {version.availability}\n",
            )
        )
    block.append("\n")
    path.write_text("".join(lines[:start] + block + lines[end:]), encoding="utf-8")


def _git_bytes(repo: Path, *args: str) -> bytes:
    """Read a Git object without printing repository content on failure."""
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"cannot read release documentation from Git ({args[0]} exited {result.returncode})")
    return result.stdout


def _tag_fern_path(repo: Path, tag: str) -> Path:
    """Find the Fern source directory used by a release tag."""
    for candidate in (Path("fern"), Path("docs/fern")):
        result = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{tag}:{candidate.as_posix()}/index.yml"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    raise ValueError(f"release tag {tag} has no Fern navigation")


def _tag_pages(navigation: str, fern_path: Path) -> dict[Path, str]:
    """Allow only known public pages referenced by a release's navigation."""
    routes = {**PAGE_ROUTES, Path("fern/pages/overview.mdx"): "overview"}
    selected: dict[Path, str] = {}
    for match in re.finditer(r"^\s*path:\s*(\S+)\s*$", navigation, re.MULTILINE):
        value = match.group(1).strip("'\"")
        path = posixpath.normpath(posixpath.join(fern_path.as_posix(), value))
        if not path.startswith("docs/"):
            raise ValueError(f"release navigation points outside public docs: {value}")
        relative = Path(path).relative_to("docs")
        if relative not in routes:
            raise ValueError(f"release navigation has an unmapped page: {relative}")
        selected[relative] = routes[relative]
    if not selected:
        raise ValueError("release navigation has no public pages")
    return selected


def _write_release_snapshot(
    repo: Path,
    destination_fern: Path,
    slug: str,
    tag: str,
    notes: dict[str, Note],
    dates: dict[str, str | None],
    validate_payloads: Callable[[Mapping[str, bytes]], None],
) -> None:
    """Compose one immutable public-doc snapshot from a release tag."""
    fern_path = _tag_fern_path(repo, tag)
    navigation_path = fern_path / "index.yml"
    navigation = _git_bytes(repo, "show", f"{tag}:{navigation_path.as_posix()}").decode("utf-8")
    page_routes = _tag_pages(navigation, fern_path)
    with tempfile.TemporaryDirectory(prefix="bioir-release-docs-") as directory:
        source_root = Path(directory)
        source_docs = source_root / "docs"
        source_fern = source_root / fern_path
        payloads: dict[str, bytes] = {navigation_path.as_posix(): navigation.encode("utf-8")}
        for relative in page_routes:
            path = f"docs/{relative.as_posix()}"
            payloads[path] = _git_bytes(repo, "show", f"{tag}:{path}")
        assets = _git_bytes(repo, "ls-tree", "-r", "--name-only", "-z", tag, "--", "docs/assets")
        for raw in assets.split(b"\0"):
            if not raw:
                continue
            path = raw.decode("utf-8")
            payloads[path] = _git_bytes(repo, "show", f"{tag}:{path}")
        try:
            validate_payloads(payloads)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(f"release documentation failed outbound validation for {tag}") from error
        for path, content in payloads.items():
            target = source_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        snapshot = destination_fern / f"pages-{slug}"
        _replace_snapshot(source_docs, snapshot, page_routes)
        _write_release_notes(snapshot, notes, dates, slug, repo / "docs" / NOTES_DIR / INDEX_NAME)
        # GitHub names the mirrored release tag v<version>, not release/<version>.
        github_ref = "v" + tag.removeprefix("release/")
        _rewrite_snapshot_links(snapshot, source_docs, slug, github_ref, page_routes)
        _version_navigation(
            source_fern / "index.yml",
            destination_fern / "versions" / f"{slug}.yml",
            source_docs,
            snapshot.name,
            list(notes),
            require_notes_marker=False,
        )


def sync_site(
    source_root: Path,
    site_root: Path,
    *,
    preview: bool,
    release_tags: list[tuple[str, str]] | None = None,
    validate_payloads: Callable[[Mapping[str, bytes]], None] | None = None,
) -> None:
    """Compose final release snapshots, newest first, behind an optional preview.

    Production publishes only final release tags; the newest is the default
    version under the ``latest`` slug. A preview adds the working-tree
    documentation first so a merge request can review it.
    Release snapshots require a payload validator before writing output.
    """
    source_docs, source_fern = source_paths(source_root)
    destination_fern = site_root / "fern"
    if destination_fern.resolve() == source_fern.resolve():
        raise ValueError("generated Fern site cannot replace its source directory")
    releases = final_release_tags(source_root) if release_tags is None else release_tags
    if not releases and not preview:
        raise ValueError("no final release tag to publish; production docs are built from release/X.Y.Z tags")
    if releases and validate_payloads is None:
        raise ValueError("release snapshots require an outbound validator")
    notes = load_notes(source_root)
    missing = [tag for version, tag in releases if version not in notes]
    if missing:
        raise ValueError(f"CHANGELOG.md has no section for final release tags: {', '.join(missing)}")
    dates = {version: note.released or tag_date(source_root, f"release/{version}") for version, note in notes.items()}
    if destination_fern.exists():
        shutil.rmtree(destination_fern)
    destination_fern.mkdir(parents=True, exist_ok=True)

    versions: list[Version] = []
    if preview:
        snapshot = destination_fern / f"pages-{PREVIEW_SLUG}"
        _replace_snapshot(source_docs, snapshot)
        _write_release_notes(snapshot, notes, dates, PREVIEW_SLUG, source_docs / NOTES_DIR / INDEX_NAME)
        _rewrite_snapshot_links(snapshot, source_docs, PREVIEW_SLUG, "main")
        _version_navigation(
            source_fern / "index.yml",
            destination_fern / "versions" / f"{PREVIEW_SLUG}.yml",
            source_docs,
            snapshot.name,
            list(notes),
        )
        versions.append(Version(PREVIEW_SLUG, "Preview", "beta"))
    if validate_payloads is not None:
        for position, (version, tag) in enumerate(releases):
            slug = LATEST_SLUG if position == 0 else version
            shipped = {key: note for key, note in notes.items() if version_key(key) <= version_key(version)}
            _write_release_snapshot(source_root, destination_fern, slug, tag, shipped, dates, validate_payloads)
            label = f"{version} (latest)" if position == 0 else version
            versions.append(Version(slug, label, "stable"))
    shutil.copy2(source_fern / "fern.config.json", destination_fern / "fern.config.json")
    _write_versions(destination_fern / "docs.yml", (source_fern / "docs.yml").read_text(encoding="utf-8"), versions)

    source_index = destination_fern / "index.yml"
    if source_index.exists():
        source_index.unlink()
