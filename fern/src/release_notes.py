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

"""Validate the hand-written changelog and map its citations to public history.

``CHANGELOG.md`` holds one ``## X.Y.Z (YYYY-MM-DD)`` section per final release,
newest first. The documentation build splits it into one page per release.

Notes may cite public GitHub commits. Copybara rewrites commit IDs on the way
to GitHub and mirrors only ``main``, so an internal release tag and its GitHub
counterpart are different commits. The ``GitOrigin-RevId`` trailer on every
public commit names the internal commit it came from; this module uses it to
prove each citation shipped in the release that claims it, and to name the
public commit a GitHub release tag must point at.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from common import REPO_ROOT, UNPUBLISHED_FILES, UNPUBLISHED_PATH_PREFIXES, Finding, report

CHANGELOG = Path("CHANGELOG.md")
NOTES_DIR = Path("release-notes")
INDEX_NAME = "index.md"
MAX_CHANGELOG_BYTES = 256 * 1024
# Only final releases are documented; candidates and development builds are not.
FINAL_TAG = re.compile(r"^release/(?P<version>\d+\.\d+\.\d+)$")
FINAL_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
PUBLIC_REPOSITORY = "https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime"
PUBLIC_REMOTE_PREFIX = "refs/remotes/public-github"
PUBLIC_MAIN = f"{PUBLIC_REMOTE_PREFIX}/main"
PUBLIC_TAGS = f"{PUBLIC_REMOTE_PREFIX}/tags"
# GitHub tags published before the mirror mapped them from internal tags.
# v0.1.0 points at a standalone snapshot commit outside public main.
PUBLIC_TAG_EXCEPTIONS = {"0.1.0": "a768bda05e51855b70193e70310ad80a312e942c"}
CHANGELOG_HEADER = (
    "---\n"
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "{}\n"
    "---\n\n"
    "# Changelog\n"
)
# The date is the PyPI upload date; without one, the release tag date is used.
RELEASE_HEADING = re.compile(r"## (?P<version>\d+\.\d+\.\d+)(?: \((?P<date>\d{4}-\d{2}-\d{2})\))?")
MARKDOWN_LINK = re.compile(r"\]\((https?://[^\s)]+)\)")
PUBLIC_COMMIT_URL = re.compile(rf"{re.escape(PUBLIC_REPOSITORY)}/commit/(?P<sha>[0-9a-f]{{40}})")
RAW_HTML = re.compile(r"(?is)<!--|<![A-Za-z][^>]*>|<\s*/?\s*[A-Za-z][^>]*>")
UNSAFE_SCHEME = re.compile(r"(?i)\b(?:javascript|data|vbscript|file)\s*:")
CODE_FENCE = re.compile(r"(?m)^[ \t]*(?:`{3,}|~{3,})")
SENSITIVE_CONTENT = {
    "credential-like text": re.compile(
        r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
        r"\b(?:glpat-|gh[pousr]_|sk-)[A-Za-z0-9_-]{8,}|"
        r"\b(?:password|secret|api[_-]?key|access[_-]?token|authorization)\s*[:=]|"
        r"\bBearer\s+\S+|https?://[^\s/@]+:[^\s/@]+@"
    ),
    "kernel details": re.compile(r"(?i)\b(?:kernels?|cubin|cutedsl|ptx|sass|sm\d{2,3})\b"),
    "private references": re.compile(
        "|".join(
            (
                r"gitlab[-a-z0-9]*\.nvidia\.com",
                r"atlassian\.net",
                r"\[[A-Z][A-Z0-9]*-\d+\]",
                *(re.escape("/".join(parts) + "/") for parts in UNPUBLISHED_PATH_PREFIXES),
                *(re.escape(path) for path in UNPUBLISHED_FILES),
            )
        ),
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Note:
    """One release's validated section of the changelog."""

    version: str
    # Section body with its subsections promoted to page-level headings.
    sections: str
    commits: tuple[str, ...]
    # Release date stated in the heading.
    released: str | None = None


class ChangelogError(ValueError):
    """A changelog problem at a specific line."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"{CHANGELOG}:{line}: {message}")
        self.line = line
        self.message = message


def version_key(version: str) -> tuple[int, ...]:
    """Order final release versions numerically."""
    if FINAL_VERSION.fullmatch(version) is None:
        raise ValueError(f"unsupported release version: {version!r}")
    return tuple(int(part) for part in version.split("."))


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a read-only Git command."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def final_release_tags(repo: Path) -> list[tuple[str, str]]:
    """Return local final release tags, newest first; candidates are skipped."""
    names = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/tags/release/").stdout
    releases = []
    for tag in names.splitlines():
        match = FINAL_TAG.fullmatch(tag)
        if match is not None:
            releases.append((match.group("version"), tag))
    return sorted(releases, key=lambda release: version_key(release[0]), reverse=True)


def tag_date(repo: Path, tag: str) -> str | None:
    """Return a tag's creation date, or None when it is not tagged locally."""
    created = _git(repo, "for-each-ref", "--format=%(creatordate:short)", f"refs/tags/{tag}").stdout.strip()
    return created or None


def _parse_section(version: str, line: int, body: str) -> tuple[str, tuple[str, ...]]:
    """Validate one release section and return its page body and citations."""
    body = body.strip()
    if not body:
        raise ChangelogError(line, f"release {version} has no content")
    if re.search(r"(?m)^(?:#|##|#{4,6}) ", body):
        raise ChangelogError(line, f"release {version} may use only '###' subsections")
    decoded = html.unescape(body)
    if RAW_HTML.search(decoded) or UNSAFE_SCHEME.search(decoded) or CODE_FENCE.search(decoded):
        raise ChangelogError(line, f"release {version} contains raw HTML, an unsafe link scheme, or a code fence")
    for category, pattern in SENSITIVE_CONTENT.items():
        if pattern.search(decoded):
            raise ChangelogError(line, f"release {version} contains {category}")
    if any("/commit/" in link and PUBLIC_COMMIT_URL.fullmatch(link) is None for link in MARKDOWN_LINK.findall(body)):
        raise ChangelogError(line, f"release {version} has a commit link that is not a full public GitHub commit URL")
    commits = tuple(dict.fromkeys(match.group("sha") for match in PUBLIC_COMMIT_URL.finditer(body)))
    return re.sub(r"(?m)^### ", "## ", body), commits


def parse_changelog(source: str) -> dict[str, Note]:
    """Split the changelog into validated release sections, newest first."""
    if not source.startswith(CHANGELOG_HEADER):
        raise ChangelogError(1, "needs the SPDX front matter followed by '# Changelog'")
    offset = CHANGELOG_HEADER.count("\n")
    headings: list[tuple[int, str, str | None, list[str]]] = []
    for number, text in enumerate(source.splitlines()[offset:], start=offset + 1):
        if text.startswith("## "):
            match = RELEASE_HEADING.fullmatch(text)
            if match is None:
                raise ChangelogError(number, "release headings are '## X.Y.Z (YYYY-MM-DD)' or '## X.Y.Z'")
            headings.append((number, match.group("version"), match.group("date"), []))
        elif headings:
            headings[-1][3].append(text)
        elif text.startswith("#"):
            raise ChangelogError(number, "put headings inside a '## X.Y.Z' release section")
    notes: dict[str, Note] = {}
    for number, version, released, lines in headings:
        if version in notes:
            raise ChangelogError(number, f"release {version} appears twice")
        if notes and version_key(version) > version_key(next(reversed(notes))):
            raise ChangelogError(number, "list releases newest first")
        sections, commits = _parse_section(version, number, "\n".join(lines))
        notes[version] = Note(version, sections, commits, released)
    return notes


def load_notes(source_root: Path) -> dict[str, Note]:
    """Load the changelog of a source tree, newest release first."""
    path = source_root / CHANGELOG
    if path.is_symlink() or not path.is_file():
        raise ChangelogError(1, "must be a regular file")
    if path.stat().st_size > MAX_CHANGELOG_BYTES:
        raise ChangelogError(1, f"exceeds {MAX_CHANGELOG_BYTES} bytes")
    return parse_changelog(path.read_text(encoding="utf-8"))


def render_page(note: Note, released: str | None) -> str:
    """Render one note as a generated Fern page."""
    status = f"Released {released}." if released else "Unreleased. This note describes changes under review."
    return (
        "---\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"title: {note.version}\n"
        "---\n\n"
        f"{status}\n\n"
        f"{note.sections}\n"
    )


def check(source_root: Path, releases: list[tuple[str, str]] | None = None) -> list[Finding]:
    """Validate the changelog and require a section for each final release tag."""
    path = source_root / CHANGELOG
    try:
        notes = load_notes(source_root)
    except ChangelogError as error:
        return [Finding(path, error.line, error.message)]
    releases = final_release_tags(source_root) if releases is None else releases
    return [
        Finding(path, 1, f"final release tag {tag} has no '## {version}' section")
        for version, tag in releases
        if version not in notes
    ]


def _origins(repo: Path, ref: str) -> list[tuple[str, str]]:
    """Return (public SHA, internal origin SHA) along a public ref, newest first."""
    output = _git(
        repo,
        "log",
        "--first-parent",
        "--format=%H%x09%(trailers:key=GitOrigin-RevId,valueonly,separator=)",
        ref,
    ).stdout
    pairs = []
    for line in output.splitlines():
        sha, _, origin = line.partition("\t")
        if origin.strip():
            pairs.append((sha, origin.strip()))
    return pairs


def _origin(repo: Path, sha: str) -> str | None:
    """Return the internal commit a public commit was mirrored from."""
    result = _git(repo, "log", "-1", "--format=%(trailers:key=GitOrigin-RevId,valueonly,separator=)", sha, check=False)
    if result.returncode:
        return None
    return result.stdout.strip() or None


def _ancestors(repo: Path, tag: str) -> set[str]:
    """Return every internal commit a release tag contains."""
    return set(_git(repo, "rev-list", tag).stdout.split())


def public_release_commit(repo: Path, tag: str, contained: set[str] | None = None) -> str | None:
    """Return the newest public main commit whose source shipped in an internal tag.

    That commit is what a GitHub release tag must point at. Commits made on an
    internal release branch are not mirrored, so the public commit can trail
    the internal tag.
    """
    contained = _ancestors(repo, tag) if contained is None else contained
    return next((sha for sha, origin in _origins(repo, PUBLIC_MAIN) if origin in contained), None)


def verify_public_commits(repo: Path, notes: dict[str, Note]) -> list[str]:
    """Prove each cited public commit shipped in its release, and check GitHub tags."""
    errors: list[str] = []
    releases = final_release_tags(repo)
    public_refs = _git(repo, "for-each-ref", "--format=%(refname)", PUBLIC_REMOTE_PREFIX).stdout.split()
    if PUBLIC_MAIN not in public_refs:
        return [f"public history is not fetched; expected {PUBLIC_MAIN}"]
    for index, (version, tag) in enumerate(releases):
        contained = _ancestors(repo, tag)
        previous = _ancestors(repo, releases[index + 1][1]) if index + 1 < len(releases) else set()
        for sha in notes[version].commits if version in notes else ():
            if not _git(repo, "for-each-ref", "--contains", sha, PUBLIC_REMOTE_PREFIX, check=False).stdout.strip():
                errors.append(f"{version}: {sha} is not in public GitHub history")
                continue
            origin = _origin(repo, sha)
            if origin is None:
                if PUBLIC_TAG_EXCEPTIONS.get(version) != sha:
                    errors.append(f"{version}: {sha} has no GitOrigin-RevId trailer")
            elif origin not in contained:
                errors.append(f"{version}: {sha} mirrors {origin}, which is not in {tag}")
            elif origin in previous:
                errors.append(f"{version}: {sha} mirrors {origin}, which already shipped in {releases[index + 1][1]}")
        public_tag = f"{PUBLIC_TAGS}/v{version}"
        if public_tag not in public_refs:
            continue
        actual = _git(repo, "rev-parse", f"{public_tag}^{{commit}}").stdout.strip()
        expected = PUBLIC_TAG_EXCEPTIONS.get(version) or public_release_commit(repo, tag, contained)
        if actual != expected:
            errors.append(f"GitHub tag v{version} points at {actual}; the mirror of {tag} is {expected}")
    return errors


def fetch_public_history(repo: Path) -> None:
    """Fetch public GitHub main and tags under refs/remotes/public-github."""
    _git(
        repo,
        "fetch",
        "--quiet",
        "--filter=blob:none",
        "--no-tags",
        f"{PUBLIC_REPOSITORY}.git",
        f"+refs/heads/main:{PUBLIC_MAIN}",
        f"+refs/tags/*:{PUBLIC_TAGS}/*",
    )


def main() -> int:
    """Run the public-history verification or print a GitHub tag target."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--fetch", action="store_true", help="fetch public GitHub history first")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="check every citation and GitHub release tag against the mirror")
    target = commands.add_parser("public-commit", help="print the public commit to tag on GitHub")
    target.add_argument("version")
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.fetch:
            fetch_public_history(repo)
        if args.command == "public-commit":
            if FINAL_VERSION.fullmatch(args.version) is None:
                raise ValueError(f"expected a final version such as 0.2.0, got {args.version!r}")
            sha = public_release_commit(repo, f"release/{args.version}")
            if sha is None:
                raise ValueError(f"no public commit mirrors a commit in release/{args.version}")
            print(sha)
            return 0
        errors = verify_public_commits(repo, load_notes(repo))
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return report([Finding(repo / CHANGELOG, 1, error) for error in errors])


if __name__ == "__main__":
    raise SystemExit(main())
