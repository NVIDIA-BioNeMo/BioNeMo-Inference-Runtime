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

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import release_notes
from check_fern_versions import sync_site
from common import REPO_ROOT
from release_notes import (
    CHANGELOG_HEADER,
    PUBLIC_MAIN,
    PUBLIC_REPOSITORY,
    PUBLIC_TAGS,
    final_release_tags,
    load_notes,
    parse_changelog,
    public_release_commit,
    verify_public_commits,
)


def _link(sha: str, text: str = "change") -> str:
    return f"[{text}]({PUBLIC_REPOSITORY}/commit/{sha})"


def _changelog(*sections: str) -> str:
    return CHANGELOG_HEADER + "\nIntro text.\n\n" + "\n".join(sections)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    return path


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _mirror(repo: Path, origins: list[str]) -> list[str]:
    """Build public history whose commits name internal origins, as Copybara does."""
    tree = _git(repo, "hash-object", "-t", "tree", "-w", "--stdin")
    parent: list[str] = []
    public = []
    for origin in origins:
        sha = _git(repo, "commit-tree", tree, *parent, "-m", f"public change\n\nGitOrigin-RevId: {origin}")
        parent = ["-p", sha]
        public.append(sha)
    _git(repo, "update-ref", PUBLIC_MAIN, public[-1])
    return public


def test_parse_changelog_splits_releases_and_promotes_subsections() -> None:
    sha = "a" * 40
    notes = parse_changelog(
        _changelog(
            f"## 0.2.0 (2026-11-03)\n\n### Fixed Issues\n\n- New {_link(sha)}.\n  - Detail.\n",
            "## 0.1.0\n\n### Key Features and Enhancements\n\n- First release without a link.\n",
        )
    )

    assert list(notes) == ["0.2.0", "0.1.0"]
    assert notes["0.2.0"].released == "2026-11-03"
    assert notes["0.2.0"].commits == (sha,)
    assert notes["0.2.0"].sections.startswith("## Fixed Issues")
    assert notes["0.1.0"].released is None
    assert notes["0.1.0"].commits == ()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (f"- [x](https://example.com/commit/{'a' * 40}).\n", "not a full public GitHub commit URL"),
        ("#### Deep\n\n- Detail.\n", "only '###' subsections"),
        ("- Tuned a kernel.\n", "kernel details"),
        ("- password=example\n", "credential-like text"),
        ("- [BNMTRT-1] change\n", "private references"),
        ("- <script>x</script>\n", "raw HTML"),
        ("~~~text\nprivate\n~~~\n", "code fence"),
        ("", "has no content"),
    ],
)
def test_parse_changelog_rejects_unpublishable_content(body: str, message: str) -> None:
    with pytest.raises(release_notes.ChangelogError, match=message):
        parse_changelog(_changelog(f"## 0.2.0\n\n{body}"))


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        (("## 0.2.0rc1\n\n- Candidate.\n",), "release headings are"),
        (("## 0.1.0\n\n- Old.\n", "## 0.2.0\n\n- New.\n"), "newest first"),
        (("## 0.1.0\n\n- One.\n", "## 0.1.0\n\n- Two.\n"), "appears twice"),
    ],
)
def test_parse_changelog_rejects_bad_release_headings(sections: tuple[str, ...], message: str) -> None:
    with pytest.raises(release_notes.ChangelogError, match=message):
        parse_changelog(_changelog(*sections))


def test_parse_changelog_requires_the_header() -> None:
    with pytest.raises(release_notes.ChangelogError, match="# Changelog"):
        parse_changelog("# Changelog\n\n## 0.1.0\n\n- Change.\n")


def test_load_notes_reads_the_repository_changelog() -> None:
    notes = load_notes(REPO_ROOT)

    assert list(notes)[-1] == "0.1.0"
    assert notes["0.1.0"].released == "2026-09-10"


def test_final_release_tags_skip_candidates_and_sort_numerically(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    _commit(repo, "initial")
    for tag in ("release/0.9.0", "release/0.10.0", "release/0.10.1rc1", "release/0.1.0", "v1.0.0"):
        _git(repo, "tag", tag)

    assert final_release_tags(repo) == [
        ("0.10.0", "release/0.10.0"),
        ("0.9.0", "release/0.9.0"),
        ("0.1.0", "release/0.1.0"),
    ]


def test_check_requires_a_section_for_each_final_tag(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(_changelog("## 0.1.0\n\n- First.\n"), encoding="utf-8")

    findings = release_notes.check(tmp_path, releases=[("0.2.0", "release/0.2.0"), ("0.1.0", "release/0.1.0")])

    assert [finding.message for finding in findings] == ["final release tag release/0.2.0 has no '## 0.2.0' section"]


def test_check_reports_the_failing_line(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(_changelog("## 0.1.0\n\n- First.\n", "## next\n"), encoding="utf-8")

    findings = release_notes.check(tmp_path, releases=[])

    assert len(findings) == 1
    assert findings[0].line == 15
    assert "release headings are" in findings[0].message


def test_public_citations_must_ship_in_their_release(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    first = _commit(repo, "first")
    _git(repo, "tag", "release/0.1.0")
    second = _commit(repo, "second")
    third = _commit(repo, "release branch only")
    _git(repo, "tag", "release/0.2.0")
    _git(repo, "reset", "-q", "--hard", second)
    later = _commit(repo, "after the release")
    public_first, public_second, public_later = _mirror(repo, [first, second, later])

    assert public_release_commit(repo, "release/0.2.0") == public_second
    assert third not in {public_first, public_second, public_later}

    def notes(*shas: str) -> dict[str, release_notes.Note]:
        return {"0.2.0": release_notes.Note("0.2.0", "", shas), "0.1.0": release_notes.Note("0.1.0", "", ())}

    assert verify_public_commits(repo, notes(public_second)) == []
    errors = verify_public_commits(repo, notes(public_first, public_later, "f" * 40))
    assert any("already shipped in release/0.1.0" in error for error in errors)
    assert any(f"mirrors {later}, which is not in release/0.2.0" in error for error in errors)
    assert any("is not in public GitHub history" in error for error in errors)

    _git(repo, "update-ref", f"{PUBLIC_TAGS}/v0.2.0", public_later)
    errors = verify_public_commits(repo, notes(public_second))
    assert errors == [f"GitHub tag v0.2.0 points at {public_later}; the mirror of release/0.2.0 is {public_second}"]


def test_preview_site_puts_working_tree_first(tmp_path: Path) -> None:
    sync_site(REPO_ROOT, tmp_path, preview=True, release_tags=[])
    config = (tmp_path / "fern/docs.yml").read_text()

    assert "display-name: Preview" in config
    assert (tmp_path / "fern/pages-preview/release-notes/0.1.1.md").is_file()


def test_production_site_requires_a_final_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no final release tag"):
        sync_site(REPO_ROOT, tmp_path, preview=False, release_tags=[])
