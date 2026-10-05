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

"""Check that release versions contain only their tagged public documentation."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from check_fern_versions import Version, _tag_pages, _write_release_snapshot, _write_versions, sync_site
from common import REPO_ROOT
from release_notes import Note


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.parametrize("legacy", [False, True])
def test_release_snapshot_uses_tagged_public_pages(tmp_path: Path, legacy: bool) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")

    fern = repo / ("docs/fern" if legacy else "fern")
    fern.mkdir(parents=True)
    (repo / "docs").mkdir(exist_ok=True)
    path = "../install.md" if legacy else "../docs/install.md"
    navigation = f"navigation:\n  - page: Installation\n    path: {path}\n    slug: install\n"
    if legacy:
        navigation += "  - page: Overview\n    path: ./pages/overview.mdx\n    slug: overview\n"
        (fern / "pages").mkdir()
        (fern / "pages/overview.mdx").write_text("Legacy overview\n", encoding="utf-8")
    (fern / "index.yml").write_text(navigation, encoding="utf-8")
    (repo / "docs/install.md").write_text("Tagged installation guide\n", encoding="utf-8")
    (repo / "docs/unlisted.md").write_text("unlisted\n", encoding="utf-8")
    (repo / "docs/assets").mkdir()
    (repo / "docs/assets/example.png").write_bytes(b"public asset")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: add release guide")
    _git(repo, "tag", "release/0.2.0")
    (repo / "docs/install.md").write_text("New development guide\n", encoding="utf-8")
    (repo / "docs/release-notes").mkdir()
    (repo / "docs/release-notes/index.md").write_text("# Release Notes\n", encoding="utf-8")

    output = tmp_path / "site/fern"
    notes = {"0.2.0": Note("0.2.0", "## Highlights\n\n- Tagged change.", ("a" * 40,))}
    validated: dict[str, bytes] = {}

    def validate(payloads: Mapping[str, bytes]) -> None:
        assert not output.exists()
        validated.update(payloads)

    _write_release_snapshot(repo, output, "latest", "release/0.2.0", notes, {"0.2.0": "2026-10-02"}, validate)

    assert validated["docs/install.md"] == b"Tagged installation guide\n"
    assert validated["docs/assets/example.png"] == b"public asset"
    assert f"{fern.relative_to(repo)}/index.yml" in validated
    assert "docs/unlisted.md" not in validated

    assert (output / "pages-latest/install.md").read_text() == "Tagged installation guide\n"
    assert (output / "pages-latest/assets/example.png").read_bytes() == b"public asset"
    assert not (output / "pages-latest/unlisted.md").exists()
    navigation = (output / "versions/latest.yml").read_text()
    assert "path: ../pages-latest/install.md" in navigation
    assert "section: Release Notes" in navigation
    assert "path: ../pages-latest/release-notes/0.2.0.md" in navigation
    page = (output / "pages-latest/release-notes/0.2.0.md").read_text()
    assert "title: 0.2.0\n" in page
    assert "Released 2026-10-02." in page
    assert (
        "/bionemo/inference-runtime/latest/release-notes/0.2.0"
        in (output / "pages-latest/release-notes/index.md").read_text()
    )
    if legacy:
        assert (output / "pages-latest/fern/pages/overview.mdx").read_text() == "Legacy overview\n"


def test_release_navigation_rejects_unlisted_page() -> None:
    with pytest.raises(ValueError, match="unmapped page"):
        _tag_pages("navigation:\n  - page: Unlisted\n    path: ../docs/unlisted.md\n", Path("fern"))


def test_historical_page_is_validated_before_snapshot_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "fern").mkdir()
    (repo / "fern/index.yml").write_text(
        "navigation:\n  - page: Installation\n    path: ../docs/install.md\n", encoding="utf-8"
    )
    (repo / "docs/release-notes").mkdir(parents=True)
    (repo / "docs/release-notes/index.md").write_text("# Release Notes\n", encoding="utf-8")
    (repo / "docs/install.md").write_text("password=example\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: add page")
    _git(repo, "tag", "release/0.2.0")

    output = tmp_path / "site/fern"

    def reject(payloads: Mapping[str, bytes]) -> None:
        assert payloads["docs/install.md"] == b"password=example\n"
        raise ValueError("rejected release payload")

    with pytest.raises(ValueError, match="failed outbound validation"):
        _write_release_snapshot(repo, output, "0.2.0", "release/0.2.0", {}, {}, reject)
    assert not (output / "pages-0.2.0").exists()


def test_release_requires_validator(tmp_path: Path) -> None:
    output = tmp_path / "fern"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("previous site")
    with pytest.raises(ValueError, match="require an outbound validator"):
        sync_site(REPO_ROOT, tmp_path, preview=False, release_tags=[("0.2.0", "release/0.2.0")])
    assert marker.read_text() == "previous site"


def test_versions_block_lists_entries_in_order(tmp_path: Path) -> None:
    config = tmp_path / "docs.yml"
    versions = [
        Version("preview", "Preview", "beta"),
        Version("latest", "0.2.0 (latest)", "stable"),
        Version("0.1.0", "0.1.0", "stable"),
    ]
    _write_versions(config, "versions: []\nredirects: []\n", versions)
    text = config.read_text(encoding="utf-8")

    assert text.index("display-name: Preview") < text.index("display-name: 0.2.0 (latest)")
    assert text.index("slug: latest") < text.index("slug: 0.1.0")
    assert "path: ./versions/latest.yml" in text
    assert text.endswith("redirects: []\n")
