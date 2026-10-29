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

"""Check release snapshots and explicit updates of the latest documentation."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from check_fern_versions import (
    Version,
    _tag_pages,
    _write_release_notes,
    _write_release_snapshot,
    _write_versions,
    sync_site,
)
from common import PAGE_ROUTES, REPO_ROOT
from release_notes import CHANGELOG_HEADER, Note


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_release_notes_without_date_do_not_claim_unreleased(tmp_path: Path) -> None:
    source_index = tmp_path / "index.md"
    source_index.write_text("# Releases\n", encoding="utf-8")
    notes = {
        "0.2.1": Note("0.2.1", "## Changes\n\n- Update.", ()),
        "0.2.0": Note("0.2.0", "## Changes\n\n- Earlier update.", ()),
    }
    _write_release_notes(tmp_path / "snapshot", notes, {"0.2.0": "2026-10-02"}, "main", source_index)

    directory = tmp_path / "snapshot/release-notes"
    index = (directory / "index.md").read_text(encoding="utf-8")
    undated_page = (directory / "0.2.1.md").read_text(encoding="utf-8")
    dated_page = (directory / "0.2.0.md").read_text(encoding="utf-8")
    assert "unreleased" not in index.lower()
    assert "unreleased" not in undated_page.lower()
    assert "— released 2026-10-02" in index
    assert "Released 2026-10-02." in dated_page


@pytest.fixture
def documentation_repo(tmp_path: Path) -> Path:
    """Create tagged documentation followed by a committed correction."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "fern").mkdir()
    for name in ("index.yml", "docs.yml", "fern.config.json"):
        shutil.copy2(REPO_ROOT / "fern" / name, repo / "fern" / name)
    for relative in PAGE_ROUTES:
        path = repo / "docs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Public page\n", encoding="utf-8")
    (repo / "docs/assets").mkdir()
    (repo / "docs/assets/example.png").write_bytes(b"tagged asset")
    (repo / "CHANGELOG.md").write_text(
        CHANGELOG_HEADER
        + "\n## 0.2.0\n\n- Future release.\n"
        + "\n## 0.1.1 (2026-10-02)\n\n- Second release.\n"
        + "\n## 0.1.0 (2026-09-10)\n\n- First release.\n",
        encoding="utf-8",
    )
    links = "[Quickstart](quickstart.md)\n[Examples](../examples/)\n"
    for version in ("0.1.0", "0.1.1"):
        (repo / "docs/install.md").write_text(f"Guide for {version}\n{links}", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", f"docs: release {version}")
        _git(repo, "tag", f"release/{version}")
    (repo / "docs/install.md").write_text(f"Corrected guide\n{links}", encoding="utf-8")
    (repo / "docs/assets/example.png").write_bytes(b"corrected asset")
    (repo / "docs/unlisted.md").write_text("Unlisted page\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs: correct installation")
    (repo / "docs/install.md").write_text("Uncommitted guide\n", encoding="utf-8")
    return repo


@pytest.mark.parametrize("update_version", [None, "0.1.1"])
def test_production_latest_update_preserves_older_releases(
    tmp_path: Path, documentation_repo: Path, update_version: str | None
) -> None:
    validated: list[dict[str, bytes]] = []

    def validate(payloads: Mapping[str, bytes]) -> None:
        validated.append(dict(payloads))

    sync_site(
        documentation_repo, tmp_path / "site", preview=False, update_version=update_version, validate_payloads=validate
    )
    output = tmp_path / "site/fern"
    config = (output / "docs.yml").read_text()
    assert "display-name: 0.1.1 (latest)" in config
    assert "slug: latest" in config
    assert "Preview" not in config
    assert "slug: main" not in config
    latest = (output / "pages-latest/install.md").read_text()
    expected = "Corrected guide" if update_version else "Guide for 0.1.1"
    github_ref = "main" if update_version else "v0.1.1"
    assert expected in latest
    assert "Uncommitted guide" not in latest
    assert "/bionemo/inference-runtime/latest/quickstart" in latest
    assert f"/tree/{github_ref}/examples)" in latest
    asset = b"corrected asset" if update_version else b"tagged asset"
    assert (output / "pages-latest/assets/example.png").read_bytes() == asset
    older = (output / "pages-0.1.0/install.md").read_text()
    assert "Guide for 0.1.0" in older
    assert "/bionemo/inference-runtime/0.1.0/quickstart" in older
    assert "/tree/v0.1.0/examples)" in older
    assert (output / "pages-0.1.0/assets/example.png").read_bytes() == b"tagged asset"
    assert {path.name for path in (output / "pages-latest/release-notes").iterdir()} == {
        "index.md",
        "0.1.1.md",
        "0.1.0.md",
    }
    assert "0.2.0" not in (output / "versions/latest.yml").read_text()
    assert not (output / "pages-latest/unlisted.md").exists()
    assert expected.encode() in validated[0]["docs/install.md"]
    assert validated[0]["docs/assets/example.png"] == asset
    assert b"Guide for 0.1.0" in validated[1]["docs/install.md"]
    assert all("docs/unlisted.md" not in payloads for payloads in validated)


@pytest.mark.parametrize(
    ("version", "preview", "newer_tag", "message"),
    [
        ("0.1.0", False, False, "can only update the latest final release"),
        ("0.9.9", False, False, "can only update the latest final release"),
        ("0.1.1", True, False, "require production mode"),
        ("0.1.1", False, True, "latest final release \\(0.2.0\\)"),
    ],
)
def test_release_update_rejects_invalid_target_before_replacing_site(
    tmp_path: Path, documentation_repo: Path, version: str, preview: bool, newer_tag: bool, message: str
) -> None:
    if newer_tag:
        _git(documentation_repo, "tag", "release/0.2.0")
    output = tmp_path / "site/fern"
    output.mkdir(parents=True)
    marker = output / "existing.txt"
    marker.write_text("previous site", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        sync_site(documentation_repo, output.parent, preview=preview, update_version=version)
    assert marker.read_text() == "previous site"


def test_current_release_update_requires_outbound_validation(tmp_path: Path, documentation_repo: Path) -> None:
    def reject(payloads: Mapping[str, bytes]) -> None:
        assert b"Corrected guide" in payloads["docs/install.md"]
        raise ValueError("rejected current payload")

    with pytest.raises(ValueError, match="failed outbound validation for HEAD"):
        sync_site(
            documentation_repo, tmp_path / "site", preview=False, update_version="0.1.1", validate_payloads=reject
        )
    assert not (tmp_path / "site/fern/pages-latest").exists()


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
