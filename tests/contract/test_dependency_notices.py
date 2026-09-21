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
"""Dependency-notice contract tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR_PATH = REPO_ROOT / ".gitlab/ci/scripts/validate_deps.py"
_SUBMODULES = (
    ("OpenFold3", "3rdparty/openfold-3", "https://github.com/aqlaboratory/openfold-3.git"),
    ("Protenix", "3rdparty/protenix", "https://github.com/bytedance/Protenix.git"),
)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bioir_test_validate_deps", _VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write_lock(repo: Path, dependencies: dict[str, str]) -> None:
    records = [
        "version = 1",
        "",
        "[[package]]",
        'name = "bionemo-ir"',
        "dependencies = [",
        *(f'    {{ name = "{name}" }},' for name in dependencies),
        "]",
    ]
    for name, version in dependencies.items():
        records.extend(["", "[[package]]", f'name = "{name}"', f'version = "{version}"'])
    (repo / "uv.lock").write_text("\n".join(records) + "\n")


def _write_repo(repo: Path) -> dict[str, str]:
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "BioIR Tests")
    revisions = {}
    for name, _, _ in _SUBMODULES:
        _git(repo, "commit", "--allow-empty", "-m", f"{name} fixture")
        revisions[name] = _git_output(repo, "rev-parse", "HEAD")

    gitmodules = []
    notices = [
        "# Third-Party Notices",
        "",
        "## Runtime Dependencies",
        "",
        "- **torch**",
        "  - License: MIT; see [MIT][mit]",
        "",
        "## NVIDIA-Licensed Runtime Dependencies",
        "",
    ]
    notices.extend(["## Reference Submodules", ""])
    links = []
    for name, path, url in _SUBMODULES:
        revision = revisions[name]
        gitmodules.extend([f'[submodule "{path}"]', f"\tpath = {path}", f"\turl = {url}"])
        notices.extend(
            [
                f"- **{name}**",
                f"  - Commit: `{revision}`",
                f"  - Source and license: [{name} commit][{name.lower()}]",
                "",
            ]
        )
        links.append(f"[{name.lower()}]: {url.removesuffix('.git')}/tree/{revision}")

    (repo / ".gitmodules").write_text("\n".join(gitmodules) + "\n")
    (repo / "THIRD_PARTY_NOTICES.md").write_text("\n".join([*notices, "[mit]: LICENSES/MIT.txt", *links]) + "\n")
    _write_lock(repo, {"torch": "2.12.0"})
    (repo / "LICENSES").mkdir()
    (repo / "LICENSES/MIT.txt").write_text("MIT fixture\n")

    _git(repo, "add", ".gitmodules", "THIRD_PARTY_NOTICES.md", "uv.lock", "LICENSES/MIT.txt")
    for name, path, _ in _SUBMODULES:
        revision = revisions[name]
        _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{revision},{path}")
    _git(repo, "commit", "-m", "test fixture")
    return revisions


@pytest.fixture
def notice_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    return tmp_path, _write_repo(tmp_path)


def test_valid_notices(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    assert validator.validate(repo) == []


def test_rejects_missing_lock(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    (repo / "uv.lock").unlink()

    assert "uv.lock: file is required" in validator.validate(repo)


@pytest.mark.parametrize("name,path,url", _SUBMODULES)
def test_rejects_stale_revision(notice_repo: tuple[Path, dict[str, str]], name: str, path: str, url: str) -> None:
    repo, revisions = notice_repo
    del path, url
    revision = revisions[name]
    notices = repo / "THIRD_PARTY_NOTICES.md"
    notices.write_text(notices.read_text().replace(f"`{revision}`", f"`{'f' * 40}`", 1))

    assert any(f"{name} revision" in error for error in validator.validate(repo))


@pytest.mark.parametrize("name,path,url", _SUBMODULES)
def test_rejects_stale_source_url(notice_repo: tuple[Path, dict[str, str]], name: str, path: str, url: str) -> None:
    repo, revisions = notice_repo
    del path
    revision = revisions[name]
    notices = repo / "THIRD_PARTY_NOTICES.md"
    expected = f"{url.removesuffix('.git')}/tree/{revision}"
    notices.write_text(notices.read_text().replace(expected, "https://example.com/wrong-source", 1))

    assert f"THIRD_PARTY_NOTICES.md: {name} source must be {expected}" in validator.validate(repo)


def test_rejects_new_dependency(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    _write_lock(repo, {"torch": "2.12.0", "new-dep": "1.0.0"})

    assert "THIRD_PARTY_NOTICES.md: missing runtime dependency new-dep" in validator.validate(repo)


def test_rejects_unknown_dependency(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    _write_lock(repo, {})

    assert "THIRD_PARTY_NOTICES.md: unknown runtime dependency torch" in validator.validate(repo)


def test_rejects_missing_license_declaration(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    notices = repo / "THIRD_PARTY_NOTICES.md"
    notices.write_text(notices.read_text().replace("  - License: MIT; see [MIT][mit]\n", ""))

    assert "THIRD_PARTY_NOTICES.md: missing license for torch" in validator.validate(repo)


def test_rejects_missing_license(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    (repo / "LICENSES/MIT.txt").unlink()

    assert "THIRD_PARTY_NOTICES.md: missing license file LICENSES/MIT.txt" in validator.validate(repo)


@pytest.mark.parametrize("target", ["https://example.com/LICENSE", "docs/LICENSE.txt", "LICENSES/MIT"])
def test_rejects_invalid_license_target(notice_repo: tuple[Path, dict[str, str]], target: str) -> None:
    repo, _ = notice_repo
    notices = repo / "THIRD_PARTY_NOTICES.md"
    notices.write_text(notices.read_text().replace("[mit]: LICENSES/MIT.txt", f"[mit]: {target}"))

    assert f"THIRD_PARTY_NOTICES.md: invalid license path {target}" in validator.validate(repo)


def test_rejects_missing_license_link(notice_repo: tuple[Path, dict[str, str]]) -> None:
    repo, _ = notice_repo
    notices = repo / "THIRD_PARTY_NOTICES.md"
    notices.write_text(notices.read_text().replace("[mit]: LICENSES/MIT.txt\n", ""))

    assert "THIRD_PARTY_NOTICES.md: missing license link [mit]" in validator.validate(repo)
