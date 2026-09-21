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
"""Contracts for uv-managed development and CI workflows."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text()


def test_github_workflows_use_locked_uv_groups() -> None:
    python_action = _read(".github/actions/python/action.yml")
    uv_action = _read(".github/actions/uv/action.yml")
    prek_action = _read(".github/actions/prek/action.yml")
    ci = _read(".github/workflows/ci.yml")
    pr = _read(".github/workflows/pr.yml")
    wheel = _read(".github/workflows/wheel.yml")

    assert "python-version-file: .python-version" in python_action
    assert "tool.uv.required-version" in uv_action
    assert 'python -m pip install "uv==${UV_VERSION}"' in uv_action
    assert "./.github/actions/uv" in prek_action
    assert "hashFiles('uv.lock', 'prek.toml')" in prek_action
    assert "CUDA_TAG: cu130" in ci
    assert "UV_NO_INSTALL_PROJECT: '1'" in ci
    assert "uv run --locked --only-group lint prek run" in ci
    assert "uv run --locked --only-group test pytest" in ci
    assert "CUDA_TAG: cu130" in pr
    assert "UV_NO_INSTALL_PROJECT: '1'" in pr
    assert "uv run --locked --only-group lint" in pr
    assert "scripts/build_wheel.sh --out-dir dist" in wheel
    assert "uv export --quiet --locked --only-group build" in _read("scripts/build_wheel.sh")

    published_workflows = "\n".join(path.read_text() for path in (REPO_ROOT / ".github").rglob("*.yml"))
    assert "requirements-dev.txt" not in published_workflows
    assert ".[dev]" not in published_workflows


def test_gitlab_ci_enforces_lock_freshness_and_uv_pin() -> None:
    benchmark = _read(".gitlab/ci/benchmark.yml")
    pretest = _read(".gitlab/ci/pretest.yml")
    prepare = _read(".gitlab/ci/prepare.yml")
    report = _read(".gitlab/ci/report.yml")
    ci_sources = "\n".join(
        path.read_text() for suffix in ("*.yml", "*.sh") for path in (REPO_ROOT / ".gitlab").rglob(suffix)
    )

    assert 'pip install -q "uv==${BIOIR_UV_VERSION}"' in benchmark
    assert 'pip install -q "uv==${BIOIR_UV_VERSION}"' in pretest
    assert "--frozen" not in ci_sources
    assert 'CUDA_TAG: "cu130"' in prepare
    assert 'UV_NO_INSTALL_PROJECT: "1"' in prepare
    assert "uv run --locked --only-group lint prek run" in prepare
    assert 'CUDA_TAG: "cu130"' in report
    assert 'UV_NO_INSTALL_PROJECT: "1"' in report
    assert "uv run --locked --only-group coverage bash" in report
    assert "uv sync --locked --only-group release --active" in ci_sources


def test_ngc_image_uses_locked_exports_and_separate_sboms() -> None:
    dockerfile = _read("docker/Dockerfile")
    prepare = _read(".gitlab/ci/prepare.yml")
    groups = tomllib.loads(_read("pyproject.toml"))["dependency-groups"]

    assert "COPY pyproject.toml uv.lock requirements.txt" in dockerfile
    assert dockerfile.count("uv export --quiet --locked") == 3
    assert "--prune torch --prune triton" in dockerfile
    assert "--no-deps --require-hashes" in dockerfile
    assert groups["sbom"] == ["cyclonedx-bom==7.4.0"]
    assert "project-dependencies.cdx.json" in prepare
    assert "dev-image-python-environment.cdx.json" in prepare
    assert 'cyclonedx-py" environment "${system_python}"' in prepare


def test_wheel_entrypoints_set_source_date_epoch() -> None:
    assert "SOURCE_DATE_EPOCH" in _read("docker/Dockerfile")
    assert "SOURCE_DATE_EPOCH ?=" in _read("docker/Makefile")
    assert "git show -s --format=%ct" in _read("scripts/build_wheel.sh")
    assert "git show -s --format=%ct" in _read(".gitlab/ci/scripts/build_wheel.sh")
    assert "git show -s --format=%ct" in _read(".github/workflows/wheel.yml")


def test_agent_playbooks_do_not_request_removed_dev_extra() -> None:
    playbooks = (
        ".agents/skills/make-data-pipeline/SKILL.md",
        ".agents/skills/module-onboard/SKILL.md",
        ".agents/skills/bench-perf-oss/SKILL.md",
        ".agents/skills/bench-perf-oss/environment.md",
    )
    content = "\n".join(_read(path) for path in playbooks)
    assert ".[dev]" not in content
    assert "uv sync --locked" in content
    assert "uv pip install --no-deps -e ." in content
