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
"""Static contracts for the public CUBIN build."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "cpp" / "kernels"
FAMILIES = tuple(
    path.parent.name.removeprefix("cutedsl_") for path in sorted(KERNELS_DIR.glob("cutedsl_*/launcher.cpp"))
)


def test_setup_materializes_into_build_tree() -> None:
    source = (REPO_ROOT / "setup.py").read_text()
    assert "materialize_cubin_payloads.py" in source
    # The payloads hang off the same root as the CMake tree, which is kept out
    # of `build_temp` so the generated headers a compilation-database entry
    # points at outlive a PEP 660 editable install.
    assert 'build_root / "bioir_cubins"' in source
    assert 'ROOT_DIR / "build" / "cmake"' in source
    assert "-DBIOIR_CUBIN_MATERIALIZED_DIR=" in source
    assert "prepare_cubins" not in source
    assert "ALLOW_STALE" not in source
    assert "family_indexes.append((family, index))" in source
    assert "index.resolve()" not in source
    assert "embedded_cubins.h" not in source


def test_top_level_cmake_owns_all_payload_shards() -> None:
    project = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text()
    kernels = (KERNELS_DIR / "CMakeLists.txt").read_text()
    assert "LANGUAGES CXX ASM" in project
    assert "BIOIR_CUBIN_MATERIALIZED_DIR" in kernels
    assert "materialization.json" in kernels
    assert "foreach(BIOIR_CUBIN_SHARD 0 1 2 3 4 5 6 7 8 9 a b c d e f)" in kernels
    # One loop registers every family, so no family carries its own CMakeLists.
    assert "_registry.cpp" in kernels
    assert not list(KERNELS_DIR.glob("cutedsl_*/CMakeLists.txt"))


@pytest.mark.parametrize("family", FAMILIES)
def test_family_consumes_materialized_registry(family: str) -> None:
    family_dir = KERNELS_DIR / f"cutedsl_{family}"
    launcher = (family_dir / "launcher.cpp").read_text()

    assert f'#include "{family}_registry.h"' in launcher
    assert "embedded::registry()" in launcher
    assert "embedded::kCubins" not in launcher
    assert "embedded::kCubinCount" not in launcher


def test_git_rules_track_only_indexes_and_packs() -> None:
    git = ["git", "-c", f"safe.directory={REPO_ROOT}"]
    probe = subprocess.run(
        [*git, "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("Git ignore rules require checkout metadata")
    family = FAMILIES[0]
    prefix = f"cpp/kernels/cutedsl_{family}/cubins"

    def ignored(path: str) -> bool:
        result = subprocess.run(
            [*git, "check-ignore", "--no-index", "--quiet", path],
            cwd=REPO_ROOT,
            check=False,
        )
        return result.returncode == 0

    assert not ignored(f"{prefix}/index.json")
    assert not ignored(f"{prefix}/packs/{family}_sm80_deadbeef.tar.xz")
    assert ignored(f"{prefix}/.cache/objects/deadbeef.cubin")
    assert ignored(f"{prefix}/embedded_cubins.h")

    attributes = (REPO_ROOT / ".gitattributes").read_text()
    assert "cpp/kernels/cutedsl_*/cubins/packs/*.tar.xz filter=lfs diff=lfs merge=lfs -text" in attributes


def test_sdist_manifest_includes_only_public_build_inputs() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    assert "recursive-include cpp/cmake *.py" in manifest
    assert "recursive-include cpp/kernels index.json *.tar.xz" in manifest
    assert "prune cpp/tools" in manifest


def test_nanobind_is_a_pinned_offline_build_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "nanobind==2.10.2" in pyproject["build-system"]["requires"]

    cmake = (REPO_ROOT / "cpp" / "cmake" / "deps" / "nanobind.cmake").read_text()
    assert 'set(BIOIR_NANOBIND_VERSION "2.10.2")' in cmake
    assert "find_package(nanobind ${BIOIR_NANOBIND_VERSION} EXACT CONFIG REQUIRED)" in cmake
    assert "FetchContent" not in cmake
    assert "github.com" not in cmake

    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()
    assert "pip install nanobind==2.10.2" in dockerfile


def test_docker_context_excludes_raw_and_materialized_cubins() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text()
    assert "**/cubins/*" in dockerignore
    assert "!**/cubins/index.json" in dockerignore
    assert "!**/cubins/packs/" in dockerignore
    assert "!**/cubins/packs/*.tar.xz" in dockerignore
    assert "**/bioir_cubins/" in dockerignore


@pytest.mark.skipif(
    os.environ.get("BIOIR_TEST_CMAKE_INTEGRATION") != "1",
    reason="opt-in CMake integration build",
)
def test_full_cmake_build_consumes_synthetic_materialization(tmp_path: Path) -> None:
    """Compile all launchers against real generated registry shapes."""
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("CMake and a C++ compiler are required")

    from tests.cubin.test_materialize_cubin_payloads import _write_case, materializer

    source = tmp_path / "source"
    indexes = [(family, _write_case(source, family)) for family in FAMILIES]
    materialized = materializer.materialize(indexes, tmp_path / "materialized")
    build = tmp_path / "build"
    library = tmp_path / "lib"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "cpp"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={library}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DBIOIR_CUBIN_MATERIALIZED_DIR={materialized.output_dir}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "8", "--target", "_cutedsl_kernels"],
        check=True,
    )

    extensions = list(library.glob("_cutedsl_kernels*.so"))
    assert len(extensions) == 1
    extension_bytes = extensions[0].read_bytes()
    for family in FAMILIES:
        assert extension_bytes.count(b"\x7fELF" + family.encode()) == 1


def _stage_indexed_corpus(staging: Path) -> list[tuple[str, Path]]:
    """Copy each family index and the packs it names.

    ``packs/`` can also hold leftover builder tarballs: a refresh writes new
    content-addressed names next to the previous LFS objects. This test
    materializes the indexed corpus, not a publish-pruned directory.
    """
    indexes: list[tuple[str, Path]] = []
    for family in FAMILIES:
        source = KERNELS_DIR / f"cutedsl_{family}" / "cubins"
        source_index = source / "index.json"
        destination = staging / family
        (destination / "packs").mkdir(parents=True)
        staged_index = destination / "index.json"
        shutil.copyfile(source_index, staged_index)
        payload = json.loads(source_index.read_text())
        packs = payload.get("packs") if isinstance(payload, dict) else None
        if isinstance(packs, dict):
            for record in packs.values():
                if not isinstance(record, dict):
                    continue
                relative = PurePosixPath(str(record.get("file", "")))
                if len(relative.parts) != 2 or relative.parts[0] != "packs":
                    continue
                src_pack = source / relative.as_posix()
                if src_pack.is_file():
                    shutil.copyfile(src_pack, destination / relative.as_posix())
        indexes.append((family, staged_index))
    return indexes


def test_stage_indexed_corpus_ignores_unreferenced_builder_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild may leave previous packs beside the indexed ones; only copy those."""
    family = "dual_gemm_x_x"
    monkeypatch.setattr("tests.contract.test_build_contract.KERNELS_DIR", tmp_path)
    monkeypatch.setattr("tests.contract.test_build_contract.FAMILIES", (family,))
    cubins = tmp_path / f"cutedsl_{family}" / "cubins"
    packs = cubins / "packs"
    packs.mkdir(parents=True)
    indexed_name = f"{family}_sm80_{'a' * 64}.tar.xz"
    (packs / indexed_name).write_bytes(b"indexed")
    (packs / f"{family}_sm80_{'b' * 64}.tar.xz").write_bytes(b"builder leftover")
    (cubins / "index.json").write_text(
        json.dumps({"family": family, "packs": {"sm_80": {"file": f"packs/{indexed_name}"}}})
    )

    staged_index = _stage_indexed_corpus(tmp_path / "staged")[0][1]
    assert {path.name for path in (staged_index.parent / "packs").iterdir()} == {indexed_name}


def test_committed_corpus_materializes(tmp_path: Path) -> None:
    """An LFS pointer or a corrupt indexed pack fails here, not at pip install time."""
    from tests.cubin.test_materialize_cubin_payloads import materializer

    indexes = _stage_indexed_corpus(tmp_path / "corpus")
    materializer.materialize(indexes, tmp_path / "materialized")
