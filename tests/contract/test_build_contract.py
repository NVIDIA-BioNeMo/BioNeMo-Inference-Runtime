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
"""Public CUBIN build: ignore rules, CMake consumption, and corpus materialization."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "cpp" / "kernels"
FAMILIES = tuple(
    path.parent.name.removeprefix("cutedsl_") for path in sorted(KERNELS_DIR.glob("cutedsl_*/launcher.cpp"))
)


def test_families_do_not_ship_private_cmake() -> None:
    assert not list(KERNELS_DIR.glob("cutedsl_*/CMakeLists.txt"))


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
    """Copy each family index and the artifacts it names.

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
        (destination / "records").mkdir()
        staged_index = destination / "index.json"
        shutil.copyfile(source_index, staged_index)
        payload = json.loads(source_index.read_text())
        packs = payload.get("packs") if isinstance(payload, dict) else None
        if isinstance(packs, dict):
            for target_arch, record in packs.items():
                if not isinstance(record, dict):
                    continue
                digest = record.get("sha256")
                if not isinstance(digest, str):
                    continue
                relative = PurePosixPath("packs") / f"{family}_{target_arch.replace('_', '')}_{digest}.tar.xz"
                src_pack = source / relative.as_posix()
                if src_pack.is_file():
                    shutil.copyfile(src_pack, destination / relative.as_posix())
        shards = payload.get("shards") if isinstance(payload, dict) else None
        if isinstance(shards, dict):
            for digit, record in shards.items():
                if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                    continue
                name = f"{digit}_{record['sha256']}.json"
                shutil.copyfile(source / "records" / name, destination / "records" / name)
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
        json.dumps({"family": family, "packs": {"sm_80": {"sha256": "a" * 64}}, "shards": {}})
    )

    staged_index = _stage_indexed_corpus(tmp_path / "staged")[0][1]
    assert {path.name for path in (staged_index.parent / "packs").iterdir()} == {indexed_name}


def test_committed_corpus_materializes(tmp_path: Path) -> None:
    """An LFS pointer or a corrupt indexed pack fails here, not at pip install time."""
    from tests.cubin.test_materialize_cubin_payloads import materializer

    indexes = _stage_indexed_corpus(tmp_path / "corpus")
    materializer.materialize(indexes, tmp_path / "materialized")
