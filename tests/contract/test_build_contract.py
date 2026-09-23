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
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import setuptools

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "cpp" / "kernels"
FAMILIES = tuple(
    path.parent.name.removeprefix("cutedsl_") for path in sorted(KERNELS_DIR.glob("cutedsl_*/launcher.cpp"))
)


def _setup_metadata(monkeypatch: pytest.MonkeyPatch, **environment: str) -> dict[str, object]:
    """Run setup.py for its metadata, with `setup()` stubbed so nothing is built."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    for name in ("BIOIR_PUBLIC_WHEEL", "BIOIR_VERSION_LOCAL", "CUDA_TAG"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="setup_under_test")
    return captured


def _setup_version(monkeypatch: pytest.MonkeyPatch, **environment: str) -> str:
    """The version setup.py stamps into the wheel."""
    return str(_setup_metadata(monkeypatch, **environment)["version"])


def test_wheel_version_records_the_toolkit_and_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local version is what tells two builds of one base version apart."""
    version = _setup_version(monkeypatch, CUDA_TAG="cu132", BIOIR_VERSION_LOCAL="g1a2b3c4 some/branch")

    assert version.endswith("+cu132.g1a2b3c4.some.branch"), version


def test_public_wheel_drops_the_whole_local_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public index rejects a local version, so BIOIR_PUBLIC_WHEEL drops it.

    Not just the CUDA tag: a leftover commit segment is still a local version,
    and PEP 440 sorts a longer one higher, so half the segment would be worse
    than all of it. `build_wheel.sh` derives the flag from the ref — set on
    `release/*`, where the base version alone identifies the build, and refused
    anywhere else.
    """
    version = _setup_version(monkeypatch, BIOIR_PUBLIC_WHEEL="1", BIOIR_VERSION_LOCAL="g1a2b3c4 some/branch")

    assert "+" not in version, version
    # And it never reaches for nvcc or torch: neither is present on a public
    # build's critical path once the tag it would produce is unused.
    assert version == _setup_version(monkeypatch, BIOIR_PUBLIC_WHEEL="1")


def test_long_description_survives_an_index_rendering_it_out_of_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index renders the description standalone, and the page cannot be amended.

    PyPI strips no YAML frontmatter and resolves no relative path, so the
    license header would open the page as its largest heading and every
    `docs/...` link would 404 — neither repairable without burning a version.
    setup.py rewrites both out, leaving README.md itself relative for a
    checkout and for GitHub.
    """
    description = str(_setup_metadata(monkeypatch, BIOIR_PUBLIC_WHEEL="1")["long_description"])

    assert not description.startswith("---"), description[:200]
    assert "SPDX-FileCopyrightText" not in description
    targets = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", description)
    targets += re.findall(r"^\[[^\]]+\]:[ \t]+(\S+)", description, re.MULTILINE)
    # An index renders the HTML in the description, so a raw <a>/<img> and an
    # autolink reach a reader exactly as a Markdown link does. Both are forms
    # setup.py rewrites nothing in, which is the case worth catching here.
    targets += re.findall(r"<(?:a|img)\b[^>]*(?:href|src)=[\"']([^\"']+)[\"']", description, re.IGNORECASE)
    targets += re.findall(r"<(https?://[^>\s]+)>", description)
    assert targets, "no links found — the rewrite regexes stopped matching"
    # Cleartext is as unfixable as a 404 once the version is out, and a reader
    # of the page has no way to tell it was not meant that way.
    cleartext = [target for target in targets if target.startswith("http://")]
    assert not cleartext, cleartext
    relative = [target for target in targets if not target.startswith(("https://", "#", "mailto:"))]
    assert not relative, relative
    # An image needs the bytes, so it resolves to raw rather than to a blob page.
    assert "https://raw.githubusercontent.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/main/docs/assets/" in description
    # A fenced command that merely names a relative path comes through untouched.
    assert "python examples/folding/run_demo.py --output-dir output" in description


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
