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
"""Structure checks in the bench-perf-oss dataset builder, with RCSB stubbed out."""

import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bench-perf-oss" / "dataset" / "rebuild_dataset.py"
)
_spec = importlib.util.spec_from_file_location("bench_perf_oss_rebuild_dataset", SCRIPT)
rebuild = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rebuild)


def _cif(comp_id: str, x: str, revision: str = "1 'Structure model' 1 0 2023-01-01 ?") -> str:
    return "\n".join(
        [
            "data_TEST",
            "loop_",
            "_pdbx_audit_revision_history.ordinal",
            revision,
            "loop_",
            *(f"_atom_site.{key}" for key in ("group_PDB", *rebuild._ATOM_KEYS)),
            f"ATOM A 1 1 {comp_id} CA . {x} 0.000 0.000",
            "",
        ]
    )


MEASURED = _cif("85L", "1.000")
REVISED = _cif("CYS", "1.000", revision="2 'Structure model' 3 0 2026-09-02 ?")


def _manifest(rel: str, text: str) -> dict:
    return {rel: {"sha256": hashlib.sha256(text.encode()).hexdigest(), "atoms_sha256": rebuild.atoms_digest(text)}}


def _serve(monkeypatch, pages: dict[str, bytes]) -> list[str]:
    fetched: list[str] = []

    def http_get(url, *_args, **_kwargs):
        fetched.append(url)
        return pages[url]

    monkeypatch.setattr(rebuild, "http_get", http_get)
    return fetched


def test_coordinate_change_falls_back_to_the_archived_major_version(tmp_path, monkeypatch):
    rel = "ground_truth/7PV5.cif"
    archive = rebuild.WWPDB_VERSIONED.format(mid="pv", pdb="7pv5")
    index = 'href="pdb_00007pv5_xyz_v3-0.cif.gz" href="pdb_00007pv5_xyz_v2-1.cif.gz"'
    fetched = _serve(
        monkeypatch,
        {
            rebuild.RCSB_ENTRY.format(pdb="7PV5"): REVISED.encode(),
            archive: index.encode(),
            archive + "pdb_00007pv5_xyz_v3-0.cif.gz": gzip.compress(REVISED.encode()),
            archive + "pdb_00007pv5_xyz_v2-1.cif.gz": gzip.compress(MEASURED.encode()),
        },
    )

    status, note = rebuild.fetch_structure(tmp_path, rel, 1, _manifest(rel, MEASURED), force=False)

    assert (status, note) == ("archived", "RCSB moved to a new major version; using wwPDB archive v2.1")
    assert (tmp_path / rel).read_text() == MEASURED
    # Newest major first, so the matching older copy is the last fetch.
    assert fetched[-1].endswith("_xyz_v2-1.cif.gz")
    assert rebuild.fetch_structure(tmp_path, rel, 1, _manifest(rel, MEASURED), force=False)[0] == "cached"


def test_no_matching_archive_copy_is_still_a_mismatch(tmp_path, monkeypatch):
    rel = "ground_truth/7PV5.cif"
    archive = rebuild.WWPDB_VERSIONED.format(mid="pv", pdb="7pv5")
    _serve(
        monkeypatch,
        {
            rebuild.RCSB_ENTRY.format(pdb="7PV5"): REVISED.encode(),
            archive: b'href="pdb_00007pv5_xyz_v3-0.cif.gz"',
            archive + "pdb_00007pv5_xyz_v3-0.cif.gz": gzip.compress(REVISED.encode()),
        },
    )

    status, _ = rebuild.fetch_structure(tmp_path, rel, 1, _manifest(rel, MEASURED), force=False)

    assert status == "MISMATCH"
    assert not (tmp_path / rel).exists()


def test_assembly_file_has_no_archive_fallback(tmp_path, monkeypatch):
    rel = "ground_truth/7WR3_A_C.cif"
    fetched = _serve(monkeypatch, {rebuild.RCSB_ASSEMBLY.format(pdb="7WR3", n=1): REVISED.encode()})

    status, _ = rebuild.fetch_structure(tmp_path, rel, 1, _manifest(rel, MEASURED), force=False)

    assert status == "MISMATCH"
    assert fetched == [rebuild.RCSB_ASSEMBLY.format(pdb="7WR3", n=1)]


@pytest.mark.parametrize("text", [MEASURED, _cif("85L", "1.000", revision="1 'Structure model' 1 1 2024-01-01 ?")])
def test_annotation_only_change_is_accepted(tmp_path, monkeypatch, text):
    rel = "ground_truth/7PV5.cif"
    _serve(monkeypatch, {rebuild.RCSB_ENTRY.format(pdb="7PV5"): text.encode()})

    status, _ = rebuild.fetch_structure(tmp_path, rel, 1, _manifest(rel, MEASURED), force=False)

    assert status == ("ok" if text == MEASURED else "revised")
