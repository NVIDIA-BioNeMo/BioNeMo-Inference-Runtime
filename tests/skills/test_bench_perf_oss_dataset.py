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
import io
import json
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills" / "bench-perf-oss" / "dataset" / "rebuild_dataset.py"
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


def test_removed_key_argument(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--api-key", "secret-test-value"])
    with pytest.raises(SystemExit) as error:
        rebuild.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "secret-test-value" not in captured.out + captured.err
    assert "unrecognized arguments" in captured.err


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data / "ground_truth").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Dataset path escapes data root"):
        rebuild.fetch_structure(data, "ground_truth/7PV5.cif", None, {}, False)
    assert not (outside / "7PV5.cif").exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://health.api.nvidia.com",
        "https://user:password@example.com",
        "https://",
        "https://example.com/#fragment",
    ],
)
def test_unsafe_service_url_rejected_before_network(monkeypatch, url):
    def unexpected_network(*args, **kwargs):
        pytest.fail("Rejected URL reached the network")

    monkeypatch.setattr(rebuild.urllib.request, "build_opener", unexpected_network)
    with pytest.raises(ValueError, match="HTTPS service URL"):
        rebuild.http_get(url)
    with pytest.raises(ValueError, match="HTTPS service URL"):
        rebuild.MsaClient("test-key", base=url)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_authenticated_redirects_are_refused(code):
    request = urllib.request.Request(rebuild.MSA_BASE, headers={"Authorization": "Bearer test-key"})
    with pytest.raises(urllib.error.HTTPError, match="Redirect refused"):
        rebuild._NoRedirect().redirect_request(request, io.BytesIO(), code, "redirect", {}, "https://other.example/")


def test_msa_authentication_and_async_polling(monkeypatch):
    calls = []

    class Response:
        def __init__(self, status, headers, body):
            self.status, self.headers, self.body = status, headers, body

        def read(self):
            return json.dumps(self.body).encode()

    @contextmanager
    def open_https(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        if len(calls) == 1:
            yield Response(202, {"nvcf-reqid": "test-request"}, {})
        else:
            yield Response(200, {}, {"result": "alignment"})

    monkeypatch.setattr(rebuild, "_open_https", open_https)
    assert rebuild.MsaClient("test-key")._post("/predict", {"sequence": "ACDE"}) == {"result": "alignment"}
    assert calls == [
        (rebuild.MSA_BASE + "/predict", "Bearer test-key", 900),
        (rebuild.NVCF_STATUS.format(reqid="test-request"), "Bearer test-key", 120),
    ]


def test_poll_request_id_cannot_change_request_target():
    with pytest.raises(ValueError, match="Invalid NVCF request ID"):
        rebuild.MsaClient("test-key")._poll("../../other?token=value")
