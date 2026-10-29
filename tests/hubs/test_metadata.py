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

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from bionemo_ir.hubs import metadata


def test_load_metadata_accepts_matching_download(tmp_path, monkeypatch):
    payload = b"trusted metadata"
    downloaded = tmp_path / "metadata.bin"
    downloaded.write_bytes(payload)
    prepare = Mock(return_value="prepared")
    asset = metadata.MetadataFile(
        metadata_key="asset",
        repo_id="example/repo",
        filename=downloaded.name,
        env="TEST_METADATA_PATH",
        sha256=hashlib.sha256(payload).hexdigest(),
        prepare=prepare,
    )
    monkeypatch.setitem(metadata.HF_MODEL_METADATA, "test-model", [asset])
    monkeypatch.setattr(metadata, "resolve_from_env", lambda _env: None)
    monkeypatch.setattr(metadata, "resolve_cached_metadata", lambda _env: None)
    monkeypatch.setattr(metadata, "download_hf_file", lambda *_args: str(downloaded))

    assert metadata.load_metadata("test-model", cache_dir=tmp_path) == {"asset": "prepared"}
    prepare.assert_called_once_with(str(downloaded), tmp_path)


def test_load_metadata_rejects_mismatched_download(tmp_path, monkeypatch):
    downloaded = tmp_path / "metadata.bin"
    downloaded.write_bytes(b"untrusted metadata")
    prepare = Mock()
    asset = metadata.MetadataFile(
        metadata_key="asset",
        repo_id="example/repo",
        filename=downloaded.name,
        env="TEST_METADATA_PATH",
        sha256=hashlib.sha256(b"trusted metadata").hexdigest(),
        prepare=prepare,
    )
    monkeypatch.setitem(metadata.HF_MODEL_METADATA, "test-model", [asset])
    monkeypatch.setattr(metadata, "resolve_from_env", lambda _env: None)
    monkeypatch.setattr(metadata, "resolve_cached_metadata", lambda _env: None)
    monkeypatch.setattr(metadata, "download_hf_file", lambda *_args: str(downloaded))

    with pytest.raises(ValueError, match="SHA-256 mismatch: metadata.bin"):
        metadata.load_metadata("test-model", cache_dir=tmp_path)

    prepare.assert_not_called()


@pytest.mark.parametrize(
    ("metadata_key", "expected"),
    [
        ("ccd_path", metadata.BOLTZ_CCD_SHA256),
        ("mol_dir", metadata.BOLTZ_MOLS_SHA256),
    ],
)
def test_boltz_metadata_has_sha256(metadata_key, expected):
    for model_name in (metadata.SupMat.Boltz1, metadata.SupMat.Boltz2, metadata.SupMat.Boltz2Affinity):
        asset = next(item for item in metadata.HF_MODEL_METADATA[model_name] if item.metadata_key == metadata_key)
        assert asset.sha256 == expected


def test_protenix_offline_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged native CCD can resolve without Hub or publisher requests."""
    from bionemo_ir.hubs.metadata import load_metadata

    mols = tmp_path / "mols.pkl"
    mols.write_bytes(b"mols")
    monkeypatch.setenv("BIOIR_PROTENIX_CCD_MOLS_PATH", str(mols))
    result = load_metadata("protenix-v2", cache_dir=tmp_path, local_files_only=True)
    assert result == {"ccd_mols_path": str(mols)}


def test_protenix_metadata_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOIR_PROTENIX_CCD_MOLS_PATH", raising=False)
    monkeypatch.setenv("BIOIR_METADATA", str(tmp_path / "empty"))
    monkeypatch.setattr(metadata, "download_hf_file", Mock(side_effect=AssertionError("unexpected download")))
    with pytest.raises(FileNotFoundError, match="BIOIR_PROTENIX_CCD_MOLS_PATH"):
        metadata.load_metadata("protenix-v2", cache_dir=tmp_path)
