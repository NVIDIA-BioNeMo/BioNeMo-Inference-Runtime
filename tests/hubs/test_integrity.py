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
from unittest.mock import Mock

import pytest

from bionemo_ir.hubs import hf, integrity, local


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("boltz1_conf.ckpt", "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5"),
        ("boltz2_conf.ckpt", "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"),
        ("boltz2_aff.ckpt", "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"),
    ],
)
def test_boltz_checkpoint_sha256(filename, expected):
    assert integrity.CHECKPOINT_SHA256[filename] == expected


@pytest.mark.parametrize("name", sorted(hf.HF_CHECKPOINTS))
def test_every_hub_checkpoint_is_pinned(name):
    checkpoint = hf.HF_CHECKPOINTS[name]
    assert len(integrity.HF_REVISIONS[checkpoint.repo_id]) == 40
    assert len(integrity.CHECKPOINT_SHA256[checkpoint.filename.rsplit("/", 1)[-1]]) == 64


def test_verify_checkpoint_sha256(tmp_path, monkeypatch):
    checkpoint = tmp_path / "test.ckpt"
    checkpoint.write_bytes(b"trusted checkpoint")
    monkeypatch.setitem(
        integrity.CHECKPOINT_SHA256,
        checkpoint.name,
        hashlib.sha256(b"trusted checkpoint").hexdigest(),
    )

    integrity.verify_checkpoint_sha256(checkpoint, checkpoint.name)
    checkpoint.write_bytes(b"untrusted checkpoint")

    with pytest.raises(ValueError, match="SHA-256 mismatch: test.ckpt"):
        integrity.verify_checkpoint_sha256(checkpoint, checkpoint.name)


@pytest.mark.parametrize("return_raw", [False, True])
def test_missing_digest_blocks_use(tmp_path, monkeypatch, return_raw):
    checkpoint = tmp_path / "boltz1_unreviewed.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    load_state_dict = Mock()
    monkeypatch.setattr(hf, "hf_hub_download", lambda **_kwargs: str(checkpoint))
    monkeypatch.setattr(hf, "_load_boltz_state_dict", load_state_dict)

    with pytest.raises(ValueError, match="Missing SHA-256 digest: boltz1_unreviewed.ckpt"):
        hf.load_state_dict_from_hf(
            repo_id="boltz-community/boltz-1",
            filename=checkpoint.name,
            name=local.SupMat.Boltz1,
            return_raw=return_raw,
        )

    load_state_dict.assert_not_called()


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("nz/OpenFold", integrity.HF_REVISIONS["nz/OpenFold"]),
        ("someone/fork", None),
    ],
)
def test_default_repo_downloads_pinned_revision(tmp_path, monkeypatch, repo_id, revision):
    checkpoint = tmp_path / "finetuning_2.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(integrity.CHECKPOINT_SHA256, checkpoint.name, hashlib.sha256(b"checkpoint").hexdigest())
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(hf, "hf_hub_download", download)

    hf.load_state_dict_from_hf(repo_id=repo_id, filename=checkpoint.name, return_raw=True)

    assert download.call_args.kwargs["revision"] == revision


def test_checkpoint_without_hub_source_names_its_variable():
    with pytest.raises(ValueError, match="stage the checkpoint and set PROTENIX_V2_CKPT"):
        hf.load_hf_weights(local.SupMat.ProtenixV2)
