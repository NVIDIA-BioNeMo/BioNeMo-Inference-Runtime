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

from bionemo_ir.hubs import hf, local


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("boltz2_conf.ckpt", "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"),
        ("boltz2_aff.ckpt", "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"),
    ],
)
def test_boltz_checkpoint_sha256(filename, expected):
    assert local.BOLTZ_CHECKPOINT_SHA256[filename] == expected


def test_verify_boltz_checkpoint_sha256(tmp_path, monkeypatch):
    checkpoint = tmp_path / "test.ckpt"
    checkpoint.write_bytes(b"trusted checkpoint")
    monkeypatch.setitem(
        local.BOLTZ_CHECKPOINT_SHA256,
        checkpoint.name,
        hashlib.sha256(b"trusted checkpoint").hexdigest(),
    )

    local.verify_boltz_checkpoint_sha256(checkpoint, checkpoint.name)
    checkpoint.write_bytes(b"untrusted checkpoint")

    with pytest.raises(ValueError, match="SHA-256 mismatch: test.ckpt"):
        local.verify_boltz_checkpoint_sha256(checkpoint, checkpoint.name)


def test_missing_boltz_digest_blocks_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "boltz1_conf.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    load_state_dict = Mock()
    monkeypatch.setattr(hf, "hf_hub_download", lambda **_kwargs: str(checkpoint))
    monkeypatch.setattr(hf, "_load_boltz_state_dict", load_state_dict)

    with pytest.raises(ValueError, match="Missing SHA-256 digest: boltz1_conf.ckpt"):
        hf.load_state_dict_from_hf(
            repo_id="boltz-community/boltz-1",
            filename=checkpoint.name,
            name=local.SupMat.Boltz1,
        )

    load_state_dict.assert_not_called()
