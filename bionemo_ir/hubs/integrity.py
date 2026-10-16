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

"""Artifact integrity checks."""

import hashlib
from pathlib import Path

# Hugging Face commit each default repo is downloaded at. A commit is immutable,
# so an upstream re-upload changes nothing until one of these moves. A repo not
# listed here (a caller's own `repo_id`) resolves its default branch.
HF_REVISIONS = {
    "boltz-community/boltz-1": "7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1",
    "boltz-community/boltz-2": "6fdef46d763fee7fbb83ca5501ccceff43b85607",
    "nz/OpenFold": "6344b2a680cbbae20b8afdb7f154228cc5b0cb27",
    "OpenFold/OpenFold3": "33ffa390314567dd2d61bb635de52c7810e47056",
}

# SHA-256 of every checkpoint BioIR downloads from the Hub, keyed by file name.
# Checked before deserialization; a file missing from this table is refused.
CHECKPOINT_SHA256 = {
    "boltz1_conf.ckpt": "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5",
    "boltz2_conf.ckpt": "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1",
    "boltz2_aff.ckpt": "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e",
    "finetuning_2.pt": "9511993ad138a66f3c51412d65351c5162e16bc4abd51d76f8016b6e0c392362",
    "finetuning_3.pt": "4ee7b3de7a2754c6102029b2997df3b9f32d3e9104836986939d1aba2a164494",
    "finetuning_4.pt": "a0bc9cde0c9c09c9632e496cd290319ea4e2e63456c51063e3c44db6c0cb9dc0",
    "finetuning_5.pt": "8b0951473212433443fc9970a7b902830d377da21bc3d0304dbbb01dd0d64ad2",
    "finetuning_no_templ_1.pt": "aebc5db250c4c9cc6751490329981384c82407f6b0b24026348abc2b63c40b09",
    "finetuning_no_templ_2.pt": "400e1d95a41203633925b0d009e91371617a7123107acceb3ca07e278c3ee634",
    "finetuning_no_templ_ptm_1.pt": "1a179937a6f61143c41f8b6a0b763ac1f4762c9dc63d2fb151240930c110bee1",
    "finetuning_ptm_1.pt": "11b641d4c90440b945197892f8b048f9dd8ba6224a31960682847f6c07dd7075",
    "finetuning_ptm_2.pt": "aff88a4534f500bcd4509584c2a6ce824417b31f4efaa5b17d19b4c97a8aaa49",
    "of3-p2-155k.pt": "af09eac4f29cef856633af07558cb143226fe95ebbef2c20921769d4a5f4bee4",
}


def verify_sha256(path: str | Path, expected: str) -> None:
    with open(path, "rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {Path(path).name}")


def verify_checkpoint_sha256(path: str | Path, filename: str) -> None:
    """Check a downloaded checkpoint against :data:`CHECKPOINT_SHA256`.

    Args:
        path: Local file to hash.
        filename: Repo path the file was downloaded as; only its name is used.

    Raises:
        ValueError: The name has no recorded digest, or the digest differs.
    """
    checkpoint_name = Path(filename).name
    expected = CHECKPOINT_SHA256.get(checkpoint_name)
    if expected is None:
        raise ValueError(f"Missing SHA-256 digest: {checkpoint_name}")
    verify_sha256(path, expected)
