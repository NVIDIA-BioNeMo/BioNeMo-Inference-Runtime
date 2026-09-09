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

"""Canonical unknown-amino-acid ``X`` is UNK, not an unexpected residue."""

from __future__ import annotations

import logging

from bionemo_ir.pipeline.models.openfold3.const import (
    _PROTEIN_1TO3,
    AA_1_TO_IDX,
    RESNAME_TO_IDX,
    UNK_IDX,
)
from bionemo_ir.pipeline.models.openfold3.feature_context import _build_structure_from_polymers


def test_protein_1to3_maps_x_to_unk() -> None:
    assert _PROTEIN_1TO3["X"] == "UNK"
    assert AA_1_TO_IDX["X"] == RESNAME_TO_IDX["UNK"] == UNK_IDX


def test_structure_builder_treats_x_as_unk_without_warning(
    caplog: logging.LogCaptureFixture,
) -> None:
    polymers = [{"sequence": "AXG", "chain_id": "A", "polymer_type": "protein"}]
    with caplog.at_level(logging.WARNING, logger="bionemo_ir.pipeline.models.openfold3.feature_context"):
        structure = _build_structure_from_polymers(polymers)
    assert structure["token_resnames"] == ["ALA", "UNK", "GLY"]
    assert "Unknown protein residue" not in caplog.text


def test_structure_builder_warns_on_noncanonical_protein_letter(
    caplog: logging.LogCaptureFixture,
) -> None:
    polymers = [{"sequence": "AJ", "chain_id": "A", "polymer_type": "protein"}]
    with caplog.at_level(logging.WARNING, logger="bionemo_ir.pipeline.models.openfold3.feature_context"):
        structure = _build_structure_from_polymers(polymers)
    assert structure["token_resnames"] == ["ALA", "UNK"]
    assert "Unknown protein residue 'J'" in caplog.text
