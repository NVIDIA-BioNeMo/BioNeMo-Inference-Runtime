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

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.data.writers.cif_writer import CIFWriter
from bionemo_ir.pipeline.models.boltz1.feature_collators import Boltz1FinalFeatureCollator
from bionemo_ir.pipeline.models.boltz2.const import chain_type_ids, token_ids, tokens
from bionemo_ir.pipeline.models.boltz2.feature_collators import Boltz2FinalFeatureCollator
from bionemo_ir.pipeline.models.boltz2.postprocessor import PostProcessor


def _encode_atom_names(atom_names: list[str]) -> torch.Tensor:
    codes = [[ord(char) - 32 for char in name[:4]] + [0] * (4 - len(name[:4])) for name in atom_names]
    return F.one_hot(torch.tensor(codes), num_classes=64)


def test_atp_identity_survives_shared_boltz1_boltz2_postprocessor(tmp_path: Path) -> None:
    """The shared Boltz path must use Token.res_name instead of ligand UNK restype."""
    assert Boltz1FinalFeatureCollator is Boltz2FinalFeatureCollator

    atom_names = ["PG", "O1G", "C1", "N9"]
    num_tokens = len(atom_names)
    features = {
        "atom_to_token": torch.eye(num_tokens),
        "ref_atom_name_chars": _encode_atom_names(atom_names),
        "res_type": F.one_hot(
            torch.full((num_tokens,), token_ids["UNK"]),
            num_classes=len(tokens),
        ),
        "residue_index": torch.zeros(num_tokens, dtype=torch.int64),
        "asym_id": torch.zeros(num_tokens, dtype=torch.int64),
        "mol_type": torch.full(
            (num_tokens,),
            chain_type_ids["NONPOLYMER"],
            dtype=torch.int64,
        ),
    }
    context = {"_row": {"tokens": [{"res_name": "ATP"}] * num_tokens}}

    batch = Boltz2FinalFeatureCollator(config=None)(features, context)
    result = PostProcessor(config=None)(
        batch,
        {
            "masks": torch.ones(1, num_tokens),
            "token_masks": torch.ones(1, num_tokens),
            "confidence_score": torch.zeros(1, 1),
            "coords": torch.zeros(1, 1, num_tokens, 3),
            "plddt": torch.full((1, 1, num_tokens), 50.0),
        },
    )

    assert batch["token_resnames"] == ["ATP"] * num_tokens
    assert result["residue_names"] == ["ATP"] * num_tokens
    np.testing.assert_array_equal(result["mol_types"], np.full(num_tokens, 3))
    assert int(result["atom_mask"].sum()) == num_tokens

    writer = CIFWriter(
        output_path=str(tmp_path / "atp.cif"),
        res_type_mapping=dict(enumerate(get_all_residue_types("boltz2"))),
        atom_type_mapping=dict(enumerate(get_all_atom_types("boltz2"))),
    )
    cif = writer.write(result)
    hetatm_lines = [line for line in cif.splitlines() if line.startswith("HETATM")]

    assert len(hetatm_lines) == num_tokens
    assert all("ATP" in line.split() for line in hetatm_lines)
    assert all("UNK" not in line.split() for line in hetatm_lines)
    assert "_chem_comp.id" in cif
    assert "ATP non-polymer" in cif
    assert "_pdbx_entity_nonpoly.comp_id" in cif
    assert "'Non-polymer ligand subunit' ATP" in cif
    assert " ATP 1 1 1 ATP " in cif
