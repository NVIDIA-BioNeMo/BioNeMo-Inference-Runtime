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

from bionemo_ir.data.writers.cif_writer import CIFWriter
from bionemo_ir.pipeline.models.openfold3.common import encode_atom_name_chars_one_hot
from bionemo_ir.pipeline.models.openfold3.const import NUM_RESTYPE_CLASSES, UNK_IDX
from bionemo_ir.pipeline.models.openfold3.feature_collators import OpenFold3FinalFeatureCollator
from bionemo_ir.pipeline.models.openfold3.postprocessor import PostProcessor
from tests.common.test_utils.synthetic_folding_outputs import of3_mappings


def test_atp_identity_survives_collation_postprocessing_and_cif(tmp_path: Path) -> None:
    """An atomized CCD ligand must not fall back to UNK after featurization."""
    atom_names = [
        "PG",
        "O1G",
        "O2G",
        "O3G",
        "PB",
        "O1B",
        "O2B",
        "O3B",
        "PA",
        "O1A",
        "O2A",
        "O3A",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ]
    num_tokens = len(atom_names)
    features = {
        "token_mask": torch.ones(num_tokens),
        "atom_mask": torch.ones(num_tokens),
        "atom_to_token_index": torch.arange(num_tokens),
        "ref_atom_name_chars": encode_atom_name_chars_one_hot(atom_names),
        "restype": F.one_hot(
            torch.full((num_tokens,), UNK_IDX),
            num_classes=NUM_RESTYPE_CLASSES,
        ),
        "residue_index": torch.ones(num_tokens, dtype=torch.int32),
        "asym_id": torch.ones(num_tokens, dtype=torch.int32),
    }
    context = {
        "_row": {
            "structure": {
                "token_resnames": ["ATP"] * num_tokens,
                "token_mol_types": [3] * num_tokens,
            }
        }
    }

    batch = OpenFold3FinalFeatureCollator(config=None)(features, context)
    result = PostProcessor(config=None)(
        batch,
        {"atom_positions_predicted": torch.zeros(1, num_tokens, 3)},
    )

    assert batch["token_resnames"] == ["ATP"] * num_tokens
    assert result["residue_names"] == ["ATP"] * num_tokens
    np.testing.assert_array_equal(result["mol_types"], np.full(num_tokens, 3))
    assert int(result["atom_mask"].sum()) == num_tokens

    res_type_mapping, atom_type_mapping = of3_mappings()
    writer = CIFWriter(
        output_path=str(tmp_path / "atp.cif"),
        res_type_mapping=res_type_mapping,
        atom_type_mapping=atom_type_mapping,
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
