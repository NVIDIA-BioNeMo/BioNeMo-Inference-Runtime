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

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit.Chem import GetPeriodicTable

from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.data.writers.cif_writer import CIFWriter
from bionemo_ir.data.writers.pdb_writer import PDBWriter
from bionemo_ir.pipeline.models.boltz1.feature_generators import Boltz1AtomFeatureGenerator, Boltz1TokenFeatureGenerator
from bionemo_ir.pipeline.models.boltz2.feature_collators import Boltz2FinalFeatureCollator
from bionemo_ir.pipeline.models.boltz2.feature_generators import (
    Boltz2AtomFeatureGenerator,
    Boltz2EnsembleFeatureGenerator,
    Boltz2TokenFeatureGenerator,
)
from bionemo_ir.pipeline.models.boltz2.structure import build_structure_from_input
from bionemo_ir.pipeline.models.boltz2.tokenizer_logic import tokenize_structure
from bionemo_ir.pipeline.models.openfold3.feature_collators import OpenFold3FinalFeatureCollator
from bionemo_ir.pipeline.models.openfold3.feature_context import _build_structure_from_polymers
from bionemo_ir.pipeline.models.openfold3.feature_generators import ConformerFeatureGenerator, StructureFeatureGenerator
from bionemo_ir.registry import ModelRegistry


@pytest.fixture(scope="module", params=["boltz-1", "boltz-2", "openfold3"])
def smiles_batch(request: pytest.FixtureRequest) -> tuple[str, dict, dict, list[str], list[str]]:
    model = request.param
    path = Path(__file__).resolve().parents[3] / "examples/data/samples/rna_dna_ligand/smiles_demo.json"
    polymers = json.loads(path.read_text())[0]["polymers"][1:]
    features = {}
    if model.startswith("boltz"):
        structure, molecules, _ = build_structure_from_input({"polymers": polymers}, {})
        tokens, bonds = tokenize_structure(structure)
        context = {"structure": structure, "tokens": tokens, "token_bonds": bonds, "molecules": molecules}
        generators = (
            [Boltz1TokenFeatureGenerator, Boltz1AtomFeatureGenerator]
            if model == "boltz-1"
            else [Boltz2TokenFeatureGenerator, Boltz2EnsembleFeatureGenerator, Boltz2AtomFeatureGenerator]
        )
        for generator in generators:
            features.update(generator()(features, context))
        batch = Boltz2FinalFeatureCollator()(features, context)
        n_tokens = len(tokens)
        output = {
            "masks": batch["atom_pad_mask"],
            "token_masks": batch["token_pad_mask"],
            "confidence_score": torch.tensor([[0.0, 1.0]]),
            "coords": torch.stack([batch["ref_pos"] + 1, batch["ref_pos"]], dim=1),
            "plddt": torch.full((1, 2, n_tokens), 75.0),
        }
        names = [atom.name for atom in structure.atoms]
        elements = [GetPeriodicTable().GetElementSymbol(atom.element) for atom in structure.atoms]
    else:
        structure = _build_structure_from_polymers(polymers)
        context = {"structure": structure}
        for generator in [StructureFeatureGenerator, ConformerFeatureGenerator]:
            features.update(generator()(features, context))
        batch = OpenFold3FinalFeatureCollator()(features, context)
        output = {"atom_positions_predicted": batch["ref_pos"]}
        names = structure["atom_names"]
        elements = structure["atom_elements"]
    return model, batch, output, names, elements


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_smiles_atom_roundtrip(smiles_batch: tuple, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, batch, output, names, elements = smiles_batch
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    output = {key: value.to(device) for key, value in output.items()}
    result = ModelRegistry.get_postprocessor(model)()(batch, output)
    active = result["atom_mask"] > 0
    assert int(active.sum()) == len(names) == 13
    assert sorted(result["atom_names"][active].tolist()) == sorted(names)
    assert sorted(result["atom_elements"][active].tolist()) == sorted(elements)
    for name in names:
        source = names.index(name)
        expected = (
            output["coords"][0, 1, source]
            if model.startswith("boltz")
            else output["atom_positions_predicted"][0, source]
        )
        np.testing.assert_array_equal(result["atom_positions"][result["atom_names"] == name][0], expected.cpu().numpy())

    kwargs = {
        "res_type_mapping": dict(enumerate(get_all_residue_types(model))),
        "atom_type_mapping": dict(enumerate(get_all_atom_types(model))),
        "output_path": None,
    }
    for writer in [CIFWriter, PDBWriter]:
        text = writer(**kwargs).write(result)
        records = [line for line in text.splitlines() if line.startswith("HETATM")]
        assert len(records) == len(names)
        if writer is CIFWriter:
            assert sorted(line.split()[3] for line in records) == sorted(names)
            assert sorted(line.split()[2] for line in records) == sorted(elements)
        else:
            assert sorted(line[12:16].strip() for line in records) == sorted(names)
            # SMILES residue names exceed legacy PDB width.
            assert sorted(line.split()[-1] for line in records) == sorted(elements)
