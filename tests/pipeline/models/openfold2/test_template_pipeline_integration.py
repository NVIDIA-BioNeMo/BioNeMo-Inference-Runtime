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

"""Real serial GPU coverage for monomer and multimer template pipelines."""

from __future__ import annotations

import asyncio
import gc
import json
import os
from pathlib import Path

import biotite.structure as bio_structure
import numpy as np
import torch
from biotite.structure.io import pdbx

from tensorrt_bionemo.data.schemas import InputRequest, Polymer, Template
from tensorrt_bionemo.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from tensorrt_bionemo.pipeline.stages.configs import FeatureGeneratorStageConfig, WriterStageConfig
from tensorrt_bionemo.pipeline.stages.parser_stage import ParserUDF
from tensorrt_bionemo.registry import get_model_class, get_tokenizer
from tests.common.test_utils.seeding import seed_everything

QUERY_SEQUENCE = "GMEGPLNLAHQQSRRADRLLAAGKYEEAISCHKKAAAYLSEAMKLTQSEQAHLSLELQRDSHMKQLLLIQERWKRAQREERLKA"
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "data" / "samples" / "monomers" / "templates" / "4zey.cif"
)
SEED = 0

CASES = (
    {
        "model_source": "alphafold2_1",
        "checkpoint_env": "ALPHAFOLD2_1_CKPT",
        "chain_id": "A",
        "residue_count": 84,
        "chain_count": 1,
    },
    {
        "model_source": "alphafold2_multimer_1",
        "checkpoint_env": "ALPHAFOLD2_MULTIMER_1_CKPT",
        "chain_id": ["A", "B"],
        "residue_count": 168,
        "chain_count": 2,
    },
)


def _request(case: dict, with_template: bool) -> InputRequest:
    suffix = "with_template" if with_template else "no_template"
    templates = [Template(path=str(TEMPLATE_PATH), format="cif", chain_id="A")] if with_template else None
    return InputRequest(
        input_id=f"{case['model_source']}_{suffix}",
        polymers=[
            Polymer(
                chain_id=case["chain_id"],
                sequence=QUERY_SEQUENCE,
                msas=None,
                paired_msas=None,
                templates=templates,
            )
        ],
    )


def _small_inference_config(model_source: str):
    model_class = get_model_class(model_source)
    config = model_class.get_pretrained_config(model_source)
    config.max_recycling_iters = 0
    config.max_msa_clusters = 2
    config.max_extra_msa = 1
    config.max_templates = 1
    return config


def _template_mask_state(request: InputRequest, config: object, model_source: str) -> tuple[bool, int, tuple[int, ...]]:
    parser = ParserUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=["record"],
        update_row=True,
    )
    parsed = asyncio.run(
        parser.udf_for_item(
            {
                "record": request,
                "__record_id": request["input_id"],
            }
        )
    )["parsed"]
    tokenizer = get_tokenizer(model_source)
    context_generator = tokenizer.context_generator_specs["primary"].generator(config=config)
    context = context_generator(parsed)
    atom_mask = context["template_all_atom_mask"]
    return (
        bool(context["is_template_present"].item()),
        int(torch.count_nonzero(atom_mask).item()),
        tuple(atom_mask.shape),
    )


def _assert_output(row: dict, expected_residues: int, expected_chains: int) -> np.ndarray:
    output_path = Path(row["output_path"])
    assert output_path.is_file(), f"Pipeline did not write {output_path}"
    assert output_path.stat().st_size > 0

    # ModelCIF has no auth_* fields; read label_* directly.
    model_structure = pdbx.get_structure(pdbx.CIFFile.read(str(output_path)), model=1, use_author_fields=False)
    ca_atoms = model_structure[bio_structure.filter_amino_acids(model_structure) & (model_structure.atom_name == "CA")]
    assert ca_atoms.array_length() == expected_residues
    assert len(np.unique(ca_atoms.chain_id)) == expected_chains
    assert np.isfinite(ca_atoms.coord).all()

    scores = json.loads(row["scores"])
    plddt = np.asarray(scores["plddt"], dtype=np.float64)
    assert plddt.shape == (expected_residues,)
    assert np.isfinite(plddt).all()
    return plddt


def test_real_template_pipelines_monomer_and_multimer(tmp_path: Path):
    """Exercise both model presets with paired template/no-template inputs."""
    assert torch.cuda.is_available(), "OpenFold2 integration requires CUDA"
    assert TEMPLATE_PATH.is_file(), f"Missing 4ZEY fixture: {TEMPLATE_PATH}"
    for case in CASES:
        checkpoint_value = os.environ.get(case["checkpoint_env"])
        assert checkpoint_value, f"{case['checkpoint_env']} must be provisioned for GPU CI"
        checkpoint = Path(checkpoint_value)
        assert checkpoint.is_file(), f"Missing checkpoint: {checkpoint}"

    for case in CASES:
        model_source = case["model_source"]
        config = _small_inference_config(model_source)
        requests = {
            False: _request(case, with_template=False),
            True: _request(case, with_template=True),
        }

        no_template_state = _template_mask_state(requests[False], config, model_source)
        with_template_state = _template_mask_state(requests[True], config, model_source)
        assert no_template_state[0] is False
        assert no_template_state[1] == 0
        assert no_template_state[2][-2:] == (case["residue_count"], 37)
        assert with_template_state[0] is True
        assert with_template_state[1] > 0
        assert with_template_state[2][-2:] == (case["residue_count"], 37)

        output_dir = tmp_path / model_source
        processor_config = EngineProcessorConfig(
            model_source=model_source,
            executor_backend=None,
            engine_kwargs={"config": config},
            feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": SEED}),
            writer_stage=WriterStageConfig(output_path=str(output_dir), format="cif"),
        )
        processor = build_processor(processor_config)
        records = [
            {
                "record": request,
                "__record_id": request["input_id"],
                "random_seed": SEED,
            }
            for request in requests.values()
        ]

        seed_everything(SEED)
        try:
            results = processor(records)
            assert {row["__record_id"] for row in results} == {request["input_id"] for request in requests.values()}
            by_id = {row["__record_id"]: row for row in results}
            plddt = {
                with_template: _assert_output(
                    by_id[request["input_id"]],
                    case["residue_count"],
                    case["chain_count"],
                )
                for with_template, request in requests.items()
            }
            mean_abs_delta = float(np.mean(np.abs(plddt[True] - plddt[False])))
            assert mean_abs_delta > 0.1, f"{model_source} template/no-template pLDDT delta is only {mean_abs_delta:.6f}"
        finally:
            del processor
            gc.collect()
            torch.cuda.empty_cache()
