# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.testing as npt

import ray

import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
from biotite.structure import AtomArrayStack

from tensorrt_bionemo.pipeline.processor.engine_proc import (
    EngineProcessorConfig, Processor, build_processor)
from tensorrt_bionemo.pipeline.stages.configs import (
    FeatureGeneratorStageConfig, ParserStageConfig, WriterStageConfig)
from tensorrt_bionemo.data.parsers import read_fasta
from tensorrt_bionemo.data.schemas import InputRequest, Polymer, MSARecord


SAMPLE_DIR = Path("examples") / "data" / "samples" / "monomers"


def create_sample_requests(repeat: int = 1):
    """Create sample protein folding requests."""
    requests = []
    sample_ids = ["T1031"]
    for i in range(repeat):
        for sample_id in sample_ids:
            sequence = read_fasta(str(SAMPLE_DIR / f"{sample_id}.fasta"))["sequences"][0]["sequence"]
            requests.append(
                InputRequest(
                    input_id=f"{sample_id}_{i}",
                    polymers=[Polymer(
                        chain_id="A",
                        sequence=sequence,
                        msas=[MSARecord(path=str(SAMPLE_DIR / "msas" / f"{sample_id}.a3m"))]
                    )]
                )
            )    
    return requests


def test_writer_stage_in_noop_pipe(tmp_path: Path):
    """Check that the atomic structure written with PDBWriter is the same as 
    the atomic structure written with CIFWriter..

        Use alphafold2_1.pt weights.  openfold2_ptm_1 weights fail to load in 
        OpenFold2.load_weights()

        Args:
            tmp_path: pytest construct
    """
    
    # (0) settings
    model_source = "alphafold2_1"
    os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"
    
    # (1) Create a scratch-space directory
    run_label = datetime.now().strftime('%Y%m%dT%H%M%S')
    output_path=os.path.join(
        tmp_path,
        "output/tests/pipeline/stages",
        f"test_writer_stage_in_noop_pipeline_output_{run_label}",
        f"writer_output_{run_label}"
    )

    # (2) Define dataset    
    requests = create_sample_requests()
    records = [{"record": req, "__record_id": req["input_id"]} for req in requests]
    ds = ray.data.from_items(records)
    
    # (3) Define and run processors for pdb and cif formats
    config_for_pdb = EngineProcessorConfig(
        model_source=model_source,
        parser_stage=ParserStageConfig(compute=2),
        feature_generator_stage=FeatureGeneratorStageConfig(compute=4),
        writer_stage=WriterStageConfig(
            compute=2, output_path=output_path, format="pdb"))
    config_for_cif = EngineProcessorConfig(
        model_source=model_source,
        parser_stage=ParserStageConfig(compute=2),
        feature_generator_stage=FeatureGeneratorStageConfig(compute=4),
        writer_stage=WriterStageConfig(
            compute=2, output_path=output_path, format="cif"))
    
    processor_for_pdb: Processor = build_processor(config_for_pdb)
    processor_for_cif: Processor = build_processor(config_for_cif)
    
    # (4) Run processors for pdb and cif formats
    ds_for_pdb = processor_for_pdb(ds)
    ds_for_pdb.materialize()
    
    ds_for_cif = processor_for_cif(ds)
    ds_for_cif.materialize()
    
    # (5) metrics to compare output
    for input_id in [req["input_id"] for req in requests]:
        pdb_file = pdb.PDBFile.read(
            os.path.join(output_path, f"{input_id}.pdb")
        )
        struc_from_pdb: AtomArrayStack = pdb.get_structure(pdb_file)
        cif_file = pdbx.CIFFile.read(
            os.path.join(output_path, f"{input_id}.cif")
        )
        struc_from_cif: AtomArrayStack = pdbx.get_structure(cif_file)
        
        atom_coord_from_pdb: np.array = struc_from_pdb.coord
        atom_coord_from_cif: np.array = struc_from_cif.coord
        
        npt.assert_allclose(atom_coord_from_pdb, atom_coord_from_cif, rtol=1e-3, atol=1e-3)
