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

import ray

from tensorrt_bionemo.pipeline.processor.engine_proc import (
    EngineProcessorConfig, Processor, build_processor)
from tensorrt_bionemo.pipeline.stages.configs import (
    FeatureGeneratorStageConfig, ParserStageConfig, WriterStageConfig)

if __name__ == "__main__":
    
    # set script-level parameters
    os.environ["ALPHAFOLD2_1_CKPT"] = "/workspaces/tensorrt-bionemo/checkpoints/alphafold2_1.pt"
    run_label = datetime.now().strftime('%Y%m%dT%H%M%S')
    writer_output_path=os.path.join(
        "/tmp/output/ray_stub",
        f"ray_stub_output_{run_label}",
        f"writer_output_{run_label}"
    )
    
    from tensorrt_bionemo.data.schemas import InputRequest
    req0 = InputRequest(
        fasta_file="tensorrt_bionemo/data/samples/T1031.fasta",
        a3m_files={"A": ["tensorrt_bionemo/data/samples/msas/T1031.a3m"]},
        mmcif_files={},
        is_files=True,
        is_description_formatted=False)

    req1 = InputRequest(
        fasta_file="tensorrt_bionemo/data/samples/T1033.fasta",
        a3m_files={"A": ["tensorrt_bionemo/data/samples/msas/T1033.a3m"]},
        mmcif_files={},
        is_files=True,
        is_description_formatted=False)

    req2 = InputRequest(
        fasta_file="tensorrt_bionemo/data/samples/T1094.fasta",
        a3m_files={"A": ["tensorrt_bionemo/data/samples/msas/T1094.a3m"]},
        mmcif_files={},
        is_files=True,
        is_description_formatted=False)

    req_bench = InputRequest(
        fasta_file="tensorrt_bionemo/data/samples/T1047s1.fasta",
        a3m_files={"A": ["tensorrt_bionemo/data/samples/msas/T1047s1.a3m"]},
        mmcif_files={},
        is_files=True,
        is_description_formatted=False)

    os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"
    config = EngineProcessorConfig(
        model_source="alphafold2_1",
        parser_stage=ParserStageConfig(compute=2),
        feature_generator_stage=FeatureGeneratorStageConfig(compute=4),
        writer_stage=WriterStageConfig(compute=2,
                                       output_path=writer_output_path,
                                       format="pdb"))
    processor: Processor = build_processor(config)

    records = []
    for i in range(5):
        records.append({"record": req_bench, "__record_id": f"T1047s{i}"})
    ds = ray.data.from_items(records)
    ds = processor(ds)
    # Force execution
    ds.materialize()

    # T1031_ensembled_tensors = torch.load(os.path.join(
    #     "dump", "T1031_ensembled_tensors_seed_310661746.pt"),
    #                                      weights_only=False)
    # for k, v in T1031_ensembled_tensors.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {v.shape}")
    # for row in ds.iter_rows():
    #     # print(row)
    #     print(row["__inference_error__"].get("traceback"))
