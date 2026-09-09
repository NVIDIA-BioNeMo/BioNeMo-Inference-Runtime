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

# Run one serial Boltz-2 prediction and print a compact confidence summary.

import json

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.stages.configs import FeatureGeneratorStageConfig, WriterStageConfig

# Step 1:
# ------------------------------------------------------------------------------
# Prepare a complete protein request and its required unpaired MSA.
SEQUENCE = "ACKIENIKYKGKEVESKLGSQLIDIFNDLDRAKEEYDKLSSPEFIAKFGDWINDEVERNVNEDGEPLLIQDVRQDSSKHYFFILKNGERFDLLTR"

request = InputRequest(
    input_id="T1031",
    polymers=[
        Polymer(
            chain_id=["A1"],
            sequence=SEQUENCE,
            msas=[MSARecord(content=f">T1031\n{SEQUENCE}\n")],
        )
    ],
)

# Wrap the request in the row schema accepted by the processor.
rows = [{"record": request, "__record_id": request["input_id"]}]

# Step 2:
# ------------------------------------------------------------------------------
# Configure a deterministic serial run before introducing Ray replicas.
config = EngineProcessorConfig(
    model_source="boltz-2",
    runtime_args={
        "num_sampling_steps": 50,
    },
    feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": 42}),
    writer_stage=WriterStageConfig(output_path="output/serial", format="cif"),
    engine_kwargs={"profile_inference": True},
)

# Run the prediction and decode the writer's JSON score payload.
row = build_processor(config)(rows)[0]
scores = json.loads(row["scores"])

# Print compact scalars and shapes instead of large pLDDT and PAE arrays.
summary = {
    "record_id": row["__record_id"],
    "output_path": row["output_path"],
    "model_inference_time_s": round(row["model_inference_time"], 2),
    "score_keys": sorted(scores),
    "ptm": scores["ptm"],
    "mean_plddt": round(sum(scores["plddt"]) / len(scores["plddt"]), 2),
    "pae_shape": [len(scores["pae"]), len(scores["pae"][0])],
}
print(json.dumps(summary, indent=2))
