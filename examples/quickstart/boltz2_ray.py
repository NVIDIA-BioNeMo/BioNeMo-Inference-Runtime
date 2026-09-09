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

# Scale the serial Boltz-2 example across every visible GPU with Ray replicas.

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "fatal")

import ray
import torch

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.stages.configs import (
    EngineStageConfig,
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)

# Step 1:
# ------------------------------------------------------------------------------
# Prepare one complete ubiquitin request per visible GPU and its required unpaired MSA.
# Canonical sample: RCSB PDB 1UBQ, https://www.rcsb.org/structure/1UBQ
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

replicas = torch.cuda.device_count()
if not replicas:
    raise RuntimeError("this example requires at least one visible GPU")

rows = []
for index in range(replicas):
    record_id = f"1UBQ-{index + 1}"
    request = InputRequest(
        input_id=record_id,
        polymers=[
            Polymer(
                chain_id=["A1"],
                sequence=SEQUENCE,
                msas=[MSARecord(content=f">{record_id}\n{SEQUENCE}\n")],
            )
        ],
    )
    rows.append({"record": request, "__record_id": record_id})

# Step 2:
# ------------------------------------------------------------------------------
# Keep the serial example deterministic with the same sampling count and seed.
SAMPLING_STEPS = 50
SEED = 42


def seed_worker() -> None:
    """Seed each Ray worker before it loads a pipeline stage."""
    torch.manual_seed(SEED)


# Step 3:
# ------------------------------------------------------------------------------
# Add Ray execution and one complete Boltz-2 model replica per visible GPU.
launch_dir = Path.cwd()
config = EngineProcessorConfig(
    model_source="boltz-2",
    executor_backend="ray",
    runtime_args={"num_sampling_steps": SAMPLING_STEPS},
    feature_generator_stage=FeatureGeneratorStageConfig(
        init_context={"random_seed": SEED},
    ),
    engine_stage=EngineStageConfig(compute=replicas),
    writer_stage=WriterStageConfig(
        output_path=str(launch_dir / f"output/ray-{replicas}gpu"),
        format="cif",
    ),
    should_continue_on_error=False,
)

# Start workers outside a mounted source checkout so they import the installed wheel.
os.chdir("/tmp")
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("ray.data._internal").setLevel(logging.ERROR)
ray.init(
    include_dashboard=False,
    logging_level="ERROR",
    log_to_driver=False,
    runtime_env={"worker_process_setup_hook": seed_worker},
)
data_context = ray.data.DataContext.get_current()
data_context.enable_progress_bars = False
data_context.enable_operator_progress_bars = False

# Submit the independent requests as a Ray Dataset and wait for every output.
start = time.perf_counter()
try:
    dataset = ray.data.from_items(rows)
    outputs = list(build_processor(config)(dataset).materialize().iter_rows())
    resources = ray.cluster_resources()
finally:
    elapsed = time.perf_counter() - start
    ray.shutdown()

# Decode one representative result and report whole-worklist throughput.
row = sorted(outputs, key=lambda output: output["__record_id"])[0]
scores = json.loads(row["scores"])
throughput = len(outputs) * 3600 / elapsed
summary = {
    "visible_gpus": replicas,
    "ray_gpus": resources.get("GPU", 0),
    "completed_structures": f"{len(outputs)}/{len(rows)}",
    "worklist_wall_s": round(elapsed, 2),
    "structures_per_hour": round(throughput, 2),
    "structures_per_gpu_hour": round(throughput / replicas, 2),
    "example": {
        "record_id": row["__record_id"],
        "output_path": row["output_path"],
        "ptm": scores["ptm"],
        "mean_plddt": round(sum(scores["plddt"]) / len(scores["plddt"]), 2),
        "pae_shape": [len(scores["pae"]), len(scores["pae"][0])],
    },
}
print(json.dumps(summary, indent=2))
