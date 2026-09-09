---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Scale Batch Inference with Ray

In this tutorial, you scale Boltz-2 batch inference across every visible
GPU with Ray. You create a worklist of independent protein requests,
configure one complete model replica per GPU, run the pipeline, and inspect
completion, throughput, and confidence summaries.

The complete runnable script is available at
[`examples/quickstart/boltz2_ray.py`][ray-example]. The sections
below explain its worklist, Ray configuration, and output.

[ray-example]: ../examples/quickstart/boltz2_ray.py

Ray replicas process independent requests concurrently. They do not split one
structure prediction across several GPUs.

## Before You Begin

Complete the [installation][installation] on a system with at least one
supported GPU. Run the serial [Quickstart][quickstart] first to verify the
wheel, checkpoint, and metadata.

[installation]: install.md
[quickstart]: quickstart.md

If you do not have a repository checkout, create `quickstart_ray.py` with the
following example. Both versions configure the Ray workers to import BioNeMo
Inference Runtime (BioIR) from the installed wheel rather than an editable
checkout.

Start at the top-level imports.

```python
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "fatal")

import ray
import torch

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import (
    EngineStageConfig,
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)
```

## Understand the Worklist

The first part of `quickstart_ray.py` detects the visible GPUs and creates one
prediction request for each GPU:

```python
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
```

This block creates the Ray worklist:

- `torch.cuda.device_count()` determines how many complete model replicas the
  example can run and stops with an error when no GPU is visible.
- Each `InputRequest` contains protein chain `A1`, the 1UBQ ubiquitin sequence,
  and an inline, query-only unpaired MSA.
- Each request has a unique identifier, such as `1UBQ-1` or `1UBQ-2`.
  `__record_id` becomes the CIF filename.
- The number of requests matches the number of replicas so the example verifies
  that each replica can complete one prediction.

Use a larger, representative worklist when measuring sustained throughput. Refer
to [Input Requests][input-requests] for other biomolecule types, MSAs, and
templates.

[input-requests]: ref/api.md#input-requests

## Understand the Pipeline Configuration

The next block configures deterministic feature generation and one Ray engine
actor per visible GPU:

```python
SAMPLING_STEPS = 50
SEED = 42


def seed_worker() -> None:
    """Seed each Ray worker before it loads a pipeline stage."""
    torch.manual_seed(SEED)


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
```

The configuration controls the distributed prediction pipeline:

- `executor_backend="ray"` sends rows through the Ray Data pipeline.
- `EngineStageConfig(compute=replicas)` creates one engine actor per visible
  GPU. Each actor reserves one GPU and loads one complete Boltz-2 model replica.
- `num_sampling_steps=50` shortens the diffusion stage for this example.
- `random_seed=42` seeds feature generation. The worker setup hook also seeds
  PyTorch before each worker loads its pipeline stage.
- `WriterStageConfig` uses an absolute path captured before the workers change
  directories and writes one CIF structure per request.
- `should_continue_on_error=False` stops the example if any request fails.

Refer to [`EngineProcessorConfig`][engine-processor-config],
[Runtime Args][runtime-args], and
[Ray Multi-GPU Replicas][ray-replicas] for the available
controls.

[engine-processor-config]: ref/api.md#engineprocessorconfig
[runtime-args]: ref/api.md#runtime-args
[ray-replicas]: ref/api.md#ray-multi-gpu-replicas

## Understand Ray Execution

The next block starts Ray, materializes the worklist, waits for every result,
and shuts Ray down:

```python
# Keep workers outside a mounted source checkout so they use the installed wheel.
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

start = time.perf_counter()
try:
    dataset = ray.data.from_items(rows)
    outputs = list(build_processor(config)(dataset).materialize().iter_rows())
    resources = ray.cluster_resources()
finally:
    elapsed = time.perf_counter() - start
    ray.shutdown()
```

The script starts workers from `/tmp` so a mounted source checkout cannot
shadow the installed wheel. `ray.init()` disables the dashboard and verbose
driver logs, then runs `seed_worker` in each worker process.

`ray.data.from_items(rows)` creates the distributed dataset.
`build_processor(config)` assembles the Ray pipeline, and `materialize()` waits
for every request to complete. The `finally` block records elapsed worklist time
and shuts Ray down even if inference fails.

## Understand the Results

The final block calculates whole-worklist throughput and builds a compact
summary from one representative prediction:

```python
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
```

The summary reports:

- `completed_structures` — the number of successful outputs compared with the
  number of submitted requests.
- `worklist_wall_s` — the elapsed time for materializing the complete worklist.
- `structures_per_hour` — worklist throughput across all visible GPUs.
- `structures_per_gpu_hour` — aggregate throughput divided by the number of
  replicas.
- `example` — the output path and confidence metrics from the first result,
  without printing the complete pLDDT and PAE arrays.

Refer to [Outputs][outputs] for the complete output-row schema.

[outputs]: ref/api.md#outputs

## Run the Example

From the repository root, run the maintained example:

```bash
python examples/quickstart/boltz2_ray.py
```

If you copied the example into a standalone file, run that file instead:

```bash
python quickstart_ray.py
```

## Example Output

The output path starts under the directory where you launch the script. Timing
and scores can vary across GPUs, systems, and releases.

### One NVIDIA H100 80 GB HBM3

This captured run used one 700 W H100:

```json
{
  "visible_gpus": 1,
  "ray_gpus": 1.0,
  "completed_structures": "1/1",
  "worklist_wall_s": 24.95,
  "structures_per_hour": 144.31,
  "structures_per_gpu_hour": 144.31,
  "example": {
    "record_id": "1UBQ-1",
    "output_path": "/bioir/output/ray-1gpu/1UBQ-1.cif",
    "ptm": 0.9149,
    "mean_plddt": 0.93,
    "pae_shape": [
      76,
      76
    ]
  }
}
```

### Eight NVIDIA H200 NVL GPUs

This captured run used eight 600 W H200 NVL GPUs:

```json
{
  "visible_gpus": 8,
  "ray_gpus": 8.0,
  "completed_structures": "8/8",
  "worklist_wall_s": 54.25,
  "structures_per_hour": 530.91,
  "structures_per_gpu_hour": 66.36,
  "example": {
    "record_id": "1UBQ-1",
    "output_path": "/bioir/output/ray-8gpu/1UBQ-1.cif",
    "ptm": 0.9149,
    "mean_plddt": 0.93,
    "pae_shape": [
      76,
      76
    ]
  }
}
```

Treat these values as successful-run examples, not benchmarks. Each run uses
only one short request per GPU and includes pipeline and model setup in the
worklist time. The stable result is that `completed_structures` matches the
request count and every reported output path contains a non-empty CIF file.

## Next Steps

- Configure stage resources and larger worklists in
  [Ray Multi-GPU Replicas][ray-replicas].
- Review controlled performance measurements in [Benchmarks][benchmarks].
- Check supported accelerators in the [Support Matrix][support-matrix].

[benchmarks]: ref/benchmark.md
[support-matrix]: ref/support-matrix.md
