---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Run Your First Prediction

In this tutorial, you run a complete Boltz-2 structure prediction with the
installed BioNeMo Inference Runtime (BioIR) wheel. You create a protein
request with an inline multiple sequence alignment (MSA), configure the serial
inference pipeline, generate a CIF structure, and inspect its confidence scores.

The complete runnable script is available at
[`examples/quickstart/boltz2.py`][boltz2-example]. The following steps
explain its inputs, configuration, and output.

[boltz2-example]: ../examples/quickstart/boltz2.py

## Before You Begin

Complete the [installation][installation], including its system requirements. If
you do not have a repository checkout, create `quickstart.py` and add the
following code blocks in order.

[installation]: install.md

The first run downloads the Boltz-2 checkpoint and chemical metadata from
Hugging Face. Later runs reuse the local copies. Refer to
[Model Weights][model-weights] for cache locations and offline staging.

[model-weights]: ref/model-weights.md

Start at the top-level import.

```python
import json

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import (
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)
```

## Understand the Input Request

The first part of `quickstart.py` creates the prediction request:

```python
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
rows = [{"record": request, "__record_id": request["input_id"]}]
```

This block creates one prediction request:

- `InputRequest` describes one biomolecular complex and gives it the identifier
  `T1031`.
- `Polymer` defines protein chain `A1` and its amino-acid sequence.
- `MSARecord` provides an inline A3M alignment containing only the query
  sequence. This keeps the example self-contained. Production requests should
  use a full MSA.
- `rows` wraps the request in the row format consumed by `build_processor`.
  `__record_id` becomes the output filename.

Refer to [Input Requests][input-requests] for MSAs, templates,
nucleic acids, ligands, and multiple chains.

[input-requests]: ref/api.md#input-requests

## Understand the Pipeline Configuration

The next block configures the serial prediction pipeline:

```python
config = EngineProcessorConfig(
    model_source="boltz-2",
    runtime_args={"num_sampling_steps": 50},
    feature_generator_stage=FeatureGeneratorStageConfig(
        init_context={"random_seed": 42},
    ),
    writer_stage=WriterStageConfig(
        output_path="output/serial",
        format="cif",
    ),
    engine_kwargs={"profile_inference": True},
)
```

The configuration controls the complete prediction pipeline:

- `model_source="boltz-2"` selects the Boltz-2 model, tokenizer, feature
  generator, checkpoint, and default runtime arguments.
- `num_sampling_steps=50` shortens the diffusion stage for this example. Other
  Boltz-2 arguments retain their registered defaults.
- `random_seed=42` makes feature generation reproducible.
- `WriterStageConfig` writes a CIF structure under `output/serial`.
- `profile_inference=True` adds the GPU model-forward time to the output row.
- Omitting `executor_backend` selects the serial processor. Ray replicas are
  intended for processing many independent requests across several GPUs.

Refer to [`EngineProcessorConfig`][engine-processor-config],
[Runtime Args][runtime-args], and
[Ray Multi-GPU Replicas][ray-replicas] for the available
controls.

[engine-processor-config]: ref/api.md#engineprocessorconfig
[runtime-args]: ref/api.md#runtime-args
[ray-replicas]: ref/api.md#ray-multi-gpu-replicas

## Understand Inference and Results

The final block runs inference and builds a compact score summary:

```python
row = build_processor(config)(rows)[0]
scores = json.loads(row["scores"])

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
```

`build_processor(config)` assembles the parser, tokenizer, feature generator,
folding engine, and writer. Calling the processor returns one row for each
input row. The writer stores `scores` as JSON, so the example decodes it before
reading individual metrics.

The summary prints scalar values and array shapes instead of the full pLDDT and
PAE arrays. Refer to [`build_processor`][build-processor] and
[Outputs][outputs] for the complete row schema.

[build-processor]: ref/api.md#build_processor
[outputs]: ref/api.md#outputs

## Run the Example

From the repository root, run the maintained example:

```bash
python examples/quickstart/boltz2.py
```

If you copied the code blocks into a standalone file, run that file instead:

```bash
python quickstart.py
```

BioIR logs model and asset loading before printing the summary. A successful
run resembles:

```json
{
  "record_id": "T1031",
  "output_path": "output/serial/T1031.cif",
  "model_inference_time_s": 2.69,
  "score_keys": [
    "iptm",
    "max_pae",
    "pae",
    "plddt",
    "ptm"
  ],
  "ptm": 0.508,
  "mean_plddt": 0.65,
  "pae_shape": [
    95,
    95
  ]
}
```

Inference time and scores can vary across GPUs and releases. The stable result
is a non-empty `output/serial/T1031.cif` structure and a score payload with the
keys shown in the previous example.

The summary contains:

- `output_path` — the predicted CIF structure.
- `model_inference_time_s` — the synchronized GPU model-forward time. It does
  not include asset downloads, preprocessing, or output writing.
- `ptm` — the predicted TM score returned by Boltz-2.
- `mean_plddt` — the mean of the returned pLDDT confidence values.
- `pae_shape` — the dimensions of the predicted aligned error matrix.

## Next Steps

- Learn how to provide MSAs, templates, nucleic acids, and ligands in
  [Input Requests][input-requests].
- Process independent requests across several GPUs in
  [Ray Multi-GPU Inference][ray-inference].
- Configure models and runtime arguments with
  [`build_processor`][build-processor].
- Check supported models, inputs, and GPUs in the
  [Support Matrix][support-matrix].

[ray-inference]: ray.md
[support-matrix]: ref/support-matrix.md
