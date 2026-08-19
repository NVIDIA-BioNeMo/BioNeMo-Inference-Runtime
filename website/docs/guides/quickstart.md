# Quickstart

Install BioNeMo Inference Runtime (BioIR) and fold your first sequence.

## Prerequisites

- A Linux machine with an NVIDIA GPU (driver 535 series or newer)
- Docker with the NVIDIA container runtime

## Install

Clone the repo with submodules and LFS objects:

```bash
git lfs install \
    && GIT_LFS_SKIP_SMUDGE=0 \
    git clone --recurse-submodules \
        https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime.git \
    && cd BioNeMo-Inference-Runtime
```

Build the dev image and open a shell in it:

```bash
docker/dev.sh
```

The image carries all dependencies and the checkout is bind-mounted, so
install the package once inside the container:

```bash
pip install --no-build-isolation -e '.[dev]'
```

## Fold a sequence

Stage the Boltz-2 checkpoint and run the demo:

```bash
scripts/fetch_weights.sh --model boltz-2
python examples/folding/run_demo.py --output-dir output
```

The demo parses sequences and MSAs, featurizes them, runs inference, and
writes PDB/mmCIF plus a `{id}_scores.json` sidecar with pLDDT / pTM / ipTM.
Checkpoints come from their upstream publishers; anything that cannot be
fetched is skipped.

## Use the Python API

The same five-stage pipeline, in-process:

```mermaid
graph LR
    A[Parser] --> B[Tokenizer] --> C[Feature generator]
    C --> D[Folding engine] --> E[Writer]
```

```python
from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import WriterStageConfig

config = EngineProcessorConfig(
    model_source="boltz-2",
    executor_backend=None,  # serial, in-process
    writer_stage=WriterStageConfig(output_path="output", format="cif"),
)
processor = build_processor(config)

request = InputRequest(
    input_id="demo",
    polymers=[
        Polymer(
            polymer_type="protein",
            chain_id=["A"],
            sequence="GSHMSL...",
            msas=[MSARecord(path="msa.a3m", format="a3m")],
        )
    ],
)
outputs = processor([{"record": request, "__record_id": request["input_id"]}])
# outputs[0]["output_path"] -> output/demo.cif
```

## Next steps

- [API reference](../development/ref/api.md) — `build_processor` in detail,
  model constructors, inputs/outputs, Ray multi-GPU execution.
- [Support matrix](../development/ref/support-matrix.md) — models, GPUs,
  and fused kernels.
- [Model weights](../development/ref/model-weights.md) — checkpoint
  resolution and staging.
- [Architecture](../development/ref/architecture.md) — the five-stage
  pipeline and runtime design.
