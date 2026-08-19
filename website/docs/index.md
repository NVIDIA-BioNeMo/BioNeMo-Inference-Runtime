# BioNeMo Inference Runtime

Easy, fast, and memory-efficient structure-prediction inference.
GPU-accelerated pipelines turn FASTA/MSA inputs into PDB/mmCIF structures
with confidence scores — for protein, nucleic-acid, and ligand models.
Models stay ordinary `nn.Module`s; there is no TensorRT engine build.

## Where to go

- [Quickstart](guides/quickstart.md) — install BioIR and fold your first
  sequence.
- [API Reference](reference/bionemo_ir/index.md) — every module of the
  `bionemo_ir` package, generated from docstrings.
- [Developer Docs](development/index.md) — build from source, pipeline
  architecture, config tree, support matrix, and contribution rules.
- [GitHub repository](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime)
  — source code, issues, and discussions.

## Supported models

- Boltz-1 / Boltz-2
- OpenFold2 / AlphaFold2
- OpenFold3
- Protenix

Per-model data coverage, GPU SKUs, and fused kernels:
[support matrix](development/ref/support-matrix.md).
