<div align="center">

# BioNeMo Inference Runtime

## Easy, fast, and memory-efficient structure prediction inference

GPU-accelerated inference for protein, nucleic-acid, and ligand structure
prediction models — from FASTA/MSA to PDB/mmCIF.

</div>

## About

BioNeMo Inference Runtime (BioIR) is a fast and easy-to-use library for
structure-prediction model inference. Built by NVIDIA as part of the BioNeMo
stack, it turns AlphaFold-lineage and all-atom folding models into a
production-ready GPU pipeline: parse, tokenize, featurize, fold, and write
structures with confidence scores.

On H100, BioIR delivers **2.4–3.8x** faster warm model-forward latency than
the best open-source `torch.compile` baselines for OpenFold3, Boltz-2, and
OpenFold2 — at matched accuracy (within ~0.005 lDDT). See
[Benchmarks](#benchmarks).

BioIR is fast with:

* State-of-the-art folding throughput on NVIDIA GPUs (Ampere through Blackwell)
* Fused triangle / pairwise kernels via precompiled CuTeDSL CUBINs — no
  kernel-JIT warmup on first use
* Auto backend selection (CuTeDSL, cuEquivariance, SDPA / PyTorch fallback)
* CUDA graph capture on diffusion modules for short-sequence sampling loops
* Triton fused ops (SwiGLU, layernorm + projection) across CUDA GPUs
* Pairwise-activation memory engineering — bf16 pairs, shortened lifetimes,
  never-materialize patterns, and memory-scaled auto-chunking for large *N*
* Weight fusion (QKV/KV, AdaLN, gates) with converters from upstream checkpoints
* A five-stage Ray Data pipeline that overlaps CPU prep with GPU folding and
  scales as one replica per GPU

BioIR is flexible and easy to use with:

* A single production entry point (`build_processor`) from `InputRequest` to
  PDB / mmCIF + confidence scores (pLDDT, pTM, ipTM, PAE)
* A model-only API for custom dataloaders: construct the `nn.Module`, call
  `forward` on a feature dict
* Remains PyTorch end to end — no TensorRT engine build between checkpoint and
  forward pass; models stay ordinary `nn.Module`s
* Shared optimized layers (`Pairformer`, `Evoformer`, diffusion transformers)
  you can drop into your own architecture
* Serial mode for debugging and per-request timing; Ray mode for cluster
  throughput
* Per-row fault tolerance so a bad request does not take down the batch
* Checkpoints from local cache, Hugging Face, or NGC (`BIOIR_CACHE`)
* Wheels that ship the compiled C++/CUDA extension and embedded CUBINs

BioIR supports major structure-prediction families, including:

* AlphaFold2 and AlphaFold2-Multimer
* OpenFold2 (finetuning, pTM, and no-template variants)
* Boltz-1 and Boltz-2 (protein, RNA/DNA, ligands; oligomers)
* OpenFold3 (all-atom; protein, nucleic acids, ligands)
* Protenix v2 (compute module; pipeline support in progress)

## Documentation

User guides and technical reference live under [`docs/`](docs/):

| Doc                                           | Description                                           |
| --------------------------------------------- | ----------------------------------------------------- |
| [Developer guide](docs/README.md)             | Local setup, build, and testing                       |
| [API reference](docs/ref/api.md)              | `build_processor`, model constructors, inputs/outputs |
| [Architecture](docs/ref/architecture.md)      | Five-stage pipeline and runtime design                |
| [Support matrix](docs/ref/support-matrix.md)  | Models, GPUs, and fused kernels                       |
| [Model weights](docs/model-weights.md)        | Checkpoint staging from NGC                           |
| [Coding guidelines](docs/coding.md)           | Style, naming, and tooling                            |
| [Folding example](examples/folding/README.md) | Runnable `build_processor` demo                       |

## Benchmarks

Mean warm model-forward latency on **H100 80GB HBM3**, same inputs and
hyperparameters as each model's open-source implementation (27-structure set).
Accuracy stays within ~0.005 lDDT of the OSS baselines.

| Model     | vs OSS eager | vs OSS `torch.compile` |
| --------- | ------------ | ---------------------- |
| OpenFold3 | 2.93x        | **2.39x**              |
| Boltz-2   | 3.69x        | **2.87x**              |
| OpenFold2 | 3.81x        | **3.78x**              |

Absolute latency (mean warm `model_forward_s`):

| Model     | OSS eager | OSS compile | **BioIR**   |
| --------- | --------- | ----------- | ----------- |
| OpenFold3 | 51.83 s   | 42.30 s     | **17.69 s** |
| Boltz-2   | 54.04 s   | 41.91 s     | **14.63 s** |
| OpenFold2 | 21.50 s   | 21.34 s     | **5.65 s**  |

## Getting Started

### Develop Docker

Build the development image directly from the NVIDIA PyTorch base:

```bash
make -C docker bioir_dev REGISTRY_IMAGE=bionemo-ir TAG=dev
```

Run the resulting image:

```bash
docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it \
  bionemo-ir:dev
```

The repository and development dependencies are already installed in editable
mode.

#### Testing

```bash
pytest -s $(pwd)/tests
```

#### Model weights

Model checkpoints for the test suite and benchmarks are centralized in the NGC
org `<ngc-org>/<ngc-team>` (the single source of truth for CI, local runs, and
benchmarking). Stage them — and run the full suite the way CI does — with:

```bash
export NGC_API_KEY=<NGC_API_KEY>          # read access to <ngc-org>/<ngc-team>
.gitlab/ci/scripts/run_tests.sh       # download + stage weights, then run tests
.gitlab/ci/scripts/run_tests.sh --download   # just stage weights and exit
```

See [`docs/model-weights.md`](docs/model-weights.md) for the design and for how
to **upload a new model's weights** to `bioair` so CI/benchmarks pick them up.

### Release Docker

Use the same development docker to build the wheel package:

```bash
pip install build
python -m build --wheel --no-isolation --outdir packages/
```

## Contributing

We welcome contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for policy
and [`docs/README.md`](docs/README.md) for local development.

## Citation

If you use BioIR in your research, please cite it via
[`CITATION.cff`](CITATION.cff).

## Contact / Support

* Bugs and feature requests:
  [GitHub Issues](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/issues/new/choose)
* Usage questions:
  [GitHub Discussions](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/discussions)
* Security vulnerabilities: see [`SECURITY.md`](SECURITY.md) — do **not** file
  a public issue

## License

This project is licensed under the [Apache License 2.0](LICENSE).
