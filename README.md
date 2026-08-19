<div align="center">

# BioNeMo Inference Runtime

## Easy, fast, and memory-efficient structure prediction inference

GPU-accelerated inference for protein, nucleic-acid, and ligand structure
prediction models — from FASTA/MSA to PDB/mmCIF.

</div>

## About

BioNeMo Inference Runtime (BioIR) is NVIDIA's library for structure-prediction
inference. A five-stage GPU pipeline turns AlphaFold-lineage and all-atom models
into PDB/mmCIF with confidence scores. Models stay ordinary `nn.Module`s — no
TensorRT engine build.

On H100, BioIR is **2.4–3.8x** faster than the best open-source `torch.compile`
baselines at matched accuracy. See [Benchmarks](#benchmarks).

## Documentation

User guides and technical reference live under [`docs/`](docs/):

| Doc                                           | Description                                           |
| --------------------------------------------- | ----------------------------------------------------- |
| [Developer guide](docs/dev.md)                | Build, test, stage weights, contribute                |
| [API reference](docs/ref/api.md)              | `build_processor`, model constructors, inputs/outputs |
| [Architecture](docs/ref/architecture.md)      | Five-stage pipeline and runtime design                |
| [Config architecture](docs/ref/config.md)     | Model `BaseConfig` tree and pipeline stage configs    |
| [Support matrix](docs/ref/support-matrix.md)  | Models, GPUs, and fused kernels                       |
| [Model weights](docs/ref/model-weights.md)    | Checkpoint resolution and staging                     |
| [Coding guidelines](docs/coding.md)           | Style, naming, and tooling                            |
| [Folding example](examples/folding/README.md) | Runnable `build_processor` demo                       |

## Benchmarks

Warm model-forward speedup vs each model's open-source implementation
(27-structure set, matched hyperparameters).

| SKU            | Model     | vs OSS eager | vs OSS `torch.compile` |
| -------------- | --------- | ------------ | ---------------------- |
| H100 80GB HBM3 | OpenFold3 | 2.93x        | **2.39x**              |
| H100 80GB HBM3 | Boltz-2   | 3.69x        | **2.87x**              |
| H100 80GB HBM3 | OpenFold2 | 3.81x        | **3.78x**              |

## Getting Started

You need a linux machine with a GPU, a driver from the 535 series or newer, and
Docker and the NVIDIA container runtime. Clone the repo, pull submodules and LFS
objects.

```bash
git lfs install \
    && GIT_LFS_SKIP_SMUDGE=0  \
    git clone --recurse-submodules \
        https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime.git \
    && cd BioNeMo-Inference-Runtime
```

Then, build the dev image and open a shell in it:

```bash
docker/dev.sh
```

The image carries the dependencies; your checkout is bind-mounted, so install
the package once inside and fold something:

```bash
pip install --no-build-isolation -e '.[dev]'
scripts/fetch_weights.sh --model boltz-2
python examples/folding/run_demo.py --output-dir output
```

Checkpoints come from their upstream publishers and need no NVIDIA credentials;
anything that cannot be fetched is skipped, and the tests needing it skip too.
Running `scripts/run_tests.sh` stages weights and runs the suite the way CI
does.

Building without a container needs more than a Python environment — see
[`docs/dev.md`](docs/dev.md#prerequisites) for the prerequisites and the wheel
build. The rest of that page covers daily development;
[`docker/README.md`](docker/README.md) covers the images and what
`docker/dev.sh` mounts.

## Contributing

We welcome contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for policy
and [`docs/dev.md`](docs/dev.md) for the development workflow.

## Citation

If you use BioIR in your research, please cite it via
[`CITATION.cff`](CITATION.cff).

## Contact / Support

- Bugs and feature requests:
  [GitHub Issues](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/issues/new/choose)
- Usage questions:
  [GitHub Discussions](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/discussions)
- Security vulnerabilities: see [`SECURITY.md`](SECURITY.md) — do **not** file a
  public issue

## License

This project is licensed under the [Apache License 2.0](LICENSE).
