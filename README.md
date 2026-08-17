<div align="center">

# BioNeMo Inference Runtime

## Easy, fast, and memory-efficient structure prediction inference

GPU-accelerated inference for protein, nucleic-acid, and ligand structure
prediction models — from FASTA/MSA to PDB/mmCIF.

</div>

## About

BioNeMo Inference Runtime (BioIR) is NVIDIA's library for
structure-prediction inference. A five-stage GPU pipeline turns
AlphaFold-lineage and all-atom models into PDB/mmCIF with confidence
scores. Models stay ordinary `nn.Module`s — no TensorRT engine build.

On H100, BioIR is **2.4–3.8x** faster than the best open-source
`torch.compile` baselines at matched accuracy. See
[Benchmarks](#benchmarks).

## Documentation

User guides and technical reference live under [`docs/`](docs/):

| Doc                                           | Description                                           |
| --------------------------------------------- | ----------------------------------------------------- |
| [Developer guide](docs/README.md)             | Local setup, build, and testing                       |
| [API reference](docs/ref/api.md)              | `build_processor`, model constructors, inputs/outputs |
| [Architecture](docs/ref/architecture.md)      | Five-stage pipeline and runtime design                |
| [Config architecture](docs/ref/config.md)     | Model `BaseConfig` tree and pipeline stage configs    |
| [Support matrix](docs/ref/support-matrix.md)  | Models, GPUs, and fused kernels                       |
| [Model weights](docs/model-weights.md)        | Checkpoint staging from NGC                           |
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
