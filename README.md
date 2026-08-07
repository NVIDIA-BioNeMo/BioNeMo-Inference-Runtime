<div align="center">

# TensorRT-BioNemo

GPU-accelerated structure prediction model inference
<div align="left">

## Getting Started

### Develop Docker

Build the development image directly from the NVIDIA PyTorch base:

```bash
make -C docker trtbnm_dev REGISTRY_IMAGE=tensorrt-bionemo TAG=dev
```

Run the resulting image:

```bash
docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it \
  tensorrt-bionemo:dev
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
