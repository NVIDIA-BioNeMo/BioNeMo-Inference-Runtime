---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Development Workflow

How to build, run, test, and lint BioNeMo Inference Runtime (BioIR). For coding
style and conventions, refer to [Coding Guidelines](coding.md); for the public
API, refer to [Python API](ref/api.md).

## Prerequisites for BioIR Development Workflow

- **GPU**: An NVIDIA GPU listed in the [BioIR Support Matrix][support-matrix].
  - BioIR's CuTeDSL kernels, CUDA graphs, and CUDA runtime paths use NVIDIA GPU
    capabilities; they do not run on CPU-only or non-NVIDIA systems. Which
    architectures the fused kernels cover, and what the rest fall back to, is in
    [Support Matrix][support-matrix].
  - H200, H100, A100, L40S, GB200, and GB300 are release-qualified with measured
    speed, peak memory, and accuracy in the [benchmarks][benchmarks].

- **Driver**: minimum version [580][driver-580].
  - The NVIDIA Container Toolkit (below) reads the CUDA version requirement off
    the image label and enforces it at `docker run`.

- **Docker**: [Docker Engine][docker-engine] (minimum version 23.0.1), and the
  [NVIDIA Container Toolkit][container-toolkit] (minimum version 1.13.5).
  - They are needed for `docker/dev.sh` and `make -C docker wheel`.
  - Ignore them if the host already has Python and the toolchain below, and
    docker is not used.

- **Python**: 3.12 for source development. `.python-version` selects the latest
  3.12 patch through uv, the extension is built `cp312`, and the published
  package metadata remains `requires-python >=3.12,<4.0`.

- **uv**: 0.12.x on a configured host. [Install uv][uv-install] if it is not
  already available. The development image includes the repository's minimum
  accepted version.

- **Toolchain**: a C++17 compiler and CUDA toolkit headers, only to build the
  extension from source. CMake and nanobind are declared build dependencies, so
  the build frontend supplies them.

Verify these requirements with the commands in
[Collecting System Information][system-information]. If the requirements are
not met, follow the [GPU Stack][gpu-stack] guide.

The kernels are **precompiled**. Neither building the wheel nor running it needs
`nvcc` — only the driver's `libcuda.so.1`.

Host and development-container commands use uv and the committed lock. The
container reuses its preinstalled locked dependencies for native rebuilds.

For convenience, `docker/dev.sh` builds the dev image and opens a shell in it,
with the checkout and the caches — weights, ccache, uv/pip, compiled kernels —
mounted from the host; refer to [Docker Images][docker-images].

[system-information]: ref/system-information.md
[gpu-stack]: ref/gpu-stack.md
[docker-images]: ref/docker-images.md
[support-matrix]: ref/support-matrix.md#gpus
[benchmarks]: ref/benchmark.md
[driver-580]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06
[docker-engine]: https://docs.docker.com/engine/install
[container-toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/

## Clone the Repository

Configure [SSH authentication with GitHub][github-ssh], then clone the
repository and fetch its Git LFS objects and submodules:

```bash
git lfs install &&
  GIT_LFS_SKIP_SMUDGE=0 \
    git clone --recurse-submodules \
      git@github.com:NVIDIA-BioNeMo/BioNeMo-Inference-Runtime.git &&
  cd BioNeMo-Inference-Runtime
```

That leaves you on the host checkout.

## Create the development environment

### Configured host

From the repository root, create the locked Python 3.12 environment and build
the editable native extension:

```bash
uv sync --locked
uv run --locked python -c "import bionemo_ir.libs._cutedsl_kernels; print('ok')"
uv run --locked prek install
```

uv downloads a compatible Python when the host has no Python 3.12 interpreter.
It installs the `dev` dependency group by default. If `nvcc` is not on
`PATH`, set the wheel's CUDA tag explicitly:

```bash
CUDA_TAG=cu132 uv sync --locked
```

### Development container

Build the image and open its shell from the host:

```bash
docker/dev.sh
```

The image already contains the third-party dependencies. Verify the checked-out
lock, register the editable project in its writable environment, and install the
hooks inside the shell:

```bash
uv lock --check
uv pip install --no-deps -e .
prek install
```

## Build

### Build a Wheel

From the host, invoke the containerized wheel build. The source is copied into
the image rather than mounted, so your checkout is untouched and only `dist/` is
written.

```bash
# Run from the host; this command builds the wheel in a container.
make -C docker wheel
ls dist
```

The artifact is `bionemo_ir-<version>+cu<xyz>-cp312-cp312-linux_<arch>.whl`.

To build directly on a configured host or inside the development container, use
the lock-constrained repository wrapper:

```bash
scripts/build_wheel.sh --out-dir dist
```

The wrapper exports the exact `build` group from `uv.lock`, constrains the
isolated PEP 517 environment with its hashes, and invokes `uv build`. It shares
`build/` with the [editable install](#editable-install), and it deletes the
extension from the source tree on its way through — every real build does, so a
stale copy cannot shadow the fresh one — but only an editable install puts one
back. Until you reinstall, `pytest` refuses to collect: the session starts by
requiring the extension and stops with the reason it could not load it.

Restore the host editable build with:

```bash
uv sync --locked --reinstall-package bionemo-ir
```

### Editable Install

On a configured host, `uv sync` creates `.venv`, installs the locked `dev`
group and installs BioIR in editable mode. Activate that environment for plain
`python`, `pip`, and `prek` commands:

```bash
uv sync --locked
source .venv/bin/activate
```

Inside the development container, verify the lock and register the mounted
checkout without resolving dependencies again:

```bash
uv lock --check
uv pip install --no-deps -e .
```

Both paths compile `bionemo_ir.libs._cutedsl_kernels`, a nanobind extension
embedding the CUBIN packs under `cpp/kernels/cutedsl_*/cubins/`. Verify the
active environment:

```bash
python -c "import bionemo_ir.libs._cutedsl_kernels; print('ok')"
```

uv's host project workflow builds in an isolated environment constrained by the
`build` group in `uv.lock`. The image installs the same build and development
groups on top of the NGC Python environment so it keeps the tested torch,
Triton, and CUDA stack; `--no-deps` adds only the mounted checkout to its
writable development venv. Do not use the container command in a general
interpreter because it suppresses dependencies the package requires.

One thing the isolated build environment does not have is `torch`. The wheel's
CUDA tag comes from `nvcc` first and only falls back to `torch.version.cuda`, so
this is invisible wherever `nvcc` is on `PATH`. Where it is not, set the tag
rather than reaching for the flag:

```bash
# Configured host.
CUDA_TAG=cu132 uv sync --locked

# Development container.
CUDA_TAG=cu132 uv pip install --no-deps -e .
```

On a configured host, uv tracks `setup.py`, the CMake and C++ sources,
committed CUBIN indexes and packs, `build.env`, and build-affecting environment
variables in the editable project's cache key. A later `uv run` or `uv sync`
rebuilds the extension when one of those inputs changes. Python-only edits
remain immediately visible through the editable install.

### Change dependencies

Consumer dependency ranges live in `[project.dependencies]` in
`pyproject.toml`. Development-only requirements live in
`[dependency-groups]`. The focused `build` group pins PEP 517 requirements;
`release` adds artifact inspection tools and is not installed in the default
environment.

Regenerate and verify the exact development resolution after either dependency
source changes:

```bash
uv lock
uv sync --locked
```

The lock covers Python 3.12 on Linux x86_64 and aarch64. Resolution fails when a
wheel-only dependency does not support either architecture. Upgrade
deliberately with `uv lock --upgrade`, inspect the lock diff, then run the same
locked sync and test gates; routine `uv lock` preserves versions already
selected.

## Run

### Data Dependencies

Two things are needed:

The commands below use the configured-host form. Inside the development
container, omit the `uv run --locked` prefix.

- **Inputs**: `examples/data/samples/` carries small, real inputs — monomers,
  heterooligomers, `RNA/DNA/ligand` complexes, MSAs, and templates — so both
  routes below have something to fold without assembling your own data.

- **Weights**: pull the Boltz-2 checkpoint into the shared cache.

  ```bash
  uv run --locked scripts/fetch_weights.sh --model boltz-2
  ```

  OpenFold3 uses a gated Hugging Face checkpoint. Create a Hugging Face account,
  request access and accept the terms on [`OpenFold/OpenFold3`][openfold3-hf],
  then authenticate and fetch the checkpoint:

  ```bash
  uv run --locked hf auth login
  uv run --locked scripts/fetch_weights.sh --model openfold3
  ```

  For a non-interactive environment, export `HF_TOKEN` instead of running
  `hf auth login`:

  ```bash
  export HF_TOKEN=hf_...
  uv run --locked scripts/fetch_weights.sh --model openfold3
  ```

  Refer to [Model Weights](ref/model-weights.md) for the other
  families, cache layout, token alternatives, and manual checkpoint staging.

### Run From the Wheel

The wheel is self-contained: the CUBINs are compiled into the extension, so
running needs no source checkout, no CMake, and no CUDA toolkit — only a driver
providing `libcuda.so.1`.

In the dev image every dependency is already installed, so build the venv on top
of them and install the wheel alone:

```bash
python -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install --no-deps dist/bionemo_ir-*.whl
/tmp/venv/bin/python examples/folding/run_demo.py --output-dir /tmp/bioir-demo
```

An empty environment instead makes pip resolve the whole dependency closure —
torch, ray, and the rest — from PyPI, which takes a while on a cold pip cache:

```bash
python -m venv /tmp/venv
/tmp/venv/bin/pip install dist/bionemo_ir-*.whl
```

That is the slower path and the stricter one: `--no-deps` above says nothing
about whether the wheel's dependencies are declared correctly.

Python puts the script's own directory on `sys.path`, not the repo root, so this
imports the installed package even when run from the checkout. That makes it the
check that a change survives packaging: a module that only imports because the
source tree happened to be on `sys.path` fails here. The
[minimal runtime image](ref/docker-images.md#minimal-runtime-image) applies both
checks at once, on a CUDA base carrying nothing else.

### Run From the Code

With the editable install from [Build](#build):

```bash
uv run --locked python examples/folding/run_demo.py            # Boltz-2 on the bundled T1031 sample
uv run --locked python examples/folding/run_demo.py \
    --model-source openfold3 \
    --input examples/data/samples/monomers/T1031.json \
    --output-dir output
uv run --locked python examples/folding/run_demo.py --help
```

The default run writes `T1031.cif` and `T1031_scores.json` to the output
directory, and takes a few minutes on an A100. Which model sources the demo can
drive is in [the folding demo README](../examples/folding/README.md) —
it is narrower than the support matrix.

## Test

```bash
uv run --locked scripts/run_tests.sh
```

This stages missing model weights, then runs pytest in two phases: an
xdist-parallel bulk phase, then the trees that must run serially.

Expect a green run with a large number of skips, from missing checkpoints or
inputs — what actually runs depends on what you staged.

```bash
uv run --locked scripts/run_tests.sh --no-weights
uv run --locked scripts/run_tests.sh --help
```

For one tree, file, or case, call `pytest` directly:

```bash
uv run --locked pytest -q tests/ops
uv run --locked pytest -q tests/_torch
uv run --locked pytest -q tests/ops/test_gated_sigmoid.py::test_gated_sigmoid_config_selection_is_source_free
```

## Lint and Format

Style is enforced by [prek](https://github.com/j178/prek), which runs the hooks
pinned in `prek.toml`. The default `dev` group installs it; wire up the git
hooks once:

```bash
uv run --locked prek install
```

They then run on `git commit`. To run them by hand:

```bash
uv run --locked prek run
uv run --locked prek run --all-files
```

A clean `uv run --locked prek run --all-files` is required for every PR, and
CI runs the same hooks. `prek.toml` lists them: a formatter and a linter per
language in the tree, plus the SPDX license header every source file carries.
Inside the development container, use the equivalent bare `prek` commands.

## Open a Pull Request

Fork, then sign commits with `git commit -s`; refer to
[Contributing](contributing.md). Ensure all three pass:

- **Style** — `uv run --locked prek run --all-files`
- **Build** — `uv sync --locked`, then import the extension
- **Tests** — `uv run --locked scripts/run_tests.sh`

Write the MR/PR title as the commit you want in history; squash is the default.
Refer to [Commits](coding.md#commits).

[github-ssh]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
[openfold3-hf]: https://huggingface.co/OpenFold/OpenFold3
