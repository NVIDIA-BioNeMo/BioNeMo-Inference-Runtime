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

- **Python**: 3.12+ (`requires-python`); the extension is built `cp312`.

- **Toolchain**: a C++17 compiler and CUDA toolkit headers, only to build the
  extension from source. CMake and nanobind are declared build dependencies, so
  pip supplies them.

Verify these requirements with the commands in
[Collecting System Information][system-information]. If the requirements are
not met, follow the [GPU Stack][gpu-stack] guide.

The kernels are **precompiled**. Neither building the wheel nor running it needs
`nvcc` — only the driver's `libcuda.so.1`.

Everything below works equally in a container or on a host that already has the
prerequisites.

For convenience, `docker/dev.sh` builds the dev image and opens a shell in it,
with the checkout and the caches — weights, ccache, pip, compiled kernels —
mounted from the host; refer to [Docker Images][docker-images].

[system-information]: ref/system-information.md
[gpu-stack]: ref/gpu-stack.md
[docker-images]: ref/docker-images.md
[support-matrix]: ref/support-matrix.md#gpus
[benchmarks]: ref/benchmark.md
[driver-580]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06
[docker-engine]: https://docs.docker.com/engine/install
[container-toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

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

That leaves you on the host checkout. Daily work is `docker/dev.sh` as above.
The wheel target in the next section is also run from the host, not from inside
that shell.

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

To build directly in the active environment, typically from inside the
development container, run:

```bash
# Run inside the development container or on a configured host.
pip wheel --no-deps --wheel-dir dist .
```

It shares `build/` with the [editable install](#editable-install), and it
deletes the extension from the source tree on its way through — every real build
does, so a stale copy cannot shadow the fresh one — but only an editable install
puts one back. Until you reinstall, `pytest` refuses to collect: the session
starts by requiring the extension and stops with the reason it could not load
it.

### Editable Install

```bash
pip install -e '.[dev]'
```

It compiles `bionemo_ir.libs._cutedsl_kernels`, a nanobind extension embedding
the CUBIN packs under `cpp/kernels/cutedsl_*/cubins/`. Verify it:

```bash
python -c "import bionemo_ir.libs._cutedsl_kernels; print('ok')"
```

pip builds this in an isolated environment and installs the build requirements
declared in `pyproject.toml` — `cmake`, `nanobind`, and `setuptools` — so the
command works on any interpreter, in the image or out of it.

Inside the dev image those three are already present, and `--no-build-isolation`
reuses them instead of re-resolving on every build. That is a speedup and
nothing more: pip installs the same pinned `nanobind==2.10.2` either way. **Do
not carry the flag outside the image.** It tells pip to skip installing the
build requirements, so on an interpreter that lacks them the build fails in
`cpp/cmake/deps/nanobind.cmake` with a message about build-system requirements —
which reads as a missing dependency rather than as the flag that suppressed it.

One thing the isolated build environment does not have is `torch`. The wheel's
CUDA tag comes from `nvcc` first and only falls back to `torch.version.cuda`, so
this is invisible wherever `nvcc` is on `PATH`. Where it is not, set the tag
rather than reaching for the flag:

```bash
CUDA_TAG=cu132 pip install -e '.[dev]'
```

## Run

### Data Dependencies

Two things are needed:

- **Inputs**: `examples/data/samples/` carries small, real inputs — monomers,
  heterooligomers, `RNA/DNA/ligand` complexes, MSAs, and templates — so both
  routes below have something to fold without assembling your own data.

- **Weights**: pull the Boltz-2 checkpoint into the shared cache.

  ```bash
  scripts/fetch_weights.sh --model boltz-2
  ```

  OpenFold3 uses a gated Hugging Face checkpoint. Create a Hugging Face account,
  request access and accept the terms on [`OpenFold/OpenFold3`][openfold3-hf],
  then authenticate and fetch the checkpoint:

  ```bash
  hf auth login
  scripts/fetch_weights.sh --model openfold3
  ```

  For a non-interactive environment, export `HF_TOKEN` instead of running
  `hf auth login`:

  ```bash
  export HF_TOKEN=hf_...
  scripts/fetch_weights.sh --model openfold3
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
python examples/folding/run_demo.py            # Boltz-2 on the bundled T1031 sample
python examples/folding/run_demo.py \
    --model-source openfold3 \
    --input examples/data/samples/monomers/T1031.json \
    --output-dir output
python examples/folding/run_demo.py --help
```

The default run writes `T1031.cif` and `T1031_scores.json` to the output
directory, and takes a few minutes on an A100. Which model sources the demo can
drive is in [the folding demo README](../examples/folding/README.md) —
it is narrower than the support matrix.

## Test

```bash
scripts/run_tests.sh
```

This stages missing model weights, then runs pytest in two phases: an
xdist-parallel bulk phase, then the trees that must run serially.

Expect a green run with a large number of skips, from missing checkpoints or
inputs — what actually runs depends on what you staged.

```bash
scripts/run_tests.sh --no-weights   # skip staging; run whatever already resolves
scripts/run_tests.sh --help
```

For one tree, file, or case, call `pytest` directly:

```bash
pytest -q tests/ops                     # kernel-level tests
pytest -q tests/_torch                  # module and backend tests
pytest -q tests/ops/test_gated_sigmoid.py::test_gated_sigmoid_config_selection_is_source_free
```

## Lint and Format

Style is enforced by [prek](https://github.com/j178/prek), which runs the hooks
pinned in `prek.toml`. `.[dev]` already installs it; wire up the git hooks once:

```bash
prek install
```

They then run on `git commit`. To run them by hand:

```bash
prek run              # staged files
prek run --all-files  # everything
```

A clean `prek run --all-files` is required for every PR, and CI runs the same
hooks. `prek.toml` lists them: a formatter and a linter per language in the
tree, plus the SPDX license header every source file carries.

## Open a Pull Request

Fork, then sign commits with `git commit -s`; refer to
[Contributing](contributing.md). Ensure all three pass:

- **Style** — `prek run --all-files`
- **Build** — `pip install -e '.[dev]'`, then import the extension
- **Tests** — `scripts/run_tests.sh`

Write the MR/PR title as the commit you want in history; squash is the default.
Refer to [Commits](coding.md#commits).

[github-ssh]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
[openfold3-hf]: https://huggingface.co/OpenFold/OpenFold3
