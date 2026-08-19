# Development workflow

How to build, run, test and lint BioIR. For coding style and conventions, see
[coding.md](coding.md); the [reference docs](./ref) cover everything else.

## Prerequisites

- **GPU** — an NVIDIA GPU. Which architectures the fused kernels cover, and what
  the rest fall back to, is in [`ref/support-matrix.md`](ref/support-matrix.md).
- **Driver** — new enough for the CUDA the base image carries. The container
  toolkit reads that requirement off the image label and enforces it at
  `docker run`.
- **Python** — 3.12 or newer (`requires-python`); the extension is built
  `cp312`.
- **Toolchain** — a C++17 compiler and CUDA toolkit headers, only to build the
  extension from source. CMake and nanobind are declared build dependencies, so
  pip supplies them.

The kernels are **precompiled**. Neither building the wheel nor running it needs
`nvcc` — only the driver's `libcuda.so.1`.

Everything below works equally in a container or on a host that already has the
prerequisites.

For convenience, `docker/dev.sh` builds the dev image and opens a shell in it,
with the checkout and the caches — weights, ccache, pip, compiled kernels —
mounted from the host; see [`../docker/README.md`](../docker/README.md).

## Build

### Build a wheel

Prefer the container: the source is copied into the image rather than mounted,
so your checkout is untouched and only `dist/` is written.

```bash
$ make -C docker wheel

$ ls dist
bionemo_ir-<version>+cu<xyz>-cp312-cp312-linux_<arch>.whl
```

The host build lands in the same place:

```bash
pip wheel --no-deps --wheel-dir dist .
```

It shares `build/` with the [editable install](#editable-install), and it
deletes the extension from the source tree on its way through — every real build
does, so a stale copy cannot shadow the fresh one — but only an editable install
puts one back. Until you reinstall, `pytest` refuses to collect: the session
starts by requiring the extension and stops with the reason it could not load
it.

### Editable install

```bash
pip install --no-build-isolation -e '.[dev]'
```

It compiles `bionemo_ir.libs._cutedsl_kernels`, a nanobind extension embedding
the CUBIN packs under `cpp/kernels/cutedsl_*/cubins/`. Verify it:

```bash
python -c "import bionemo_ir.libs._cutedsl_kernels; print('ok')"
```

`--no-build-isolation` reuses the build requirements the dev image already
carries — `cmake`, `nanobind` and `setuptools`, declared in `pyproject.toml` —
instead of pip building a throwaway environment and re-resolving them on every
build. It is also what pins the extension to the `nanobind` the image installed.
On a host without those three, drop the flag and let pip fetch them.

## Run

Two things are needed:

- **Inputs**: `examples/data/samples/` carries small, real inputs — monomers,
  heterooligomers, `RNA/DNA/ligand` complexes, MSAs and templates — so both
  routes below have something to fold without assembling your own data.

- **Weights**: pull the Boltz-2 checkpoint into the shared cache.

  ```bash
  scripts/fetch_weights.sh --model boltz-2
  ```

  See [`ref/model-weights.md`](ref/model-weights.md) for the other families, the
  cache layout, and the free HuggingFace token that gated checkpoints need.

### From the wheel

The wheel is self-contained: the CUBINs are compiled into the extension, so
running needs no source checkout, no CMake and no CUDA toolkit — only a driver
providing `libcuda.so.1`.

In the dev image every dependency is already installed, so build the venv on top
of them and install the wheel alone:

```bash
python -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install --no-deps dist/bionemo_ir-*.whl
/tmp/venv/bin/python examples/folding/run_demo.py --output-dir /tmp/bioir-demo
```

An empty environment instead makes pip resolve the whole dependency closure —
torch, ray and the rest — from PyPI, which takes a while on a cold pip cache:

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
[minimal runtime image](../docker/README.md#minimal-runtime-image) applies both
checks at once, on a CUDA base carrying nothing else.

### From the code

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
drive is in [`../examples/folding/README.md`](../examples/folding/README.md) —
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

For one tree, file or case, call `pytest` directly:

```bash
pytest -q tests/ops                     # kernel-level tests
pytest -q tests/_torch                  # module and backend tests
pytest -q tests/ops/test_gated_sigmoid.py::test_gated_sigmoid_config_selection_is_source_free
```

## Lint and format

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
tree, plus the SPDX licence header every source file carries.

## Open a pull request

Ensure all three pass:

- **Style** — `prek run --all-files`
- **Build** — `pip install -e '.[dev]'`, then import the extension
- **Tests** — `scripts/run_tests.sh`

PRs will be squash-merged. The title and description will become commit message
and body. So, write them the way you want the squashed commit to read.

See also [coding.md](coding.md#commits).
