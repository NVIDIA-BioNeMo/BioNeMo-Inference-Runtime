# Development workflow

This doc discusses how to build, run, test and check BioIR. For coding style and
conventions, see [coding.md](coding.md). Explore the [reference docs](./ref) for
advanced topics.

## Contents

- [Prerequisites](#prerequisites)
- [Build](#build)
- [Run](#run)
- [Test](#test)
- [Lint and format](#lint-and-format)
- [Before you open a pull request](#before-you-open-a-pull-request)

## Prerequisites

|           | Requirement                                      | Why                                                                                                                             |
| --------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| GPU       | NVIDIA GPU, compute capability `8.0/8.6/8.9/9.0` | The fused kernels ship as precompiled CUBINs for these architectures. Other GPUs fall back to portable PyTorch implementations. |
| Driver    | 535 or newer                                     | Enforced at `docker run` by the container toolkit.                                                                              |
| Python    | 3.12+                                            | `requires-python`; the extension is built `cp312`.                                                                              |
| Toolchain | A C++17 compiler and CUDA toolkit headers        | Only to build the extension from source. CMake and nanobind are declared build dependencies, so pip supplies them.              |

The kernels are **precompiled**. Neither building the wheel nor running it
needs `nvcc` — only the driver's `libcuda.so.1`.

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

# example wheel name
$ ls dist
bionemo_ir-0.6.0.dev0+cu132-cp312-cp312-linux_aarch64.whl
```

The host build lands in the same place:

```bash
pip wheel --no-deps --wheel-dir dist .
```

It shares `build/` with the [editable install](#editable-install), and it
deletes the extension from the source tree on its way through — every real build
does, so a stale copy cannot shadow the fresh one — but only an editable install
puts one back. Until you reinstall, the kernel tests quietly skip.

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

To run the project, we need:

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
about whether the wheel's dependencies are declared correctly. So does the
[minimal runtime image](../docker/README.md#minimal-runtime-image), on a CUDA
base with nothing else on it.

Python puts the script's own directory on `sys.path`, not the repo root, so this
imports the installed package even when run from the checkout. That makes it the
check that a change survives packaging: a module that only imports because the
source tree happened to be on `sys.path` fails here. The minimal runtime image
applies the same check on a CUDA base with no toolkit installed — see
[`../docker/README.md`](../docker/README.md).

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

This stages model weights, then runs pytest in two phases: an xdist-parallel
bulk phase, then the trees that must run serially. It writes JUnit XML under
`tmp/reports/`. Coverage is off by default because nothing local reads the
report; `COVERAGE=1` turns it on, which is what CI does.

Expect a green run with a large number of skips: every test whose checkpoint is
unstaged skips itself, so what you actually run depends on what staged. Read the
pass/skip counts the run prints at the end — those, not a green exit on its own,
are what tell you how much you covered.

```bash
scripts/run_tests.sh --no-weights   # skip staging; run whatever already resolves
scripts/run_tests.sh --help
```

For one tree, file or case, call pytest directly:

```bash
pytest -q tests/ops                     # kernel-level tests
pytest -q tests/_torch                  # module and backend tests
pytest -q tests/ops/test_gated_sigmoid.py::test_gated_sigmoid_config_selection_is_source_free
```

**Tests skip rather than fail when their inputs are missing.** A test needing a
checkpoint you have not staged skips itself, so a run with no weights at all
still exercises the kernels, the config loaders and the pipeline scaffolding. A
green run with many skips is expected; check the skip reasons to see what
staging would unlock. That includes the OpenFold3 module tests, whose checkpoint
is gated — see [`ref/model-weights.md`](ref/model-weights.md).

Tests that read CuTeDSL kernel _sources_ assert on a kernel class's tile schema.
Where those sources are absent, these tests skip.

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

A clean `prek run --all-files` is the whole style gate — nothing runs it for you
after you push. `prek.toml` lists the hooks: a formatter and a linter per
language in the tree, plus the SPDX licence header every source file carries.

## Before you open a pull request

Nothing runs automatically on a pull request yet, so no check will catch a break
for you. Run all three yourself:

| Check | Command                                              |
| ----- | ---------------------------------------------------- |
| Style | `prek run --all-files`                               |
| Build | `pip install -e '.[dev]'` then import the extension  |
| Tests | `scripts/run_tests.sh`                               |

The first two need no GPU. The suite does, and no part of it is GPU-free — if
you have none, say so in the pull request and a maintainer runs it on the
internal pipeline, which is where the GPU runners are.

Title your PR the way you want the squashed commit to read — see
[coding.md](coding.md#commits). Merged PRs are imported internally, merged
there, and synced back, so your PR is closed rather than showing as merged. Your
authorship and trailers are preserved.
