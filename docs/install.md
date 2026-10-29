---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# BioIR Package Installation

The following sections provide the steps to install the BioNeMo Inference
Runtime (BioIR) package, along with the software and hardware requirements.

## Requirements to Install the BioIR Wheel

The release wheel has the following software and hardware requirements:

- **Operating system:**
  [A Linux distribution that supports the latest NVIDIA drivers][linux-requirements],
  with glibc 2.34 or newer. Release wheels are tagged `manylinux_2_34`, so pip
  reports no matching distribution on an older distribution such as Ubuntu 20.04
  or RHEL 8.
- **CPU architecture:** x86_64 (amd64) or aarch64 (arm64).
- **GPU:** An NVIDIA GPU listed in the [support matrix][support-matrix].
- **NVIDIA Driver:** [580][driver-580] or higher.
- **Python:** CPython 3.12, 3.13, or 3.14. Release wheels use `cp312-abi3`.
- **Build tools:** A C/C++ compiler and the active Python's development headers.
  See [Install Build Tools][build-tools].

[linux-requirements]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/introduction.html#linux-system-requirements
[support-matrix]: ref/support-matrix.md#gpus
[driver-580]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06
[build-tools]: #install-build-tools

Use the commands in [Collecting System Information][system-information] to
verify these requirements. If the installed NVIDIA driver is older than version
580, refer to the [GPU Stack][gpu-stack] guide.

[system-information]: ref/system-information.md
[gpu-stack]: ref/gpu-stack.md

### GPU Requirements

H200, H100, A100, L40S, GB200, and GB300 are release-qualified with measured
speed, peak memory, and accuracy in the [benchmarks][benchmarks].

[benchmarks]: ref/benchmark.md

BioIR also runs on other NVIDIA GPUs, including Ampere, Ada Lovelace, Hopper,
and Blackwell. The [support matrix][support-matrix] distinguishes backend
compatibility from release qualification and lists the optimized kernels for
each architecture.

### Driver Requirements

Use NVIDIA driver 580 or newer. BioIR is built against CUDA 13.2, but the wheel
does not need an NGC image or a CUDA Toolkit installed on the host. It requires
the driver-provided `libcuda.so.1`.

<Error>
Drivers older than 580 are unsupported. The CUDA forward-compatibility shim
caused hangs and crashes during inference testing on older drivers.
</Error>

Check the installed GPU and driver with commands at [Collecting System
Information][system-information].

## Choose an Installation Method

Choose one of the following installation methods based on your task:

- **Release wheel (recommended):** Install BioIR into a Python environment to
  call its models, optimized modules, and prediction pipeline from your code.
  This method is the simplest way to start using BioIR.
- **Source build:** Use the development container when changing BioIR or running
  its test suite. Follow the
  [development workflow][development-prerequisites]
  for Docker, NVIDIA Container Toolkit, compiler, and CUDA-header requirements.

[development-prerequisites]: dev.md#prerequisites-for-bioir-development-workflow

## Install the Release Wheel

BioIR is published on [PyPI][pypi] as `bionemo-ir`, with one wheel for each
supported CPU architecture. Each wheel supports CPython 3.12 through 3.14 using
the Python stable ABI. pip selects the wheel that matches the host.

[pypi]: https://pypi.org/project/bionemo-ir/

### Create a Python Environment

The release wheel supports CPython 3.12 through 3.14. A dedicated Python
environment is **strongly recommended**. It isolates BioIR and its dependencies
from the system Python and other projects. The examples below use Python 3.12;
replace `3.12` with `3.13` or `3.14` to use another supported version.

Choose one of the following methods to create and activate an environment:

[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
[conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

<Tabs>
<Tab title="uv (recommended)">

[Install `uv`][uv-install]. Its installer script needs `curl` or `wget`; on
Ubuntu, install `curl` with
`sudo apt-get update && sudo apt-get install --yes curl`. Open a new shell so
`uv` is on `PATH`, then run:

```bash
uv venv --python 3.12 --managed-python --seed .venv
source .venv/bin/activate
```

The `--managed-python` flag uses a uv-managed Python, which includes the
`Python.h` headers, instead of a system Python. The `--seed` flag installs
`pip` in the environment for the shared install commands below.

</Tab>
<Tab title="conda">
[Install conda][conda], then create and activate a named environment:

```bash
conda create --name bioir python=3.12
conda activate bioir
```

</Tab>
<Tab title="Python venv">
Install Python 3.12 on the host. On Ubuntu and Debian, the `venv` module ships
in a separate package; without it, `python3.12 -m venv` fails with
`ensurepip is not available`. Install it, or use the uv tab, which needs no
administrator access:

```bash
sudo apt-get update
sudo apt-get install --yes python3.12 python3.12-venv
```

On RHEL-family systems, the `python3.12` package includes `venv`:

```bash
sudo dnf install --assumeyes python3.12
```

Then create and activate an environment with Python's built-in `venv` module:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

</Tab>
</Tabs>

Keep the environment activated for the rest of this guide and for later work,
including the [Quickstart][quickstart]. Confirm that it uses your chosen
supported Python version, then upgrade `pip`:

[quickstart]: quickstart.md

```bash
python --version
python -m pip install --upgrade pip
```

### Install Build Tools

Every host needs a C compiler and the `Python.h` headers of the environment's
Python interpreter. Triton, a PyTorch dependency, compiles a small C launcher
with them the first time BioIR runs a model.

pip also builds a dependency from source when PyPI has no wheel for the
environment's Python version and CPU architecture. The dependencies below need
native build tools in the release's tested dependency set. Other resolutions may
differ; source builds add minutes to the install.

| Python | x86_64    | aarch64                    |
| ------ | --------- | -------------------------- |
| 3.12   | None      | `kalign-python`, `biotraj` |
| 3.13   | None      | `kalign-python`, `biotraj` |
| 3.14   | `biotraj` | `kalign-python`, `biotraj` |

pip builds `ihm` from source on every host. Its C extension is optional, so
`ihm` installs without a compiler.

uv-managed and conda Pythons include `Python.h`. A Python from the
distribution's package manager needs its matching development package as well.
Check the environment's interpreter:

```bash
python -c "import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_path('include'), 'Python.h').is_file())"
```

Install the compilers, then the development package only when the check
prints `False`. The development package examples use Python 3.12; substitute
your environment's Python minor version when it differs:

<Tabs>
<Tab title="Ubuntu">

```bash
sudo apt-get update
sudo apt-get install --yes build-essential
```

```bash
sudo apt-get install --yes python3.12-dev
```

</Tab>
<Tab title="RHEL family">

```bash
sudo dnf install --assumeyes gcc gcc-c++ make
```

```bash
sudo dnf install --assumeyes python3.12-devel
```

</Tab>
</Tabs>

To list the dependencies that build from source on this host, run the
following check before installing BioIR. It compiles nothing:

```bash
python -m pip install --dry-run --only-binary=:all: --no-binary=ihm bionemo-ir
```

The result means:

- **Ends with `Would install`:** Every dependency except `ihm` installs from a
  wheel.
- **`ResolutionImpossible`:** The lines under `The conflict is caused by:`
  name a dependency that builds from source, such as `kalign-python`.
- **`No matching distribution found for bionemo-ir`:** The Python version or
  glibc does not match the release wheel. Check the
  [requirements][requirements].
- **`No matching distribution found for` another package:** That dependency
  builds from source.

pip names only the first dependency that builds from source.

[requirements]: #requirements-to-install-the-bioir-wheel

### Install the Package

The [external materials notice][external-materials] applies to the dependencies
downloaded separately by pip.

[external-materials]: overview.mdx#notice-and-disclaimer

Install BioIR into the active environment:

```bash
python -m pip install bionemo-ir
```

pip resolves PyTorch, the CUDA libraries, and the remaining dependencies from
the same index. The wheel carries the GPU kernels precompiled, so neither
installing nor running BioIR needs `nvcc` or the CUDA Toolkit.

If pip fails to build a dependency, or the first prediction fails to compile
Triton's launcher, match the error to the [build tools][build-tools] it needs:

- `fatal error: Python.h: No such file or directory`,
  `Could NOT find Python (missing: ... Development.Module)`, or
  `Imported target "pybind11::module" includes non-existent path`: Install
  the `-dev` or `-devel` package for your Python version, for example
  `python3.12-dev` or `python3.12-devel`. During a prediction, the first error
  appears above a `CalledProcessError` naming Triton's `driver.c`.
- `Failed to find C compiler`, `No such file or directory: 'gcc'` or `'cc'`,
  `no acceptable C compiler found`, or a CMake error naming
  `CMAKE_C_COMPILER` or `CMAKE_CXX_COMPILER`: Install the compilers.

Verify the install:

```bash
python -c "import bionemo_ir; print(bionemo_ir.__version__)"
```

<Note>
On aarch64, `python -m pip check` can report
`nvidia-cusparselt-cu13 0.8.1 is not supported on this platform` because the
upstream cuSPARSELt wheel declares an incorrect internal platform tag. This
warning does not prevent installation or package imports. No post-install
repair is required.
</Note>

## Next Steps

- Run your first Boltz-2 prediction in the [Quickstart][quickstart].
- Use the [`build_processor` API][build-processor] to run a supported
  structure-prediction pipeline.
- Review the [model and GPU support matrix][model-gpu-support].
- Follow the [development workflow][development] to build, test, or contribute
  to BioIR.

[build-processor]: ref/api.md#build_processor
[model-gpu-support]: ref/support-matrix.md
[development]: dev.md
