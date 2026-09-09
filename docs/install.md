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
  [A Linux distribution that supports the latest NVIDIA drivers][linux-requirements].
- **CPU architecture:** x86_64 (amd64) or aarch64 (arm64).
- **GPU:** An NVIDIA GPU listed in the [support matrix][support-matrix].
- **NVIDIA Driver:** [580][driver-580] or higher.
- **Python:** Python 3.12. Release wheels are tagged `cp312`.
- **aarch64 build tools:** A C/C++ compiler and Python 3.12 development
  headers. Some transitive dependencies build from source on aarch64.

[linux-requirements]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/introduction.html#linux-system-requirements
[support-matrix]: ref/support-matrix.md#gpus
[driver-580]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06

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

Use NVIDIA driver 580 or newer. BioIR is built against CUDA 13.2, but the CUDA
Toolkit does not need to be installed on the host. BioIR requires the
driver-provided `libcuda.so.1`.

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

The latest [GitHub release][releases] contains one wheel for each supported CPU
architecture. Install the [`gh` CLI][gh-cli] and log in with read access to
`NVIDIA-BioNeMo/BioNeMo-Inference-Runtime`.

[releases]: https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/releases
[gh-cli]: https://cli.github.com/

### Install aarch64 Build Prerequisites

On Ubuntu 24.04 aarch64, install the compiler toolchain and Python headers used
to build transitive dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes build-essential python3.12-dev
```

The `python3.12-dev` package provides `Python.h`.

### Create a Python Environment

The release wheel requires Python 3.12 (`cp312`). A dedicated Python
environment is **strongly recommended**. It isolates BioIR and its dependencies
from the system Python and other projects.

Choose one of the following methods to create and activate an environment:

[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
[conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

<Tabs>
<Tab title="uv (recommended)">

[Install `uv`][uv-install], then run:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
```

The `--seed` flag installs `pip` in the environment for the shared install
commands below.

</Tab>
<Tab title="conda">
[Install conda][conda], then create and activate a named environment:

```bash
conda create --name bioir python=3.12
conda activate bioir
```

</Tab>
<Tab title="Python venv">
Install Python 3.12 on the host, then create and activate an environment with
Python's built-in `venv` module:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

</Tab>
</Tabs>

Keep the environment activated for the rest of this guide and for later work,
including the [Quickstart][quickstart]. Confirm that it uses Python 3.12,
then upgrade `pip`:

[quickstart]: quickstart.md

```bash
python --version
python -m pip install --upgrade pip
```

### Download the Wheel

Download the wheel that matches your CPU architecture. `uname -m` reports
`x86_64` or `aarch64`. With no release tag, `gh` selects the latest release.

Run this command on an `x86_64` host:

```bash
gh release download \
  --repo NVIDIA-BioNeMo/BioNeMo-Inference-Runtime \
  --pattern "*_amd64_bionemo_ir-*.whl" --pattern SHA256SUMS
```

Run this command on an `aarch64` host:

```bash
gh release download \
  --repo NVIDIA-BioNeMo/BioNeMo-Inference-Runtime \
  --pattern "*_arm64_bionemo_ir-*.whl" --pattern SHA256SUMS
```

### Verify the Checksum

Verify the downloaded assets against `SHA256SUMS`:

```bash
sha256sum --check --ignore-missing SHA256SUMS
```

### Restore the Wheel Filename

The downloaded asset name includes a bundle prefix, and GitHub replaces the
wheel version's `+` with `.`. Restore a pip-valid name after the checksum
check:

```bash
for asset in *_bionemo_ir-*.whl; do
  wheel=bionemo_ir-${asset#*_bionemo_ir-}
  wheel=$(printf '%s\n' "$wheel" | sed -E 's/\.cu([0-9]+)([.-])/+cu\1\2/')
  mv "$asset" "$wheel"
done
```

### Install the Wheel

Install the restored wheel into the active environment:

```bash
python -m pip install ./bionemo_ir-*.whl
```

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
