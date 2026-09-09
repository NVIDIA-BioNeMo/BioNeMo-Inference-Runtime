---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{}
---

# GPU Stack

This page gives instructions or references to install software prerequisites for
the BioNeMo Inference Runtime (BioIR). The installation and setup workflows have
been tested with the following system architectures:

- Ubuntu 24.04 with amd64 or arm64 (aarch64)

Recall

- [Requirements to install the BioIR wheel][wheel-requirements]
- [Prerequisites for BioIR development workflow][development-prerequisites]

[wheel-requirements]: ../install.md#requirements-to-install-the-bioir-wheel
[development-prerequisites]: ../dev.md#prerequisites-for-bioir-development-workflow

If the NVIDIA Driver, Docker, and NVIDIA Container Toolkit requirements in the
chosen workflow above were not met, refer to the following.

## Install the NVIDIA Driver, Docker, and NVIDIA Container Toolkit Stack

### Installation Instructions by GPU System Category

The installation process for the NVIDIA Driver, Docker, and NVIDIA Container
Toolkit depends on the category of GPU on your host.

Run the command [Determine the GPU category][gpu-category]. With the output, use
the table below to find the instructions for installation the NVIDIA Driver,
Docker, and NVIDIA Container Toolkit Stack

[gpu-category]: system-information.md#3-determine-the-gpu-category

| GPU category       | Installation instructions                                           |
| ------------------ | ------------------------------------------------------------------- |
| DGX, HGX           | [NVIDIA DGX OS 7 User Guide: Installing the GPU Driver][dgx]        |
| Tesla (DataCenter) | [NVIDIA Tesla (datacenter) Driver Installation Guide][tesla-driver] |
| GeForce            | BioIR does not support GeForce devices                              |

[dgx]: https://docs.nvidia.com/dgx/dgx-os-7-user-guide/installing_on_ubuntu.html#installing-the-gpu-driver
[tesla-driver]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest

If the Driver, Docker, or NVIDIA Container Toolkit installation fails, refer to
[Troubleshooting][troubleshooting].
Refer to [References][references] below.

[troubleshooting]: #troubleshooting
[references]: #references

### References

- [NVIDIA DGX OS 7 User Guide: Installing the GPU Driver][dgx]
- [NVIDIA Tesla (datacenter) Driver Installation Guide][tesla-driver]
- [Docker Engine Installation][docker-engine]
- [NVIDIA Container Toolkit][container-toolkit]
- [NVIDIA CUDA Compatibility][cuda-compatibility]

[docker-engine]: https://docs.docker.com/engine/install/
[container-toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[cuda-compatibility]: https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html

## Troubleshooting

### Common Issues

**Driver version mismatch**: If `nvidia-smi` shows an older driver version,
ensure you have rebooted after installation.

**CUDA version**: The driver must support CUDA 13.0 or higher. Check
the CUDA version in the `nvidia-smi` output. If your system shows CUDA 12.x or
lower, you need to install a driver with major index 580 or higher.

To verify CUDA compatibility:

1. Check current driver: `nvidia-smi`
2. Verify CUDA version shows 13.0 or higher
3. If not, refer to [NVIDIA CUDA Compatibility][cuda-compatibility]

**Secure Boot**: If you have Secure Boot enabled, you might need to sign the
NVIDIA kernel modules or disable Secure Boot in your BIOS.

**Library version conflicts**: If you encounter library version conflicts,
ensure all old NVIDIA packages are removed before installing the new driver.

### Architecture-Specific Troubleshooting

#### amd64 / x86_64

**Remove previous driver versions:**

```bash
# Remove old drivers
sudo apt-get remove --purge nvidia-*
sudo apt-get autoremove

# Verify removal
ls /usr/lib/x86_64-linux-gnu/ | grep -i nvidia
```

**Package conflicts:**

```bash
# Clean package cache
sudo apt-get clean
sudo apt-get update

# Try installation again
sudo apt-get install -y cuda-drivers
```

#### arm64 / aarch64 DGX

**Kernel version issues:**

```bash
# Check current kernel
uname -r

# List available kernels
dpkg --list | grep linux-image

# Configure grub to use correct kernel, see https://docs.nvidia.com/dgx/dgx-os-7-user-guide/
```

**DGX-specific issues:**

```bash
# Verify fabricmanager (if using NVSwitch)
systemctl status nvidia-fabricmanager

# Check NVIDIA services
systemctl status nvidia-persistenced
systemctl status nvidia-dcgm
```

**Build errors for older kernel:**

- Ignore build errors for modules built for `6.14.0-1015-nvidia-64k`
- These errors are expected and do not affect functionality

### Getting Additional Help

If you continue to experience issues:

1. Check NVIDIA driver logs: `dmesg | grep -i nvidia`
2. Review Docker logs: `sudo journalctl -u docker.service`
3. Consult [NVIDIA Driver Installation Guide][tesla-driver]
4. For DGX systems: [NVIDIA DGX OS 7 User Guide][dgx]
