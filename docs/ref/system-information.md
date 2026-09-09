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

# Collect System Information

To help you with the installation of BioNeMo Inference Runtime (BioIR), system
information can be obtained with the commands below.

## 1. Determine the OS Distribution Installed on the Host

```bash
# Check OS version
cat /etc/os-release
# Example output for Ubuntu:
# NAME="Ubuntu"
# VERSION="24.04.3 LTS (Noble Numbat)"
# ID=ubuntu
# VERSION_ID="24.04"
```

## 2. Determine the CPU Architecture

```bash
# Set CPU arch as environment variable, on Ubuntu/Debian system
export CPU_ARCH=$(dpkg --print-architecture)
echo "CPU_ARCH: ${CPU_ARCH}"
# Example output:
# amd64

# Set CPU arch as environment variable, on a non-Ubuntu/Debian system
export CPU_ARCH=$(uname -m)
echo "CPU_ARCH: ${CPU_ARCH}"
# Example output:
# x86_64
```

## 3. Determine the GPU Category

From the root of your BioNeMo-Inference-Runtime clone, run the script below with
possible output values:

- (a) DGX
- (b) HGX
- (c) Tesla / datacenter
- (d) GeForce

```bash
./scripts/print_gpu_system_category.sh
# Example output
(a) DGX
system     : Dell  Inc. Dell Pro Max with Station GB300
gpu        : NVIDIA GB300 x1 (brand: NVIDIA)
nvswitches : 0
--------------------------
```

## 4. Determine the GPU Model

```bash
# Check GPU model
nvidia-smi --query-gpu=name --format=csv
# Example output:
# name
# NVIDIA GB300
```

If you see a message like `Command 'nvidia-smi' not found`, then attempt to
determine GPU model with the command below:

```bash
# Check GPU model
lspci | grep -Ei "VGA|3D|Display" | grep -i NVIDIA
# Example output:
# 01:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
```

## 5. Determine the NVIDIA Driver Version Installed on the Host

If the `nvidia-smi` command in Item 4 above is successful, then run
the command below. Otherwise, the NVIDIA Driver is not correctly installed.

```bash
# Check NVIDIA Driver version
nvidia-smi --query-gpu=driver_version --format=csv
# Example output:
# driver_version
# 580.65.06
```

## 6. Determine the Docker Version Installed on the Host

```bash
# Check Docker version
docker --version
# Example output:
# Docker version 29.2.1, build a5c7197
```

## 7. Determine the NVIDIA Container Toolkit Version Installed on the Host

Refer to [NVIDIA Container Toolkit][container-toolkit] for version details.

```bash
# Check NVIDIA Container Toolkit version
nvidia-ctk --version
# Example output:
# NVIDIA Container Toolkit CLI version 1.19.0
# commit: ec7b4e2fa2caecad6d89be4a26029b831fe7503a
```

[container-toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
