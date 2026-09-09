#!/usr/bin/env bash
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

# Classify the host as DGX, HGX, datacenter (Tesla-class), or GeForce.
set -uo pipefail

dmi() { tr -d '\0' <"/sys/class/dmi/id/$1" 2>/dev/null | head -1; }

vendor=$(dmi sys_vendor)
product=$(dmi product_name)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "unknown — no NVIDIA driver (nvidia-smi not found): $vendor $product"
  exit 1
fi

smi=$(nvidia-smi -q 2>/dev/null)
brand=$(awk -F': +' '/Product Brand/ {print $2; exit}' <<<"$smi")
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
# shellcheck disable=SC2012
switches=$(ls -A /proc/driver/nvidia-nvswitch/devices 2>/dev/null | wc -l)

if [[ -r /etc/dgx-release ]] || [[ $vendor == NVIDIA* && $product == *DGX* ]]; then
  class="(a) DGX"
elif ((switches > 0)); then
  class="(b) HGX"
elif [[ $brand == GeForce* || $gpu == *GeForce* ]]; then
  class="(d) GeForce"
else
  class="(c) Tesla / datacenter"
fi

printf '%s\n' "$class"
printf '  system     : %s %s\n' "$vendor" "$product"
printf '  gpu        : %s x%s (brand: %s)\n' "$gpu" "$gpus" "$brand"
printf '  nvswitches : %s\n' "$switches"
