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
"""Kernel configs for the windowed attention kernels.

One file per ``(head_dim, n_queries, n_keys, sm)``, named ``D{head_dim}_W{n_queries}_K{n_keys}_sm{sm}.json``; a
window shape and GPU without one is not served. Its ``kernel_abi`` names the kernel the GPU runs: ``sm90``
(cp.async and WGMMA) on SM90, ``sm80`` (cp.async and warp-level MMA) on the others. The kernels derive their
tiling from the window shape, so the file's single ``default`` entry holds no tile parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_KEY = "default"
KERNEL_ABIS = ("sm80", "sm90")


def config_file_name(sm: int, head_dim: int, n_queries: int, n_keys: int) -> str:
    """Return the config file name for one window shape on one SM."""
    return get_config_file_name(sm, D=head_dim, W=n_queries, K=n_keys)


def validate_config(kernel_abi: str, kernel_variant: str | None, configs: Mapping[str, Any], name: str) -> str:
    """Return the kernel ABI of a config file's contents, or raise if they name anything else."""
    if kernel_abi not in KERNEL_ABIS:
        raise ValueError(f"{name}: window attention has no {kernel_abi!r} kernel; expected one of {KERNEL_ABIS}")
    if kernel_variant is not None:
        raise ValueError(f"{name}: window attention kernels have no variants")
    if dict(configs) != {CONFIG_KEY: {}}:
        raise ValueError(f"{name}: window attention configs must be {{{CONFIG_KEY!r}: {{}}}}")
    return kernel_abi


def config_kernel_abi(sm: int, head_dim: int, n_queries: int, n_keys: int) -> str | None:
    """The kernel ABI configured for one window shape on ``sm``, or ``None`` when there is no config."""
    name = config_file_name(sm, head_dim, n_queries, n_keys)
    bundle = load_kernel_configs(str(CONFIGS_DIR), name)
    if bundle is None:
        return None
    return validate_config(bundle.kernel_abi, bundle.kernel_variant, bundle.configs, name)
