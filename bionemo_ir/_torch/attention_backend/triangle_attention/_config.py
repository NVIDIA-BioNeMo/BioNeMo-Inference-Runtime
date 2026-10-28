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
"""Tuned CuTeDSL source-kernel configuration for triangle attention."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo_ir._torch.utils.kernel import (
    CutlassNumeric,
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
    resolve_implementation,
)

_VARIANT_KEY_RE = re.compile(r"^S=(\d+)$")


@dataclass(frozen=True)
class TriangleAttentionLeftMaskKernelConfig:
    """A resolved source kernel and its architecture-specific validators."""

    arch: str
    kernel_factory: Callable[[int], Any]
    can_implement: Callable[[CutlassNumeric, int], bool]
    cache_identity: tuple[object, ...] = ()


_TRI_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_config_bundle(sm_version: int, head_dim: int):
    bundle = load_kernel_configs(
        _TRI_CONFIGS_DIR,
        get_config_file_name(sm_version, D=head_dim),
    )
    if bundle is None:
        raise ValueError(f"No triangle-attention config file for SM{sm_version}, head_dim={head_dim}.")
    return bundle


def _nearest_variant(configs: dict[str, Any], S: int) -> tuple[int, dict[str, Any]]:
    """Return the tile whose ``S=<anchor>`` key is nearest to ``S``."""
    if S < 0:
        raise ValueError(f"Triangle-attention S must be non-negative; got {S}")

    candidates: list[tuple[int, str]] = []
    for key in configs:
        match = _VARIANT_KEY_RE.fullmatch(key)
        if match is not None:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise ValueError(f"No triangle-attention S anchors are registered; available keys: {sorted(configs)}")

    anchor, key = min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))
    return anchor, dict(configs[key])


def get_nearest_bucket(sm_version: int, head_dim: int, S: int) -> int:
    """Return the nearest tuned per-sample side-length anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    bucket, _ = _nearest_variant(bundle.configs, S)
    return bucket


def get_kernel_config(
    sm_version: int,
    head_dim: int,
    S: int,
) -> TriangleAttentionLeftMaskKernelConfig:
    """Resolve the source kernel at the nearest tuned ``S`` anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, S)
    source = load_source_module(__package__)
    implementation = source.source_implementation(bundle.kernel_abi, None, bundle.kernel_variant)
    kernel_cls = resolve_implementation(implementation)
    if bundle.kernel_abi == "sm100":
        if tile_params:
            raise ValueError(f"SM100 triangle attention has no tunable tile parameters; got {tile_params}")
        return source._build_sm100_config(kernel_cls)
    if "mma_tiler_mn" in tile_params:
        return source._build_sm90_config(kernel_cls, **tile_params)
    return source._build_sm80_config(kernel_cls, sm_version=sm_version, **tile_params)
