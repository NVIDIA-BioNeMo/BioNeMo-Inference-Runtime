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
"""Tuned CuTeDSL source-kernel configuration for pairwise attention."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo_ir._torch.utils.kernel import (
    CutlassNumeric,
    KernelConfigBundle,
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
    resolve_implementation,
)

_VARIANT_KEY_RE = re.compile(r"^S=(\d+)$")


@dataclass(frozen=True)
class PairwiseAttentionLeftMaskKernelConfig:
    """A resolved source kernel and its architecture-specific validators."""

    arch: str
    kernel_factory: Callable[[int], Any]
    can_implement: Callable[[CutlassNumeric, int], bool]
    # Everything the bound kernel's machine code depends on beyond the variant.
    cache_identity: tuple = ()


_PW_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_config_bundle(sm_version: int, head_dim: int):
    bundle = load_kernel_configs(
        _PW_CONFIGS_DIR,
        get_config_file_name(sm_version, D=head_dim),
    )
    if bundle is None:
        raise ValueError(f"No pairwise-attention config file for SM{sm_version}, head_dim={head_dim}.")
    return bundle


def _nearest_variant(configs: dict[str, Any], S: int) -> tuple[int, dict[str, Any]]:
    """Return the tile whose ``S=<anchor>`` key is nearest to ``S``."""
    if S < 0:
        raise ValueError(f"Pairwise-attention S must be non-negative; got {S}")

    candidates: list[tuple[int, str]] = []
    for key in configs:
        match = _VARIANT_KEY_RE.fullmatch(key)
        if match is not None:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise ValueError(f"No pairwise-attention S anchors are registered; available keys: {sorted(configs)}")

    anchor, key = min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))
    return anchor, dict(configs[key])


def get_nearest_bucket(sm_version: int, head_dim: int, S: int) -> int:
    """Return the nearest tuned per-side sequence-length anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    bucket, _ = _nearest_variant(bundle.configs, S)
    return bucket


def _anchor_kernel_abi(bundle: KernelConfigBundle, tile_params: dict[str, Any]) -> str:
    """Pop an anchor's own ``kernel_abi``, which overrides the bundle's for that anchor."""
    kernel_abi = tile_params.pop("kernel_abi", bundle.kernel_abi)
    if kernel_abi not in ("sm80", "sm90", "sm100"):
        raise ValueError(f"{bundle.source_path}: unsupported pairwise-attention kernel_abi {kernel_abi!r}")
    return kernel_abi


def get_bucket_kernel_abi(sm_version: int, head_dim: int, bucket: int) -> str:
    """Return the kernel ABI that serves the tuned ``S=<bucket>`` anchor.

    An anchor may name its own ``kernel_abi``: on SM100 and SM103 short
    sequences run the Ampere kernel, whose smaller tiles pad less than the
    Blackwell kernel's 256-row work items.
    """
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, bucket)
    return _anchor_kernel_abi(bundle, tile_params)


def get_kernel_config(
    sm_version: int,
    head_dim: int,
    S: int,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Resolve the source kernel at the nearest tuned ``S`` anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, S)
    kernel_abi = _anchor_kernel_abi(bundle, tile_params)
    source = load_source_module(__package__)
    implementation = source.source_implementation(kernel_abi, None, bundle.kernel_variant)
    kernel_cls = resolve_implementation(implementation)
    if kernel_abi == "sm100":
        unknown = set(tile_params) - {"tail_split", "dynamic_scheduler"}
        if unknown:
            raise ValueError(f"Unknown SM100 pairwise-attention parameters {sorted(unknown)} in {bundle.source_path}")
        return source._build_sm100_config(kernel_cls, **tile_params)
    if "mma_tiler_mn" in tile_params:
        return source._build_sm90_config(kernel_cls, **tile_params)
    return source._build_sm80_config(kernel_cls, sm_version=sm_version, **tile_params)
