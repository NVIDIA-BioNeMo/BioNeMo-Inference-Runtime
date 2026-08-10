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

import cutlass

from tensorrt_bionemo._torch._kernel_config_loader import (
    get_config_file_name,
    load_kernel_configs,
    resolve_implementation,
)

_VARIANT_KEY_RE = re.compile(r"^S=(\d+)$")


@dataclass(frozen=True)
class PairwiseAttentionLeftMaskKernelConfig:
    """A resolved source kernel and its architecture-specific validators."""

    arch: str
    kernel_factory: Callable[[int], Any]
    can_implement: Callable[[type, int], bool]


def _build_sm80_config(
    kernel_cls: type,
    m_block_size: int,
    n_block_size: int,
    sm_version: int,
    num_threads: int = 128,
    swizzle_b: int = 3,
    load_bias_before_gemm: bool = True,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Bind an Ampere kernel class to one tuned tile configuration."""

    def factory(head_dim: int):
        return kernel_cls(
            head_dim,
            m_block_size,
            n_block_size,
            num_threads,
            swizzle_b=swizzle_b,
            load_bias_before_gemm=load_bias_before_gemm,
        )

    def can_impl(ct_dtype: type, head_dim: int) -> bool:
        return kernel_cls.can_implement(
            ct_dtype,
            head_dim,
            m_block_size,
            n_block_size,
            num_threads,
            sm_version,
        )

    return PairwiseAttentionLeftMaskKernelConfig(arch="sm80", kernel_factory=factory, can_implement=can_impl)


def _build_sm90_config(
    kernel_cls: type,
    mma_tiler_mn: tuple[int, int],
    is_persistent: bool,
    kv_stage: int = 5,
    raster_factor: int = 0,
    qk_acc_dtype: type = cutlass.Float32,
    pv_acc_dtype: type = cutlass.Float32,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Bind a Hopper kernel class to one tuned tile configuration."""
    mma_mn_tuple = tuple(mma_tiler_mn)

    def factory(head_dim: int):
        return kernel_cls(
            qk_acc_dtype,
            pv_acc_dtype,
            (mma_mn_tuple[0], mma_mn_tuple[1], head_dim),
            is_persistent,
            kv_stage=kv_stage,
            raster_factor=raster_factor,
        )

    def can_impl(ct_dtype: type, head_dim: int) -> bool:
        # Shape-independent config validation still needs representative
        # shapes that satisfy Hopper's batch/head divisibility checks.
        ok, _ = kernel_cls.can_implement(
            (1, 64, 1, head_dim),
            (1, 64, 1, head_dim),
            ct_dtype,
            qk_acc_dtype,
            pv_acc_dtype,
            mma_mn_tuple,
            is_persistent,
            1.0,
            1,
            kv_stage=kv_stage,
        )
        return ok

    return PairwiseAttentionLeftMaskKernelConfig(arch="sm90", kernel_factory=factory, can_implement=can_impl)


_PW_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "pairwise_attention")


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


def get_kernel_config(
    sm_version: int,
    head_dim: int,
    S: int,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Resolve the source kernel at the nearest tuned ``S`` anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, S)
    kernel_cls = resolve_implementation(bundle.implementation)
    if "mma_tiler_mn" in tile_params:
        return _build_sm90_config(kernel_cls, **tile_params)
    return _build_sm80_config(kernel_cls, sm_version=sm_version, **tile_params)
