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
"""Tuned kernel configuration for gated sigmoid GEMM."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo_ir._torch._kernel_config_loader import (
    get_config_file_name,
    load_kernel_configs,
    resolve_implementation,
)
from bionemo_ir._torch._kernel_source_loader import load_source_module

M_SHORT_THRESHOLD = 1024
M_MEDIUM_THRESHOLD = 2048

_GS_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# JSON configs are keyed by (K, N, SM) and M-bucket left edge.
_M_BUCKET_KEY = {
    "short": 0,
    "medium": M_SHORT_THRESHOLD + 1,
    "long": M_MEDIUM_THRESHOLD + 1,
}


@dataclass(frozen=True)
class GatedSigmoidKernelConfig:
    """Resolved source-kernel configuration."""

    kernel_factory: Callable[[type, bool], Any]
    can_implement: Callable[[type], bool]
    tile_params: dict[str, Any]


def _build_kernel_config(
    kernel_cls: type,
    m_block_size: int = 128,
    n_block_size: int = 128,
    k_block_size: int = 32,
    num_stages: int = 3,
    atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
    raster_factor: int = 1,
) -> GatedSigmoidKernelConfig:
    """Wrap a gated sigmoid kernel class + tile params into a config entry."""
    atom_layout_mnk = tuple(atom_layout_mnk)

    def factory(ct_dtype: type, has_bias: bool):
        return kernel_cls(
            ab_dtype=ct_dtype,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            num_stages=num_stages,
            atom_layout_mnk=atom_layout_mnk,
            raster_factor=raster_factor,
            has_bias=has_bias,
        )

    def can_impl(ct_dtype: type) -> bool:
        return kernel_cls.can_implement(ct_dtype)

    return GatedSigmoidKernelConfig(
        kernel_factory=factory,
        can_implement=can_impl,
        tile_params={
            "m_block_size": m_block_size,
            "n_block_size": n_block_size,
            "k_block_size": k_block_size,
            "num_stages": num_stages,
            "atom_layout_mnk": list(atom_layout_mnk),
            "raster_factor": raster_factor,
        },
    )


def _make_sm80_config(**tile_params) -> GatedSigmoidKernelConfig:
    """Build a config entry using the default SM80 kernel class."""
    return _build_lazy_kernel_config(_DEFAULT_KERNEL_ABI, **tile_params)


def _build_lazy_kernel_config(kernel_abi: str, **tile_params) -> GatedSigmoidKernelConfig:
    """Build a config whose kernel class resolves only when executed."""

    def kernel_cls() -> type:
        source = load_source_module(__package__)
        return resolve_implementation(source.source_implementation(kernel_abi))

    config = _build_kernel_config(object, **tile_params)
    return GatedSigmoidKernelConfig(
        kernel_factory=lambda ct_dtype, has_bias: _build_kernel_config(kernel_cls(), **tile_params).kernel_factory(
            ct_dtype, has_bias
        ),
        can_implement=lambda ct_dtype: kernel_cls().can_implement(ct_dtype),
        tile_params=config.tile_params,
    )


def _classify_m_range(M: int) -> str:
    if M <= M_SHORT_THRESHOLD:
        return "short"
    elif M <= M_MEDIUM_THRESHOLD:
        return "medium"
    return "long"


def _params_from_json(raw: dict) -> dict:
    """Normalize JSON tile params for :func:`_build_kernel_config`."""
    params = dict(raw)
    layout = params.get("atom_layout_mnk")
    if layout is not None:
        params["atom_layout_mnk"] = tuple(layout)
    return params


_DEFAULT_KERNEL_ABI = "sm80"

# Heuristic tiles must also appear in the builder's shipped tile set.
_SMALL_N_THRESHOLD = 128
_AMPERE_TILES = {
    "small_n": (64, 64, 32, 3),
    "short": (64, 64, 32, 3),
    "medium": (64, 128, 32, 3),
    "long": (128, 64, 32, 3),
}
_HEURISTIC_TILES: dict[int, dict[str, tuple[int, int, int, int]]] = {
    80: _AMPERE_TILES,
    86: _AMPERE_TILES,
    89: _AMPERE_TILES,
    # SM90 autotuned with bias; no-bias is within 1%.
    90: {
        "small_n": (64, 64, 32, 3),
        "short": (64, 64, 32, 3),
        "medium": (64, 64, 32, 3),
        "long": (64, 128, 64, 2),
    },
}


def _heuristic_tile_params(sm_version: int, N: int, m_range: str) -> dict[str, Any]:
    """Return the fallback tile for an unregistered ``(K, N)`` pair."""
    tiles = _HEURISTIC_TILES.get(sm_version)
    if tiles is None:
        raise ValueError(f"No gated-sigmoid fallback tiles registered for SM{sm_version}.")
    bucket = "small_n" if N <= _SMALL_N_THRESHOLD else m_range
    m_block, n_block, k_block, num_stages = tiles[bucket]
    return {
        "m_block_size": m_block,
        "n_block_size": n_block,
        "k_block_size": k_block,
        "num_stages": num_stages,
        "atom_layout_mnk": (2, 2, 1),
        "raster_factor": 1,
    }


def get_m_bucket(M: int) -> int:
    """Return the tuned M-bucket left edge covering ``M``."""
    return _M_BUCKET_KEY[_classify_m_range(M)]


def get_tile_params(sm_version: int, K: int, N: int, M: int) -> dict[str, Any]:
    """Return tuned or fallback tile parameters."""
    m_range = _classify_m_range(M)
    bundle = load_kernel_configs(_GS_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    m_key = str(_M_BUCKET_KEY[m_range])
    if bundle is not None and m_key in bundle.configs:
        return _params_from_json(bundle.configs[m_key])
    return _heuristic_tile_params(sm_version, N, m_range)


def get_kernel_abi(sm_version: int, K: int, N: int) -> str:
    """Return the tuned source generation for one ``(sm, K, N)``."""
    bundle = load_kernel_configs(_GS_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    if bundle is None:
        return _DEFAULT_KERNEL_ABI
    return bundle.kernel_abi


def get_kernel_config(sm_version: int, K: int, N: int, M: int) -> GatedSigmoidKernelConfig:
    """Resolve the source-kernel config for one runtime key."""
    return _build_lazy_kernel_config(
        get_kernel_abi(sm_version, K, N),
        **get_tile_params(sm_version, K, N, M),
    )
