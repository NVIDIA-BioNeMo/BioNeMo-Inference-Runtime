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
from dataclasses import dataclass, replace
from typing import Any

from bionemo_ir._torch.utils.kernel import (
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
    resolve_implementation,
)

M_TINY_THRESHOLD = 256
M_SHORT_THRESHOLD = 768
M_SHORT_HI_THRESHOLD = 1024
M_MEDIUM_THRESHOLD = 2048

_GS_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# JSON configs are keyed by (K, N, SM) and M-bucket left edge. A bucket the file does not list runs the entry of the
# nearest lower short bucket it does list (``short_hi`` -> ``short`` -> ``tiny``), so a file written before a split
# keeps its meaning.
_M_BUCKET_KEY = {
    "tiny": 0,
    "short": M_TINY_THRESHOLD + 1,
    "short_hi": M_SHORT_THRESHOLD + 1,
    "medium": M_SHORT_HI_THRESHOLD + 1,
    "long": M_MEDIUM_THRESHOLD + 1,
}


@dataclass(frozen=True)
class GatedSigmoidKernelConfig:
    """Resolved source-kernel configuration.

    ``kernel_factory(ct_dtype, has_bias, has_residual=False, has_mask=False)``
    builds the kernel.
    """

    kernel_factory: Callable[..., Any]
    can_implement: Callable[[type], bool]
    tile_params: dict[str, Any]
    kernel_abi: str = "sm80"


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

    def factory(ct_dtype: type, has_bias: bool, has_residual: bool = False, has_mask: bool = False):
        return kernel_cls(
            ab_dtype=ct_dtype,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            num_stages=num_stages,
            atom_layout_mnk=atom_layout_mnk,
            raster_factor=raster_factor,
            has_bias=has_bias,
            has_residual=has_residual,
            has_mask=has_mask,
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


def _build_sm90_kernel_config(
    kernel_cls: type,
    n_block_size: int = 64,
    k_block_size: int = 64,
    num_stages: int = 3,
    unroll: int = 2,
) -> GatedSigmoidKernelConfig:
    """Wrap the SM90 streaming kernel class + tile params into a config entry."""

    def factory(ct_dtype: type, has_bias: bool, has_residual: bool = False, has_mask: bool = False):
        return kernel_cls(
            ab_dtype=ct_dtype,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            num_stages=num_stages,
            unroll=unroll,
            has_bias=has_bias,
            has_residual=has_residual,
            has_mask=has_mask,
        )

    return GatedSigmoidKernelConfig(
        kernel_factory=factory,
        can_implement=lambda ct_dtype: kernel_cls.can_implement(ct_dtype),
        tile_params={
            "n_block_size": n_block_size,
            "k_block_size": k_block_size,
            "num_stages": num_stages,
            "unroll": unroll,
        },
        kernel_abi="sm90",
    )


def _build_sm100_kernel_config(kernel_cls: type, **tile_params) -> GatedSigmoidKernelConfig:
    """Wrap the persistent TMA ``tcgen05`` kernel class. Its CTA is one 128-row tile with a single MMA warp, so the atom
    layout ``(1, 1, 1)`` names the tile in the CUBIN registry and no Ampere or Hopper tile shares that key."""
    tile_params.setdefault("m_block_size", 128)
    tile_params.setdefault("atom_layout_mnk", (1, 1, 1))
    return replace(_build_kernel_config(kernel_cls, **tile_params), kernel_abi="sm100")


_CONFIG_BUILDERS: dict[str, Callable[..., GatedSigmoidKernelConfig]] = {
    "sm80": _build_kernel_config,
    "sm90": _build_sm90_kernel_config,
    "sm100": _build_sm100_kernel_config,
}


def build_kernel_config_for_abi(kernel_abi: str, kernel_cls: type, **tile_params) -> GatedSigmoidKernelConfig:
    """Wrap ``kernel_cls`` with the tile params of its ``kernel_abi``."""
    try:
        builder = _CONFIG_BUILDERS[kernel_abi]
    except KeyError:
        raise ValueError(f"Unsupported gated sigmoid kernel ABI {kernel_abi!r}") from None
    return builder(kernel_cls, **tile_params)


def _make_sm80_config(**tile_params) -> GatedSigmoidKernelConfig:
    """Build a config entry using the default SM80 kernel class."""
    return _build_lazy_kernel_config(_DEFAULT_KERNEL_ABI, **tile_params)


def _build_lazy_kernel_config(kernel_abi: str, **tile_params) -> GatedSigmoidKernelConfig:
    """Build a config whose kernel class resolves only when executed."""

    def kernel_cls() -> type:
        source = load_source_module(__package__)
        return resolve_implementation(source.source_implementation(kernel_abi))

    config = build_kernel_config_for_abi(kernel_abi, object, **tile_params)
    return GatedSigmoidKernelConfig(
        kernel_factory=lambda *args, **kwargs: build_kernel_config_for_abi(
            kernel_abi, kernel_cls(), **tile_params
        ).kernel_factory(*args, **kwargs),
        can_implement=lambda ct_dtype: kernel_cls().can_implement(ct_dtype),
        tile_params=config.tile_params,
        kernel_abi=kernel_abi,
    )


def _classify_m_range(M: int) -> str:
    if M <= M_TINY_THRESHOLD:
        return "tiny"
    if M <= M_SHORT_THRESHOLD:
        return "short"
    if M <= M_SHORT_HI_THRESHOLD:
        return "short_hi"
    if M <= M_MEDIUM_THRESHOLD:
        return "medium"
    return "long"


def kernel_smem_bytes(kernel: Any, ct_dtype: type) -> int:
    """Dynamic shared memory ``kernel`` launches with, in bytes."""
    if hasattr(kernel, "smem_bytes"):
        return int(kernel.smem_bytes())
    if hasattr(kernel, "n_mhaout_stages"):
        return int(
            kernel.dynamic_smem_bytes(
                ct_dtype,
                kernel.bM,
                kernel.bN,
                kernel.bK,
                kernel.num_stages,
                kernel.n_mhaout_stages,
                kernel.has_residual,
            )
        )
    return int(kernel.dynamic_smem_bytes(ct_dtype, kernel.bN, kernel.bK, kernel.num_stages))


def _params_from_json(raw: dict) -> dict:
    """Normalize JSON tile params for :func:`_build_kernel_config`.

    A bucket may name its own ``kernel_abi``; :func:`get_kernel_abi` reads it, so it is not a tile parameter.
    """
    params = dict(raw)
    params.pop("kernel_abi", None)
    layout = params.get("atom_layout_mnk")
    if layout is not None:
        params["atom_layout_mnk"] = tuple(layout)
    return params


# The source generation of the heuristic tiles and of any bucket a file does not list: the warp-level ``mma.sync``
# kernel, which the Blackwell targets also run.
_DEFAULT_KERNEL_ABI = "sm80"
# The ``sm100`` kernel is a ``tcgen05`` kernel and needs tensor memory, which SM120 and SM121 lack.
_TCGEN05_SMS = (100, 103)


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
    # SM100: the most frequent 2x2x1 winner per bucket of its own B200 sweep; ties go to the smaller tile area.
    100: {
        "small_n": (32, 64, 32, 4),
        "short": (32, 32, 32, 4),
        "medium": (64, 64, 64, 3),
        "long": (64, 64, 64, 3),
    },
    # SM103: the same rule applied to its own B300 sweep.
    103: {
        "small_n": (32, 64, 32, 4),
        "short": (32, 32, 64, 3),
        "medium": (64, 64, 64, 3),
        "long": (64, 64, 64, 3),
    },
    # SM120: the same rule applied to its own RTX PRO 6000 Blackwell sweep. Its 99 KB of shared memory rules out
    # the 128x128 tiles once a residual is staged.
    120: {
        "small_n": (32, 32, 32, 3),
        "short": (32, 64, 32, 4),
        "medium": (64, 64, 32, 4),
        "long": (64, 64, 64, 3),
    },
    # SM121: the same rule applied to its own GB10 sweep. Ties between equally frequent winners go to the
    # smaller tile area, then the squarer tile.
    121: {
        "small_n": (32, 32, 32, 3),
        "short": (32, 32, 32, 3),
        "medium": (64, 64, 32, 4),
        "long": (64, 64, 32, 4),
    },
}


def _heuristic_tile_params(sm_version: int, N: int, m_range: str) -> dict[str, Any]:
    """Return the fallback tile for an unregistered ``(K, N)`` pair."""
    tiles = _HEURISTIC_TILES.get(sm_version)
    if tiles is None:
        raise ValueError(f"No gated-sigmoid fallback tiles registered for SM{sm_version}.")
    bucket = "small_n" if N <= _SMALL_N_THRESHOLD else ("short" if m_range in ("tiny", "short_hi") else m_range)
    m_block, n_block, k_block, num_stages = tiles[bucket]
    return {
        "m_block_size": m_block,
        "n_block_size": n_block,
        "k_block_size": k_block,
        "num_stages": num_stages,
        "atom_layout_mnk": (2, 2, 1),
        "raster_factor": 1,
    }


_SHORT_INHERITANCE = {"short": ("short", "tiny"), "short_hi": ("short_hi", "short", "tiny")}


def _bucket_entry(configs: dict, m_range: str) -> tuple[int, dict] | None:
    """Return ``(left edge, entry)`` of the bucket a file lists for ``m_range``, or ``None``.

    ``short_hi`` inherits ``short`` and ``short`` inherits ``tiny`` when the file lists no key of its own.
    """
    for name in _SHORT_INHERITANCE.get(m_range, (m_range,)):
        edge = _M_BUCKET_KEY[name]
        entry = configs.get(str(edge))
        if entry is not None:
            return edge, entry
    return None


@dataclass(frozen=True)
class ResolvedTile:
    """What one ``(sm, K, N, M)`` runs: the source generation, its tile and the M-bucket key its CUBINs carry."""

    kernel_abi: str
    tile_params: dict[str, Any]
    m_bucket: int


def resolve(sm_version: int, K: int, N: int, M: int) -> ResolvedTile:
    """Resolve one runtime key from its tuned file with a single lookup, or from the heuristic tiles.

    A bucket entry may name its own ``kernel_abi``; the file's ``kernel_abi`` runs every bucket that names none. A
    bucket the file does not list uses the heuristic tile, which belongs to :data:`_DEFAULT_KERNEL_ABI`.
    """
    m_range = _classify_m_range(M)
    bundle = load_kernel_configs(_GS_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    found = None if bundle is None else _bucket_entry(bundle.configs, m_range)
    if found is None:
        return ResolvedTile(_DEFAULT_KERNEL_ABI, _heuristic_tile_params(sm_version, N, m_range), _M_BUCKET_KEY[m_range])
    edge, entry = found
    abi = entry.get("kernel_abi", bundle.kernel_abi)
    if abi not in _CONFIG_BUILDERS:
        raise ValueError(f"Unsupported gated sigmoid kernel ABI {abi!r} in K{K}_N{N}")
    # The TMA source zero-fills a partial last K block, so any K works; only the SM's tensor memory matters.
    if abi == "sm100" and sm_version not in _TCGEN05_SMS:
        raise ValueError(f"K{K}_N{N}: the sm100 gated-sigmoid kernel runs on SM{_TCGEN05_SMS}, not SM{sm_version}")
    return ResolvedTile(abi, _params_from_json(entry), edge)


def get_m_bucket(M: int, sm_version: int | None = None, K: int | None = None, N: int | None = None) -> int:
    """Return the M-bucket left edge that holds the config for ``M``.

    With ``sm_version``, ``K`` and ``N`` the edge is the one the tuned file lists, which is the key its CUBIN entries
    carry; without them it is the edge of the range of ``M``.
    """
    if sm_version is None:
        return _M_BUCKET_KEY[_classify_m_range(M)]
    return resolve(sm_version, K, N, M).m_bucket


def get_tile_params(sm_version: int, K: int, N: int, M: int) -> dict[str, Any]:
    """Return tuned or fallback tile parameters."""
    return resolve(sm_version, K, N, M).tile_params


def get_kernel_abi(sm_version: int, K: int, N: int, M: int | None = None) -> str:
    """Return the tuned source generation for one ``(sm, K, N)``, and for the M bucket of ``M`` when given.

    Without ``M`` the answer is the file's ``kernel_abi``.
    """
    if M is not None:
        return resolve(sm_version, K, N, M).kernel_abi
    bundle = load_kernel_configs(_GS_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    return _DEFAULT_KERNEL_ABI if bundle is None else bundle.kernel_abi


def kernel_config(resolved: ResolvedTile) -> GatedSigmoidKernelConfig:
    """The source-kernel config of one resolved runtime key."""
    return _build_lazy_kernel_config(resolved.kernel_abi, **resolved.tile_params)


def get_kernel_config(sm_version: int, K: int, N: int, M: int) -> GatedSigmoidKernelConfig:
    """Resolve the source-kernel config for one runtime key."""
    return kernel_config(resolve(sm_version, K, N, M))
