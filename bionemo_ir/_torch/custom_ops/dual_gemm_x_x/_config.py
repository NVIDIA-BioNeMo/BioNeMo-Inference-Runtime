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
"""Tuned configuration lookup for the dual-GEMM ``x_x`` kernel."""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo_ir._torch.utils.kernel import (
    KernelConfigBundle,
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
)
from bionemo_ir.logger import logger

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

_TUNED_SMS: tuple[int, ...] = (80, 86, 89, 90, 100, 103)
_FALLBACK_SM = 80
#: SM90/Blackwell kernel that keeps each CTA's weight slice resident and multicasts X across a cluster.
RESIDENT_VARIANT = "resident"
_VARIANT_KEY_RE = re.compile(r"^S=(\d+)\|t=(\d+)$")
_DEFAULT_GATES = ("sigmoid",)
_SUPPORTED_GATES = frozenset({"sigmoid", "silu"})


@dataclass(frozen=True)
class DualGemmXxKernelConfig:
    """A source kernel factory and its selected tile parameters."""

    kernel_factory: Callable[[], Any]
    tile_params: dict[str, Any]


@dataclass(frozen=True)
class _DualGemmXxConfigSelection:
    """Source-independent result of one tuning lookup."""

    kernel_abi: str
    kernel_variant: str | None
    chosen_key: str
    bucket: int
    tile_params: dict[str, Any]


def _fallback_sm(sm_version: int) -> int:
    """Return the tuned SM used when ``sm_version`` has no direct config."""
    if sm_version in _TUNED_SMS:
        return sm_version
    return _FALLBACK_SM


def _variant_key(S: int, transpose_out: bool) -> str:
    """Compose the JSON key for one sequence anchor and output layout."""
    return f"S={int(S)}|t={int(transpose_out)}"


def _nearest_variant(
    configs: dict[str, Any],
    S: int,
    transpose_out: bool,
) -> tuple[str, dict[str, Any]]:
    """Select the nearest matching ``S`` anchor, preferring the lower tie."""
    t_flag = int(transpose_out)
    candidates: list[tuple[int, str]] = []
    for key in configs:
        match = _VARIANT_KEY_RE.fullmatch(key)
        if match is None or int(match.group(2)) != t_flag:
            continue
        candidates.append((int(match.group(1)), key))
    if not candidates:
        raise ValueError(f"No dual_gemm x_x variant for t={t_flag}; available: {sorted(configs)}")

    _, best_key = min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))
    return best_key, dict(configs[best_key])


def _optional_config_bundle(sm_version: int, K: int, N: int) -> KernelConfigBundle | None:
    """Load the direct or fallback bundle without requiring one to exist."""
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    if bundle is None:
        bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(_fallback_sm(sm_version), K=K, N=N))
    return bundle


@functools.cache
def _declared_gates(source_path: str) -> tuple[str, ...]:
    """Return the machine-code epilogues declared by one shape bundle."""
    with open(source_path) as config_file:
        raw = json.load(config_file)
    gates = raw.get("gates", list(_DEFAULT_GATES))
    if not isinstance(gates, list) or not gates:
        raise ValueError(f"Invalid dual_gemm x_x gates in {source_path}: expected a non-empty list")
    if len(set(gates)) != len(gates) or any(gate not in _SUPPORTED_GATES for gate in gates):
        raise ValueError(
            f"Invalid dual_gemm x_x gates in {source_path}: expected unique values from {sorted(_SUPPORTED_GATES)}"
        )
    return tuple(gates)


def _has_direct_config_for_gate(sm_version: int, K: int, N: int, gate: str) -> bool:
    """Whether this exact target ships a tuning bundle for ``gate``.

    Source lookup can fall back from an unknown target to SM80 tuning, but the
    CUBIN corpus cannot: each target has its own SASS image. Backend dispatch
    therefore requires a direct file so source and source-free builds select
    the same implementation.
    """
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    return bundle is not None and gate in _declared_gates(bundle.source_path)


def _load_config_bundle(sm_version: int, K: int, N: int) -> KernelConfigBundle:
    """Load the direct config, preserving the existing SM80 fallback."""
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    if bundle is not None:
        return bundle

    fallback = _fallback_sm(sm_version)
    logger.warning(f"No dual_gemm x_x config for SM{sm_version} K={K} N={N}; falling back to SM{fallback} tuning.")
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(fallback, K=K, N=N))
    if bundle is None:
        expected = get_config_file_name(sm_version, K=K, N=N)
        raise ValueError(
            f"No dual_gemm x_x config file for SM{sm_version} or fallback SM{fallback} "
            f"for K={K}, N={N}; expected {_CONFIGS_DIR}/{expected}"
        )
    return bundle


def _get_config_selection(
    sm_version: int,
    K: int,
    N: int,
    S: int,
    transpose_out: bool,
) -> _DualGemmXxConfigSelection:
    """Return implementation metadata and the nearest tuned tile.

    ``kernel_variant`` is normally the bundle-wide default, but one bucket's
    tile_params may name its own ``"kernel_variant"`` to select a different
    kernel class for just that anchor (e.g. the resident-weight kernel wins
    at larger N but regresses at the smallest anchor, where the bundle
    default stays in force). Popped from ``tile_params`` so it never leaks
    into a ``KernelConfig.from_dict`` call as an unrecognized field.
    """
    bundle = _load_config_bundle(sm_version, K, N)
    chosen_key, tile_params = _nearest_variant(bundle.configs, S, transpose_out)
    match = _VARIANT_KEY_RE.fullmatch(chosen_key)
    if match is None:
        raise ValueError(f"Invalid dual_gemm x_x variant key {chosen_key!r}")
    kernel_variant = tile_params.pop("kernel_variant", bundle.kernel_variant)
    return _DualGemmXxConfigSelection(
        kernel_abi=bundle.kernel_abi,
        kernel_variant=kernel_variant,
        chosen_key=chosen_key,
        bucket=int(match.group(1)),
        tile_params=tile_params,
    )


def _get_bucket_ranges(
    sm_version: int,
    K: int,
    N: int,
    transpose_out: bool,
) -> list[tuple[int, str]] | None:
    """Return sorted anchors for a shape and transpose mode."""
    bundle = _optional_config_bundle(sm_version, K, N)
    if bundle is None:
        return None

    t_flag = int(transpose_out)
    ranges = [
        (int(match.group(1)), key)
        for key in bundle.configs
        if (match := _VARIANT_KEY_RE.fullmatch(key)) is not None and int(match.group(2)) == t_flag
    ]
    if not ranges:
        return None
    ranges.sort(key=lambda candidate: candidate[0])
    return ranges


def _kernel_is_sm90(sm_version: int, K: int, N: int) -> bool:
    """Whether the selected source generation uses the raw SM90 signature."""
    bundle = _optional_config_bundle(sm_version, K, N)
    return bundle is not None and bundle.kernel_abi == "sm90"


def _uses_resident_kernel(sm_version: int, K: int, N: int, S: int, transpose_out: bool) -> bool:
    """Whether this (shape, anchor) selects the resident-weight SM90 kernel, which needs a live mask.

    The Blackwell resident kernel null-checks the mask like every other image, so
    only the SM90 one qualifies. Bucket-aware (not just bundle-wide) because a
    bucket's tile_params may override ``kernel_variant`` for just that anchor --
    goes through the same ``_get_config_selection`` resolution every other caller
    uses, so this can never disagree with which kernel actually gets built.
    """
    bundle = _optional_config_bundle(sm_version, K, N)
    if bundle is None or bundle.kernel_abi != "sm90":
        return False
    selection = _get_config_selection(sm_version, K, N, S, transpose_out)
    return selection.kernel_variant == RESIDENT_VARIANT


def get_nearest_bucket(
    sm_version: int,
    K: int,
    N: int,
    S: int,
    transpose_out: bool,
) -> int:
    """Return the nearest tuned sequence-length bucket."""
    return _get_config_selection(sm_version, K, N, S, transpose_out).bucket


def get_kernel_config(
    sm_version: int,
    K: int,
    N: int,
    S: int,
    transpose_out: bool,
    has_bias: bool,
    dtype_str: str,
    gate: str = "sigmoid",
) -> DualGemmXxKernelConfig:
    """Resolve and build the development-time source kernel configuration.

    Importing the package does not import kernel source implementations. This
    compatibility entry point resolves them only when explicitly called.
    """
    source = load_source_module(__package__)
    selection = _get_config_selection(sm_version, K, N, S, transpose_out)
    return source.build_source_kernel_config(
        selection,
        has_bias=has_bias,
        transpose_out=transpose_out,
        dtype_str=dtype_str,
        gate=gate,
    )
