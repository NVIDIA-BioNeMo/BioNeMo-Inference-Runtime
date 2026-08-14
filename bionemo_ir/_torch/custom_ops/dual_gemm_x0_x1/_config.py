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
"""Source-independent configuration lookup for dual-GEMM ``x0_x1``."""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bionemo_ir._torch._kernel_config_loader import (
    get_config_file_name,
    load_kernel_configs,
    resolve_implementation,
)
from bionemo_ir.logger import logger

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "dual_gemm_x0_x1")

# Compare implementation paths without importing private kernels.
_SM80_IMPLEMENTATION = "bionemo_ir.dsl_kernels.cute.sm80_dualgemm_x0x1_splitkv1._DualGemmX0X1KernelSplitKv1"
_SM90_IMPLEMENTATION = "bionemo_ir.dsl_kernels.cute.sm90_dual_gemm.DualGemmSm90Pingpong"

# Hopper uses a distinct call ABI and is shared with the x_x variant.
_SM90_KERNEL_NAME = "DualGemmSm90Pingpong"

# N/K/SM are in the filename; keys are ``S=<anchor>[|b=<0|1>]``.
_VARIANT_KEY_RE = re.compile(r"^S=(?P<s>\d+)(?:\|b=(?P<b>[01]))?$")

# SM80/86/89 use Ampere split-K; SM90 uses Hopper ping-pong.
_TUNED_SMS: tuple[int, ...] = (80, 86, 89, 90)
_FALLBACK_SM: int = 80


@dataclass(frozen=True)
class _ParsedKey:
    s: int  # per-sample side-length anchor
    b: int | None  # has_bias filter (None: applies regardless)
    raw: str  # original key string


def _parse_variant_key(key: str) -> _ParsedKey | None:
    match = _VARIANT_KEY_RE.match(key)
    if match is None:
        return None
    b_str = match.group("b")
    return _ParsedKey(s=int(match.group("s")), b=int(b_str) if b_str is not None else None, raw=key)


@dataclass(frozen=True)
class DualGemmX0X1KernelConfig:
    """Resolved implementation, tile, bucket, and resource validator."""

    arch: str
    kernel_factory: Callable[[], Any]
    can_implement: Callable[[type, int, int], bool]
    tile_params: dict[str, Any]
    chosen_key: str
    bucket: int


def _kernel_config_dataclass(kernel_cls: type) -> type:
    """Return the sibling ``KernelConfig`` type for ``kernel_cls``."""
    module = sys.modules.get(kernel_cls.__module__)
    if module is None:
        module = importlib.import_module(kernel_cls.__module__)
    dataclass_type = getattr(module, "KernelConfig", None)
    if dataclass_type is None:
        raise TypeError(
            f"{kernel_cls.__module__} does not define a sibling `KernelConfig` "
            f"dataclass; cannot build {kernel_cls.__name__}."
        )
    return dataclass_type


def implementation_is_sm90(implementation: str) -> bool:
    """Whether a JSON ``implementation`` path names the Hopper kernel."""
    return implementation.rsplit(".", 1)[-1] == _SM90_KERNEL_NAME


def _build_kernel_config(
    kernel_cls: type,
    tile_params: dict[str, Any],
    has_bias: bool,
    dtype_str: str,
    chosen_key: str,
    bucket: int,
    *,
    sm_version: int,
) -> DualGemmX0X1KernelConfig:
    """Bind one implementation and tile to runtime dtype and bias flags."""
    full_params = dict(tile_params)
    full_params["ab_dtype"] = dtype_str
    is_sm90 = kernel_cls.__name__ == _SM90_KERNEL_NAME
    kernel_config = _kernel_config_dataclass(kernel_cls).from_dict(full_params)

    def factory():
        # x0_x1 is always N-major and unmasked.
        if is_sm90:
            return kernel_cls(
                config=kernel_config,
                variant="x0_x1",
                has_bias=has_bias,
                has_mask=False,
                transpose_out=False,
            )
        return kernel_cls(config=kernel_config, has_bias=has_bias, transpose_out=False)

    def can_implement(ct_dtype: type, K: int, N: int) -> bool:
        if is_sm90:
            # Hopper derives its stage count from SM90 shared-memory capacity.
            return True
        return bool(
            kernel_cls.can_implement(
                ct_dtype,
                K,
                N,
                cta_tiler=tuple(full_params["cta_tiler"]),
                num_stages=int(full_params["num_stages"]),
                # Resource limits come from the target SM, not the ABI.
                sm_version=sm_version,
            )
        )

    return DualGemmX0X1KernelConfig(
        arch="sm90" if is_sm90 else "sm80",
        kernel_factory=factory,
        can_implement=can_implement,
        tile_params=full_params,
        chosen_key=chosen_key,
        bucket=bucket,
    )


def _fallback_sm(sm_version: int) -> int:
    """Use an exact tuning when available, otherwise SM80."""
    if sm_version in _TUNED_SMS:
        return sm_version
    return _FALLBACK_SM


def load_bundle(sm_version: int, K: int, N: int):
    """Load the tuned bundle for ``(sm_version, K, N)``, falling back to SM80."""
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(sm_version, K=K, N=N))
    if bundle is not None:
        return bundle
    fallback = _fallback_sm(sm_version)
    logger.warning(f"No dual_gemm x0_x1 config for SM{sm_version} K={K} N={N}; falling back to SM{fallback} tuning.")
    bundle = load_kernel_configs(_CONFIGS_DIR, get_config_file_name(fallback, K=K, N=N))
    if bundle is None:
        raise ValueError(
            f"No dual_gemm x0_x1 config file for SM{sm_version} or fallback "
            f"SM{fallback} for K={K}, N={N}; expected "
            f"{_CONFIGS_DIR}/{get_config_file_name(sm_version, K=K, N=N)}"
        )
    return bundle


def bucket_anchors(configs: dict[str, Any], has_bias: bool) -> list[tuple[int, str]]:
    """Return sorted anchors, preferring bias-specific over universal keys."""
    b_flag = int(bool(has_bias))
    exact: list[tuple[int, str]] = []
    universal: list[tuple[int, str]] = []
    for key in configs:
        parsed = _parse_variant_key(key)
        if parsed is None:
            continue
        if parsed.b == b_flag:
            exact.append((parsed.s, key))
        elif parsed.b is None:
            universal.append((parsed.s, key))
    candidates = exact or universal
    candidates.sort()
    return candidates


def _nearest_variant(configs: dict[str, Any], S: int, has_bias: bool) -> tuple[int, str, dict]:
    """Pick the nearest ``S`` anchor; ties choose the lower one."""
    candidates = bucket_anchors(configs, has_bias)
    if not candidates:
        raise ValueError(
            f"No dual_gemm x0_x1 variants found for has_bias={has_bias}; available keys: {sorted(configs)}"
        )
    anchor, key = min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))
    return anchor, key, dict(configs[key])


def compute_S(M_rows: int) -> int:
    """Map flattened rows ``M=S*S`` to the nearest side length."""
    return int(round(math.sqrt(max(M_rows, 1))))


def get_nearest_bucket(sm_version: int, K: int, N: int, S: int, has_bias: bool) -> int:
    """Return the nearest tuned per-sample side-length anchor."""
    bundle = load_bundle(sm_version, K, N)
    bucket, _, _ = _nearest_variant(bundle.configs, S, has_bias)
    return bucket


def kernel_is_sm90(sm_version: int, K: int, N: int) -> bool:
    """Whether this config bundle selects the Hopper call ABI."""
    try:
        bundle = load_bundle(sm_version, K, N)
    except ValueError:
        return False
    return implementation_is_sm90(bundle.implementation)


def get_kernel_config(
    sm_version: int,
    K: int,
    N: int,
    S: int,
    has_bias: bool,
    dtype_str: str,
) -> DualGemmX0X1KernelConfig:
    """Resolve the nearest tuned source kernel.

    Args:
        sm_version: Device SM as ``major * 10 + minor``.
        K: GEMM inner dimension.
        N: GEMM output dimension.
        S: Side-length tuning anchor.
        has_bias: Whether both bias operands are present.
        dtype_str: ``"fp16"`` or ``"bf16"``.
    """
    bundle = load_bundle(sm_version, K, N)
    bucket, chosen_key, tile_params = _nearest_variant(bundle.configs, S, has_bias)
    kernel_cls = resolve_implementation(bundle.implementation)
    return _build_kernel_config(
        kernel_cls,
        tile_params,
        has_bias=has_bias,
        dtype_str=dtype_str,
        chosen_key=chosen_key,
        bucket=bucket,
        sm_version=sm_version,
    )
