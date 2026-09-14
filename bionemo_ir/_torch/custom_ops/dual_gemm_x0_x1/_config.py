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

import cutlass.utils as utils

from bionemo_ir._torch.utils.kernel import (
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
    resolve_implementation,
)
from bionemo_ir.logger import logger

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

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


def _build_kernel_config(
    kernel_cls: type,
    tile_params: dict[str, Any],
    has_bias: bool,
    dtype_str: str,
    chosen_key: str,
    bucket: int,
    *,
    sm_version: int,
    kernel_abi: str,
    asymmetric: bool = False,
    K: int | None = None,
    K1: int | None = None,
    N: int | None = None,
) -> DualGemmX0X1KernelConfig:
    """Bind one implementation and tile to runtime dtype, bias flags, and shape.

    ``K``/``K1``/``N`` are the widths the caller will run. The SM90 asymmetric
    kernel bakes them into its tile counts and shared-memory budget, so it has
    to be told; they are passed to the kernel config rather than added to
    ``tile_params`` so the tuning JSON stays the single source of tile choices
    and the CUBIN variant identity, which already carries the shape, does not
    change for tunings that predate this.
    """
    full_params = dict(tile_params)
    full_params["ab_dtype"] = dtype_str
    is_sm90 = kernel_abi == "sm90"
    kernel_params = dict(full_params)
    if N is not None:
        kernel_params["n"] = int(N)
    if K is not None:
        kernel_params["k0"] = int(K)
        kernel_params["k1"] = int(K if K1 is None else K1)
    kernel_config = _kernel_config_dataclass(kernel_cls).from_dict(kernel_params)

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
        if asymmetric:
            vector_elements = 128 // ct_dtype.width
            if K % vector_elements != 0 or N % vector_elements != 0:
                return False
            smem_bytes = kernel_cls.dynamic_smem_bytes(
                ct_dtype,
                tuple(full_params["cta_tiler"]),
                int(full_params["num_stages"]),
            )
            return smem_bytes <= int(utils.get_smem_capacity_in_bytes(f"sm_{sm_version}"))
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


def needs_independent_operand_strides(K0: int, K1: int, N: int) -> bool:
    """Whether ``X1`` and the output need their own extents and row strides.

    The compact operand signature shares one row-stride symbol across ``X0``,
    ``X1`` and ``out``, which constrains both ``K1`` and ``N`` to equal ``K0``
    at launch. Unequal ``K1`` is not the only way to break that, so this keys
    on the output extent too: the 128x256 out-projection has ``K0 == K1`` with
    ``N != K0`` and needs the wide signature just as much.

    This decides the traced signature only, never variant identity. Every shape
    tuned before that out-projection has ``N == K0`` whenever ``K0 == K1``, so
    it agrees with the old ``K1 != K0`` test on all of them and leaves their
    identities, disk cache keys, and shipped CUBINs untouched.
    """
    return K1 != K0 or N != K0


def config_file_name(sm: int, K0: int, K1: int, N: int) -> str:
    """Return the equal-width or explicit asymmetric config filename.

    Equal widths keep the legacy ``K{K}_N{N}`` name so existing tunings stay
    addressable; asymmetric widths use
    ``K0{K0}_K1{K1}_N{N}``.
    """
    if K0 == K1:
        return get_config_file_name(sm, K=K0, N=N)
    return get_config_file_name(sm, K0=K0, K1=K1, N=N)


def load_bundle(sm_version: int, K: int, N: int, K1: int | None = None):
    """Load the tuned bundle for ``(sm_version, K, N)``, falling back to SM80.

    ``K`` is the first inner dimension; ``K1`` defaults to ``K`` for the
    equal-width case.
    """
    K1 = K if K1 is None else K1
    name = config_file_name(sm_version, K, K1, N)
    bundle = load_kernel_configs(_CONFIGS_DIR, name)
    if bundle is not None:
        return bundle
    fallback = _fallback_sm(sm_version)
    logger.warning(
        f"No dual_gemm x0_x1 config for SM{sm_version} K0={K} K1={K1} N={N}; falling back to SM{fallback} tuning."
    )
    bundle = load_kernel_configs(_CONFIGS_DIR, config_file_name(fallback, K, K1, N))
    if bundle is None:
        raise ValueError(
            f"No dual_gemm x0_x1 config file for SM{sm_version} or fallback "
            f"SM{fallback} for K0={K}, K1={K1}, N={N}; expected "
            f"{_CONFIGS_DIR}/{name}"
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


def get_nearest_bucket(sm_version: int, K: int, N: int, S: int, has_bias: bool, K1: int | None = None) -> int:
    """Return the nearest tuned per-sample side-length anchor."""
    bundle = load_bundle(sm_version, K, N, K1)
    bucket, _, _ = _nearest_variant(bundle.configs, S, has_bias)
    return bucket


def kernel_is_sm90(sm_version: int, K: int, N: int, K1: int | None = None) -> bool:
    """Whether this config bundle selects the Hopper call ABI."""
    try:
        bundle = load_bundle(sm_version, K, N, K1)
    except ValueError:
        return False
    return bundle.kernel_abi == "sm90"


def get_kernel_config(
    sm_version: int,
    K: int,
    N: int,
    S: int,
    has_bias: bool,
    dtype_str: str,
    K1: int | None = None,
    K0: int | None = None,
) -> DualGemmX0X1KernelConfig:
    """Resolve the nearest tuned source kernel.

    Args:
        sm_version: Device SM as ``major * 10 + minor``.
        K: First GEMM inner dimension (``K0``).
        N: GEMM output dimension.
        S: Side-length tuning anchor.
        has_bias: Whether both bias operands are present.
        dtype_str: ``"fp16"`` or ``"bf16"``.
        K1: Second GEMM inner dimension; defaults to ``K``.
        K0: Explicit alias for ``K``; when provided, it must match.
    """
    if K0 is not None and K0 != K:
        raise ValueError(f"K={K} and K0={K0} must match")
    bundle = load_bundle(sm_version, K, N, K1)
    bucket, chosen_key, tile_params = _nearest_variant(bundle.configs, S, has_bias)
    implementation = load_source_module(__package__).source_implementation(
        bundle.kernel_abi, tile_params, bundle.kernel_variant
    )
    kernel_cls = resolve_implementation(implementation)
    return _build_kernel_config(
        kernel_cls,
        tile_params,
        has_bias=has_bias,
        dtype_str=dtype_str,
        chosen_key=chosen_key,
        bucket=bucket,
        sm_version=sm_version,
        kernel_abi=bundle.kernel_abi,
        asymmetric=K1 is not None and K1 != K,
        K=K,
        K1=K1,
        N=N,
    )
