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
"""M-bucket schedules and shipped variants for fused AdaLN."""

from __future__ import annotations

import os

import cutlass
import torch

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# Architectures with shipped payloads.
SUPPORTED_SMS = (80, 86, 89, 90, 100, 103)

# N is compile-time and must have a shipped payload.
SHIPPED_N = (128, 256, 384, 512, 768, 1024)

_ADALN_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# JSON keys are inclusive M-bucket left edges; the shipping table is closed.
_BUCKET_BIG_M: dict = {"tpr_override": 128, "num_threads_override": 128}


def _unshipped_n_schedule(sm_version: int, N: int) -> list[tuple[int | None, dict]]:
    """Return the development-only schedule for an unshipped N."""
    if sm_version < 90:
        return [(None, {})]
    if N < 1024:
        return [(256, {"num_threads_override": 256}), (None, {})]
    if N <= 1024:
        return [(2048, {}), (None, _BUCKET_BIG_M)]
    if N < 6144:
        return [(None, _BUCKET_BIG_M)]
    return [(None, {})]


def resolve_geometry(N: int, cfg: dict) -> tuple[int, int]:
    """Resolve ``(threads_per_row, num_threads)``."""
    tpr_override = cfg.get("tpr_override")
    if tpr_override is not None:
        threads_per_row = int(tpr_override)
    else:
        threads_per_row = 256
        for limit, threads in ((64, 8), (128, 16), (384, 16), (3072, 32), (6144, 64), (16384, 128)):
            if N <= limit:
                threads_per_row = threads
                break
    num_threads_override = cfg.get("num_threads_override")
    num_threads = int(num_threads_override) if num_threads_override is not None else (128 if N <= 16384 else 256)
    return threads_per_row, num_threads


# Defaults for every machine-code configuration knob.
_KERNEL_CFG_DEFAULTS: dict = {
    "tpr_override": None,
    "num_threads_override": None,
    "cluster_n_override": None,
    "async_s_copy": None,
    "single_pass": True,
    "direct_load": None,
}


def config_identity(cfg: dict) -> tuple[tuple[str, object], ...]:
    """Return the canonical compile-cache identity."""
    keys = set(_KERNEL_CFG_DEFAULTS) | set(cfg)
    return tuple(sorted((k, cfg.get(k, _KERNEL_CFG_DEFAULTS.get(k))) for k in keys))


# These knobs are represented by the payload geometry key.
_GEOMETRY_KNOBS = frozenset({"tpr_override", "num_threads_override"})


def is_cubin_representable(cfg: dict) -> bool:
    """Return whether the CUBIN registry can represent this config."""

    def opaque(c: dict) -> tuple:
        return tuple(kv for kv in config_identity(c) if kv[0] not in _GEOMETRY_KNOBS)

    return opaque(cfg) == opaque({})


def bucket_variants(sm_version: int, N: int) -> list[tuple[int | None, dict, tuple[int, int]]]:
    """Return ``(m_max, config, geometry)`` for each bucket."""
    bundle = load_kernel_configs(_ADALN_CONFIGS_DIR, get_config_file_name(sm_version, N=N))
    if bundle is None:
        if sm_version in SUPPORTED_SMS and N in SHIPPED_N:
            raise RuntimeError(
                f"no AdaLN bucket schedule for SM{sm_version}, N={N}. The table is closed over the "
                f"shipping matrix, so every (SM, N) in SUPPORTED_SMS x SHIPPED_N needs a "
                f"adaln_layernorm_sigmoid/configs/{get_config_file_name(sm_version, N=N)}"
            )
        return [(m_max, cfg, resolve_geometry(N, cfg)) for m_max, cfg in _unshipped_n_schedule(sm_version, N)]
    edges = sorted(int(key) for key in bundle.configs)
    if not edges or edges[0] != 0:
        raise RuntimeError(f"AdaLN bucket schedule for SM{sm_version}, N={N} must start at left edge 0; got {edges}")
    variants = []
    for index, edge in enumerate(edges):
        m_max = edges[index + 1] - 1 if index + 1 < len(edges) else None
        cfg = dict(bundle.configs[str(edge)])
        variants.append((m_max, cfg, resolve_geometry(N, cfg)))
    return variants


def select_bucket(sm_version: int, N: int, M: int) -> tuple[dict, tuple[int, int]]:
    """Return the bucket config and geometry covering ``M``."""
    for m_max, cfg, geometry in bucket_variants(sm_version, N):
        if m_max is None or M <= m_max:
            return cfg, geometry
    raise RuntimeError(f"No bucket matched M={M} (last bucket must be catch-all)")


def cutlass_dtype(t: torch.Tensor) -> type:
    ty = _TORCH_TO_CUTLASS_DTYPE.get(t.dtype)
    if ty is None:
        raise TypeError(f"AdaLN layernorm-sigmoid kernel expects float16, bfloat16 or float32; got {t.dtype}")
    return ty
