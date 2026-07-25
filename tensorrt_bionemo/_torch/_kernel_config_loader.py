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
"""Load CuTe kernel tuning configs from JSON files (vLLM fused_moe pattern).

One JSON file per (problem shape, SM) pair, named
``K{K}_N{N}_sm{sm}.json`` for gated sigmoid / dual GEMM and
``D{D}_sm{sm}.json`` for attention. Each file has::

    {
        "implementation": "<dotted.module.ClassName>",
        "configs": {
            "<op-specific lookup key>": { ...op-specific value... },
            ...
        }
    }

``implementation`` is the dotted import path of the CuTe kernel class to
construct. The inner key format under ``configs`` is op-specific: a
stringified M bucket for gated sigmoid and attention; a flat pipe-separated
key for dual GEMM (e.g. ``"S=512|t=0"``) that maps directly to tile params
(no nested table — the dual-GEMM loader filters by ``t=`` and picks the
closest ``S=`` anchor).
The loader treats all keys as raw strings and leaves parsing to the
caller.

Override search path: set ``TRTBNM_TUNED_CONFIG_FOLDER`` to a directory
containing the same filenames (checked before the package defaults).
"""

from __future__ import annotations

import functools
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any

from tensorrt_bionemo.logger import logger

_TUNED_CONFIG_ENV = "TRTBNM_TUNED_CONFIG_FOLDER"


def get_config_file_name(sm: int, **dims: int) -> str:
    """Build a config filename from shape dimensions and SM version.

    Args:
        sm: SM version as ``major * 10 + minor`` (e.g. 80).
        **dims: Named shape keys, e.g. ``K=768, N=768`` or ``D=64``.

    Returns:
        Filename like ``K768_N768_sm80.json`` or ``D64_sm80.json``.
    """
    parts = [f"{k}{v}" for k, v in dims.items()]
    parts.append(f"sm{sm}")
    return "_".join(parts) + ".json"


@dataclass(frozen=True)
class KernelConfigBundle:
    """Parsed JSON contents for one ``(problem, sm)`` config file.

    Attributes:
        implementation: Dotted import path of the kernel class.
        configs: Raw config map; keys and structure are op-specific.
        source_path: Filesystem path the bundle was loaded from.
    """

    implementation: str
    configs: dict[str, Any]
    source_path: str


@functools.lru_cache(maxsize=None)
def load_kernel_configs(
    configs_dir: str,
    file_name: str,
) -> KernelConfigBundle | None:
    """Load a kernel config bundle from JSON, with optional user override.

    Args:
        configs_dir: Package directory containing default JSON files.
        file_name: Result of :func:`get_config_file_name`.

    Returns:
        :class:`KernelConfigBundle` or ``None`` if no file exists.
    """
    candidates: list[str] = []
    user_dir = os.environ.get(_TUNED_CONFIG_ENV)
    if user_dir:
        candidates.append(os.path.join(user_dir, file_name))
    else:    
        candidates.append(os.path.join(configs_dir, file_name))

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        logger.info_once(
            "Using kernel configuration from %s",
            path,
            key=("kernel_config", path),
        )
        return KernelConfigBundle(
            implementation=raw["implementation"],
            configs=dict(raw["configs"]),
            source_path=path,
        )

    return None


@functools.lru_cache(maxsize=None)
def resolve_implementation(dotted_path: str) -> type:
    """Import a class from its dotted path (cached)."""
    module_path, _, name = dotted_path.rpartition(".")
    if not module_path or not name:
        raise ValueError(
            f"Invalid implementation path {dotted_path!r}; expected "
            f"'pkg.module.ClassName'.")
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
