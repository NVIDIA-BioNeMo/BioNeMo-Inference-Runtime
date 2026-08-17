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
``D{D}_sm{sm}.json`` for attention. Private source checkouts use::

    {
        "implementation": "<dotted.module.ClassName>",
        "configs": {
            "<op-specific lookup key>": { ...op-specific value... },
            ...
        }
    }

Public source-free artifacts replace ``implementation`` with a non-sensitive
``kernel_arch`` marker (``"sm80"`` or ``"sm90"``). Source-only entry points
require the private field, while CUBIN selection can still use the public
architecture marker. The loader treats config keys as raw strings and leaves
parsing to the caller. Gated sigmoid uses stringified legacy bucket keys.
Pairwise and triangle attention use ``"S=<anchor>"`` and select the nearest
per-sample side-length anchor. Dual GEMM uses a flat pipe-separated key (for
example ``"S=512|t=0"``), filters its non-nearest axes, and likewise selects
the closest ``S`` anchor.

Override search path: set ``BIOIR_TUNED_CONFIG_FOLDER`` to a directory
containing the same filenames (checked before the package defaults).
"""

from __future__ import annotations

import functools
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any

from bionemo_ir.logger import logger

_TUNED_CONFIG_ENV = "BIOIR_TUNED_CONFIG_FOLDER"
_SUPPORTED_KERNEL_ARCHES = frozenset({"sm80", "sm90"})


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
        implementation: Private dotted kernel class, or ``None`` after strip.
        kernel_arch: Public source-generation marker used for ABI selection.
        configs: Raw config map; keys and structure are op-specific.
        source_path: Filesystem path the bundle was loaded from.
    """

    implementation: str | None
    kernel_arch: str
    configs: dict[str, Any]
    source_path: str


def _implementation_kernel_arch(implementation: str) -> str:
    """Derive the public source generation from a private implementation path."""
    return "sm90" if ".sm90_" in implementation else "sm80"


def require_source_implementation(implementation: str | None, source_path: str) -> str:
    """Return a private implementation path or report a source-free config."""
    if implementation is None:
        raise ImportError(
            f"Kernel source metadata was stripped from {source_path}; use the packaged CUBIN runtime instead."
        )
    return implementation


@functools.cache
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
        implementation = raw.get("implementation")
        if implementation is not None and (not isinstance(implementation, str) or not implementation):
            raise ValueError(f"Invalid implementation in {path}: expected a non-empty string")
        kernel_arch = raw.get("kernel_arch")
        if kernel_arch is None:
            if implementation is None:
                raise ValueError(f"Invalid kernel config {path}: missing implementation and kernel_arch")
            kernel_arch = _implementation_kernel_arch(implementation)
        if kernel_arch not in _SUPPORTED_KERNEL_ARCHES:
            raise ValueError(
                f"Invalid kernel_arch in {path}: expected one of {sorted(_SUPPORTED_KERNEL_ARCHES)}, "
                f"got {kernel_arch!r}"
            )
        if implementation is not None and kernel_arch != _implementation_kernel_arch(implementation):
            raise ValueError(f"kernel_arch {kernel_arch!r} disagrees with implementation in {path}")
        logger.info_once(
            "Using kernel configuration from %s",
            path,
            key=("kernel_config", path),
        )
        return KernelConfigBundle(
            implementation=implementation,
            kernel_arch=kernel_arch,
            configs=dict(raw["configs"]),
            source_path=path,
        )

    return None


@functools.cache
def resolve_implementation(dotted_path: str) -> type:
    """Import a class from its dotted path (cached)."""
    module_path, _, name = dotted_path.rpartition(".")
    if not module_path or not name:
        raise ValueError(f"Invalid implementation path {dotted_path!r}; expected 'pkg.module.ClassName'.")
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
