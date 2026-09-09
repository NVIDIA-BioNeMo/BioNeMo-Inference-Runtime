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
"""Load CuTe kernel tuning configs from JSON, one file per (problem shape, SM).

Named ``K{K}_N{N}_sm{sm}.json`` or ``D{D}_sm{sm}.json``::

    {"kernel_abi": "sm80", "kernel_variant": "full_k", "configs": {...}}

``kernel_abi`` is the ABI family, ``sm{version}``; ``_SUPPORTED_KERNEL_ABIS`` is
the set that ships. ``kernel_variant`` optionally picks between kernels sharing
one ABI. Each op package maps the pair to an implementation and parses its own
``configs`` keys. A legacy ``implementation`` key is accepted only as an
alternate spelling of the ABI; the path itself selects nothing.

Set ``BIOIR_TUNED_CONFIG_FOLDER`` to override the search path.
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
_SUPPORTED_KERNEL_ABIS = frozenset({"sm80", "sm90"})


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
        implementation: Dotted kernel class named by the config, or ``None``.
        kernel_abi: Kernel ABI family used for implementation selection.
        kernel_variant: Op-specific choice within ``kernel_abi``, or ``None``.
        configs: Raw config map; keys and structure are op-specific.
        source_path: Filesystem path the bundle was loaded from.
    """

    implementation: str | None
    kernel_abi: str
    kernel_variant: str | None
    configs: dict[str, Any]
    source_path: str


def _implementation_kernel_abi(implementation: str) -> str:
    """Derive the kernel ABI family from an implementation path."""
    return "sm90" if ".sm90_" in implementation else "sm80"


@functools.cache
def _load_kernel_configs_cached(
    user_dir: str | None,
    configs_dir: str,
    file_name: str,
) -> KernelConfigBundle | None:
    """Load a kernel config bundle; ``user_dir`` is part of the cache key."""
    candidates: list[str] = []
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
        kernel_abi = raw.get("kernel_abi")
        if kernel_abi is None:
            if implementation is None:
                raise ValueError(f"Invalid kernel config {path}: missing implementation and kernel_abi")
            kernel_abi = _implementation_kernel_abi(implementation)
        # Checked before the membership test: a JSON list or object is unhashable,
        # so `in` against the frozenset raises TypeError instead of this ValueError.
        if not isinstance(kernel_abi, str) or not kernel_abi:
            raise ValueError(f"Invalid kernel_abi in {path}: expected a non-empty string, got {kernel_abi!r}")
        if kernel_abi not in _SUPPORTED_KERNEL_ABIS:
            raise ValueError(
                f"Invalid kernel_abi in {path}: expected one of {sorted(_SUPPORTED_KERNEL_ABIS)}, got {kernel_abi!r}"
            )
        if implementation is not None and kernel_abi != _implementation_kernel_abi(implementation):
            raise ValueError(f"kernel_abi {kernel_abi!r} disagrees with implementation in {path}")
        kernel_variant = raw.get("kernel_variant")
        if kernel_variant is not None and (not isinstance(kernel_variant, str) or not kernel_variant):
            raise ValueError(f"Invalid kernel_variant in {path}: expected a non-empty string, got {kernel_variant!r}")
        logger.info_once(
            "Using kernel configuration from %s",
            path,
            key=("kernel_config", path),
        )
        return KernelConfigBundle(
            implementation=implementation,
            kernel_abi=kernel_abi,
            kernel_variant=kernel_variant,
            configs=dict(raw["configs"]),
            source_path=path,
        )

    return None


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
    return _load_kernel_configs_cached(os.environ.get(_TUNED_CONFIG_ENV), configs_dir, file_name)


load_kernel_configs.cache_clear = _load_kernel_configs_cached.cache_clear


@functools.cache
def resolve_implementation(dotted_path: str) -> type:
    """Import a class from its dotted path (cached)."""
    module_path, _, name = dotted_path.rpartition(".")
    if not module_path or not name:
        raise ValueError(f"Invalid implementation path {dotted_path!r}; expected 'pkg.module.ClassName'.")
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
