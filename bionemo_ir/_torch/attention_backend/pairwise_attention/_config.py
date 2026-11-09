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

from bionemo_ir._torch.utils.kernel import (
    CutlassNumeric,
    KernelConfigBundle,
    get_config_file_name,
    load_kernel_configs,
    load_source_module,
    resolve_implementation,
)

_VARIANT_KEY_RE = re.compile(r"^S=(\d+)$")


@dataclass(frozen=True)
class PairwiseAttentionLeftMaskKernelConfig:
    """A resolved source kernel and its architecture-specific validators."""

    arch: str
    kernel_factory: Callable[[int], Any]
    can_implement: Callable[[CutlassNumeric, int], bool]
    # Everything the bound kernel's machine code depends on beyond the variant.
    cache_identity: tuple = ()


_PW_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_config_bundle(sm_version: int, head_dim: int):
    bundle = load_kernel_configs(
        _PW_CONFIGS_DIR,
        get_config_file_name(sm_version, D=head_dim),
    )
    if bundle is None:
        raise ValueError(f"No pairwise-attention config file for SM{sm_version}, head_dim={head_dim}.")
    return bundle


def _source_variants(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The measured source kernels an ``S=<anchor>`` entry names, or ``None``."""
    variants = entry.get("source_variants")
    if variants is not None and not isinstance(variants, dict):
        raise ValueError("source_variants must be an object")
    return variants


def _general_tile(entry: dict[str, Any]) -> dict[str, Any] | None:
    """An entry's general (CUBIN-built) tile without its ``source_variants``.

    ``None`` for an entry that only names source variants. An empty dict is a
    general tile with default parameters.
    """
    if _source_variants(entry) is None:
        return dict(entry)
    if set(entry) == {"source_variants"}:
        return None
    return {key: value for key, value in entry.items() if key != "source_variants"}


# The measured source kernels exist for these targets only.
TUNED_SMS = (100, 103)
TUNED_HEAD_DIMS = (32, 48, 64)
_TUNED_FAMILIES = ("splitn", "native", "tensor_bias")

_ResolvedKey = tuple[str | int | bool, ...]


@dataclass(frozen=True)
class _TunedVariant:
    """One ``source_variants`` entry, validated when its bundle is first read."""

    anchor: int
    key: _ResolvedKey
    min_sequence: int
    max_sequence: int | None
    mult_ranges: tuple[tuple[int, int], ...]

    def serves(self, mult: int, seqlen: int) -> bool:
        if seqlen < self.min_sequence or (self.max_sequence is not None and seqlen > self.max_sequence):
            return False
        return any(start <= mult <= end for start, end in self.mult_ranges)


@dataclass
class _TunedTable:
    bundle: KernelConfigBundle
    variants: tuple[_TunedVariant, ...]
    # Lengths with their own variants; any other length looks for a nearby anchor.
    anchors: frozenset[int]
    # Nearby anchors serve sequences up to the longest one.
    max_anchor: int
    resolutions: dict[tuple[int, int], _ResolvedKey | None]


# The config loader caches each bundle. Keep its table with the bundle object
# so clearing the loader cache or changing override folders also refreshes it.
_TUNED_TABLES: dict[str, _TunedTable] = {}


def _parse_ranges(ranges: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("mult_ranges must be a non-empty list")
    parsed = []
    for bounds in ranges:
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(item) is not int for item in bounds)
            or bounds[0] < 1
            or bounds[0] > bounds[1]
        ):
            raise ValueError("mult_ranges must contain positive inclusive [start, end] pairs")
        parsed.append((bounds[0], bounds[1]))
    return tuple(parsed)


def _parse_tuned_variant(anchor: int, variant: Any) -> _TunedVariant:
    if not isinstance(variant, dict):
        raise ValueError("a source variant must be an object")
    family, params = variant.get("kernel"), variant.get("params")
    if family not in _TUNED_FAMILIES or not isinstance(params, list):
        raise ValueError(f"a source variant needs a kernel in {_TUNED_FAMILIES} and a parameter list")
    key = (family, *params)
    if family == "splitn" and (len(key) < 5 or type(key[3]) is not int or type(key[4]) is not bool):
        raise ValueError("invalid SplitN tile or full-tiles flag")
    # A variant may be faster on only one side of its tuning anchor.
    min_sequence = variant.get("min_sequence", 1)
    max_sequence = variant.get("max_sequence")
    if type(min_sequence) is not int or min_sequence < 1:
        raise ValueError("min_sequence must be a positive integer")
    if "max_sequence" in variant and (type(max_sequence) is not int or max_sequence < min_sequence):
        raise ValueError("max_sequence must be an integer greater than or equal to min_sequence")
    return _TunedVariant(anchor, key, min_sequence, max_sequence, _parse_ranges(variant.get("mult_ranges")))


def _overlap(a: _TunedVariant, b: _TunedVariant) -> bool:
    """Whether one (mult, seqlen) can select both variants of an anchor."""
    a_max = a.max_sequence if a.max_sequence is not None else float("inf")
    b_max = b.max_sequence if b.max_sequence is not None else float("inf")
    if a.min_sequence > b_max or b.min_sequence > a_max:
        return False
    return any(a0 <= b1 and b0 <= a1 for a0, a1 in a.mult_ranges for b0, b1 in b.mult_ranges)


def _tuned_table(sm_version: int, head_dim: int) -> _TunedTable:
    """Validate a bundle's ``source_variants`` as a whole, so a bad table fails on first use, not on one shape."""
    bundle = _load_config_bundle(sm_version, head_dim)
    table = _TUNED_TABLES.get(bundle.source_path)
    if table is not None and table.bundle is bundle:
        return table
    variants: list[_TunedVariant] = []
    try:
        for anchor_key, entry in bundle.configs.items():
            source_variants = _source_variants(entry)
            if source_variants is None:
                continue
            anchor_match = _VARIANT_KEY_RE.fullmatch(anchor_key)
            if anchor_match is None:
                raise ValueError(f"a sequence-range anchor must be named S=<integer>; got {anchor_key!r}")
            anchor = int(anchor_match.group(1))
            parsed = [_parse_tuned_variant(anchor, variant) for variant in source_variants.values()]
            for index, variant in enumerate(parsed):
                if any(_overlap(variant, other) for other in parsed[:index]):
                    raise ValueError(f"overlapping mult ranges at S={anchor}")
            variants.extend(parsed)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid SM{sm_version} source config in {bundle.source_path}: {error}") from error
    anchors = frozenset(variant.anchor for variant in variants)
    table = _TunedTable(bundle, tuple(variants), anchors, max(anchors, default=0), {})
    _TUNED_TABLES[bundle.source_path] = table
    return table


def tuned_key(mult: int, head_dim: int, seqlen: int, sm_version: int = 100) -> _ResolvedKey | None:
    """Select the measured SM100 or SM103 source kernel for ``mult`` rows of length ``seqlen``.

    An anchor's ``source_variants`` name a kernel per range of multiplicities.
    Returns ``None`` where no measured kernel applies, so the general kernel runs.
    """
    if mult <= 0 or seqlen <= 0 or head_dim not in TUNED_HEAD_DIMS or sm_version not in TUNED_SMS:
        return None
    table = _tuned_table(sm_version, head_dim)
    lookup = (mult, seqlen)
    if lookup in table.resolutions:
        return table.resolutions[lookup]

    # An exact anchor answers alone; otherwise the nearest compatible anchor
    # does, up to the longest tuned sequence.
    exact = seqlen in table.anchors
    selected = best_rank = None
    if exact or seqlen <= table.max_anchor:
        for variant in table.variants:
            if (exact and variant.anchor != seqlen) or not variant.serves(mult, seqlen):
                continue
            key = variant.key
            # A nearby SplitN tile must still cover the sequence.
            if not exact and key[0] == "splitn" and seqlen > key[3]:
                continue
            # Choose the closest compatible tile; ties use the lower anchor.
            # Validation rules out two variants of one anchor serving the same call.
            rank = (abs(variant.anchor - seqlen), variant.anchor)
            if best_rank is None or rank < best_rank:
                # Only the anchor's own length is known to fill every SplitN tile.
                selected = (*key[:4], False, *key[5:]) if key[0] == "splitn" and not exact else key
                best_rank = rank
    if len(table.resolutions) >= 4096:
        table.resolutions.clear()
    table.resolutions[lookup] = selected
    return selected


def _nearest_variant(configs: dict[str, Any], S: int) -> tuple[int, dict[str, Any]]:
    """Return the general tile whose ``S=<anchor>`` key is nearest to ``S``."""
    if S < 0:
        raise ValueError(f"Pairwise-attention S must be non-negative; got {S}")

    candidates: list[tuple[int, dict[str, Any]]] = []
    for key, entry in configs.items():
        match = _VARIANT_KEY_RE.fullmatch(key)
        if match is not None and (tile := _general_tile(entry)) is not None:
            candidates.append((int(match.group(1)), tile))
    if not candidates:
        raise ValueError(f"No pairwise-attention S anchors are registered; available keys: {sorted(configs)}")

    return min(candidates, key=lambda candidate: (abs(candidate[0] - S), candidate[0]))


def get_nearest_bucket(sm_version: int, head_dim: int, S: int) -> int:
    """Return the nearest tuned per-side sequence-length anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    bucket, _ = _nearest_variant(bundle.configs, S)
    return bucket


def _anchor_kernel_abi(bundle: KernelConfigBundle, tile_params: dict[str, Any]) -> str:
    """Pop an anchor's own ``kernel_abi``, which overrides the bundle's for that anchor."""
    kernel_abi = tile_params.pop("kernel_abi", bundle.kernel_abi)
    if kernel_abi not in ("sm80", "sm90", "sm100"):
        raise ValueError(f"{bundle.source_path}: unsupported pairwise-attention kernel_abi {kernel_abi!r}")
    return kernel_abi


def get_bucket_kernel_abi(sm_version: int, head_dim: int, bucket: int) -> str:
    """Return the kernel ABI that serves the tuned ``S=<bucket>`` anchor.

    An anchor may name its own ``kernel_abi``: on SM100 and SM103 short
    sequences run the Ampere kernel, whose smaller tiles pad less than the
    Blackwell kernel's 256-row work items.
    """
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, bucket)
    return _anchor_kernel_abi(bundle, tile_params)


def get_kernel_config(
    sm_version: int,
    head_dim: int,
    S: int,
) -> PairwiseAttentionLeftMaskKernelConfig:
    """Resolve the source kernel at the nearest tuned ``S`` anchor."""
    bundle = _load_config_bundle(sm_version, head_dim)
    _, tile_params = _nearest_variant(bundle.configs, S)
    kernel_abi = _anchor_kernel_abi(bundle, tile_params)
    source = load_source_module(__package__)
    implementation = source.source_implementation(kernel_abi, None, bundle.kernel_variant)
    kernel_cls = resolve_implementation(implementation)
    if kernel_abi == "sm100":
        unknown = set(tile_params) - {"tail_split", "dynamic_scheduler"}
        if unknown:
            raise ValueError(f"Unknown SM100 pairwise-attention parameters {sorted(unknown)} in {bundle.source_path}")
        return source._build_sm100_config(kernel_cls, **tile_params)
    if "mma_tiler_mn" in tile_params:
        return source._build_sm90_config(kernel_cls, **tile_params)
    return source._build_sm80_config(kernel_cls, sm_version=sm_version, **tile_params)
