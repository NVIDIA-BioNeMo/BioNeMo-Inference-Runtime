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
"""Shipped transition MLP variants and their tuning.

One ``configs/W{C}_H{H}_sm{sm}.json`` bundle per shape and SM declares the ``kernel_abi`` that runs it
and maps keys to tile parameters, which the ABI's kernel takes as keyword arguments. A key reads
``[S=<bucket>|]act=<activation>|bias=<0|1>|mask=<0|1>[|res=<0|1>]``:

- Without ``res``, an entry serves both residual states; an entry naming the call's state wins.
- With ``S``, an entry is tuned for that pseudo sequence length, and a call takes the nearest
  bucket, ties going to the smaller. Without it, the entry serves every size. The entries a variant
  resolves to are all bucketed or a single size-independent one.

A call's pseudo sequence length is ``round(sqrt(rows))`` over its flattened rows, the measure the
dual GEMM buckets on: ``sqrt(I * J)`` for a pair representation ``[I, J, C]``. The CUBIN builder
enumerates the SM90 bundles through :func:`bundle_variants`, so each image is the tile a call of
its bucket selects; the SM80 kernel runs from source on SM80, SM86 and SM89.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
ACTIVATIONS = ("relu", "silu_gate")
KERNEL_ABIS = {80: "sm80", 86: "sm80", 89: "sm80", 90: "sm90"}
# The bucket of a size-independent entry.
ANY_SIZE = 0
_FILE_RE = re.compile(r"^W(?P<width>\d+)_H(?P<hidden>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(
    r"^(?:S=(?P<bucket>[1-9]\d*)\|)?act=(?P<activation>relu|silu_gate)\|bias=(?P<bias>[01])\|mask=(?P<mask>[01])"
    r"(?:\|res=(?P<residual>[01]))?$"
)


class TransitionMlpVariant(NamedTuple):
    """Every axis that changes the kernel's machine code or argument list."""

    activation: str
    has_bias: bool
    has_mask: bool
    has_residual: bool
    width: int
    hidden: int


class ConfigKey(NamedTuple):
    """One parsed ``configs`` key; a ``None`` axis applies to every value."""

    bucket: int | None
    activation: str
    has_bias: bool
    has_mask: bool
    has_residual: bool | None


def variant_key(variant: TransitionMlpVariant) -> str:
    """The size-independent ``configs`` key that serves both of a variant's residual states."""
    return config_key(variant.activation, variant.has_bias, variant.has_mask)


def config_key(
    activation: str,
    has_bias: bool,
    has_mask: bool,
    *,
    bucket: int | None = None,
    has_residual: bool | None = None,
) -> str:
    """Format a ``configs`` key; the inverse of :func:`parse_config_key`."""
    key = f"act={activation}|bias={int(has_bias)}|mask={int(has_mask)}"
    if bucket is not None:
        key = f"S={bucket}|{key}"
    if has_residual is not None:
        key = f"{key}|res={int(has_residual)}"
    return key


def parse_config_key(key: str) -> ConfigKey:
    """Parse one ``configs`` key."""
    match = _KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(
            f"invalid transition MLP config key {key!r}; expected [S=<bucket>|]act=...|bias=0|1|mask=0|1[|res=0|1]"
        )
    bucket, residual = match["bucket"], match["residual"]
    return ConfigKey(
        None if bucket is None else int(bucket),
        match["activation"],
        match["bias"] == "1",
        match["mask"] == "1",
        None if residual is None else residual == "1",
    )


def pseudo_seqlen(rows: int) -> int:
    """The bucket coordinate of a call over ``rows`` flattened rows: ``sqrt(I * J)`` for pair features."""
    return round(math.sqrt(max(rows, 1)))


def _variant_entries(configs: dict[str, Any], variant: TransitionMlpVariant) -> dict[int, tuple[str, dict[str, Any]]]:
    """``bucket -> (key, tile)`` for ``variant``, empty when the bundle doesn't declare it."""
    exact: dict[int, tuple[str, dict[str, Any]]] = {}
    shared: dict[int, tuple[str, dict[str, Any]]] = {}
    for key, tile in configs.items():
        parsed = parse_config_key(key)
        if (parsed.activation, parsed.has_bias, parsed.has_mask) != variant[:3]:
            continue
        if parsed.has_residual is None:
            group = shared
        elif parsed.has_residual == variant.has_residual:
            group = exact
        else:
            continue
        group[ANY_SIZE if parsed.bucket is None else parsed.bucket] = (key, dict(tile))
    entries = exact or shared
    if ANY_SIZE in entries and len(entries) > 1:
        keys = sorted(key for key, _ in entries.values())
        raise ValueError(f"transition MLP {variant} resolves to bucketed and size-independent entries: {keys}")
    return entries


def bundle_variants(
    configs: dict[str, Any], width: int, hidden: int
) -> dict[TransitionMlpVariant, dict[int, tuple[str, dict[str, Any]]]]:
    """Every variant a bundle's ``configs`` declare, each with its ``bucket -> (key, tile)`` entries."""
    variants = set()
    for key in configs:
        parsed = parse_config_key(key)
        for has_residual in (False, True) if parsed.has_residual is None else (parsed.has_residual,):
            variants.add(
                TransitionMlpVariant(parsed.activation, parsed.has_bias, parsed.has_mask, has_residual, width, hidden)
            )
    return {variant: _variant_entries(configs, variant) for variant in variants}


def _bundle_configs(sm: int, file_name: str) -> dict[str, Any] | None:
    """A bundle's configs if it runs ``sm``'s kernel ABI and names no kernel variant, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi or bundle.kernel_variant is not None:
        return None
    return bundle.configs


def _configs(sm: int, variant: TransitionMlpVariant) -> dict[str, Any] | None:
    return _bundle_configs(sm, get_config_file_name(sm, W=variant.width, H=variant.hidden))


def buckets(sm: int, variant: TransitionMlpVariant) -> tuple[int, ...]:
    """The buckets ``variant`` ships on ``sm`` in ascending order; ``(ANY_SIZE,)`` for a size-independent entry."""
    configs = _configs(sm, variant)
    return () if configs is None else tuple(sorted(_variant_entries(configs, variant)))


def nearest_bucket(anchors: tuple[int, ...], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(anchors, key=lambda anchor: (abs(anchor - seqlen), anchor))


def get_tile_params(sm: int, variant: TransitionMlpVariant, bucket: int | None = None) -> dict[str, Any] | None:
    """Tile parameters for ``variant`` at ``bucket`` on ``sm``, or ``None`` when that entry does not ship.

    ``bucket=None`` takes the smallest shipped bucket, so it answers whether the variant ships at all.
    """
    configs = _configs(sm, variant)
    if configs is None:
        return None
    entries = _variant_entries(configs, variant)
    if not entries:
        return None
    entry = entries.get(min(entries) if bucket is None else bucket)
    return None if entry is None else dict(entry[1])


def select_config(sm: int, variant: TransitionMlpVariant, seqlen: int) -> tuple[int, dict[str, Any]] | None:
    """The bucket a call of pseudo sequence length ``seqlen`` selects and its tile, or ``None`` if unshipped."""
    anchors = buckets(sm, variant)
    if not anchors:
        return None
    bucket = nearest_bucket(anchors, seqlen)
    return bucket, get_tile_params(sm, variant, bucket)


def shipped_variants(sm: int) -> frozenset[TransitionMlpVariant]:
    """Every variant the packaged configs declare for ``sm``, from the bundles :func:`get_tile_params` reads."""
    if sm not in KERNEL_ABIS:
        return frozenset()
    variants = set()
    for path in sorted(CONFIGS_DIR.glob(f"W*_H*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        configs = _bundle_configs(sm, path.name)
        if configs is None:
            continue
        declared = bundle_variants(configs, int(match["width"]), int(match["hidden"]))
        variants.update(variant for variant, entries in declared.items() if entries)
    return frozenset(variants)
