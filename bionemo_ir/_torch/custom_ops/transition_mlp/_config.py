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
and maps keys to tile parameters, which the ABI's kernel takes as keyword arguments. An entry may
name its own ``kernel_abi``. A key reads ``[S=<bucket>|]act=<activation>|bias=<0|1>|mask=<0|1>[|res=<0|1>]``:

- Without ``res``, an entry serves both residual states; an entry naming the call's state wins.
- With ``S``, an entry is tuned for that pseudo sequence length, and a call takes the nearest
  bucket, ties going to the smaller. Without it, the entry serves every size. The entries a variant
  resolves to are all bucketed or a single size-independent one.

A call's pseudo sequence length is ``round(sqrt(rows))`` over its flattened rows, the measure the
dual GEMM buckets on: ``sqrt(I * J)`` for a pair representation ``[I, J, C]``. The CUBIN builder
enumerates every bundle through :func:`bundle_variants`, so each image is the tile a call of its
bucket selects. SM80, SM86 and SM89 run the SM80 kernel. SM90 runs the SM90 kernel, or the SM80
kernel where a bundle or an entry declares it: the SM90 kernel's CTA multiplies 128 rows by every
weight, so a call of few rows reaches few SMs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import KernelConfigBundle, get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
# W1 row blocks per activation, as the kernels count them: the ReLU input; the SwiGLU value and gate; the
# 3-way SwiGLU value, gate and second value.
ACTIVATIONS = {"relu": 1, "silu_gate": 2, "silu_gate_3way": 3}
# Each SM's own kernel ABI, and every ABI its bundles may declare.
KERNEL_ABIS = {80: "sm80", 86: "sm80", 89: "sm80", 90: "sm90"}
_RUNNABLE_ABIS = {80: {"sm80"}, 86: {"sm80"}, 89: {"sm80"}, 90: {"sm90", "sm80"}}
# The bucket of a size-independent entry.
ANY_SIZE = 0
_FILE_RE = re.compile(r"^W(?P<width>\d+)_H(?P<hidden>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(
    r"^(?:S=(?P<bucket>[1-9]\d*)\|)?act=(?P<activation>relu|silu_gate|silu_gate_3way)\|bias=(?P<bias>[01])"
    r"\|mask=(?P<mask>[01])(?:\|res=(?P<residual>[01]))?$"
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


class TileEntry(NamedTuple):
    """One resolved ``configs`` entry: its key, the kernel ABI that runs it and that kernel's keyword arguments."""

    key: str
    kernel_abi: str
    tile_params: dict[str, Any]


def w1_rows(variant: TransitionMlpVariant) -> int:
    """Rows of the first weight: the hidden width once per W1 block of the variant's activation."""
    return ACTIVATIONS[variant.activation] * variant.hidden


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


def _variant_entries(configs: dict[str, Any], variant: TransitionMlpVariant, kernel_abi: str) -> dict[int, TileEntry]:
    """``bucket -> entry`` for ``variant``, empty when the bundle doesn't declare it.

    ``kernel_abi`` is the bundle's, which runs every entry that names none of its own.
    """
    exact: dict[int, TileEntry] = {}
    shared: dict[int, TileEntry] = {}
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
        tile_params = dict(tile)
        entry_abi = tile_params.pop("kernel_abi", kernel_abi)
        group[ANY_SIZE if parsed.bucket is None else parsed.bucket] = TileEntry(key, entry_abi, tile_params)
    entries = exact or shared
    if ANY_SIZE in entries and len(entries) > 1:
        keys = sorted(entry.key for entry in entries.values())
        raise ValueError(f"transition MLP {variant} resolves to bucketed and size-independent entries: {keys}")
    return entries


def bundle_variants(
    configs: dict[str, Any], width: int, hidden: int, kernel_abi: str
) -> dict[TransitionMlpVariant, dict[int, TileEntry]]:
    """Every variant a bundle's ``configs`` declare, each with its ``bucket -> entry`` map.

    ``kernel_abi`` is the bundle's, which runs every entry that names none of its own.
    """
    variants = set()
    for key in configs:
        parsed = parse_config_key(key)
        for has_residual in (False, True) if parsed.has_residual is None else (parsed.has_residual,):
            variants.add(
                TransitionMlpVariant(parsed.activation, parsed.has_bias, parsed.has_mask, has_residual, width, hidden)
            )
    return {variant: _variant_entries(configs, variant, kernel_abi) for variant in variants}


def _bundle(sm: int, file_name: str) -> KernelConfigBundle | None:
    """A bundle if ``sm`` runs its kernel ABI and it names no kernel variant, else ``None``.

    Raises:
        ValueError: An entry of the bundle names a kernel ABI ``sm`` can't run.
    """
    runnable = _RUNNABLE_ABIS.get(sm)
    if runnable is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_variant is not None or bundle.kernel_abi not in runnable:
        return None
    for key, tile in bundle.configs.items():
        if tile.get("kernel_abi", bundle.kernel_abi) not in runnable:
            raise ValueError(f"{file_name}: {key} names kernel_abi {tile['kernel_abi']!r}, which SM{sm} cannot run")
    return bundle


def _entries(sm: int, variant: TransitionMlpVariant) -> dict[int, TileEntry]:
    """``bucket -> entry`` for ``variant`` on ``sm``, empty when no bundle declares it."""
    bundle = _bundle(sm, get_config_file_name(sm, W=variant.width, H=variant.hidden))
    return {} if bundle is None else _variant_entries(bundle.configs, variant, bundle.kernel_abi)


def _entry(sm: int, variant: TransitionMlpVariant, bucket: int | None) -> TileEntry | None:
    entries = _entries(sm, variant)
    if not entries:
        return None
    return entries.get(min(entries) if bucket is None else bucket)


def kernel_abi(sm: int, variant: TransitionMlpVariant, bucket: int) -> str | None:
    """The kernel ABI that runs ``variant`` at ``bucket`` on ``sm``, or ``None`` when that entry does not ship."""
    entry = _entry(sm, variant, bucket)
    return None if entry is None else entry.kernel_abi


def buckets(sm: int, variant: TransitionMlpVariant) -> tuple[int, ...]:
    """The buckets ``variant`` ships on ``sm`` in ascending order; ``(ANY_SIZE,)`` for a size-independent entry."""
    return tuple(sorted(_entries(sm, variant)))


def nearest_bucket(anchors: tuple[int, ...], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(anchors, key=lambda anchor: (abs(anchor - seqlen), anchor))


def get_tile_params(sm: int, variant: TransitionMlpVariant, bucket: int | None = None) -> dict[str, Any] | None:
    """Tile parameters for ``variant`` at ``bucket`` on ``sm``, or ``None`` when that entry does not ship.

    ``bucket=None`` takes the smallest shipped bucket, so it answers whether the variant ships at all.
    """
    entry = _entry(sm, variant, bucket)
    return None if entry is None else dict(entry.tile_params)


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
        bundle = _bundle(sm, path.name)
        if bundle is None:
            continue
        declared = bundle_variants(bundle.configs, int(match["width"]), int(match["hidden"]), bundle.kernel_abi)
        variants.update(variant for variant, entries in declared.items() if entries)
    return frozenset(variants)
