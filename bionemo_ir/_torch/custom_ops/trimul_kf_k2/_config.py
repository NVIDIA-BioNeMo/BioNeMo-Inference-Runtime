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
"""Shipped SM90 TriMul KF K2 tiles and their sequence anchors.

One ``configs/D{D}_sm90.json`` bundle per hidden width ``D`` declares the ``kernel_abi`` and maps
``S=<anchor>`` keys to the contraction tile tuned for that token count::

    {"kernel_abi": "sm90", "configs": {"S=512": {"kernel_variant": "K2_0", "tile_n": 256, "cluster_m": 2,
                                                 "defer_kmin": 8, "split_epi": false}}}

``K2_0`` takes ``tile_n``, ``cluster_m`` (a 2-CTA cluster along M shares B), ``defer_kmin`` and
``split_epi``; ``K2_2`` takes the same keys with ``cluster_m`` 2 and pairs its CTAs along N instead
(sharing A); ``K2_1`` takes only ``tile_n``, and runs a 2-CTA cluster at ``tile_n`` 192. A call over
``N`` tokens takes the anchor nearest ``N``, ties going to the smaller. An anchor names one tile for
both contraction directions or one per direction as ``{"out": tile, "in": tile}``, and may list tiles
of distinct widths, of which a call runs the one that pads ``N`` least (:func:`least_padding`). The
contraction itself does not depend on ``D``, which only sets the batch count ``B * D``, so bundles of
different widths share their machine code.

A ``K2_1`` tile of 192 needs an even number of column tiles to fill its clusters; a call with an
odd number runs :func:`fallback` instead, which every shipped bundle therefore also ships.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90"}
KERNEL_VARIANTS = ("K2_0", "K2_1", "K2_2")
#: Contraction k-block, fixed by every variant.
TILE_K = 64
_FILE_RE = re.compile(r"^D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")
_K2_0_KEYS = frozenset({"kernel_variant", "tile_n", "cluster_m", "defer_kmin", "split_epi"})
#: K2_0 / K2_2 column tiles; MN-major (incoming) operands take widths off 64 columns on a 32 B swizzle.
_K2_0_TILE_N = range(128, 257, 16)
_K2_1_KEYS = frozenset({"kernel_variant", "tile_n"})


class TrimulKFK2Tile(NamedTuple):
    """Every axis that changes a K2 image's machine code, besides the direction."""

    kernel_variant: str
    tile_n: int
    cluster_m: int = 1
    defer_kmin: int = 0
    split_epi: bool = False

    @property
    def cluster_n(self) -> int:
        """CTAs per cluster along N: ``K2_1`` multicasts A across a pair at ``tile_n`` 192."""
        return 2 if self.kernel_variant == "K2_1" and self.tile_n == 192 else 1


def parse_tile(entry: Any, source: str) -> TrimulKFK2Tile:
    """Validate one config entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: expected an object")
    variant = entry.get("kernel_variant")
    if variant in ("K2_0", "K2_2"):
        if set(entry) != _K2_0_KEYS:
            raise ValueError(f"{source}: {variant} takes exactly {sorted(_K2_0_KEYS)}")
        tile = TrimulKFK2Tile(
            variant, int(entry["tile_n"]), int(entry["cluster_m"]), int(entry["defer_kmin"]), bool(entry["split_epi"])
        )
        clusters = (1, 2) if variant == "K2_0" else (2,)
        if tile.tile_n not in _K2_0_TILE_N or tile.cluster_m not in clusters or tile.defer_kmin < 0:
            raise ValueError(f"{source}: unsupported {variant} tile {tile}")
        if tile.split_epi and tile.tile_n < 192:
            raise ValueError(f"{source}: split_epi needs tile_n 192 or wider")
        return tile
    if variant == "K2_1":
        if set(entry) != _K2_1_KEYS:
            raise ValueError(f"{source}: K2_1 takes exactly {sorted(_K2_1_KEYS)}")
        tile = TrimulKFK2Tile("K2_1", int(entry["tile_n"]))
        if tile.tile_n not in (128, 192):
            raise ValueError(f"{source}: unsupported K2_1 tile_n {tile.tile_n}")
        return tile
    raise ValueError(f"{source}: unknown K2 variant {variant!r}; expected one of {KERNEL_VARIANTS}")


def parse_tiles(entry: Any, source: str) -> tuple[TrimulKFK2Tile, ...]:
    """One direction's tile, or the tiles a list names (distinct widths), widest first."""
    if not isinstance(entry, list):
        return (parse_tile(entry, source),)
    tiles = tuple(sorted((parse_tile(item, f"{source}[{i}]") for i, item in enumerate(entry)), key=lambda t: -t.tile_n))
    if len(tiles) < 2 or len({tile.tile_n for tile in tiles}) != len(tiles):
        raise ValueError(f"{source}: a tile list names two or more tiles of distinct widths")
    return tiles


def parse_entry(entry: Any, source: str) -> tuple[tuple[TrimulKFK2Tile, ...], tuple[TrimulKFK2Tile, ...]]:
    """``(outgoing tiles, incoming tiles)`` of one anchor: tiles for both directions, or ``{"out": .., "in": ..}``."""
    if isinstance(entry, dict) and set(entry) == {"out", "in"}:
        return parse_tiles(entry["out"], f"{source}: out"), parse_tiles(entry["in"], f"{source}: in")
    tiles = parse_tiles(entry, source)
    return tiles, tiles


def parse_configs(
    configs: dict[str, Any], source: str = "configs", outgoing: bool = True
) -> dict[int, tuple[TrimulKFK2Tile, ...]]:
    """``anchor -> tiles`` (widest first) of one direction for one bundle's ``configs``, in ascending anchor order."""
    entries: dict[int, tuple[TrimulKFK2Tile, ...]] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K2 config key {key!r}; expected 'S=<anchor>'")
        entries[int(match["bucket"])] = parse_entry(entry, f"{source}: {key}")[0 if outgoing else 1]
    if not entries:
        raise ValueError(f"{source}: declares no K2 tile")
    return dict(sorted(entries.items()))


def fallback(tile: TrimulKFK2Tile) -> TrimulKFK2Tile | None:
    """The tile run instead of ``tile`` when its clusters cannot be filled, or ``None``."""
    if tile.cluster_n > 1:
        return TrimulKFK2Tile("K2_1", 128)
    return None


def shipped_tiles(tiles: Iterable[TrimulKFK2Tile | tuple[TrimulKFK2Tile, ...]]) -> tuple[TrimulKFK2Tile, ...]:
    """Configured tiles (every width an anchor lists) plus the fallbacks a call may need, in a stable order."""
    shipped = {tile for entry in tiles for tile in ((entry,) if isinstance(entry, TrimulKFK2Tile) else entry)}
    shipped.update([alternate for tile in shipped if (alternate := fallback(tile)) is not None])
    return tuple(sorted(shipped))


def least_padding(tiles: tuple[TrimulKFK2Tile, ...], n: int) -> TrimulKFK2Tile:
    """The width among an anchor's ``tiles`` that pads ``n`` columns least, ties going to the wider."""
    return min(tiles, key=lambda tile: (-(-n // tile.tile_n) * tile.tile_n, -tile.tile_n))


def runtime_tile(tile: TrimulKFK2Tile, n: int) -> TrimulKFK2Tile:
    """The tile that actually runs ``tile``'s anchor at ``n`` tokens.

    Raises:
        ValueError: The tile defers its stores past more k-blocks than ``n`` has.
    """
    if tile.cluster_n > 1 and -(-n // tile.tile_n) % tile.cluster_n:
        tile = fallback(tile)
    if tile.defer_kmin > -(-n // TILE_K):
        raise ValueError(f"trimul KF K2 tile {tile} defers {tile.defer_kmin} k-blocks; N={n} has fewer")
    return tile


def _bundle_tiles(sm: int, file_name: str, outgoing: bool = True) -> dict[int, tuple[TrimulKFK2Tile, ...]] | None:
    """A bundle's anchors for one direction if it runs ``sm``'s kernel ABI, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi:
        return None
    if bundle.kernel_variant is not None:
        raise ValueError(f"{bundle.source_path}: name the K2 variant per anchor, not per bundle")
    return parse_configs(bundle.configs, bundle.source_path, outgoing)


def anchors(sm: int, D: int, outgoing: bool = True) -> dict[int, tuple[TrimulKFK2Tile, ...]] | None:
    """``anchor -> tiles`` shipped for hidden width ``D`` and one direction on ``sm``, or ``None``."""
    return _bundle_tiles(sm, get_config_file_name(sm, D=D), outgoing)


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, D: int, n: int, outgoing: bool = True) -> tuple[int, TrimulKFK2Tile] | None:
    """The anchor a call over ``n`` tokens selects and the tile that runs it in that direction, or ``None`` if ``D``
    does not ship."""
    entries = anchors(sm, D, outgoing)
    if not entries:
        return None
    bucket = nearest_bucket(list(entries), n)
    return bucket, runtime_tile(least_padding(entries[bucket], n), n)


def shipped_widths(sm: int) -> frozenset[int]:
    """Every hidden width ``D`` the packaged configs declare for ``sm``."""
    if sm not in KERNEL_ABIS:
        return frozenset()
    widths = set()
    for path in sorted(CONFIGS_DIR.glob(f"D*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is not None and _bundle_tiles(sm, path.name):
            widths.add(int(match["D"]))
    return frozenset(widths)
