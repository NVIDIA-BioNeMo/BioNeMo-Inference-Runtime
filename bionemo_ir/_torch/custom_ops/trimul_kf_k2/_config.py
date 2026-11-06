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
"""Shipped TriMul KF K2 tiles and their sequence anchors, for SM90 and SM100/SM103.

One ``configs/D{D}_sm{sm}.json`` bundle per hidden width ``D`` declares the ``kernel_abi`` and maps
``S=<anchor>`` keys to the contraction tile tuned for that token count::

    {"kernel_abi": "sm90", "configs": {"S=512": {"kernel_variant": "K2_0", "tile_n": 256, "cluster_m": 2,
                                                 "defer_kmin": 8, "split_epi": false}}}

``K2_0`` takes ``tile_n``, ``cluster_m`` (a 2-CTA cluster along M shares B), ``defer_kmin`` and
``split_epi``; ``K2_2`` takes the same keys with ``cluster_m`` 2 and pairs its CTAs along N instead
(sharing A); ``K2_1`` takes only ``tile_n``, and runs a 2-CTA cluster at ``tile_n`` 192. ``K2_0`` and
``K2_2`` also take an optional ``dynamic`` (default false): its clusters claim work units from a counter
instead of a static round-robin, which keeps the sibling tiles of a plane in flight together. A call over
``N`` tokens takes the anchor nearest ``N``, ties going to the smaller. An anchor names one tile for
both contraction directions or one per direction as ``{"out": tile, "in": tile}``, and may list tiles
of distinct widths, of which a call runs the one that pads ``N`` least (:func:`least_padding`). The
contraction itself does not depend on ``D``, which only sets the batch count ``B * D``, so bundles of
different widths share their machine code.

A ``K2_1`` tile of 192 needs an even number of column tiles to fill its clusters; a call with an
odd number runs :func:`fallback` instead, which every shipped bundle therefore also ships.

SM100 and SM103 run the ``sm100`` ABI from identical ``D{D}_sm100.json`` and ``D{D}_sm103.json`` bundles,
whose tiles (:class:`TrimulKFK2Sm100Tile`) name the Blackwell kernels instead::

    {"kernel_abi": "sm100", "configs": {"S=128": {"kernel_variant": "K2_B0", "tile_m": 128, "tile_n": 64},
                                        "S=512": {"kernel_variant": "K2_B1", "tile_n": 256, "cluster_n": 1,
                                                  "clc": false}}}

``K2_B0`` takes ``tile_m`` (128, or 256 on a CTA pair), ``tile_n`` and optionally ``cluster_n`` (2: two 1-SM
CTAs on adjacent column tiles share A; 1 by default); ``K2_B1`` runs 256-row tiles on CTA pairs and takes
``tile_n``, ``cluster_n`` (two pairs along N share A) and ``clc`` (cluster launch control hands out the work
instead of a static split); ``K2_B2`` runs 256 x 256 pair tiles with tail tiles ``tail_w`` wide (a multiple
of 32 up to 224) in both directions. Anchors and tile lists work as on SM90, except that a list may also hold tiles of one
width that pad ``N`` differently: :meth:`TrimulKFK2Sm100Tile.padded` counts the column tiles a cluster runs
past ``N`` and the columns tail tiles save, and ties go to the wider tile, then to the one without tails. A
``K2_B0`` tile on paired CTAs runs them unpaired where ``N`` takes an odd number of its column tiles, as anchors
sit at multiples of 8 and a pair would otherwise compute a whole column tile past ``N``: its :func:`fallback`,
which shipped bundles therefore also compile.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90", 100: "sm100", 103: "sm100"}
KERNEL_VARIANTS = ("K2_0", "K2_1", "K2_2")
#: The ``sm100`` ABI's kernels: ``K2_B0`` sized to small planes, ``K2_B1`` on CTA pairs, ``K2_B2`` with tail tiles.
SM100_KERNEL_VARIANTS = ("K2_B0", "K2_B1", "K2_B2")
#: Contraction k-block, fixed by every variant.
TILE_K = 64
#: Words of a dynamic tile's work counter (one used; the rest keep it on its own 16-byte line).
COUNTER_WORDS = 4
_FILE_RE = re.compile(r"^D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")
_K2_0_KEYS = frozenset({"kernel_variant", "tile_n", "cluster_m", "defer_kmin", "split_epi"})
_K2_0_OPTIONAL = frozenset({"dynamic"})
#: K2_0 / K2_2 column tiles; MN-major (incoming) operands take widths off 64 columns on a 32 B swizzle.
_K2_0_TILE_N = range(128, 257, 16)
_K2_1_KEYS = frozenset({"kernel_variant", "tile_n"})
_SM100_KEYS = {
    "K2_B0": frozenset({"kernel_variant", "tile_m", "tile_n"}),
    "K2_B1": frozenset({"kernel_variant", "tile_n", "cluster_n", "clc"}),
    "K2_B2": frozenset({"kernel_variant", "tail_w"}),
}
#: Keys an SM100 variant may add to its required ones.
_SM100_OPTIONAL = {"K2_B0": frozenset({"cluster_n"})}
#: SM100 column tiles: whole 32-wide epilogue subtiles, which also split evenly across a CTA pair's halves of B.
_SM100_TILE_N = range(32, 257, 32)
#: Rows of a CTA-pair tile, and the column block whose remainder ``K2_B2``'s tail tiles cover.
_PAIR_TILE = 256


class TrimulKFK2Tile(NamedTuple):
    """Every axis that changes a K2 image's machine code, besides the direction."""

    kernel_variant: str
    tile_n: int
    cluster_m: int = 1
    defer_kmin: int = 0
    split_epi: bool = False
    #: Clusters claim work units from a global counter (``K2_0`` and ``K2_2``).
    dynamic: bool = False

    @property
    def cluster_n(self) -> int:
        """CTAs per cluster along N: ``K2_1`` multicasts A across a pair at ``tile_n`` 192."""
        return 2 if self.kernel_variant == "K2_1" and self.tile_n == 192 else 1

    def padded(self, n: int) -> int:
        """Columns a call over ``n`` tokens computes, padding included; a cluster that cannot fill falls back.

        ``K2_2`` pairs two column tiles along N (its ``cluster_m`` CTAs share A), so its clusters compute whole
        ``cluster_m * tile_n`` column spans and one past ``N`` still runs.
        """
        span = self.tile_n * (self.cluster_m if self.kernel_variant == "K2_2" else 1)
        return -(-n // span) * span


class TrimulKFK2Sm100Tile(NamedTuple):
    """Every axis that changes an SM100/SM103 K2 image's machine code, besides the direction."""

    kernel_variant: str
    tile_m: int = _PAIR_TILE
    tile_n: int = _PAIR_TILE
    cluster_n: int = 1
    clc: bool = False
    tail_w: int = 0

    @property
    def dynamic(self) -> bool:
        """SM100/SM103 tiles run a static schedule (or CLC), never the SM90 work counter."""
        return False

    def padded(self, n: int) -> int:
        """Columns a call over ``n`` tokens computes, padding included.

        The ``cluster_n`` CTAs or pairs of a cluster take adjacent column tiles, so one past ``N`` still runs; tail
        tiles cover the last ``N % 256`` columns ``tail_w`` wide when the remainder fits in them.
        """
        rem = n % _PAIR_TILE
        if self.tail_w and n > _PAIR_TILE and 0 < rem <= self.tail_w:
            return n - rem + self.tail_w
        span = self.tile_n * self.cluster_n
        return -(-n // span) * span


K2Tile = TrimulKFK2Tile | TrimulKFK2Sm100Tile


def _parse_sm100_tile(entry: dict[str, Any], source: str) -> TrimulKFK2Sm100Tile:
    """Validate one ``sm100`` config entry."""
    variant = entry.get("kernel_variant")
    keys = _SM100_KEYS.get(variant)
    if keys is None:
        raise ValueError(f"{source}: unknown SM100 K2 variant {variant!r}; expected one of {SM100_KERNEL_VARIANTS}")
    optional = _SM100_OPTIONAL.get(variant, frozenset())
    if not keys <= set(entry) <= keys | optional:
        also = f", optionally {sorted(optional)}" if optional else ""
        raise ValueError(f"{source}: {variant} takes exactly {sorted(keys)}{also}")
    if variant == "K2_B0":
        tile = TrimulKFK2Sm100Tile(
            variant, tile_m=int(entry["tile_m"]), tile_n=int(entry["tile_n"]), cluster_n=int(entry.get("cluster_n", 1))
        )
        supported = (
            tile.tile_m in (128, _PAIR_TILE)
            and tile.tile_n in _SM100_TILE_N
            and (tile.cluster_n == 1 or (tile.cluster_n == 2 and tile.tile_m == 128))
        )
    elif variant == "K2_B1":
        if not isinstance(entry["clc"], bool):
            raise ValueError(f"{source}: K2_B1 takes clc as true or false")
        tile = TrimulKFK2Sm100Tile(
            variant, tile_n=int(entry["tile_n"]), cluster_n=int(entry["cluster_n"]), clc=entry["clc"]
        )
        supported = tile.tile_n in _SM100_TILE_N and tile.cluster_n in (1, 2)
    else:
        tile = TrimulKFK2Sm100Tile(variant, tail_w=int(entry["tail_w"]))
        supported = tile.tail_w % 32 == 0 and 32 <= tile.tail_w <= 224
    if not supported:
        raise ValueError(f"{source}: unsupported {variant} tile {tile}")
    return tile


def parse_tile(entry: Any, source: str, kernel_abi: str = "sm90") -> K2Tile:
    """Validate one config entry of a ``kernel_abi`` bundle."""
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: expected an object")
    if kernel_abi == "sm100":
        return _parse_sm100_tile(entry, source)
    variant = entry.get("kernel_variant")
    if variant in ("K2_0", "K2_2"):
        if not _K2_0_KEYS <= set(entry) <= _K2_0_KEYS | _K2_0_OPTIONAL:
            raise ValueError(
                f"{source}: {variant} takes exactly {sorted(_K2_0_KEYS)}, optionally {sorted(_K2_0_OPTIONAL)}"
            )
        tile = TrimulKFK2Tile(
            variant,
            int(entry["tile_n"]),
            int(entry["cluster_m"]),
            int(entry["defer_kmin"]),
            bool(entry["split_epi"]),
            bool(entry.get("dynamic", False)),
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


def parse_tiles(entry: Any, source: str, kernel_abi: str = "sm90") -> tuple[K2Tile, ...]:
    """One direction's tile, or the tiles a list names (each padding N its own way), widest first."""
    if not isinstance(entry, list):
        return (parse_tile(entry, source, kernel_abi),)
    tiles = tuple(
        sorted(
            (parse_tile(item, f"{source}[{i}]", kernel_abi) for i, item in enumerate(entry)),
            key=lambda t: (-t.tile_n, t.kernel_variant),
        )
    )
    if kernel_abi == "sm100":
        if len(tiles) < 2 or len({(tile.tile_n * tile.cluster_n, tile.tail_w) for tile in tiles}) != len(tiles):
            raise ValueError(f"{source}: a tile list names two or more tiles that pad N differently")
        return tiles
    if len(tiles) < 2 or len({tile.tile_n for tile in tiles}) != len(tiles):
        raise ValueError(f"{source}: a tile list names two or more tiles of distinct widths")
    return tiles


def parse_entry(entry: Any, source: str, kernel_abi: str = "sm90") -> tuple[tuple[K2Tile, ...], tuple[K2Tile, ...]]:
    """``(outgoing tiles, incoming tiles)`` of one anchor: tiles for both directions, or ``{"out": .., "in": ..}``."""
    if isinstance(entry, dict) and set(entry) == {"out", "in"}:
        out = parse_tiles(entry["out"], f"{source}: out", kernel_abi)
        incoming = parse_tiles(entry["in"], f"{source}: in", kernel_abi)
    else:
        out = incoming = parse_tiles(entry, source, kernel_abi)
    return out, incoming


def parse_configs(
    configs: dict[str, Any], source: str = "configs", outgoing: bool = True, kernel_abi: str = "sm90"
) -> dict[int, tuple[K2Tile, ...]]:
    """``anchor -> tiles`` (widest first) of one direction for one ``kernel_abi`` bundle's ``configs``, in ascending
    anchor order."""
    entries: dict[int, tuple[K2Tile, ...]] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K2 config key {key!r}; expected 'S=<anchor>'")
        entries[int(match["bucket"])] = parse_entry(entry, f"{source}: {key}", kernel_abi)[0 if outgoing else 1]
    if not entries:
        raise ValueError(f"{source}: declares no K2 tile")
    return dict(sorted(entries.items()))


def fallback(tile: K2Tile) -> K2Tile | None:
    """The tile run instead of ``tile`` when its clusters cannot be filled, or ``None``."""
    if isinstance(tile, TrimulKFK2Sm100Tile):
        # K2_B1's pairs of pairs run past N by design (padded() counts them); K2_B0's paired CTAs run alone.
        return tile._replace(cluster_n=1) if tile.kernel_variant == "K2_B0" and tile.cluster_n > 1 else None
    if tile.cluster_n > 1:
        return TrimulKFK2Tile("K2_1", 128)
    return None


def shipped_tiles(tiles: Iterable[K2Tile | tuple[K2Tile, ...]]) -> tuple[K2Tile, ...]:
    """Configured tiles (every width an anchor lists) plus the fallbacks a call may need, in a stable order."""
    shipped = {
        tile
        for entry in tiles
        for tile in ((entry,) if isinstance(entry, (TrimulKFK2Tile, TrimulKFK2Sm100Tile)) else entry)
    }
    shipped.update([alternate for tile in shipped if (alternate := fallback(tile)) is not None])
    return tuple(sorted(shipped))


def least_padding(tiles: tuple[K2Tile, ...], n: int) -> K2Tile:
    """The tile among an anchor's ``tiles`` that pads ``n`` columns least, ties going to the wider, then to the one
    without tail tiles."""
    return min(tiles, key=lambda tile: (tile.padded(n), -tile.tile_n, tile.kernel_variant == "K2_B2"))


def runtime_tile(tile: K2Tile, n: int) -> K2Tile:
    """The tile that actually runs ``tile``'s anchor at ``n`` tokens.

    Raises:
        ValueError: The tile defers its stores past more k-blocks than ``n`` has.
    """
    if isinstance(tile, TrimulKFK2Sm100Tile):
        alternate = fallback(tile)
        return alternate if alternate is not None and -(-n // tile.tile_n) % tile.cluster_n else tile
    if tile.cluster_n > 1 and -(-n // tile.tile_n) % tile.cluster_n:
        tile = fallback(tile)
    if tile.defer_kmin > -(-n // TILE_K):
        raise ValueError(f"trimul KF K2 tile {tile} defers {tile.defer_kmin} k-blocks; N={n} has fewer")
    return tile


def _bundle_tiles(sm: int, file_name: str, outgoing: bool = True) -> dict[int, tuple[K2Tile, ...]] | None:
    """A bundle's anchors for one direction if it runs ``sm``'s kernel ABI, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi:
        return None
    if bundle.kernel_variant is not None:
        raise ValueError(f"{bundle.source_path}: name the K2 variant per anchor, not per bundle")
    return parse_configs(bundle.configs, bundle.source_path, outgoing, kernel_abi)


def anchors(sm: int, D: int, outgoing: bool = True) -> dict[int, tuple[K2Tile, ...]] | None:
    """``anchor -> tiles`` shipped for hidden width ``D`` and one direction on ``sm``, or ``None``."""
    return _bundle_tiles(sm, get_config_file_name(sm, D=D), outgoing)


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, D: int, n: int, outgoing: bool = True) -> tuple[int, K2Tile] | None:
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
