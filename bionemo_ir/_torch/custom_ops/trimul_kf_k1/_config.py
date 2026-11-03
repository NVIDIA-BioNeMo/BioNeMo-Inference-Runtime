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
"""Shipped TriMul KF K1 variants and their sequence anchors, for SM90 and SM100/SM103.

One ``configs/C{C}_D{D}_sm{sm}.json`` bundle per input width ``C``, hidden width ``D`` and SM declares
the ``kernel_abi`` that runs it and maps ``S=<anchor>`` keys to the K1 variant tuned for that token
count::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K1_0"}}}

A call over ``N`` tokens takes the anchor nearest ``N``, ties going to the smaller. The variant
fixes two contracts with the rest of the chain: the weight fold it reads (the
:data:`INTERLEAVED_VARIANTS` read one interleaved ``[4D, C]`` fold, the others separate projection
and gate folds) and whether it hands its input-LayerNorm row statistics to K3
(:data:`STATS_VARIANTS`, which ``trimul_kf_k3`` pairs with its statistics-reading variant). The
:data:`CENTRED_VARIANTS` drop the LayerNorm mean term and read a fold whose rows are centred
(``fold_input_weights(..., centred=True)``); the others apply the mean term and read either fold. The
fold carries the projection biases, so every variant serves projections with and without them.

Each kernel ABI runs its own variants (:data:`ABI_VARIANTS`): SM90 the WGMMA kernels ``K1_0`` to
``K1_2``, SM100 and SM103 (one ``sm100`` ABI, a bundle per SM) the tcgen05 kernels ``K1_B0``
(C >= 64) and ``K1_B1`` (C = 32). A ``K1_B0`` anchor also names its tile, every
:class:`TrimulKFK1Tile` field::

    {"kernel_abi": "sm100", "configs": {"S=96": {"kernel_variant": "K1_B0", "bnc": 128, "epi_n": 16, "ctas": 2,
                                                 "w_slots": 4, "s_stages": 2, "split": 1, "store_warps": 1,
                                                 "epi_groups": 2}}}

The SM100 bundles place their anchors so the nearest-anchor rule switches tiles exactly where the
measurements did: ``S=32`` serves N <= 64, ``S=96`` up to the next switch, and so on.

An anchor may also name the rule that picks the layout of the ``a``/``b`` planes K1 hands to K2 from
the token count (:data:`AB_LAYOUT_RULES`, default ``dense``). The rule covers every token count that
takes the anchor, so a bundle sets where padding starts by where it puts its anchors: a dense anchor
at ``S`` and a padding one at ``2 M - S`` pad from ``M`` tokens up. This bundle pads from 512::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K1_2"},
                                       "S=896": {"kernel_variant": "K1_2", "ab_layout": "pad_n_mod_16_8"}}}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90", 100: "sm100", 103: "sm100"}
#: The variants each kernel ABI runs.
ABI_VARIANTS = {"sm90": ("K1_0", "K1_1", "K1_2"), "sm100": ("K1_B0", "K1_B1")}
KERNEL_VARIANTS = tuple(variant for variants in ABI_VARIANTS.values() for variant in variants)
#: Variants reading the interleaved ``[4D, C]`` weight fold instead of separate projection and gate folds.
INTERLEAVED_VARIANTS = frozenset({"K1_1", "K1_2"})
#: Variants writing the input-LayerNorm row statistics that ``trimul_kf_k3``'s ``K3_1`` and ``K3_3`` read.
STATS_VARIANTS = frozenset({"K1_2"})
CENTRED_VARIANTS = frozenset({"K1_2"})
#: Variants whose anchors name a :class:`TrimulKFK1Tile`.
TILED_VARIANTS = frozenset({"K1_B0"})
#: Layouts of the ``a``/``b`` planes K1 writes; see :func:`ab_pitch`.
AB_LAYOUTS = ("dense", "padded")
#: Row alignment of the ``padded`` layout, in elements: one 128-byte line of bf16.
AB_PAD_ALIGN = 64
#: Variants that write the ``padded`` layout (a compile-time flavour of each kernel).
PADDED_VARIANTS = frozenset(KERNEL_VARIANTS)
#: Rules an anchor names for its layout, see :func:`ab_layout`.
AB_LAYOUT_RULES = ("dense", "pad_n_mod_16_8", "pad_n_mod_64_56")
_FILE_RE = re.compile(r"^C(?P<C>\d+)_D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")


class TrimulKFK1Tile(NamedTuple):
    """The tuning a ``K1_B0`` anchor names: every axis of its machine code besides ``(C, D)`` and the plane layout.

    A tile's ``2D`` output channels form units of ``bnc`` channels; a unit is one ``M x 2*bnc`` UMMA
    over its projection rows, then its gate rows.
    """

    #: Channels per unit.
    bnc: int
    #: Channels per epilogue subtile; divides ``bnc`` and ``D``.
    epi_n: int
    #: 1, or 2 for one ``cta_group::2`` UMMA over a CTA pair, each CTA holding half of the weights.
    ctas: int
    #: 0 keeps the weights resident in shared memory; otherwise the streamed (unit, K-block) slots.
    w_slots: int
    #: Row-statistics slots.
    s_stages: int
    #: Work items per 128-position tile; more items fill the SMs at small N.
    split: int
    #: Warps per TMA store: 1 stores 32 positions, 2 stores 64 (whole 128-byte rows).
    store_warps: int
    #: Epilogue warp groups: 2 drain one unit each, 4 split every unit's subtiles between two groups.
    epi_groups: int


class _Anchor(NamedTuple):
    """One anchor's variant, its a/b layout rule and, for :data:`TILED_VARIANTS`, its tile."""

    kernel_variant: str
    ab_layout: str
    tile: TrimulKFK1Tile | None


def ab_pitch(n: int, ab_layout: str = "dense") -> int:
    """Row pitch ``P`` of the ``a``/``b`` planes for ``n`` tokens: ``[B, D, N, N]`` views of ``[B, D, P, P]``."""
    if ab_layout == "dense":
        return n
    if ab_layout == "padded":
        return -(-n // AB_PAD_ALIGN) * AB_PAD_ALIGN
    raise ValueError(f"unknown trimul KF a/b layout {ab_layout!r}; expected one of {AB_LAYOUTS}")


def ab_layout(rule: str, n: int) -> str:
    """The a/b layout ``rule`` picks for ``n`` tokens.

    ``pad_n_mod_16_8`` pads where dense rows would start mid-sector (``N % 16 == 8``), which makes
    K2's TMA re-fetch partial lines. ``pad_n_mod_64_56`` pads only those whose rows padding lengthens
    by 8 columns (``N % 64 == 56``): K1 writes every padded column, so at smaller planes wider padding
    costs K1 about what aligned rows save K2. Whether padding pays depends on the shapes, so the token
    counts a rule covers come from the bundle's anchors, not from the rule.
    """
    if rule not in AB_LAYOUT_RULES:
        raise ValueError(f"unknown trimul KF a/b layout rule {rule!r}; expected one of {AB_LAYOUT_RULES}")
    if rule == "dense" or n % 16 != 8 or (rule == "pad_n_mod_64_56" and n % AB_PAD_ALIGN != 56):
        return "dense"
    return "padded"


class TrimulKFK1Selection(NamedTuple):
    """The anchor a call selects, the K1 variant tuned for it, the layout of the planes it writes and, for
    ``K1_B0``, its tile."""

    bucket: int
    kernel_variant: str
    ab_layout: str = "dense"
    tile: TrimulKFK1Tile | None = None

    @property
    def interleaved(self) -> bool:
        """Whether the variant reads the interleaved weight fold."""
        return self.kernel_variant in INTERLEAVED_VARIANTS

    @property
    def writes_stats(self) -> bool:
        """Whether the variant hands its input-LayerNorm row statistics to K3."""
        return self.kernel_variant in STATS_VARIANTS

    @property
    def centred(self) -> bool:
        """Whether the variant drops the input-LayerNorm mean term, so it reads a centred fold."""
        return self.kernel_variant in CENTRED_VARIANTS

    def ab_pitch(self, n: int) -> int:
        """Row pitch of the ``a``/``b`` planes this selection writes for ``n`` tokens."""
        return ab_pitch(n, self.ab_layout)


def parse_tile(entry: dict[str, Any], source: str) -> TrimulKFK1Tile:
    """The :class:`TrimulKFK1Tile` a ``K1_B0`` entry names, validated as far as it goes without ``(C, D)``."""
    missing = set(TrimulKFK1Tile._fields) - set(entry)
    if missing:
        raise ValueError(
            f"{source}: a K1_B0 tile names every one of {TrimulKFK1Tile._fields}; missing {sorted(missing)}"
        )
    fields = {field: entry[field] for field in TrimulKFK1Tile._fields}
    # bool is an int subclass; JSON true/false is not a tile size
    if any(type(value) is not int for value in fields.values()):
        raise ValueError(f"{source}: K1_B0 tile fields are integers, got {fields}")
    tile = TrimulKFK1Tile(**fields)
    if (
        tile.bnc <= 0
        or tile.bnc % 16
        or tile.epi_n <= 0
        or tile.bnc % tile.epi_n
        or tile.ctas not in (1, 2)
        or tile.w_slots < 0
        or tile.s_stages <= 0
        or tile.split <= 0
        or tile.store_warps not in (1, 2)
        or tile.epi_groups not in (2, 4)
        or (tile.bnc // tile.epi_n) % (tile.epi_groups // 2)
    ):
        raise ValueError(f"{source}: unsupported K1_B0 tile {tile}")
    return tile


def _parse_anchors(configs: dict[str, Any], source: str, kernel_abi: str) -> dict[int, _Anchor]:
    """Every anchor of one bundle's ``configs``, validated against ``kernel_abi``, in ascending order."""
    variants = ABI_VARIANTS.get(kernel_abi)
    if variants is None:
        raise ValueError(
            f"{source}: trimul KF K1 ships no {kernel_abi!r} kernels; expected one of {tuple(ABI_VARIANTS)}"
        )
    anchors: dict[int, _Anchor] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K1 config key {key!r}; expected 'S=<anchor>'")
        if not isinstance(entry, dict) or "kernel_variant" not in entry:
            raise ValueError(f"{source}: {key} must hold one 'kernel_variant' and optionally its 'ab_layout' rule")
        variant = entry["kernel_variant"]
        if variant not in variants:
            raise ValueError(
                f"{source}: {key} names unknown {kernel_abi} K1 variant {variant!r}; expected one of {variants}"
            )
        allowed = {"kernel_variant", "ab_layout"}
        if variant in TILED_VARIANTS:
            allowed |= set(TrimulKFK1Tile._fields)
        if set(entry) - allowed:
            raise ValueError(f"{source}: {key} names {sorted(set(entry) - allowed)}, which {variant} does not take")
        rule = entry.get("ab_layout", "dense")
        if rule not in AB_LAYOUT_RULES or (rule != "dense" and variant not in PADDED_VARIANTS):
            raise ValueError(f"{source}: {key} names a/b layout rule {rule!r} {variant} cannot follow")
        tile = parse_tile(entry, f"{source}: {key}") if variant in TILED_VARIANTS else None
        anchors[int(match["bucket"])] = _Anchor(variant, rule, tile)
    if not anchors:
        raise ValueError(f"{source}: declares no K1 variant")
    return dict(sorted(anchors.items()))


def parse_configs(configs: dict[str, Any], source: str = "configs", kernel_abi: str = "sm90") -> dict[int, str]:
    """``anchor -> kernel_variant`` for one ``kernel_abi`` bundle's ``configs``, in ascending anchor order."""
    return {bucket: anchor.kernel_variant for bucket, anchor in _parse_anchors(configs, source, kernel_abi).items()}


def parse_layout_rules(configs: dict[str, Any], source: str = "configs", kernel_abi: str = "sm90") -> dict[int, str]:
    """``anchor -> a/b layout rule`` for one ``kernel_abi`` bundle's ``configs``, in ascending anchor order."""
    return {bucket: anchor.ab_layout for bucket, anchor in _parse_anchors(configs, source, kernel_abi).items()}


def parse_tiles(
    configs: dict[str, Any], source: str = "configs", kernel_abi: str = "sm90"
) -> dict[int, TrimulKFK1Tile | None]:
    """``anchor -> tile`` (``None`` for an untiled variant) for one ``kernel_abi`` bundle's ``configs``."""
    return {bucket: anchor.tile for bucket, anchor in _parse_anchors(configs, source, kernel_abi).items()}


def _bundle_anchors(sm: int, file_name: str) -> dict[int, _Anchor] | None:
    """A bundle's anchors if it runs ``sm``'s kernel ABI, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi:
        return None
    if bundle.kernel_variant is not None:
        raise ValueError(f"{bundle.source_path}: name the K1 variant per anchor, not per bundle")
    return _parse_anchors(bundle.configs, bundle.source_path, kernel_abi)


def anchors(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> kernel_variant`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    entries = _bundle_anchors(sm, get_config_file_name(sm, C=C, D=D))
    return None if entries is None else {bucket: anchor.kernel_variant for bucket, anchor in entries.items()}


def layout_rules(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> a/b layout rule`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    entries = _bundle_anchors(sm, get_config_file_name(sm, C=C, D=D))
    return None if entries is None else {bucket: anchor.ab_layout for bucket, anchor in entries.items()}


def tiles(sm: int, C: int, D: int) -> dict[int, TrimulKFK1Tile | None] | None:
    """``anchor -> tile`` shipped for ``(C, D)`` on ``sm`` (``None`` for an untiled variant), or ``None``."""
    entries = _bundle_anchors(sm, get_config_file_name(sm, C=C, D=D))
    return None if entries is None else {bucket: anchor.tile for bucket, anchor in entries.items()}


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, C: int, D: int, n: int) -> TrimulKFK1Selection | None:
    """The anchor, K1 variant, a/b layout and tile a call over ``n`` tokens runs, or ``None`` if ``(C, D)`` does
    not ship."""
    entries = _bundle_anchors(sm, get_config_file_name(sm, C=C, D=D))
    if not entries:
        return None
    bucket = nearest_bucket(list(entries), n)
    anchor = entries[bucket]
    return TrimulKFK1Selection(bucket, anchor.kernel_variant, ab_layout(anchor.ab_layout, n), anchor.tile)


def shipped_shapes(sm: int) -> frozenset[tuple[int, int]]:
    """Every ``(C, D)`` the packaged configs declare for ``sm``."""
    if sm not in KERNEL_ABIS:
        return frozenset()
    shapes = set()
    for path in sorted(CONFIGS_DIR.glob(f"C*_D*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is not None and _bundle_anchors(sm, path.name):
            shapes.add((int(match["C"]), int(match["D"])))
    return frozenset(shapes)
