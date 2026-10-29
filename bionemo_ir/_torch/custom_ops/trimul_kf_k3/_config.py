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
"""Shipped SM90 TriMul KF K3 variants and their sequence anchors.

One ``configs/C{C}_D{D}_sm90.json`` bundle per output width ``C`` and hidden width ``D`` declares
the ``kernel_abi`` and maps ``S=<anchor>`` keys to the K3 variant tuned for that token count::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K3_0"}}}

A call over ``N`` tokens takes the anchor nearest ``N``, ties going to the smaller. The variant must
match the K1 that ran before it: :data:`STATS_VARIANTS` read the input-LayerNorm row statistics
``trimul_kf_k1``'s statistics-writing variant leaves, the others re-reduce them from ``x``. Each
variant ships with and without the fused residual. ``K3_0`` / ``K3_1`` are the cooperative kernels,
``K3_2`` / ``K3_3`` their ping-pong counterparts. ``K3_4`` keeps a column tile's weights resident in
shared memory, reads them centred (:data:`CENTRED_VARIANTS`), takes every batch item in one launch and
ships for C=384, D=256 at every sequence length.

The :data:`PITCHED_VARIANTS` also read a product whose rows sit further apart than ``N``, as
``trimul_kf_k2`` writes it at K1's padded a/b pitch, from a second image of the variant that tiles
the product per row. An anchor may name the rule that picks, from the token count, where its variant
reads such a product (:data:`PROD_LAYOUT_RULES`, default ``dense``); the chain then has K2 write the
product pitched exactly where K1 pads its planes and this rule reads a pitch::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K3_4",
                                                 "prod_layout": "pitched_n_mod_64_ge_56"}}}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90"}
KERNEL_VARIANTS = ("K3_0", "K3_1", "K3_2", "K3_3", "K3_4")
#: Variants reading the input-LayerNorm row statistics K1 handed off.
STATS_VARIANTS = frozenset({"K3_1", "K3_3", "K3_4"})
#: Variants reading the fold's centred, row-permuted weights (:attr:`TrimulKFOutputFold.centred`).
CENTRED_VARIANTS = frozenset({"K3_4"})
#: Variants whose row tiles may straddle batch items, so every batch shares one launch.
ONE_LAUNCH_VARIANTS = frozenset({"K3_4"})
#: Variants that also read a product at a row pitch past N, from per-row tiles: every variant a shipped shape selects.
PITCHED_VARIANTS = frozenset({"K3_0", "K3_2", "K3_3", "K3_4"})
PROD_LAYOUTS = ("dense", "pitched")
PROD_LAYOUT_RULES = ("dense", "pitched", "pitched_n_mod_64_ge_56")
_FILE_RE = re.compile(r"^C(?P<C>\d+)_D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")


def prod_layout(rule: str, n: int) -> str:
    """The product layout (:data:`PROD_LAYOUTS`) ``rule`` reads for ``n`` tokens where K1 pads its planes.

    ``pitched`` reads K2's pitched product wherever K1 pads. ``pitched_n_mod_64_ge_56`` reads one only where 64-column
    row chunks overrun N by at most 8 columns (``N % 64 >= 56``; K1 pads only ``N % 16 == 8``): a per-row tile's
    columns past N are work the flat tiles do not do, and with more of them the aligned product no longer pays for it.
    """
    if rule not in PROD_LAYOUT_RULES:
        raise ValueError(f"unknown trimul KF product layout rule {rule!r}; expected one of {PROD_LAYOUT_RULES}")
    if rule == "pitched_n_mod_64_ge_56":
        return "pitched" if n % 64 >= 56 else "dense"
    return rule


class TrimulKFK3Selection(NamedTuple):
    """The anchor a call selects, the K3 variant tuned for it and the product layout it reads where K1 pads."""

    bucket: int
    kernel_variant: str
    prod_layout: str = "dense"

    @property
    def reads_stats(self) -> bool:
        """Whether the variant reads K1's input-LayerNorm row statistics."""
        return self.kernel_variant in STATS_VARIANTS

    @property
    def centred(self) -> bool:
        """Whether the variant reads the fold's centred, row-permuted weights."""
        return self.kernel_variant in CENTRED_VARIANTS

    @property
    def one_launch(self) -> bool:
        """Whether the variant takes every batch item in one launch."""
        return self.kernel_variant in ONE_LAUNCH_VARIANTS

    @property
    def reads_pitch(self) -> bool:
        """Whether the call reads K2's ``pitched`` product (rows further apart than ``N``) where K1 pads its planes."""
        return self.prod_layout == "pitched"


def supports(kernel_variant: str, C: int, D: int) -> bool:
    """Whether ``kernel_variant`` compiles for output width ``C`` and hidden width ``D``.

    The :data:`ONE_LAUNCH_VARIANTS` keep a 128-column tile's rows of both weights in shared memory
    (``128 (C + D)`` bf16) and need the first four of their ``C / 64`` x chunks and their ``D / 64`` P chunks to
    carry the previous slab's epilogue and the gate; the other variants take every shipped shape.
    """
    if kernel_variant in ONE_LAUNCH_VARIANTS:
        return C % 128 == 0 and D % 64 == 0 and C >= 384 and D >= 256 and C + D <= 640
    return kernel_variant in KERNEL_VARIANTS


def parse_configs(configs: dict[str, Any], source: str = "configs") -> dict[int, str]:
    """``anchor -> kernel_variant`` for one bundle's ``configs``, in ascending anchor order."""
    entries: dict[int, str] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K3 config key {key!r}; expected 'S=<anchor>'")
        if (
            not isinstance(entry, dict)
            or "kernel_variant" not in entry
            or set(entry) - {"kernel_variant", "prod_layout"}
        ):
            raise ValueError(f"{source}: {key} must hold one 'kernel_variant' and optionally its 'prod_layout' rule")
        variant = entry["kernel_variant"]
        if variant not in KERNEL_VARIANTS:
            raise ValueError(f"{source}: {key} names unknown K3 variant {variant!r}; expected one of {KERNEL_VARIANTS}")
        rule = entry.get("prod_layout", "dense")
        if rule not in PROD_LAYOUT_RULES or (rule != "dense" and variant not in PITCHED_VARIANTS):
            raise ValueError(f"{source}: {key} names product layout rule {rule!r} {variant} cannot follow")
        entries[int(match["bucket"])] = variant
    if not entries:
        raise ValueError(f"{source}: declares no K3 variant")
    return dict(sorted(entries.items()))


def parse_layout_rules(configs: dict[str, Any], source: str = "configs") -> dict[int, str]:
    """``anchor -> product layout rule`` for one bundle's ``configs`` (validated by :func:`parse_configs`)."""
    parse_configs(configs, source)
    rules = {int(_KEY_RE.fullmatch(key)["bucket"]): entry.get("prod_layout", "dense") for key, entry in configs.items()}
    return dict(sorted(rules.items()))


def _bundle_entries(sm: int, file_name: str, parse: Any = parse_configs) -> dict[int, str] | None:
    """A bundle's anchors (``parse`` of its configs) if it runs ``sm``'s kernel ABI, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi:
        return None
    if bundle.kernel_variant is not None:
        raise ValueError(f"{bundle.source_path}: name the K3 variant per anchor, not per bundle")
    return parse(bundle.configs, bundle.source_path)


def anchors(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> kernel_variant`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    return _bundle_entries(sm, get_config_file_name(sm, C=C, D=D))


def layout_rules(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> product layout rule`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    return _bundle_entries(sm, get_config_file_name(sm, C=C, D=D), parse_layout_rules)


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, C: int, D: int, n: int) -> TrimulKFK3Selection | None:
    """The anchor, K3 variant and product layout a call over ``n`` tokens runs, or ``None`` when ``(C, D)`` does not
    ship."""
    entries = anchors(sm, C, D)
    if not entries:
        return None
    bucket = nearest_bucket(list(entries), n)
    return TrimulKFK3Selection(bucket, entries[bucket], prod_layout(layout_rules(sm, C, D)[bucket], n))


def shipped_shapes(sm: int) -> frozenset[tuple[int, int]]:
    """Every ``(C, D)`` the packaged configs declare for ``sm``."""
    if sm not in KERNEL_ABIS:
        return frozenset()
    shapes = set()
    for path in sorted(CONFIGS_DIR.glob(f"C*_D*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is not None and _bundle_entries(sm, path.name):
            shapes.add((int(match["C"]), int(match["D"])))
    return frozenset(shapes)
