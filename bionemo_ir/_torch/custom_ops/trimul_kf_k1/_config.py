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
"""Shipped SM90 TriMul KF K1 variants and their sequence anchors.

One ``configs/C{C}_D{D}_sm90.json`` bundle per input width ``C`` and hidden width ``D`` declares the
``kernel_abi`` that runs it and maps ``S=<anchor>`` keys to the K1 variant tuned for that token
count::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K1_0"}}}

A call over ``N`` tokens takes the anchor nearest ``N``, ties going to the smaller. The variant
fixes two contracts with the rest of the chain: the weight fold it reads (the
:data:`INTERLEAVED_VARIANTS` read one interleaved ``[4D, C]`` fold, ``K1_0`` separate projection
and gate folds) and whether it hands its input-LayerNorm row statistics to K3
(:data:`STATS_VARIANTS`, which ``trimul_kf_k3`` pairs with its statistics-reading variant). The
fold carries the projection biases, so every variant serves projections with and without them.

An anchor may also name the rule that picks the layout of the ``a``/``b`` planes K1 hands to K2 from
the token count (:data:`AB_LAYOUT_RULES`, default ``dense``)::

    {"kernel_abi": "sm90", "configs": {"S=128": {"kernel_variant": "K1_2", "ab_layout": "pad_n_mod_16_8"}}}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90"}
KERNEL_VARIANTS = ("K1_0", "K1_1", "K1_2")
#: Variants reading the interleaved ``[4D, C]`` weight fold instead of separate projection and gate folds.
INTERLEAVED_VARIANTS = frozenset({"K1_1", "K1_2"})
#: Variants writing the input-LayerNorm row statistics that ``trimul_kf_k3``'s ``K3_1`` and ``K3_3`` read.
STATS_VARIANTS = frozenset({"K1_2"})
#: Layouts of the ``a``/``b`` planes K1 writes; see :func:`ab_pitch`.
AB_LAYOUTS = ("dense", "padded")
#: Row alignment of the ``padded`` layout, in elements: one 128-byte line of bf16.
AB_PAD_ALIGN = 64
#: Variants that write the ``padded`` layout (a compile-time flavour of each kernel).
PADDED_VARIANTS = frozenset({"K1_0", "K1_1", "K1_2"})
#: Rules an anchor names for its layout, see :func:`ab_layout`.
AB_LAYOUT_RULES = ("dense", "pad_n_mod_16_8")
#: No rule pads below this many tokens.
AB_PAD_MIN_N = 512
_FILE_RE = re.compile(r"^C(?P<C>\d+)_D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")


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
    K2's TMA re-fetch partial lines.
    """
    if rule not in AB_LAYOUT_RULES:
        raise ValueError(f"unknown trimul KF a/b layout rule {rule!r}; expected one of {AB_LAYOUT_RULES}")
    if rule == "dense" or n < AB_PAD_MIN_N or n % 16 != 8:
        return "dense"
    return "padded"


class TrimulKFK1Selection(NamedTuple):
    """The anchor a call selects, the K1 variant tuned for it and the layout of the planes it writes."""

    bucket: int
    kernel_variant: str
    ab_layout: str = "dense"

    @property
    def interleaved(self) -> bool:
        """Whether the variant reads the interleaved weight fold."""
        return self.kernel_variant in INTERLEAVED_VARIANTS

    @property
    def writes_stats(self) -> bool:
        """Whether the variant hands its input-LayerNorm row statistics to K3."""
        return self.kernel_variant in STATS_VARIANTS

    def ab_pitch(self, n: int) -> int:
        """Row pitch of the ``a``/``b`` planes this selection writes for ``n`` tokens."""
        return ab_pitch(n, self.ab_layout)


def parse_configs(configs: dict[str, Any], source: str = "configs") -> dict[int, str]:
    """``anchor -> kernel_variant`` for one bundle's ``configs``, in ascending anchor order."""
    entries: dict[int, str] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K1 config key {key!r}; expected 'S=<anchor>'")
        if not isinstance(entry, dict) or "kernel_variant" not in entry or set(entry) - {"kernel_variant", "ab_layout"}:
            raise ValueError(f"{source}: {key} must hold one 'kernel_variant' and optionally its 'ab_layout' rule")
        variant = entry["kernel_variant"]
        if variant not in KERNEL_VARIANTS:
            raise ValueError(f"{source}: {key} names unknown K1 variant {variant!r}; expected one of {KERNEL_VARIANTS}")
        rule = entry.get("ab_layout", "dense")
        if rule not in AB_LAYOUT_RULES or (rule != "dense" and variant not in PADDED_VARIANTS):
            raise ValueError(f"{source}: {key} names a/b layout rule {rule!r} {variant} cannot follow")
        entries[int(match["bucket"])] = variant
    if not entries:
        raise ValueError(f"{source}: declares no K1 variant")
    return dict(sorted(entries.items()))


def parse_layout_rules(configs: dict[str, Any], source: str = "configs") -> dict[int, str]:
    """``anchor -> a/b layout rule`` for one bundle's ``configs`` (validated by :func:`parse_configs`)."""
    parse_configs(configs, source)
    rules = {int(_KEY_RE.fullmatch(key)["bucket"]): entry.get("ab_layout", "dense") for key, entry in configs.items()}
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
        raise ValueError(f"{bundle.source_path}: name the K1 variant per anchor, not per bundle")
    return parse(bundle.configs, bundle.source_path)


def anchors(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> kernel_variant`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    return _bundle_entries(sm, get_config_file_name(sm, C=C, D=D))


def layout_rules(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> a/b layout rule`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    return _bundle_entries(sm, get_config_file_name(sm, C=C, D=D), parse_layout_rules)


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, C: int, D: int, n: int) -> TrimulKFK1Selection | None:
    """The anchor, K1 variant and a/b layout a call over ``n`` tokens runs, or ``None`` if ``(C, D)`` does not ship."""
    entries = anchors(sm, C, D)
    if not entries:
        return None
    bucket = nearest_bucket(list(entries), n)
    rule = layout_rules(sm, C, D)[bucket]
    return TrimulKFK1Selection(bucket, entries[bucket], ab_layout(rule, n))


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
