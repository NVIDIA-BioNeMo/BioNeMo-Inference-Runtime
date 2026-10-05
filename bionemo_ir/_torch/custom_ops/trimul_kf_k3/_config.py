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
``K3_2`` / ``K3_3`` their ping-pong counterparts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
KERNEL_ABIS = {90: "sm90"}
KERNEL_VARIANTS = ("K3_0", "K3_1", "K3_2", "K3_3")
#: Variants reading the input-LayerNorm row statistics K1 handed off.
STATS_VARIANTS = frozenset({"K3_1", "K3_3"})
_FILE_RE = re.compile(r"^C(?P<C>\d+)_D(?P<D>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^S=(?P<bucket>[1-9]\d*)$")


class TrimulKFK3Selection(NamedTuple):
    """The anchor a call selects and the K3 variant tuned for it."""

    bucket: int
    kernel_variant: str

    @property
    def reads_stats(self) -> bool:
        """Whether the variant reads K1's input-LayerNorm row statistics."""
        return self.kernel_variant in STATS_VARIANTS


def parse_configs(configs: dict[str, Any], source: str = "configs") -> dict[int, str]:
    """``anchor -> kernel_variant`` for one bundle's ``configs``, in ascending anchor order."""
    entries: dict[int, str] = {}
    for key, entry in configs.items():
        match = _KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{source}: invalid trimul KF K3 config key {key!r}; expected 'S=<anchor>'")
        if not isinstance(entry, dict) or set(entry) != {"kernel_variant"}:
            raise ValueError(f"{source}: {key} must hold exactly one 'kernel_variant'")
        variant = entry["kernel_variant"]
        if variant not in KERNEL_VARIANTS:
            raise ValueError(f"{source}: {key} names unknown K3 variant {variant!r}; expected one of {KERNEL_VARIANTS}")
        entries[int(match["bucket"])] = variant
    if not entries:
        raise ValueError(f"{source}: declares no K3 variant")
    return dict(sorted(entries.items()))


def _bundle_entries(sm: int, file_name: str) -> dict[int, str] | None:
    """A bundle's anchors if it runs ``sm``'s kernel ABI, else ``None``."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    bundle = load_kernel_configs(str(CONFIGS_DIR), file_name)
    if bundle is None or bundle.kernel_abi != kernel_abi:
        return None
    if bundle.kernel_variant is not None:
        raise ValueError(f"{bundle.source_path}: name the K3 variant per anchor, not per bundle")
    return parse_configs(bundle.configs, bundle.source_path)


def anchors(sm: int, C: int, D: int) -> dict[int, str] | None:
    """``anchor -> kernel_variant`` shipped for ``(C, D)`` on ``sm``, or ``None``."""
    return _bundle_entries(sm, get_config_file_name(sm, C=C, D=D))


def nearest_bucket(buckets: tuple[int, ...] | list[int], seqlen: int) -> int:
    """The anchor closest to ``seqlen``, ties going to the smaller."""
    return min(buckets, key=lambda anchor: (abs(anchor - seqlen), anchor))


def select(sm: int, C: int, D: int, n: int) -> TrimulKFK3Selection | None:
    """The anchor and K3 variant a call over ``n`` tokens runs, or ``None`` when ``(C, D)`` does not ship."""
    entries = anchors(sm, C, D)
    if not entries:
        return None
    bucket = nearest_bucket(list(entries), n)
    return TrimulKFK3Selection(bucket, entries[bucket])


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
