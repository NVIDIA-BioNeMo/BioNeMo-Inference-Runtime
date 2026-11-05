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
"""Configs of the conditioned transition MLP.

One ``configs/CT{width}_H{hidden}_sm{sm}.json`` per shape the kernel serves. Each entry, keyed
``act=<activation>|tile=<tile>``, holds the kernel's constructor arguments for one kind of call::

    {"kernel_abi": "sm100", "configs": {"act=silu_gate|tile=base": {...}, "act=silu_gate_3way|tile=base": {...}}}

The activation is the 2-way SwiGLU (``silu_gate``) or the 3-way one (``silu_gate_3way``), and each has a ``base`` tile,
which runs any call. ``staged`` stages each atom's conditioning in shared memory, which pays off when a call has at
most one row tile per CTA pair. ``sample_major`` and ``sample_stream`` take a call whose rows are ``samples`` copies of
the atoms in sample-major tiles of whole atoms, which read each atom's conditioning once; the first stages it, the
second streams it. :func:`choose_config` picks one of an activation's tiles per call. The file names carry ``CT`` so
that the transition families' files can share one ``BIOIR_TUNED_CONFIG_FOLDER``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
# SM103 runs the SM100 kernel ABI from images compiled for it. Its configs copy SM100's (B200-measured) until a B300
# tunes them.
KERNEL_ABIS = {100: "sm100", 103: "sm100"}
#: W1 row blocks per activation: the SwiGLU's value and gate rows, then for the 3-way one its second value rows.
W1_BLOCKS = {"silu_gate": 2, "silu_gate_3way": 3}
ACTIVATIONS = tuple(W1_BLOCKS)
#: Every tile a bundle may name, by the calls it serves; see the module docstring.
TILES = ("base", "staged", "sample_major", "sample_stream")
#: Rows of a CTA's tile; a CTA pair takes two.
CTA_ROWS = 128
# The kernel's constructor arguments a config may set.
_PARAMS = frozenset(
    {
        "x_stages",
        "w2_stages",
        "gemm2_lag",
        "out_stages",
        "norm_ahead",
        "gate_ahead",
        "norm_group",
        "norm_tree",
        "cond_stage",
        "gate_group",
        "sample_major",
    }
)
_FILE_RE = re.compile(r"^CT(?P<width>\d+)_H(?P<hidden>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(rf"^act=(?P<activation>{'|'.join(ACTIVATIONS)})\|tile=(?P<tile>{'|'.join(TILES)})$")


def config_file_name(sm: int, width: int, hidden: int) -> str:
    """The config file name for one ``(width, hidden)`` on one SM."""
    return get_config_file_name(sm, CT=width, H=hidden)


def config_key(activation: str, tile: str) -> str:
    """The ``configs`` entry of ``activation``'s ``tile``; the inverse of :func:`parse_config_key`."""
    return f"act={activation}|tile={tile}"


def parse_config_key(key: str) -> tuple[str, str]:
    """``(activation, tile)`` of one ``configs`` key."""
    match = _KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(
            f"invalid conditioned transition config key {key!r}; expected act=<{'|'.join(ACTIVATIONS)}>|"
            f"tile=<{'|'.join(TILES)}>"
        )
    return match["activation"], match["tile"]


def image_key(params: Mapping[str, int]) -> tuple[bool, int]:
    """``(cond_stage, samples)`` of a config: with the activation, the axes that tell its image apart, on which the
    CUBIN library keys it."""
    return bool(params.get("cond_stage", 0)), int(params.get("sample_major", 0))


def validate_config(
    kernel_abi: str, kernel_variant: str | None, configs: Mapping[str, Any], name: str
) -> tuple[str, dict[str, dict[str, dict[str, int]]]]:
    """Return a config file's kernel ABI and, per activation, its configs by tile; raise if it names anything else,
    an activation lacks its ``base`` tile, a config's image axes (:func:`image_key`) do not match its tile, or an
    activation's sample-major tiles take different samples."""
    if kernel_abi not in KERNEL_ABIS.values():
        raise ValueError(
            f"{name}: the conditioned transition has no {kernel_abi!r} kernel; expected one of "
            f"{sorted(set(KERNEL_ABIS.values()))}"
        )
    if kernel_variant is not None:
        raise ValueError(f"{name}: the conditioned transition has no kernel variants")
    parsed: dict[str, dict[str, dict[str, int]]] = {}
    for key, entry in configs.items():
        activation, tile = parse_config_key(key)
        if (
            not isinstance(entry, Mapping)
            or not set(entry) <= _PARAMS
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in entry.values()
            )
        ):
            raise ValueError(f"{name}: config {key!r} takes non-negative integers for {sorted(_PARAMS)}")
        staged, samples = image_key(entry)
        if staged != (tile in ("staged", "sample_major")) or (samples > 0) != tile.startswith("sample_"):
            raise ValueError(f"{name}: config {key!r} has cond_stage={int(staged)}, sample_major={samples}")
        parsed.setdefault(activation, {})[tile] = dict(entry)
    if not parsed:
        raise ValueError(f"{name}: a conditioned transition config names at least one activation")
    for activation, tiles in parsed.items():
        if "base" not in tiles:
            raise ValueError(f"{name}: {activation} has no base tile")
        if len({image_key(params)[1] for tile, params in tiles.items() if tile.startswith("sample_")}) > 1:
            raise ValueError(f"{name}: {activation}'s sample-major tiles must take the same samples")
    ordered = {
        activation: {tile: parsed[activation][tile] for tile in TILES if tile in parsed[activation]}
        for activation in ACTIVATIONS
        if activation in parsed
    }
    return kernel_abi, ordered


def configs(sm: int, width: int, hidden: int, activation: str = "silu_gate") -> dict[str, dict[str, int]]:
    """``activation``'s configs in the packaged bundle for ``(width, hidden)`` on ``sm``'s kernel ABI, by tile; none
    without one."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return {}
    name = config_file_name(sm, width, hidden)
    bundle = load_kernel_configs(str(CONFIGS_DIR), name)
    if bundle is None:
        return {}
    bundle_abi, parsed = validate_config(bundle.kernel_abi, bundle.kernel_variant, bundle.configs, name)
    return parsed.get(activation, {}) if bundle_abi == kernel_abi else {}


def shipped_shapes(sm: int) -> frozenset[tuple[int, int, str]]:
    """Every ``(width, hidden, activation)`` the packaged configs ship for ``sm``."""
    shapes = set()
    for path in sorted(CONFIGS_DIR.glob(f"CT*_H*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        width, hidden = int(match["width"]), int(match["hidden"])
        shapes.update(
            (width, hidden, activation) for activation in ACTIVATIONS if configs(sm, width, hidden, activation)
        )
    return frozenset(shapes)


def sample_atoms(samples: int) -> int:
    """Atoms per CTA in a sample-major tile: its ``samples`` rows each, in whole 8-row blocks within a CTA's tile."""
    return CTA_ROWS // samples // 8 * 8


def choose_config(rows: int, atoms: int, pairs: int, samples: int, tiles: Collection[str] = TILES) -> str:
    """The tile of ``tiles`` for a call of ``rows`` rows over ``atoms`` atoms on a GPU of ``pairs`` CTA pairs, whose
    sample-major tiles take ``samples`` samples."""
    count = -(-rows // (2 * CTA_ROWS))
    if samples and rows == atoms * samples:
        sample_tiles = -(-atoms // (2 * sample_atoms(samples)))
        # Staging the conditioning in shared memory pays off only once tiles fill over half the pairs.
        if "sample_major" in tiles and pairs < 2 * sample_tiles <= 2 * pairs:
            return "sample_major"
        # Sample-major tiles hold fewer rows than the CTA's 128: they win unless that adds a round of tiles.
        if "sample_stream" in tiles and -(-sample_tiles // pairs) <= -(-count // pairs):
            return "sample_stream"
    if "staged" in tiles and count <= pairs:
        return "staged"
    return "base"
