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
"""Shipped and candidate shapes of the SM90 transition with its LayerNorm.

One ``configs/C{width}_H{hidden}_sm90.json`` per ``(width, hidden)``, with one entry per activation the kernel serves
at that shape. A shape ships only where the fused kernel beats the LayerNorm pass and ``transition_mlp`` it replaces,
so an entry is the shipping decision: an activation without one is not served. An entry marked as a candidate is
built into the CUBIN library and runs when a caller asks for candidates, but it does not ship until a measurement
shows it wins. The kernel derives its tiling from the shape, so an entry holds no tile parameters::

    {"kernel_abi": "sm90", "configs": {"act=relu": {}, "act=silu_gate": {"candidate": true}}}

An entry may list launch layouts for calls too small to fill the GPU, each the largest call it takes in waves (the
call's row tiles of the full tile, over the device's SM count) and the layout it launches: ``tile_wgs`` warpgroups per
row tile and ``split`` CTAs per tile. A call takes the first layout whose ``waves`` it does not exceed, and the full
tile with split 1 past the last::

    "act=relu": {"layouts": [{"waves": 0.07, "tile_wgs": 1, "split": 4}, {"waves": 0.25, "tile_wgs": 1, "split": 2}]}

The file names carry ``C`` rather than ``transition_mlp``'s ``W``, so the two families' files can share one
``BIOIR_TUNED_CONFIG_FOLDER``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).with_name("configs")
# W1 row blocks per activation, as the kernel counts them: the ReLU input; the SwiGLU value and gate.
W1_BLOCKS = {"relu": 1, "silu_gate": 2}
#: The hidden activations: ReLU, and SwiGLU ``silu(gate) * value`` with W1's first ``hidden`` rows making ``value`` and
#: the next ``hidden`` rows ``gate``.
ACTIVATIONS = tuple(W1_BLOCKS)
KERNEL_ABIS = {90: "sm90"}
_FILE_RE = re.compile(r"^C(?P<width>\d+)_H(?P<hidden>\d+)_sm(?P<sm>\d+)\.json$")
_KEY_RE = re.compile(r"^act=(?P<activation>relu|silu_gate)$")
# An entry's keys: whether it is a candidate, and its launch layouts for small calls.
_ENTRY_KEYS = frozenset({"candidate", "layouts"})
_LAYOUT_KEYS = frozenset({"waves", "tile_wgs", "split"})
# The warpgroups a row tile may take, and the most CTAs a cluster may: the portable cluster size.
_TILE_WGS = (1, 2, 4)
_MAX_SPLIT = 8


def config_file_name(sm: int, width: int, hidden: int) -> str:
    """The config file name for one ``(width, hidden)`` on one SM."""
    return get_config_file_name(sm, C=width, H=hidden)


def config_key(activation: str) -> str:
    """The ``configs`` entry that ships ``activation``; the inverse of :func:`parse_config_key`."""
    if activation not in W1_BLOCKS:
        raise ValueError(f"the fused transition has no {activation!r} activation; expected one of {ACTIVATIONS}")
    return f"act={activation}"


def parse_config_key(key: str) -> str:
    """The activation one ``configs`` key ships."""
    match = _KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid fused transition config key {key!r}; expected act=relu or act=silu_gate")
    return match["activation"]


def image_elements(width: int, hidden: int, activation: str) -> int:
    """bf16 elements of the packed weight image: every W1 block, then W2."""
    return (W1_BLOCKS[activation] + 1) * width * hidden


def vec_elements(width: int, hidden: int, activation: str) -> int:
    """fp32 elements of the packed bias vector: one per W1 row, then one per output channel."""
    return W1_BLOCKS[activation] * hidden + width


def layout_bit(tile_wgs: int, split: int) -> int:
    """The bit of ``(tile_wgs, split)`` in a kernel's layout mask (the launcher's ``KernelSpec.layouts``)."""
    if tile_wgs not in _TILE_WGS or not 1 <= split <= _MAX_SPLIT:
        raise ValueError(f"no fused transition layout tile_wgs={tile_wgs}, split={split}")
    return _TILE_WGS.index(tile_wgs) * _MAX_SPLIT + split - 1


def layout_mask(pairs: Iterable[tuple[int, int]]) -> int:
    """The mask of ``(tile_wgs, split)`` pairs a kernel launches with (its ``layouts``)."""
    return sum(1 << layout_bit(tile_wgs, split) for tile_wgs, split in set(pairs))


def choose_layout(
    rows: int,
    tile_m: int,
    num_wg: int,
    num_sms: int,
    layouts: Iterable[tuple[float, int, int]],
    allowed: int,
) -> tuple[int, int]:
    """``(tile_wgs, split)`` for a call of ``rows`` rows: the first of ``layouts`` whose waves the call does not
    exceed, counting its full row tiles (``tile_m`` rows, ``num_wg`` warpgroups) over ``num_sms`` SMs, and that the
    kernel's ``allowed`` mask (:func:`layout_mask`) holds; the full tile with split 1 otherwise."""
    waves = -(-rows // tile_m) / num_sms
    for max_waves, tile_wgs, split in layouts:
        if waves <= max_waves:
            if tile_wgs <= num_wg and allowed >> layout_bit(tile_wgs, split) & 1:
                return tile_wgs, split
            break
    return num_wg, 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_layouts(entry: Mapping[str, Any], where: str) -> tuple[tuple[float, int, int], ...]:
    """An entry's launch layouts as ``(waves, tile_wgs, split)``, ascending in ``waves``; raise if malformed."""
    layouts = entry.get("layouts", [])
    if not isinstance(layouts, list):
        raise ValueError(f"{where}: layouts must be a list")
    parsed = []
    for layout in layouts:
        if not isinstance(layout, Mapping) or set(layout) != _LAYOUT_KEYS:
            raise ValueError(f"{where}: a layout holds exactly {sorted(_LAYOUT_KEYS)}, got {layout!r}")
        waves, tile_wgs, split = layout["waves"], layout["tile_wgs"], layout["split"]
        if not isinstance(waves, int | float) or isinstance(waves, bool) or not waves > 0:
            raise ValueError(f"{where}: a layout's waves must be a positive number, got {waves!r}")
        if not _is_int(tile_wgs) or tile_wgs not in _TILE_WGS:
            raise ValueError(f"{where}: a layout's tile_wgs must be one of {_TILE_WGS}, got {tile_wgs!r}")
        if not _is_int(split) or not 1 <= split <= _MAX_SPLIT:
            raise ValueError(f"{where}: a layout's split must be 1 to {_MAX_SPLIT}, got {split!r}")
        if parsed and waves <= parsed[-1][0]:
            raise ValueError(f"{where}: layouts must be in ascending waves")
        parsed.append((float(waves), tile_wgs, split))
    return tuple(parsed)


def validate_config(
    kernel_abi: str, kernel_variant: str | None, configs: Mapping[str, Any], name: str
) -> tuple[str, dict[str, bool]]:
    """Return a config file's kernel ABI and, for each activation it names, whether that entry is a candidate; raise
    if the file names anything else or a malformed layout."""
    if kernel_abi not in KERNEL_ABIS.values():
        raise ValueError(
            f"{name}: the fused transition has no {kernel_abi!r} kernel; expected one of {sorted(KERNEL_ABIS.values())}"
        )
    if kernel_variant is not None:
        raise ValueError(f"{name}: the fused transition has no kernel variants")
    if not configs:
        raise ValueError(f"{name}: a fused transition config must name at least one activation")
    entries = {}
    for key, entry in configs.items():
        if not isinstance(entry, Mapping) or not set(entry) <= _ENTRY_KEYS or entry.get("candidate", True) is not True:
            raise ValueError(
                f"{name}: the fused transition's {key!r} entry takes no tile parameters: it holds at most "
                '"candidate": true and "layouts"'
            )
        parse_layouts(entry, f"{name} {key}")
        entries[parse_config_key(key)] = "candidate" in entry
    return kernel_abi, dict(sorted(entries.items()))


def _bundle(sm: int, width: int, hidden: int) -> Any:
    """The packaged config bundle for ``(width, hidden)`` on ``sm``'s kernel ABI, validated, or None."""
    kernel_abi = KERNEL_ABIS.get(sm)
    if kernel_abi is None:
        return None
    name = config_file_name(sm, width, hidden)
    bundle = load_kernel_configs(str(CONFIGS_DIR), name)
    if bundle is None:
        return None
    bundle_abi, _ = validate_config(bundle.kernel_abi, bundle.kernel_variant, bundle.configs, name)
    return bundle if bundle_abi == kernel_abi else None


def _entries(sm: int, width: int, hidden: int) -> dict[str, bool]:
    """Per activation the packaged config for ``(width, hidden)`` names on ``sm``'s kernel ABI, whether its entry is a
    candidate."""
    bundle = _bundle(sm, width, hidden)
    if bundle is None:
        return {}
    return validate_config(bundle.kernel_abi, bundle.kernel_variant, bundle.configs, "")[1]


def layouts(sm: int, width: int, hidden: int, activation: str = "relu") -> tuple[tuple[float, int, int], ...]:
    """The launch layouts ``(waves, tile_wgs, split)`` the config entry for ``(width, hidden, activation)`` lists for
    ``sm``, ascending in waves; none without an entry."""
    bundle = _bundle(sm, width, hidden)
    entry = None if bundle is None else bundle.configs.get(config_key(activation))
    return () if entry is None else parse_layouts(entry, config_file_name(sm, width, hidden))


def ships(sm: int, width: int, hidden: int, activation: str = "relu", *, candidates: bool = False) -> bool:
    """Whether a config entry ships ``(width, hidden, activation)`` for ``sm``'s kernel ABI; with ``candidates``, a
    candidate entry counts too."""
    entries = _entries(sm, width, hidden)
    return activation in entries and (candidates or not entries[activation])


def _shapes(sm: int, candidate: bool) -> frozenset[tuple[int, int, str]]:
    if sm not in KERNEL_ABIS:
        return frozenset()
    shapes = set()
    for path in sorted(CONFIGS_DIR.glob(f"C*_H*_sm{sm}.json")):
        match = _FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        width, hidden = int(match["width"]), int(match["hidden"])
        entries = _entries(sm, width, hidden)
        shapes.update(
            (width, hidden, activation) for activation, is_candidate in entries.items() if is_candidate == candidate
        )
    return frozenset(shapes)


def shipped_shapes(sm: int) -> frozenset[tuple[int, int, str]]:
    """Every ``(width, hidden, activation)`` the packaged configs ship for ``sm``."""
    return _shapes(sm, candidate=False)


def candidate_shapes(sm: int) -> frozenset[tuple[int, int, str]]:
    """Every ``(width, hidden, activation)`` the packaged configs build for ``sm`` but do not ship yet."""
    return _shapes(sm, candidate=True)
