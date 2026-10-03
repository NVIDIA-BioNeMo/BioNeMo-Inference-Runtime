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
"""Tuning configs for the fused attention epilogue.

One file per ``(heads, head_dim, channels, sm)``, named
``H{heads}_D{head_dim}_C{channels}_sm{sm}.json``, holding one entry of kernel
tile parameters per ``R=<rows>`` anchor. A call takes the entry whose anchor
is nearest its folded rows ``B*I*J``, the lower one on a tie. An entry may
name its own ``kernel_abi`` and ``kernel_variant``: the kernel that serves few
rows best is not always the one that serves many.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
ANCHOR_KEY_RE = re.compile(r"^R=(?P<rows>\d+)$")
ENTRY_KERNEL_ABIS = ("sm80", "sm90")

# The CUBIN builder ships output-gate variants at OUTPUT_GATE_WIDTHS and
# STREAMED_WIDTHS, so both live in this fingerprinted module.
#
# The output gate's tile doubles the residual stages, which leaves no room
# beside the 128 KB resident weight of H*D == 512.
OUTPUT_GATE_WIDTHS = (128, 256)
# Layers projecting to more than 128 channels, such as Protenix's 256-channel
# triangle attention and the diffusion token transformers, take the streamed
# SM90 kernel, or for few rows the channel-tiled SM80 one, with any head split
# of these H*D. SM80, SM86 and SM89 take the channel-tiled kernel at every
# size.
STREAMED_WIDTHS = (256, 768)


@dataclass(frozen=True)
class TunedConfig:
    """One anchor's tuning: the kernel it names and that kernel's tile parameters."""

    rows: int
    kernel_abi: str
    kernel_variant: str | None
    tile_params: dict[str, Any]


def config_file_name(sm: int, heads: int, head_dim: int, channels: int) -> str:
    """Return the tuning file name for one layer shape on one SM."""
    return get_config_file_name(sm, H=heads, D=head_dim, C=channels)


def parse_tunings(
    configs: Mapping[str, Any], kernel_abi: str, kernel_variant: str | None, name: str
) -> list[TunedConfig]:
    """Resolve a tuning file's ``R=<rows>`` entries against its kernel defaults, sorted by anchor."""
    tunings = []
    for key, entry in configs.items():
        match = ANCHOR_KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"{name}: invalid attention epilogue config key {key!r}; expected R=<rows>")
        tile_params = dict(entry)
        entry_abi = tile_params.pop("kernel_abi", kernel_abi)
        entry_variant = tile_params.pop("kernel_variant", kernel_variant)
        if entry_abi not in ENTRY_KERNEL_ABIS:
            raise ValueError(f"{name}: {key} names kernel_abi {entry_abi!r}; expected one of {ENTRY_KERNEL_ABIS}")
        tunings.append(TunedConfig(int(match.group("rows")), entry_abi, entry_variant, tile_params))
    return sorted(tunings, key=lambda tuning: tuning.rows)


def tuned_configs(sm: int, heads: int, head_dim: int, channels: int) -> list[TunedConfig]:
    """Return one layer shape's tunings sorted by anchor, or none when it is untuned."""
    name = config_file_name(sm, heads, head_dim, channels)
    bundle = load_kernel_configs(str(CONFIGS_DIR), name)
    if bundle is None:
        return []
    return parse_tunings(bundle.configs, bundle.kernel_abi, bundle.kernel_variant, name)


def nearest_anchor(anchors: Sequence[int], rows: int) -> int:
    """Pick the anchor nearest ``rows``; ties choose the lower one."""
    return min(anchors, key=lambda anchor: (abs(anchor - rows), anchor))
