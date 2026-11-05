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

On SM100 and SM103 an entry may also name the Blackwell kernels: ``sm100``,
the triangle attentions' 2-SM kernel, or ``sm100_token``, the token
transformers' kernel, whose variants ``SM100_TOKEN_TUNING`` holds because each
call picks its own. Their fixed cost per call loses to the SM80 kernel on few
rows, so the files' lower anchors name that one.
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
# Blackwell data-center SMs, whose files may also name their own kernels.
BLACKWELL_SMS = (100, 103)
BLACKWELL_ENTRY_KERNEL_ABIS = ("sm80", "sm100", "sm100_token")

# The CUBIN builder ships output-gate variants at OUTPUT_GATE_WIDTHS and
# STREAMED_WIDTHS, and residual-free variants at NO_RESIDUAL_WIDTHS, so all
# three live in this fingerprinted module.
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
# Token transformers that add the residual themselves, after a gate of their
# own, take an epilogue that writes the bare projection.
NO_RESIDUAL_WIDTHS = (768,)

# The ``sm100`` entries tune the triangle kernel's ``o``/``g`` ring slots and
# residual slots, ``og_stages`` and ``z_stages``. Swept on B200 at N = 1024 by
# CUDA-graph replay with L2 flushed: 3 and 3 at both widths; deeper rings ran
# 1-9% slower. B300 (SM103) keeps them: swept at every triangle call of the 17
# benchmark samples (N = 32-1736, starting and ending nodes, SM clock locked at
# 2032 MHz), they stay within 4% of the best depths at C = 128 and 0.4% at
# C = 256.
#
# The ``sm80`` entries tune the SM80 kernel for small calls, at anchors of
# N <= 128 per side, or 128 tokens of 5 samples at the token width. Every tile
# the kernel accepts (tile_m of 16 to 64) was timed on B200 against the native
# kernel by CUDA-graph replay of back-to-back calls, on the calls the layers
# make at each width: square triangle calls without the gate, N = 16 to 192 in
# steps of 8; at C = 128 also the atom transformers' gated calls of 5 samples
# by 16 to 7168 atoms; at the token width 5 samples of 8 to 192 tokens, and
# RFProteína's biased calls of 2 by 8 to 600 tokens. A call takes the nearest
# anchor, so each tile serves an interval of rows: the anchors, their tiles and
# the native kernel's anchor minimize the summed log time ratio to the native
# kernel over every call timed, with a cost per anchor, and leave none of them
# more than 2% slower. SM103 takes the same files, untuned on B300.
#
# Widths at which the triangle kernel also takes an output gate. Its logits take a slot beside each residual
# slot: 48 KB more at these tunings, which fits beside the 16 KB of Wo each CTA holds at C = 128 but not beside
# the 64 KB at C = 256.
SM100_OUTPUT_GATE_WIDTHS = (128,)
# The token kernel serves ``H*D = C = TOKEN_WIDTH`` with an output gate, on
# 128-row tiles whose gated chunks six CTAs of a cluster share. ``triples``
# runs 1-SM CTAs of 128 channels, three to a tile, for calls of up to one wave
# of tiles; ``pairs`` runs 2-SM pairs of 256 channels, two tiles each, for
# larger ones. Slots (gated ring, staging pool, Wo) swept on B200 at 373, 959
# and 1734 tokens: 6/4/4 best; 6/5/3, 6/6/2, 3/7/4 and 3/8/3 ran 1-17% slower.
# B300 keeps them unswept.
TOKEN_WIDTH = 768
_TOKEN_RINGS = {"a_stages": 6, "p_stages": 4, "w_stages": 4}
_SM100_TOKEN_VARIANTS = {
    "triples": {"pairs": False, "share": 3, **_TOKEN_RINGS},
    "pairs": {"pairs": True, **_TOKEN_RINGS},
}
SM100_TOKEN_TUNING = dict.fromkeys(BLACKWELL_SMS, _SM100_TOKEN_VARIANTS)


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
    configs: Mapping[str, Any],
    kernel_abi: str,
    kernel_variant: str | None,
    name: str,
    kernel_abis: Sequence[str] = ENTRY_KERNEL_ABIS,
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
        if entry_abi not in kernel_abis:
            raise ValueError(f"{name}: {key} names kernel_abi {entry_abi!r}; expected one of {tuple(kernel_abis)}")
        tunings.append(TunedConfig(int(match.group("rows")), entry_abi, entry_variant, tile_params))
    return sorted(tunings, key=lambda tuning: tuning.rows)


def tuned_configs(sm: int, heads: int, head_dim: int, channels: int) -> list[TunedConfig]:
    """Return one layer shape's tunings sorted by anchor, or none when it is untuned."""
    name = config_file_name(sm, heads, head_dim, channels)
    bundle = load_kernel_configs(str(CONFIGS_DIR), name)
    if bundle is None:
        return []
    kernel_abis = BLACKWELL_ENTRY_KERNEL_ABIS if sm in BLACKWELL_SMS else ENTRY_KERNEL_ABIS
    return parse_tunings(bundle.configs, bundle.kernel_abi, bundle.kernel_variant, name, kernel_abis)


def nearest_anchor(anchors: Sequence[int], rows: int) -> int:
    """Pick the anchor nearest ``rows``; ties choose the lower one."""
    return min(anchors, key=lambda anchor: (abs(anchor - rows), anchor))
