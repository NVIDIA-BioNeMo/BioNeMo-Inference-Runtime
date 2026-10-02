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
"""Token-padding specs: which tensors a model pads to an aligned token count.

Model configs carry them; ``bionemo_ir._torch.layers.token_padding`` applies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureDictPadSpec:
    """Which ``input_feature_dict`` keys need padding, and their shape kind.

    Every key's token dim(s) must equal the region's current (pre-pad) N for
    that key to be a true token-count feature; keys not listed, absent, or
    ``None`` are left untouched. Deliberately explicit per key rather than
    shape-sniffed: heuristically padding "any dim that happens to equal N"
    risks silently padding a dimension that isn't semantically a token count.
    """

    single_last: tuple[str, ...] = field(default_factory=tuple)  # [..., N], e.g. asym_id
    single_channel: tuple[str, ...] = field(default_factory=tuple)  # [..., N, C]
    pair_last: tuple[str, ...] = field(default_factory=tuple)  # [..., N, N], e.g. pseudo_beta_mask
    pair_channel: tuple[str, ...] = field(default_factory=tuple)  # [..., N, N, C], e.g. distogram
    single_matrix: tuple[str, ...] = field(default_factory=tuple)  # [..., N, R, C], e.g. frame rotations


@dataclass(frozen=True)
class TrunkPadSpec:
    """Which named tensors a padded region (a trunk, a confidence pairformer) pads, by shape kind."""

    single_channel: tuple[str, ...] = field(default_factory=tuple)  # [..., N, C], e.g. s, MSA m
    pair_channel: tuple[str, ...] = field(default_factory=tuple)  # [..., N, N, C]
    single_last: tuple[str, ...] = field(default_factory=tuple)  # [..., N] no channel, e.g. masks, msa/*
    pair_last: tuple[str, ...] = field(default_factory=tuple)  # [..., N, N] no channel, e.g. pair masks
    feature_dict: FeatureDictPadSpec | None = None
