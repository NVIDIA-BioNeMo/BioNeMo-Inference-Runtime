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

"""Private helpers shared by Protenix production modules."""

from __future__ import annotations

# Reference-conformer + windowing keys consumed by the atom encoder / diffusion
# cache path. Order matches the atom encoder's keyword arguments.
ATOM_ENCODER_FEATURE_KEYS: tuple[str, ...] = (
    "atom_to_token_idx",
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_atom_name_chars",
    "ref_element",
    "d_lm",
    "v_lm",
    "pad_info",
)

DIFFUSION_CONSUMED_FEATURES: tuple[str, ...] = ("relp",) + tuple(
    k for k in ATOM_ENCODER_FEATURE_KEYS if k != "atom_to_token_idx"
)


def atom_encoder_kwargs(features: dict) -> dict:
    """Assemble atom-encoder keyword args from a feature dict (stable key set)."""
    return {k: features[k] for k in ATOM_ENCODER_FEATURE_KEYS}
