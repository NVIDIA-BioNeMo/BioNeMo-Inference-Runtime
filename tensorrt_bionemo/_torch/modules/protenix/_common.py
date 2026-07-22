# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private helpers shared by Protenix production modules."""

from __future__ import annotations

# Reference-conformer + windowing keys consumed by the atom encoder / diffusion
# cache path. Order matches the historical ``prepare_cache`` / encoder kwargs.
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

DIFFUSION_CONSUMED_FEATURES: tuple[str, ...] = ("relp", ) + tuple(
    k for k in ATOM_ENCODER_FEATURE_KEYS if k != "atom_to_token_idx")


def atom_encoder_kwargs(features: dict) -> dict:
    """Assemble atom-encoder keyword args from a feature dict (stable key set)."""
    return {k: features[k] for k in ATOM_ENCODER_FEATURE_KEYS}
