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
"""Validate and materialize RFProteína DNA/RNA model outputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from bionemo_ir._torch.layers.transformers.local_latents_decoder import get_residue_atom_mask

from .atom_utils import condition_pair, optional_feature, prepare_coordinate_condition
from .contracts import RFProteinaNucleicAcidOutput
from .representation_builder import (
    ATOM37_SIZE,
    CONDITION_COORDINATE,
    CONDITION_COORDINATE_MASK,
    CONDITION_SEQUENCE,
    CONDITION_SEQUENCE_MASK,
    UNSPECIFIED_RESIDUE_TOKEN,
)

RNA_TOKEN_TYPE = 1
DNA_TOKEN_TYPE = 2
RNA_RESIDUE_TYPE_RANGE = (22, 27)
DNA_RESIDUE_TYPE_RANGE = (27, 32)
UNIFIED_RESIDUE_TYPE_COUNT = 33
LEGACY_RESIDUE_TYPE_COUNT = 20


def _is_unified(decoder_num_residue_types: int) -> bool:
    """Whether a decoder carries the unified vocabulary, in its 33-class form or a wider one.

    A wider unified vocabulary extends the 33-class one without renumbering it, so RNA is
    22..26 and DNA 27..31 whatever the decoder's width. Classes past 32 (glycans, modified
    residues) are never valid for an RNA or DNA token: the decoder restricts each token kind to
    its own classes, and ``_validate_residue_kinds`` rejects anything outside the ranges above.
    """
    return decoder_num_residue_types >= UNIFIED_RESIDUE_TYPE_COUNT


def _nucleic_token_masks(
    static_features: Mapping[str, torch.Tensor],
    token_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validated padding/token-type tensors and RNA/DNA masks."""
    missing = [key for key in ("padding_mask", "token_type") if key not in static_features]
    if missing:
        raise ValueError(f"nucleic-acid tokens require static features: {missing}")
    padding_mask = static_features["padding_mask"]
    token_type = static_features["token_type"]
    for key, value in (("padding_mask", padding_mask), ("token_type", token_type)):
        if value.shape != token_shape:
            raise ValueError(f"{key} must have shape {token_shape}, got {tuple(value.shape)}")
    padding_mask = padding_mask.to(device=device, dtype=torch.bool)
    token_type = token_type.to(device=device, dtype=torch.long)
    rna = padding_mask & (token_type == RNA_TOKEN_TYPE)
    dna = padding_mask & (token_type == DNA_TOKEN_TYPE)
    return padding_mask, token_type, rna, dna


def _validate_residue_kinds(
    residue_types: torch.Tensor, rna_mask: torch.Tensor, dna_mask: torch.Tensor, name: str
) -> None:
    """Require unified residue IDs to agree with RNA/DNA token types."""
    rna_min, rna_max = RNA_RESIDUE_TYPE_RANGE
    dna_min, dna_max = DNA_RESIDUE_TYPE_RANGE
    invalid_rna = rna_mask & ((residue_types < rna_min) | (residue_types >= rna_max))
    invalid_dna = dna_mask & ((residue_types < dna_min) | (residue_types >= dna_max))
    invalid = invalid_rna | invalid_dna
    if torch.any(invalid):
        bad = torch.unique(residue_types[invalid]).detach().cpu().tolist()
        raise ValueError(f"{name} must use unified RNA IDs 22..26 and DNA IDs 27..31; got {bad}")


def _active_na_secondary_structure(static_features: Mapping[str, torch.Tensor]) -> bool:
    """Return whether NA-secondary-structure conditions are active."""
    for key in ("na_ss_state_mask", "na_ss_pair_mask"):
        value = static_features.get(key)
        if value is not None and bool(torch.any(value.to(torch.bool))):
            return True
    return False


def validate_nucleic_acid_contract(
    static_features: Mapping[str, torch.Tensor],
    can_be_encoded: torch.Tensor,
    decoder_num_residue_types: int,
    supports_na_secondary_structure: bool = False,
) -> None:
    """Fail early when a batch cannot produce a complete DNA/RNA output.

    A unified decoder, the 33-class one or a wider vocabulary that extends it,
    supports generated RNA/DNA. A legacy 20-class
    decoder can only carry fully coordinate- and sequence-conditioned NA
    context through a design: its final NA rows come entirely from the input
    conditions. Active NA-secondary-structure conditions require the latest
    representation input schema; legacy checkpoints reject them rather than
    silently changing the requested problem.
    """
    if decoder_num_residue_types != LEGACY_RESIDUE_TYPE_COUNT and not _is_unified(decoder_num_residue_types):
        raise ValueError(
            "RFProteína DNA/RNA support requires a 20-class decoder or a unified decoder of "
            f"{UNIFIED_RESIDUE_TYPE_COUNT} or more classes, got {decoder_num_residue_types}"
        )
    if can_be_encoded.ndim != 2:
        raise ValueError(f"can_be_encoded must have shape [B, N], got {tuple(can_be_encoded.shape)}")
    token_shape = tuple(can_be_encoded.shape)
    device = can_be_encoded.device
    _, _, rna, dna = _nucleic_token_masks(static_features, token_shape, device)
    nucleic = rna | dna
    if not torch.any(nucleic):
        return
    if not supports_na_secondary_structure and _active_na_secondary_structure(static_features):
        raise ValueError(
            "active na_ss_state/na_ss_pair conditions are not consumed by "
            "the released TRT-BioNeMo RFProteína checkpoint"
        )

    required = ("atom37_mask", "residue_type")
    missing = [key for key in required if key not in static_features]
    if missing:
        raise ValueError(f"nucleic-acid tokens require static features: {missing}")
    atom_mask = static_features["atom37_mask"]
    expected_atom_shape = (*token_shape, ATOM37_SIZE)
    if atom_mask.shape != expected_atom_shape:
        raise ValueError(f"atom37_mask must have shape {expected_atom_shape}, got {tuple(atom_mask.shape)}")
    atom_mask = atom_mask.to(device=device, dtype=torch.bool)
    input_residue_types = static_features["residue_type"]
    if input_residue_types.shape != token_shape:
        raise ValueError(f"residue_type must have shape {token_shape}, got {tuple(input_residue_types.shape)}")
    input_residue_types = input_residue_types.to(device=device, dtype=torch.long)
    _validate_residue_kinds(input_residue_types, rna, dna, "residue_type")

    _, coordinate_mask = condition_pair(
        static_features,
        CONDITION_COORDINATE,
        CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )
    _, sequence_mask = condition_pair(
        static_features,
        CONDITION_SEQUENCE,
        CONDITION_SEQUENCE_MASK,
        unspecified_token=UNSPECIFIED_RESIDUE_TOKEN,
    )
    if coordinate_mask is None:
        coordinate_mask = torch.zeros_like(atom_mask)
    elif coordinate_mask.shape != expected_atom_shape:
        raise ValueError(
            f"{CONDITION_COORDINATE_MASK} must have shape {expected_atom_shape}, got {tuple(coordinate_mask.shape)}"
        )
    coordinate_mask = coordinate_mask.to(device=device, dtype=torch.bool) & atom_mask
    fully_coordinate_conditioned = (coordinate_mask | ~atom_mask).all(dim=-1)
    if sequence_mask is None:
        sequence_mask = torch.zeros(token_shape, dtype=torch.bool, device=device)
    elif sequence_mask.shape != token_shape:
        raise ValueError(f"{CONDITION_SEQUENCE_MASK} must have shape {token_shape}, got {tuple(sequence_mask.shape)}")
    sequence_mask = sequence_mask.to(device=device, dtype=torch.bool)

    if decoder_num_residue_types == LEGACY_RESIDUE_TYPE_COUNT:
        if torch.any(nucleic & can_be_encoded.to(device=device, dtype=torch.bool)):
            raise ValueError("DNA/RNA tokens must be non-encodable with the legacy 20-class protein decoder")
        unsupported = nucleic & (~fully_coordinate_conditioned | ~sequence_mask)
        if torch.any(unsupported):
            raise ValueError(
                "the legacy 20-class protein decoder only supports DNA/RNA "
                "tokens whose complete atom37 coordinates and sequence are "
                "conditioned; use a unified autoencoder for "
                "sequence-context, partial, or joint NA generation"
            )
    else:
        unavailable = nucleic & ~can_be_encoded.to(device=device, dtype=torch.bool)
        incomplete = unavailable & (~fully_coordinate_conditioned | ~sequence_mask)
        if torch.any(incomplete):
            raise ValueError(
                "non-encodable DNA/RNA tokens require complete coordinate "
                "and sequence conditions so a full output can be materialized"
            )


@torch.no_grad()
def materialize_nucleic_acid_output(
    coordinates: torch.Tensor,
    residue_types: torch.Tensor,
    atom_mask: torch.Tensor,
    static_features: Mapping[str, torch.Tensor],
    decoder_num_residue_types: int,
) -> RFProteinaNucleicAcidOutput | None:
    """Build a canonical DNA/RNA view and hard-restore fixed conditions."""
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (ATOM37_SIZE, 3):
        raise ValueError(f"coordinates must have shape [B, N, 37, 3], got {tuple(coordinates.shape)}")
    token_shape = tuple(coordinates.shape[:2])
    if residue_types.shape != token_shape:
        raise ValueError(f"residue_types must have shape {token_shape}, got {tuple(residue_types.shape)}")
    expected_atom_shape = (*token_shape, ATOM37_SIZE)
    if atom_mask.shape != expected_atom_shape:
        raise ValueError(f"atom_mask must have shape {expected_atom_shape}, got {tuple(atom_mask.shape)}")
    device = coordinates.device
    _, token_type, rna, dna = _nucleic_token_masks(static_features, token_shape, device)
    nucleic = rna | dna
    if not torch.any(nucleic):
        return None

    input_atom_mask = static_features["atom37_mask"]
    if input_atom_mask.shape != expected_atom_shape:
        raise ValueError(f"atom37_mask must have shape {expected_atom_shape}, got {tuple(input_atom_mask.shape)}")
    input_atom_mask = input_atom_mask.to(device=device, dtype=torch.bool) & nucleic.unsqueeze(-1)
    input_residue_types = static_features["residue_type"]
    if input_residue_types.shape != token_shape:
        raise ValueError(f"residue_type must have shape {token_shape}, got {tuple(input_residue_types.shape)}")
    input_residue_types = input_residue_types.to(device=device, dtype=torch.long)
    _validate_residue_kinds(input_residue_types, rna, dna, "residue_type")

    generated_atom_mask = atom_mask.to(device=device, dtype=torch.bool) & nucleic.unsqueeze(-1)
    generated_coordinates = torch.where(generated_atom_mask.unsqueeze(-1), coordinates, torch.zeros_like(coordinates))
    if _is_unified(decoder_num_residue_types):
        generated_residue_types = residue_types.to(device=device, dtype=torch.long)
        decoded = nucleic & (generated_residue_types >= 0)
        _validate_residue_kinds(generated_residue_types, rna & decoded, dna & decoded, "decoded residue_types")
    elif decoder_num_residue_types == LEGACY_RESIDUE_TYPE_COUNT:
        generated_residue_types = torch.full(token_shape, -1, device=device, dtype=torch.long)
        generated_atom_mask = torch.zeros_like(generated_atom_mask)
        generated_coordinates = torch.zeros_like(generated_coordinates)
    else:
        raise ValueError(
            f"decoder_num_residue_types must be {LEGACY_RESIDUE_TYPE_COUNT} or at least "
            f"{UNIFIED_RESIDUE_TYPE_COUNT}, got {decoder_num_residue_types}"
        )

    coordinate_values, coordinate_mask = condition_pair(
        static_features,
        CONDITION_COORDINATE,
        CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )
    if coordinate_mask is None:
        coordinate_mask = torch.zeros_like(input_atom_mask)
        coordinate_values_nm = torch.zeros_like(coordinates)
    else:
        coordinate_values_nm, coordinate_mask = prepare_coordinate_condition(
            coordinate_values,
            coordinate_mask,
            coordinates,
            intersect_mask=input_atom_mask,
            value_name=CONDITION_COORDINATE,
            mask_name=CONDITION_COORDINATE_MASK,
        )

    sequence_values, sequence_mask = condition_pair(
        static_features,
        CONDITION_SEQUENCE,
        CONDITION_SEQUENCE_MASK,
        unspecified_token=UNSPECIFIED_RESIDUE_TOKEN,
    )
    if sequence_mask is None:
        sequence_mask = torch.zeros(token_shape, device=device, dtype=torch.bool)
        sequence_values = torch.zeros(token_shape, device=device, dtype=torch.long)
    else:
        if sequence_values.shape != token_shape:
            raise ValueError(f"{CONDITION_SEQUENCE} must have shape {token_shape}, got {tuple(sequence_values.shape)}")
        if sequence_mask.shape != token_shape:
            raise ValueError(
                f"{CONDITION_SEQUENCE_MASK} must have shape {token_shape}, got {tuple(sequence_mask.shape)}"
            )
        sequence_values = sequence_values.to(device=device, dtype=torch.long)
        sequence_mask = sequence_mask.to(device=device, dtype=torch.bool) & nucleic
        _validate_residue_kinds(sequence_values, rna & sequence_mask, dna & sequence_mask, CONDITION_SEQUENCE)

    if decoder_num_residue_types == LEGACY_RESIDUE_TYPE_COUNT:
        fully_coordinate_conditioned = (coordinate_mask | ~input_atom_mask).all(dim=-1)
        unsupported = nucleic & (~fully_coordinate_conditioned | ~sequence_mask)
        if torch.any(unsupported):
            raise ValueError(
                "the legacy 20-class protein decoder needs complete DNA/RNA coordinate and sequence conditions"
            )

    final_residue_types = torch.where(sequence_mask, sequence_values, generated_residue_types)
    unavailable = nucleic & (final_residue_types < 0)
    if torch.any(unavailable):
        raise ValueError(
            "DNA/RNA output has no residue identity; provide sequence conditions or use a unified autoencoder"
        )
    _validate_residue_kinds(final_residue_types, rna, dna, "final residue_types")
    # RNA and DNA classes sit in the 33-class prefix every unified vocabulary shares, and their
    # occupancy rows are identical in a wider vocabulary, so this table serves any unified decoder.
    final_atom_mask = get_residue_atom_mask(
        final_residue_types,
        UNIFIED_RESIDUE_TYPE_COUNT,
        nucleic,
    )
    # Preserve any resolved conditioned slots beyond the predicted canonical
    # layout (for example, terminal atoms on an unknown input residue).
    final_atom_mask |= coordinate_mask
    if decoder_num_residue_types == LEGACY_RESIDUE_TYPE_COUNT:
        final_atom_mask = input_atom_mask

    final_coordinates = torch.where(coordinate_mask.unsqueeze(-1), coordinate_values_nm, coordinates)
    final_coordinates = torch.where(
        final_atom_mask.unsqueeze(-1), final_coordinates, torch.zeros_like(final_coordinates)
    )
    final_coordinates = torch.where(
        nucleic.unsqueeze(-1).unsqueeze(-1), final_coordinates, torch.zeros_like(final_coordinates)
    )
    return RFProteinaNucleicAcidOutput(
        token_mask=nucleic,
        rna_token_mask=rna,
        dna_token_mask=dna,
        generated_coordinates=generated_coordinates,
        coordinates=final_coordinates,
        generated_atom_mask=generated_atom_mask,
        atom_mask=final_atom_mask,
        coordinate_condition_mask=coordinate_mask,
        generated_residue_types=generated_residue_types,
        residue_types=final_residue_types,
        sequence_condition_mask=sequence_mask,
        token_type=token_type,
        input_residue_types=input_residue_types,
        chain_iid=optional_feature(static_features, "chain_iid", token_shape, device),
        molecule_iid=optional_feature(static_features, "molecule_iid", token_shape, device),
    )
