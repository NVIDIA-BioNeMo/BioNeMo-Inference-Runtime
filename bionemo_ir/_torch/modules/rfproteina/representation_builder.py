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
"""Native tensor feature construction for RFProteína's velocity network."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.custom_ops import get_adaln_layernorm_sigmoid_op
from bionemo_ir._torch.sampling.random import randperm
from bionemo_ir.utils import str_dtype_to_torch

from .atom_utils import angstrom_to_nanometer, finite_atom_mask
from .chirality import CHIRAL_FRAME_ATOMS, CHIRAL_SLOT_OWNER, CHIRALITY, align_chirality
from .contracts import RFProteinaRepresentations
from .groups import CONDITION_COORDINATE_GROUP

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaRepresentationConfig, RFProteinaTrunkConfig

# Atomworks/RFProteína condition names are part of the serialized tensor-batch
# contract. Keeping the strings here avoids importing either package.
CONDITION_SEQUENCE = "condition_sequence_1_residue"
CONDITION_SEQUENCE_MASK = "mask_sequence_1_residue"
# Unified encoding stores unspecified residues as the mask token and omits
# mask_sequence_1_residue.
UNSPECIFIED_RESIDUE_TOKEN = 0
CONDITION_COORDINATE = "condition_coordinate_1_atom"
CONDITION_COORDINATE_MASK = "mask_coordinate_1_atom"
CONDITION_INTERNAL_COORDINATE = "condition_InternalCoordinate_1_atom"
CONDITION_INTERNAL_COORDINATE_MASK = "mask_InternalCoordinate_1_atom"
CONDITION_RASA = "condition_rasa-binned_1_atom"
CONDITION_SECONDARY_STRUCTURE = "condition_secondary-structure_1_residue"
CONDITION_HBOND = "condition_hbond_1_atom"
CONDITION_PISTACK = "mask_pistack_1_atom"
CONDITION_PISTACK_SPECIFIED = "condition_pistack_1_atom"
CONDITION_PICATION = "condition_pication_1_atom"
CONDITION_SALT_BRIDGE = "condition_saltbridge_1_atom"
CONDITION_HYDROPHOBIC = "condition_hydrophobic_1_atom"
CONDITION_METAL_COORDINATION = "condition_metal-coord_1_atom"
CONDITION_NA_SS_STATE = "na_ss_state"
CONDITION_NA_SS_STATE_MASK = "na_ss_state_mask"
CONDITION_HOTSPOT = "mask_hotspot_1_atom"
CONDITION_HOTSPOT_SPECIFIED = "condition_hotspot_1_atom"
CONDITION_NOT_CYSTEINE = "mask_not-cysteine_1_residue"
CONDITION_CHAIN_CENTROID = "condition_ChainCentroid_1_atom"
CONDITION_CATH = "condition_cath-label_1_residue"
CONDITION_INDEX_GROUP = "condition_index-group_1_token"
NA_SS_PAIR = "na_ss_pair"
NA_SS_PAIR_MASK = "na_ss_pair_mask"

ATOM37_SIZE = 37
COORDINATE_GROUP_UNKNOWN = -1
# RFProteina's C_CRDGRP sentinels: -1 is unconditioned and 0 is absolute
# coordinate conditioning, so a positive group marks internal-coordinate
# conditioning (rfproteina/conditions.py).
COORDINATE_GROUP_ABSOLUTE = 0
TIME_EMBEDDING_MAX_POSITIONS = 2000
RECYCLE_EMBEDDING_MAX_POSITIONS = 100
# Unified standard-polymer tokens, excluding unknown, mask, atomized, and gap.
SEQUENCE_PROBABILITY_TOKENS = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "A",
    "C",
    "G",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
)


class _PairAdaptiveLayerNorm(nn.Module):
    """Upstream-compatible pair AdaLN used by representation construction.

    The condition holds one row per design, so its gamma and beta are projected once per design rather than per
    pair. One fused AdaLN kernel normalizes every pair row and applies them; pairs outside the mask are zeroed.
    """

    def __init__(self, pair_dim: int, condition_dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.norm_cond = nn.RMSNorm(condition_dim, eps=eps, dtype=dtype)
        # The kernel applies the sigmoid; the module keeps it for the checkpoint layout.
        self.to_gamma = nn.Sequential(nn.Linear(condition_dim, pair_dim, dtype=dtype), nn.Sigmoid())
        self.to_beta = nn.Linear(condition_dim, pair_dim, bias=False, dtype=dtype)
        self._adaln = get_adaln_layernorm_sigmoid_op(dtype, pair_dim, rms_norm=True)

    def forward(self, pair: torch.Tensor, condition: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Condition every pair on its design.

        Args:
            pair: ``[B, N, N, pair_dim]`` pair features in the module dtype.
            condition: ``[B, condition_dim]`` condition, one row per design, in the module dtype.
            pair_mask: ``[B, N, N]`` valid pairs.

        Returns:
            ``[B, N, N, pair_dim]`` ``sigmoid(gamma) * RMSNorm(pair) + beta``, zero outside ``pair_mask``.
        """
        condition = self.norm_cond(condition)
        rows = pair.reshape(pair.shape[0], -1, pair.shape[-1])
        output = self._adaln(
            rows,
            self.to_gamma[0](condition).unsqueeze(1),
            self.to_beta(condition).unsqueeze(1),
            out=torch.empty_like(rows),
            eps=self.eps,
        )
        return output.view_as(pair).mul_(pair_mask.unsqueeze(-1).to(output.dtype))


def _time_embedding(
    t: torch.Tensor, embedding_dim: int, max_positions: int = TIME_EMBEDDING_MAX_POSITIONS
) -> torch.Tensor:
    """Sinusoidal diffusion-time embedding used by the released model."""
    if t.ndim != 1:
        raise ValueError(f"Expected time tensor [B], got {tuple(t.shape)}")
    half_dim = embedding_dim // 2
    exponent = math.log(max_positions) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -exponent)
    phase = (t * max_positions).float().unsqueeze(-1)
    embedding = torch.cat([torch.sin(phase * frequencies), torch.cos(phase * frequencies)], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _active_scale(active: bool | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(active, device=reference.device, dtype=reference.dtype)
    while scale.ndim < reference.ndim:
        scale = scale.unsqueeze(-1)
    return scale


class RFProteinaRepresentationBuilder(nn.Module):
    """Build released-checkpoint single, pair, and condition representations.

    ``static_features`` is an already-tensorized dictionary. Motif conditioning
    is ``condition_coordinate_1_atom [B, N, 37, 3]`` in angstroms. Finite values
    mark the conditioned atoms; ``mask_coordinate_1_atom [B, N, 37]`` overrides
    that selection when the batch carries it. Internal-coordinate values and
    masks use the same atom37 layout, sequence values and masks are ``[B, N]``,
    and ``condition_index-group_1_token`` is ``[B, N]``. Only tensor math lives
    here; parsing, topology construction, tokenization, and annotation
    generation remain outside the model package.
    """

    def __init__(self, config: RFProteinaRepresentationConfig, trunk_config: RFProteinaTrunkConfig) -> None:
        super().__init__()
        self.config = config
        # Feature construction stays in the configured (normally FP32) dtype.
        # Only the terminal pair AdaLN has its own reduced-precision boundary;
        # outputs are cast once for the velocity transformer.
        self.dtype = config.torch_dtype
        self.pair_adaln_dtype = str_dtype_to_torch(config.pair_adaln_dtype)
        self.output_dtype = trunk_config.torch_dtype
        self.latent_dim = trunk_config.latent_dim
        self.token_dim = trunk_config.token_dim
        self.pair_dim = trunk_config.pair_dim
        self.condition_dim = trunk_config.condition_dim

        if config.latent_dim != self.latent_dim:
            raise ValueError(
                "RFProteína representation/trunk latent dimensions must "
                f"match, got {config.latent_dim} and {self.latent_dim}"
            )

        self.token_type_embedding = nn.Embedding(
            config.num_token_kinds, config.token_type_embedding_dim, dtype=self.dtype
        )
        # The sequence-probability release replaces this frozen one-hot table
        # with explicit probability and mask channels, which have no parameters.
        self.conditional_residue_embedding = None
        if config.input_schema != "sequence_probability":
            self.conditional_residue_embedding = nn.Embedding(
                config.num_conditional_residue_types, config.num_conditional_residue_types, dtype=self.dtype
            )
            self.conditional_residue_embedding.weight.requires_grad_(False)
        self.secondary_structure_embedding = nn.Embedding(
            4, config.secondary_structure_embedding_dim, padding_idx=0, dtype=self.dtype
        )
        self.charge_embedding = nn.Embedding(config.num_charge_classes, config.charge_embedding_dim, dtype=self.dtype)
        self.element_embedding = nn.Embedding(config.max_atomic_number, config.element_embedding_dim, dtype=self.dtype)
        self.single_projection = nn.Linear(config.single_input_dim, self.token_dim, bias=False, dtype=self.dtype)

        self.cath_class_embedding = nn.Embedding(6, config.cath_embedding_dim, dtype=self.dtype)
        self.cath_architecture_embedding = nn.Embedding(44, config.cath_embedding_dim, dtype=self.dtype)
        self.cath_topology_embedding = nn.Embedding(1473, config.cath_embedding_dim, dtype=self.dtype)
        self.condition_projection = nn.Linear(
            config.condition_input_dim, self.condition_dim, bias=False, dtype=self.dtype
        )

        self.pair_projection = nn.Linear(config.pair_input_dim, self.pair_dim, bias=False, dtype=self.dtype)
        if config.factory_norm == "rms_norm":
            self.pair_norm = nn.RMSNorm(self.pair_dim, eps=config.norm_epsilon, dtype=self.dtype)
            self.pair_condition_norm = nn.RMSNorm(self.condition_dim, eps=config.norm_epsilon, dtype=self.dtype)
        elif config.factory_norm == "layer_norm":
            self.pair_norm = nn.LayerNorm(self.pair_dim, eps=config.norm_epsilon, dtype=self.dtype)
            self.pair_condition_norm = nn.LayerNorm(self.condition_dim, eps=config.norm_epsilon, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported representation factory norm: {config.factory_norm}")
        self.pair_condition_projection = nn.Linear(
            config.pair_condition_input_dim, self.condition_dim, bias=False, dtype=self.dtype
        )
        self.pair_adaln = _PairAdaptiveLayerNorm(
            self.pair_dim,
            self.condition_dim,
            eps=config.norm_epsilon,
            dtype=self.pair_adaln_dtype,
        )

    def _zeros(self, state: Mapping[str, torch.Tensor], trailing_shape: tuple[int, ...]) -> torch.Tensor:
        batch, tokens = state["bb_ca"].shape[:2]
        return torch.zeros((batch, tokens, *trailing_shape), device=state["bb_ca"].device, dtype=self.dtype)

    def _conditional_coordinates(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        coordinate_key: str,
        mask_key: str,
        *,
        finite_marks_conditioned: bool = False,
        group_marks_conditioned: bool = False,
    ) -> torch.Tensor:
        """Atom37 coordinates in nanometres, flattened, with a mask channel.

        Args:
            static_features: Tensorized batch. Coordinates are angstroms.
            state: Product-space state, used only for the zero-feature shape.
            coordinate_key: ``[B, N, 37, 3]`` coordinate tensor.
            mask_key: Optional ``[B, N, 37]`` bool mask. When present it selects
                the conditioned atoms.
            finite_marks_conditioned: When the mask is absent, treat an atom as
                conditioned if any coordinate component is finite. Absolute
                motif coordinates use this. Internal coordinates do not --
                an internal-coordinate payload may be finite everywhere, so
                finiteness does not discriminate.
            group_marks_conditioned: When the mask is absent, read the selection
                off ``condition_CoordinateGroup_1_atom``: a positive group marks
                internal-coordinate conditioning. Internal coordinates use this,
                because upstream never emits ``mask_InternalCoordinate_1_atom``
                -- it carries the selection in the coordinate group instead
                (``ConditionalAtom37NanometersInternalCoorsSeqFeat`` in
                rfproteina's feature factory). Without it a batch that conditions
                only on internal coordinates silently contributes 148 zero
                channels and the condition is lost.

        Returns:
            ``[B, N, 148]``: 37 xyz vectors followed by the 37 mask values.
        """
        if coordinate_key not in static_features:
            return self._zeros(state, (ATOM37_SIZE * 4,))
        raw = static_features[coordinate_key]
        if mask_key in static_features:
            mask = static_features[mask_key].to(torch.bool)
        elif finite_marks_conditioned:
            mask = finite_atom_mask(raw)
        elif group_marks_conditioned and CONDITION_COORDINATE_GROUP in static_features:
            groups = static_features[CONDITION_COORDINATE_GROUP].long()
            mask = (groups != COORDINATE_GROUP_UNKNOWN) & (groups != COORDINATE_GROUP_ABSOLUTE)
        else:
            return self._zeros(state, (ATOM37_SIZE * 4,))
        coordinates = angstrom_to_nanometer(torch.nan_to_num(raw, nan=0.0))
        coordinates = coordinates * mask.unsqueeze(-1)
        return torch.cat([coordinates.flatten(-2), mask.to(coordinates.dtype)], dim=-1)

    def _conditional_residue_type(
        self, static_features: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if CONDITION_SEQUENCE not in static_features or CONDITION_SEQUENCE_MASK not in static_features:
            return self._zeros(state, (self.config.num_conditional_residue_types,))
        mask = static_features[CONDITION_SEQUENCE_MASK].to(torch.bool)
        residue_type = static_features[CONDITION_SEQUENCE].long() * mask
        return self.conditional_residue_embedding(residue_type) * mask.unsqueeze(-1)

    def _atom_annotation(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        if key not in static_features:
            return self._zeros(state, (ATOM37_SIZE,))
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        return static_features[key].to(self.dtype) * atom_mask

    def _atom_role_annotation(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        """Encode NONE/FIRST/SECOND/BOTH atom roles as two binary channels."""
        if key not in static_features:
            return self._zeros(state, (2 * ATOM37_SIZE,))
        values = static_features[key].long()
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        first = ((values == 1) | (values == 3)) & atom_mask
        second = ((values == 2) | (values == 3)) & atom_mask
        return torch.cat((first, second), dim=-1).to(self.dtype)

    def _na_state(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if CONDITION_NA_SS_STATE not in static_features or CONDITION_NA_SS_STATE_MASK not in static_features:
            return self._zeros(state, (2,))
        value = static_features[CONDITION_NA_SS_STATE].to(self.dtype).unsqueeze(-1)
        mask = static_features[CONDITION_NA_SS_STATE_MASK].to(self.dtype).unsqueeze(-1)
        return torch.cat((value, mask), dim=-1)

    @property
    def conditions_per_recycle_pass(self) -> bool:
        """Whether each trunk pass gets its own condition track."""
        return self.config.recycle_feature is not None

    def _sequence_probabilities(
        self, static_features: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Probability value and explicit-mask channels in unified token order."""
        del state
        channels = []
        missing = []
        for token in SEQUENCE_PROBABILITY_TOKENS:
            key = f"condition_sequence-probability-{token}_1_residue"
            value = static_features.get(key)
            if value is None:
                missing.append(key)
            else:
                channels.append(value)
        if missing:
            preview = ", ".join(missing[:3])
            if len(missing) > 3:
                preview = f"{preview}, ..."
            raise KeyError(f"sequence_probabilities requires {preview}")
        probabilities = torch.stack(channels, dim=-1).to(self.dtype)
        specified = ~torch.isnan(probabilities)
        return torch.cat((probabilities.masked_fill(~specified, 0), specified.to(self.dtype)), dim=-1)

    def _rasa_binned(
        self, static_features: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Binned RASA; atoms outside the atom37 mask are unspecified (0)."""
        if CONDITION_RASA not in static_features:
            return self._zeros(state, (ATOM37_SIZE,))
        rasa = static_features[CONDITION_RASA].to(self.dtype)
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        return rasa.masked_fill(~atom_mask, 0)

    def _masked_atom_values(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        key: str,
        *,
        nonzero: bool,
    ) -> torch.Tensor:
        """Atom37 contact flags, zero on atoms outside the atom37 mask."""
        if key not in static_features:
            return self._zeros(state, (ATOM37_SIZE,))
        values = static_features[key]
        values = (values != 0).to(self.dtype) if nonzero else values.to(self.dtype)
        return values * static_features["atom37_mask"].to(self.dtype)

    def _chirality(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Packed chirality vectors, in the noisy state's frame when the checkpoint was trained that way."""
        vectors = static_features[CHIRALITY]
        rotate, translate = self.config.noise_align_chirality, self.config.noise_align_chirality_translate
        if rotate or translate:
            missing = [key for key in (CHIRAL_FRAME_ATOMS, CHIRAL_SLOT_OWNER) if key not in static_features]
            if missing:
                raise KeyError(f"noise-aligned chirality requires {' and '.join(missing)}")
            vectors = align_chirality(
                vectors,
                state["bb_ca"],
                padding_mask,
                static_features[CHIRAL_FRAME_ATOMS],
                static_features[CHIRAL_SLOT_OWNER],
                rotate=rotate,
                translate=translate,
            )
        return vectors.to(self.dtype)

    def _coordinate_group(self, static_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        groups = static_features[CONDITION_COORDINATE_GROUP].long().clone()
        populated = groups != COORDINATE_GROUP_UNKNOWN
        unique_groups, inverse = torch.unique(groups[populated], return_inverse=True)
        permutation = randperm(len(unique_groups), device=groups.device)
        groups[populated] = permutation[inverse].to(groups.dtype)
        return groups.to(self.dtype)

    def _single(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_type = static_features["token_type"].long()
        atomized = (token_type == self.config.atomized_token_type).unsqueeze(-1)

        if CONDITION_RASA in static_features:
            atom_mask = static_features["atom37_mask"].to(torch.bool)
            rasa = torch.where(
                atom_mask, static_features[CONDITION_RASA], torch.zeros_like(static_features[CONDITION_RASA])
            ).to(self.dtype)
        else:
            rasa = self._zeros(state, (ATOM37_SIZE,))

        if CONDITION_SECONDARY_STRUCTURE in static_features:
            secondary_structure = self.secondary_structure_embedding(
                static_features[CONDITION_SECONDARY_STRUCTURE].long()
            )
        else:
            secondary_structure = self._zeros(state, (self.config.secondary_structure_embedding_dim,))

        if CONDITION_NOT_CYSTEINE in static_features:
            not_cysteine = static_features[CONDITION_NOT_CYSTEINE].to(self.dtype).unsqueeze(-1)
        else:
            not_cysteine = self._zeros(state, (1,))

        if "charge" in static_features:
            charge = (static_features["charge"][:, :, 1].long() + self.config.charge_offset).clamp(
                0, self.config.num_charge_classes - 1
            )
            charge = self.charge_embedding(charge) * atomized
        else:
            charge = self._zeros(state, (self.config.charge_embedding_dim,))

        if "element" in static_features:
            element = static_features["element"][:, :, 1].long().clamp(0, self.config.max_atomic_number - 1)
            element = self.element_embedding(element) * atomized
        else:
            element = self._zeros(state, (self.config.element_embedding_dim,))

        if CHIRALITY in static_features:
            chirality = self._chirality(static_features, state, padding_mask)
        else:
            chirality = self._zeros(state, (self.config.chirality_dim,))

        if CONDITION_CHAIN_CENTROID in static_features:
            centroid = static_features[CONDITION_CHAIN_CENTROID].float()
            included = torch.isfinite(centroid).all(dim=-1, keepdim=True)
            centroid = torch.where(included, angstrom_to_nanometer(centroid), torch.zeros_like(centroid))
        else:
            centroid = self._zeros(state, (3,))

        if self.config.input_schema == "sequence_probability":
            features = [
                state["bb_ca"],
                state["local_latents"],
                self.token_type_embedding(token_type),
                self._sequence_probabilities(static_features, state),
                self._conditional_coordinates(
                    static_features,
                    state,
                    CONDITION_COORDINATE,
                    CONDITION_COORDINATE_MASK,
                    finite_marks_conditioned=True,
                ),
                self._conditional_coordinates(
                    static_features,
                    state,
                    CONDITION_INTERNAL_COORDINATE,
                    CONDITION_INTERNAL_COORDINATE_MASK,
                    group_marks_conditioned=True,
                ),
                self._rasa_binned(static_features, state),
                secondary_structure,
                self._atom_role_annotation(static_features, state, CONDITION_HBOND),
                self._masked_atom_values(static_features, state, CONDITION_PISTACK_SPECIFIED, nonzero=False),
                self._atom_role_annotation(static_features, state, CONDITION_PICATION),
                self._masked_atom_values(static_features, state, CONDITION_SALT_BRIDGE, nonzero=True),
                self._masked_atom_values(static_features, state, CONDITION_HYDROPHOBIC, nonzero=False),
                self._masked_atom_values(static_features, state, CONDITION_METAL_COORDINATION, nonzero=True),
                self._na_state(static_features, state),
                self._masked_atom_values(static_features, state, CONDITION_HOTSPOT_SPECIFIED, nonzero=False),
                charge,
                element,
                self._coordinate_group(static_features),
                chirality,
                centroid,
            ]
        else:
            common_features = [
                self.token_type_embedding(token_type),
                self._conditional_residue_type(static_features, state),
                self._conditional_coordinates(
                    static_features,
                    state,
                    CONDITION_COORDINATE,
                    CONDITION_COORDINATE_MASK,
                    finite_marks_conditioned=True,
                ),
                self._conditional_coordinates(
                    static_features,
                    state,
                    CONDITION_INTERNAL_COORDINATE,
                    CONDITION_INTERNAL_COORDINATE_MASK,
                    group_marks_conditioned=True,
                ),
                rasa,
                secondary_structure,
            ]
            trailing_features = [
                self._atom_annotation(static_features, state, CONDITION_HOTSPOT),
                not_cysteine,
                charge,
                element,
                self._coordinate_group(static_features),
                chirality,
                centroid,
            ]
            if self.config.input_schema == "latest":
                features = [
                    state["bb_ca"],
                    state["local_latents"],
                    *common_features,
                    self._atom_role_annotation(static_features, state, CONDITION_HBOND),
                    self._atom_annotation(static_features, state, CONDITION_PISTACK),
                    self._atom_role_annotation(static_features, state, CONDITION_PICATION),
                    self._na_state(static_features, state),
                    *trailing_features,
                ]
            else:
                active = _active_scale(self_conditioning_active, state["bb_ca"])
                features = [
                    state["bb_ca"],
                    state["local_latents"],
                    self_conditioning["bb_ca"] * active,
                    self_conditioning["local_latents"]
                    * _active_scale(self_conditioning_active, self_conditioning["local_latents"]),
                    *common_features,
                    self._atom_annotation(static_features, state, CONDITION_HBOND),
                    *trailing_features,
                ]
        features = torch.cat(features, dim=-1).to(self.dtype)
        valid = padding_mask.unsqueeze(-1).to(self.dtype)
        return self.single_projection(features * valid) * valid

    def _recycle_feature(
        self, feature: str, recycle_index: int, recycle_depth: int, batch: int, device: torch.device
    ) -> torch.Tensor:
        """The recycle track one trunk pass reads, ``[B, width]``, for a one-based pass index.

        ``recycle_index`` embeds the pass itself. ``recycle_remaining`` counts the passes
        still to come, saturating at the last slot; the first pass of any run reads the
        all-zero slot, which is also the whole of a single-pass run.
        """
        if feature == "recycle_index":
            scaled = torch.full(
                (batch,),
                recycle_index / RECYCLE_EMBEDDING_MAX_POSITIONS,
                device=device,
                dtype=torch.float32,
            )
            return _time_embedding(
                scaled,
                self.config.time_embedding_dim,
                max_positions=RECYCLE_EMBEDDING_MAX_POSITIONS,
            )
        if recycle_index > recycle_depth:
            raise ValueError(f"recycle pass {recycle_index} of a {recycle_depth}-pass run has no remaining count")
        width = self.config.recycle_remaining_dim
        remaining = torch.zeros((batch, width), device=device, dtype=torch.float32)
        if recycle_index > 1:
            remaining[:, min(recycle_depth - recycle_index, width - 1)] = 1.0
        return remaining

    def _condition(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        recycle_index: int = 1,
        recycle_depth: int = 1,
    ) -> torch.Tensor:
        cath_codes = static_features[CONDITION_CATH].long() + 1
        cath = torch.cat(
            [
                self.cath_class_embedding(cath_codes[..., 0]),
                self.cath_architecture_embedding(cath_codes[..., 1]),
                self.cath_topology_embedding(cath_codes[..., 2]),
            ],
            dim=-1,
        )
        batch, tokens = state["bb_ca"].shape[:2]
        time = (
            torch.cat(
                [
                    _time_embedding(times["bb_ca"], self.config.time_embedding_dim),
                    _time_embedding(times["local_latents"], self.config.time_embedding_dim),
                ],
                dim=-1,
            )
            .unsqueeze(1)
            .expand(batch, tokens, -1)
        )
        features = [time, cath]
        feature = self.config.recycle_feature
        if feature is not None:
            if recycle_index < 1:
                raise ValueError(f"recycle_index is one-based, got {recycle_index}")
            recycle = self._recycle_feature(feature, recycle_index, recycle_depth, batch, state["bb_ca"].device)
            features.append(recycle.unsqueeze(1).expand(batch, tokens, -1))
        valid = padding_mask.unsqueeze(-1).to(self.dtype)
        projected = torch.cat(features, dim=-1).to(self.dtype)
        return self.condition_projection(projected * valid) * valid

    def condition_stack(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        recycle_depth: int,
    ) -> torch.Tensor:
        """One projected condition per trunk pass, ``[recycle_depth, B, N, condition_dim]``.

        Pass 0 uses recycle index 1. Only a checkpoint whose condition carries a
        recycle feature reads a different row on every pass.
        """
        if not self.conditions_per_recycle_pass:
            raise ValueError("condition_stack requires a checkpoint whose condition carries a recycle feature")
        if recycle_depth < 1:
            raise ValueError(f"recycle_depth must be positive, got {recycle_depth}")
        padding_mask = padding_mask.to(torch.bool)
        return torch.stack(
            [
                self._condition(
                    static_features, state, times, padding_mask, recycle_index=cycle + 1, recycle_depth=recycle_depth
                )
                for cycle in range(recycle_depth)
            ]
        ).to(self.output_dtype)

    def _distance_indices(self, coordinates: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(coordinates.unsqueeze(2) - coordinates.unsqueeze(1), dim=-1)
        boundaries = torch.linspace(
            self.config.pair_distance_min,
            self.config.pair_distance_max,
            self.config.pair_distance_bins - 1,
            device=coordinates.device,
        )
        return torch.bucketize(distances, boundaries)

    def _pair(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens = padding_mask.shape
        zeros = torch.zeros((batch, tokens, tokens), device=padding_mask.device, dtype=self.dtype)
        index_group = static_features[CONDITION_INDEX_GROUP].long()
        molecule = static_features["molecule_iid"].long()
        path_length = static_features["token_bond_path_length"].long().clamp(max=self.config.token_bond_path_bins - 1)
        pair_mask = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)
        xt_distance = F.one_hot(self._distance_indices(state["bb_ca"]), num_classes=self.config.pair_distance_bins).to(
            self.dtype
        )
        if "bond_order" in static_features:
            bond_order = F.one_hot(static_features["bond_order"].long(), num_classes=self.config.num_bond_orders).to(
                self.dtype
            )
        else:
            bond_order = torch.zeros((*zeros.shape, self.config.num_bond_orders), device=zeros.device, dtype=self.dtype)
        pair_features = [
            xt_distance,
        ]
        if self.config.input_schema == "legacy_self_conditioned":
            self_conditioned_distance = F.one_hot(
                self._distance_indices(self_conditioning["bb_ca"]), num_classes=self.config.pair_distance_bins
            ).to(self.dtype)
            self_conditioned_distance = self_conditioned_distance * _active_scale(
                self_conditioning_active, self_conditioned_distance
            )
            pair_features.append(self_conditioned_distance)
        pair_features.extend(
            [
                static_features.get("bond_mask", zeros).to(self.dtype).unsqueeze(-1),
                bond_order,
                static_features.get("is_aromatic", zeros).to(self.dtype).unsqueeze(-1),
                (index_group.unsqueeze(-1) == index_group.unsqueeze(-2)).to(self.dtype).unsqueeze(-1),
                (molecule.unsqueeze(-1) == molecule.unsqueeze(-2)).to(self.dtype).unsqueeze(-1),
            ]
        )
        if self.config.input_schema != "legacy_self_conditioned":
            entity = static_features.get("molecule_entity_id", molecule).long()
            pair_features.append((entity.unsqueeze(-1) == entity.unsqueeze(-2)).to(self.dtype).unsqueeze(-1))
        pair_features.extend(
            [
                F.one_hot(path_length, num_classes=self.config.token_bond_path_bins).to(self.dtype),
                static_features.get("canonical_polymer_ordering", zeros).to(self.dtype).unsqueeze(-1),
            ]
        )
        if self.config.input_schema != "legacy_self_conditioned":
            na_pair = static_features.get(NA_SS_PAIR, zeros).to(self.dtype).unsqueeze(-1)
            na_pair_mask = static_features.get(NA_SS_PAIR_MASK, zeros).to(self.dtype).unsqueeze(-1)
            pair_features.extend((na_pair, na_pair_mask))
        pair_features = torch.cat(pair_features, dim=-1)
        pair_valid = pair_mask.unsqueeze(-1).to(self.dtype)
        pair = self.pair_norm(self.pair_projection(pair_features * pair_valid)) * pair_valid

        pair_time = torch.cat(
            [
                _time_embedding(times["bb_ca"], self.config.time_embedding_dim),
                _time_embedding(times["local_latents"], self.config.time_embedding_dim),
            ],
            dim=-1,
        ).to(self.dtype)
        # One condition row per design; the AdaLN broadcasts it over the pairs and masks its output.
        pair_condition = self.pair_condition_norm(self.pair_condition_projection(pair_time))
        return self.pair_adaln(pair.to(self.pair_adaln_dtype), pair_condition.to(self.pair_adaln_dtype), pair_mask)

    def forward(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        recycle_index: int = 1,
    ) -> RFProteinaRepresentations:
        """Construct all velocity-network input tracks.

        ``recycle_index`` is the one-based trunk pass used by the
        sequence-probability condition feature. Pass 1 is the default, which
        matches the condition compared before a recycled trunk run.
        """
        if padding_mask is None:
            padding_mask = static_features["padding_mask"]
        padding_mask = padding_mask.to(torch.bool)

        # Match upstream evaluation order; coordinate-group randomization in
        # the single factory consumes the global random stream.
        condition = self._condition(static_features, state, times, padding_mask, recycle_index=recycle_index)
        single = self._single(static_features, state, self_conditioning, self_conditioning_active, padding_mask)
        pair = self._pair(static_features, state, times, self_conditioning, self_conditioning_active, padding_mask)
        return RFProteinaRepresentations(
            single=single.to(self.output_dtype),
            pair=pair.to(self.output_dtype),
            condition=condition.to(self.output_dtype),
        )
