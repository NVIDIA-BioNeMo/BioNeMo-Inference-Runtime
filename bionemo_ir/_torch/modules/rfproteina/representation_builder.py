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

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.sampling.random import randperm
from bionemo_ir.dsl_kernels.triton.rfd4_pair_init import rfd4_pair_init
from bionemo_ir.utils import str_dtype_to_torch

from .atom_utils import angstrom_to_nanometer, finite_atom_mask
from .chirality import CHIRAL_FRAME_ATOMS, CHIRAL_SLOT_OWNER, CHIRALITY, align_chirality, chiral_frames
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
    """Upstream-compatible parameters of the pair AdaLN that conditions the pair track on its design.

    The condition holds one row per design, so its scale and bias are projected once per design rather than per
    pair; :func:`rfd4_pair_init` applies them to every pair as ``sigmoid(scale) * RMSNorm(pair) + bias``.
    """

    def __init__(self, pair_dim: int, condition_dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.norm_cond = nn.RMSNorm(condition_dim, eps=eps, dtype=dtype)
        # The kernel applies the sigmoid; the module keeps it for the checkpoint layout.
        self.to_gamma = nn.Sequential(nn.Linear(condition_dim, pair_dim, dtype=dtype), nn.Sigmoid())
        self.to_beta = nn.Linear(condition_dim, pair_dim, bias=False, dtype=dtype)

    def modulation(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each design's ``[B, pair_dim]`` scale, before the kernel's sigmoid, and bias from its ``[B, condition_dim]``
        condition."""
        condition = self.norm_cond(condition)
        return self.to_gamma[0](condition), self.to_beta(condition)


@functools.cache
def _frequencies(half_dim: int, max_positions: int, device: torch.device) -> torch.Tensor:
    """The sinusoidal embedding's ``[half_dim]`` FP32 frequencies, built once per device as a normal tensor."""
    exponent = math.log(max_positions) / (half_dim - 1)
    with torch.inference_mode(False):
        return torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -exponent)


def _time_embedding(
    t: torch.Tensor, embedding_dim: int, max_positions: int = TIME_EMBEDDING_MAX_POSITIONS
) -> torch.Tensor:
    """Sinusoidal diffusion-time embedding used by the released model."""
    if t.ndim != 1:
        raise ValueError(f"Expected time tensor [B], got {tuple(t.shape)}")
    phase = (t * max_positions).float().unsqueeze(-1)
    angles = phase * _frequencies(embedding_dim // 2, max_positions, t.device)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _active_scale(active: bool | torch.Tensor, reference: torch.Tensor) -> torch.Tensor | float:
    """``active`` as a factor for ``reference``. A host value stays on the host, which spares a blocking copy."""
    if not isinstance(active, torch.Tensor):
        return float(active)
    scale = active.to(device=reference.device, dtype=reference.dtype)
    while scale.ndim < reference.ndim:
        scale = scale.unsqueeze(-1)
    return scale


class _CoordinateGroups(NamedTuple):
    """A design's ``[B, N]`` coordinate groups, which of them are populated, and each populated one's rank among the
    ``count`` distinct labels; a step only permutes the labels."""

    labels: torch.Tensor
    populated: torch.Tensor
    ranks: torch.Tensor
    count: int


class _PairFeatures(NamedTuple):
    """A design's static pair features as :func:`rfd4_pair_init` reads them, with the projection rows they select."""

    #: FP32 ``[pair_distance_bins - 1]`` edges of the distance bins.
    edges: torch.Tensor
    #: int32 ``[B, N, N, 2]`` bond-order class, ``-1`` for none, and bond-path bin.
    classes: torch.Tensor
    #: FP32 ``[B, N, N, S]`` scalar channels.
    scalars: torch.Tensor
    #: ``[B, N]`` mask of real tokens.
    mask: torch.Tensor
    #: FP32 projection rows per distance bin, bond order, bond-path bin and scalar channel, each ``[*, pair_dim]``.
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    #: The pair norm's FP32 weight, its bias for a LayerNorm (``None`` for an RMSNorm), and its epsilon.
    norm: tuple[torch.Tensor, torch.Tensor | None, float]
    #: FP32 projection rows ``[bins, pair_dim]`` of the self-conditioning distance bins, for the legacy schema.
    self_conditioned: torch.Tensor | None


@dataclass(frozen=True)
class StaticTrackParts:
    """The parts of one design's input tracks that its static features alone fix.

    The builder's ``prepare`` builds them once for a rollout. Each step then adds only what its sampling state
    changes, and concatenates the same values in the same order as a build from the features alone.
    """

    #: ``[B, N, 1]`` token mask in the builder dtype.
    valid: torch.Tensor
    #: Single features between the state's and the coordinate group.
    single_static: torch.Tensor
    #: Chirality vectors, or ``None`` when the checkpoint aligns them to each step's noisy state.
    chirality: torch.Tensor | None
    #: :func:`chiral_frames` for that alignment, or ``None`` when the frame tensors are missing.
    chiral_frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    #: Chain centroids, the last single features.
    centroid: torch.Tensor
    #: Coordinate groups, whose labels each step permutes.
    coordinate_groups: _CoordinateGroups
    #: ``[B, N, 3 * cath_embedding_dim]`` CATH embeddings of the condition track.
    cath: torch.Tensor
    #: The static pair features as :func:`rfd4_pair_init` reads them.
    pair_features: _PairFeatures


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

    def _zeros(self, padding_mask: torch.Tensor, trailing_shape: tuple[int, ...]) -> torch.Tensor:
        """``[B, N, *trailing_shape]`` zero features for a ``[B, N]`` token mask."""
        return torch.zeros((*padding_mask.shape[:2], *trailing_shape), device=padding_mask.device, dtype=self.dtype)

    def _conditional_coordinates(
        self,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        coordinate_key: str,
        mask_key: str,
        *,
        finite_marks_conditioned: bool = False,
        group_marks_conditioned: bool = False,
    ) -> torch.Tensor:
        """Atom37 coordinates in nanometres, flattened, with a mask channel.

        Args:
            static_features: Tensorized batch. Coordinates are angstroms.
            padding_mask: ``[B, N]`` token mask, read only for the zero-feature shape.
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
            return self._zeros(padding_mask, (ATOM37_SIZE * 4,))
        raw = static_features[coordinate_key]
        if mask_key in static_features:
            mask = static_features[mask_key].to(torch.bool)
        elif finite_marks_conditioned:
            mask = finite_atom_mask(raw)
        elif group_marks_conditioned and CONDITION_COORDINATE_GROUP in static_features:
            groups = static_features[CONDITION_COORDINATE_GROUP].long()
            mask = (groups != COORDINATE_GROUP_UNKNOWN) & (groups != COORDINATE_GROUP_ABSOLUTE)
        else:
            return self._zeros(padding_mask, (ATOM37_SIZE * 4,))
        coordinates = angstrom_to_nanometer(torch.nan_to_num(raw, nan=0.0))
        coordinates = coordinates * mask.unsqueeze(-1)
        return torch.cat([coordinates.flatten(-2), mask.to(coordinates.dtype)], dim=-1)

    def _conditional_residue_type(
        self, static_features: Mapping[str, torch.Tensor], padding_mask: torch.Tensor
    ) -> torch.Tensor:
        if CONDITION_SEQUENCE not in static_features or CONDITION_SEQUENCE_MASK not in static_features:
            return self._zeros(padding_mask, (self.config.num_conditional_residue_types,))
        mask = static_features[CONDITION_SEQUENCE_MASK].to(torch.bool)
        residue_type = static_features[CONDITION_SEQUENCE].long() * mask
        return self.conditional_residue_embedding(residue_type) * mask.unsqueeze(-1)

    def _atom_annotation(
        self,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if key not in static_features:
            return self._zeros(padding_mask, (ATOM37_SIZE,))
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        return static_features[key].to(self.dtype) * atom_mask

    def _atom_role_annotation(
        self,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Encode NONE/FIRST/SECOND/BOTH atom roles as two binary channels."""
        if key not in static_features:
            return self._zeros(padding_mask, (2 * ATOM37_SIZE,))
        values = static_features[key].long()
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        first = ((values == 1) | (values == 3)) & atom_mask
        second = ((values == 2) | (values == 3)) & atom_mask
        return torch.cat((first, second), dim=-1).to(self.dtype)

    def _na_state(
        self,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if CONDITION_NA_SS_STATE not in static_features or CONDITION_NA_SS_STATE_MASK not in static_features:
            return self._zeros(padding_mask, (2,))
        value = static_features[CONDITION_NA_SS_STATE].to(self.dtype).unsqueeze(-1)
        mask = static_features[CONDITION_NA_SS_STATE_MASK].to(self.dtype).unsqueeze(-1)
        return torch.cat((value, mask), dim=-1)

    @property
    def conditions_per_recycle_pass(self) -> bool:
        """Whether each trunk pass gets its own condition track."""
        return self.config.recycle_feature is not None

    def _sequence_probabilities(self, static_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Probability value and explicit-mask channels in unified token order."""
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

    def _rasa_binned(self, static_features: Mapping[str, torch.Tensor], padding_mask: torch.Tensor) -> torch.Tensor:
        """Binned RASA; atoms outside the atom37 mask are unspecified (0)."""
        if CONDITION_RASA not in static_features:
            return self._zeros(padding_mask, (ATOM37_SIZE,))
        rasa = static_features[CONDITION_RASA].to(self.dtype)
        atom_mask = static_features["atom37_mask"].to(torch.bool)
        return rasa.masked_fill(~atom_mask, 0)

    def _masked_atom_values(
        self,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        key: str,
        *,
        nonzero: bool,
    ) -> torch.Tensor:
        """Atom37 contact flags, zero on atoms outside the atom37 mask."""
        if key not in static_features:
            return self._zeros(padding_mask, (ATOM37_SIZE,))
        values = static_features[key]
        values = (values != 0).to(self.dtype) if nonzero else values.to(self.dtype)
        return values * static_features["atom37_mask"].to(self.dtype)

    def _chirality(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        frame_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Packed chirality vectors, in the noisy state's frame when the checkpoint was trained that way.

        ``frame_index`` is the rollout's :func:`chiral_frames`, found here when omitted.
        """
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
                frame_index=frame_index,
            )
        return vectors.to(self.dtype)

    def _coordinate_groups(self, static_features: Mapping[str, torch.Tensor]) -> _CoordinateGroups:
        """The design's coordinate groups and the ranks of their distinct labels.

        Finding the distinct labels waits on the device, so a rollout does it once, in :meth:`prepare`.
        """
        groups = static_features[CONDITION_COORDINATE_GROUP].long()
        populated = groups != COORDINATE_GROUP_UNKNOWN
        labels, inverse = torch.unique(groups[populated], return_inverse=True)
        ranks = torch.zeros_like(groups)
        ranks[populated] = inverse
        return _CoordinateGroups(groups, populated, ranks, len(labels))

    def _coordinate_group(self, groups: _CoordinateGroups) -> torch.Tensor:
        """The coordinate groups with their labels randomly permuted, one draw from the sampling stream."""
        labels = groups.labels
        permutation = randperm(groups.count, device=labels.device)
        if groups.count:
            labels = torch.where(groups.populated, permutation[groups.ranks].to(labels.dtype), labels)
        return labels.to(self.dtype)

    def _single_static(
        self, static_features: Mapping[str, torch.Tensor], padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """The single features the static features alone fix.

        Returns:
            The features between the state's and the coordinate group, the chirality vectors unless the checkpoint
            aligns them to the noisy state (then ``None``), and the chain centroids.
        """
        token_type = static_features["token_type"].long()
        atomized = (token_type == self.config.atomized_token_type).unsqueeze(-1)

        if CONDITION_RASA in static_features:
            atom_mask = static_features["atom37_mask"].to(torch.bool)
            rasa = torch.where(
                atom_mask, static_features[CONDITION_RASA], torch.zeros_like(static_features[CONDITION_RASA])
            ).to(self.dtype)
        else:
            rasa = self._zeros(padding_mask, (ATOM37_SIZE,))

        if CONDITION_SECONDARY_STRUCTURE in static_features:
            secondary_structure = self.secondary_structure_embedding(
                static_features[CONDITION_SECONDARY_STRUCTURE].long()
            )
        else:
            secondary_structure = self._zeros(padding_mask, (self.config.secondary_structure_embedding_dim,))

        if CONDITION_NOT_CYSTEINE in static_features:
            not_cysteine = static_features[CONDITION_NOT_CYSTEINE].to(self.dtype).unsqueeze(-1)
        else:
            not_cysteine = self._zeros(padding_mask, (1,))

        if "charge" in static_features:
            charge = (static_features["charge"][:, :, 1].long() + self.config.charge_offset).clamp(
                0, self.config.num_charge_classes - 1
            )
            charge = self.charge_embedding(charge) * atomized
        else:
            charge = self._zeros(padding_mask, (self.config.charge_embedding_dim,))

        if "element" in static_features:
            element = static_features["element"][:, :, 1].long().clamp(0, self.config.max_atomic_number - 1)
            element = self.element_embedding(element) * atomized
        else:
            element = self._zeros(padding_mask, (self.config.element_embedding_dim,))

        if CHIRALITY not in static_features:
            chirality = self._zeros(padding_mask, (self.config.chirality_dim,))
        elif self.config.noise_align_chirality or self.config.noise_align_chirality_translate:
            chirality = None
        else:
            chirality = static_features[CHIRALITY].to(self.dtype)

        if CONDITION_CHAIN_CENTROID in static_features:
            centroid = static_features[CONDITION_CHAIN_CENTROID].float()
            included = torch.isfinite(centroid).all(dim=-1, keepdim=True)
            centroid = torch.where(included, angstrom_to_nanometer(centroid), torch.zeros_like(centroid))
        else:
            centroid = self._zeros(padding_mask, (3,))

        if self.config.input_schema == "sequence_probability":
            features = [
                self.token_type_embedding(token_type),
                self._sequence_probabilities(static_features),
                self._conditional_coordinates(
                    static_features,
                    padding_mask,
                    CONDITION_COORDINATE,
                    CONDITION_COORDINATE_MASK,
                    finite_marks_conditioned=True,
                ),
                self._conditional_coordinates(
                    static_features,
                    padding_mask,
                    CONDITION_INTERNAL_COORDINATE,
                    CONDITION_INTERNAL_COORDINATE_MASK,
                    group_marks_conditioned=True,
                ),
                self._rasa_binned(static_features, padding_mask),
                secondary_structure,
                self._atom_role_annotation(static_features, padding_mask, CONDITION_HBOND),
                self._masked_atom_values(static_features, padding_mask, CONDITION_PISTACK_SPECIFIED, nonzero=False),
                self._atom_role_annotation(static_features, padding_mask, CONDITION_PICATION),
                self._masked_atom_values(static_features, padding_mask, CONDITION_SALT_BRIDGE, nonzero=True),
                self._masked_atom_values(static_features, padding_mask, CONDITION_HYDROPHOBIC, nonzero=False),
                self._masked_atom_values(static_features, padding_mask, CONDITION_METAL_COORDINATION, nonzero=True),
                self._na_state(static_features, padding_mask),
                self._masked_atom_values(static_features, padding_mask, CONDITION_HOTSPOT_SPECIFIED, nonzero=False),
                charge,
                element,
            ]
        else:
            common_features = [
                self.token_type_embedding(token_type),
                self._conditional_residue_type(static_features, padding_mask),
                self._conditional_coordinates(
                    static_features,
                    padding_mask,
                    CONDITION_COORDINATE,
                    CONDITION_COORDINATE_MASK,
                    finite_marks_conditioned=True,
                ),
                self._conditional_coordinates(
                    static_features,
                    padding_mask,
                    CONDITION_INTERNAL_COORDINATE,
                    CONDITION_INTERNAL_COORDINATE_MASK,
                    group_marks_conditioned=True,
                ),
                rasa,
                secondary_structure,
            ]
            trailing_features = [
                self._atom_annotation(static_features, padding_mask, CONDITION_HOTSPOT),
                not_cysteine,
                charge,
                element,
            ]
            if self.config.input_schema == "latest":
                features = [
                    *common_features,
                    self._atom_role_annotation(static_features, padding_mask, CONDITION_HBOND),
                    self._atom_annotation(static_features, padding_mask, CONDITION_PISTACK),
                    self._atom_role_annotation(static_features, padding_mask, CONDITION_PICATION),
                    self._na_state(static_features, padding_mask),
                    *trailing_features,
                ]
            else:
                features = [
                    *common_features,
                    self._atom_annotation(static_features, padding_mask, CONDITION_HBOND),
                    *trailing_features,
                ]
        return torch.cat(features, dim=-1), chirality, centroid

    def _single(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        padding_mask: torch.Tensor,
        parts: StaticTrackParts,
    ) -> torch.Tensor:
        """The projected single track: the state's features around the static ones in ``parts``."""
        features = [state["bb_ca"], state["local_latents"]]
        if self.config.input_schema == "legacy_self_conditioned":
            active = _active_scale(self_conditioning_active, state["bb_ca"])
            features += [
                self_conditioning["bb_ca"] * active,
                self_conditioning["local_latents"]
                * _active_scale(self_conditioning_active, self_conditioning["local_latents"]),
            ]
        chirality = parts.chirality
        if chirality is None:
            chirality = self._chirality(static_features, state, padding_mask, parts.chiral_frames)
        features += [parts.single_static, self._coordinate_group(parts.coordinate_groups), chirality, parts.centroid]
        features = torch.cat(features, dim=-1).to(self.dtype)
        return self.single_projection(features * parts.valid) * parts.valid

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

    def _cath(self, static_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, N, 3 * cath_embedding_dim]`` class, architecture and topology embeddings of the CATH labels."""
        cath_codes = static_features[CONDITION_CATH].long() + 1
        return torch.cat(
            [
                self.cath_class_embedding(cath_codes[..., 0]),
                self.cath_architecture_embedding(cath_codes[..., 1]),
                self.cath_topology_embedding(cath_codes[..., 2]),
            ],
            dim=-1,
        )

    def _time(self, times: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, 2 * time_embedding_dim]`` backbone and latent time embeddings, which the condition and pair share."""
        return torch.cat(
            [
                _time_embedding(times["bb_ca"], self.config.time_embedding_dim),
                _time_embedding(times["local_latents"], self.config.time_embedding_dim),
            ],
            dim=-1,
        )

    def _condition_from(
        self,
        time: torch.Tensor,
        cath: torch.Tensor,
        valid: torch.Tensor,
        recycle_index: int,
        recycle_depth: int,
    ) -> torch.Tensor:
        """One trunk pass's projected condition from the step's time embedding and the design's CATH embeddings."""
        batch, tokens = cath.shape[:2]
        features = [time.unsqueeze(1).expand(batch, tokens, -1), cath]
        feature = self.config.recycle_feature
        if feature is not None:
            if recycle_index < 1:
                raise ValueError(f"recycle_index is one-based, got {recycle_index}")
            recycle = self._recycle_feature(feature, recycle_index, recycle_depth, batch, cath.device)
            features.append(recycle.unsqueeze(1).expand(batch, tokens, -1))
        projected = torch.cat(features, dim=-1).to(self.dtype)
        return self.condition_projection(projected * valid) * valid

    def _condition(
        self,
        static_features: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        recycle_index: int = 1,
        recycle_depth: int = 1,
    ) -> torch.Tensor:
        valid = padding_mask.unsqueeze(-1).to(self.dtype)
        return self._condition_from(self._time(times), self._cath(static_features), valid, recycle_index, recycle_depth)

    def _conditions(
        self,
        time: torch.Tensor,
        cath: torch.Tensor,
        valid: torch.Tensor,
        recycle_depth: int,
        first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[recycle_depth, B, N, condition_dim]`` conditions in the output dtype; ``first`` is pass 1's if held."""
        passes = [
            first
            if cycle == 0 and first is not None
            else self._condition_from(time, cath, valid, cycle + 1, recycle_depth).to(self.output_dtype)
            for cycle in range(recycle_depth)
        ]
        return torch.stack(passes)

    def condition_stack(
        self,
        static_features: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        recycle_depth: int,
    ) -> torch.Tensor:
        """One projected condition per trunk pass, ``[recycle_depth, B, N, condition_dim]``.

        Pass 0 uses recycle index 1. Only a checkpoint whose condition carries a
        recycle feature reads a different row on every pass; the passes share
        one time embedding and the design's CATH embeddings.
        """
        if not self.conditions_per_recycle_pass:
            raise ValueError("condition_stack requires a checkpoint whose condition carries a recycle feature")
        if recycle_depth < 1:
            raise ValueError(f"recycle_depth must be positive, got {recycle_depth}")
        valid = padding_mask.to(torch.bool).unsqueeze(-1).to(self.dtype)
        return self._conditions(self._time(times), self._cath(static_features), valid, recycle_depth)

    def _pair(
        self,
        parts: StaticTrackParts,
        state: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The pair track, from one :func:`rfd4_pair_init` launch over the state's distance bins and ``parts``.

        ``out`` are padded buffers that both receive it, as the trunk's initial-pair buffers do.
        """
        features = parts.pair_features
        # One condition row per design; the kernel's AdaLN broadcasts it over the pairs.
        pair_condition = self.pair_condition_norm(self.pair_condition_projection(time.to(self.dtype)))
        scale, bias = self.pair_adaln.modulation(pair_condition.to(self.pair_adaln_dtype))
        conditioned = None
        if features.self_conditioned is not None:
            coordinates = self_conditioning["bb_ca"].float()
            batch = coordinates.shape[0]
            if isinstance(self_conditioning_active, torch.Tensor):
                active = self_conditioning_active.to(coordinates.device, torch.float32).reshape(-1).expand(batch)
                active = active.contiguous()
            else:
                # A fill, unlike a tensor made from the host value, does not wait on the device.
                active = coordinates.new_full((batch,), float(self_conditioning_active))
            conditioned = (coordinates, active, features.self_conditioned)
        norm_weight, norm_bias, norm_eps = features.norm
        return rfd4_pair_init(
            state["bb_ca"].float(),
            features.classes,
            features.scalars,
            features.mask,
            features.edges,
            features.weights,
            norm_weight,
            norm_bias,
            scale,
            bias,
            norm_eps=norm_eps,
            adaln_eps=self.pair_adaln.eps,
            out_dtype=self.pair_adaln_dtype,
            self_conditioning=conditioned,
            out=out,
        )

    def _pair_features(self, static_features: Mapping[str, torch.Tensor], padding_mask: torch.Tensor) -> _PairFeatures:
        """The static pair features as :func:`rfd4_pair_init` reads them: one-hot classes as indices, plus scalars.

        The projection's input columns are the distance bins (twice for the legacy schema, which also bins the
        self-conditioning coordinates), the bond mask, the bond-order one-hot, the aromatic, index-group and
        molecule flags, the entity flag, the bond-path one-hot, the canonical-ordering channel, and the
        nucleic-acid pair channel with its mask; the legacy schema has neither the entity flag nor the nucleic-acid
        channels. A one-hot class selects one row of the transposed projection.
        """
        if not padding_mask.is_cuda:
            raise RuntimeError("the pair track runs on CUDA")
        config = self.config
        legacy = config.input_schema == "legacy_self_conditioned"
        bins, orders, paths = config.pair_distance_bins, config.num_bond_orders, config.token_bond_path_bins
        start = bins * (2 if legacy else 1)
        after_bond = start + 1 + orders
        path_start = after_bond + (3 if legacy else 4)
        tail = path_start + paths
        rows = self.pair_projection.weight.detach().T.float()
        if rows.shape[0] != tail + (1 if legacy else 3):
            raise ValueError(f"pair projection takes {rows.shape[0]} features, expected {tail + (1 if legacy else 3)}")

        batch, tokens = padding_mask.shape
        zeros = torch.zeros((batch, tokens, tokens), device=padding_mask.device)
        index_group = static_features[CONDITION_INDEX_GROUP].long()
        molecule = static_features["molecule_iid"].long()
        path = static_features["token_bond_path_length"].long().clamp(max=paths - 1)
        has_bond = "bond_order" in static_features
        bond = static_features["bond_order"].long() if has_bond else torch.full_like(path, -1)
        # One host read for the class ranges the one-hot encoding requires.
        if not bool(((bond >= 0 if has_bond else bond == -1) & (bond < orders) & (path >= 0)).all()):
            raise ValueError(f"bond_order must lie in [0, {orders}) and token_bond_path_length must be non-negative")

        def same(ids: torch.Tensor) -> torch.Tensor:
            return ids.unsqueeze(-1) == ids.unsqueeze(-2)

        # The scalar channels by their projection column, which also orders them.
        channels = {
            start: static_features.get("bond_mask", zeros),
            after_bond: static_features.get("is_aromatic", zeros),
            after_bond + 1: same(index_group),
            after_bond + 2: same(molecule),
            tail: static_features.get("canonical_polymer_ordering", zeros),
        }
        if not legacy:
            channels[after_bond + 3] = same(static_features.get("molecule_entity_id", molecule).long())
            channels[tail + 1] = static_features.get(NA_SS_PAIR, zeros)
            channels[tail + 2] = static_features.get(NA_SS_PAIR_MASK, zeros)
        scalar_rows = sorted(channels)
        norm = self.pair_norm
        return _PairFeatures(
            edges=torch.linspace(
                config.pair_distance_min, config.pair_distance_max, bins - 1, device=padding_mask.device
            ),
            classes=torch.stack((bond, path), dim=-1).to(torch.int32),
            scalars=torch.stack([channels[row].to(torch.float32) for row in scalar_rows], dim=-1),
            mask=padding_mask,
            weights=(
                rows[:bins].contiguous(),
                rows[start + 1 : after_bond].contiguous(),
                rows[path_start:tail].contiguous(),
                rows[scalar_rows].contiguous(),
            ),
            norm=(
                norm.weight.detach().float(),
                norm.bias.detach().float() if isinstance(norm, nn.LayerNorm) else None,
                norm.eps,
            ),
            self_conditioned=rows[bins:start].contiguous() if legacy else None,
        )

    def prepare(
        self, static_features: Mapping[str, torch.Tensor], padding_mask: torch.Tensor | None = None
    ) -> StaticTrackParts:
        """Build the parts of a design's tracks that its static features alone fix, once for its rollout.

        A step then builds only what its sampling state changes: the noisy coordinates and latents, the
        coordinate-group permutation it draws, chirality aligned to the noisy state, the time and recycle
        conditions, and the pair distance bins. Finding the coordinate groups' distinct labels and the chiral
        frames waits on the device, so a step that reuses these parts does neither.

        Args:
            static_features: The design's tensorized features.
            padding_mask: ``[B, N]`` mask of real tokens; ``static_features["padding_mask"]`` when omitted.

        Returns:
            The design's :class:`StaticTrackParts`.
        """
        if padding_mask is None:
            padding_mask = static_features["padding_mask"]
        padding_mask = padding_mask.to(torch.bool)
        single_static, chirality, centroid = self._single_static(static_features, padding_mask)
        frames = None
        if chirality is None and CHIRAL_FRAME_ATOMS in static_features and CHIRAL_SLOT_OWNER in static_features:
            frames = chiral_frames(static_features[CHIRAL_FRAME_ATOMS], padding_mask)
            if frames[0].numel() == 0:
                # Without a frame the alignment returns the supplied vectors, zero on padding, at every step.
                chirality = (static_features[CHIRALITY].float() * padding_mask[..., None]).to(self.dtype)
        return StaticTrackParts(
            valid=padding_mask.unsqueeze(-1).to(self.dtype),
            single_static=single_static,
            chirality=chirality,
            chiral_frames=frames,
            centroid=centroid,
            coordinate_groups=self._coordinate_groups(static_features),
            cath=self._cath(static_features),
            pair_features=self._pair_features(static_features, padding_mask),
        )

    def forward(
        self,
        static_features: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
        times: Mapping[str, torch.Tensor],
        self_conditioning: Mapping[str, torch.Tensor],
        self_conditioning_active: bool | torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        recycle_index: int = 1,
        parts: StaticTrackParts | None = None,
        recycle_depth: int = 1,
        pair_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> RFProteinaRepresentations:
        """Construct all velocity-network input tracks.

        ``recycle_index`` is the one-based trunk pass used by the
        sequence-probability condition feature. Pass 1 is the default, which
        matches the condition compared before a recycled trunk run. A
        ``recycle_depth`` above 1, on a checkpoint whose condition carries a
        recycle feature, gives one condition per trunk pass instead, as
        :meth:`condition_stack` builds them. ``parts`` are this design's
        :meth:`prepare` output, which a rollout builds once and hands to every
        step; a call without them builds its own. ``pair_out`` are two padded
        buffers in ``pair_adaln_dtype`` that both receive the pair track. When
        that is also the output dtype, the returned pair is a view of the first.
        """
        if padding_mask is None:
            padding_mask = static_features["padding_mask"]
        padding_mask = padding_mask.to(torch.bool)
        if parts is None:
            parts = self.prepare(static_features, padding_mask)
        time = self._time(times)

        # Match upstream evaluation order; coordinate-group randomization in
        # the single factory consumes the global random stream.
        condition = self._condition_from(time, parts.cath, parts.valid, recycle_index, 1)
        single = self._single(static_features, state, self_conditioning, self_conditioning_active, padding_mask, parts)
        pair = self._pair(parts, state, time, self_conditioning, self_conditioning_active, out=pair_out)
        condition = condition.to(self.output_dtype)
        if recycle_depth > 1 and self.conditions_per_recycle_pass:
            first = condition if recycle_index == 1 else None
            condition = self._conditions(time, parts.cath, parts.valid, recycle_depth, first)
        return RFProteinaRepresentations(
            single=single.to(self.output_dtype),
            pair=pair.to(self.output_dtype),
            condition=condition,
        )
