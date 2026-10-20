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
"""Shared runtime contracts for RFProteína model components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.layers.transformers.local_latents_decoder import get_residue_atom_mask

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaReferenceConfig

#: Product-space state: modality name -> tensor.
RFProteinaState = dict[str, torch.Tensor]
#: Velocity-network output: modality name -> {"v": velocity}, plus an optional
#: ``"insertion"`` head.
RFProteinaPrediction = dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True)
class ModalityReference:
    """Resolved reference distribution for one modality."""

    dim: int
    config: RFProteinaReferenceConfig


@dataclass(frozen=True)
class RFProteinaCorruptionOutput:
    """Clean/reference/corrupted states from a partial forward marginal."""

    clean_state: RFProteinaState
    reference_state: RFProteinaState
    corrupted_state: RFProteinaState
    times: dict[str, torch.Tensor]


class _CorruptionView:
    """Expose states from a composed corruption result."""

    corruption: RFProteinaCorruptionOutput

    @property
    def clean_state(self) -> RFProteinaState:
        return self.corruption.clean_state

    @property
    def reference_state(self) -> RFProteinaState:
        return self.corruption.reference_state

    @property
    def corrupted_state(self) -> RFProteinaState:
        return self.corruption.corrupted_state


@dataclass
class RFProteinaSamplingOutput:
    """Final product-space state and optional sampling diagnostics.

    Every :data:`RFProteinaState` here holds one tensor per modality: ``bb_ca``
    ``[B, N, 3]`` in nanometres and ``local_latents`` ``[B, N, latent_dim]``.
    """

    #: State at ``t = 1``, the end of the rollout.
    state: RFProteinaState
    #: The network's final clean-data estimate.
    clean_prediction: RFProteinaState
    #: Token-pair insertion scores, absent when the head is disabled.
    insertion_logits: torch.Tensor | None
    #: Requested rollout snapshots.
    trajectory: list[dict[str, RFProteinaState]]
    #: Final-step per-token sequence-class scores from the trunk's readout
    #: probe, absent when the head is disabled. Distinct from the decoder's
    #: ``RFProteinaPredictionOutput.sequence_logits``.
    trunk_sequence_logits: torch.Tensor | None = None
    #: Seed of the rollout's random stream.
    seed: int | None = None


@dataclass
class RFProteinaPartialSamplingOutput(_CorruptionView):
    """Partial rollout plus its exact forward-marginal initialization."""

    sampling: RFProteinaSamplingOutput
    corruption: RFProteinaCorruptionOutput
    start_times: dict[str, float]
    num_steps: int


@dataclass(frozen=True)
class RFProteinaCleanStateEncoding:
    """Posterior statistics and clean product-space endpoint from the AE."""

    clean_state: RFProteinaState
    mean: torch.Tensor
    log_scale: torch.Tensor
    encoding_mask: torch.Tensor


@dataclass(frozen=True)
class RFProteinaPartialRedesignOutput(_CorruptionView):
    """Diagnostics and fixed-coordinate output view for partial redesign."""

    #: Resolved product-modality start times.
    start_times: dict[str, float]
    #: Number of schedule intervals actually executed.
    num_steps: int
    #: Forward-marginal clean, reference, and corrupted states.
    corruption: RFProteinaCorruptionOutput
    #: Raw decoder output before coordinate restoration, ``[B,N,37,3]``.
    raw_coordinates: torch.Tensor
    #: Decoder output with fixed absolute coordinates restored.
    coordinates: torch.Tensor
    #: Atom slots restored from absolute coordinate conditions.
    fixed_coordinate_mask: torch.Tensor


@dataclass(frozen=True)
class RFProteinaRepresentations:
    """Dense representations consumed by the velocity trunk."""

    single: torch.Tensor
    pair: torch.Tensor
    condition: torch.Tensor


@dataclass(frozen=True)
class RFProteinaMaterializedOutput:
    """Common coordinate and identity view for non-protein token classes."""

    #: Tokens selected by this output, ``[B, N]``.
    token_mask: torch.Tensor
    #: Coordinates before fixed conditions are restored, ``[B,N,37,3]``.
    generated_coordinates: torch.Tensor
    #: Canonical final coordinates, ``[B,N,37,3]``.
    coordinates: torch.Tensor
    #: Final occupied atom slots, ``[B, N, 37]``.
    atom_mask: torch.Tensor
    #: Atoms restored from absolute coordinate conditions.
    coordinate_condition_mask: torch.Tensor
    #: Canonical unified residue identities.
    residue_types: torch.Tensor
    #: Static token-kind IDs.
    token_type: torch.Tensor

    @property
    def fixed_token_mask(self) -> torch.Tensor:
        """Return selected tokens with at least one fixed coordinate."""
        return self.coordinate_condition_mask.any(dim=-1)


@dataclass(frozen=True)
class RFProteinaAtomizedOutput(RFProteinaMaterializedOutput):
    """Batch-aligned model output for atomized non-polymer tokens."""

    element: torch.Tensor | None = None
    charge: torch.Tensor | None = None
    chirality: torch.Tensor | None = None
    molecule_iid: torch.Tensor | None = None
    chain_iid: torch.Tensor | None = None
    bond_mask: torch.Tensor | None = None
    bond_order: torch.Tensor | None = None
    is_aromatic: torch.Tensor | None = None
    token_bond_path_length: torch.Tensor | None = None


@dataclass(frozen=True)
class RFProteinaNucleicAcidOutput(RFProteinaMaterializedOutput):
    """Batch-aligned DNA/RNA output in the unified residue space."""

    rna_token_mask: torch.Tensor
    dna_token_mask: torch.Tensor
    generated_atom_mask: torch.Tensor
    generated_residue_types: torch.Tensor
    sequence_condition_mask: torch.Tensor
    input_residue_types: torch.Tensor
    chain_iid: torch.Tensor | None = None
    molecule_iid: torch.Tensor | None = None

    @property
    def fully_fixed_token_mask(self) -> torch.Tensor:
        """Return tokens whose complete final atom layout is conditioned."""
        return self.token_mask & (self.coordinate_condition_mask | ~self.atom_mask).all(dim=-1)


@dataclass(frozen=True)
class RFProteinaInsertionOutput:
    """Tensor view of generated tokens with motif conditions materialized."""

    assignment: torch.Tensor
    source_token_index: torch.Tensor
    generated_token_mask: torch.Tensor
    target_mask: torch.Tensor
    coordinates: torch.Tensor
    coordinate_mask: torch.Tensor
    residue_types: torch.Tensor
    sequence_mask: torch.Tensor


@dataclass(frozen=True)
class RFProteinaPredictionOutput:
    """End-to-end sampling result and decoded product-space prediction."""

    sampling: RFProteinaSamplingOutput
    coordinates: torch.Tensor
    sequence_logits: torch.Tensor
    residue_types: torch.Tensor
    atom_mask: torch.Tensor
    insertion: RFProteinaInsertionOutput | None = None
    nucleic_acid: RFProteinaNucleicAcidOutput | None = None
    atomized: RFProteinaAtomizedOutput | None = None
    partial_redesign: RFProteinaPartialRedesignOutput | None = None
    residue_type_offset: int = 0

    @property
    def insertion_logits(self) -> torch.Tensor | None:
        """Return final token-pair insertion scores, when enabled."""
        return self.sampling.insertion_logits

    @property
    def insertion_assignment(self) -> torch.Tensor | None:
        """Return unindexed-source to generated-target assignments."""
        return None if self.insertion is None else self.insertion.assignment

    @property
    def coordinates_with_insertion(self) -> torch.Tensor:
        """Return coordinates with motif references materialized."""
        return self.coordinates if self.insertion is None else self.insertion.coordinates

    @property
    def residue_types_with_insertion(self) -> torch.Tensor:
        """Return residue types with motif sequence conditions materialized."""
        return self.residue_types if self.insertion is None else self.insertion.residue_types

    @property
    def canonical_residue_types(self) -> torch.Tensor:
        """Return canonical unified IDs without hard-fixing redesign sequence."""
        residue_types = self.residue_types_with_insertion if self.partial_redesign is None else self.residue_types
        return torch.where(residue_types >= 0, residue_types + self.residue_type_offset, residue_types)

    @property
    def complex_coordinates(self) -> torch.Tensor:
        """Return the final protein/NA/ligand atom37 coordinate view."""
        coordinates = self.coordinates_with_insertion
        for materialized in (self.nucleic_acid, self.atomized):
            if materialized is not None:
                coordinates = torch.where(
                    materialized.token_mask.unsqueeze(-1).unsqueeze(-1),
                    materialized.coordinates,
                    coordinates,
                )
        return coordinates

    @property
    def complex_residue_types(self) -> torch.Tensor:
        """Return final unified protein, DNA/RNA, and ligand identities."""
        residue_types = self.canonical_residue_types
        for materialized in (self.nucleic_acid, self.atomized):
            if materialized is not None:
                residue_types = torch.where(materialized.token_mask, materialized.residue_types, residue_types)
        return residue_types

    @property
    def complex_atom_mask(self) -> torch.Tensor:
        """Return final atom37 occupancy after all materialization."""
        atom_mask = self.atom_mask
        if self.partial_redesign is None and self.insertion is not None and torch.any(self.insertion.sequence_mask):
            conditioned_mask = get_residue_atom_mask(
                self.canonical_residue_types,
                33,
                self.insertion.generated_token_mask,
            )
            atom_mask = torch.where(
                self.insertion.sequence_mask.unsqueeze(-1),
                conditioned_mask,
                atom_mask,
            )
        for materialized in (self.nucleic_acid, self.atomized):
            if materialized is not None:
                atom_mask = torch.where(materialized.token_mask.unsqueeze(-1), materialized.atom_mask, atom_mask)
        return atom_mask

    @property
    def final_coordinates(self) -> torch.Tensor:
        """Return the canonical coordinates for writing or downstream use."""
        return self.complex_coordinates if self.partial_redesign is None else self.partial_redesign.coordinates

    @property
    def final_residue_types(self) -> torch.Tensor:
        """Return unified output IDs matching redesign sequence semantics."""
        return self.complex_residue_types

    @property
    def final_atom_mask(self) -> torch.Tensor:
        """Return atom37 occupancy matching :attr:`final_residue_types`."""
        return self.complex_atom_mask


@dataclass(frozen=True)
class RFProteinaSearchCost:
    """Work performed by one metric-weighted beam search."""

    num_requests: int
    beam_width: int
    n_branch: int
    num_checkpoints: int
    completion_rollouts: int
    velocity_steps: int


@dataclass(frozen=True)
class RFProteinaSearchCandidate:
    """One scored beam member or kept intermediate."""

    prediction: RFProteinaPredictionOutput
    metrics: dict[str, float]
    reward: float
    rank: int
    request_index: int
    parent_id: str | None
    candidate_id: str
    checkpoint: int
    seed: int
    is_intermediate: bool


@dataclass(frozen=True)
class RFProteinaSearchOutput:
    """Intermediates followed by the final beam, plus search cost."""

    candidates: tuple[RFProteinaSearchCandidate, ...]
    cost: RFProteinaSearchCost

    @property
    def final_candidates(self) -> tuple[RFProteinaSearchCandidate, ...]:
        """Return survivors of the last checkpoint."""
        return tuple(candidate for candidate in self.candidates if not candidate.is_intermediate)

    @property
    def intermediate_candidates(self) -> tuple[RFProteinaSearchCandidate, ...]:
        """Return optional intermediates that passed the save criterion."""
        return tuple(candidate for candidate in self.candidates if candidate.is_intermediate)


@dataclass(frozen=True)
class RFProteinaConversionReport:
    """Checkpoint conversion coverage."""

    loaded_keys: tuple[str, ...]
    ignored_source_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]


RFProteinaRepresentationBuilderFn = Callable[
    [RFProteinaState, dict[str, torch.Tensor], RFProteinaState, bool], RFProteinaRepresentations
]
RFProteinaPredictFn = Callable[
    [RFProteinaState, dict[str, torch.Tensor], RFProteinaState, bool, int],
    RFProteinaPrediction,
]

__all__ = [
    "ModalityReference",
    "RFProteinaAtomizedOutput",
    "RFProteinaCleanStateEncoding",
    "RFProteinaConversionReport",
    "RFProteinaCorruptionOutput",
    "RFProteinaInsertionOutput",
    "RFProteinaMaterializedOutput",
    "RFProteinaNucleicAcidOutput",
    "RFProteinaPartialRedesignOutput",
    "RFProteinaPartialSamplingOutput",
    "RFProteinaPredictFn",
    "RFProteinaPrediction",
    "RFProteinaPredictionOutput",
    "RFProteinaRepresentationBuilderFn",
    "RFProteinaRepresentations",
    "RFProteinaSamplingOutput",
    "RFProteinaSearchCandidate",
    "RFProteinaSearchCost",
    "RFProteinaSearchOutput",
    "RFProteinaState",
]
