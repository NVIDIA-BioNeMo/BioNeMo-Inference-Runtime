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
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

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


class RFProteinaTrajectoryFrame(TypedDict):
    """One numerical step's updated state, clean estimate, and optional readout."""

    x_t: RFProteinaState
    x_1: RFProteinaState
    sequence_logits: torch.Tensor | None


class RFProteinaDecodedOutput(TypedDict):
    """Decoder tensors in the caller's original batch and token layout.

    Coordinates are in nanometres. Residue IDs index ``sequence_logits``;
    non-latent tokens have ID -1. Raw presence logits precede residue-table
    filtering and are absent on checkpoints without an atom-presence head.
    """

    coords: torch.Tensor
    sequence_logits: torch.Tensor
    residue_type: torch.Tensor
    atom_presence_logits: NotRequired[torch.Tensor]


@dataclass
class RFProteinaSimulationOutput:
    """Final product-space state and optional sampling diagnostics.

    Every :data:`RFProteinaState` here holds one tensor per modality: ``bb_ca``
    ``[B, N, 3]`` in nanometres and ``local_latents`` ``[B, N, latent_dim]``.
    """

    #: State at the requested interval end (``t = 1`` for a full rollout).
    state: RFProteinaState
    #: The network's final clean-data estimate.
    clean_prediction: RFProteinaState
    #: Token-pair insertion scores, absent when the head is disabled.
    insertion_logits: torch.Tensor | None
    #: Requested rollout snapshots.
    trajectory: list[RFProteinaTrajectoryFrame]
    #: Final-step per-token sequence-class scores from the trunk's readout
    #: probe, absent when the head is disabled. Distinct from the decoder's
    #: ``RFProteinaDecodedOutput.sequence_logits``.
    trunk_sequence_logits: torch.Tensor | None = None
    #: Seed of the rollout's random stream.
    seed: int | None = None


@dataclass(frozen=True)
class RFProteinaCleanStateEncoding:
    """Posterior statistics and clean product-space endpoint from the AE."""

    clean_state: RFProteinaState
    mean: torch.Tensor
    log_scale: torch.Tensor
    encoding_mask: torch.Tensor


@dataclass(frozen=True)
class RFProteinaRepresentations:
    """Dense representations consumed by the velocity trunk."""

    single: torch.Tensor
    pair: torch.Tensor
    condition: torch.Tensor


@dataclass(frozen=True)
class RFProteinaConversionReport:
    """Checkpoint conversion coverage."""

    loaded_keys: tuple[str, ...]
    ignored_source_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]


RFProteinaPredictFn = Callable[
    [RFProteinaState, dict[str, torch.Tensor], RFProteinaState, bool, int],
    RFProteinaPrediction,
]
