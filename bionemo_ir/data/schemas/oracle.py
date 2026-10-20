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
"""Shared data schemas for custom structure-oracle integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .basic import AtomType, FoldingOutput, InputRequest, PolymerType, ResType


class OracleStatus(StrEnum):
    """Outcome of evaluating one oracle request."""

    OK = "ok"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ResidueIdentity:
    """Source index resolved to BioIR's canonical residue schema.

    Attributes:
        source_index: Integer in the source model's residue vocabulary.
        residue_type: Canonical BioIR residue type.
        one_letter_code: Sequence code accepted by the target oracle. This may
            differ from ``residue_type.name`` for DNA, RNA, and custom
            components.
        polymer_type: Canonical BioIR polymer kind.
    """

    source_index: int
    residue_type: ResType
    one_letter_code: str | None = None
    polymer_type: PolymerType | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.residue_type, ResType):
            raise TypeError(f"residue_type must be ResType, got {type(self.residue_type).__name__}")
        if self.one_letter_code is not None and not self.one_letter_code:
            raise ValueError("one_letter_code must be None or a non-empty string")
        if self.polymer_type is not None and not isinstance(self.polymer_type, PolymerType):
            raise TypeError(f"polymer_type must be PolymerType, got {type(self.polymer_type).__name__}")

    @property
    def component_id(self) -> str:
        """Return the canonical PDB/CCD component identifier."""
        return self.residue_type.canonical_name


@dataclass(frozen=True)
class AtomIdentity:
    """Source index resolved to BioIR's canonical atom schema."""

    source_index: int
    atom_type: AtomType
    element: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.atom_type, AtomType):
            raise TypeError(f"atom_type must be AtomType, got {type(self.atom_type).__name__}")
        if self.element is not None and not self.element:
            raise ValueError("element must be None or a non-empty string")

    @property
    def atom_name(self) -> str:
        """Return the canonical atom name."""
        return self.atom_type.name


@dataclass(frozen=True)
class EncodedStructure:
    """Source-model structure before residue and atom identities are resolved.

    Coordinates use ångströms at the oracle boundary. Source adapters may use
    ``metadata`` to preserve model-specific arrays needed by custom resolvers.
    The first dimension contains only tokens included in this oracle request.

    Attributes:
        coordinates: Dense coordinates ``[tokens, atom_slots, 3]`` in ångströms.
        atom_mask: Occupancy mask ``[tokens, atom_slots]``.
        residue_type_indices: Source vocabulary indices ``[tokens]``.
        atom_type_indices: Source atom vocabulary indices
            ``[tokens, atom_slots]``.
        source_token_indices: Token positions in the source-model output
            ``[tokens]``.
        token_type_indices: Optional source token-kind indices ``[tokens]``.
        chain_indices: Optional source chain indices ``[tokens]``.
        metadata: Optional source-specific values available to resolvers.
    """

    coordinates: np.ndarray
    atom_mask: np.ndarray
    residue_type_indices: np.ndarray
    atom_type_indices: np.ndarray
    source_token_indices: np.ndarray
    token_type_indices: np.ndarray | None = None
    chain_indices: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError(
                f"coordinates must have shape [tokens, atom_slots, 3], got {tuple(self.coordinates.shape)}"
            )
        token_atom_shape = self.coordinates.shape[:2]
        if self.atom_mask.shape != token_atom_shape:
            raise ValueError(f"atom_mask must have shape {token_atom_shape}, got {tuple(self.atom_mask.shape)}")
        if self.atom_type_indices.shape != token_atom_shape:
            raise ValueError(
                f"atom_type_indices must have shape {token_atom_shape}, got {tuple(self.atom_type_indices.shape)}"
            )
        token_shape = (token_atom_shape[0],)
        for name, value in (
            ("residue_type_indices", self.residue_type_indices),
            ("source_token_indices", self.source_token_indices),
            ("token_type_indices", self.token_type_indices),
            ("chain_indices", self.chain_indices),
        ):
            if value is not None and value.shape != token_shape:
                raise ValueError(f"{name} must have shape {token_shape}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class ResolvedStructure:
    """Encoded coordinates paired with canonical BioIR molecular types."""

    encoded: EncodedStructure
    residues: tuple[ResidueIdentity, ...]
    atoms: tuple[tuple[AtomIdentity | None, ...], ...]

    def __post_init__(self) -> None:
        tokens, atom_slots = self.encoded.atom_mask.shape
        if len(self.residues) != tokens:
            raise ValueError(f"residues must contain {tokens} entries, got {len(self.residues)}")
        if len(self.atoms) != tokens or any(len(row) != atom_slots for row in self.atoms):
            raise ValueError(f"atoms must have shape [{tokens}, {atom_slots}]")
        occupied = self.encoded.atom_mask.astype(bool, copy=False)
        for token_index, row in enumerate(self.atoms):
            for atom_slot, identity in enumerate(row):
                if occupied[token_index, atom_slot] != (identity is not None):
                    raise ValueError(
                        "atoms must contain an identity exactly where atom_mask is true; "
                        f"mismatch at token {token_index}, atom slot {atom_slot}"
                    )

    @property
    def sequence(self) -> str:
        """Return a one-letter sequence, requiring every residue to define one."""
        missing = [index for index, residue in enumerate(self.residues) if residue.one_letter_code is None]
        if missing:
            raise ValueError(f"residues at token positions {missing} do not define one_letter_code")
        return "".join(residue.one_letter_code for residue in self.residues if residue.one_letter_code is not None)


@dataclass(frozen=True)
class OracleRequest:
    """One canonical structure-oracle request.

    ``input_request`` is reserved for a pipeline-ready BioIR request. Custom
    providers can consume ``structure`` directly today; a future BioIR
    provider can consume the same schema without changing this contract.
    """

    request_id: str
    structure: ResolvedStructure
    input_request: InputRequest | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.input_request is not None and not isinstance(self.input_request, InputRequest):
            raise TypeError(f"input_request must be InputRequest, got {type(self.input_request).__name__}")


@dataclass(frozen=True)
class OracleResponse:
    """Result for one oracle request.

    ``folding_output`` is the shared BioIR result schema. ``output`` remains an
    escape hatch for custom provider-native predictions. Search consumes only
    finite scalar entries in ``metrics``.
    """

    request_id: str
    status: OracleStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    folding_output: FoldingOutput | None = None
    output: object | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(self.status, OracleStatus):
            raise TypeError(f"status must be OracleStatus, got {type(self.status).__name__}")
        if self.folding_output is not None and not isinstance(self.folding_output, FoldingOutput):
            raise TypeError(f"folding_output must be FoldingOutput, got {type(self.folding_output).__name__}")


__all__ = [
    "AtomIdentity",
    "EncodedStructure",
    "OracleRequest",
    "OracleResponse",
    "OracleStatus",
    "ResidueIdentity",
    "ResolvedStructure",
]
