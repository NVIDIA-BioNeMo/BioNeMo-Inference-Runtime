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
"""Build canonical custom-oracle requests from RFProteína predictions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import torch

from bionemo_ir._torch.modules.rfproteina.atom_utils import NANOMETER_TO_ANGSTROM
from bionemo_ir._torch.modules.rfproteina.contracts import RFProteinaPredictionOutput
from bionemo_ir._torch.search import MetricEvaluationContext
from bionemo_ir.data.schemas import EncodedStructure, OracleRequest, OracleResponse, OracleStatus
from bionemo_ir.oracles.resolvers import (
    AtomTypeResolver,
    OracleResolutionError,
    ResidueTypeResolver,
    resolve_structure,
)

_DEFAULT_METADATA_KEYS = (
    "element",
    "charge",
    "chirality",
    "molecule_iid",
    "chain_iid",
    "bond_mask",
    "bond_order",
    "is_aromatic",
    "token_bond_path_length",
)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    value_host = value.detach().cpu()
    if value_host.dtype is torch.bfloat16:
        value_host = value_host.to(torch.float32)
    return value_host.numpy()


def _batch_row(value: torch.Tensor, batch_index: int, batch_size: int, name: str) -> torch.Tensor:
    if value.ndim < 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must have batch dimension {batch_size}, got {tuple(value.shape)}")
    return value[batch_index]


class RFProteinaOracleRequestAdapter:
    """Resolve RFProteína's tensor output into custom-oracle requests.

    The adapter deliberately does not infer residue or atom semantics from
    vocabulary width. Callers provide both resolvers. ``metadata_keys`` keeps
    selected source-row arrays available to contextual resolvers; ``None``
    preserves every static feature.
    """

    def __init__(
        self,
        residue_type_resolver: ResidueTypeResolver,
        atom_type_resolver: AtomTypeResolver,
        *,
        metadata_keys: Collection[str] | None = _DEFAULT_METADATA_KEYS,
        atom_type_indices_key: str | None = None,
    ) -> None:
        self.residue_type_resolver = residue_type_resolver
        self.atom_type_resolver = atom_type_resolver
        self.metadata_keys = None if metadata_keys is None else tuple(metadata_keys)
        self.atom_type_indices_key = atom_type_indices_key

    def batch_size(self, prediction: object) -> int:
        """Return the RFProteína decoded batch size."""
        if not isinstance(prediction, RFProteinaPredictionOutput):
            raise TypeError(f"expected RFProteinaPredictionOutput, got {type(prediction).__name__}")
        if prediction.final_coordinates.ndim != 4:
            raise ValueError(
                "RFProteina final_coordinates must have shape [batch, tokens, atom_slots, 3], "
                f"got {tuple(prediction.final_coordinates.shape)}"
            )
        return prediction.final_coordinates.shape[0]

    def _metadata_for_row(
        self,
        static_features: Mapping[str, object],
        batch_index: int,
        batch_size: int,
    ) -> dict[str, object]:
        items = (
            static_features.items()
            if self.metadata_keys is None
            else ((key, static_features[key]) for key in self.metadata_keys if key in static_features)
        )
        metadata: dict[str, object] = {}
        for key, value in items:
            if torch.is_tensor(value):
                row = value[batch_index] if value.ndim >= 1 and value.shape[0] == batch_size else value
                metadata[key] = _to_numpy(row)
            else:
                metadata[key] = value
        return metadata

    def _optional_token_vector(
        self,
        static_features: Mapping[str, object],
        key: str,
        batch_index: int,
        batch_size: int,
        token_count: int,
        included: torch.Tensor,
    ) -> np.ndarray | None:
        value = static_features.get(key)
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise TypeError(f"{key} must be a torch.Tensor")
        row = _batch_row(value, batch_index, batch_size, key)
        if row.shape != (token_count,):
            raise ValueError(f"{key} must have shape [{batch_size}, {token_count}], got {tuple(value.shape)}")
        return _to_numpy(row[included.to(device=row.device)].to(torch.long))

    def _atom_type_indices(
        self,
        static_features: Mapping[str, object],
        batch_index: int,
        batch_size: int,
        token_count: int,
        atom_slots: int,
        included: torch.Tensor,
    ) -> np.ndarray:
        if self.atom_type_indices_key is None:
            count = int(included.sum().item())
            return np.broadcast_to(np.arange(atom_slots, dtype=np.int64), (count, atom_slots)).copy()
        value = static_features.get(self.atom_type_indices_key)
        if not torch.is_tensor(value):
            raise TypeError(f"{self.atom_type_indices_key} must be a torch.Tensor")
        row = _batch_row(value, batch_index, batch_size, self.atom_type_indices_key)
        expected = (token_count, atom_slots)
        if row.shape != expected:
            raise ValueError(
                f"{self.atom_type_indices_key} must have shape [{batch_size}, {token_count}, {atom_slots}], "
                f"got {tuple(value.shape)}"
            )
        return _to_numpy(row[included.to(device=row.device)].to(torch.long))

    def build_requests(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Sequence[OracleRequest | OracleResponse]:
        """Build one resolved request or unsupported response per batch row."""
        batch_size = self.batch_size(prediction)
        if not isinstance(prediction, RFProteinaPredictionOutput):
            raise TypeError(f"expected RFProteinaPredictionOutput, got {type(prediction).__name__}")
        padding_mask = static_features.get("padding_mask")
        if not torch.is_tensor(padding_mask):
            raise TypeError("static_features['padding_mask'] must be a torch.Tensor")

        coordinates = prediction.final_coordinates
        atom_mask = prediction.final_atom_mask
        residue_type_indices = prediction.final_residue_types
        if residue_type_indices.ndim != 2:
            raise ValueError(
                f"final_residue_types must have shape [batch, tokens], got {tuple(residue_type_indices.shape)}"
            )
        if atom_mask.ndim != 3:
            raise ValueError(
                f"final_atom_mask must have shape [batch, tokens, atom_slots], got {tuple(atom_mask.shape)}"
            )
        expected_coordinate_shape = (*residue_type_indices.shape, atom_mask.shape[-1], 3)
        if coordinates.shape != expected_coordinate_shape:
            raise ValueError(
                f"final_coordinates must have shape {expected_coordinate_shape}, got {tuple(coordinates.shape)}"
            )
        if atom_mask.shape != expected_coordinate_shape[:-1]:
            raise ValueError(
                f"final_atom_mask must have shape {expected_coordinate_shape[:-1]}, got {tuple(atom_mask.shape)}"
            )
        if padding_mask.shape != residue_type_indices.shape:
            raise ValueError(
                f"padding_mask must have shape {tuple(residue_type_indices.shape)}, got {tuple(padding_mask.shape)}"
            )

        _, token_count = residue_type_indices.shape
        atom_slots = atom_mask.shape[-1]
        if context.request_ids and len(context.request_ids) != batch_size:
            raise ValueError(f"metric context has {len(context.request_ids)} request IDs for a batch of {batch_size}")
        checkpoint = "posthoc" if context.checkpoint is None else str(context.checkpoint)
        request_ids = context.request_ids or tuple(
            f"rfproteina:{checkpoint}:{batch_index}" for batch_index in range(batch_size)
        )

        items: list[OracleRequest | OracleResponse] = []
        for batch_index, request_id in enumerate(request_ids):
            included = padding_mask[batch_index].to(device=coordinates.device, dtype=torch.bool)
            if not bool(included.any()):
                items.append(
                    OracleResponse(
                        request_id=request_id,
                        status=OracleStatus.UNSUPPORTED,
                        error="RFProteina prediction contains no non-padding tokens",
                    )
                )
                continue

            encoded = EncodedStructure(
                coordinates=(_to_numpy(coordinates[batch_index, included].to(torch.float32)) * NANOMETER_TO_ANGSTROM),
                atom_mask=_to_numpy(atom_mask[batch_index, included].to(torch.bool)),
                residue_type_indices=_to_numpy(residue_type_indices[batch_index, included].to(torch.long)),
                atom_type_indices=self._atom_type_indices(
                    static_features,
                    batch_index,
                    batch_size,
                    token_count,
                    atom_slots,
                    included,
                ),
                source_token_indices=_to_numpy(
                    torch.arange(token_count, device=included.device, dtype=torch.long)[included]
                ),
                token_type_indices=self._optional_token_vector(
                    static_features,
                    "token_type",
                    batch_index,
                    batch_size,
                    token_count,
                    included,
                ),
                chain_indices=self._optional_token_vector(
                    static_features,
                    "chain_iid",
                    batch_index,
                    batch_size,
                    token_count,
                    included,
                ),
                metadata=self._metadata_for_row(static_features, batch_index, batch_size),
            )
            try:
                resolved = resolve_structure(
                    encoded,
                    self.residue_type_resolver,
                    self.atom_type_resolver,
                )
            except OracleResolutionError as error:
                items.append(
                    OracleResponse(
                        request_id=request_id,
                        status=OracleStatus.UNSUPPORTED,
                        error=str(error),
                    )
                )
                continue
            items.append(
                OracleRequest(
                    request_id=request_id,
                    structure=resolved,
                    tags=context.tags,
                    metadata={"source": "rfproteina", "checkpoint": context.checkpoint},
                )
            )
        return items


__all__ = ["RFProteinaOracleRequestAdapter"]
