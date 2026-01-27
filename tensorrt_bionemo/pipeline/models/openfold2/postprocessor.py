# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.pipeline.base import PostProcessorBase


class PostProcessorConfig(BaseModel):
    subtract_plddt: bool = False
    multimer_ri_gap: int = 200


class PostProcessor(PostProcessorBase):

    def __init__(self, config: Optional[BaseModel] = None) -> None:
        super().__init__(config)
        if self.config is None:
            self.config = PostProcessorConfig()

    def _normalize_residue_indices(self,
                                   residue_index: np.ndarray) -> np.ndarray:
        """Normalize residue indices for multi-chain FASTAs.

        Converts global residue indices (with gaps between chains) to
        per-chain residue indices starting from 0 for each chain.

        The algorithm detects chain boundaries by examining when the chain ID
        (computed as (residue_index[i] - i) // multimer_ri_gap) changes.

        Args:
            residue_index: Global residue indices with multimer_ri_gap between chains

        Returns:
            Normalized residue indices (0-based for each chain)
        """
        if len(residue_index) == 0:
            return residue_index

        # Calculate which chain each residue belongs to
        position_in_sequence = np.arange(residue_index.shape[0])
        chain_ids = ((residue_index - position_in_sequence) /
                     self.config.multimer_ri_gap).astype(np.int64)

        # Detect chain transitions
        chain_changes = np.concatenate([[True], chain_ids[1:]
                                        != chain_ids[:-1]])

        # Calculate offsets at chain boundaries: position + chain_id * multimer_ri_gap
        chain_offsets_at_boundaries = (
            position_in_sequence +
            chain_ids * self.config.multimer_ri_gap) * chain_changes

        # Forward-fill: propagate each offset to all positions until the next change
        # Use cummax to get the most recent chain offset for each position
        offsets = np.maximum.accumulate(chain_offsets_at_boundaries)

        # Normalize residue indices within each chain
        return residue_index - offsets

    def _get_chain_indices(self, np_batch: dict[str,
                                                np.ndarray]) -> np.ndarray:
        """Extract or infer chain indices from batch data.

        Args:
            np_batch: Batch dictionary containing residue data

        Returns:
            Array of chain indices (0-based) for each residue
        """
        if 'asym_id' in np_batch:
            # Use explicit asymmetric unit IDs if available (convert to 0-based)
            return np_batch["asym_id"] - 1
        else:
            # Default to single chain (all zeros)
            return np.zeros_like(np_batch["aatype"])

    def __call__(self, batch: dict[str, Any],
                 output: dict[str, Any]) -> FoldingOutput:
        np_batch = {}
        # Get only the last element of the tensor, we don't need all the recycling steps
        for k in [
                "residue_index", "aatype", "asym_id", "final_atom_positions",
                "final_atom_mask"
        ]:
            if k in batch:
                np_batch[k] = np.array(batch[k][..., -1].cpu())
        plddt = output["plddt"].cpu().numpy()
        plddt_b_factors = np.repeat(plddt[..., None],
                                    rc.atom_type_num,
                                    axis=-1)

        if self.config.subtract_plddt:
            plddt_b_factors = 100 - plddt_b_factors

        # Normalize residue indices for multi-chain sequences
        normalized_residue_index = self._normalize_residue_indices(
            np_batch["residue_index"])

        # Extract chain indices from batch data
        chain_indices = self._get_chain_indices(np_batch)

        return FoldingOutput(
            residue_types=np_batch["aatype"],
            atom_positions=output["final_atom_positions"].cpu().numpy(),
            atom_mask=output["final_atom_mask"].cpu().numpy(),
            residue_indices=normalized_residue_index + 1,
            b_factors=plddt_b_factors,
            chain_indices=chain_indices)
