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

"""Boltz1 feature generators.

Boltz1 has fewer feature groups than Boltz2:
- Token features (with pocket_feature, no method/modified/affinity/contact)
- Atom features (no ensemble, no chirality/backbone/bfactor/plddt)
- MSA features (one-hot msa)
- Residue constraint features (empty)
- Chain constraint features
"""

from typing import Any

import torch

from bionemo_ir.pipeline.base import FeatureGeneratorBase
from bionemo_ir.pipeline.models.boltz2.const import max_paired_seqs
from bionemo_ir.pipeline.models.boltz2.feature_generators import _row
from bionemo_ir.pipeline.models.boltz2.featurizer import (
    process_chain_feature_constraints,
    process_residue_constraint_features,
)

from .featurizer import process_atom_features, process_msa_features, process_token_features

# OSS boltz1 inference (boltz.data.module.inference.BoltzInferenceDataModule)
# calls the featurizer with ``max_seqs=const.max_msa_seqs`` = 16384. The boltz2
# TRT const sets ``max_msa_seqs=8192`` (boltz2's predict-path default), which
# would truncate boltz1 MSAs deeper than OSS. Use boltz1's 16384 here.
BOLTZ1_MAX_MSA_SEQS = 16384


class Boltz1TokenFeatureGenerator(FeatureGeneratorBase):
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        return process_token_features(row["tokens"], row["token_bonds"], row["structure"])


class Boltz1AtomFeatureGenerator(FeatureGeneratorBase):
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        return process_atom_features(row["structure"], row["tokens"], row["molecules"])


class Boltz1MsaFeatureGenerator(FeatureGeneratorBase):
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        return process_msa_features(
            row["tokens"],
            msa_parsed_per_chain=row["msa_parsed_per_chain"],
            paired_msa_per_chain=row.get("paired_msa_per_chain"),
            max_seqs=BOLTZ1_MAX_MSA_SEQS,
            max_paired=max_paired_seqs,
        )


class Boltz1ResidueConstraintFeatureGenerator(FeatureGeneratorBase):
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        # Upstream boltz1 inference runs with compute_constraint_features=True
        # (``BoltzInferenceDataset.__getitem__`` in
        # ``boltz/data/module/inference.py``), so ligand RDKit constraints
        # (rdkit_bounds / chiral / stereo / planar) must be populated and fed
        # to the diffusion steering potentials. The constraints dict is built
        # by the inherited Boltz2 context generator and lives in the row.
        row = _row(context)
        constraints = row.get("residue_constraints") or {}
        return process_residue_constraint_features(constraints=constraints)


class Boltz1ChainConstraintFeatureGenerator(FeatureGeneratorBase):
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        structure = row.get("structure")
        return process_chain_feature_constraints(structure)
