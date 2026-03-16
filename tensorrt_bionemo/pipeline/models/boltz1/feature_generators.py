# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from tensorrt_bionemo.pipeline.base import FeatureGeneratorBase
from tensorrt_bionemo.pipeline.models.boltz2.const import (max_msa_seqs,
                                                           max_paired_seqs)
from tensorrt_bionemo.pipeline.models.boltz2.feature_generators import _row
from tensorrt_bionemo.pipeline.models.boltz2.featurizer import (
    process_chain_feature_constraints, process_residue_constraint_features)

from .featurizer import (process_atom_features, process_msa_features,
                         process_token_features)


class Boltz1TokenFeatureGenerator(FeatureGeneratorBase):

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        return process_token_features(row["tokens"], row["token_bonds"],
                                      row["structure"])


class Boltz1AtomFeatureGenerator(FeatureGeneratorBase):

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        return process_atom_features(row["structure"], row["tokens"],
                                     row["molecules"])


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
            max_seqs=max_msa_seqs,
            max_paired=max_paired_seqs)


class Boltz1ResidueConstraintFeatureGenerator(FeatureGeneratorBase):

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return process_residue_constraint_features()


class Boltz1ChainConstraintFeatureGenerator(FeatureGeneratorBase):

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        structure = row.get("structure")
        return process_chain_feature_constraints(structure)
