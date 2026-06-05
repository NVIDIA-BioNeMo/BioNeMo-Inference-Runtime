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
"""Boltz2 feature generators: one per internal step of compute_features.

Per base.py: "Feature Generator: Generates feature tensors based on context
tensors." Each generator reads context["_row"] (structure, tokens, molecules,
MSA) and/or batch (outputs of prior generators) and returns one group of
tensors. The merger combines them; the collator produces the final feature set.
"""

from typing import Any

import numpy as np
import torch
from rdkit import Chem

from tensorrt_bionemo.pipeline.base import FeatureGeneratorBase

from .const import Structure, Token, TokenBond, max_msa_seqs, max_paired_seqs
from .featurizer import (load_dummy_templates_features, process_atom_features,
                         process_chain_feature_constraints,
                         process_contact_feature_constraints,
                         process_ensemble_features, process_msa_features,
                         process_residue_constraint_features,
                         process_token_features)


def _row(context: dict[str, Any]) -> dict[str, Any]:
    """Get the raw row from context, deserialising dataclass dicts on first access."""
    row = context.get("_row") or context

    if "structure" in row and isinstance(row["structure"], dict):
        row["structure"] = Structure.from_dict(row["structure"])

    if "tokens" in row and row.get("tokens") is not None:
        tokens = row["tokens"]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], dict):
            row["tokens"] = [Token.from_dict(t) for t in tokens]

    if "token_bonds" in row and row.get("token_bonds") is not None:
        tbs = row["token_bonds"]
        if isinstance(tbs, np.ndarray):
            tbs = tbs.tolist()
        if tbs and isinstance(tbs[0], dict):
            row["token_bonds"] = [TokenBond.from_dict(tb) for tb in tbs]

    if "molecules" in row and row.get("molecules") is not None:
        mols = row["molecules"]
        if mols and isinstance(next(iter(mols.values())), (bytes, bytearray)):
            row["molecules"] = {k: Chem.Mol(v) for k, v in mols.items()}

    return row


class Boltz2TokenFeatureGenerator(FeatureGeneratorBase):
    """Generates token-level feature tensors from structure, tokens, token_bonds."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        structure = row["structure"]
        tokens = row["tokens"]
        token_bonds = row["token_bonds"]
        return process_token_features(
            tokens,
            token_bonds,
            structure,
            override_method=None,
            max_tokens=None,
        )


class Boltz2EnsembleFeatureGenerator(FeatureGeneratorBase):
    """Generates ensemble indices (single conformer)."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return process_ensemble_features()


class Boltz2AtomFeatureGenerator(FeatureGeneratorBase):
    """Generates atom-level feature tensors; requires ensemble_ref_idxs from batch."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        structure = row["structure"]
        tokens = row["tokens"]
        molecules = row["molecules"]
        ensemble_ref_idxs = batch["ensemble_ref_idxs"]
        return process_atom_features(
            structure,
            tokens,
            molecules,
            ensemble_ref_idxs=ensemble_ref_idxs,
            max_tokens=None,
        )


class Boltz2MsaFeatureGenerator(FeatureGeneratorBase):
    """Generates MSA feature tensors from tokens and msa_parsed_per_chain."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        tokens = row["tokens"]
        msa_parsed_per_chain = row["msa_parsed_per_chain"]
        paired_msa_per_chain = row.get("paired_msa_per_chain")
        return process_msa_features(
            tokens,
            msa_parsed_per_chain=msa_parsed_per_chain,
            paired_msa_per_chain=paired_msa_per_chain,
            max_seqs=max_msa_seqs,
            max_paired=max_paired_seqs,
            max_tokens=None,
        )


class Boltz2TemplateFeatureGenerator(FeatureGeneratorBase):
    """Generates dummy template features; num_tokens from batch (e.g. token_index shape)."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        num_tok = batch["token_index"].shape[0]
        return load_dummy_templates_features(1, num_tok)


class Boltz2ResidueConstraintFeatureGenerator(FeatureGeneratorBase):
    """Generates residue constraint feature tensors from per-residue RDKit data."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        constraints = row.get("residue_constraints") or {}
        return process_residue_constraint_features(constraints=constraints)


class Boltz2ChainConstraintFeatureGenerator(FeatureGeneratorBase):
    """Generates chain constraint feature tensors."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _row(context)
        structure = row.get("structure")
        return process_chain_feature_constraints(structure)


class Boltz2ContactConstraintFeatureGenerator(FeatureGeneratorBase):
    """Generates contact constraint feature tensors."""

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return process_contact_feature_constraints()
