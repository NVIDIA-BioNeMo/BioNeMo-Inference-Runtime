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
"""Vectorised OF3 MSA row encoding matches the per-character dispatch."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3 import feature_context
from bionemo_ir.pipeline.models.openfold3.const import (
    GAP_IDX,
    MOL_TYPE_DNA,
    MOL_TYPE_LIGAND,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    NUM_MSA_CLASSES,
)
from bionemo_ir.pipeline.models.openfold3.feature_generators import (
    MsaFeatureGenerator,
    _deletion_rows,
    _encode_msa_rows,
    _pad_columns,
)

ALPHABET = list("ACDEFGHIKLMNPQRSTVWYXBZUO-.acgtnx")


def _reference_rows(seqs: list[str], mol_type: int, width: int) -> np.ndarray:
    rows = np.full((len(seqs), width), GAP_IDX, dtype=np.int64)
    for i, seq in enumerate(seqs):
        for j in range(min(len(seq), width)):
            rows[i, j] = feature_context._resolve_msa_char(seq[j], mol_type)
    return rows


@pytest.fixture(autouse=True)
def _reset_warning_cache() -> None:
    feature_context._seen_unknown_msa_chars.clear()


@pytest.mark.parametrize("mol_type", [MOL_TYPE_PROTEIN, MOL_TYPE_RNA, MOL_TYPE_DNA, MOL_TYPE_LIGAND, 7])
def test_encode_rows_matches_per_char_dispatch(mol_type: int) -> None:
    rng = np.random.default_rng(mol_type)
    seqs = ["".join(rng.choice(ALPHABET, size=int(n))) for n in rng.integers(0, 40, size=64)]
    seqs += ["", "-.-.", "ACéEF", "ΔCDE"]
    width = 30
    expected = _reference_rows(seqs, mol_type, width)
    feature_context._seen_unknown_msa_chars.clear()
    assert np.array_equal(_encode_msa_rows(seqs, mol_type, width), expected)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Unknown RNA MSA char" in r.getMessage()]


def test_rna_unknown_chars_warn_like_per_char_dispatch(caplog: pytest.LogCaptureFixture) -> None:
    seqs = ["ACGUNXn", "NNxxZz", "acgu-.J"]
    logger = "bionemo_ir.pipeline.models.openfold3.feature_context"
    with caplog.at_level(logging.WARNING, logger=logger):
        expected_rows = _reference_rows(seqs, MOL_TYPE_RNA, 7)
    expected_warnings = _warnings(caplog)
    assert expected_warnings, "the reference must warn on at least one character"
    caplog.clear()
    feature_context._seen_unknown_msa_chars.clear()
    with caplog.at_level(logging.WARNING, logger=logger):
        rows = _encode_msa_rows(seqs, MOL_TYPE_RNA, 7)
    assert np.array_equal(rows, expected_rows)
    assert _warnings(caplog) == expected_warnings


def _reference_deletions(raw: str) -> list[int]:
    counts, run = [], 0
    for char in raw:
        if char.islower():
            run += 1
        else:
            counts.append(run)
            run = 0
    return counts


def test_deletion_rows_match_per_row_reference() -> None:
    raw = ["", "ACDEF", "aaACdeF-.gG", "abc", "AébÉC", "-.-.", "ACDEFGHIKLMNPQRSTVWYxxxACDEFGHIKLMNPQRSTVWY"]
    seqs = ["".join(c for c in r if not c.islower()) for r in raw] + ["ACD", "AC"]
    seqs[1] = "AC"  # Shorter aligned row caps its deletion columns.
    width = max(map(len, seqs))
    expected = np.zeros((len(seqs), width), dtype=np.int64)
    for i, seq in enumerate(seqs):
        counts = _reference_deletions(raw[i] if i < len(raw) else seq)
        n = min(len(seq), len(counts))
        expected[i, :n] = counts[:n]
    assert np.array_equal(_deletion_rows(raw, seqs, width), expected)


def test_pad_columns_crops_and_pads() -> None:
    rows = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert _pad_columns(rows, 3, GAP_IDX) is rows
    padded = _pad_columns(rows, 5, GAP_IDX)
    assert padded.tolist() == [[0, 1, 2, GAP_IDX, GAP_IDX], [3, 4, 5, GAP_IDX, GAP_IDX]]
    assert _pad_columns(rows, 2, 0).tolist() == [[0, 1], [3, 4]]


def _context(chains: list[tuple[str, int, int]], msa: list, paired: list) -> dict:
    chain_ids: list[str] = []
    mol_types: list[int] = []
    for cid, mol_type, n_res in chains:
        chain_ids += [cid] * n_res
        mol_types += [mol_type] * n_res
    return {
        "structure": {"n_tokens": len(chain_ids), "token_chain_ids": chain_ids, "token_mol_types": mol_types},
        "msa_per_chain": msa,
        "paired_msa_per_chain": paired,
    }


def test_generator_query_dedup_crop_and_profile() -> None:
    main = {"sequences": ["ACDEFG", "FGHIKL", "KLMNPQ", "FGHIKL"], "raw": ["ACDEFG", "FGhhHIKL", "KLMNPQ", "FGHIKL"]}
    paired = {"sequences": ["FGHIKL", "PQRSTV"]}
    feats = MsaFeatureGenerator()({}, _context([("A", MOL_TYPE_PROTEIN, 4)], [main], [paired]))

    ref = _reference_rows(main["sequences"], MOL_TYPE_PROTEIN, 6)
    # Query is file row 0 cropped; rows equal to a full-width paired row are dropped.
    expected_rows = np.stack([ref[0, :4], ref[0, :4], ref[2, :4]])
    msa_idx = feats["msa"].argmax(dim=-1).numpy()
    assert np.array_equal(msa_idx, expected_rows)
    assert feats["msa"].dtype == torch.int32 and feats["msa"].shape == (3, 4, NUM_MSA_CLASSES)
    assert feats["has_deletion"].tolist() == [[0.0] * 4] * 3
    # deletion_mean and profile use all four full-width rows, then crop.
    dels = np.zeros((4, 6))
    dels[1, 2] = 2
    assert torch.allclose(feats["deletion_mean"], torch.tensor(dels.mean(axis=0)[:4], dtype=torch.float32))
    assert feats["profile"].shape == (4, NUM_MSA_CLASSES)
    assert torch.allclose(feats["profile"].sum(dim=-1), torch.ones(4))
    assert feats["num_paired_seqs"].tolist() == [1]
    assert feats["msa_mask"].tolist() == [[1.0] * 4] * 3


def test_generator_pads_short_rows_and_broadcasts_shared_msa() -> None:
    shared = {"sequences": ["ACD", "KL"], "raw": ["ACD", "KL"]}
    feats = MsaFeatureGenerator()(
        {}, _context([("A", MOL_TYPE_PROTEIN, 4), ("B", MOL_TYPE_PROTEIN, 4)], [shared, shared], [])
    )
    msa_idx = feats["msa"].argmax(dim=-1).numpy()
    ref = _reference_rows(shared["sequences"], MOL_TYPE_PROTEIN, 4)
    assert np.array_equal(msa_idx[:, :4], np.stack([ref[0], ref[0], ref[1]]))
    assert np.array_equal(msa_idx[:, 4:], msa_idx[:, :4])
    assert msa_idx[2, 2] == GAP_IDX and msa_idx[2, 3] == GAP_IDX


def test_generator_without_msa_emits_gap_row_and_zero_profile() -> None:
    feats = MsaFeatureGenerator()({}, _context([("A", MOL_TYPE_PROTEIN, 3)], [None], []))
    assert feats["msa"].shape == (1, 3, NUM_MSA_CLASSES)
    assert feats["msa"].argmax(dim=-1).tolist() == [[GAP_IDX] * 3]
    assert torch.equal(feats["profile"], torch.zeros(3, NUM_MSA_CLASSES))
    assert torch.equal(feats["deletion_mean"], torch.zeros(3))
