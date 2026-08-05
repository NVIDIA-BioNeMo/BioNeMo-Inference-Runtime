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

"""Unit tests for the Boltz-2 template alignment core.

Cover the pure-Python sequence-alignment + chain-assignment functions
(``get_global_alignment_score`` / ``get_local_alignments`` /
``get_template_records_from_{search,matching}``) that decide which template
chain maps to which query chain and at what offset. Biopython + SciPy only
(no CCD/GPU/golden), so they run fast in CI. Full 12-tensor featurization is
covered separately, against the upstream Boltz implementation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tensorrt_bionemo.pipeline.models.boltz2.template_logic import (
    Alignment,
    TemplateMatch,
    _load_modified_mol,
    global_alignment_score,
    local_alignments,
    parse_template_structure,
    template_records_from_matching,
    template_records_from_search,
    tokenize_template,
)

_REPO = Path(__file__).resolve().parents[4]
# Small committed template CIF (chain A, 211 protein residues); parses without
# any CCD mol data (mol_dir defaults to None for standard protein residues).
_CIF_8WLE = _REPO / "examples/data/samples/monomers/templates/8wle_A.cif"

# Two deterministic, dissimilar protein sequences (real amino acids).
SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
SEQ_B = "GSHMWDEACFPQRNTYLVIKGSHMWDEACFPQRNTYLVIKAAEEDDKKRRTT"
_HEAD, _TAIL = 15, 8  # offset-case truncation


# --------------------------------------------------------------------------- #
# global_alignment_score
# --------------------------------------------------------------------------- #
def test_global_alignment_score_self_exceeds_cross():
    """A sequence scores higher against itself than against a distinct one."""
    self_score = global_alignment_score(SEQ_A, SEQ_A)
    cross_score = global_alignment_score(SEQ_A, SEQ_B)
    assert self_score > cross_score


def test_global_alignment_score_is_symmetric():
    assert global_alignment_score(SEQ_A, SEQ_B) == global_alignment_score(SEQ_B, SEQ_A)


# --------------------------------------------------------------------------- #
# local_alignments
# --------------------------------------------------------------------------- #
def test_local_alignments_self_is_full_ungapped_block():
    """Self-alignment yields a single block spanning the whole sequence."""
    blocks = local_alignments(SEQ_A, SEQ_A)
    assert len(blocks) >= 1
    a = blocks[0]
    assert (a.query_st, a.query_en) == (0, len(SEQ_A))
    assert (a.template_st, a.template_en) == (0, len(SEQ_A))


def test_local_alignments_recovers_offset():
    """A query cut from the middle of the template aligns back at that offset."""
    query = SEQ_A[_HEAD : len(SEQ_A) - _TAIL]
    a = local_alignments(query, SEQ_A)[0]
    assert a.query_st == 0
    assert a.template_st == _HEAD
    assert a.template_st - a.query_st == _HEAD  # the offset the featurizer uses
    assert a.query_en - a.query_st == len(query)


# --------------------------------------------------------------------------- #
# template_records_from_matching (explicit 1:1 chain mapping)
# --------------------------------------------------------------------------- #
def test_records_from_matching_offset_and_flags_propagate():
    query = SEQ_A[_HEAD : len(SEQ_A) - _TAIL]
    recs = template_records_from_matching(
        template_id="tmpl",
        chain_ids=["A"],
        sequences={"A": query},
        template_chain_ids=["A"],
        template_sequences={"A": SEQ_A},
        force=True,
        threshold=0.5,
    )
    assert len(recs) == 1
    r = recs[0]
    assert r.name == "tmpl"
    assert r.query_chain == "A" and r.template_chain == "A"
    assert r.template_st - r.query_st == _HEAD
    assert r.force is True
    assert r.threshold == 0.5


def test_records_from_matching_default_threshold_is_inf():
    recs = template_records_from_matching(
        template_id="t",
        chain_ids=["A"],
        sequences={"A": SEQ_A},
        template_chain_ids=["A"],
        template_sequences={"A": SEQ_A},
    )
    assert recs[0].threshold == float("inf")
    assert recs[0].force is False


def test_records_from_matching_multichain_pairs_by_position():
    """matching zips the chain lists positionally (no re-assignment)."""
    recs = template_records_from_matching(
        template_id="t",
        chain_ids=["A", "B"],
        sequences={"A": SEQ_A, "B": SEQ_B},
        template_chain_ids=["TA", "TB"],
        template_sequences={"TA": SEQ_A, "TB": SEQ_B},
    )
    mapping = {(r.query_chain, r.template_chain) for r in recs}
    assert ("A", "TA") in mapping
    assert ("B", "TB") in mapping


# --------------------------------------------------------------------------- #
# template_records_from_search (Hungarian chain assignment)
# --------------------------------------------------------------------------- #
def test_records_from_search_assigns_by_sequence_not_order():
    """Templates listed in swapped order are still matched to the right query
    chain via linear_sum_assignment on the global-score matrix."""
    recs = template_records_from_search(
        template_id="t",
        chain_ids=["A", "B"],
        sequences={"A": SEQ_A, "B": SEQ_B},
        # T1 carries B's sequence, T2 carries A's — order is deliberately swapped.
        template_chain_ids=["T1", "T2"],
        template_sequences={"T1": SEQ_B, "T2": SEQ_A},
    )
    assignment = {r.query_chain: r.template_chain for r in recs}
    assert assignment["A"] == "T2"  # A matched to the chain holding SEQ_A
    assert assignment["B"] == "T1"  # B matched to the chain holding SEQ_B


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
def test_templatematch_is_frozen():
    m = TemplateMatch(
        name="t", query_chain="A", query_st=0, query_en=1, template_chain="A", template_st=0, template_en=1
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.query_st = 5  # frozen dataclass -> mutation forbidden


def test_alignment_fields():
    a = Alignment(query_st=1, query_en=10, template_st=3, template_en=12)
    assert (a.query_st, a.query_en, a.template_st, a.template_en) == (1, 10, 3, 12)


# --------------------------------------------------------------------------- #
# _load_modified_mol: the residue name comes from an untrusted CIF and is used
# to build a pickle path, so it must reject anything that could traverse out of
# mol_dir into an attacker-controlled pickle.
# --------------------------------------------------------------------------- #
def test_load_modified_mol_rejects_unsafe_names(tmp_path):
    import pickle

    # Plant a pickle one directory above mol_dir; a traversal name must not reach it.
    (tmp_path.parent / "evil.pkl").write_bytes(pickle.dumps({"x": 1}))
    for unsafe in ["../evil", "..", "a/b", "a.b", r"a\b", "toolong6", "MSE/../x"]:
        assert _load_modified_mol(unsafe, str(tmp_path)) is None
    # A well-formed CCD id whose file is absent returns None (no error).
    assert _load_modified_mol("MSE", str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# gemmi CIF parse + tokenization (committed fixture, no CCD)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _CIF_8WLE.is_file(), reason="8wle_A.cif fixture missing")
def test_parse_template_structure_chain_and_token_count():
    struct, seqs = parse_template_structure(str(_CIF_8WLE), fmt="cif")
    assert set(seqs) == {"A"}
    assert len(seqs["A"]) == 211
    # One token per protein residue.
    assert len(tokenize_template(struct)) == 211


@pytest.mark.skipif(not _CIF_8WLE.is_file(), reason="8wle_A.cif fixture missing")
def test_parse_then_self_align_is_full_coverage_offset_zero():
    """End-to-end parse -> align: a chain self-templates at offset 0, full span."""
    _, seqs = parse_template_structure(str(_CIF_8WLE), fmt="cif")
    seq = seqs["A"]
    recs = template_records_from_matching(
        template_id="8wle",
        chain_ids=["A"],
        sequences={"A": seq},
        template_chain_ids=["A"],
        template_sequences={"A": seq},
    )
    assert len(recs) == 1
    r = recs[0]
    assert r.template_st - r.query_st == 0
    assert r.query_en - r.query_st == len(seq)
