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
"""Unit tests for the shared writer helpers in ``base_writer``.

The integration tests in ``test_cif_writer`` / ``test_pdb_writer`` exercise
``_classify_chain`` and ``_IHM_REMAP`` indirectly. These tests pin down
the corner cases — empty sets, padding-only sets, mixed protein with X,
and the chain-id base-N scheme — so a regression in the helper itself
surfaces directly without having to interpret a writer's output.
"""

from __future__ import annotations

import pytest

from tensorrt_bionemo.data.writers.base_writer import (
    _IHM_REMAP,
    _MOL_TYPE_TO_KIND,
    _classify_chain,
)
from tensorrt_bionemo.data.writers.cif_writer import _chain_id_from_index


class TestClassifyChain:
    """Cover all branches of ``_classify_chain``."""

    def test_pure_rna(self):
        assert _classify_chain(("RA", "RG", "RC", "RU")) == "rna"

    def test_pure_rna_with_unknown(self):
        """``RX`` is the unknown-nucleotide stand-in and still classifies as RNA."""
        assert _classify_chain(("RA", "RX", "RG")) == "rna"

    def test_pure_dna(self):
        assert _classify_chain(("DA", "DG", "DC", "DT")) == "dna"

    def test_pure_dna_with_unknown(self):
        assert _classify_chain(("DA", "DX", "DT")) == "dna"

    def test_all_x_is_nonpoly(self):
        """Boltz2 NONPOLYMER tokenizer emits per-atom residues all of type ``X``."""
        assert _classify_chain(("X", "X", "X")) == "nonpoly"

    def test_single_x_is_nonpoly(self):
        assert _classify_chain(("X",)) == "nonpoly"

    def test_protein_default(self):
        assert _classify_chain(("A", "G", "L", "Y")) == "protein"

    def test_protein_with_x_is_still_protein(self):
        """A real protein chain with one UNK residue is NOT a ligand chain."""
        assert _classify_chain(("A", "X", "L")) == "protein"

    def test_mixed_rna_dna_is_protein_fallback(self):
        """RNA+DNA in the same chain falls through both purity checks; the
        writer treats it as a protein chain (most permissive bucket)."""
        assert _classify_chain(("RA", "DA")) == "protein"

    def test_only_pad_falls_through_to_protein(self):
        """All-pad residues leave an empty set after _PAD_RESNAMES subtraction;
        empty cannot be a subset-of-anything, so the fallthrough is ``protein``."""
        assert _classify_chain(("-", "<PAD>", "-")) == "protein"

    def test_rna_plus_pad_is_rna(self):
        """Pad residues are filtered before the subset check."""
        assert _classify_chain(("-", "RA", "RG")) == "rna"


class TestIhmRemap:
    """``_IHM_REMAP`` maps internal short codes → CIF/PDB-standard codes."""

    def test_protein_unknown_becomes_unk(self):
        assert _IHM_REMAP["X"] == "UNK"

    def test_rna_short_codes_become_single_letters(self):
        assert _IHM_REMAP["RA"] == "A"
        assert _IHM_REMAP["RG"] == "G"
        assert _IHM_REMAP["RC"] == "C"
        assert _IHM_REMAP["RU"] == "U"

    def test_rna_unknown_becomes_n(self):
        assert _IHM_REMAP["RX"] == "N"

    def test_dna_unknown_becomes_dn(self):
        assert _IHM_REMAP["DX"] == "DN"

    def test_dna_canonical_codes_passthrough(self):
        """DA/DG/DC/DT are already CIF/PDB-standard — not in the remap."""
        for code in ("DA", "DG", "DC", "DT"):
            assert code not in _IHM_REMAP


class TestMolTypeToKind:
    """Canonical mol-type ints → chain-classification labels."""

    def test_all_four_kinds_present(self):
        assert _MOL_TYPE_TO_KIND == {
            0: "protein",
            1: "rna",
            2: "dna",
            3: "nonpoly",
        }


class TestChainIdFromIndex:
    """CIF asym_id base-N scheme: A-Z, a-z, then AA, AB, ...; unbounded."""

    @pytest.mark.parametrize(
        "idx,expected",
        [
            (0, "A"),
            (1, "B"),
            (25, "Z"),
            (26, "a"),
            (51, "z"),
            (52, "AA"),
            (53, "AB"),
            (52 + 25, "AZ"),
            (52 + 26, "Aa"),
            # Spot-check a far value just to confirm there's no hardcoded cap
            # (the function is supposed to grow as needed).
            (52 + 52, "BA"),
        ],
    )
    def test_index_to_asym_id(self, idx, expected):
        assert _chain_id_from_index(idx) == expected

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _chain_id_from_index(-1)
