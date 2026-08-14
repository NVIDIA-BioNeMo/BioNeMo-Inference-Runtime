# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile

import numpy as np
import pytest

from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.data.writers.cif_writer import CIFWriter
from tests.common.test_utils.data import get_sample_folding_output


class TestCIFWriter:
    """Test suite for CIFWriter class."""

    @pytest.fixture
    def res_type_mapping(self):
        """Fixture providing residue type mapping for openfold2."""
        res_types = get_all_residue_types("openfold2", include_gap=False)
        return dict(enumerate(res_types))

    @pytest.fixture
    def atom_type_mapping(self):
        """Fixture providing atom type mapping for openfold2."""
        atom_types = get_all_atom_types("openfold2")
        return dict(enumerate(atom_types))

    @pytest.fixture
    def sample_folding_output(self):
        """Fixture providing sample folding output data."""
        return get_sample_folding_output()

    @pytest.fixture
    def temp_output_file(self):
        """Fixture providing a temporary output file path."""
        fd, path = tempfile.mkstemp(suffix=".cif")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_initialization_with_both_mappings(self, res_type_mapping, atom_type_mapping, temp_output_file):
        """Test CIFWriter initialization with both residue and atom type mappings."""
        writer = CIFWriter(
            output_path=temp_output_file, res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        assert writer.output_path == temp_output_file
        assert len(writer.res_types) == len(res_type_mapping)
        assert len(writer.atom_types) == len(atom_type_mapping)

    def test_set_output_path(self, res_type_mapping, atom_type_mapping):
        """Test setting output path after initialization."""
        writer = CIFWriter(
            output_path="initial.cif", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        new_path = "new_output.cif"
        writer.set_output_path(new_path)

        assert writer.output_path == new_path

    def test_write_output_basics(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that write() produces valid CIF structure."""
        writer = CIFWriter(
            output_path="test.cif", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        cif_content: str = writer.write(sample_folding_output)

        # Check basic CIF structure
        assert isinstance(cif_content, str)
        assert len(cif_content) > 0

        # CIF files should contain data blocks
        assert "_atom_site" in cif_content.lower()
        assert "atom" in cif_content.lower()

        # Should contain coordinate-related fields
        # mmCIF uses _atom_site.Cartn_x, _atom_site.Cartn_y, _atom_site.Cartn_z
        assert "Cartn" in cif_content

    def test_write_respects_atom_mask(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that write() respects the atom mask."""
        writer = CIFWriter(
            output_path="test.cif", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        # Count non-zero atoms in mask
        atom_mask = sample_folding_output["atom_mask"]
        expected_atoms = np.sum(atom_mask > 0.5)

        cif_content = writer.write(sample_folding_output)

        # The output should have content
        assert len(cif_content) > 0
        assert expected_atoms > 0  # Ensure we have atoms to write

        cif_content_as_lines: list[str] = cif_content.split("\n")
        count_of_ATOM_in_cif = sum([1 for x in cif_content_as_lines if "ATOM" in x])
        count_of_HETATM_in_cif = sum([1 for x in cif_content_as_lines if "HETATM" in x])

        assert count_of_ATOM_in_cif + count_of_HETATM_in_cif == expected_atoms

    def test_write_contains_model_info(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that output contains model information."""
        writer = CIFWriter(
            output_path="test.cif", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        cif_content = writer.write(sample_folding_output)

        # Should contain model-related information
        assert "model" in cif_content.lower()

    def test_write_handles_multi_chain(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that multi-chain structures are handled properly."""
        writer = CIFWriter(
            output_path="test.cif", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        chain_indices = sample_folding_output.get("chain_indices")
        cif_content = writer.write(sample_folding_output)

        if chain_indices is not None:
            unique_chains = len(np.unique(chain_indices))
            # CIF should contain chain information
            assert len(cif_content) > 0
            if unique_chains > 1:
                # Should have entity or asym information for multiple chains
                assert "_entity" in cif_content.lower() or "asym" in cif_content.lower()

    def test_write_file(self, res_type_mapping, atom_type_mapping, sample_folding_output, temp_output_file):
        """Test that write() returns the same string that is written to file."""
        writer = CIFWriter(
            output_path=temp_output_file, res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        cif_content = writer.write(sample_folding_output)

        # Check that file was written
        assert os.path.exists(temp_output_file)

        # Read file and verify content matches returned buffer
        with open(temp_output_file) as f:
            file_content = f.read()

        assert file_content == cif_content


# ---------------------------------------------------------------------------
# Multi-polymer test suite (RNA / DNA / non-polymer ligand chains)
# ---------------------------------------------------------------------------

from tests.common.test_utils.synthetic_folding_outputs import (  # noqa: E402 -- fixtures imported beside the suite that uses them, kept after the single-polymer tests
    dna_only_folding,
    many_chains_folding,
    multi_polymer_folding,
    nonpoly_ligand_folding,
    of3_mappings,
    rna_only_folding,
)


class TestCIFWriterMultiPolymer:
    """Exercises CIFWriter on RNA / DNA / non-polymer / mixed chain inputs."""

    @pytest.fixture
    def writer(self):
        res_map, atom_map = of3_mappings()
        return CIFWriter(res_type_mapping=res_map, atom_type_mapping=atom_map, output_path="test.cif")

    # ----- RNA -------------------------------------------------------

    def test_rna_only_with_mol_types(self, writer):
        """A pure RNA chain (mol_types=1) emits ATOM rows with single-letter
        residue codes (A/G/C/U) from the IHM RNA alphabet."""
        out = writer.write(rna_only_folding(sequence="AGCU", with_residue_names=True, with_mol_types=True))
        # Plain mmCIF ATOM rows
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "RNA chain should produce ATOM rows"
        # Residue names appear as the bare single letter in the mmCIF output
        # via the ihm RNAAlphabet path (not 'RA'/'RG'/...).
        joined = "\n".join(atom_lines)
        assert " A " in joined or "\tA\t" in joined or '"A"' in joined or " A\n" in joined
        # No HETATM for a polymer chain
        assert not any(l.startswith("HETATM") for l in out.split("\n"))

    def test_rna_only_heuristic_classification(self, writer):
        """Without mol_types, the heuristic classifies all-RNA chains as ``rna``."""
        out = writer.write(rna_only_folding(sequence="AGC", with_residue_names=False, with_mol_types=False))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "RNA heuristic should produce ATOM rows"

    # ----- DNA -------------------------------------------------------

    def test_dna_only_with_mol_types(self, writer):
        """A pure DNA chain emits ATOM rows; residue codes come from DNAAlphabet."""
        out = writer.write(dna_only_folding(sequence="ACGT", with_residue_names=True, with_mol_types=True))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "DNA chain should produce ATOM rows"
        # No HETATM for a polymer chain
        assert not any(l.startswith("HETATM") for l in out.split("\n"))

    def test_dna_only_heuristic_classification(self, writer):
        out = writer.write(dna_only_folding(sequence="AT"))
        assert any(l.startswith("ATOM") for l in out.split("\n"))

    # ----- Non-polymer (ligand) --------------------------------------

    def test_nonpoly_emits_hetatm_with_ccd_code(self, writer):
        """A non-polymer chain with ``residue_names=NAG`` emits HETATM rows."""
        out = writer.write(
            nonpoly_ligand_folding(
                atom_names=["C1", "C2", "N2"],
                ccd_code="NAG",
                with_mol_types=True,
            )
        )
        hetatm_lines = [l for l in out.split("\n") if l.startswith("HETATM")]
        assert hetatm_lines, "non-polymer chain should produce HETATM rows"
        # CCD code surfaces in the entity description / hetatm rows
        assert "NAG" in out

    def test_nonpoly_heuristic_falls_back_to_unk(self, writer):
        """Without residue_names or mol_types, all-X residues classify as
        nonpoly and emit HETATM rows with the ``UNK`` fallback."""
        out = writer.write(
            nonpoly_ligand_folding(
                atom_names=["C1", "N2"],
                ccd_code=None,
                with_mol_types=False,
            )
        )
        hetatm_lines = [l for l in out.split("\n") if l.startswith("HETATM")]
        assert hetatm_lines, "all-X chain should still be emitted as HETATM"
        assert "UNK" in out

    def test_mol_types_overrides_heuristic(self, writer):
        """mol_types=3 (nonpoly) should force HETATM even when residue codes
        look polymer-like (here we abuse 'A' restype on a nonpoly chain)."""
        # Build an A-residue but mol_types=3 → writer treats it as nonpoly.
        from tests.common.test_utils.synthetic_folding_outputs import _build_folding, _of3_indices

        res_idx_map, _ = _of3_indices()
        out_fo = _build_folding(
            res_indices=[res_idx_map["A"]],
            residue_indices=[1],
            chain_indices=[0],
            per_residue_atom_names=[["C1"]],
            residue_names=["XYZ"],
            mol_types=[3],
        )
        out = writer.write(out_fo)
        # Expect HETATM because mol_types says nonpoly.
        assert any(l.startswith("HETATM") for l in out.split("\n"))

    # ----- Multi-polymer combo ---------------------------------------

    def test_multi_polymer_emits_all_chain_kinds(self, writer):
        """Protein + RNA + DNA + nonpoly in one structure produces a mix of
        ATOM (polymer chains) and HETATM (nonpoly) rows."""
        out = writer.write(multi_polymer_folding(with_residue_names=True, with_mol_types=True))
        lines = out.split("\n")
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        hetatm_lines = [l for l in lines if l.startswith("HETATM")]
        assert atom_lines, "polymer chains should emit ATOM rows"
        assert hetatm_lines, "nonpoly chain should emit HETATM rows"
        # All four CCD codes should surface (ALA/TYR/RA-RC/DA-DC/NAG)
        for code in ("ALA", "TYR", "NAG"):
            assert code in out, f"expected {code!r} in multi-polymer CIF"

    # ----- Chain ID mapping (CIF supports unlimited via multi-char) --

    def test_uppercase_chain_ids(self, writer):
        """26 chains → asym_ids A..Z (single uppercase letters)."""
        out = writer.write(many_chains_folding(n_chains=26))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert letter in out

    def test_extended_chain_ids_beyond_26(self, writer):
        """Chain index ≥ 26 produces lowercase ascii (CIF can handle this;
        legacy PDB can't)."""
        out = writer.write(many_chains_folding(n_chains=28))
        # The 27th chain should produce 'a', 28th 'b' (per
        # _chain_id_from_index in cif_writer.py).
        assert any(" a " in l or "'a'" in l for l in out.split("\n"))

    # ----- File round-trip -------------------------------------------

    def test_write_to_file_contains_residue_names(self, tmp_path):
        res_map, atom_map = of3_mappings()
        out_path = tmp_path / "multi.cif"
        writer = CIFWriter(res_type_mapping=res_map, atom_type_mapping=atom_map, output_path=str(out_path))
        writer.write(multi_polymer_folding())
        text = out_path.read_text()
        assert "NAG" in text and ("ALA" in text or "ALA " in text)
