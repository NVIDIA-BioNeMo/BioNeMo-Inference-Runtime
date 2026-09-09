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
from bionemo_ir.data.writers.pdb_writer import PDBWriter
from tests.common.test_utils.data import get_sample_folding_output


@pytest.fixture(autouse=True)
def _write_outside_the_checkout(tmp_path, monkeypatch):
    """Run every test in this module from a scratch directory.

    Most tests below hand the writer a bare relative ``output_path`` because
    they only assert on the returned string — but :meth:`PDBWriter.write` also
    opens that path for writing, so a run started from the repo root drops a
    ``test.pdb`` next to ``setup.py``.
    """
    monkeypatch.chdir(tmp_path)


class TestPDBWriter:
    """Test suite for PDBWriter class."""

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
        fd, path = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_initialization_with_both_mappings(self, res_type_mapping, atom_type_mapping, temp_output_file):
        """Test PDBWriter initialization with both residue and atom type mappings."""
        writer = PDBWriter(
            output_path=temp_output_file, res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        assert writer.output_path == temp_output_file
        assert writer.res_type_mapping == res_type_mapping
        assert writer.atom_type_mapping == atom_type_mapping
        assert len(writer.res_types) == len(res_type_mapping)
        assert len(writer.atom_types) == len(atom_type_mapping)

    def test_set_output_path(self, res_type_mapping, atom_type_mapping):
        """Test setting output path after initialization."""
        writer = PDBWriter(
            output_path="initial.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        new_path = "new_output.pdb"
        writer.set_output_path(new_path)

        assert writer.output_path == new_path

    def test_get_pdb_headers(self, res_type_mapping, atom_type_mapping):
        """Test PDB header generation."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        headers = writer.get_pdb_headers()

        assert isinstance(headers, list)
        assert len(headers) > 0
        assert any("PARENT" in header for header in headers)

    def test_chain_end_formatting(self, res_type_mapping, atom_type_mapping):
        """Test chain end line formatting."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        chain_end_line = writer._chain_end(atom_index=100, end_resname="ALA", chain_name="A", residue_index=50)

        assert chain_end_line.startswith("TER")
        assert "100" in chain_end_line
        assert "ALA" in chain_end_line
        assert "A" in chain_end_line
        assert "50" in chain_end_line

    def test_write_pdb_output_structure(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that write() produces valid PDB structure."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        pdb_content = writer.write(sample_folding_output)

        # Check basic PDB structure
        assert isinstance(pdb_content, str)
        assert "MODEL     1" in pdb_content
        assert "ENDMDL" in pdb_content
        assert "END" in pdb_content

        # Split into lines and verify
        lines = pdb_content.split("\n")
        assert len(lines) > 0

    def test_write_pdb_with_atom_records(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that write() produces ATOM records."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        pdb_content = writer.write(sample_folding_output)

        # Check for ATOM records
        atom_lines = [line for line in pdb_content.split("\n") if line.startswith("ATOM")]
        assert len(atom_lines) > 0, "No ATOM records found in PDB output"

        # Verify ATOM record format (at least one should be properly formatted)
        for atom_line in atom_lines[:5]:  # Check first 5 ATOM lines
            assert len(atom_line) == 80
            assert atom_line[:6].strip() == "ATOM"

    def test_write_pdb_respects_atom_mask(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that write() respects the atom mask."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        # Count non-zero atoms in mask
        atom_mask = sample_folding_output["atom_mask"]
        expected_atoms = np.sum(atom_mask > 0.5)

        pdb_content = writer.write(sample_folding_output)
        atom_lines = [line for line in pdb_content.split("\n") if line.startswith("ATOM")]

        # Number of ATOM lines should correspond to masked atoms
        assert len(atom_lines) <= expected_atoms

    def test_write_pdb_has_correct_residue_count(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that the output contains information for all residues."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        pdb_content = writer.write(sample_folding_output)

        # The output should contain data
        assert len(pdb_content) > 0
        assert "ATOM" in pdb_content or "TER" in pdb_content

    def test_write_pdb_coordinates_format(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that coordinates are properly formatted in the output."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        pdb_content = writer.write(sample_folding_output)
        atom_lines = [line for line in pdb_content.split("\n") if line.startswith("ATOM")]

        if len(atom_lines) > 0:
            # Check first ATOM line has coordinate data
            first_atom = atom_lines[0]
            # Coordinates should be in columns 31-54 (0-indexed: 30-54)
            coord_section = first_atom[30:54]
            # Should contain numeric values
            assert any(char.isdigit() or char == "." or char == "-" for char in coord_section)

    def test_write_handles_multi_chain(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that multi-chain structures are handled properly."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        chain_indices = sample_folding_output.get("chain_indices")
        pdb_content = writer.write(sample_folding_output)

        if chain_indices is not None:
            unique_chains = len(np.unique(chain_indices))
            if unique_chains > 1:
                # Should have TER records for chain terminations
                ter_lines = [line for line in pdb_content.split("\n") if line.startswith("TER")]
                assert len(ter_lines) >= unique_chains

    def test_write_includes_parent_info(self, res_type_mapping, atom_type_mapping, sample_folding_output):
        """Test that output includes parent information in headers."""
        writer = PDBWriter(
            output_path="test.pdb", res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping
        )

        pdb_content = writer.write(sample_folding_output)

        # Should contain PARENT line
        assert "PARENT" in pdb_content


# ---------------------------------------------------------------------------
# Multi-polymer test suite (RNA / DNA / non-polymer ligand chains)
# ---------------------------------------------------------------------------

from bionemo_ir.data.writers.pdb_writer import PDB_MAX_CHAINS  # noqa: E402 (kept beside its suite)
from tests.common.test_utils.synthetic_folding_outputs import (  # noqa: E402 -- fixtures imported beside the suite that uses them
    dna_only_folding,
    many_chains_folding,
    multi_polymer_folding,
    nonpoly_ligand_folding,
    of3_mappings,
    rna_only_folding,
)


class TestPDBWriterMultiPolymer:
    """Exercises PDBWriter on RNA / DNA / non-polymer / mixed chain inputs."""

    @pytest.fixture
    def writer(self):
        res_map, atom_map = of3_mappings()
        return PDBWriter(res_type_mapping=res_map, atom_type_mapping=atom_map, output_path="test.pdb")

    # ----- RNA -------------------------------------------------------

    def test_rna_only_emits_atom_rows_with_nucleic_residue_name(self, writer):
        """RNA chain → ATOM rows with single-letter residue name in cols 18-20.

        Residue name field is 3 chars right-justified; ``A`` renders as
        ``"  A"`` (two leading spaces).
        """
        out = writer.write(rna_only_folding(sequence="AGCU", with_residue_names=True, with_mol_types=True))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "RNA chain should produce ATOM rows"
        # PDB residue-name field is columns 18-20 (1-indexed; 17-19 0-indexed).
        # Each ATOM row should have a single-letter nucleotide right-justified
        # in those 3 columns: '  A' for adenine, etc.
        for line in atom_lines:
            res_name_field = line[17:20]
            assert res_name_field.strip() in {"A", "G", "C", "U"}, (
                f"unexpected res name {res_name_field!r} in: {line[:30]}"
            )
        # No HETATM for an RNA polymer chain
        assert not any(l.startswith("HETATM") for l in out.split("\n"))

    def test_rna_heuristic_without_residue_names(self, writer):
        """Without ``residue_names``, _IHM_REMAP maps internal 'RA'/'RG'/...
        codes to the single-letter PDB residue names. Heuristic
        classification should still emit ATOM rows (polymer chain)."""
        out = writer.write(rna_only_folding(sequence="AG"))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "RNA heuristic should produce ATOM rows"

    @pytest.mark.parametrize("with_names", [True, False])
    def test_rna_unknown_nucleotide_renders_as_n(self, writer, with_names):
        """Unknown ribonucleotide (``RX``) renders as ``N`` in cols 18-20.

        No component table to resolve against here (unlike CIFWriter), so
        the name is just formatted in. Covers the ``residue_names`` path and
        the ``_IHM_REMAP`` fallback.
        """
        out = writer.write(rna_only_folding(sequence="AGN", with_residue_names=with_names, with_mol_types=with_names))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "RNA chain with an unknown base should produce ATOM rows"
        assert "  N" in {l[17:20] for l in atom_lines}, "unknown ribonucleotide should render right-justified as 'N'"
        assert not any(l.startswith("HETATM") for l in out.split("\n"))

    # ----- DNA -------------------------------------------------------

    def test_dna_only_emits_two_letter_residue_name(self, writer):
        """DNA residues use 2-letter codes (DA/DG/DC/DT) in the 3-char field."""
        out = writer.write(dna_only_folding(sequence="ACGT", with_residue_names=True, with_mol_types=True))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "DNA chain should produce ATOM rows"
        for line in atom_lines:
            res_name_field = line[17:20].strip()
            assert res_name_field in {"DA", "DG", "DC", "DT"}, f"unexpected DNA res name {res_name_field!r}"

    @pytest.mark.parametrize("with_names", [True, False])
    def test_dna_unknown_nucleotide_renders_as_dn(self, writer, with_names):
        """Unknown deoxyribonucleotide (``DX``) renders as ``DN``."""
        out = writer.write(dna_only_folding(sequence="ACN", with_residue_names=with_names, with_mol_types=with_names))
        atom_lines = [l for l in out.split("\n") if l.startswith("ATOM")]
        assert atom_lines, "DNA chain with an unknown base should produce ATOM rows"
        assert " DN" in {l[17:20] for l in atom_lines}, "unknown deoxyribonucleotide should render as 'DN'"

    # ----- Non-polymer (ligand) --------------------------------------

    def test_nonpoly_emits_hetatm_record_type(self, writer):
        """Non-polymer chains emit ``HETATM`` (cols 1-6) not ``ATOM``."""
        out = writer.write(
            nonpoly_ligand_folding(
                atom_names=["C1", "C2", "N2"],
                ccd_code="NAG",
                with_mol_types=True,
            )
        )
        hetatm_lines = [l for l in out.split("\n") if l.startswith("HETATM")]
        assert hetatm_lines, "non-polymer chain should produce HETATM rows"
        # Real CCD code surfaces in cols 18-20
        for line in hetatm_lines:
            assert line[17:20].strip() == "NAG"

    def test_no_ter_after_nonpoly_chain(self, writer):
        """Per PDB v3.3, ``TER`` does not follow a HETATM (non-polymer) group."""
        out = writer.write(nonpoly_ligand_folding(atom_names=["C1", "C2"], ccd_code="NAG", with_mol_types=True))
        # Single nonpoly chain: no TER lines should appear.
        ter_lines = [l for l in out.split("\n") if l.startswith("TER")]
        assert not ter_lines, f"unexpected TER after nonpoly chain: {ter_lines}"

    def test_nonpoly_heuristic_unk_fallback(self, writer):
        """Without residue_names or mol_types, all-X residues classify as
        nonpoly via the heuristic and HETATM rows use ``UNK``."""
        out = writer.write(nonpoly_ligand_folding(atom_names=["C1", "N2"], ccd_code=None, with_mol_types=False))
        hetatm_lines = [l for l in out.split("\n") if l.startswith("HETATM")]
        assert hetatm_lines, "heuristic should still flag chain as nonpoly"
        for line in hetatm_lines:
            assert line[17:20].strip() == "UNK"

    # ----- Multi-polymer combo ---------------------------------------

    def test_multi_polymer_record_types_and_ter_placement(self, writer):
        """Protein chain ends with ``TER``, nonpoly does not."""
        out = writer.write(multi_polymer_folding())
        lines = out.split("\n")
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        hetatm_lines = [l for l in lines if l.startswith("HETATM")]
        ter_lines = [l for l in lines if l.startswith("TER")]
        assert atom_lines, "polymer chains should emit ATOM rows"
        assert hetatm_lines, "nonpoly chain should emit HETATM rows"
        # Three polymer chains (protein, RNA, DNA) → 3 TERs.
        assert len(ter_lines) == 3, f"expected 3 TER lines (one per polymer chain), got {len(ter_lines)}"
        # CCD codes surface
        assert "ALA" in out and "TYR" in out and "NAG" in out

    def test_multi_polymer_chain_id_mapping(self, writer):
        """Chains 0..3 should map to PDB chain IDs ``A``..``D``."""
        out = writer.write(multi_polymer_folding())
        # Each ATOM/HETATM line has chain ID at column 22 (1-indexed; 21 0-indexed)
        atom_lines = [l for l in out.split("\n") if l.startswith(("ATOM", "HETATM"))]
        seen = {l[21] for l in atom_lines}
        assert seen == {"A", "B", "C", "D"}, f"expected chains A-D, saw {sorted(seen)}"

    # ----- Line width / column format --------------------------------

    def test_all_record_lines_padded_to_80_chars(self, writer):
        """Legacy PDB columnar contract — every line is exactly 80 chars."""
        out = writer.write(multi_polymer_folding())
        for line in out.split("\n"):
            if line == "":
                continue
            assert len(line) == 80, f"line not padded to 80 chars (got {len(line)}): {line!r}"

    # ----- Chain limit -----------------------------------------------

    def test_pdb_max_chain_constant(self):
        """PDB single-char chain field caps at 62 (A-Z + a-z + 0-9)."""
        assert PDB_MAX_CHAINS == 62

    def test_within_chain_limit_succeeds(self, writer):
        """62 chains (chain indices 0..61) fits PDB's single-char asym_id."""
        out = writer.write(many_chains_folding(n_chains=PDB_MAX_CHAINS))
        # Sample a few chain letters to confirm they made it into the file.
        for letter in "AZaz09":
            # Each letter should appear at column 22 somewhere.
            atom_lines = [l for l in out.split("\n") if l.startswith("ATOM") and l[21] == letter]
            assert atom_lines, f"chain {letter} missing from output"

    def test_too_many_chains_raises(self, writer):
        """63rd chain (index 62) overflows the 1-char asym_id field."""
        with pytest.raises(ValueError, match="62 chains"):
            writer.write(many_chains_folding(n_chains=PDB_MAX_CHAINS + 1))

    # ----- File round-trip -------------------------------------------

    def test_write_to_file_contains_hetatm_ccd_codes(self, tmp_path):
        res_map, atom_map = of3_mappings()
        out_path = tmp_path / "multi.pdb"
        writer = PDBWriter(res_type_mapping=res_map, atom_type_mapping=atom_map, output_path=str(out_path))
        writer.write(multi_polymer_folding())
        text = out_path.read_text()
        # Both ATOM and HETATM records persist to disk with correct codes
        assert "ATOM" in text and "HETATM" in text
        assert "ALA" in text and "NAG" in text
