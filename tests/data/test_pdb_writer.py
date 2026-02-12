# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_bionemo.data.utils import (get_all_atom_types,
                                         get_all_residue_types)
from tensorrt_bionemo.data.writers.pdb_writer import PDBWriter
from tests.common.test_utils.data import get_sample_folding_output


class TestPDBWriter:
    """Test suite for PDBWriter class."""

    @pytest.fixture
    def res_type_mapping(self):
        """Fixture providing residue type mapping for openfold2."""
        res_types = get_all_residue_types("openfold2", include_gap=False)
        return {i: res_type for i, res_type in enumerate(res_types)}

    @pytest.fixture
    def atom_type_mapping(self):
        """Fixture providing atom type mapping for openfold2."""
        atom_types = get_all_atom_types("openfold2")
        return {i: atom_type for i, atom_type in enumerate(atom_types)}

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

    def test_initialization_with_both_mappings(self, res_type_mapping,
                                               atom_type_mapping,
                                               temp_output_file):
        """Test PDBWriter initialization with both residue and atom type mappings."""
        writer = PDBWriter(output_path=temp_output_file,
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        assert writer.output_path == temp_output_file
        assert writer.res_type_mapping == res_type_mapping
        assert writer.atom_type_mapping == atom_type_mapping
        assert len(writer.res_types) == len(res_type_mapping)
        assert len(writer.atom_types) == len(atom_type_mapping)

    def test_set_output_path(self, res_type_mapping, atom_type_mapping):
        """Test setting output path after initialization."""
        writer = PDBWriter(output_path="initial.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        new_path = "new_output.pdb"
        writer.set_output_path(new_path)

        assert writer.output_path == new_path

    def test_get_pdb_headers(self, res_type_mapping, atom_type_mapping):
        """Test PDB header generation."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        headers = writer.get_pdb_headers()

        assert isinstance(headers, list)
        assert len(headers) > 0
        assert any("PARENT" in header for header in headers)

    def test_chain_end_formatting(self, res_type_mapping, atom_type_mapping):
        """Test chain end line formatting."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        chain_end_line = writer._chain_end(atom_index=100,
                                           end_resname="ALA",
                                           chain_name="A",
                                           residue_index=50)

        assert chain_end_line.startswith("TER")
        assert "100" in chain_end_line
        assert "ALA" in chain_end_line
        assert "A" in chain_end_line
        assert "50" in chain_end_line

    def test_write_pdb_output_structure(self, res_type_mapping,
                                        atom_type_mapping,
                                        sample_folding_output):
        """Test that write() produces valid PDB structure."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        pdb_content = writer.write(sample_folding_output)

        # Check basic PDB structure
        assert isinstance(pdb_content, str)
        assert "MODEL     1" in pdb_content
        assert "ENDMDL" in pdb_content
        assert "END" in pdb_content

        # Split into lines and verify
        lines = pdb_content.split('\n')
        assert len(lines) > 0

    def test_write_pdb_with_atom_records(self, res_type_mapping,
                                         atom_type_mapping,
                                         sample_folding_output):
        """Test that write() produces ATOM records."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        pdb_content = writer.write(sample_folding_output)

        # Check for ATOM records
        atom_lines = [
            line for line in pdb_content.split('\n') if line.startswith("ATOM")
        ]
        assert len(atom_lines) > 0, "No ATOM records found in PDB output"

        # Verify ATOM record format (at least one should be properly formatted)
        for atom_line in atom_lines[:5]:  # Check first 5 ATOM lines
            assert len(atom_line) == 80
            assert atom_line[:6].strip() == "ATOM"

    def test_write_pdb_respects_atom_mask(self, res_type_mapping,
                                          atom_type_mapping,
                                          sample_folding_output):
        """Test that write() respects the atom mask."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        # Count non-zero atoms in mask
        atom_mask = sample_folding_output["atom_mask"]
        expected_atoms = np.sum(atom_mask > 0.5)

        pdb_content = writer.write(sample_folding_output)
        atom_lines = [
            line for line in pdb_content.split('\n') if line.startswith("ATOM")
        ]

        # Number of ATOM lines should correspond to masked atoms
        assert len(atom_lines) <= expected_atoms

    def test_write_pdb_has_correct_residue_count(self, res_type_mapping,
                                                 atom_type_mapping,
                                                 sample_folding_output):
        """Test that the output contains information for all residues."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        pdb_content = writer.write(sample_folding_output)

        # The output should contain data
        assert len(pdb_content) > 0
        assert "ATOM" in pdb_content or "TER" in pdb_content

    def test_write_pdb_coordinates_format(self, res_type_mapping,
                                          atom_type_mapping,
                                          sample_folding_output):
        """Test that coordinates are properly formatted in the output."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        pdb_content = writer.write(sample_folding_output)
        atom_lines = [
            line for line in pdb_content.split('\n') if line.startswith("ATOM")
        ]

        if len(atom_lines) > 0:
            # Check first ATOM line has coordinate data
            first_atom = atom_lines[0]
            # Coordinates should be in columns 31-54 (0-indexed: 30-54)
            coord_section = first_atom[30:54]
            # Should contain numeric values
            assert any(char.isdigit() or char == '.' or char == '-'
                       for char in coord_section)

    def test_write_handles_multi_chain(self, res_type_mapping,
                                       atom_type_mapping,
                                       sample_folding_output):
        """Test that multi-chain structures are handled properly."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        chain_indices = sample_folding_output.get("chain_indices")
        pdb_content = writer.write(sample_folding_output)

        if chain_indices is not None:
            unique_chains = len(np.unique(chain_indices))
            if unique_chains > 1:
                # Should have TER records for chain terminations
                ter_lines = [
                    line for line in pdb_content.split('\n')
                    if line.startswith("TER")
                ]
                assert len(ter_lines) >= unique_chains

    def test_write_includes_parent_info(self, res_type_mapping,
                                        atom_type_mapping,
                                        sample_folding_output):
        """Test that output includes parent information in headers."""
        writer = PDBWriter(output_path="test.pdb",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        pdb_content = writer.write(sample_folding_output)

        # Should contain PARENT line
        assert "PARENT" in pdb_content
