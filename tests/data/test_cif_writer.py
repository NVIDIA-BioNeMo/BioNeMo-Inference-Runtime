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
from tensorrt_bionemo.data.writers.cif_writer import CIFWriter
from tests.common.test_utils.data import get_sample_folding_output


class TestCIFWriter:
    """Test suite for CIFWriter class."""

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
        fd, path = tempfile.mkstemp(suffix=".cif")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_initialization_with_both_mappings(self, res_type_mapping,
                                               atom_type_mapping,
                                               temp_output_file):
        """Test CIFWriter initialization with both residue and atom type mappings."""
        writer = CIFWriter(output_path=temp_output_file,
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        assert writer.output_path == temp_output_file
        assert len(writer.res_types) == len(res_type_mapping)
        assert len(writer.atom_types) == len(atom_type_mapping)

    def test_initialization_with_only_res_type(self, res_type_mapping,
                                               temp_output_file):
        """Test CIFWriter initialization with only residue type mapping."""
        writer = CIFWriter(output_path=temp_output_file,
                           res_type_mapping=res_type_mapping)

        assert writer.output_path == temp_output_file
        assert len(writer.res_types) == len(res_type_mapping)

    def test_set_output_path(self, res_type_mapping, atom_type_mapping):
        """Test setting output path after initialization."""
        writer = CIFWriter(output_path="initial.cif",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        new_path = "new_output.cif"
        writer.set_output_path(new_path)

        assert writer.output_path == new_path

    def test_write_output_basics(self, res_type_mapping,
                                        atom_type_mapping,
                                        sample_folding_output):
        """Test that write() produces valid CIF structure."""
        writer = CIFWriter(output_path="test.cif",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

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

    def test_write_respects_atom_mask(self, res_type_mapping,
                                          atom_type_mapping,
                                          sample_folding_output):
        """Test that write() respects the atom mask."""
        writer = CIFWriter(output_path="test.cif",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        # Count non-zero atoms in mask
        atom_mask = sample_folding_output["atom_mask"]
        expected_atoms = np.sum(atom_mask > 0.5)

        cif_content = writer.write(sample_folding_output)
        
        # The output should have content
        assert len(cif_content) > 0
        assert expected_atoms > 0  # Ensure we have atoms to write
        
        cif_content_as_lines: list[str] = cif_content.split("\n")
        count_of_ATOM_in_cif = sum([
            1 for x in cif_content_as_lines if "ATOM" in x])
        count_of_HETATM_in_cif = sum([
            1 for x in cif_content_as_lines if "HETATM" in x])
        
        assert count_of_ATOM_in_cif + count_of_HETATM_in_cif == expected_atoms
        
    def test_write_contains_model_info(self, res_type_mapping,
                                           atom_type_mapping,
                                           sample_folding_output):
        """Test that output contains model information."""
        writer = CIFWriter(output_path="test.cif",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        cif_content = writer.write(sample_folding_output)

        # Should contain model-related information
        assert "model" in cif_content.lower()

    def test_write_handles_multi_chain(self, res_type_mapping,
                                       atom_type_mapping,
                                       sample_folding_output):
        """Test that multi-chain structures are handled properly."""
        writer = CIFWriter(output_path="test.cif",
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        chain_indices = sample_folding_output.get("chain_indices")
        cif_content = writer.write(sample_folding_output)

        if chain_indices is not None:
            unique_chains = len(np.unique(chain_indices))
            # CIF should contain chain information
            assert len(cif_content) > 0
            if unique_chains > 1:
                # Should have entity or asym information for multiple chains
                assert "_entity" in cif_content.lower() or "asym" in cif_content.lower()

    def test_write_file(self, res_type_mapping,
                               atom_type_mapping,
                               sample_folding_output,
                               temp_output_file):
        """Test that write() returns the same string that is written to file."""
        writer = CIFWriter(output_path=temp_output_file,
                           res_type_mapping=res_type_mapping,
                           atom_type_mapping=atom_type_mapping)

        cif_content = writer.write(sample_folding_output)

        # Check that file was written
        assert os.path.exists(temp_output_file)
        
        # Read file and verify content matches returned buffer
        with open(temp_output_file, "r") as f:
            file_content = f.read()
        
        assert file_content == cif_content
