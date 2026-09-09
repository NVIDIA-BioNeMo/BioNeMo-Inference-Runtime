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

import numpy as np

from bionemo_ir.pipeline.models.openfold2.postprocessor import PostProcessor, PostProcessorConfig


class TestPostProcessorNormalizeResidueIndices:
    """Test suite for PostProcessor residue index normalization functionality."""

    def test_normalize_residue_indices_single_chain(self):
        """Test residue index normalization for a single chain."""
        config = PostProcessorConfig()
        postprocessor = PostProcessor(config)

        # Single chain: residue indices 0, 1, 2, 3, 4
        residue_index = np.array([0, 1, 2, 3, 4])

        normalized = postprocessor._normalize_residue_indices(residue_index)

        # Should remain unchanged for single chain
        np.testing.assert_array_equal(normalized, residue_index)

    def test_normalize_residue_indices_two_chains(self):
        """Test residue index normalization for two chains.

        Note: The original algorithm uses (residue_index - position) / gap to detect chains.
        For [0, 1, 2, 200, 201, 202], the chain IDs are all 0 because (200-3)/200=0.985->0.
        So this is actually treated as a single chain by the original code.
        """
        config = PostProcessorConfig(multimer_ri_gap=200)
        postprocessor = PostProcessor(config)

        residue_index = np.array([0, 1, 2, 200, 201, 202])
        normalized = postprocessor._normalize_residue_indices(residue_index)

        # Original code treats this as single chain (no chain transition detected)
        expected = np.array([0, 1, 2, 200, 201, 202])
        np.testing.assert_array_equal(normalized, expected)

    def test_normalize_residue_indices_three_chains(self):
        """Test residue index normalization for three chains.

        For [0, 1, 200, 201, 202, 400, 401]:
        - Positions 0-4: chain_id = 0
        - Positions 5-6: chain_id = 1 (because (400-5)/200=1.975->1, (401-6)/200=1.975->1)
        """
        config = PostProcessorConfig(multimer_ri_gap=200)
        postprocessor = PostProcessor(config)

        residue_index = np.array([0, 1, 200, 201, 202, 400, 401])
        normalized = postprocessor._normalize_residue_indices(residue_index)

        # Chain transition at position 5
        # Offset for positions 5-6 is: 5 + 1*200 = 205
        expected = np.array([0, 1, 200, 201, 202, 195, 196])
        np.testing.assert_array_equal(normalized, expected)

    def test_normalize_residue_indices_with_offsets(self):
        """Test residue index normalization when chains have sufficient gaps.

        For [5, 6, 7, 210, 211, 212]:
        - Positions 0-2: chain_id = 0 (5-0=5, 6-1=5, 7-2=5, all / 200 = 0)
        - Positions 3-5: chain_id = 1 (210-3=207, 211-4=207, 212-5=207, all / 200 = 1)
        """
        config = PostProcessorConfig(multimer_ri_gap=200)
        postprocessor = PostProcessor(config)

        residue_index = np.array([5, 6, 7, 210, 211, 212])
        normalized = postprocessor._normalize_residue_indices(residue_index)

        # Chain transition at position 3
        # Offset for positions 3-5 is: 3 + 1*200 = 203
        expected = np.array([5, 6, 7, 7, 8, 9])
        np.testing.assert_array_equal(normalized, expected)

    def test_normalize_residue_indices_edge_case_empty(self):
        """Test residue index normalization with empty input."""
        config = PostProcessorConfig()
        postprocessor = PostProcessor(config)

        residue_index = np.array([])

        normalized = postprocessor._normalize_residue_indices(residue_index)

        # Should handle empty array gracefully
        assert len(normalized) == 0


class TestPostProcessorGetChainIndices:
    """Test suite for PostProcessor chain index extraction functionality."""

    def test_get_chain_indices_with_asym_id(self):
        """Test chain index extraction when asym_id is provided."""
        config = PostProcessorConfig()
        postprocessor = PostProcessor(config)

        np_batch = {
            "aatype": np.array([1, 2, 3, 4]),
            "asym_id": np.array([1, 1, 2, 2]),  # 1-based asym_id
        }

        chain_indices = postprocessor._get_chain_indices(np_batch)

        # Expected: 0-based chain indices
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(chain_indices, expected)

    def test_get_chain_indices_without_asym_id(self):
        """Test chain index extraction when asym_id is not provided."""
        config = PostProcessorConfig()
        postprocessor = PostProcessor(config)

        np_batch = {"aatype": np.array([1, 2, 3, 4])}

        chain_indices = postprocessor._get_chain_indices(np_batch)

        # Expected: all zeros (single chain)
        expected = np.zeros(4, dtype=np.int64)
        np.testing.assert_array_equal(chain_indices, expected)

    def test_get_chain_indices_multiple_chains(self):
        """Test chain index extraction with multiple chains."""
        config = PostProcessorConfig()
        postprocessor = PostProcessor(config)

        np_batch = {
            "aatype": np.array([1, 2, 3, 4, 5, 6]),
            "asym_id": np.array([1, 1, 2, 2, 3, 3]),  # 1-based asym_id
        }

        chain_indices = postprocessor._get_chain_indices(np_batch)

        # Expected: 0-based chain indices for 3 chains
        expected = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(chain_indices, expected)
