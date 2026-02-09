# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest

from tensorrt_bionemo.data.parsers import (SequenceParsed, parse_fasta_content,
                                           read_fasta)
from tensorrt_bionemo.data.parsers.fasta import _generate_chain_id
from tensorrt_bionemo.data.schemas import PolymerType

SAMPLES_DIR = Path(
    __file__
).parent.parent.parent.parent / "examples" / "data" / "samples" / "monomers"


class TestReadFasta:

    def test_read_single_molecule(self):
        result = read_fasta(SAMPLES_DIR / "T1031.fasta")
        assert isinstance(result, SequenceParsed)
        assert len(result["sequences"]) == 1
        assert result["sequences"][0]["chain_id"] == "A"

    def test_read_returns_protein_type(self):
        result = read_fasta(SAMPLES_DIR / "T1033.fasta")
        assert result["sequences"][0][
            "polymer_type"] == PolymerType.PROTEIN.value

    @pytest.mark.parametrize("filename,expected_len", [
        ("T1031.fasta", 95),
        ("T1033.fasta", 100),
    ])
    def test_sequence_lengths(self, filename, expected_len):
        result = read_fasta(SAMPLES_DIR / filename)
        assert len(result["sequences"][0]["sequence"]) == expected_len


class TestParseFastaContent:

    def test_parse_from_string(self):
        content = StringIO(">test_seq\nACDEFGHIKLMNPQRSTVWY")
        result = parse_fasta_content(content)
        assert result["sequences"][0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_parse_multiline_sequence(self):
        content = StringIO(">test\nACDE\nFGHI\nKLMN")
        result = parse_fasta_content(content)
        assert result["sequences"][0]["sequence"] == "ACDEFGHIKLMN"

    def test_parse_multiple_sequences(self):
        content = StringIO(">chain1\nACDE\n>chain2\nFGHI")
        result = parse_fasta_content(content)
        assert len(result["sequences"]) == 2
        # mmCIF-style: first two chains are A and B
        assert result["sequences"][0]["chain_id"] == "A"
        assert result["sequences"][1]["chain_id"] == "B"

    def test_return_as_list(self):
        content = StringIO(">seq1\nACDE\n>seq2\nFGHI")
        seqs, descs = parse_fasta_content(content, return_as_list=True)
        assert seqs == ["ACDE", "FGHI"]
        assert len(descs) == 2

    def test_description_preserved(self):
        content = StringIO(">my_protein description here\nACDE")
        result = parse_fasta_content(content)
        assert "my_protein" in result["descriptions"][0]


class TestChainIdGeneration:
    """Test suite for chain ID generation using mmCIF-style conventions.

    Chain IDs must be 1-4 alphanumeric characters as enforced by
    Polymer._validate_chain_id (regex pattern: ^[A-Za-z0-9]{1,4}$).

    mmCIF convention uses base-26 encoding with letters A-Z:
    - 0-25: A-Z
    - 26-701: AA-ZZ
    - 702-18277: AAA-ZZZ
    - 18278-475253: AAAA-ZZZZ
    """

    def test_generate_chain_id_first_sequence(self):
        """Test chain ID for the first sequence (index 0)."""
        assert _generate_chain_id(0) == "A"

    def test_generate_chain_id_single_letter_range(self):
        """Test chain IDs for indices 0-25 (single letter: A-Z)."""
        assert _generate_chain_id(0) == "A"
        assert _generate_chain_id(1) == "B"
        assert _generate_chain_id(25) == "Z"

    def test_generate_chain_id_two_letter_start(self):
        """Test transition to two-letter chain IDs (mmCIF style)."""
        assert _generate_chain_id(25) == "Z"
        assert _generate_chain_id(26) == "AA"
        assert _generate_chain_id(27) == "AB"

    def test_generate_chain_id_two_letter_range(self):
        """Test two-letter chain ID boundaries."""
        assert _generate_chain_id(26) == "AA"  # First 2-letter
        assert _generate_chain_id(51) == "AZ"  # End of first "row"
        assert _generate_chain_id(52) == "BA"  # Start of second "row"
        assert _generate_chain_id(100) == "CW"
        assert _generate_chain_id(701) == "ZZ"  # Last 2-letter

    def test_generate_chain_id_three_letter_start(self):
        """Test transition to three-letter chain IDs."""
        assert _generate_chain_id(701) == "ZZ"
        assert _generate_chain_id(702) == "AAA"
        assert _generate_chain_id(703) == "AAB"

    def test_generate_chain_id_three_letter_range(self):
        """Test three-letter chain ID boundaries."""
        assert _generate_chain_id(702) == "AAA"  # First 3-letter
        assert _generate_chain_id(1000) == "ALM"
        assert _generate_chain_id(5000) == "GJI"
        assert _generate_chain_id(18277) == "ZZZ"  # Last 3-letter

    def test_generate_chain_id_four_letter_start(self):
        """Test transition to four-letter chain IDs."""
        assert _generate_chain_id(18277) == "ZZZ"
        assert _generate_chain_id(18278) == "AAAA"
        assert _generate_chain_id(18279) == "AAAB"

    def test_generate_chain_id_four_letter_range(self):
        """Test four-letter chain IDs."""
        assert _generate_chain_id(18278) == "AAAA"  # First 4-letter
        assert _generate_chain_id(50000) == "BUYC"
        assert _generate_chain_id(100000) == "EQXE"
        assert _generate_chain_id(475253) == "ZZZZ"  # Last valid

    def test_generate_chain_id_max_length(self):
        """Test that all generated IDs are at most 4 characters."""
        test_indices = [
            0, 25, 26, 701, 702, 18277, 18278, 50000, 100000, 475253
        ]
        for idx in test_indices:
            chain_id = _generate_chain_id(idx)
            assert len(
                chain_id
            ) <= 4, f"Chain ID '{chain_id}' for index {idx} exceeds 4 characters"
            assert len(chain_id) >= 1, f"Chain ID for index {idx} is empty"

    def test_generate_chain_id_exceeds_max(self):
        """Test that exceeding maximum index raises clear error."""
        max_index = 26 + 26**2 + 26**3 + 26**4 - 1  # 475,253
        with pytest.raises(ValueError) as exc_info:
            _generate_chain_id(max_index + 1)

        error_msg = str(exc_info.value)
        assert "475254" in error_msg  # The invalid index
        assert "475253" in error_msg  # The maximum allowed
        assert "Polymer._validate_chain_id" in error_msg or "mmCIF" in error_msg

    def test_generate_chain_id_uniqueness(self):
        """Test that generated chain IDs are unique for different indices."""
        chain_ids = [_generate_chain_id(i) for i in range(10000)]
        assert len(chain_ids) == len(
            set(chain_ids)), "Generated chain IDs are not unique"

    def test_chain_id_matches_validation_pattern(self):
        """Test that generated chain IDs match Polymer._validate_chain_id pattern."""
        import re
        pattern = re.compile(r'^[A-Za-z0-9]{1,4}$')

        # Test various indices across different ranges
        test_indices = [
            0, 10, 25, 26, 100, 701, 702, 5000, 18277, 18278, 50000, 475253
        ]
        for idx in test_indices:
            chain_id = _generate_chain_id(idx)
            assert pattern.match(chain_id), \
                f"Chain ID '{chain_id}' for index {idx} does not match validation pattern"

    def test_mmcif_style_alphabetical_order(self):
        """Test that chain IDs follow alphabetical order (mmCIF convention)."""
        # First 26 should be A-Z
        for i in range(26):
            assert _generate_chain_id(i) == chr(ord('A') + i)

        # Next should be AA, AB, AC...
        assert _generate_chain_id(26) == "AA"
        assert _generate_chain_id(27) == "AB"
        assert _generate_chain_id(28) == "AC"


class TestFastaWithManySequences:
    """Test FASTA parsing with many sequences to verify chain ID generation."""

    def test_parse_many_sequences_old_limit(self):
        """Test parsing sequences beyond the old algorithm's limit (260 sequences)."""
        # Old algorithm would fail around sequence 260 (when chain_id becomes "A10")
        # with 4 chars, and definitely at 2600 with "A100" (5 chars)
        num_sequences = 300
        fasta_lines = []
        # Create truly unique sequences by encoding the index into the sequence itself
        base_seq = "ACDEFGHIKLMNPQRSTVWY"
        for i in range(num_sequences):
            fasta_lines.append(f">seq{i}")
            # Make each sequence unique by repeating base_seq different number of times
            # and adding a single amino acid based on index
            fasta_lines.append(base_seq * (i + 1))

        content = StringIO("\n".join(fasta_lines))
        result = parse_fasta_content(content)

        assert len(result["sequences"]) == num_sequences
        # Verify all chain IDs are valid (1-4 chars)
        for seq in result["sequences"]:
            chain_id = seq["chain_id"]
            assert len(
                chain_id) <= 4, f"Chain ID '{chain_id}' exceeds 4 characters"
            assert len(chain_id) >= 1

    def test_parse_sequences_across_boundaries(self):
        """Test parsing sequences at mmCIF boundary transitions."""
        # Test around key boundaries: 26 (A->AA), 702 (ZZ->AAA), 18278 (ZZZ->AAAA)
        test_counts = [30, 710, 18285]

        for num_seq in test_counts:
            fasta_lines = []
            for i in range(num_seq):
                fasta_lines.append(f">seq{i}")
                fasta_lines.append(f"SEQ{i}")

            content = StringIO("\n".join(fasta_lines))
            result = parse_fasta_content(content)

            assert len(result["sequences"]) == num_seq
            # Check all chain IDs are valid
            for seq in result["sequences"]:
                assert len(seq["chain_id"]) <= 4

    def test_chain_id_assignment_order(self):
        """Test that chain IDs are assigned in the expected mmCIF order."""
        fasta_lines = []
        for i in range(40):
            fasta_lines.append(f">seq{i}")
            fasta_lines.append(f"SEQ{i}")

        content = StringIO("\n".join(fasta_lines))
        result = parse_fasta_content(content)

        expected_ids = [_generate_chain_id(i) for i in range(40)]
        actual_ids = [seq["chain_id"] for seq in result["sequences"]]

        assert actual_ids == expected_ids

        # Verify mmCIF-style progression
        assert actual_ids[0] == "A"
        assert actual_ids[25] == "Z"
        assert actual_ids[26] == "AA"
        assert actual_ids[27] == "AB"

    def test_error_message_for_too_many_sequences(self):
        """Test that exceeding maximum sequences gives clear error."""
        max_index = 26 + 26**2 + 26**3 + 26**4 - 1  # 475,253
        # We can't actually create 475k+ sequences in a test, so test with a smaller mock
        # Instead, we'll test the error directly from the generation function
        with pytest.raises(ValueError) as exc_info:
            _generate_chain_id(max_index + 1)

        error_msg = str(exc_info.value)
        # Should mention the sequence index that failed
        assert "475254" in error_msg
        # Should reference the validation constraint
        assert "Polymer._validate_chain_id" in error_msg or "4 characters" in error_msg

    def test_mmcif_style_compatibility(self):
        """Test that algorithm follows mmCIF conventions for first sequences."""
        # For the first 26 sequences, should be A-Z (standard mmCIF)
        fasta_lines = []
        for i in range(26):
            fasta_lines.append(f">seq{i}")
            fasta_lines.append(f"SEQ{i}")

        content = StringIO("\n".join(fasta_lines))
        result = parse_fasta_content(content)

        # Should be A through Z
        expected_chain_ids = [chr(ord('A') + i) for i in range(26)]
        actual_chain_ids = [seq["chain_id"] for seq in result["sequences"]]
        assert actual_chain_ids == expected_chain_ids

        # All should be single letter
        for seq in result["sequences"]:
            assert len(seq["chain_id"]) == 1
            assert seq["chain_id"].isupper()

    def test_two_letter_chain_ids_follow_mmcif(self):
        """Test that two-letter chain IDs follow mmCIF conventions."""
        # Create 52 sequences total (0-51), so we get chain IDs from A to AZ
        # The first 26 will be A-Z, the next 26 will be AA-AZ
        fasta_lines = []
        for i in range(52):
            fasta_lines.append(f">seq{i}")
            fasta_lines.append(f"SEQ{i}")

        content = StringIO("\n".join(fasta_lines))
        result = parse_fasta_content(content)

        # Get all chain IDs
        all_chain_ids = [seq["chain_id"] for seq in result["sequences"]]

        # First 26 should be A-Z (single letter)
        assert all_chain_ids[0] == "A"
        assert all_chain_ids[25] == "Z"

        # Next 26 should be AA-AZ (two letters)
        two_letter_ids = all_chain_ids[26:]
        assert two_letter_ids[0] == "AA"
        assert two_letter_ids[1] == "AB"
        assert two_letter_ids[-1] == "AZ"

        # Verify all two-letter IDs are 2 characters
        for chain_id in two_letter_ids:
            assert len(chain_id) == 2
            assert chain_id.isupper()
