# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest

from tensorrt_bionemo.data.parsers import parse_fasta_content, read_fasta, SequenceParsed
from tensorrt_bionemo.data.schemas import PolymerType

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "data" / "samples"


class TestReadFasta:

    def test_read_single_molecule(self):
        result = read_fasta(SAMPLES_DIR / "T1031.fasta")
        assert isinstance(result, SequenceParsed)
        assert len(result["sequences"]) == 1
        assert result["sequences"][0]["chain_id"] == "A"

    def test_read_returns_protein_type(self):
        result = read_fasta(SAMPLES_DIR / "T1033.fasta")
        assert result["sequences"][0]["polymer_type"] == PolymerType.PROTEIN.value

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
