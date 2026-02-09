# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest

from tensorrt_bionemo.data.parsers import parse_a3m_content, read_a3m
from tensorrt_bionemo.data.schemas import MSAParsed

SAMPLES_DIR = Path(
    __file__
).parent.parent.parent.parent / "examples" / "data" / "samples" / "monomers"


class TestReadA3M:

    def test_read_returns_a3m_parsed(self):
        result = read_a3m(SAMPLES_DIR / "msas" / "T1031.a3m")
        assert isinstance(result, MSAParsed)
        assert "sequences" in result
        assert "raw" in result
        assert "descriptions" in result

    def test_read_has_sequences(self):
        result = read_a3m(SAMPLES_DIR / "msas" / "T1033.a3m")
        assert len(result["sequences"]) > 0

    @pytest.mark.parametrize("target", ["T1031", "T1033", "T1047s1", "T1094"])
    def test_all_a3m_files_readable(self, target):
        result = read_a3m(SAMPLES_DIR / "msas" / f"{target}.a3m")
        assert len(result["sequences"]) >= 1


class TestParseA3MContent:

    def test_parse_simple_a3m(self):
        content = StringIO(">seq1\nACDEFG\n>seq2\nACDEFG")
        result = parse_a3m_content(content)
        assert result["sequences"] == ["ACDEFG", "ACDEFG"]

    def test_lowercase_removal(self):
        content = StringIO(">seq1\nACdeFG")
        result = parse_a3m_content(content)
        assert result["sequences"] == ["ACFG"]
        assert result["raw"] == ["ACdeFG"]

    def test_raw_preserved(self):
        content = StringIO(">seq1\nACdeFGhi")
        result = parse_a3m_content(content)
        assert result["raw"][0] == "ACdeFGhi"
        assert result["sequences"][0] == "ACFG"


class TestCommentFiltering:
    """Test suite for comment line filtering in A3M parser.

    These tests verify that lines starting with '#' (after lstrip()) are
    properly filtered out before FASTA parsing, and that sequences are
    parsed identically with or without comments present.
    """

    def test_comment_lines_filtered_out(self):
        """Test that comment lines are removed during parsing."""
        content_with_comments = StringIO("# This is a comment\n"
                                         ">seq1\n"
                                         "ACDEFG\n"
                                         "# Another comment\n"
                                         ">seq2\n"
                                         "GHIKLM")
        result = parse_a3m_content(content_with_comments)
        assert result["sequences"] == ["ACDEFG", "GHIKLM"]
        assert result["descriptions"] == ["seq1", "seq2"]

    def test_identical_parsing_with_and_without_comments(self):
        """Test that sequences are parsed identically with or without comments."""
        content_without_comments = StringIO(">seq1\n"
                                            "ACDEFG\n"
                                            ">seq2\n"
                                            "GHIKLM\n"
                                            ">seq3\n"
                                            "NQSTVWY")
        result_without = parse_a3m_content(content_without_comments)

        content_with_comments = StringIO("# Header comment\n"
                                         ">seq1\n"
                                         "ACDEFG\n"
                                         "# Middle comment\n"
                                         ">seq2\n"
                                         "GHIKLM\n"
                                         "# Another comment\n"
                                         ">seq3\n"
                                         "NQSTVWY\n"
                                         "# Trailing comment")
        result_with = parse_a3m_content(content_with_comments)

        # Verify sequences are identical
        assert result_without["sequences"] == result_with["sequences"]
        assert result_without["raw"] == result_with["raw"]
        assert result_without["descriptions"] == result_with["descriptions"]

    def test_comments_with_leading_whitespace(self):
        """Test that comments with leading whitespace are filtered."""
        content = StringIO("   # Comment with leading spaces\n"
                           ">seq1\n"
                           "ACDEFG\n"
                           "\t# Comment with leading tab\n"
                           ">seq2\n"
                           "GHIKLM")
        result = parse_a3m_content(content)
        assert result["sequences"] == ["ACDEFG", "GHIKLM"]

    def test_multiple_consecutive_comments(self):
        """Test multiple consecutive comment lines."""
        content = StringIO("# Comment 1\n"
                           "# Comment 2\n"
                           "# Comment 3\n"
                           ">seq1\n"
                           "ACDEFG")
        result = parse_a3m_content(content)
        assert result["sequences"] == ["ACDEFG"]

    def test_comments_not_preserved_by_default(self):
        """Test that comments are not included by default."""
        content = StringIO("# This is a comment\n"
                           ">seq1\n"
                           "ACDEFG")
        result = parse_a3m_content(content)
        assert result["comments"] is None

    def test_preserve_comments_option(self):
        """Test that preserve_comments option collects comment lines."""
        content = StringIO("# First comment\n"
                           ">seq1\n"
                           "ACDEFG\n"
                           "# Second comment\n"
                           ">seq2\n"
                           "GHIKLM\n"
                           "# Third comment")
        result = parse_a3m_content(content, preserve_comments=True)

        assert result["comments"] is not None
        assert len(result["comments"]) == 3
        assert result["comments"][0] == "# First comment"
        assert result["comments"][1] == "# Second comment"
        assert result["comments"][2] == "# Third comment"

        # Verify sequences are still parsed correctly
        assert result["sequences"] == ["ACDEFG", "GHIKLM"]

    def test_preserve_comments_strips_whitespace(self):
        """Test that preserved comments have leading/trailing whitespace stripped."""
        content = StringIO("   # Comment with spaces   \n"
                           ">seq1\n"
                           "ACDEFG\n"
                           "\t# Comment with tab\t\n")
        result = parse_a3m_content(content, preserve_comments=True)

        assert result["comments"] == [
            "# Comment with spaces", "# Comment with tab"
        ]

    def test_empty_file_with_only_comments(self):
        """Test parsing file containing only comments."""
        content = StringIO("# Comment 1\n"
                           "# Comment 2\n"
                           "# Comment 3")
        result = parse_a3m_content(content, preserve_comments=True)

        assert result["sequences"] == []
        assert result["comments"] == [
            "# Comment 1", "# Comment 2", "# Comment 3"
        ]

    def test_comments_with_lowercase_deletions(self):
        """Test comment filtering with sequences containing lowercase deletions."""
        content = StringIO("# Comment before deletions\n"
                           ">seq1\n"
                           "ACdeFGhi\n"
                           "# Comment after deletions\n"
                           ">seq2\n"
                           "GHikLM")
        result = parse_a3m_content(content, preserve_comments=True)

        assert result["sequences"] == ["ACFG", "GHLM"]
        assert result["raw"] == ["ACdeFGhi", "GHikLM"]
        assert len(result["comments"]) == 2

    def test_hash_not_at_line_start_not_treated_as_comment(self):
        """Test that '#' not at the start of line (after lstrip) is not a comment."""
        content = StringIO(">seq1 description with # hash\n"
                           "ACDEFG")
        result = parse_a3m_content(content)

        # The sequence should be parsed normally
        assert result["sequences"] == ["ACDEFG"]
        # Description might contain the hash
        assert "seq1" in result["descriptions"][0]
