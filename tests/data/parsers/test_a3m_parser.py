# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

import pytest

from tensorrt_bionemo.data.parsers import parse_a3m_content, read_a3m
from tensorrt_bionemo.data.schemas import MSAParsed

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "data" / "samples"


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
