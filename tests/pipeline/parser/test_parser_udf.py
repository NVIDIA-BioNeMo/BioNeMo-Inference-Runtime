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

import asyncio
from pathlib import Path

import pytest

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.stages.parser_stage import ParserUDF

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "data" / "samples" / "monomers"


class TestParserUDFBasicParsing:
    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    @pytest.fixture
    def sample_sequence(self):
        return "ACDEFGHIKLMNPQRSTVWY"

    def test_parse_single_polymer_with_sequence_only(self, parser_udf, sample_sequence):
        request = InputRequest(input_id="test_seq", polymers=[Polymer(chain_id="A", sequence=sample_sequence)])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        assert result["parsed"]["input_id"] == "test_seq"
        assert len(result["parsed"]["polymers"]) == 1
        assert result["parsed"]["polymers"][0]["sequence"] == sample_sequence

    @pytest.mark.skipif(not (SAMPLES_DIR / "msas" / "T1031.a3m").exists(), reason="Sample MSA file not found")
    def test_parse_single_polymer_with_msa_file(self, parser_udf, sample_sequence):
        a3m_path = str(SAMPLES_DIR / "msas" / "T1031.a3m")
        request = InputRequest(
            input_id="test_msa",
            polymers=[Polymer(chain_id="A", sequence=sample_sequence, msas=[MSARecord(path=a3m_path)])],
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["msas"] is not None
        assert len(polymer["msas"]) == 1
        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = polymer["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert len(msa_parsed["sequences"]) > 0
        assert msa_parsed["sequences"][0] is not None

    def test_parse_single_polymer_with_msa_content(self, parser_udf, sample_sequence):
        a3m_content = f">query\n{sample_sequence}\n>hit1\n{sample_sequence}\n"
        request = InputRequest(
            input_id="test_msa_content",
            polymers=[Polymer(chain_id="A", sequence=sample_sequence, msas=[MSARecord(content=a3m_content)])],
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["msas"] is not None
        assert len(polymer["msas"]) == 1
        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = polymer["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert len(msa_parsed["sequences"]) == 2  # query + hit1
        assert msa_parsed["sequences"][0] == sample_sequence
        assert msa_parsed["descriptions"][0] == "query"
        assert msa_parsed["sequences"][1] == sample_sequence
        assert msa_parsed["descriptions"][1] == "hit1"

    def test_parse_multiple_polymers_multimer(self, parser_udf):
        request = InputRequest(
            input_id="multimer_test",
            polymers=[
                Polymer(chain_id="A", sequence="ACDEFGHIKL"),
                Polymer(chain_id="B", sequence="MNPQRSTVWY"),
            ],
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        assert len(result["parsed"]["polymers"]) == 2
        assert result["parsed"]["polymers"][0]["chain_id"] == "A"
        assert result["parsed"]["polymers"][1]["chain_id"] == "B"

    def test_parse_polymer_with_different_chain_ids(self, parser_udf):
        request = InputRequest(
            input_id="chain_test",
            polymers=[
                Polymer(chain_id="X", sequence="ACDEF"),
                Polymer(chain_id="Y", sequence="GHIKL"),
                Polymer(chain_id="Z", sequence="MNPQR"),
            ],
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert len(result["parsed"]["polymers"]) == 3
        chain_ids = [p["chain_id"] for p in result["parsed"]["polymers"]]
        assert chain_ids == ["X", "Y", "Z"]


class TestParserUDFEdgeCases:
    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    def test_parse_empty_polymers_list(self, parser_udf):
        request = InputRequest(input_id="empty", polymers=[])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        assert result["parsed"]["polymers"] == []

    def test_parse_polymer_without_msas(self, parser_udf):
        request = InputRequest(input_id="no_msa", polymers=[Polymer(chain_id="A", sequence="ACDEFGHIKL", msas=None)])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["msas"] is None

    def test_parse_polymer_without_templates(self, parser_udf):
        request = InputRequest(
            input_id="no_template", polymers=[Polymer(chain_id="A", sequence="ACDEFGHIKL", templates=None)]
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["templates"] is None

    def test_parse_polymer_with_empty_msas_list(self, parser_udf):
        request = InputRequest(
            input_id="empty_msa_list", polymers=[Polymer(chain_id="A", sequence="ACDEFGHIKL", msas=[])]
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["msas"] is None  # Empty list becomes None after parsing


class TestParserUDFErrorHandling:
    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    def test_on_row_error_returns_none_parsed(self, parser_udf):
        error = ValueError("Test error")
        row = {"record": {}, "__record_id": "test"}

        result = parser_udf.on_row_error(row, error)

        assert result["parsed"] is None

    def test_invalid_msa_path_raises_error(self, parser_udf):
        request = InputRequest(
            input_id="bad_path",
            polymers=[
                Polymer(chain_id="A", sequence="ACDEFGHIKL", msas=[MSARecord(path="/nonexistent/path/to/file.a3m")])
            ],
        )
        row = {"record": dict(request), "__record_id": "test"}

        with pytest.raises(FileNotFoundError):
            asyncio.run(parser_udf.udf_for_item(row))


class TestParserUDFContentExtraction:
    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    def test_parse_msa_content_from_inline_content(self, parser_udf):
        """Test that MSA content is correctly parsed into MSAParsed."""
        content = ">seq1\nACDEF\n>seq2\nGHIKL\n"
        request = InputRequest(
            input_id="test", polymers=[Polymer(chain_id="A", sequence="ACDEF", msas=[MSARecord(content=content)])]
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = result["parsed"]["polymers"][0]["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert "sequences" in msa_parsed
        assert "raw" in msa_parsed
        assert "descriptions" in msa_parsed
        assert len(msa_parsed["sequences"]) == 2
        assert msa_parsed["sequences"][0] == "ACDEF"
        assert msa_parsed["descriptions"][0] == "seq1"
        assert msa_parsed["sequences"][1] == "GHIKL"
        assert msa_parsed["descriptions"][1] == "seq2"

    def test_parse_msa_content_from_dict(self, parser_udf):
        """Test that MSA can be parsed from dict format."""
        content = ">seq1\nACDEF\n"
        request = InputRequest(
            input_id="test",
            polymers=[
                Polymer(chain_id="A", sequence="ACDEF", msas=[{"content": content, "path": None, "format": "a3m"}])
            ],
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = result["parsed"]["polymers"][0]["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert len(msa_parsed["sequences"]) == 1
        assert msa_parsed["sequences"][0] == "ACDEF"
        assert msa_parsed["descriptions"][0] == "seq1"

    @pytest.mark.skipif(not (SAMPLES_DIR / "msas" / "T1031.a3m").exists(), reason="Sample MSA file not found")
    def test_parse_msa_content_from_file_path(self, parser_udf):
        """Test that MSA content is loaded and parsed from file path."""
        a3m_path = str(SAMPLES_DIR / "msas" / "T1031.a3m")
        request = InputRequest(
            input_id="test", polymers=[Polymer(chain_id="A", sequence="ACDEF", msas=[MSARecord(path=a3m_path)])]
        )
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = result["parsed"]["polymers"][0]["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert "sequences" in msa_parsed
        assert len(msa_parsed["sequences"]) > 0
        assert msa_parsed["sequences"][0] is not None
        assert len(msa_parsed["sequences"][0]) > 0
