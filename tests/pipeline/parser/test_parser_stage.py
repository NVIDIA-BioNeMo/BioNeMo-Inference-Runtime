# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pickle
from pathlib import Path

import pytest

from tensorrt_bionemo.data.schemas import InputRequest, MSARecord, Polymer
from tensorrt_bionemo.pipeline.stages.base import StatefulStageUDF
from tensorrt_bionemo.pipeline.stages.configs import ParserStageConfig
from tensorrt_bionemo.pipeline.stages.parser_stage import (ParserStage,
                                                           ParserUDF)


def _unpack_columnar(output):
    """Unpack DATA_COLUMN format back to flat columnar dict for test assertions."""
    data_col = output.get(StatefulStageUDF.DATA_COLUMN)
    if data_col is None:
        return output
    rows = [pickle.loads(d) if isinstance(d, bytes) else d for d in data_col]
    n = len(rows)
    flat = {
        "__inference_error__":
        output.get("__inference_error__", [None] * n),
        "__record_id":
        output.get(StatefulStageUDF.RECORD_ID_IN_BATCH_COLUMN, [None] * n),
    }
    all_keys: set = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in all_keys:
        flat[key] = [row.get(key) for row in rows]
    return flat


SAMPLES_DIR = Path(
    __file__).parent.parent.parent.parent / "examples" / "data" / "samples"


class TestParserStageConfiguration:

    def test_stage_has_correct_fn_class(self):
        stage = ParserStage(fn=ParserUDF)
        assert stage.fn == ParserUDF

    def test_required_input_keys(self):
        stage = ParserStage(fn=ParserUDF)
        required_keys = stage.get_required_input_keys()

        assert "record" in required_keys
        assert isinstance(required_keys["record"], str)

    def test_stage_initialization_with_defaults(self):
        stage = ParserStage(fn=ParserUDF)

        assert stage.fn == ParserUDF
        assert stage.fn_constructor_kwargs == {}
        assert stage.compute_by_rows is True

    def test_stage_configuration_with_custom_compute(self):
        stage = ParserStage(
            fn=ParserUDF,
            map_batches_kwargs={"concurrency": 4},
        )

        assert stage.map_batches_kwargs["concurrency"] == 4


class TestParserStageConfig:

    def test_default_config(self):
        config = ParserStageConfig()

        assert config.compute is None
        assert config.compute_by_rows is True
        assert config.enabled is True

    def test_config_with_custom_compute(self):
        config = ParserStageConfig(compute=8)

        assert config.compute == 8

    def test_config_with_compute_by_rows_false(self):
        config = ParserStageConfig(compute_by_rows=False)

        assert config.compute_by_rows is False


class TestParserStageGetDatasetKwargs:

    def test_get_dataset_map_batches_kwargs(self):
        stage = ParserStage(
            fn=ParserUDF,
            fn_constructor_kwargs={},
            compute_by_rows=True,
            drop_keys=["temp_data"],
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=32)

        assert kwargs["batch_size"] == 32
        assert kwargs["fn_constructor_kwargs"]["compute_by_rows"] is True
        assert kwargs["fn_constructor_kwargs"]["drop_keys"] == ["temp_data"]

    def test_get_dataset_map_batches_kwargs_with_expected_keys(self):
        stage = ParserStage(fn=ParserUDF)

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "expected_input_keys" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["expected_input_keys"] == [
            "record"
        ]


class TestParserUDFAsyncBatchProcessing:

    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    def test_batch_processing_single_row(self, parser_udf):

        async def run_batch():
            batch = {
                "record": [
                    dict(
                        InputRequest(input_id="r1",
                                     polymers=[
                                         Polymer(chain_id="A",
                                                 sequence="ACDEFGHIKL")
                                     ])),
                ],
                "__record_id": ["r1"],
            }

            results = []
            async for output in parser_udf(batch):
                results.append(output)
            return results

        results = asyncio.run(run_batch())

        assert len(results) == 1
        output = _unpack_columnar(results[0])
        assert "parsed" in output
        assert len(output["parsed"]) == 1

    def test_batch_processing_multiple_rows(self, parser_udf):

        async def run_batch():
            batch = {
                "record": [
                    dict(
                        InputRequest(
                            input_id="r1",
                            polymers=[Polymer(chain_id="A",
                                              sequence="ACDEF")])),
                    dict(
                        InputRequest(
                            input_id="r2",
                            polymers=[Polymer(chain_id="A",
                                              sequence="GHIKL")])),
                    dict(
                        InputRequest(
                            input_id="r3",
                            polymers=[Polymer(chain_id="A",
                                              sequence="MNPQR")])),
                ],
                "__record_id": ["r1", "r2", "r3"],
            }

            results = []
            async for output in parser_udf(batch):
                results.append(output)
            return results

        results = asyncio.run(run_batch())

        assert len(results) == 1
        output = _unpack_columnar(results[0])
        assert len(output["parsed"]) == 3

    def test_batch_preserves_record_ids(self, parser_udf):

        async def run_batch():
            batch = {
                "record": [
                    dict(
                        InputRequest(
                            input_id="id_100",
                            polymers=[Polymer(chain_id="A",
                                              sequence="ACDEF")])),
                    dict(
                        InputRequest(
                            input_id="id_200",
                            polymers=[Polymer(chain_id="A",
                                              sequence="GHIKL")])),
                ],
                "__record_id": ["id_100", "id_200"],
            }

            results = []
            async for output in parser_udf(batch):
                results.append(output)
            return results

        results = asyncio.run(run_batch())

        assert results[0]["__record_id"] == ["id_100", "id_200"]


class TestParserOutputSchema:

    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    def test_output_contains_polymers(self, parser_udf):
        """Test that parsed output contains polymers with sequence info."""
        request = InputRequest(
            input_id="schema_test",
            polymers=[Polymer(chain_id="A", sequence="ACDEFGHIKL")])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "polymers" in result["parsed"]
        assert len(result["parsed"]["polymers"]) == 1
        assert result["parsed"]["polymers"][0]["sequence"] == "ACDEFGHIKL"

    def test_output_msa_structure_in_polymer(self, parser_udf):
        """Test that MSAs are stored within each polymer as MSAParsed."""
        a3m_content = ">query\nACDEF\n>hit1\nACDEF\n"
        request = InputRequest(
            input_id="msa_schema_test",
            polymers=[
                Polymer(chain_id="A",
                        sequence="ACDEF",
                        msas=[MSARecord(content=a3m_content)]),
                Polymer(chain_id="B",
                        sequence="GHIKL",
                        msas=[MSARecord(content=a3m_content)]),
            ])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert len(result["parsed"]["polymers"]) == 2
        assert result["parsed"]["polymers"][0]["msas"] is not None
        assert result["parsed"]["polymers"][1]["msas"] is not None
        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = result["parsed"]["polymers"][0]["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert len(msa_parsed["sequences"]) == 2
        assert msa_parsed["sequences"][0] == "ACDEF"
        assert msa_parsed["descriptions"][0] == "query"

    def test_output_input_id_preserved(self, parser_udf):
        """Test that input_id is preserved in parsed output."""
        request = InputRequest(input_id="test_id_123", polymers=[])
        row = {"record": dict(request), "__record_id": "test"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert result["parsed"]["input_id"] == "test_id_123"


class TestParserWithRealSampleFiles:

    @pytest.fixture
    def parser_udf(self):
        return ParserUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["record"],
            update_row=True,
        )

    @pytest.mark.skipif(not (SAMPLES_DIR / "T1031.fasta").exists(),
                        reason="Sample FASTA file not found")
    def test_parse_with_real_fasta_sequence(self, parser_udf):
        from tensorrt_bionemo.data.parsers.fasta import read_fasta

        fasta_path = SAMPLES_DIR / "T1031.fasta"
        parsed_fasta = read_fasta(fasta_path)
        sequence = parsed_fasta["sequences"][0]["sequence"]

        request = InputRequest(
            input_id="T1031",
            polymers=[Polymer(chain_id="A", sequence=sequence)])
        row = {"record": dict(request), "__record_id": "T1031"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        assert result["parsed"]["polymers"] is not None
        assert len(result["parsed"]["polymers"]) == 1

    @pytest.mark.skipif(not (SAMPLES_DIR / "msas" / "T1031.a3m").exists(),
                        reason="Sample A3M file not found")
    def test_parse_with_real_msa_file(self, parser_udf):
        a3m_path = str(SAMPLES_DIR / "msas" / "T1031.a3m")

        request = InputRequest(input_id="T1031_msa",
                               polymers=[
                                   Polymer(chain_id="A",
                                           sequence="ACDEFGHIKLMNPQRSTVWY",
                                           msas=[MSARecord(path=a3m_path)])
                               ])
        row = {"record": dict(request), "__record_id": "T1031"}

        result = asyncio.run(parser_udf.udf_for_item(row))

        assert "parsed" in result
        polymer = result["parsed"]["polymers"][0]
        assert polymer["msas"] is not None
        # MSAParsed is MSAParsed with sequences, raw, descriptions
        msa_parsed = polymer["msas"][0]
        assert isinstance(msa_parsed, dict)
        assert "sequences" in msa_parsed
        assert len(msa_parsed["sequences"]) > 0
        assert msa_parsed["sequences"][0] is not None
        assert len(msa_parsed["sequences"][0]) > 0
