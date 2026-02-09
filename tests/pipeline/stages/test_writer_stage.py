# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensorrt_bionemo.pipeline.stages.writer_stage import (WriterStage,
                                                           WriterUDF)


@pytest.fixture
def sample_row():
    n_res = 64
    return {
        "atom_positions": np.random.randn(n_res, 37, 3),
        "residue_types": np.random.randint(0, 20, n_res),
        "atom_mask": np.ones((n_res, 37)),
        "residue_indices": np.arange(1, n_res + 1),
        "b_factors": np.random.rand(n_res, 37) * 100,
        "chain_indices": np.zeros(n_res, dtype=np.int64),
        "__record_id": "test_protein",
        "__idx_in_batch": 0,
    }


@pytest.fixture
def writer_udf():
    return WriterUDF(
        compute_by_rows=True,
        drop_keys=[],
        expected_input_keys=[
            "atom_positions", "residue_types", "atom_mask", "residue_indices"
        ],
        update_row=False,
        mappings={},
        format="pdb",
        output_path=None,
    )


class TestWriterUDFInit:

    def test_init_with_defaults(self):
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},
        )
        assert udf.format == "pdb"
        assert udf.output_path is None
        assert udf.mappings == {}

    def test_init_with_custom_format_pdb(self):
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},
            format="pdb",
        )
        assert udf.format == "pdb"

    def test_init_with_custom_format_cif(self):
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},
            format="cif",
        )
        assert udf.format == "cif"

    def test_init_with_output_path(self):
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},
            output_path="/tmp/output",
        )
        assert udf.output_path == "/tmp/output"

    def test_init_with_mappings(self):
        mappings = {"res_type_mapping": {0: "ALA"}}
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings=mappings,
        )
        assert udf.mappings == mappings


class TestWriterUDFGetWriter:

    def test_get_writer_pdb_format(self):
        from tensorrt_bionemo.data.schemas.basic import AtomTypes, ResTypes
        from tensorrt_bionemo.data.writers import PDBWriter
        basic_20 = ResTypes.basic_20_residue_types()
        res_type_mapping = {i: basic_20[i] for i in range(len(basic_20))}
        all_atom_types = AtomTypes.all_types()
        atom_type_mapping = {
            i: all_atom_types[i]
            for i in range(len(all_atom_types))
        }
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={
                "res_type_mapping": res_type_mapping,
                "atom_type_mapping": atom_type_mapping
            },
            format="pdb",
        )
        writer, ext = udf._get_writer_and_ext()
        assert ext == ".pdb"
        assert isinstance(writer, PDBWriter)

    def test_get_writer_cif_format(self):
        from tensorrt_bionemo.data.schemas.basic import AtomTypes, ResTypes
        from tensorrt_bionemo.data.writers import CIFWriter
        basic_20 = ResTypes.basic_20_residue_types()
        res_type_mapping = {i: basic_20[i] for i in range(len(basic_20))}
        all_atom_types = AtomTypes.all_types()
        atom_type_mapping = {
            i: all_atom_types[i]
            for i in range(len(all_atom_types))
        }
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={
                "res_type_mapping": res_type_mapping,
                "atom_type_mapping": atom_type_mapping
            },
            format="cif",
        )
        writer, ext = udf._get_writer_and_ext()
        assert ext == ".cif"
        assert isinstance(writer, CIFWriter)

    def test_get_writer_with_default_mappings(self):
        """Test that writers work without explicit mappings (use defaults)."""
        from tensorrt_bionemo.data.writers import PDBWriter
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},  # No mappings provided
            format="pdb",
        )
        writer, ext = udf._get_writer_and_ext()
        assert ext == ".pdb"
        assert isinstance(writer, PDBWriter)
        # Verify writer has default mappings from BaseWriter
        assert writer.res_type_mapping is not None
        assert writer.atom_type_mapping is not None

    def test_get_writer_invalid_format(self):
        udf = WriterUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=[],
            update_row=False,
            mappings={},
            format="invalid",
        )
        with pytest.raises(ValueError,
                           match="Invalid format.*Supported formats"):
            udf._get_writer_and_ext()


class TestWriterUDFProcessing:

    def test_udf_for_item_returns_output_dict(self, writer_udf, sample_row):
        with patch.object(writer_udf,
                          '_get_writer_and_ext') as mock_get_writer:
            mock_writer = MagicMock()
            mock_writer.write.return_value = "ATOM..."
            mock_get_writer.return_value = (mock_writer, ".pdb")

            result = asyncio.run(writer_udf.udf_for_item(sample_row))

            assert "output_path" in result
            assert "format" in result
            assert "output_raw" in result

    def test_udf_for_item_format_in_output(self, writer_udf, sample_row):
        with patch.object(writer_udf,
                          '_get_writer_and_ext') as mock_get_writer:
            mock_writer = MagicMock()
            mock_writer.write.return_value = "ATOM..."
            mock_get_writer.return_value = (mock_writer, ".pdb")

            result = asyncio.run(writer_udf.udf_for_item(sample_row))

            assert result["format"] == "pdb"

    def test_udf_for_item_with_output_path(self, sample_row):
        with tempfile.TemporaryDirectory() as tmpdir:
            udf = WriterUDF(
                compute_by_rows=True,
                drop_keys=[],
                expected_input_keys=[],
                update_row=False,
                mappings={},
                format="pdb",
                output_path=tmpdir,
            )

            with patch.object(udf, '_get_writer_and_ext') as mock_get_writer:
                mock_writer = MagicMock()
                mock_writer.write.return_value = "ATOM..."
                mock_get_writer.return_value = (mock_writer, ".pdb")

                result = asyncio.run(udf.udf_for_item(sample_row))

                expected_path = os.path.join(tmpdir, "test_protein.pdb")
                assert result["output_path"] == expected_path

    def test_udf_for_item_without_record_id(self, sample_row):
        del sample_row["__record_id"]

        with tempfile.TemporaryDirectory() as tmpdir:
            udf = WriterUDF(
                compute_by_rows=True,
                drop_keys=[],
                expected_input_keys=[],
                update_row=False,
                mappings={},
                format="pdb",
                output_path=tmpdir,
            )

            with patch.object(udf, '_get_writer_and_ext') as mock_get_writer:
                mock_writer = MagicMock()
                mock_writer.write.return_value = "ATOM..."
                mock_get_writer.return_value = (mock_writer, ".pdb")

                result = asyncio.run(udf.udf_for_item(sample_row))

                expected_path = os.path.join(tmpdir, "0.pdb")
                assert result["output_path"] == expected_path

    def test_udf_for_item_cif_format(self, sample_row):
        """Test that CIF format produces correct output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            udf = WriterUDF(
                compute_by_rows=True,
                drop_keys=[],
                expected_input_keys=[],
                update_row=False,
                mappings={},
                format="cif",
                output_path=tmpdir,
            )

            with patch.object(udf, '_get_writer_and_ext') as mock_get_writer:
                mock_writer = MagicMock()
                mock_writer.write.return_value = "data_structure\n#..."
                mock_get_writer.return_value = (mock_writer, ".cif")

                result = asyncio.run(udf.udf_for_item(sample_row))

                expected_path = os.path.join(tmpdir, "test_protein.cif")
                assert result["output_path"] == expected_path
                assert result["format"] == "cif"


class TestWriterUDFErrorHandling:

    def test_on_row_error_returns_none_values(self, writer_udf, sample_row):
        error = ValueError("Test error")
        result = writer_udf.on_row_error(sample_row, error)

        assert result["output_path"] is None
        assert result["output_raw"] is None
        assert result["format"] == "pdb"


class TestWriterStage:

    def test_stage_fn_is_writer_udf(self):
        stage = WriterStage()
        assert stage.fn == WriterUDF

    def test_stage_update_row_is_false(self):
        stage = WriterStage()
        assert stage.update_row is False

    def test_get_required_input_keys(self):
        stage = WriterStage()
        required_keys = stage.get_required_input_keys()

        assert "atom_positions" in required_keys
        assert "residue_types" in required_keys
        assert "atom_mask" in required_keys
        assert "residue_indices" in required_keys
