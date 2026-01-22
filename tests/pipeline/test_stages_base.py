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

import pyarrow
import pytest

from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class MockRowUDF(StatefulStageUDF):
    """Mock UDF for row-based processing tests."""

    async def udf_for_rows(self, rows):
        """Simple UDF that adds a computed column to each row."""
        for row in rows:
            # Compute new value
            row["computed_value"] = row.get("value", 0) * 2
            # Must preserve and yield the __idx_in_batch column for proper alignment
            yield row


class MockBatchUDF(StatefulStageUDF):
    """Mock UDF for batch-based processing tests."""

    async def udf_for_batch(self, batch):
        """Simple UDF that operates on the entire batch."""
        output = batch.copy()
        output["computed_values"] = [v * 2 for v in batch["value"]]
        return output


class TestStatefulStageUDF:
    """Test suite for StatefulStageUDF class."""

    def test_row_based_processing_transforms_data_correctly(self):
        """Test: Row-based processing correctly transforms columnar to row format and back.

        This test verifies that:
        - Columnar batch data is converted to row format
        - Each row is processed by udf_for_rows
        - Output is transformed back to columnar format
        - Original columns are preserved (update_row=True)
        - New columns are added
        """

        async def run_test():
            udf = MockRowUDF(compute_by_rows=True,
                             expected_input_keys=["value"],
                             update_row=True)

            # Create a batch with columnar format
            batch = {
                "value": [1, 2, 3],
                "name": ["a", "b", "c"],
                "__record_id": [10, 20, 30]
            }

            # Process the batch
            results = []
            async for output in udf(batch):
                results.append(output)

            # Verify single output batch
            assert len(results) == 1
            output = results[0]

            # Verify output structure
            assert "value" in output
            assert "name" in output
            assert "computed_value" in output
            assert "__record_id" in output

            # Verify values
            assert output["value"] == [1, 2, 3]
            assert output["name"] == ["a", "b", "c"]
            assert output["computed_value"] == [2, 4, 6]  # value * 2
            assert output["__record_id"] == [10, 20, 30]

        asyncio.run(run_test())

    def test_batch_based_processing_operates_on_whole_batch(self):
        """Test: Batch-based processing operates on entire batch without row conversion.

        This test verifies that:
        - Data is processed directly in columnar format
        - No row transformation occurs
        - Record IDs are preserved
        """

        async def run_test():
            udf = MockBatchUDF(compute_by_rows=False,
                               expected_input_keys=["value"])

            # Create a batch with columnar format
            batch = {
                "value": [5, 10, 15],
                "category": ["x", "y", "z"],
                "__record_id": [100, 200, 300]
            }

            # Process the batch
            results = []
            async for output in udf(batch):
                results.append(output)

            # Verify single output batch
            assert len(results) == 1
            output = results[0]

            # Verify output structure
            assert "value" in output
            assert "category" in output
            assert "computed_values" in output
            assert "__record_id" in output

            # Verify values
            assert output["value"] == [5, 10, 15]
            assert output["category"] == ["x", "y", "z"]
            assert output["computed_values"] == [10, 20, 30]  # value * 2
            assert output["__record_id"] == [100, 200, 300]

        asyncio.run(run_test())

    def test_error_rows_are_skipped_but_preserved_in_output(self):
        """Test: Rows with inference errors are skipped during processing but included in output.

        This test verifies that:
        - Rows with __inference_error__ are not processed
        - Error rows maintain their error state in output
        - Normal rows are processed correctly
        - All rows appear in output in correct order
        """

        async def run_test():
            udf = MockRowUDF(compute_by_rows=True,
                             expected_input_keys=["value"],
                             update_row=True)

            # Create a batch with some error rows
            batch = {
                "value": [1, 2, 3, 4],
                "name": ["a", "b", "c", "d"],
                "__record_id": [10, 20, 30, 40],
                "__inference_error__": [
                    {
                        "error_msg": None,
                        "traceback": None
                    },  # Normal
                    {
                        "error_msg": "Error occurred",
                        "traceback": "stack trace"
                    },  # Error
                    {
                        "error_msg": None,
                        "traceback": None
                    },  # Normal
                    {
                        "error_msg": "Another error",
                        "traceback": "another trace"
                    }  # Error
                ]
            }

            # Process the batch
            results = []
            async for output in udf(batch):
                results.append(output)

            # Verify single output batch
            assert len(results) == 1
            output = results[0]

            # Verify error information is preserved
            assert output["__inference_error__"][0]["error_msg"] is None
            assert output["__inference_error__"][1][
                "error_msg"] == "Error occurred"
            assert output["__inference_error__"][2]["error_msg"] is None
            assert output["__inference_error__"][3][
                "error_msg"] == "Another error"

            # Verify normal rows were processed (indices 0 and 2)
            assert output["computed_value"][0] == 2  # 1 * 2
            assert output["computed_value"][2] == 6  # 3 * 2

            # Error rows should not have computed_value or it should be None
            assert output["computed_value"][1] is None
            assert output["computed_value"][3] is None

        asyncio.run(run_test())

    def test_drop_keys_removes_specified_columns_from_output(self):
        """Test: Drop keys functionality removes specified columns from output.

        This test verifies that:
        - Columns specified in drop_keys are removed from output
        - Other columns are preserved
        - Functionality works in row-based mode
        """

        async def run_test():
            udf = MockRowUDF(compute_by_rows=True,
                             expected_input_keys=["value"],
                             drop_keys=["temporary_data"],
                             update_row=True)

            # Create a batch with a column to be dropped
            batch = {
                "value": [1, 2, 3],
                "name": ["a", "b", "c"],
                "temporary_data": ["temp1", "temp2", "temp3"],
                "__record_id": [10, 20, 30]
            }

            # Process the batch
            results = []
            async for output in udf(batch):
                results.append(output)

            # Verify single output batch
            assert len(results) == 1
            output = results[0]

            # Verify temporary_data was dropped
            assert "temporary_data" not in output

            # Verify other columns are preserved
            assert "value" in output
            assert "name" in output
            assert "computed_value" in output
            assert output["value"] == [1, 2, 3]
            assert output["computed_value"] == [2, 4, 6]

        asyncio.run(run_test())

    def test_validation_raises_error_for_missing_required_keys(self):
        """Test: Validation raises ValueError when required input keys are missing.

        This test verifies that:
        - Missing required keys trigger validation error
        - Error message indicates which keys are missing
        - Validation occurs before processing
        """

        async def run_test():
            udf = MockRowUDF(compute_by_rows=True,
                             expected_input_keys=["value", "required_field"],
                             update_row=True)

            # Create a batch missing the required_field
            batch = {
                "value": [1, 2, 3],
                "name": ["a", "b", "c"],
                "__record_id": [10, 20, 30]
            }

            # Verify that processing raises ValueError for missing required key
            with pytest.raises(ValueError) as exc_info:
                async for _ in udf(batch):
                    pass

            # Verify error message mentions missing key
            assert "required_field" in str(exc_info.value)
            assert "Required input keys" in str(exc_info.value)

        asyncio.run(run_test())


class TestStatefulStage:
    """Test suite for StatefulStage Pydantic model."""

    def test_stage_initialization_with_defaults(self):
        """Test that StatefulStage initializes with correct default values."""
        stage = StatefulStage(fn=MockRowUDF)

        assert stage.fn == MockRowUDF
        assert stage.fn_constructor_kwargs == {}
        assert stage.map_batches_kwargs == {"concurrency": 1}
        assert stage.compute_by_rows is True
        assert stage.drop_keys is None
        assert stage.update_row is True

    def test_get_dataset_map_batches_kwargs_merges_configuration(self):
        """Test that get_dataset_map_batches_kwargs correctly merges all configuration."""
        stage = StatefulStage(fn=MockRowUDF,
                              fn_constructor_kwargs={"custom_param": "value"},
                              map_batches_kwargs={"concurrency": 2},
                              compute_by_rows=True,
                              drop_keys=["temp"],
                              update_row=False)

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=100)

        # Verify map_batches_kwargs are included
        assert kwargs["concurrency"] == 2
        assert kwargs["batch_size"] == 100

        # Verify fn_constructor_kwargs includes user params and injected params
        fn_kwargs = kwargs["fn_constructor_kwargs"]
        assert fn_kwargs["custom_param"] == "value"
        assert fn_kwargs["compute_by_rows"] is True
        assert fn_kwargs["drop_keys"] == ["temp"]
        assert fn_kwargs["expected_input_keys"] == []
        assert fn_kwargs["update_row"] is False

    def test_stage_raises_error_if_compute_by_rows_in_constructor_kwargs(self):
        """Test that stage raises error if compute_by_rows is manually set in fn_constructor_kwargs."""
        stage = StatefulStage(
            fn=MockRowUDF,
            fn_constructor_kwargs={"compute_by_rows": True},  # Invalid!
            compute_by_rows=True)

        with pytest.raises(ValueError) as exc_info:
            stage.get_dataset_map_batches_kwargs(batch_size=100)

        assert "compute_by_rows" in str(exc_info.value)
        assert "cannot be used" in str(exc_info.value)

    def test_empty_pyarrow_table_returns_empty_dict(self):
        """Test that empty PyArrow tables are handled gracefully."""

        async def run_test():
            udf = MockRowUDF(compute_by_rows=True)

            # Create empty PyArrow table
            empty_table = pyarrow.table({})

            results = []
            async for output in udf(empty_table):
                results.append(output)

            # Verify empty dict is returned
            assert len(results) == 1
            assert results[0] == {}

        asyncio.run(run_test())

    def test_update_row_false_replaces_instead_of_merges(self):
        """Test that update_row=False replaces row data instead of merging."""

        class ReplaceUDF(StatefulStageUDF):
            """UDF that returns only new data."""

            async def udf_for_rows(self, rows):
                for row in rows:
                    idx = row[self.IDX_IN_BATCH_COLUMN]
                    # Return only new columns, not preserving old ones
                    yield {
                        self.IDX_IN_BATCH_COLUMN: idx,
                        "new_value": row["value"] * 10
                    }

        async def run_test():
            udf = ReplaceUDF(
                compute_by_rows=True,
                expected_input_keys=["value"],
                update_row=False  # Replace mode
            )

            batch = {
                "value": [1, 2, 3],
                "old_column": ["a", "b", "c"],
                "__record_id": [10, 20, 30]
            }

            results = []
            async for output in udf(batch):
                results.append(output)

            assert len(results) == 1
            output = results[0]

            # Verify old columns are not preserved (replaced)
            assert "old_column" not in output or all(
                v is None for v in output.get("old_column", []))

            # Verify new value is present
            assert "new_value" in output
            assert output["new_value"] == [10, 20, 30]

            # Verify __record_id is still preserved (special handling)
            assert output["__record_id"] == [10, 20, 30]

        asyncio.run(run_test())
