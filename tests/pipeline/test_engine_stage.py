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

# isort: off
import asyncio
from unittest.mock import Mock, patch

import pytest

from tensorrt_bionemo.pipeline.stages.engine_stage import (FoldingEngineStage,
                                                           FoldingEngineUDF,
                                                           FoldingEngineWrapper
                                                           )
# isort: on


class TestFoldingEngineWrapper:
    """Test suite for FoldingEngineWrapper class."""

    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.DeviceConfig')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.EngineConfig')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.get_model_class')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.get_postprocessor')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngine')
    def test_initialization_creates_engine_with_correct_config(
            self, mock_folding_engine, mock_get_postprocessor,
            mock_get_model_class, mock_engine_config_cls,
            mock_device_config_cls):
        """Test that FoldingEngineWrapper initializes with correct configuration.

        This test verifies that:
        - Model class and postprocessor are retrieved from registries
        - Pretrained config is loaded if not provided
        - Engine is created with correct parameters
        """
        # Setup mocks
        mock_model_class = Mock()
        mock_model_config = Mock(max_batch_size=8)
        mock_postprocessor_class = Mock()
        mock_device_config = Mock()
        mock_engine_config = Mock()

        mock_get_model_class.return_value = mock_model_class
        mock_model_class.get_pretrained_config.return_value = mock_model_config
        mock_get_postprocessor.return_value = mock_postprocessor_class
        mock_device_config_cls.return_value = mock_device_config
        mock_engine_config_cls.return_value = mock_engine_config

        # Create wrapper
        engine_kwargs = {
            "accelerated_configs": {
                "enable_opt": True
            },
            "postprocessor_config": {
                "param": "value"
            }
        }
        wrapper = FoldingEngineWrapper(model="test_model",
                                       engine_kwargs=engine_kwargs,
                                       max_pending_requests=10)

        # Verify model class and postprocessor were retrieved
        mock_get_model_class.assert_called_once_with("test_model")
        mock_get_postprocessor.assert_called_once_with("test_model")

        # Verify pretrained config was loaded
        mock_model_class.get_pretrained_config.assert_called_once_with(
            "test_model")

        # Verify EngineConfig was created with correct parameters
        mock_engine_config_cls.assert_called_once_with(
            name="test_model",
            model=mock_model_config,
            device=mock_device_config,
            accelerated={"enable_opt": True},
            postprocessor={"param": "value"})

        # Verify FoldingEngine was created
        mock_folding_engine.assert_called_once_with(mock_engine_config,
                                                    mock_model_class,
                                                    mock_postprocessor_class)

        # Verify max_pending_requests and model_config are set
        assert wrapper.max_pending_requests == 10
        assert wrapper.model_config == mock_model_config

    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.DeviceConfig')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.EngineConfig')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.get_model_class')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.get_postprocessor')
    @patch('tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngine')
    def test_predict_async_executes_single_row(self, mock_folding_engine,
                                               mock_get_postprocessor,
                                               mock_get_model_class,
                                               mock_engine_config_cls,
                                               mock_device_config_cls):
        """Test that predict_async executes prediction for a single row.

        This test verifies that:
        - Single row prediction works correctly
        - Execution time is measured
        - Output format matches expected structure
        """
        # Setup mocks
        mock_model_class = Mock()
        mock_model_config = Mock(max_batch_size=8)
        mock_device_config = Mock()
        mock_engine_config = Mock()

        mock_model_class.get_pretrained_config.return_value = mock_model_config
        mock_get_model_class.return_value = mock_model_class
        mock_get_postprocessor.return_value = Mock()
        mock_device_config_cls.return_value = mock_device_config
        mock_engine_config_cls.return_value = mock_engine_config

        mock_engine_instance = Mock()
        mock_prediction = {"structure": "xyz", "confidence": 0.95}
        mock_engine_instance.execute.return_value = mock_prediction
        mock_folding_engine.return_value = mock_engine_instance

        # Create wrapper
        wrapper = FoldingEngineWrapper(model="test_model", engine_kwargs={})

        # Execute prediction
        async def run_test():
            rows = [{"sequence": "ACGT"}]
            outputs, time_takens = await wrapper.predict_async(rows)

            # Verify engine.execute was called with correct input
            mock_engine_instance.execute.assert_called_once_with(
                {"sequence": "ACGT"})

            # Verify output structure
            assert len(outputs) == 1
            assert outputs[0] == mock_prediction
            assert len(time_takens) == 1
            assert isinstance(time_takens[0], float)
            assert time_takens[0] >= 0

        asyncio.run(run_test())


class TestFoldingEngineUDF:
    """Test suite for FoldingEngineUDF class."""

    @patch(
        'tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngineWrapper')
    def test_successful_prediction_returns_correct_output_structure(
            self, mock_wrapper_class):
        """Test that successful prediction returns correct output structure.

        This test verifies that:
        - Predictions are executed for each row
        - Output includes all expected fields
        - Error tracking shows no errors
        - Index in batch is preserved
        """
        # Setup mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 2
        mock_wrapper_class.return_value = mock_wrapper

        # Mock successful prediction
        async def mock_predict_async(rows):
            outputs = []
            times = []
            for _ in rows:
                outputs.append({"structure": "ATOM...", "confidence": 0.92})
                times.append(0.5)
            return outputs, times

        mock_wrapper.predict_async = mock_predict_async

        # Create UDF
        udf = FoldingEngineUDF(compute_by_rows=True,
                               drop_keys=[],
                               expected_input_keys=["sequence"],
                               update_row=False,
                               model="test_model",
                               engine_kwargs={},
                               should_continue_on_error=False)

        # Execute prediction
        async def run_test():
            batch = [
                {
                    "sequence": "ACGT",
                    "__idx_in_batch": 0
                },
                {
                    "sequence": "TGCA",
                    "__idx_in_batch": 1
                },
            ]

            results = []
            async for output in udf.udf_for_rows(batch):
                results.append(output)

            # Verify we got 2 outputs
            assert len(results) == 2

            # Verify output structure for first result
            assert "structure" in results[0]
            assert "confidence" in results[0]
            assert "time_taken" in results[0]
            assert "__inference_error__" in results[0]
            assert results[0]["__idx_in_batch"] == 0

            # Verify no errors
            assert results[0]["__inference_error__"]["error_msg"] is None
            assert results[0]["__inference_error__"]["traceback"] is None
            assert results[1]["__inference_error__"]["error_msg"] is None

        asyncio.run(run_test())

    @patch(
        'tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngineWrapper')
    def test_error_handling_raises_when_should_continue_on_error_false(
            self, mock_wrapper_class):
        """Test that errors raise ValueError when should_continue_on_error is False.

        This test verifies that:
        - Exceptions during prediction are caught
        - ValueError is raised with appropriate message
        - Processing stops on error
        """
        # Setup mock wrapper that raises an error
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 2
        mock_wrapper_class.return_value = mock_wrapper

        async def mock_predict_async_with_error(rows):
            raise RuntimeError("Model inference failed")

        mock_wrapper.predict_async = mock_predict_async_with_error

        # Create UDF with should_continue_on_error=False
        udf = FoldingEngineUDF(compute_by_rows=True,
                               drop_keys=[],
                               expected_input_keys=["sequence"],
                               update_row=False,
                               model="test_model",
                               engine_kwargs={},
                               should_continue_on_error=False)

        # Execute prediction and expect error
        async def run_test():
            batch = [{"sequence": "ACGT", "__idx_in_batch": 0}]

            with pytest.raises(ValueError) as exc_info:
                async for _ in udf.udf_for_rows(batch):
                    pass

            # Verify error message
            assert "Error predicting folding output" in str(exc_info.value)
            assert "Model inference failed" in str(exc_info.value)

        asyncio.run(run_test())

    @patch(
        'tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngineWrapper')
    def test_error_handling_continues_when_should_continue_on_error_true(
            self, mock_wrapper_class):
        """Test that errors are captured but processing continues when should_continue_on_error is True.

        This test verifies that:
        - Exceptions during prediction are caught and recorded
        - Processing continues for all rows
        - Error information is included in output
        - Traceback is captured
        """
        # Setup mock wrapper that raises an error
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 2
        mock_wrapper_class.return_value = mock_wrapper

        async def mock_predict_async_with_error(rows):
            raise RuntimeError("Model inference failed")

        mock_wrapper.predict_async = mock_predict_async_with_error

        # Create UDF with should_continue_on_error=True
        udf = FoldingEngineUDF(compute_by_rows=True,
                               drop_keys=[],
                               expected_input_keys=["sequence"],
                               update_row=False,
                               model="test_model",
                               engine_kwargs={},
                               should_continue_on_error=True)

        # Execute prediction
        async def run_test():
            batch = [
                {
                    "sequence": "ACGT",
                    "__idx_in_batch": 0
                },
                {
                    "sequence": "TGCA",
                    "__idx_in_batch": 1
                },
            ]

            results = []
            async for output in udf.udf_for_rows(batch):
                results.append(output)

            # Verify we got 2 outputs even with errors
            assert len(results) == 2

            # Verify error information is captured
            assert "__inference_error__" in results[0]
            assert results[0]["__inference_error__"][
                "error_msg"] == "RuntimeError: Model inference failed"
            assert results[0]["__inference_error__"]["traceback"] is not None
            assert "RuntimeError" in results[0]["__inference_error__"][
                "traceback"]

            # Verify index is preserved
            assert results[0]["__idx_in_batch"] == 0
            assert results[1]["__idx_in_batch"] == 1

        asyncio.run(run_test())

    @patch(
        'tensorrt_bionemo.pipeline.stages.engine_stage.FoldingEngineWrapper')
    def test_batching_splits_large_batches_into_sub_batches(
            self, mock_wrapper_class):
        """Test that large batches are split into sub-batches based on max_batch_size.

        This test verifies that:
        - Batches larger than max_batch_size are split
        - Multiple async tasks are created for sub-batches
        - All rows are processed
        - Results are yielded as they complete
        """
        # Setup mock wrapper with max_batch_size=2
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 2
        mock_wrapper_class.return_value = mock_wrapper

        # Track calls to predict_async to verify batching
        predict_calls = []

        async def mock_predict_async(rows):
            predict_calls.append(len(rows))
            outputs = []
            times = []
            for _ in rows:
                outputs.append({"structure": "ATOM...", "confidence": 0.92})
                times.append(0.5)
            return outputs, times

        mock_wrapper.predict_async = mock_predict_async

        # Create UDF
        udf = FoldingEngineUDF(compute_by_rows=True,
                               drop_keys=[],
                               expected_input_keys=["sequence"],
                               update_row=False,
                               model="test_model",
                               engine_kwargs={},
                               should_continue_on_error=False)

        # Execute prediction with 5 rows (should split into 3 batches: 2, 2, 1)
        async def run_test():
            batch = [{
                "sequence": f"SEQ{i}",
                "__idx_in_batch": i
            } for i in range(5)]

            results = []
            async for output in udf.udf_for_rows(batch):
                results.append(output)

            # Verify all 5 rows were processed
            assert len(results) == 5

            # Verify batching: 3 calls with sizes [2, 2, 1]
            assert len(predict_calls) == 3
            assert predict_calls[0] == 2  # First sub-batch
            assert predict_calls[1] == 2  # Second sub-batch
            assert predict_calls[2] == 1  # Third sub-batch

            # Verify all indices are present
            result_indices = sorted([r["__idx_in_batch"] for r in results])
            assert result_indices == [0, 1, 2, 3, 4]

        asyncio.run(run_test())


class TestFoldingEngineStage:
    """Test suite for FoldingEngineStage configuration."""

    def test_stage_initialization_sets_gpu_requirements(self):
        """Test that FoldingEngineStage correctly configures Ray remote args with GPU requirements.

        This test verifies that:
        - num_gpus is set to 1 in ray_remote_args
        - accelerator_type is passed through if provided
        - Configuration is injected into map_batches_kwargs
        """
        # Test with accelerator_type
        stage_values = {
            "fn": FoldingEngineUDF,
            "map_batches_kwargs": {
                "accelerator_type": "cuda",
                "concurrency": 2
            },
            "fn_constructor_kwargs": {
                "model": "test_model",
                "engine_kwargs": {}
            }
        }

        # Call the validator
        result = FoldingEngineStage.post_init(stage_values)

        # Verify GPU configuration
        assert "num_gpus" in result["map_batches_kwargs"]
        assert result["map_batches_kwargs"]["num_gpus"] == 1
        assert result["map_batches_kwargs"]["accelerator_type"] == "cuda"
        assert result["map_batches_kwargs"]["concurrency"] == 2

    def test_stage_initialization_without_accelerator_type(self):
        """Test that FoldingEngineStage works without accelerator_type specified."""
        stage_values = {
            "fn": FoldingEngineUDF,
            "map_batches_kwargs": {
                "concurrency": 1
            },
            "fn_constructor_kwargs": {
                "model": "test_model",
                "engine_kwargs": {}
            }
        }

        # Call the validator
        result = FoldingEngineStage.post_init(stage_values)

        # Verify GPU configuration is added
        assert "num_gpus" in result["map_batches_kwargs"]
        assert result["map_batches_kwargs"]["num_gpus"] == 1

        # Verify accelerator_type is not added if not present
        assert "accelerator_type" not in result[
            "map_batches_kwargs"] or result["map_batches_kwargs"].get(
                "accelerator_type") == ""
