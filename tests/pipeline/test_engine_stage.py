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
from typing import Any
from collections.abc import AsyncIterator
from unittest.mock import Mock, patch

import pytest
import ray

from bionemo_ir.pipeline.processor.utils import get_available_gpu_count
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF, unpack_pipeline_row
from bionemo_ir.pipeline.stages.configs import ParallelismMode
from bionemo_ir.pipeline.stages.engine_stage import (
    FoldingEngineStage,
    FoldingEngineUDF,
    FoldingEngineWrapper,
    FoldingPredictionError,
)
# isort: on

# Minimum GPUs required for multi-GPU replica tests
MIN_GPUS_FOR_MULTI_GPU_REPLICA = 2


class TestFoldingEngineWrapper:
    """Test suite for FoldingEngineWrapper class."""

    @patch("bionemo_ir.pipeline.stages.engine_stage.DeviceConfig")
    @patch("bionemo_ir.pipeline.stages.engine_stage.EngineConfig")
    @patch("bionemo_ir.pipeline.stages.engine_stage.get_model_class")
    @patch("bionemo_ir.pipeline.stages.engine_stage.get_postprocessor")
    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngine")
    def test_initialization_creates_engine_with_correct_config(
        self,
        mock_folding_engine,
        mock_get_postprocessor,
        mock_get_model_class,
        mock_engine_config_cls,
        mock_device_config_cls,
    ):
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
        engine_kwargs = {"accelerated_configs": {"enable_opt": True}, "postprocessor_config": {"param": "value"}}
        wrapper = FoldingEngineWrapper(model="test_model", engine_kwargs=engine_kwargs, max_pending_requests=10)

        # Verify model class and postprocessor were retrieved
        mock_get_model_class.assert_called_once_with("test_model")
        mock_get_postprocessor.assert_called_once_with("test_model")

        # Verify pretrained config was loaded
        mock_model_class.get_pretrained_config.assert_called_once_with("test_model")

        # Verify EngineConfig was created with correct parameters
        mock_engine_config_cls.assert_called_once_with(
            name="test_model",
            model=mock_model_config,
            device=mock_device_config,
            accelerated={"enable_opt": True},
            postprocessor={"param": "value"},
            profile_inference=False,
        )

        # Verify FoldingEngine was created
        mock_folding_engine.assert_called_once_with(
            mock_engine_config, mock_model_class, mock_postprocessor_class, runtime_args=None
        )

        # Verify max_pending_requests and model_config are set
        assert wrapper.max_pending_requests == 10
        assert wrapper.model_config == mock_model_config

    @patch("bionemo_ir.pipeline.stages.engine_stage.DeviceConfig")
    @patch("bionemo_ir.pipeline.stages.engine_stage.EngineConfig")
    @patch("bionemo_ir.pipeline.stages.engine_stage.get_model_class")
    @patch("bionemo_ir.pipeline.stages.engine_stage.get_postprocessor")
    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngine")
    def test_predict_async_executes_single_row(
        self,
        mock_folding_engine,
        mock_get_postprocessor,
        mock_get_model_class,
        mock_engine_config_cls,
        mock_device_config_cls,
    ):
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
            mock_engine_instance.execute.assert_called_once_with({"sequence": "ACGT"})

            # Verify output structure
            assert len(outputs) == 1
            assert outputs[0] == mock_prediction
            assert len(time_takens) == 1
            assert isinstance(time_takens[0], float)
            assert time_takens[0] >= 0

        asyncio.run(run_test())


class TestFoldingEngineUDF:
    """Test suite for FoldingEngineUDF class."""

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_successful_prediction_returns_correct_output_structure(self, mock_wrapper_class):
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
        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=False,
        )

        # Execute prediction
        async def run_test():
            batch = [
                {"sequence": "ACGT", "__idx_in_batch": 0},
                {"sequence": "TGCA", "__idx_in_batch": 1},
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

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_error_handling_raises_when_should_continue_on_error_false(self, mock_wrapper_class):
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
        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=False,
        )

        # Execute prediction and expect error
        async def run_test():
            batch = [{"sequence": "ACGT", "__idx_in_batch": 0}]

            with pytest.raises(FoldingPredictionError) as exc_info:
                async for _ in udf.udf_for_rows(batch):
                    pass

            # Verify exception chaining preserves original cause
            assert exc_info.value.__cause__ is not None
            assert "Model inference failed" in str(exc_info.value.__cause__)

        asyncio.run(run_test())

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_error_handling_continues_when_should_continue_on_error_true(self, mock_wrapper_class):
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
        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=True,
        )

        # Execute prediction
        async def run_test():
            batch = [
                {"sequence": "ACGT", "__idx_in_batch": 0},
                {"sequence": "TGCA", "__idx_in_batch": 1},
            ]

            results = []
            async for output in udf.udf_for_rows(batch):
                results.append(output)

            # Verify we got 2 outputs even with errors
            assert len(results) == 2

            # Verify error information is captured
            assert "__inference_error__" in results[0]
            assert results[0]["__inference_error__"]["error_msg"] == "RuntimeError: Model inference failed"
            assert results[0]["__inference_error__"]["traceback"] is not None
            assert "RuntimeError" in results[0]["__inference_error__"]["traceback"]

            # Verify index is preserved
            assert results[0]["__idx_in_batch"] == 0
            assert results[1]["__idx_in_batch"] == 1

        asyncio.run(run_test())

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_sticky_cuda_error_fails_the_worker_without_touching_the_device(self, mock_wrapper_class):
        """A sticky CUDA error leaves the context unusable for every later record.

        Continuing produces a cascade of misleading failures at unrelated call sites,
        and ``cleanup``'s ``empty_cache`` raises the same sticky error, burying the
        original traceback. So it must propagate even with
        ``should_continue_on_error=True``, and cleanup must not run.
        """
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 2
        mock_wrapper_class.return_value = mock_wrapper

        async def mock_predict_async_with_ima(rows):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        mock_wrapper.predict_async = mock_predict_async_with_ima

        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=True,
        )

        async def run_test():
            batch = [{"sequence": "ACGT", "__idx_in_batch": 0}]
            with pytest.raises(FoldingPredictionError) as exc_info:
                async for _ in udf.udf_for_rows(batch):
                    pass
            assert "illegal memory access" in str(exc_info.value.__cause__)

        asyncio.run(run_test())
        mock_wrapper.cleanup.assert_not_called()

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_sticky_cuda_error_cancels_pending_sub_batches(self, mock_wrapper_class):
        """A poisoned CUDA context must not keep running later sub-batches.

        ``udf_for_rows`` fans each sub-batch out with ``create_task``. If one of
        them hits a sticky CUDA error, the remaining tasks would otherwise keep
        launching work on that context after ``FoldingPredictionError`` has
        already left ``predict_async``.
        """
        mock_wrapper = Mock()
        mock_wrapper.get_max_batch_size.return_value = 1
        mock_wrapper_class.return_value = mock_wrapper

        release_siblings = asyncio.Event()
        finished_later: list[int] = []

        async def mock_predict_async(rows):
            idx = rows[0]["__idx_in_batch"]
            if idx == 0:
                await asyncio.sleep(0)
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            await release_siblings.wait()
            finished_later.append(idx)
            return [{"structure": "ATOM..."}], [0.0]

        mock_wrapper.predict_async = mock_predict_async

        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=True,
        )

        async def run_test():
            batch = [{"sequence": f"SEQ{i}", "__idx_in_batch": i} for i in range(3)]
            with pytest.raises(FoldingPredictionError) as exc_info:
                async for _ in udf.udf_for_rows(batch):
                    pass
            assert "illegal memory access" in str(exc_info.value.__cause__)
            # The worker loop stays up. Releasing the gate would let any still
            # pending sibling finish if cancellation did not settle them first.
            release_siblings.set()
            await asyncio.sleep(0)
            assert finished_later == []

        asyncio.run(run_test())
        mock_wrapper.cleanup.assert_not_called()

    @patch("bionemo_ir.pipeline.stages.engine_stage.FoldingEngineWrapper")
    def test_batching_splits_large_batches_into_sub_batches(self, mock_wrapper_class):
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
        udf = FoldingEngineUDF(
            compute_by_rows=True,
            drop_keys=[],
            expected_input_keys=["sequence"],
            update_row=False,
            model="test_model",
            engine_kwargs={},
            should_continue_on_error=False,
        )

        # Execute prediction with 5 rows (should split into 3 batches: 2, 2, 1)
        async def run_test():
            batch = [{"sequence": f"SEQ{i}", "__idx_in_batch": i} for i in range(5)]

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
            "map_batches_kwargs": {"accelerator_type": "cuda", "concurrency": 2},
            "fn_constructor_kwargs": {"model": "test_model", "engine_kwargs": {}},
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
            "map_batches_kwargs": {"concurrency": 1},
            "fn_constructor_kwargs": {"model": "test_model", "engine_kwargs": {}},
        }

        # Call the validator
        result = FoldingEngineStage.post_init(stage_values)

        # Verify GPU configuration is added
        assert "num_gpus" in result["map_batches_kwargs"]
        assert result["map_batches_kwargs"]["num_gpus"] == 1

        # Verify accelerator_type is not added if not present
        assert (
            "accelerator_type" not in result["map_batches_kwargs"]
            or result["map_batches_kwargs"].get("accelerator_type") == ""
        )


def _get_nvml_gpu_id(torch_gpu_id):
    """
    Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

    If the latter isn't set return the same id
    """
    import os

    # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        return ids[torch_gpu_id]  # remap
    else:
        return torch_gpu_id


def _get_device_uuid(device_id: int):
    """Return GPU UUID for device_id if pynvml is available and initialized; else fallback."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None
    try:
        import pynvml

        device_id = _get_nvml_gpu_id(device_id)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        return uuid.decode() if isinstance(uuid, bytes) else uuid
    except (ImportError, Exception):
        return None


def _get_current_device_id() -> int:
    """Return current GPU device id for the calling process (e.g. Ray worker)."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except ImportError:
        pass
    return 0


class MockFoldingEngineUDF(StatefulStageUDF):
    """Mock engine UDF for testing replica mode under Ray map_batches.

    Same constructor signature as FoldingEngineUDF so stage kwargs match;
    returns fake prediction outputs without loading a real model.
    Includes device_id so tests can verify the number of GPUs used.
    """

    def __init__(
        self,
        compute_by_rows: bool,
        drop_keys: list[str],
        expected_input_keys: list[str],
        update_row: bool,
        model: str,
        engine_kwargs: dict[str, Any],
        max_pending_requests: Any = None,
        should_continue_on_error: bool = False,
        parallelism_mode: Any = ParallelismMode.REPLICA,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            compute_by_rows=compute_by_rows,
            drop_keys=drop_keys or [],
            expected_input_keys=expected_input_keys or [],
            update_row=update_row,
        )
        self._model = model
        self._max_batch_size = 2

    async def udf_for_rows(self, batch: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        device_id = _get_current_device_id()
        n_iters = (len(batch) + self._max_batch_size - 1) // self._max_batch_size
        for i in range(n_iters):
            start = i * self._max_batch_size
            end = min(start + self._max_batch_size, len(batch))
            sub = batch[start:end]
            device_uuid = _get_device_uuid(device_id)
            for row in sub:
                idx = row.get("__idx_in_batch", 0)
                yield {
                    "structure": "MOCK_ATOM",
                    "confidence": 0.99,
                    "time_taken": 0.01,
                    "device_id": device_id,
                    "device_uuid": device_uuid,
                    "__inference_error__": {"error_msg": None, "traceback": None},
                    "__idx_in_batch": idx,
                }


class TestFoldingEngineStageReplicaMapBatches:
    """Test engine stage replica mode under Ray Dataset map_batches."""

    @pytest.fixture(autouse=True, scope="class")
    def ray_local(self):
        """Run Ray in worker mode (avoids PeekObjectRefStream bug with async UDF in local_mode).

        Class-scoped: a cold ``ray.init``/``ray.shutdown`` pair costs tens of seconds,
        and at function scope every test in this class paid one. The cluster carries no
        per-test state these tests depend on — each builds its own dataset and its own
        ``ActorPoolStrategy`` pool, and nothing in the package uses named/detached actors
        — so one cluster serves the class. Phase 2 runs serial (see run_tests.sh), so
        there is no xdist worker to fight over it.
        """
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        yield
        ray.shutdown()

    def test_replica_mode_map_batches_completes(self, ray_local):
        """Replica mode stage runs under map_batches and materializes."""
        batch_size = 4
        num_rows = 6
        stage = StatefulStage(
            fn=MockFoldingEngineUDF,
            fn_constructor_kwargs={
                "model": "test_model",
                "engine_kwargs": {},
                "parallelism_mode": ParallelismMode.REPLICA,
            },
            map_batches_kwargs={
                "num_gpus": 0,
                "batch_size": batch_size,
                "compute": ray.data.ActorPoolStrategy(min_size=1, max_size=1),
            },
            compute_by_rows=True,
            drop_keys=None,
            update_row=False,
        )
        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=batch_size)

        ds = ray.data.from_items([{"key": f"row_{i}", "__record_id": f"id_{i}"} for i in range(num_rows)])
        result = ds.map_batches(stage.fn, **kwargs)
        result = result.materialize()
        out = [unpack_pipeline_row(r) for r in result.take_all()]

        assert len(out) == num_rows
        for row in out:
            assert "structure" in row
            assert row["structure"] == "MOCK_ATOM"
            assert "time_taken" in row
            assert "device_id" in row
            assert isinstance(row["device_id"], int)
            assert "__inference_error__" in row

    def test_replica_mode_map_batches_output_structure(self, ray_local):
        """Replica mode output has expected inference columns."""
        stage = StatefulStage(
            fn=MockFoldingEngineUDF,
            fn_constructor_kwargs={
                "model": "test_model",
                "engine_kwargs": {},
                "parallelism_mode": ParallelismMode.REPLICA,
            },
            map_batches_kwargs={
                "num_gpus": 0,
                "batch_size": 2,
                "compute": ray.data.ActorPoolStrategy(min_size=1, max_size=1),
            },
            compute_by_rows=True,
            drop_keys=None,
            update_row=False,
        )
        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=2)

        ds = ray.data.from_items(
            [
                {"key": "a", "__record_id": "id_a"},
                {"key": "b", "__record_id": "id_b"},
            ]
        )
        out = [unpack_pipeline_row(r) for r in ds.map_batches(stage.fn, **kwargs).materialize().take_all()]

        assert len(out) == 2
        for row in out:
            assert row.get("__inference_error__") is not None
            assert row["__inference_error__"].get("error_msg") is None
            assert "time_taken" in row
            assert "structure" in row
            assert "confidence" in row
            assert "device_id" in row
            assert isinstance(row["device_id"], int)

    def test_replica_mode_real_stage_config_under_map_batches(self, ray_local):
        """FoldingEngineStage config (num_gpus, compute) works with map_batches when using a mock UDF.

        Testing the real FoldingEngineUDF under map_batches would require loading a registered
        model in the Ray worker; use MockFoldingEngineUDF (above) to test the map_batches path
        without a real model.
        """
        stage = FoldingEngineStage(
            fn_constructor_kwargs={
                "model": "alphafold2_1",
                "engine_kwargs": {},
                "parallelism_mode": ParallelismMode.REPLICA,
            },
            map_batches_kwargs={
                "num_gpus": 0,
                "batch_size": 1,
                "compute": ray.data.ActorPoolStrategy(min_size=1, max_size=1),
            },
            compute_by_rows=True,
            drop_keys=[],
        )
        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=1)
        assert stage.fn is FoldingEngineUDF
        assert kwargs["fn_constructor_kwargs"]["parallelism_mode"] == ParallelismMode.REPLICA
        assert kwargs.get("num_gpus") == 0 or kwargs.get("num_gpus") == 1

    @pytest.mark.skipif(
        get_available_gpu_count() < MIN_GPUS_FOR_MULTI_GPU_REPLICA,
        reason=f"Need at least {MIN_GPUS_FOR_MULTI_GPU_REPLICA} GPUs for multi-GPU replica test",
    )
    def test_replica_mode_map_batches_multiple_gpus(self, ray_local):
        """Replica mode with multiple GPUs: 2 replicas (1 GPU each) under map_batches."""
        num_gpus_available = get_available_gpu_count()
        num_replicas = min(2, num_gpus_available)
        assert num_replicas >= MIN_GPUS_FOR_MULTI_GPU_REPLICA, "skipif should have skipped"

        stage = StatefulStage(
            fn=MockFoldingEngineUDF,
            fn_constructor_kwargs={
                "model": "test_model",
                "engine_kwargs": {},
                "parallelism_mode": ParallelismMode.REPLICA,
            },
            map_batches_kwargs={
                "num_gpus": 1,
                "batch_size": 2,
                "compute": ray.data.ActorPoolStrategy(
                    min_size=num_replicas,
                    max_size=num_replicas,
                ),
            },
            compute_by_rows=True,
            drop_keys=None,
            update_row=False,
        )
        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=2)

        num_rows = 4
        ds = ray.data.from_items([{"key": f"row_{i}", "__record_id": f"id_{i}"} for i in range(num_rows)])
        result = ds.map_batches(stage.fn, **kwargs)
        result = result.materialize()
        out = [unpack_pipeline_row(r) for r in result.take_all()]

        assert len(out) == num_rows
        device_ids = {row["device_id"] for row in out}
        # When Ray pins each actor to a different GPU, we see num_replicas distinct device_ids.
        # When Ray does not (e.g. same CUDA_VISIBLE_DEVICES per worker), all rows may have the same device_id.
        assert len(device_ids) >= 1, f"Expected at least one device_id, got {device_ids}"
        device_uuids = {row["device_uuid"] for row in out}
        assert len(device_uuids) == num_replicas, (
            f"Expected {num_replicas} distinct device_uuids, got {len(device_uuids)}: {device_uuids}"
        )
        for row in out:
            assert "structure" in row
            assert row["structure"] == "MOCK_ATOM"
            assert "time_taken" in row
            assert "device_id" in row
            assert isinstance(row["device_id"], int)
            assert "__inference_error__" in row
