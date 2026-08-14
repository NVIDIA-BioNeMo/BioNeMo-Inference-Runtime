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
import gc
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

import torch
from pydantic import model_validator

from bionemo_ir.configs.base import DeviceConfig, EngineConfig
from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.logger import logger
from bionemo_ir.pipeline.engine import FoldingEngine
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF
from bionemo_ir.pipeline.stages.configs import ParallelismMode
from bionemo_ir.registry import get_model_class, get_postprocessor


class FoldingPredictionError(RuntimeError):
    """Raised when model prediction fails for a batch of records."""

    def __init__(self, record_ids: list, cause: Exception):
        self.record_ids = record_ids
        super().__init__(f"Prediction failed for record_ids={record_ids}")


class FoldingEngineWrapper:
    def __init__(
        self,
        model: str,
        engine_kwargs: dict[str, Any],
        max_pending_requests: int = -1,
        runtime_args: dict[str, Any] | None = None,
    ) -> None:
        model_class = get_model_class(model)
        model_config = engine_kwargs.get("config", None)
        if model_config is None:
            model_config = model_class.get_pretrained_config(model)
        accelerated_configs = engine_kwargs.get("accelerated_configs", None)

        postprocessor_config = engine_kwargs.get("postprocessor_config", None)
        postprocessor_class = get_postprocessor(model)
        device_config = engine_kwargs.get("device", None) or DeviceConfig()
        profile_inference = engine_kwargs.get("profile_inference", False)
        engine_config = EngineConfig(
            name=model,
            model=model_config,
            device=device_config,
            accelerated=accelerated_configs,
            postprocessor=postprocessor_config,
            profile_inference=profile_inference,
        )
        self.engine = FoldingEngine(engine_config, model_class, postprocessor_class, runtime_args=runtime_args)
        self.model_config = model_config
        self.max_pending_requests = max_pending_requests
        self.is_cuda_device = device_config.device_type == "cuda"

    def get_max_batch_size(self) -> int:
        return self.model_config.max_batch_size

    def cleanup(self):
        gc.collect()
        if self.is_cuda_device:
            torch.cuda.empty_cache()

    async def predict_async(self, rows: list[dict[str, Any]]) -> tuple[list[FoldingOutput], list[float]]:
        assert len(rows) == 1, "Currently, support len(rows) == 1."
        row = rows[0]
        t = time.perf_counter()
        prediction = self.engine.execute(row)
        time_taken = time.perf_counter() - t
        return [prediction], [time_taken]


class FoldingEngineUDF(StatefulStageUDF):
    def __init__(
        self,
        compute_by_rows: bool,
        drop_keys: list[str],
        expected_input_keys: list[str],
        update_row: bool,
        model: str,
        engine_kwargs: dict[str, Any],
        max_pending_requests: int | None = None,
        should_continue_on_error: bool = False,
        parallelism_mode: ParallelismMode = ParallelismMode.REPLICA,
        runtime_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            compute_by_rows=compute_by_rows,
            drop_keys=drop_keys,
            expected_input_keys=expected_input_keys,
            update_row=update_row,
        )
        self.parallelism_mode = parallelism_mode
        self.should_continue_on_error = should_continue_on_error
        max_pending_requests = max_pending_requests or 1

        self.folding = FoldingEngineWrapper(
            model=model,
            engine_kwargs=engine_kwargs,
            max_pending_requests=max_pending_requests,
            runtime_args=runtime_args,
        )

    def _create_success_response(
        self, row: dict[str, Any], output: dict[str, Any], time_taken: float
    ) -> dict[str, Any]:
        """Create a successful prediction response."""
        resp = {
            **output,
            "time_taken": time_taken,
            "__inference_error__": {
                "error_msg": None,
                "traceback": None,
            },
            self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
        }
        # This stage uses update_row=False (replaces the row), so explicitly carry
        # the upstream per-stage timing forward and tag the engine's own time (always on).
        _timing = dict(row.get("stage_timing_s") or {})
        _timing["FoldingEngine"] = output.get("model_inference_time", time_taken)
        resp["stage_timing_s"] = _timing
        return resp

    def _create_error_response(self, row: dict[str, Any], error_msg: str, traceback_str: str) -> dict[str, Any]:
        """Create an error response for a failed prediction."""
        return {
            "__inference_error__": {
                "error_msg": error_msg,
                "traceback": traceback_str,
            },
            self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
        }

    async def _predict_with_error_handling(self, sub_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate output for a sub-batch, catching errors if should_continue_on_error is set.

        In the future the folding flow should be:
        1. Put requests to queue
        2. Get requests from queue and do padding if any to create a batch to model.
        3. Execute the engine.
        Currently, support len(rows) == 1.

        Args:
            sub_batch: List of input rows to process.

        Returns:
            List of prediction results or error responses.

        Raises:
            FoldingPredictionError: If prediction fails and
                should_continue_on_error is False.  The original exception
                is chained via ``__cause__``.
        """
        try:
            outputs, time_takens = await self.folding.predict_async(sub_batch)
            return [
                self._create_success_response(row, output, time_taken)
                for row, output, time_taken in zip(sub_batch, outputs, time_takens, strict=False)
            ]
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error("=== Exception in _predict_with_error_handling ===")
            logger.error(traceback_str)
            logger.error("================================================")
            if not self.should_continue_on_error:
                record_ids = [row.get(self.RECORD_ID_IN_BATCH_COLUMN) for row in sub_batch]
                raise FoldingPredictionError(record_ids, e) from e

            self.folding.cleanup()
            error_msg = f"{type(e).__name__}: {str(e)}"
            return [self._create_error_response(row, error_msg, traceback_str) for row in sub_batch]

    async def udf_for_rows(self, batch: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """
        For each row in the batch, predict the folding output.
        We don't use the udf_for_item function, the model will handle the batching.
        """
        max_batch_size = self.folding.get_max_batch_size()
        n_iters = (len(batch) + max_batch_size - 1) // max_batch_size

        batch_start_time = time.perf_counter()
        tasks = []
        for i in range(n_iters):
            start_idx = i * max_batch_size
            end_idx = min(start_idx + max_batch_size, len(batch))
            sub_batch = batch[start_idx:end_idx]
            task = asyncio.create_task(self._predict_with_error_handling(sub_batch))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            results: list[dict[str, Any]] = await task
            for item in results:
                yield item

        batch_time_taken = time.perf_counter() - batch_start_time
        logger.debug(
            "Elapsed time for batch %s with size %d: %s",
            len(batch),
            batch_time_taken,
        )


class FoldingEngineStage(StatefulStage):
    """
    A stage that runs folding engine.
    """

    fn: type[StatefulStageUDF] = FoldingEngineUDF
    update_row: bool = False

    @model_validator(mode="before")
    def post_init(cls, values):
        map_batches_kwargs = values.get("map_batches_kwargs", {})
        accelerator_type = map_batches_kwargs.get("accelerator_type", "")

        ray_remote_args = {}
        if accelerator_type:
            ray_remote_args["accelerator_type"] = accelerator_type

        if "num_gpus" not in map_batches_kwargs:
            ray_remote_args["num_gpus"] = 1

        map_batches_kwargs.update(ray_remote_args)
        values["map_batches_kwargs"] = map_batches_kwargs
        return values
