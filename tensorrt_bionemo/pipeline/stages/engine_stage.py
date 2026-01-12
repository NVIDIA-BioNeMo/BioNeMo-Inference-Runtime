import asyncio
import time
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type

from pydantic import model_validator

from tensorrt_bionemo.configs.base import DeviceConfig, EngineConfig
from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.pipeline.engine import FoldingEngine
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)
from tensorrt_bionemo.registry import get_model_class, get_postprocessor


class FoldingEngineWrapper:

    def __init__(self,
                 model: str,
                 engine_kwargs: Dict[str, Any],
                 max_pending_requests: int = -1) -> None:
        model_class = get_model_class(model)
        model_config = engine_kwargs.get("config", None)
        if model_config is None:
            model_config = model_class.get_pretrained_config(model)
        accelerated_configs = engine_kwargs.get("accelerated_configs", None)
        postprocessor_config = engine_kwargs.get("postprocessor_config", None)
        postprocessor_class = get_postprocessor(model)
        device_config = engine_kwargs.get("device", None) or DeviceConfig()
        engine_config = EngineConfig(name=model,
                                     model=model_config,
                                     device=device_config,
                                     accelerated=accelerated_configs,
                                     postprocessor=postprocessor_config)
        self.engine = FoldingEngine(engine_config, model_class,
                                    postprocessor_class)
        self.model_config = model_config
        self.max_pending_requests = max_pending_requests
        if self.max_pending_requests > 0:
            self.semaphore = asyncio.Semaphore(self.max_pending_requests)
        else:
            # self.semaphore = asyncio.NullContext()
            pass

    def get_max_batch_size(self) -> int:
        return self.model_config.max_batch_size

    async def predict_async(
            self,
            rows: List[Dict[str,
                            Any]]) -> Tuple[List[FoldingOutput], List[float]]:
        assert len(rows) == 1, "Currently, support len(rows) == 1."
        row = rows[0]
        t = time.perf_counter()
        # async with self.semaphore:
        prediction = self.engine.execute(row)
        time_taken = time.perf_counter() - t
        return [prediction], [time_taken]


class FoldingEngineUDF(StatefulStageUDF):

    def __init__(self,
                 compute_by_rows: bool,
                 drop_keys: List[str],
                 expected_input_keys: List[str],
                 update_row: bool,
                 model: str,
                 engine_kwargs: Dict[str, Any],
                 max_pending_requests: Optional[int] = None,
                 should_continue_on_error: bool = False) -> None:
        super().__init__(
            compute_by_rows=compute_by_rows,
            drop_keys=drop_keys,
            expected_input_keys=expected_input_keys,
            update_row=update_row,
        )
        self.should_continue_on_error = should_continue_on_error
        max_pending_requests = max_pending_requests or 1
        self.folding = FoldingEngineWrapper(
            model=model,
            engine_kwargs=engine_kwargs,
            max_pending_requests=max_pending_requests)

    async def _predict_with_error_handling(
            self, sub_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate output for a single row, catching errors if should_continue_on_error is set.

         In the future the folding flow should be:
        1. Put requests to queue
        2. Get requests from queue and do padding if any to create a batch to model.
        3. Execute the engine.
        Currently, support len(rows) == 1.
        """
        try:
            outputs, time_takens = await self.folding.predict_async(sub_batch)
            ret = []
            for row, output, time_taken in zip(sub_batch, outputs,
                                               time_takens):
                idx_in_batch = row[self.IDX_IN_BATCH_COLUMN]
                ret.append({
                    **output,
                    "time_taken": time_taken,
                    "__inference_error__": {
                        "error_msg": None,
                        "traceback": None,
                    },
                    self.IDX_IN_BATCH_COLUMN: idx_in_batch,
                })
            return ret
        except Exception as e:
            if not self.should_continue_on_error:
                raise ValueError(f"Error predicting folding output: {e}")
            error_msg = f"{type(e).__name__}: {str(e)}"
            ret = []
            for row in sub_batch:
                idx_in_batch = row[self.IDX_IN_BATCH_COLUMN]
                ret.append({
                    "__inference_error__": {
                        "error_msg": error_msg,
                        "traceback": traceback.format_exc(),
                    },
                    self.IDX_IN_BATCH_COLUMN: idx_in_batch,
                })

    async def udf_for_rows(
            self, batch: List[Dict[str,
                                   Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        For each row in the batch, predict the folding output.
        """
        max_batch_size = self.folding.get_max_batch_size()
        n_iters = (len(batch) + max_batch_size - 1) // max_batch_size

        batch_start_time = time.perf_counter()
        tasks = []
        for i in range(n_iters):
            start_idx = i * max_batch_size
            end_idx = min(start_idx + max_batch_size, len(batch))
            sub_batch = batch[start_idx:end_idx]
            task = asyncio.create_task(
                self._predict_with_error_handling(sub_batch))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            results: List[Dict[str, Any]] = await task
            for item in results:
                yield item

        batch_time_taken = time.perf_counter() - batch_start_time
        logger.info(
            "Elapsed time for batch %s with size %d: %s",
            len(batch),
            batch_time_taken,
        )


class FoldingEngineStage(StatefulStage):
    """
    A stage that runs folding engine.
    """

    fn: Type[StatefulStageUDF] = FoldingEngineUDF
    update_row: bool = False

    @model_validator(mode="before")
    def post_init(cls, values):
        map_batches_kwargs = values["map_batches_kwargs"]
        accelerator_type = map_batches_kwargs.get("accelerator_type", "")

        ray_remote_args = {}
        if accelerator_type:
            ray_remote_args["accelerator_type"] = accelerator_type

        # TODO: When we have the executor support, we should use the executor to get the number of GPUs.
        # For now, we hardcode it to 1.
        ray_remote_args["num_gpus"] = 1

        map_batches_kwargs.update(ray_remote_args)
        return values
