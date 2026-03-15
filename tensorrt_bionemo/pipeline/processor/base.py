from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from tensorrt_bionemo.pipeline.stages.base import StatefulStage


class ProcessorConfig(BaseModel):
    """The processor configuration."""

    batch_size: int = Field(
        default=1,
        description=
        "Large batch sizes are likely to saturate the compute resources "
        "and could achieve higher throughput. On the other hand, small batch sizes "
        "are more fault-tolerant and could reduce bubbles in the data pipeline. "
        "You can tune the batch size to balance the throughput and fault-tolerance "
        "based on your use case. Defaults to 1.",
    )
    accelerator_type: Optional[str] = Field(
        default=None,
        description="The accelerator type used by the LLM stage in a processor. "
        "Default to None, meaning that only the CPU will be used.",
    )
    concurrency: Union[int, Tuple[int, int]] = Field(
        default=1,
        description="The number of workers for data parallelism. Default to 1. "
        "If ``concurrency`` is a ``tuple`` ``(m, n)``, Ray creates an autoscaling "
        "actor pool that scales between ``m`` and ``n`` workers (``1 <= m <= n``). "
        "If ``concurrency`` is an ``int`` ``n``, Ray uses either a fixed pool of ``n`` "
        "workers or an autoscaling pool from ``1`` to ``n`` workers, depending on "
        "the processor and stage.",
    )

    model_source: str = Field(
        description="The model source to use for the offline processing.", )
    runtime_env: Optional[Dict[str, Any]] = Field(
        default=None,
        description=
        "The runtime environment to use for the offline processing.",
    )
    max_pending_requests: Optional[int] = Field(
        default=None,
        description="The maximum number of pending requests. If not specified, "
        "will use the default value from the backend engine.",
    )
    max_concurrent_batches: int = Field(
        default=8,
        description="The maximum number of concurrent batches in the engine. "
        "This is to overlap the batch processing to avoid the tail latency of "
        "each batch. The default value may not be optimal when the batch size "
        "or the batch processing latency is too small, but it should be good "
        "enough for batch size >= 32.",
    )
    should_continue_on_error: bool = Field(
        default=False,
        description=
        "If True, continue processing when inference fails for a row "
        "instead of raising an exception. Failed rows will have a non-null "
        "'__inference_error__' column containing the error message, and other "
        "output columns will be None. Error rows bypass postprocess. "
        "If False (default), any inference error will raise an exception.",
    )
    executor_backend: Optional[Literal["ray"]] = Field(
        default=None,
        description="Execution backend. None = serial (no Ray, stages run "
        "sequentially in-process — useful for debugging and testing). "
        "'ray' = distributed execution via Ray Data.",
    )

    @field_validator("concurrency")
    def validate_concurrency(
        cls, concurrency: Union[int,
                                Tuple[int,
                                      int]]) -> Union[int, Tuple[int, int]]:
        """Validate that `concurrency` is either:
        - a positive int, or
        - a 2-tuple `(min, max)` of positive ints with `min <= max`.
        """

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        if isinstance(concurrency, int):
            require(
                concurrency > 0,
                f"A positive integer for `concurrency` is expected! Got: `{concurrency}`.",
            )
        elif isinstance(concurrency, tuple):
            require(
                all(c > 0 for c in concurrency),
                f"`concurrency` tuple items must be positive integers! Got: `{concurrency}`.",
            )

            min_concurrency, max_concurrency = concurrency
            require(
                min_concurrency <= max_concurrency,
                f"min > max in the concurrency tuple `{concurrency}`!",
            )
        return concurrency

    def get_concurrency(self,
                        autoscaling_enabled: bool = True) -> Tuple[int, int]:
        """Return a normalized `(min, max)` worker range from `self.concurrency`.

        Behavior:
        - If `concurrency` is an int `n`:
          - `autoscaling_enabled` is True  -> return `(1, n)` (autoscaling).
          - `autoscaling_enabled` is False -> return `(n, n)` (fixed-size pool).
        - If `concurrency` is a 2-tuple `(m, n)`, return it unchanged
          (the `autoscaling_enabled` flag is ignored).

        Args:
            autoscaling_enabled: When False, treat an integer `concurrency` as fixed `(n, n)`;
                otherwise treat it as a range `(1, n)`. Defaults to True.

        Returns:
            tuple[int, int]: The allowed worker range `(min, max)`.

        Examples:
            >>> self.concurrency = (2, 4)
            >>> self.get_concurrency()
            (2, 4)
            >>> self.concurrency = 4
            >>> self.get_concurrency()
            (1, 4)
            >>> self.get_concurrency(autoscaling_enabled=False)
            (4, 4)
        """
        if isinstance(self.concurrency, int):
            if autoscaling_enabled:
                return 1, self.concurrency
            else:
                return self.concurrency, self.concurrency
        return self.concurrency

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class _ProcessorBase:
    """Shared bookkeeping for both serial and Ray processors."""

    def __init__(self, config: ProcessorConfig, stages: List[StatefulStage]):
        self.config = config
        self.stages: OrderedDict[str, StatefulStage] = OrderedDict()
        for stage in stages:
            self._append_stage(stage)

    def _append_stage(self, stage: StatefulStage) -> None:
        stage_name = type(stage).__name__
        if stage_name in self.stages:
            num_same_type_stage = sum(1 for s in self.stages.values()
                                      if type(s) is type(stage))
            stage_name = f"{stage_name}_{num_same_type_stage}"
        self.stages[stage_name] = stage

    def list_stage_names(self) -> List[str]:
        return list(self.stages.keys())

    def get_stage_by_name(self, name: str) -> StatefulStage:
        if name in self.stages:
            return self.stages[name]
        raise ValueError(f"Stage {name} not found")


class Processor(_ProcessorBase):
    """Ray-based distributed processor."""

    def __init__(self, config: ProcessorConfig, stages: List[StatefulStage]):
        import ray as _ray
        from ray.data import Dataset  # noqa: F401

        super().__init__(config, stages)

        data_context = _ray.data.DataContext.get_current()
        data_context.wait_for_min_actors_s = 600
        data_context._enable_actor_pool_on_exit_hook = True

    def __call__(self, dataset: "Dataset") -> "Dataset":
        for stage in self.stages.values():
            kwargs = stage.get_dataset_map_batches_kwargs(
                batch_size=self.config.batch_size)
            dataset = dataset.map_batches(stage.fn, **kwargs)
        return dataset


class SerialProcessor(_ProcessorBase):
    """In-process serial processor — no Ray dependency.

    Runs each stage sequentially on every input row.  Useful for debugging,
    testing, and environments where Ray is not available.
    """

    def __init__(self, config: ProcessorConfig, stages: List[StatefulStage]):
        super().__init__(config, stages)
        self._udf_instances: OrderedDict[str, Any] = OrderedDict()

    def _get_or_create_udf(self, name: str, stage: StatefulStage):
        if name not in self._udf_instances:
            ctor_kwargs = stage.fn_constructor_kwargs.copy()
            ctor_kwargs["compute_by_rows"] = stage.compute_by_rows
            ctor_kwargs["drop_keys"] = stage.drop_keys
            ctor_kwargs["expected_input_keys"] = list(
                stage.get_required_input_keys().keys())
            ctor_kwargs["update_row"] = stage.update_row
            self._udf_instances[name] = stage.fn(**ctor_kwargs)
        return self._udf_instances[name]

    def __call__(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run all stages serially on the given records.

        Args:
            records: List of input dicts, same format as rows passed to
                ``ray.data.from_items``.

        Returns:
            List of output dicts (flat, no packing).
        """
        import asyncio

        batch: Dict[str, Any] = self._rows_to_columnar(records)

        for name, stage in self.stages.items():
            udf = self._get_or_create_udf(name, stage)
            batch = asyncio.run(self._run_udf(udf, batch))

        return self._columnar_to_rows(batch)

    @staticmethod
    async def _run_udf(udf, batch: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        async for chunk in udf(batch):
            for k, v in chunk.items():
                if k in merged:
                    if isinstance(merged[k], list) and isinstance(v, list):
                        merged[k].extend(v)
                    else:
                        merged[k] = v
                else:
                    merged[k] = v
        return merged

    @staticmethod
    def _rows_to_columnar(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        all_keys: set = set()
        for row in rows:
            all_keys.update(row.keys())
        return {k: [row.get(k) for row in rows] for k in all_keys}

    @staticmethod
    def _columnar_to_rows(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not batch:
            return []
        first_col = next(iter(batch.values()))
        n = len(first_col) if isinstance(first_col, list) else 1
        rows: List[Dict[str, Any]] = [{} for _ in range(n)]
        for k, vals in batch.items():
            if not isinstance(vals, list):
                vals = [vals]
            for i, v in enumerate(vals):
                rows[i][k] = v
        return rows
