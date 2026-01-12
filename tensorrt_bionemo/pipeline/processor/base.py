from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from pydantic import BaseModel, Field, field_validator
from ray.data import Dataset

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


class Processor:

    def __init__(self, config: ProcessorConfig, stages: List[StatefulStage]):
        self.config = config
        self.stages: OrderedDict[str, StatefulStage] = OrderedDict()

        # FIXES: https://github.com/ray-project/ray/issues/53124
        # TODO (Kourosh): Remove this once the issue is fixed
        data_context = ray.data.DataContext.get_current()
        data_context.wait_for_min_actors_s = 600
        # TODO: Remove this when https://github.com/ray-project/ray/issues/53169
        # is fixed.
        data_context._enable_actor_pool_on_exit_hook = True

        for stage in stages:
            self._append_stage(stage)

    def __call__(self, dataset: Dataset) -> Dataset:
        """Execute the processor:
        preprocess -> stages -> postprocess.
        Note that the dataset won't be materialized during the execution.

        Args:
            dataset: The input dataset.

        Returns:
            The output dataset.
        """
        # Apply stages.
        for stage in self.stages.values():
            kwargs = stage.get_dataset_map_batches_kwargs(
                batch_size=self.config.batch_size)
            dataset = dataset.map_batches(stage.fn, **kwargs)
        return dataset

    def _append_stage(self, stage: StatefulStage) -> None:
        """Append a stage before postprocess. The stage class name will be used as
        the stage name. If there are multiple stages with the same type, a suffix
        will be added to the stage name to avoid conflicts.

        Args:
            stage: The stage to append.
        """
        stage_name = type(stage).__name__

        # When a processor has multiple stages with the same type,
        # append a index suffix to the stage name to avoid conflicts.
        if stage_name in self.stages:
            num_same_type_stage = len(
                [s for s in self.stages.values() if s is stage])
            stage_name = f"{stage_name}_{num_same_type_stage + 1}"
        self.stages[stage_name] = stage

    def list_stage_names(self) -> List[str]:
        """List the stage names of this processor in order. Preprocess and postprocess
        are not included.

        Returns:
            A list of stage names.
        """
        return list(self.stages.keys())

    def get_stage_by_name(self, name: str) -> StatefulStage:
        """Get a particular stage by its name. If the stage is not found,
        a ValueError will be raised.

        Args:
            name: The stage name.

        Returns:
            The pipeline stage.
        """
        if name in self.stages:
            return self.stages[name]
        raise ValueError(f"Stage {name} not found")
