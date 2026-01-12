"""The base class for all stages."""
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import pyarrow
from pydantic import BaseModel, Field

from tensorrt_bionemo.logger import logger


class StatefulStageUDF:
    """A stage UDF wrapper that processes the input and output columns
    before and after the UDF.

    Args:
        data_column: The internal data column name of the processor. The
                     __call__ method takes the data column as the input of the UDF
                     method, and encapsulates the output of the UDF method into the data
                     column for the next stage.
        expected_input_keys: The expected input keys of the stage.
    """

    # The internal column name for the index of the row in the batch.
    # This is used to align the output of the UDF with the input batch.
    IDX_IN_BATCH_COLUMN: str = "__idx_in_batch"
    RECORD_ID_IN_BATCH_COLUMN: str = "__record_id"

    def __init__(self,
                 compute_by_rows: bool = True,
                 drop_keys: Optional[List[str]] = None,
                 expected_input_keys: Optional[List[str]] = None,
                 update_row: bool = True):
        self.expected_input_keys = set(expected_input_keys or [])
        self.compute_by_rows = compute_by_rows
        self.drop_keys = drop_keys
        self.update_row = update_row

    async def __call__(self,
                       batch: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        if isinstance(batch, pyarrow.lib.Table) and batch.num_rows == 0:
            yield {}
            return

        if self.compute_by_rows:
            # Transform:
            # batch: [col0: [row0, row1], col1: [row0, row1]]
            # to:
            # list of rows: row0: [col0, col1], row1: [col0, col1]
            # then call the udf_for_rows method
            rows = []
            for column, values in batch.items():
                if len(rows) == 0:
                    rows = [{} for _ in range(len(values))]
                for row, value in zip(rows, values):
                    row[column] = value
            self.validate_rows_input(rows)
            for idx, row in enumerate[dict](rows):
                row[self.IDX_IN_BATCH_COLUMN] = idx

            normal_rows = []
            error_row_indices = set()
            for idx, row in enumerate(rows):
                infer_err = row.get("__inference_error__", {
                    "error_msg": None,
                    "traceback": None
                })
                if infer_err["error_msg"] is not None:
                    error_row_indices.add(idx)
                else:
                    normal_rows.append(row)
            not_outputed_rows = set(range(len(rows))) - error_row_indices
            if normal_rows:
                async for output in self.udf_for_rows(normal_rows):
                    if self.IDX_IN_BATCH_COLUMN not in output:
                        raise ValueError(
                            "The output of the UDF must contain the column "
                            f"{self.IDX_IN_BATCH_COLUMN}.")
                    idx_in_batch = output.pop(self.IDX_IN_BATCH_COLUMN)
                    if idx_in_batch not in not_outputed_rows:
                        raise ValueError(
                            f"The row {idx_in_batch} is outputted twice. "
                            "This is likely due to the UDF is not one-to-one.")
                    not_outputed_rows.remove(idx_in_batch)

                    # Add stage outputs to the data column of the row.
                    rows[idx_in_batch].pop(self.IDX_IN_BATCH_COLUMN)
                    _id = rows[idx_in_batch][self.RECORD_ID_IN_BATCH_COLUMN]
                    if self.update_row:
                        rows[idx_in_batch].update(output)
                    else:
                        rows[idx_in_batch] = output
                    if _id is not None:
                        # Keep the __record_id for the row.
                        rows[idx_in_batch][
                            self.RECORD_ID_IN_BATCH_COLUMN] = _id
            if not_outputed_rows:
                raise ValueError(
                    f"The rows {not_outputed_rows} are not outputted.")
            # Clean up idx column from error rows (normal rows already cleaned above)
            for idx in error_row_indices:
                rows[idx].pop(self.IDX_IN_BATCH_COLUMN, None)

            # Transform back
            output = {}
            output["__inference_error__"] = [
                row.get("__inference_error__") for row in rows
            ]
            if self.drop_keys:
                for key in self.drop_keys:
                    for row in rows:
                        if key in row:
                            del row[key]
            gather_keys = list(rows[0].keys())
            for key in gather_keys:
                output[key] = [row.get(key) for row in rows]
            yield output
        else:
            output = await self.udf_for_batch(batch)
            if self.drop_keys:
                for key in self.drop_keys:
                    if key in output:
                        del output[key]
            _ids = batch.get(self.RECORD_ID_IN_BATCH_COLUMN, None)
            if _ids is not None:
                output[self.RECORD_ID_IN_BATCH_COLUMN] = _ids
            yield output

    def validate_batch_input(self, batch: Dict[str, Any]):
        """Validate the batch to make sure the required keys are present.
        """
        input_keys = set(batch.keys())
        missing_required = self.expected_input_keys - input_keys
        if missing_required:
            raise ValueError(
                f"Required input keys {missing_required} not found at the input of "
                f"{self.__class__.__name__}. Input keys: {input_keys}")

    def validate_rows_input(self, inputs: List[Dict[str, Any]]):
        """Validate the inputs to make sure the required keys are present.

        Args:
            inputs: The inputs.

        Raises:
            ValueError: If the required keys are not found.
        """
        for inp in inputs:
            input_keys = set(inp.keys())

            if self.IDX_IN_BATCH_COLUMN in input_keys:
                raise ValueError(
                    f"The input column {self.IDX_IN_BATCH_COLUMN} is reserved "
                    "for internal use.")

            if not self.expected_input_keys:
                continue

            missing_required = self.expected_input_keys - input_keys
            if missing_required:
                raise ValueError(
                    f"Required input keys {missing_required} not found at the input of "
                    f"{self.__class__.__name__}. Input keys: {input_keys}")

    async def udf_for_batch(
            self, batch: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError(
            "StageUDF must implement the udf_for_batch method")

    async def udf_for_rows(
            self, rows: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError(
            "StageUDF must implement the udf_for_rows method")


class StatefulStage(BaseModel):
    """
    A basic building block to compose a Processor.
    """

    fn: Type[StatefulStageUDF] = Field(
        description="The well-optimized stateful UDF for this stage.")
    fn_constructor_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="The keyword arguments of the UDF constructor.",
    )
    map_batches_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"concurrency": 1},
        description=
        "The arguments of .map_batches(). Default {'concurrency': 1}.",
    )

    compute_by_rows: bool = Field(
        default=True,
        description="Convert to rows mode and compute.",
    )
    drop_keys: Optional[List[str]] = Field(
        default=None,
        description="The keys to drop from the output.",
    )
    update_row: bool = Field(
        default=True,
        description=
        "Whether to update the input with the output, else replace the input with the output.",
    )

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {}

    def get_optional_input_keys(self) -> Dict[str, str]:
        """The optional input keys of the stage and their descriptions."""
        return {}

    def get_dataset_map_batches_kwargs(self,
                                       batch_size: int) -> Dict[str, Any]:
        """We separate fn and fn_constructor_kwargs in Stage for better UX,
        so we combine them with other map_batches_kwargs together in this method.

        Args:
            batch_size: The batch size set by the processor config.
            compute_by_rows: Convert to rows mode and compute.
            drop_keys: The keys to drop from the output.

        Returns:
            The dataset map_batches kwargs.
        """
        kwargs = self.map_batches_kwargs.copy()
        batch_size_in_kwargs = kwargs.get("batch_size", batch_size)
        if batch_size_in_kwargs != batch_size:
            logger.warning(
                "batch_size is set to %d in map_batches_kwargs, but it will be "
                "overridden by the batch size configured by the processor %d.",
                batch_size_in_kwargs,
                batch_size,
            )
        kwargs["batch_size"] = batch_size

        kwargs.update(
            {"fn_constructor_kwargs": self.fn_constructor_kwargs.copy()})
        if "compute_by_rows" in kwargs["fn_constructor_kwargs"]:
            raise ValueError(
                "'compute_by_rows' cannot be used as in fn_constructor_kwargs."
            )

        kwargs["fn_constructor_kwargs"][
            "compute_by_rows"] = self.compute_by_rows
        kwargs["fn_constructor_kwargs"]["drop_keys"] = self.drop_keys
        kwargs["fn_constructor_kwargs"]["expected_input_keys"] = list(
            self.get_required_input_keys().keys())
        kwargs["fn_constructor_kwargs"]["update_row"] = self.update_row
        return kwargs

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
