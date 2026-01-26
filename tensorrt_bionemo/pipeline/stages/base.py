# Adapted from https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/batch/stages/base.py
# But modify for both of the row mode and batch mode.
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import pyarrow
from pydantic import BaseModel, ConfigDict, Field

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
        """Process a batch of data through the stage UDF.

        This method serves as the main entry point for processing data through the stage.
        It handles the transformation between columnar and row-based formats, manages
        error propagation, and ensures proper alignment of outputs with inputs.

        The method supports two processing modes:
        1. Row-based processing (compute_by_rows=True): Transforms columnar batch data
           into individual rows, processes each row through udf_for_rows, and transforms
           back to columnar format.
        2. Batch-based processing (compute_by_rows=False): Processes the entire batch
           directly through udf_for_batch.

        Args:
            batch: A dictionary mapping column names to lists of values, representing
                   a batch of data in columnar format. For example:
                   {"col1": [val1, val2, ...], "col2": [val1, val2, ...]}

                   Special columns:
                   - __record_id: Optional record identifier that is preserved through
                     processing
                   - __inference_error__: List of error dictionaries with "error_msg"
                     and "traceback" keys. Rows with errors are skipped during processing
                     but included in output.

        Yields:
            A dictionary in columnar format containing the processed results. The output
            includes:
            - All input columns (unless dropped via drop_keys)
            - New columns added by the UDF
            - Updated columns modified by the UDF (if update_row=True)
            - __inference_error__: Preserved error information for all rows
            - __record_id: Preserved record identifiers (if present in input)

        Raises:
            ValueError: If the UDF output is missing the required __idx_in_batch column
                       (row-based mode only).
            ValueError: If a row index is outputted multiple times, indicating the UDF
                       is not one-to-one (row-based mode only).
            ValueError: If some rows are not outputted by the UDF (row-based mode only).

        Notes:
            - In row-based mode, the method adds an internal __idx_in_batch column to
              track row positions and ensure proper alignment of outputs.
            - Rows with existing inference errors are preserved and passed through
              without processing, maintaining their error state.
            - The method validates that all normal (non-error) rows are processed
              exactly once by the UDF.
            - Empty PyArrow tables are handled gracefully by yielding an empty dict.
            - Keys specified in drop_keys are removed from the output after processing.
        """
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
            for idx, row in enumerate(rows):
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
                    # The output may be a reference of the row, so we need to check
                    # They are same reference to pop the idx_in_batch column.
                    if id(rows[idx_in_batch]) != id(output):
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
        for row in rows:
            idx = row[self.IDX_IN_BATCH_COLUMN]
            try:
                result = await self.udf_for_item(row)
                result["__inference_error__"] = {
                    "error_msg": None,
                    "traceback": None
                }
            except Exception as e:
                result = self.on_row_error(row, e)
                result["__inference_error__"] = {
                    "error_msg": f"{type(e).__name__}: {str(e)}",
                    "traceback": traceback.format_exc()
                }
            result[self.IDX_IN_BATCH_COLUMN] = idx
            yield result

    async def udf_for_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "StageUDF must implement the udf_for_item method")


class StatefulStage(BaseModel):
    """A Pydantic configuration model for Ray Data map_batches pipeline stages.

    This class serves as a declarative configuration for a single processing stage
    in a data pipeline. It encapsulates a stateful UDF (User-Defined Function) along
    with its construction parameters and Ray Data map_batches execution settings.

    The StatefulStage is designed to work with Ray Data's map_batches API, providing
    a clean separation between:
    1. The UDF implementation (fn)
    2. UDF initialization parameters (fn_constructor_kwargs)
    3. Ray Data execution parameters (map_batches_kwargs)
    4. Processing behavior flags (compute_by_rows, drop_keys, update_row)

    Attributes:
        fn: The StatefulStageUDF class (not instance) that will be instantiated
            to process data. This class must implement either udf_for_rows or
            udf_for_batch methods.
        fn_constructor_kwargs: Keyword arguments passed to the fn constructor when
            instantiating the UDF. These are user-defined parameters specific to
            the UDF implementation. Default: empty dict.
        map_batches_kwargs: Arguments passed to Ray Data's map_batches method,
            controlling execution behavior like concurrency, batch format, etc.
            Default: {"concurrency": 1}.
        compute_by_rows: If True, converts columnar batch data to row format before
            processing and transforms back after. If False, processes data in batch
            (columnar) format. Default: True.
        drop_keys: Optional list of column keys to remove from the output after
            processing. Useful for cleaning up intermediate columns. Default: None.
        update_row: If True, merges UDF output with input row (preserves existing
            columns). If False, replaces input with UDF output entirely. Only
            applies when compute_by_rows=True. Default: True.

    Example:
        ```python
        class MyUDF(StatefulStageUDF):
            def __init__(self, threshold: float, **kwargs):
                super().__init__(**kwargs)
                self.threshold = threshold

            async def udf_for_rows(self, rows):
                for row in rows:
                    row["filtered"] = row["value"] > self.threshold
                    yield row

        stage = StatefulStage(
            fn=MyUDF,
            fn_constructor_kwargs={"threshold": 0.5},
            map_batches_kwargs={"concurrency": 2, "batch_size": 100},
            compute_by_rows=True,
            drop_keys=["temporary_col"],
            update_row=True
        )

        # Later used in Ray Data pipeline:
        # ds = ds.map_batches(stage.fn, **stage.get_dataset_map_batches_kwargs(batch_size=100))
        ```

    Notes:
        - This is a Pydantic BaseModel, so all fields are validated and type-checked.
        - The batch_size in map_batches_kwargs will be overridden by the processor's
          batch_size configuration if they differ.
        - The compute_by_rows, drop_keys, expected_input_keys, and update_row parameters
          are automatically injected into fn_constructor_kwargs by get_dataset_map_batches_kwargs.
        - Subclasses should override get_required_input_keys() and get_optional_input_keys()
          to document their data requirements.
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
        """Get the required input keys for this stage and their descriptions.

        Subclasses should override this method to declare which input columns
        must be present in the data batch for the stage to function correctly.
        These keys are automatically validated by the StatefulStageUDF before
        processing.

        Returns:
            A dictionary mapping required column names to human-readable descriptions
            of what each column contains. Default: empty dict (no requirements).

        Example:
            ```python
            def get_required_input_keys(self) -> Dict[str, str]:
                return {
                    "sequence": "Protein or DNA sequence string",
                    "organism": "Source organism identifier"
                }
            ```
        """
        return {}

    def get_optional_input_keys(self) -> Dict[str, str]:
        """Get the optional input keys for this stage and their descriptions.

        Subclasses should override this method to document which input columns
        the stage can use if available, but are not strictly required. This
        helps document the stage's full capabilities without enforcing their
        presence.

        Returns:
            A dictionary mapping optional column names to human-readable descriptions
            of what each column contains. Default: empty dict (no optional inputs).

        Example:
            ```python
            def get_optional_input_keys(self) -> Dict[str, str]:
                return {
                    "metadata": "Additional sequence metadata",
                    "quality_score": "Sequence quality scores (0-100)"
                }
            ```
        """
        return {}

    def get_dataset_map_batches_kwargs(self,
                                       batch_size: int) -> Dict[str, Any]:
        """Construct the complete kwargs dictionary for Ray Data's map_batches call.

        This method combines the stage configuration into a single dictionary suitable
        for passing to Ray Data's Dataset.map_batches() method. It merges:
        1. User-specified map_batches_kwargs (concurrency, etc.)
        2. The batch_size from the processor configuration
        3. UDF constructor kwargs with injected stage configuration

        The method automatically injects several parameters into fn_constructor_kwargs:
        - compute_by_rows: Controls row vs batch processing mode
        - drop_keys: Columns to remove from output
        - expected_input_keys: Required input columns (from get_required_input_keys)
        - update_row: Whether to merge or replace row data

        Args:
            batch_size: The batch size configured at the processor level. This will
                       override any batch_size specified in map_batches_kwargs, with
                       a warning logged if they differ.

        Returns:
            A dictionary containing all parameters for Ray Data's map_batches call,
            including the merged fn_constructor_kwargs with injected configuration.

        Raises:
            ValueError: If 'compute_by_rows' is manually specified in fn_constructor_kwargs
                       (it must be set via the compute_by_rows field instead).

        Example:
            ```python
            stage = StatefulStage(
                fn=MyUDF,
                fn_constructor_kwargs={"threshold": 0.5},
                map_batches_kwargs={"concurrency": 2},
                compute_by_rows=True,
                drop_keys=["temp"]
            )

            kwargs = stage.get_dataset_map_batches_kwargs(batch_size=100)
            # Returns:
            # {
            #     "concurrency": 2,
            #     "batch_size": 100,
            #     "fn_constructor_kwargs": {
            #         "threshold": 0.5,
            #         "compute_by_rows": True,
            #         "drop_keys": ["temp"],
            #         "expected_input_keys": [...],
            #         "update_row": True
            #     }
            # }
            ```
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

    model_config = ConfigDict(arbitrary_types_allowed=True,
                              validate_assignment=True)
