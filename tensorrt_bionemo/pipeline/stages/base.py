# Adapted from https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/batch/stages/base.py
# All per-row data is packed into a single pickled DATA_COLUMN between stages
# so that PyArrow never has to infer schemas for complex/heterogeneous nested dicts.
import pickle
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import pyarrow
from pydantic import BaseModel, ConfigDict, Field

from tensorrt_bionemo.logger import logger


def unpack_pipeline_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Unpack a row from the pipeline's packed data-column format.

    After pipeline execution, each row has a pickled ``__data__`` column
    containing the actual payload.  Call this when iterating over pipeline
    output rows (e.g. ``ds.iter_rows()``) to get the original flat dict.
    """
    data = row.get(StatefulStageUDF.DATA_COLUMN)
    if data is None:
        return row
    unpacked = pickle.loads(data) if isinstance(data, bytes) else data
    unpacked["__inference_error__"] = row.get("__inference_error__")
    record_id = row.get(StatefulStageUDF.RECORD_ID_IN_BATCH_COLUMN)
    if record_id is not None:
        unpacked[StatefulStageUDF.RECORD_ID_IN_BATCH_COLUMN] = record_id
    return unpacked


class StatefulStageUDF:
    """A stage UDF wrapper that processes the input and output columns
    before and after the UDF.

    Between stages, all per-row data is packed into a single pickled column
    (``DATA_COLUMN``) so that PyArrow never needs to infer schemas for complex
    nested Python objects.  Only ``__inference_error__`` and ``__record_id``
    are kept as plain top-level columns (they have uniform, simple types).

    The first stage in the pipeline receives flat columns from
    ``ray.data.from_items``; subsequent stages receive the packed format.
    """

    IDX_IN_BATCH_COLUMN: str = "__idx_in_batch"
    RECORD_ID_IN_BATCH_COLUMN: str = "__record_id"
    DATA_COLUMN: str = "__data__"

    # Keys that live as top-level Arrow columns (not inside DATA_COLUMN).
    _TOP_LEVEL_KEYS = frozenset({"__inference_error__", "__record_id"})

    pack_output: bool = True

    def __init__(self,
                 compute_by_rows: bool = True,
                 drop_keys: Optional[List[str]] = None,
                 expected_input_keys: Optional[List[str]] = None,
                 update_row: bool = True):
        self.expected_input_keys = set(expected_input_keys or [])
        self.compute_by_rows = compute_by_rows
        self.drop_keys = drop_keys
        self.update_row = update_row

    # ------------------------------------------------------------------
    # Row unpacking helpers
    # ------------------------------------------------------------------

    def _unpack_rows_from_batch(self,
                                batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a columnar batch into a list of row dicts.

        If the batch contains DATA_COLUMN (packed format from a previous
        stage), each element is unpickled and top-level columns are merged in.
        Otherwise, the batch is in flat-column format (from ray.data.from_items)
        and is converted the traditional way.
        """
        if self.DATA_COLUMN in batch:
            packed = batch[self.DATA_COLUMN]
            if hasattr(packed, "tolist"):
                packed = packed.tolist()
            n_rows = len(packed)

            error_col = batch.get("__inference_error__")
            if error_col is None:
                error_col = [None] * n_rows
            elif hasattr(error_col, "tolist"):
                error_col = error_col.tolist()

            record_col = batch.get(self.RECORD_ID_IN_BATCH_COLUMN)
            if record_col is None:
                record_col = [None] * n_rows
            elif hasattr(record_col, "tolist"):
                record_col = record_col.tolist()

            rows: List[Dict[str, Any]] = []
            for i in range(n_rows):
                row = pickle.loads(packed[i]) if isinstance(
                    packed[i], bytes) else packed[i]
                row["__inference_error__"] = error_col[i]
                row[self.RECORD_ID_IN_BATCH_COLUMN] = record_col[i]
                rows.append(row)
            return rows

        # Flat columns (first stage from ray.data.from_items)
        rows: List[Dict[str, Any]] = []
        for column, values in batch.items():
            if hasattr(values, "tolist"):
                values = values.tolist()
            if len(rows) == 0:
                rows = [{} for _ in range(len(values))]
            for row, value in zip(rows, values):
                row[column] = value
        return rows

    def _pack_rows_to_output(self, rows: List[Dict[str,
                                                   Any]]) -> Dict[str, Any]:
        """Pack processed rows back into columnar format.

        All per-row data (except top-level keys) is pickled into
        DATA_COLUMN so Arrow only sees uniform ``bytes`` columns.
        """
        output: Dict[str, Any] = {}

        output["__inference_error__"] = [
            row.pop("__inference_error__", None) for row in rows
        ]
        record_ids = [
            row.pop(self.RECORD_ID_IN_BATCH_COLUMN, None) for row in rows
        ]
        output[self.RECORD_ID_IN_BATCH_COLUMN] = record_ids

        if self.drop_keys:
            for key in self.drop_keys:
                for row in rows:
                    row.pop(key, None)

        output[self.DATA_COLUMN] = [pickle.dumps(row) for row in rows]
        return output

    def _flatten_rows_to_output(self, rows: List[Dict[str,
                                                      Any]]) -> Dict[str, Any]:
        """Convert processed rows back to flat columnar format.

        Used by terminal stages (e.g. writer) whose output schema is simple
        and Arrow-friendly, so packing into DATA_COLUMN is unnecessary.
        """
        if self.drop_keys:
            for key in self.drop_keys:
                for row in rows:
                    row.pop(key, None)

        output: Dict[str, Any] = {}
        if not rows:
            return output

        all_keys: set = set()
        for row in rows:
            all_keys.update(row.keys())

        for key in all_keys:
            output[key] = [row.get(key) for row in rows]
        return output

    # ------------------------------------------------------------------

    async def __call__(self,
                       batch: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Process a batch of data through the stage UDF.

        Supports two processing modes:
        1. Row-based (compute_by_rows=True): columnar → rows → UDF → packed columnar.
        2. Batch-based (compute_by_rows=False): direct batch UDF.
        """
        if isinstance(batch, pyarrow.lib.Table) and batch.num_rows == 0:
            yield {}
            return

        if self.compute_by_rows:
            rows = self._unpack_rows_from_batch(batch)

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

                    if id(rows[idx_in_batch]) != id(output):
                        rows[idx_in_batch].pop(self.IDX_IN_BATCH_COLUMN)
                    _id = rows[idx_in_batch][self.RECORD_ID_IN_BATCH_COLUMN]
                    if self.update_row:
                        rows[idx_in_batch].update(output)
                    else:
                        rows[idx_in_batch] = output
                    if _id is not None:
                        rows[idx_in_batch][
                            self.RECORD_ID_IN_BATCH_COLUMN] = _id
            if not_outputed_rows:
                raise ValueError(
                    f"The rows {not_outputed_rows} are not outputted.")
            for idx in error_row_indices:
                rows[idx].pop(self.IDX_IN_BATCH_COLUMN, None)

            if self.pack_output:
                yield self._pack_rows_to_output(rows)
            else:
                yield self._flatten_rows_to_output(rows)
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
            infer_err = inp.get("__inference_error__", {
                "error_msg": None,
                "traceback": None
            })
            if infer_err["error_msg"] is not None:
                continue
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
        import time as _time
        for row in rows:
            idx = row[self.IDX_IN_BATCH_COLUMN]
            try:
                _t0 = _time.perf_counter()
                result = await self.udf_for_item(row)
                # Accumulate per-stage wall time (always on). Carrying the prior dict
                # means it survives both update_row=True (merge) and =False (replace).
                _timing = dict(row.get("stage_timing_s") or {})
                _timing[self.__class__.__name__] = _time.perf_counter() - _t0
                result["stage_timing_s"] = _timing
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

    def on_row_error(self, row: Dict[str, Any],
                     error: Exception) -> Dict[str, Any]:
        return {}

    async def udf_for_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} inherits from StatefulStageUDF must implement the udf_for_item method"
        )


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
        metadata: Optional dict (e.g. ccd_path, mol_dir) used only by the
            Tokenizer stage. Passed to context generators (e.g. Boltz2
            ccd_path, mol_dir). Merged into fn_constructor_kwargs when set.

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
