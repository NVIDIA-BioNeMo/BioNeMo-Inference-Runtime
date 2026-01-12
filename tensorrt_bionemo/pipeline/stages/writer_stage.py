import os
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.data.writers import PDBWriter
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class WriterUDF(StatefulStageUDF):

    def __init__(self,
                 compute_by_rows: bool,
                 drop_keys: List[str],
                 expected_input_keys: List[str],
                 update_row: bool,
                 mappings: dict[str, Any],
                 format: Optional[str] = "pdb",
                 output_path: Optional[str] = None):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys,
                         update_row)
        self.format = format
        if self.format is None:
            self.format = "pdb"
        self.mappings = mappings
        self.output_path = output_path

    async def udf_for_rows(
            self, batch: List[Dict[str,
                                   Any]]) -> AsyncIterator[Dict[str, Any]]:
        results = []
        if self.format == "pdb":
            writer = PDBWriter(
                res_type_mapping=self.mappings.get("res_type_mapping", None),
                atom_type_mapping=self.mappings.get("atom_type_mapping", None))
            ext = ".pdb"
        else:
            raise ValueError(f"Invalid format: {self.format}")
        for row in batch:
            record: FoldingOutput = FoldingOutput(
                atom_positions=row.get("atom_positions", None),
                residue_types=row.get("residue_types", None),
                atom_mask=row.get("atom_mask", None),
                residue_indices=row.get("residue_indices", None),
                b_factors=row.get("b_factors", None),
                chain_indices=row.get("chain_indices", None))
            row_id = row.get(self.RECORD_ID_IN_BATCH_COLUMN)
            try:
                if self.output_path and row_id:
                    # If the output path and record ID are provided, use the record ID as the file name
                    output_path = os.path.join(self.output_path,
                                               f"{row_id}{ext}")
                elif self.output_path:
                    # If the output path is provided but the record ID is not, use the index in the batch as the file name
                    output_path = os.path.join(
                        self.output_path,
                        f"{row[self.IDX_IN_BATCH_COLUMN]}{ext}")
                else:
                    # Return the raw output to the user
                    output_path = None
                writer.set_output_path(output_path)
                output_raw = writer.write(record)
                results.append({
                    "output_path":
                    output_path,
                    "format":
                    self.format,
                    "output_raw":
                    output_raw,
                    self.RECORD_ID_IN_BATCH_COLUMN:
                    row_id,
                    self.IDX_IN_BATCH_COLUMN:
                    row[self.IDX_IN_BATCH_COLUMN],
                    "__inference_error__": {
                        "error_msg": None,
                        "traceback": None
                    },
                })
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                results.append({
                    "output_path":
                    None,
                    "format":
                    self.format,
                    "output_raw":
                    None,
                    "__inference_error__": {
                        "error_msg": error_msg,
                        "traceback": traceback.format_exc()
                    },
                    self.IDX_IN_BATCH_COLUMN:
                    row[self.IDX_IN_BATCH_COLUMN]
                })
        assert len(batch) == len(results)
        for row, result in zip(batch, results):
            yield result


class WriterStage(StatefulStage):
    """
    A stage that parses the input.
    """

    fn: Type[StatefulStageUDF] = WriterUDF
    update_row: bool = False

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "atom_positions": "The atom positions of the output. ",
            "residue_types": "The residue types of the output. ",
            "atom_mask": "The atom mask of the output. ",
            "residue_indices": "The residue indices of the output. ",
        }
