# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Any, Dict, List, Optional, Type

import numpy as np

from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.data.writers import CIFWriter, PDBWriter
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
        self.format = format or "pdb"
        self.mappings = mappings
        self.output_path = output_path

    def _get_writer_and_ext(self):
        """Get the appropriate writer instance and file extension based on format.

        Both PDBWriter and CIFWriter inherit from BaseWriter and support optional
        mappings. If mappings are not provided, they use sensible defaults.

        Returns:
            tuple: (writer_instance, file_extension)

        Raises:
            ValueError: If the format is not supported
        """
        res_type_mapping = self.mappings.get("res_type_mapping", None)
        atom_type_mapping = self.mappings.get("atom_type_mapping", None)

        if self.format == "pdb":
            writer = PDBWriter(res_type_mapping=res_type_mapping,
                               atom_type_mapping=atom_type_mapping)
            return writer, ".pdb"
        elif self.format == "cif":
            writer = CIFWriter(res_type_mapping=res_type_mapping,
                               atom_type_mapping=atom_type_mapping)
            return writer, ".cif"
        else:
            raise ValueError(
                f"Invalid format: {self.format}. Supported formats: 'pdb', 'cif'"
            )

    async def udf_for_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        writer, ext = self._get_writer_and_ext()

        chain_indices = row.get("chain_indices")
        if chain_indices is None:
            residue_indices = row.get("residue_indices")
            if residue_indices is not None:
                chain_indices = np.zeros_like(residue_indices, dtype=np.int64)

        b_factors = row.get("b_factors")
        if b_factors is None:
            atom_mask = row.get("atom_mask")
            if atom_mask is not None:
                b_factors = np.zeros_like(atom_mask, dtype=np.float32)

        record = FoldingOutput(atom_positions=row.get("atom_positions", None),
                               residue_types=row.get("residue_types", None),
                               atom_mask=row.get("atom_mask", None),
                               residue_indices=row.get("residue_indices",
                                                       None),
                               b_factors=b_factors,
                               chain_indices=chain_indices)
        row_id = row.get(self.RECORD_ID_IN_BATCH_COLUMN)

        if self.output_path and row_id:
            output_path = os.path.join(self.output_path, f"{row_id}{ext}")
        elif self.output_path:
            output_path = os.path.join(
                self.output_path, f"{row[self.IDX_IN_BATCH_COLUMN]}{ext}")
        else:
            output_path = None

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        writer.set_output_path(output_path)
        output_raw = writer.write(record)

        return {
            "output_path": output_path,
            "format": self.format,
            "output_raw": output_raw,
            self.RECORD_ID_IN_BATCH_COLUMN: row_id,
        }

    def on_row_error(self, row: Dict[str, Any],
                     error: Exception) -> Dict[str, Any]:
        return {
            "output_path": None,
            "format": self.format,
            "output_raw": None,
        }


class WriterStage(StatefulStage):

    fn: Type[StatefulStageUDF] = WriterUDF
    update_row: bool = False

    def get_required_input_keys(self) -> Dict[str, str]:
        return {
            "atom_positions": "The atom positions of the output. ",
            "residue_types": "The residue types of the output. ",
            "atom_mask": "The atom mask of the output. ",
            "residue_indices": "The residue indices of the output. ",
        }
