# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.writers import CIFWriter, PDBWriter
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF

_SUPPORTED_FORMATS = {"pdb", "cif"}
_EXT_MAP = {"pdb": ".pdb", "cif": ".cif"}


class WriterUDF(StatefulStageUDF):
    """Terminal pipeline stage that writes predicted structures to disk.

    ``pack_output`` is ``False`` so the returned row is a flat Arrow-friendly
    dict rather than the packed ``DATA_COLUMN`` format used between earlier
    stages.

    Supports writing **multiple formats** in a single pass when ``format``
    is a list (e.g. ``["pdb", "cif"]``).  Both files are produced from the
    same ``FoldingOutput``, so atom coordinates are guaranteed identical.

    Output row schema
    -----------------
    Each call to :meth:`udf_for_item` returns a dict with these keys:

    * ``output_path`` (*str | None*) - filesystem path of the first (or
      only) written structure file.
    * ``output_paths`` (*str*) - **JSON-encoded** dict mapping each
      format to its filesystem path, e.g.
      ``'{"pdb": "/out/id.pdb", "cif": "/out/id.cif"}'``.
    * ``format`` (*str*) - the primary format written (first element when
      a list was configured).
    * ``output_raw`` (*str*) - the raw file content of the primary format.
    * ``scores`` (*str*) - **JSON-encoded** string of prediction quality
      metrics (pLDDT, pTM, ipTM, ...).  Encoded as a JSON string (rather
      than a raw dict) so the column has a uniform ``string`` type in
      PyArrow, avoiding schema-inference errors when score dicts have
      varying structures across rows.  To access the scores dict::

          import json
          scores = json.loads(row["scores"])
          plddt = scores.get("plddt")

    * ``__record_id`` (*str | None*) - propagated record identifier.
    """

    pack_output = False

    def __init__(
        self,
        compute_by_rows: bool,
        drop_keys: list[str],
        expected_input_keys: list[str],
        update_row: bool,
        mappings: dict[str, Any],
        format: str | list[str] | None = "pdb",
        output_path: str | None = None,
    ):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys, update_row)
        raw = format or "pdb"
        self.formats: list[str] = [raw] if isinstance(raw, str) else list(raw)
        for fmt in self.formats:
            if fmt not in _SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported writer format '{fmt}'. Supported: {sorted(_SUPPORTED_FORMATS)}")
        self.mappings = mappings
        self.output_path = output_path

    @property
    def format(self) -> str:
        """Primary (first) format — kept for backward compatibility."""
        return self.formats[0]

    def round_floats(self, o: Any, precision: int = 4) -> Any:
        if isinstance(o, float):
            return round(o, precision)
        if isinstance(o, dict):
            return {k: self.round_floats(v, precision) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.round_floats(x, precision) for x in o]
        return o

    def _create_writer(self, fmt: str):
        res_type_mapping = self.mappings.get("res_type_mapping", None)
        atom_type_mapping = self.mappings.get("atom_type_mapping", None)
        if res_type_mapping is None or atom_type_mapping is None:
            raise ValueError(
                "WriterUDF requires both 'res_type_mapping' and "
                "'atom_type_mapping' to write PDB or CIF. "
                "Ensure WriterStage is configured with proper mappings."
            )
        if fmt == "pdb":
            return PDBWriter(res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping)
        if fmt == "cif":
            return CIFWriter(res_type_mapping=res_type_mapping, atom_type_mapping=atom_type_mapping)
        raise ValueError(f"Invalid format: {fmt}")

    def _resolve_path(self, row: dict[str, Any], row_id: Any, suffix: str) -> str | None:
        """Resolve one output destination inside the configured output directory.

        Record identifiers come from user input and may contain ``..`` segments
        or an absolute path, either of which would place the file outside the
        output directory, so the resolved destination is checked against the
        resolved output root.

        Args:
            row: Row being written, used for the fallback batch-index name.
            row_id: Record identifier, or a falsy value to name the file by
                batch index instead.
            suffix: Extension or scores-file suffix to append to the name.

        Returns:
            The destination path, or ``None`` when no output directory is set.

        Raises:
            ValueError: The destination resolves outside the output directory.
        """
        if not self.output_path:
            return None

        output_dir = Path(self.output_path)
        filename = f"{row_id if row_id else row[self.IDX_IN_BATCH_COLUMN]}{suffix}"
        output_path = output_dir / filename
        if not output_path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Output path escapes output directory")
        return str(output_path)

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        """Write one prediction in every configured format, plus its scores.

        Args:
            row: Folding outputs for a single record.

        Returns:
            The written paths, the primary serialized structure, the scores,
            and any timing fields carried on the input row.
        """
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

        record = FoldingOutput(
            atom_positions=row.get("atom_positions", None),
            residue_types=row.get("residue_types", None),
            atom_mask=row.get("atom_mask", None),
            residue_indices=row.get("residue_indices", None),
            b_factors=b_factors,
            chain_indices=chain_indices,
            plddt=row.get("plddt", None),
            ptm=row.get("ptm", None),
            iptm=row.get("iptm", None),
            pae=row.get("pae", None),
            max_pae=row.get("max_pae", None),
            residue_names=row.get("residue_names", None),
            mol_types=row.get("mol_types", None),
        )
        row_id = row.get(self.RECORD_ID_IN_BATCH_COLUMN)

        output_paths: dict[str, str | None] = {}
        primary_raw: str | None = None

        for fmt in self.formats:
            ext = _EXT_MAP[fmt]
            out_path = self._resolve_path(row, row_id, ext)
            if out_path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            writer = self._create_writer(fmt)
            writer.set_output_path(out_path)
            raw = writer.write(record)

            output_paths[fmt] = out_path
            if primary_raw is None:
                primary_raw = raw

        scores = self.round_floats(record.get_scores())

        score_path = self._resolve_path(row, row_id, "_scores.json")

        if score_path:
            with open(score_path, "w") as f:
                json.dump(scores, f)

        primary_path = output_paths.get(self.format)
        result = {
            "output_path": primary_path,
            "output_paths": json.dumps(output_paths),
            "format": self.format,
            "output_raw": primary_raw,
            "scores": json.dumps(scores),
            self.RECORD_ID_IN_BATCH_COLUMN: row_id,
        }
        for timing_key in ("time_taken", "model_inference_time", "model_inference_time_samples", "stage_timing_s"):
            if timing_key in row:
                result[timing_key] = row[timing_key]
        return result

    def on_row_error(self, row: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "output_path": None,
            "output_paths": json.dumps({}),
            "format": self.format,
            "output_raw": None,
        }


class WriterStage(StatefulStage):
    """Pipeline stage configuration for :class:`WriterUDF`.

    This is a terminal stage (``update_row=False``): its output replaces the
    input row entirely.  The output schema is flat and Arrow-friendly —
    see :class:`WriterUDF` for the exact column definitions.

    .. note::
       The ``scores`` column is a **JSON string**, not a dict.  Consumers
       must call ``json.loads(row["scores"])`` to obtain the scores dict.
       This ensures uniform PyArrow typing across rows with heterogeneous
       score structures.
    """

    fn: type[StatefulStageUDF] = WriterUDF
    update_row: bool = False

    def get_required_input_keys(self) -> dict[str, str]:
        return {
            "atom_positions": "The atom positions of the output. ",
            "residue_types": "The residue types of the output. ",
            "atom_mask": "The atom mask of the output. ",
            "residue_indices": "The residue indices of the output. ",
        }
