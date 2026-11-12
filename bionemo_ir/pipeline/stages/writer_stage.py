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
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.writers import CIFWriter, PDBWriter, _text
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF
from bionemo_ir.pipeline.utils import STAGE_TIMING_COLUMN

_SUPPORTED_FORMATS = {"pdb", "cif"}
_EXT_MAP = {"pdb": ".pdb", "cif": ".cif"}
_SCORE_PRECISION = 4
_SCORE_SCALE = 10**_SCORE_PRECISION
# Rounded scores in [0, _SCORE_GRID_MAX], which covers PAE in angstroms, are spelled from one shared token table.
_SCORE_GRID_MAX = 32
# Score sets with at least this many values encode on a helper thread; smaller ones take less than the handoff.
_SCORE_THREAD_MIN_VALUES = 65536


def _encode_float_array(values: np.ndarray) -> str:
    """Return ``json.dumps(values.tolist())`` for a float64 array.

    Rounded scores repeat few distinct values: an ``N x N`` PAE matrix rounded
    to 0.001 within ``[0, 32)`` has at most 32,000. Each distinct value is
    encoded once and the text is joined from that table, which avoids encoding
    every element separately.
    """
    if values.size == 0:
        return json.dumps(values.tolist())
    # Bit patterns keep -0.0 distinct from 0.0, which ``json`` encodes differently.
    bits, inverse = np.unique(np.ascontiguousarray(values).view(np.int64), return_inverse=True)
    # One encoder call for the whole table; a float token never contains ", ".
    table = np.array(json.dumps(bits.view(np.float64).tolist())[1:-1].split(", "), dtype=object)
    return _join_json_tokens(table[inverse.reshape(values.shape)])


def _encode_rounded_float32_array(values: np.ndarray) -> str:
    """Return :func:`_encode_float_array` of ``np.round(values.astype(np.float64), _SCORE_PRECISION)``.

    ``np.round`` multiplies by ``10**_SCORE_PRECISION``, rounds to an integer and
    divides. For values in ``[0, _SCORE_GRID_MAX]`` those integers index the token
    table of :func:`_text.grid_tokens`, which spells the same text without the
    division or a per-value encoding.
    """
    scaled = values.astype(np.float64)
    scaled *= _SCORE_SCALE
    np.rint(scaled, out=scaled)
    # ``signbit`` rejects negatives and -0.0; NaN fails the upper bound.
    if (
        _SCORE_PRECISION == _text.GRID_DECIMALS
        and scaled.ndim
        and scaled.size
        and not np.signbit(scaled).any()
        and scaled.max() <= _SCORE_GRID_MAX * _SCORE_SCALE
    ):
        text = _text.json_grid_text(scaled.astype(np.int32), _text.grid_tokens(_SCORE_GRID_MAX))
        if text is not None:
            return text
    scaled /= _SCORE_SCALE
    return _encode_float_array(scaled)


def _join_json_tokens(tokens: np.ndarray) -> str:
    if tokens.ndim == 1:
        return "[" + ", ".join(tokens.tolist()) + "]"
    return "[" + ", ".join(_join_json_tokens(row) for row in tokens) + "]"


class _PendingScores:
    """Score JSON, encoded on a helper thread when the score arrays are large.

    The array encoders spend most of their time in NumPy, which releases the
    GIL, so a helper thread overlaps them with the structure text that the
    caller builds meanwhile. Small score sets encode when :meth:`result` runs,
    because a thread handoff would cost more than it hides.
    """

    def __init__(self, encode: Callable[[FoldingOutput], str], record: FoldingOutput) -> None:
        self._encode = encode
        self._record = record
        self._future: Future[str] | None = None
        values = sum(value.size for value in record.get_score_values().values() if isinstance(value, np.ndarray))
        if values >= _SCORE_THREAD_MIN_VALUES:
            self._future = Future()
            threading.Thread(target=self._run, name="bioir-writer-scores", daemon=True).start()

    def _run(self) -> None:
        try:
            self._future.set_result(self._encode(self._record))
        except BaseException as error:
            self._future.set_exception(error)

    def result(self) -> str:
        """Return the score JSON, raising any error from encoding it."""
        if self._future is None:
            return self._encode(self._record)
        return self._future.result()

    def settle(self) -> None:
        """Wait for a helper thread without raising, before another error propagates."""
        if self._future is not None:
            self._future.exception()


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

    def round_floats(self, o: Any, precision: int = _SCORE_PRECISION) -> Any:
        if isinstance(o, float):
            return round(o, precision)
        if isinstance(o, dict):
            return {k: self.round_floats(v, precision) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.round_floats(x, precision) for x in o]
        return o

    def _encode_scores(self, record: FoldingOutput) -> str:
        """Round the scores to :data:`_SCORE_PRECISION` decimals and encode them once.

        float32 arrays are rounded in NumPy. Their product with ``10**4`` is exact
        in float64 (a 24-bit significand times 625 fits in 34 bits), and both
        ``np.round`` and Python ``round`` break ties to even and divide with
        correct rounding, so the result is bit-identical to :meth:`round_floats`
        without the per-element Python loop; :func:`_encode_rounded_float32_array`
        encodes them without converting them to lists. Every other value keeps
        :meth:`round_floats`. The text is joined once, since a PAE field alone can
        run to tens of megabytes.
        """
        parts = ["{"]
        for key, value in record.get_score_values().items():
            if isinstance(value, np.ndarray) and value.dtype == np.float32:
                encoded = _encode_rounded_float32_array(value)
            elif isinstance(value, np.ndarray):
                encoded = json.dumps(self.round_floats(value.tolist()))
            else:
                encoded = json.dumps(self.round_floats(value))
            if len(parts) > 1:
                parts.append(", ")
            parts += (json.dumps(key), ": ", encoded)
        parts.append("}")
        return "".join(parts)

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
            # Ligand atoms whose names fall outside the ``AtomTypes`` vocabulary
            # sit in overflow slots, so the slot's canonical name is unrelated to
            # the atom. Dropping these two here made every such atom inherit its
            # slot name, which collides inside a residue ("Multiple atoms with
            # same atom_id") and loses the element.
            atom_names=row.get("atom_names", None),
            atom_elements=row.get("atom_elements", None),
        )
        row_id = row.get(self.RECORD_ID_IN_BATCH_COLUMN)

        output_paths: dict[str, str | None] = {}
        primary_raw: str | None = None

        scores = _PendingScores(self._encode_scores, record)
        try:
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
        except BaseException:
            scores.settle()
            raise

        scores_json = scores.result()

        score_path = self._resolve_path(row, row_id, "_scores.json")

        if score_path:
            with open(score_path, "w") as f:
                f.write(scores_json)

        primary_path = output_paths.get(self.format)
        result = {
            "output_path": primary_path,
            "output_paths": json.dumps(output_paths),
            "format": self.format,
            "output_raw": primary_raw,
            "scores": scores_json,
            self.RECORD_ID_IN_BATCH_COLUMN: row_id,
        }
        for timing_key in ("time_taken", "model_inference_time", "model_inference_time_samples", STAGE_TIMING_COLUMN):
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
