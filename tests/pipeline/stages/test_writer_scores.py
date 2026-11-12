# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The writer's score JSON is byte-identical to Python-rounded ``get_scores()``."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import bionemo_ir.pipeline.stages.writer_stage as writer_stage
from bionemo_ir.data.schemas.basic import FoldingOutput
from bionemo_ir.pipeline.stages.writer_stage import WriterUDF, _encode_float_array, _encode_rounded_float32_array


def _udf(output_path: str | None) -> WriterUDF:
    return WriterUDF(
        compute_by_rows=True,
        drop_keys=[],
        expected_input_keys=[],
        update_row=False,
        mappings={},
        format="pdb",
        output_path=output_path,
    )


def _row(plddt: np.ndarray | None, pae: np.ndarray | None, **scalars: float) -> dict:
    n_res = 8 if plddt is None else len(plddt)
    return {
        "atom_positions": np.zeros((n_res, 37, 3), dtype=np.float32),
        "residue_types": np.zeros(n_res, dtype=np.int64),
        "atom_mask": np.ones((n_res, 37), dtype=np.float32),
        "residue_indices": np.arange(1, n_res + 1),
        "chain_indices": np.zeros(n_res, dtype=np.int64),
        "plddt": plddt,
        "pae": pae,
        "__record_id": "rec",
        "__idx_in_batch": 0,
        **scalars,
    }


def _expected_scores(udf: WriterUDF, row: dict) -> str:
    keys = ("atom_positions", "residue_types", "atom_mask", "residue_indices", "chain_indices", "plddt", "pae")
    record = FoldingOutput(**{k: row.get(k) for k in keys}, **{k: row.get(k) for k in ("ptm", "iptm", "max_pae")})
    return json.dumps(udf.round_floats(record.get_scores()))


def _write(udf: WriterUDF, row: dict) -> dict:
    with patch.object(udf, "_create_writer") as mock_create:
        mock_writer = MagicMock()
        mock_writer.write.return_value = "ATOM..."
        mock_create.return_value = mock_writer
        return asyncio.run(udf.udf_for_item(row))


def _tie_and_edge_values() -> np.ndarray:
    # k / 32 has an exact 4-decimal tie in binary (x * 10**4 ends in .5).
    ties = np.arange(1, 64, 2, dtype=np.float64) / 32.0
    edges = np.array([0.0, -0.0, np.nan, np.inf, 12.34565, 99.99995, 0.00005, 31.75], dtype=np.float64)
    return np.concatenate([ties, edges]).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scores_match_python_rounding(dtype: type, tmp_path) -> None:
    rng = np.random.default_rng(0)
    edge = _tie_and_edge_values()
    n = len(edge)
    plddt = edge.astype(dtype)
    pae = np.round(rng.uniform(0, 32, (n, n)).astype(np.float32), 3).astype(dtype)
    pae[0] = edge.astype(dtype)
    row = _row(plddt, pae, ptm=0.8123456, iptm=float("nan"), max_pae=31.75)
    udf = _udf(str(tmp_path))

    result = _write(udf, row)

    expected = _expected_scores(udf, row)
    assert result["scores"] == expected
    with open(os.path.join(tmp_path, "rec_scores.json"), "rb") as f:
        assert f.read() == expected.encode()
    assert json.loads(result["scores"])["iptm"] is None


def test_random_float32_scores_match_python_rounding() -> None:
    rng = np.random.default_rng(1)
    n = 96
    plddt = rng.uniform(0, 100, n).astype(np.float32)
    pae = np.round(rng.uniform(0, 32, (n, n)).astype(np.float32), 3)
    row = _row(plddt, pae, ptm=0.5, iptm=0.25, max_pae=float(pae.max()))
    udf = _udf(None)

    result = _write(udf, row)

    assert result["scores"] == _expected_scores(udf, row)
    assert result["output_path"] is None


def test_scores_without_arrays_keep_none_fields() -> None:
    row = _row(None, None)
    udf = _udf(None)

    result = _write(udf, row)

    assert result["scores"] == _expected_scores(udf, row)
    assert json.loads(result["scores"]) == {"plddt": None, "ptm": None, "iptm": None, "pae": None, "max_pae": None}


@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 3), (5,), (2, 3, 4)])
def test_float_array_encoding_matches_json(shape: tuple[int, ...]) -> None:
    values = np.random.default_rng(2).uniform(-2, 2, shape).round(1)
    values.flat[: min(values.size, 2)] = [-0.0, np.nan][: min(values.size, 2)]

    assert _encode_float_array(values) == json.dumps(values.tolist())


def test_get_score_values_keeps_arrays_and_drops_nan_scalars() -> None:
    plddt = np.arange(4, dtype=np.float32)
    record = FoldingOutput(
        atom_positions=np.zeros((4, 37, 3), dtype=np.float32),
        residue_types=np.zeros(4, dtype=np.int64),
        atom_mask=np.ones((4, 37), dtype=np.float32),
        residue_indices=np.arange(4),
        plddt=plddt,
        ptm=np.float32(0.5),
        iptm=float("nan"),
    )
    values = record.get_score_values()
    assert values["plddt"] is plddt
    assert values["pae"] is None
    assert values["ptm"] == 0.5 and isinstance(values["ptm"], float)
    assert values["iptm"] is None and values["max_pae"] is None
    assert record.get_scores() == {"plddt": plddt.tolist(), "ptm": 0.5, "iptm": None, "pae": None, "max_pae": None}


@pytest.mark.parametrize("shape", [(320_001,), (256, 256), (4, 128, 128)])
def test_rounded_float32_array_spells_every_grid_token(shape: tuple[int, ...]) -> None:
    """Every four-decimal score in [0, 32] comes out of the shared token table as ``json`` spells it."""
    scale = writer_stage._SCORE_SCALE
    values = np.resize(np.arange(writer_stage._SCORE_GRID_MAX * scale + 1) / scale, shape).astype(np.float32)
    assert _encode_rounded_float32_array(values) == json.dumps(np.round(values.astype(np.float64), 4).tolist())


def test_large_float32_scores_match_python_rounding() -> None:
    udf = _udf(None)
    row = _large_row()
    assert _write(udf, row)["scores"] == _expected_scores(udf, row)


@pytest.mark.parametrize("special", [None, -0.00001, -0.1, 32.00005, 32.0001, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 3), (300,), (256, 256), (2, 200, 200)])
def test_float32_array_encoding_matches_rounded_json(special: float | None, shape: tuple[int, ...]) -> None:
    """Grid indices from the rounding step give the same text as ``np.round``, at any size."""
    values = np.random.default_rng(5).uniform(0, 32, shape).astype(np.float32)
    if special is not None:
        values.flat[values.size // 2] = special
    expected = json.dumps(np.round(values.astype(np.float64), 4).tolist())
    assert _encode_rounded_float32_array(values) == expected


def _large_row() -> dict:
    """A row whose PAE alone reaches the helper thread's threshold."""
    rng = np.random.default_rng(6)
    n = math.isqrt(writer_stage._SCORE_THREAD_MIN_VALUES)
    return _row(rng.uniform(0, 100, n).astype(np.float32), rng.uniform(0, 32, (n, n)).astype(np.float32))


def test_large_scores_encode_on_helper_thread(tmp_path) -> None:
    """Large score arrays are encoded off the calling thread, with the same bytes."""
    row = _large_row()
    udf = _udf(str(tmp_path))
    threads = []
    encode = udf._encode_scores

    def record_thread(record: FoldingOutput) -> str:
        threads.append(threading.current_thread())
        return encode(record)

    with patch.object(udf, "_encode_scores", side_effect=record_thread):
        result = _write(udf, row)
    assert result["scores"] == _expected_scores(udf, row)
    assert threads and threads[0] is not threading.current_thread()


def test_structure_error_wins_over_score_error(tmp_path) -> None:
    """A structure failure propagates as before, after the helper thread finishes."""
    udf = _udf(str(tmp_path))
    finished = threading.Event()

    def failing_scores(record: FoldingOutput) -> str:
        time.sleep(0.01)
        finished.set()
        raise RuntimeError("scores")

    with (
        patch.object(udf, "_encode_scores", side_effect=failing_scores),
        patch.object(udf, "_create_writer") as create,
    ):
        create.return_value.write.side_effect = ValueError("structure")
        with pytest.raises(ValueError, match="structure"):
            asyncio.run(udf.udf_for_item(_large_row()))
        assert finished.is_set()


def test_score_error_propagates_after_structure(tmp_path) -> None:
    udf = _udf(str(tmp_path))
    with patch.object(udf, "_encode_scores", side_effect=RuntimeError("scores")):
        with pytest.raises(RuntimeError, match="scores"):
            _write(udf, _large_row())
