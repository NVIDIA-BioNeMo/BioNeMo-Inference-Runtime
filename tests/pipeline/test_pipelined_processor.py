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

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.processor.base import ProcessorConfig, SerialProcessor
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.processor.pipelined import PipelinedSerialProcessor, _pack_batch, _ThreadWorker, _unpack_batch
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF

WORKERS = ["thread", "process"]


class _PrologueUDF(StatefulStageUDF):
    """Seeds the global RNGs per row like the model pre_init hooks, then draws from them."""

    prepared = 0

    def __init__(self, *args: Any, fail_on: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on

    def prepare(self) -> None:
        type(self).prepared += 1

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        if row["value"] == self.fail_on:
            raise ValueError(f"bad value {row['value']}")
        np.random.seed(row["random_seed"])
        torch.manual_seed(row["random_seed"])
        return {
            "value": row["value"],
            "random_seed": row["random_seed"],
            "features": torch.randn(4, 3),
            "draw": float(np.random.rand()),
            "mask": np.arange(row["value"]) % 2 == 0,
            "prologue_pid": os.getpid(),
            "prologue_thread": threading.get_ident(),
            "prepared": type(self).prepared,
        }


class _DeviceUDF(StatefulStageUDF):
    """Stand-in for the engine: records the call order and may raise."""

    calls: list[Any] = []

    def __init__(self, *args: Any, raise_on: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raise_on = raise_on

    async def udf_for_rows(self, rows: list[dict[str, Any]]) -> Any:
        for row in rows:
            type(self).calls.append(row["value"])
            if row["value"] == self.raise_on:
                raise RuntimeError("engine failure")
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                "value": row["value"],
                "draw": row["draw"],
                "features": row["features"],
                "mask": row["mask"],
                "prologue_pid": row["prologue_pid"],
                "prologue_thread": row["prologue_thread"],
                "prepared": row["prepared"],
                "coords": (row["features"] * 2).numpy(),
                "engine_thread": threading.get_ident(),
                "stage_timing_s": {**row.get("stage_timing_s", {}), "Device": 0.0},
                "__inference_error__": {"error_msg": None, "traceback": None},
            }


class _EpilogueUDF(StatefulStageUDF):
    pack_output = False

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "value": row["value"],
            "draw": row["draw"],
            "features_sum": float(row["features"].sum()),
            "coords_sum": float(row["coords"].sum()),
            "mask_sum": int(row["mask"].sum()),
            "prologue_pid": row["prologue_pid"],
            "prologue_thread": row["prologue_thread"],
            "engine_thread": row["engine_thread"],
            "epilogue_pid": os.getpid(),
            "prepared": row["prepared"],
        }


class _WritingEpilogueUDF(_EpilogueUDF):
    def __init__(self, *args: Any, output_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.output_path = Path(output_path)

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        output = await super().udf_for_item(row)
        (self.output_path / f"{row['value']}.txt").write_text(str(output))
        return output


def _stages(fail_on: str | None = None, raise_on: Any = None) -> list[StatefulStage]:
    return [
        StatefulStage(fn=_PrologueUDF, fn_constructor_kwargs={"fail_on": fail_on}),
        StatefulStage(fn=_DeviceUDF, fn_constructor_kwargs={"raise_on": raise_on}, map_batches_kwargs={"num_gpus": 1}),
        StatefulStage(fn=_EpilogueUDF),
    ]


def _records(count: int) -> list[dict[str, Any]]:
    return [{"value": i + 1, "random_seed": 100 + i, "__record_id": f"r{i}"} for i in range(count)]


_INPUT_KEYS = {"__record_id", "value", "draw", "features_sum", "coords_sum", "mask_sum"}


def _comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k in _INPUT_KEYS} for row in rows]


@pytest.fixture(autouse=True)
def _reset_device_calls() -> None:
    _DeviceUDF.calls = []
    _PrologueUDF.prepared = 0


@pytest.mark.parametrize("workers", WORKERS)
def test_matches_serial_and_keeps_order(workers: str) -> None:
    records = _records(5)
    expected = SerialProcessor(ProcessorConfig(model_source="test"), _stages())(records)
    _DeviceUDF.calls = []
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), _stages(), workers=workers) as processor:
        actual = processor(records)
        again = processor(records)

    assert _comparable(actual) == _comparable(expected)
    assert _comparable(again) == _comparable(expected)
    assert [row["value"] for row in actual] == [1, 2, 3, 4, 5]
    assert _DeviceUDF.calls == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert all((row.get("__inference_error__") or {}).get("error_msg") is None for row in actual)


@pytest.mark.parametrize("workers", WORKERS)
def test_cpu_stages_run_off_the_engine_thread(workers: str) -> None:
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), _stages(), workers=workers) as processor:
        rows = processor(_records(3))

    for row in rows:
        assert row["engine_thread"] == threading.get_ident()
        if workers == "thread":
            assert row["prologue_pid"] == os.getpid()
            assert row["prologue_thread"] != row["engine_thread"]
        else:
            assert row["prologue_pid"] != os.getpid()
            assert row["epilogue_pid"] not in (os.getpid(), row["prologue_pid"])
        assert row["prepared"] == 1


@pytest.mark.parametrize("workers", WORKERS)
def test_row_error_does_not_stop_the_stream(workers: str) -> None:
    records = _records(4)
    with PipelinedSerialProcessor(
        ProcessorConfig(model_source="test"), _stages(fail_on=2), workers=workers
    ) as processor:
        rows = processor(records)

    assert [row["__record_id"] for row in rows] == ["r0", "r1", "r2", "r3"]
    assert "ValueError: bad value 2" in rows[1]["__inference_error__"]["error_msg"]
    assert rows[1].get("draw") is None
    assert [row["value"] for row in rows if row["__inference_error__"]["error_msg"] is None] == [1, 3, 4]
    assert _DeviceUDF.calls == [1, 3, 4]


@pytest.mark.parametrize("workers", WORKERS)
@pytest.mark.parametrize("raise_on", [1, 4])
def test_engine_exception_aborts_and_processor_recovers(workers: str, raise_on: int, tmp_path: Path) -> None:
    stages = _stages(raise_on=raise_on)
    stages[-1] = StatefulStage(fn=_WritingEpilogueUDF, fn_constructor_kwargs={"output_path": str(tmp_path)})
    config = ProcessorConfig(model_source="test")
    with pytest.raises(RuntimeError, match="engine failure"):
        SerialProcessor(config, stages)(_records(4))
    assert not list(tmp_path.iterdir())

    processor = PipelinedSerialProcessor(config, stages, workers=workers)
    with pytest.raises(RuntimeError, match="engine failure"):
        processor(_records(4))
    assert processor._prologue is None and processor._epilogue is None
    assert not list(tmp_path.iterdir())

    records = _records(1)
    records[0]["value"] = 5
    rows = processor(records)
    processor.close()
    assert [row["value"] for row in rows] == [5]
    assert [path.name for path in tmp_path.iterdir()] == ["5.txt"]


def test_unnamed_outputs_keep_batch_indices(tmp_path: Path) -> None:
    class _IndexedWriter(_EpilogueUDF):
        async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
            (tmp_path / f"{row[self.IDX_IN_BATCH_COLUMN]}.txt").write_text(str(row["value"]))
            return await super().udf_for_item(row)

    stages = _stages()
    stages[-1] = StatefulStage(fn=_IndexedWriter)
    records = [{**row, "__record_id": None} for row in _records(4)]
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages) as processor:
        processor(records)
    assert {path.name: path.read_text() for path in tmp_path.iterdir()} == {f"{i}.txt": str(i + 1) for i in range(4)}


@pytest.mark.parametrize("values", [[1, 2], [2, 1]])
def test_epilogue_preserves_heterogeneous_columns(values: list[int]) -> None:
    class _SparseDevice(StatefulStageUDF):
        pack_output = False

        async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
            return {"first" if row["value"] == 1 else "second": row["value"]}

    class _IdentityEpilogue(StatefulStageUDF):
        pack_output = False

        async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
            return row

    stages = [
        StatefulStage(fn=_SparseDevice, update_row=False, map_batches_kwargs={"num_gpus": 1}),
        StatefulStage(fn=_IdentityEpilogue),
    ]
    records = [{"value": value, "__record_id": f"r{value}"} for value in values]
    config = ProcessorConfig(model_source="test")
    expected = SerialProcessor(config, stages)(records)
    with PipelinedSerialProcessor(config, stages) as processor:
        actual = processor(records)

    assert [{k: v for k, v in row.items() if k != "stage_timing_s"} for row in actual] == [
        {k: v for k, v in row.items() if k != "stage_timing_s"} for row in expected
    ]
    assert [row["first"] for row in actual] == [1 if value == 1 else None for value in values]
    assert [row["second"] for row in actual] == [2 if value == 2 else None for value in values]


def test_thread_abort_waits_for_active_work() -> None:
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class _Blocking(_PrologueUDF):
        async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
            started.set()
            release.wait(timeout=10)
            return await super().udf_for_item(row)

    worker = _ThreadWorker([("prologue", StatefulStage(fn=_Blocking))])
    worker.submit(0, SerialProcessor._rows_to_columnar(_records(1)))
    assert started.wait(timeout=10)

    def close() -> None:
        worker.close(abort=True)
        closed.set()

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert not closed.wait(timeout=0.1)
    finally:
        release.set()
        closer.join(timeout=10)
    assert closed.is_set()
    assert not worker._thread.is_alive()


def test_prefetch_bounds_the_prologue() -> None:
    seen: list[int] = []

    class _Recording(_PrologueUDF):
        async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
            seen.append(row["value"])
            return await super().udf_for_item(row)

    stages = _stages()
    stages[0] = StatefulStage(fn=_Recording)
    started = threading.Event()
    release = threading.Event()

    class _Blocking(_DeviceUDF):
        async def udf_for_rows(self, rows: list[dict[str, Any]]) -> Any:
            started.set()
            release.wait(timeout=10)
            async for item in super().udf_for_rows(rows):
                yield item

    stages[1] = StatefulStage(fn=_Blocking, map_batches_kwargs={"num_gpus": 1})
    processor = PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages, workers="thread", prefetch=2)
    worker = threading.Thread(target=processor, args=(_records(8),))
    worker.start()
    assert started.wait(timeout=10)
    deadline = threading.Event()
    deadline.wait(0.3)
    # Engine holds request 1; only requests 2 and 3 may be prepared ahead.
    assert sorted(seen) == [1, 2, 3]
    release.set()
    worker.join(timeout=20)
    processor.close()
    assert sorted(seen) == list(range(1, 9))


def test_without_device_stage_runs_serially() -> None:
    stages = [StatefulStage(fn=_PrologueUDF)]
    records = _records(3)
    expected = SerialProcessor(ProcessorConfig(model_source="test"), stages)(records)
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages) as processor:
        rows = processor(records)
    assert [row["__data__"]["draw"] for row in rows] == [row["__data__"]["draw"] for row in expected]
    assert processor._prologue is None


def test_pack_batch_round_trip_preserves_values_and_order() -> None:
    readonly = np.arange(6, dtype=np.float32).reshape(2, 3)
    readonly.setflags(write=False)
    row = {
        "a": torch.arange(10, dtype=torch.int64),
        "b": np.array([True, False, True]),
        "c": torch.empty(0, 3),
        "d": readonly,
        "e": np.array(2.5),
        "f": np.array(["x", "y"]),
        "g": np.array([{"k": 1}], dtype=object),
        "h": "text",
        "i": torch.ones(2, 2, dtype=torch.float16)[:, :1],
    }
    batch = {"__record_id": ["r0"], "__data__": [row]}

    packed = _pack_batch(batch)
    assert packed["__data__"][0]["a"] is None
    assert packed["__data__"][0]["h"] == "text"
    assert packed["__data__"][0]["f"] is row["f"]
    assert row["a"] is not None

    restored = _unpack_batch(packed)["__data__"][0]
    assert list(restored) == list(row)
    for key in "abcdei":
        value, original = restored[key], row[key]
        assert type(value) is type(original), key
        assert tuple(value.shape) == tuple(original.shape), key
        if isinstance(value, torch.Tensor):
            assert value.dtype == original.dtype and torch.equal(value, original), key
        else:
            assert value.dtype == original.dtype and np.array_equal(value, original), key
    assert restored["d"].flags.writeable
    assert restored["f"].tolist() == ["x", "y"]
    assert restored["g"][0] == {"k": 1}
    assert _unpack_batch({"x": [1]}) == {"x": [1]}


def test_prefetch_rejects_zero() -> None:
    with pytest.raises(ValueError, match="prefetch"):
        PipelinedSerialProcessor(ProcessorConfig(model_source="test"), [], prefetch=0)


def test_build_processor_uses_threads() -> None:
    config = EngineProcessorConfig(model_source="test")
    with (
        patch("bionemo_ir.pipeline.processor.engine_proc._build_stages", return_value=_stages()),
        patch("bionemo_ir.pipeline.processor.engine_proc._resolve_metadata"),
        patch("bionemo_ir.pipeline.processor.engine_proc._resolve_runtime_args"),
    ):
        processor = build_processor(config)
    assert type(processor) is PipelinedSerialProcessor
    assert processor.workers == "thread"
    assert processor.prefetch == 2
    assert "stage_overlap" not in type(config).model_fields
    assert "prefetch" not in type(config).model_fields


_ORPHAN_SCRIPT = """
import os, signal, sys
sys.path.insert(0, sys.argv[1])
import test_pipelined_processor as fakes
from bionemo_ir.pipeline.processor.base import ProcessorConfig
from bionemo_ir.pipeline.processor.pipelined import PipelinedSerialProcessor

if __name__ == "__main__":
    processor = PipelinedSerialProcessor(ProcessorConfig(model_source="test"), fakes._stages(), workers="process")
    processor(fakes._records(2))
    print(processor._prologue._process.pid, processor._epilogue._process.pid, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_process_workers_exit_when_parent_dies() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _ORPHAN_SCRIPT, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    pids = [int(pid) for pid in proc.stdout.split()]
    assert len(pids) == 2
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.2)
        assert not any(_alive(pid) for pid in pids), "stage workers outlived the parent"
    finally:
        for pid in pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
