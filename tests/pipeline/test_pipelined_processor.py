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
from bionemo_ir.pipeline.processor.pipelined import (
    PipelinedSerialProcessor,
    _pack_batch,
    _ProcessWorker,
    _ThreadWorker,
    _unpack_batch,
)
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


class _FailingPrepareUDF(_PrologueUDF):
    def prepare(self) -> None:
        raise OSError("no CCD")


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
def test_live_device_stage_configuration_is_used(workers: str) -> None:
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), _stages(), workers=workers) as processor:
        prologue, device, epilogue = processor.list_stage_names()
        udf = processor.get_stage_udf(device)
        assert processor.get_stage_udf(device) is udf
        for name in (prologue, epilogue):
            with pytest.raises(ValueError, match="executes in a worker"):
                processor.get_stage_udf(name)
        udf.raise_on = 1
        with pytest.raises(RuntimeError, match="engine failure"):
            processor(_records(1))


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


def test_prologue_prepares_before_the_first_call() -> None:
    prepared = threading.Event()

    class _Preparing(_PrologueUDF):
        def prepare(self) -> None:
            super().prepare()
            prepared.set()

    stages = _stages()
    stages[0] = StatefulStage(fn=_Preparing)
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages, workers="thread") as processor:
        assert prepared.is_set()
        rows = processor(_records(2))
    assert [row["prepared"] for row in rows] == [1, 1]


@pytest.mark.parametrize("workers", WORKERS)
def test_prepare_failure_surfaces_on_the_first_call(workers: str) -> None:
    stages = _stages()
    stages[0] = StatefulStage(fn=_FailingPrepareUDF)
    with PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages, workers=workers) as processor:
        with pytest.raises(RuntimeError, match="no CCD"):
            processor(_records(1))


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


def test_process_processor_can_be_constructed_at_module_scope(tmp_path: Path) -> None:
    script = tmp_path / "module_processor.py"
    script.write_text(
        """
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import test_pipelined_processor as fakes
from bionemo_ir.pipeline.processor.base import ProcessorConfig
from bionemo_ir.pipeline.processor.pipelined import PipelinedSerialProcessor

processor = PipelinedSerialProcessor(ProcessorConfig(model_source="test"), fakes._stages(), workers="process")

if __name__ == "__main__":
    with processor:
        rows = processor(fakes._records(2))
        assert [row["value"] for row in rows] == [1, 2]
        assert all(row["prepared"] == 1 for row in rows)
    print("processed", flush=True)
"""
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).parent), str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "processed"


_PROLOGUE = [("prologue", StatefulStage(fn=_PrologueUDF))]


def test_dropped_thread_worker_stops_its_thread() -> None:
    worker = _ThreadWorker(_PROLOGUE)
    thread = worker._thread
    del worker
    assert not thread.is_alive()


@pytest.mark.parametrize("workers", WORKERS)
def test_interrupted_close_can_be_retried(workers: str) -> None:
    worker = _ThreadWorker(_PROLOGUE) if workers == "thread" else _ProcessWorker(_PROLOGUE)
    stop = "_stop_thread" if workers == "thread" else "_stop_process"
    with patch(f"bionemo_ir.pipeline.processor.pipelined.{stop}", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            worker.close()
    worker.close()
    handle = worker._thread if isinstance(worker, _ThreadWorker) else worker._process
    assert not handle.is_alive()


# Shared header for the exit scripts. The module-level reference to the pipelined
# module keeps it alive into shutdown, where Python clears its globals before
# __main__ releases the processor. _stop_* print once they finish, so a script
# shows which workers stopped at exit.
_EXIT_HEADER = """
import os, sys, threading
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import test_pipelined_processor as fakes
from bionemo_ir.pipeline.processor import pipelined
from bionemo_ir.pipeline.processor.base import ProcessorConfig
from bionemo_ir.pipeline.stages.base import StatefulStage


def _logged(stop, name):
    def run(*args, **kwargs):
        stop(*args, **kwargs)
        print(name, flush=True)
    return run


pipelined._stop_thread = _logged(pipelined._stop_thread, "stopped thread")
pipelined._stop_process = _logged(pipelined._stop_process, "stopped process")
"""


def _run_exit_script(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "exit_processor.py"
    script.write_text(_EXIT_HEADER + body)
    return subprocess.run(
        [sys.executable, str(script), str(Path(__file__).parent), str(Path(__file__).resolve().parents[2]), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("workers", WORKERS)
def test_processor_alive_at_exit_closes_quietly(workers: str, tmp_path: Path) -> None:
    proc = _run_exit_script(
        tmp_path,
        """
processor = pipelined.PipelinedSerialProcessor(ProcessorConfig(model_source="test"), fakes._stages(), workers=sys.argv[3])

if __name__ == "__main__":
    rows = processor(fakes._records(2))
    assert [row["value"] for row in rows] == [1, 2]
    print("processed", flush=True)
""",
        workers,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr
    # Process workers stop at exit; daemon thread workers are abandoned.
    stopped = ["stopped process"] * 2 if workers == "process" else []
    assert proc.stdout.split("\n")[:-1] == ["processed", *stopped]


def test_exit_does_not_wait_for_a_busy_thread_worker(tmp_path: Path) -> None:
    proc = _run_exit_script(
        tmp_path,
        """
started = threading.Event()


class _Blocked(fakes._PrologueUDF):
    async def udf_for_item(self, row):
        started.set()
        threading.Event().wait()


if __name__ == "__main__":
    stages = fakes._stages()
    stages[0] = StatefulStage(fn=_Blocked)
    processor = pipelined.PipelinedSerialProcessor(ProcessorConfig(model_source="test"), stages)
    threading.Thread(target=processor, args=(fakes._records(1),), daemon=True).start()
    assert started.wait(timeout=60)
    print("exiting", flush=True)
""",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split("\n")[:-1] == ["exiting"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_leaves_parent_process_workers_alone(tmp_path: Path) -> None:
    proc = _run_exit_script(
        tmp_path,
        """
if __name__ == "__main__":
    processor = pipelined.PipelinedSerialProcessor(ProcessorConfig(model_source="test"), fakes._stages(), workers="process")
    processor(fakes._records(1))
    if os.fork() == 0:
        sys.exit(0)
    os.wait()
    print("child exited", flush=True)
""",
    )
    assert proc.returncode == 0, proc.stderr
    # The child runs the inherited finalizers at exit; they must return without touching
    # the parent's workers, which would raise "can only test a child process".
    assert "_stop_process" not in proc.stderr, proc.stderr
    assert "child exited" in proc.stdout.split("\n")


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
        if sys.platform == "linux":
            state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
            return state not in {"Z", "X", "x"}
    except (ProcessLookupError, FileNotFoundError):
        return False
    return True


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process states")
def test_alive_excludes_zombies() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _alive(proc.pid):
            time.sleep(0.01)
        assert not _alive(proc.pid)
        assert _alive(os.getpid())
    finally:
        proc.wait(timeout=10)


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
