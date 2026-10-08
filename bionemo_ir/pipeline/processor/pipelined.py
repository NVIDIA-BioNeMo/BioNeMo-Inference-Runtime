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

"""Serial executor that overlaps the CPU stages of neighbouring requests with the device stage."""

import asyncio
import contextlib
import multiprocessing
import os
import pickle
import queue
import threading
import traceback
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.multiprocessing  # noqa: F401  registers the shared-memory tensor reducers

from bionemo_ir.logger import logger
from bionemo_ir.pipeline.processor.base import ProcessorConfig, SerialProcessor, build_stage_udf
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF

StageGroup = list[tuple[str, StatefulStage]]
# index, batch, formatted traceback
_Message = tuple[int, dict[str, Any] | None, str | None]

_SHARED_KEY = "__shared_tensors__"
_ALIGN = 16


def _run_stages(udfs: Sequence[StatefulStageUDF], batch: dict[str, Any]) -> dict[str, Any]:
    for udf in udfs:
        batch = asyncio.run(SerialProcessor._run_udf(udf, batch))
    return batch


def _build_udfs(stages: StageGroup) -> list[StatefulStageUDF]:
    udfs = [build_stage_udf(stage) for _, stage in stages]
    for udf in udfs:
        udf.prepare()
    return udfs


def _as_cpu_tensor(value: Any) -> torch.Tensor | None:
    """Return a CPU tensor sharing ``value``'s memory, or None when pickling must handle it."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if not isinstance(value, np.ndarray) or value.dtype == object:
        return None
    # ascontiguousarray would promote 0-d arrays to 1-d.
    array = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    if not array.flags.writeable:
        array = array.copy()
    try:
        return torch.from_numpy(array)
    except TypeError:
        return None


def _pack_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Move the tensor and array values of packed rows into one shared-memory buffer.

    One buffer per batch keeps the handoff to one file descriptor; the receiving
    process maps it once and views every value in place.
    """
    rows = batch.get(StatefulStageUDF.DATA_COLUMN)
    if not isinstance(rows, list):
        return batch
    slots: list[tuple[int, str, bool, torch.dtype, tuple[int, ...], int, int]] = []
    chunks: list[tuple[int, torch.Tensor]] = []
    total = 0
    out_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        out: dict[str, Any] = {}
        for key, value in row.items():
            tensor = _as_cpu_tensor(value)
            if tensor is None:
                out[key] = value
                continue
            flat = tensor.contiguous().view(-1)
            nbytes = flat.numel() * flat.element_size()
            offset = -(-total // _ALIGN) * _ALIGN
            slots.append(
                (row_index, key, isinstance(value, np.ndarray), flat.dtype, tuple(tensor.shape), offset, nbytes)
            )
            chunks.append((offset, flat))
            total = offset + nbytes
            out[key] = None
        out_rows.append(out)
    if not slots:
        return batch
    buffer = torch.empty(0, dtype=torch.uint8).set_(torch.UntypedStorage._new_shared(max(total, 1)))
    for offset, flat in chunks:
        if flat.numel():
            buffer[offset : offset + flat.numel() * flat.element_size()].view(flat.dtype).copy_(flat)
    return {**batch, StatefulStageUDF.DATA_COLUMN: out_rows, _SHARED_KEY: (slots, buffer)}


def _unpack_batch(batch: dict[str, Any]) -> dict[str, Any]:
    shared = batch.pop(_SHARED_KEY, None)
    if shared is None:
        return batch
    slots, buffer = shared
    rows = batch[StatefulStageUDF.DATA_COLUMN]
    for row_index, key, is_array, dtype, shape, offset, nbytes in slots:
        if nbytes:
            tensor = buffer[offset : offset + nbytes].view(dtype).view(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        rows[row_index][key] = tensor.numpy() if is_array else tensor
    return batch


class _StageGroupWorker:
    """Runs one stage group over a stream of batches, preserving submission order."""

    def submit(self, index: int, batch: dict[str, Any]) -> None:
        raise NotImplementedError

    def result(self) -> tuple[int, dict[str, Any]]:
        raise NotImplementedError

    def close(self, abort: bool = False) -> None:
        raise NotImplementedError


def _next_request(requests: Any, parent_pid: int | None) -> tuple[int, dict[str, Any]] | None:
    """Block for the next request; None once the queue closes or the parent process is gone."""
    while True:
        try:
            return requests.get(timeout=1.0)
        except queue.Empty:
            if parent_pid is not None and os.getppid() != parent_pid:
                return None


def _serve(stages: StageGroup, requests: Any, results: Any, shared_memory: bool, parent_pid: int | None = None) -> None:
    try:
        udfs = _build_udfs(stages)
    except Exception:
        results.put((-1, None, traceback.format_exc()))
        return
    while (item := _next_request(requests, parent_pid)) is not None:
        index, batch = item
        try:
            if shared_memory:
                batch = _unpack_batch(batch)
            batch = _run_stages(udfs, batch)
            if shared_memory:
                batch = _pack_batch(batch)
        except Exception:
            results.put((index, None, traceback.format_exc()))
        else:
            results.put((index, batch, None))


def _serve_process(stages_blob: bytes, requests: Any, results: Any, parent_pid: int) -> None:
    # A spawned worker outlives a killed parent; the pid check lets it exit instead of idling forever.
    _serve(pickle.loads(stages_blob), requests, results, shared_memory=True, parent_pid=parent_pid)


def _drain(requests: Any) -> None:
    while True:
        try:
            requests.get_nowait()
        except queue.Empty:
            return


class _ThreadWorker(_StageGroupWorker):
    def __init__(self, stages: StageGroup) -> None:
        self._requests: queue.Queue[tuple[int, dict[str, Any]] | None] = queue.Queue()
        self._results: queue.Queue[_Message] = queue.Queue()
        self._thread = threading.Thread(
            target=_serve,
            args=(stages, self._requests, self._results, False),
            name="bioir-stage-group",
            daemon=True,
        )
        self._thread.start()

    def submit(self, index: int, batch: dict[str, Any]) -> None:
        self._requests.put((index, batch))

    def result(self) -> tuple[int, dict[str, Any]]:
        index, batch, error = self._results.get()
        if error is not None:
            raise RuntimeError(f"stage worker failed:\n{error}")
        return index, batch

    def close(self, abort: bool = False) -> None:
        _drain(self._requests)
        self._requests.put(None)
        self._thread.join()


class _ProcessWorker(_StageGroupWorker):
    def __init__(self, stages: StageGroup) -> None:
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue()
        self._results = context.Queue()
        self._process = context.Process(
            target=_serve_process,
            args=(pickle.dumps(stages), self._requests, self._results, os.getpid()),
            name="bioir-stage-group",
            daemon=True,
        )
        self._process.start()

    def submit(self, index: int, batch: dict[str, Any]) -> None:
        self._requests.put((index, _pack_batch(batch)))

    def result(self) -> tuple[int, dict[str, Any]]:
        while True:
            try:
                index, batch, error = self._results.get(timeout=1.0)
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError(f"stage worker exited with code {self._process.exitcode}") from None
                continue
            if error is not None:
                raise RuntimeError(f"stage worker failed:\n{error}")
            return index, _unpack_batch(batch)

    def close(self, abort: bool = False) -> None:
        if self._process.is_alive():
            if abort:
                self._process.terminate()
            else:
                self._requests.put(None)
            self._process.join(timeout=60.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        self._requests.close()
        self._results.close()


class PipelinedSerialProcessor(SerialProcessor):
    """Serial processor whose CPU stages run alongside the device stage of neighbouring requests.

    Stages before the first device stage (one with ``num_gpus`` in its
    ``map_batches_kwargs``) form the prologue and stages after the last device
    stage the epilogue. Each group runs on its own worker and the device stages
    run on the calling thread, so request ``N+1`` is parsed and featurized
    while the engine runs request ``N``. The epilogue receives buffered
    outputs only after every device batch succeeds, preventing writer side
    effects on engine failure. Workers start on the first call and persist
    across calls; :meth:`close` stops them.

    Output rows keep the order of the input records and the per-row error
    semantics of :class:`SerialProcessor`; an engine exception still aborts the
    call. ``stage_timing_s`` still holds each stage's own wall time, but the
    stages of one request no longer run back to back, so the values do not add
    up to the request's wall time.

    Args:
        config: Processor configuration.
        stages: Pipeline stages in execution order.
        workers: ``"thread"`` shares the interpreter with the engine; ``"process"``
            spawns one interpreter per stage group and hands rows over through
            shared memory, which keeps the engine's launch thread clear of the
            GIL and isolates the per-request RNG seeding.
        prefetch: Requests prepared ahead of the engine. Device outputs
            buffer for the entire call.
    """

    def __init__(
        self,
        config: ProcessorConfig,
        stages: list[StatefulStage],
        workers: Literal["thread", "process"] = "thread",
        prefetch: int = 1,
    ) -> None:
        super().__init__(config, stages)
        if prefetch < 1:
            raise ValueError(f"prefetch must be positive, got {prefetch}")
        self.workers = workers
        self.prefetch = prefetch
        self._prologue: _StageGroupWorker | None = None
        self._epilogue: _StageGroupWorker | None = None

    def _stage_groups(self) -> tuple[StageGroup, StageGroup, StageGroup]:
        items = list(self.stages.items())
        device = [i for i, (_, stage) in enumerate(items) if stage.map_batches_kwargs.get("num_gpus")]
        if not device:
            return items, [], []
        return items[: device[0]], items[device[0] : device[-1] + 1], items[device[-1] + 1 :]

    def get_stage_udf(self, name: str) -> StatefulStageUDF:
        """Return a live UDF on the calling thread, such as the engine stage.

        Worker-owned CPU stages cannot be configured through this accessor.
        Without a device stage, every stage executes on the calling thread.

        Args:
            name: Stage name from :meth:`list_stage_names`.

        Returns:
            The cached stage UDF. Unknown or worker-owned names raise
            ``ValueError``.
        """
        self.get_stage_by_name(name)
        _, device_stages, _ = self._stage_groups()
        if device_stages and name not in dict(device_stages):
            raise ValueError(f"Stage {name} executes in a worker; its live UDF is unavailable")
        return super().get_stage_udf(name)

    def _spawn(self, stages: StageGroup) -> _StageGroupWorker | None:
        if not stages:
            return None
        logger.debug("Starting %s worker for stages %s", self.workers, [name for name, _ in stages])
        if self.workers == "process":
            return _ProcessWorker(stages)
        return _ThreadWorker(stages)

    def __call__(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run all stages on the given records, overlapping CPU stages across requests.

        Args:
            records: List of input dicts, same format as rows passed to
                ``ray.data.from_items``.

        Returns:
            List of output dicts (flat, no packing), one per record, in order.
        """
        self._validate_input_keys(key for record in records for key in record)
        prologue_stages, device_stages, epilogue_stages = self._stage_groups()
        if not records or not device_stages:
            return super().__call__(records)
        if self._prologue is None:
            self._prologue = self._spawn(prologue_stages)
        if self._epilogue is None:
            self._epilogue = self._spawn(epilogue_stages)
        batches = [self._rows_to_columnar([record]) for record in records]
        try:
            outputs = self._run_pipeline(batches, device_stages)
        except BaseException:
            self.close(abort=True)
            raise
        return [row for batch in outputs for row in self._columnar_to_rows(batch)]

    def _run_pipeline(self, batches: list[dict[str, Any]], device_stages: StageGroup) -> list[dict[str, Any]]:
        count = len(batches)
        prologue, epilogue = self._prologue, self._epilogue
        submitted = 0

        def feed(limit: int) -> None:
            nonlocal submitted
            while submitted < min(count, limit):
                prologue.submit(submitted, batches[submitted])
                submitted += 1

        if prologue is not None:
            feed(self.prefetch + 1)
        # Model construction on the first call overlaps the prologue of the first requests.
        udfs = [self._get_or_create_udf(name, stage) for name, stage in device_stages]

        outputs: list[dict[str, Any] | None] = [None] * count
        for index in range(count):
            if prologue is not None:
                got, batch = prologue.result()
                if got != index:
                    raise RuntimeError(f"prologue returned request {got}, expected {index}")
                feed(index + 1 + self.prefetch)
            else:
                batch = batches[index]
            outputs[index] = _run_stages(udfs, batch)
        if epilogue is not None:
            # Preserve batch indices for unnamed output filenames.
            rows = [row for output in outputs for row in self._columnar_to_rows(output)]
            batch = self._rows_to_columnar(rows)
            epilogue.submit(0, batch)
            _, batch = epilogue.result()
            return [batch]
        return outputs

    def close(self, abort: bool = False) -> None:
        """Stop the stage workers; they restart on the next call.

        Args:
            abort: Drop queued requests instead of finishing them.
        """
        for worker in (self._prologue, self._epilogue):
            if worker is not None:
                worker.close(abort)
        self._prologue = None
        self._epilogue = None

    def __enter__(self) -> "PipelinedSerialProcessor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close(abort=True)
