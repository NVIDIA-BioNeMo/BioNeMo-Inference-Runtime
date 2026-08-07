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
"""Memory estimation and the pre-capture capacity gate for CUDA graphs.

Capturing a ``torch.cuda.CUDAGraph`` pins a private memory pool sized to the
module's working set (its static input/output buffers plus the graph's own
bookkeeping). Before capturing, the tracker calls
:func:`check_capacity_for_capture` to confirm there is enough free GPU (and
host) memory; if not, it reverts to eager execution instead of risking an OOM
mid-capture. The driver/host queries (:func:`gpu_free_bytes`,
:func:`host_available_bytes`) are module-level so tests can monkeypatch them.
"""

from typing import Any, NamedTuple

import torch

MB = 1 << 20

# Fixed overhead charged on top of the measured working set: the CUDA graph's
# instantiated executable and per-node metadata that do not show up as input/
# output tensor bytes.
GRAPH_METADATA_BYTES = 2 * MB

# Always keep at least this much GPU headroom free after a capture, even for a
# tiny working set, so other allocations (and the caching allocator's own
# fragmentation) have room.
MIN_GPU_MARGIN_BYTES = 64 * MB


class MemoryCheck(NamedTuple):
    """Result of :func:`check_capacity_for_capture`.

    Attributes:
        ok: Whether there is enough memory to safely capture.
        reason: Human-readable explanation (``"ok"`` when ``ok`` is ``True``).
        needed_gpu_bytes: Estimated GPU bytes the capture would require.
        free_gpu_bytes: GPU bytes free at the time of the check.
    """

    ok: bool
    reason: str
    needed_gpu_bytes: int
    free_gpu_bytes: int


def tensor_bytes(value: Any) -> int:
    """Recursively sum the byte sizes of every tensor in ``value``.

    Walks nested dicts / lists / tuples; non-tensor leaves contribute 0.
    """
    if isinstance(value, torch.Tensor):
        return value.element_size() * value.numel()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def estimate_capture_gpu_bytes(working_set_bytes: int) -> int:
    """Estimate the GPU bytes a capture needs for a given working set.

    Adds the fixed graph metadata overhead and a safety margin (the larger of
    :data:`MIN_GPU_MARGIN_BYTES` and 10% of the working set). Always strictly
    larger than ``working_set_bytes``.

    ``working_set_bytes`` should be the memory the capture holds resident: the
    static input/output buffers plus the forward's intermediate activations. The
    caller measures the latter during warmup (see the tracker's ``_warmup_call``)
    since it is not derivable from the input/output shapes alone; when that
    measurement is unavailable it degrades to an input/output-only estimate.
    """
    margin = max(MIN_GPU_MARGIN_BYTES, int(0.10 * working_set_bytes))
    return working_set_bytes + GRAPH_METADATA_BYTES + margin


def gpu_free_bytes(device: torch.device | int | None = None) -> int:
    """Return the number of free bytes on the CUDA ``device`` (default current)."""
    free, _total = torch.cuda.mem_get_info(device)
    return free


def host_available_bytes() -> int:
    """Return available host RAM in bytes (``MemAvailable`` from /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # value is in kB
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # Fallback: assume plenty so the gate does not block when we cannot measure.
    return 1 << 62


def container_device(value: Any) -> torch.device | None:
    """Return the device of the first tensor found in ``value`` (recursively),
    or ``None`` when it holds no tensor.

    Walks nested dicts / lists / tuples like :func:`tensor_bytes`; used to pin the
    GPU memory check (and the warmup activation measurement) to the device the
    (static) inputs actually live on.
    """
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for v in value.values():
            device = container_device(v)
            if device is not None:
                return device
    elif isinstance(value, (list, tuple)):
        for v in value:
            device = container_device(v)
            if device is not None:
                return device
    return None


def check_capacity_for_capture(working_set_bytes: int, input_container: Any = None) -> MemoryCheck:
    """Decide whether a graph for ``working_set_bytes`` can be safely captured.

    The GPU free-memory check targets the device the inputs live on, taken from
    the first tensor in ``input_container`` (e.g. a key's ``static_input_kwargs``);
    when ``input_container`` holds no tensor the current CUDA device is used.

    Blocks (``ok=False``) when the estimated GPU requirement exceeds free GPU
    memory, or when host RAM is below the working set plus graph metadata
    (capture also allocates host-side bookkeeping). The ``reason`` mentions
    ``"GPU"`` or ``"host"`` so the caller can log which gate tripped.
    """
    device = container_device(input_container)
    needed_gpu = estimate_capture_gpu_bytes(working_set_bytes)
    free_gpu = gpu_free_bytes(device)
    if needed_gpu > free_gpu:
        return MemoryCheck(
            False, f"insufficient GPU memory: need {needed_gpu} B, free {free_gpu} B", needed_gpu, free_gpu
        )

    needed_host = working_set_bytes + GRAPH_METADATA_BYTES
    host_free = host_available_bytes()
    if host_free < needed_host:
        return MemoryCheck(
            False, f"insufficient host memory: need {needed_host} B, free {host_free} B", needed_gpu, free_gpu
        )

    return MemoryCheck(True, "ok", needed_gpu, free_gpu)
