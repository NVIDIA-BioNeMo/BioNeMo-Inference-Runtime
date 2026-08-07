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
"""CUDA-graph memory estimates and pre-capture capacity checks.

Capture is rejected when its resident working set would exhaust GPU or host
memory.
"""

from typing import Any, NamedTuple

import torch

MB = 1 << 20

# Executable and node metadata omitted from tensor-byte accounting.
GRAPH_METADATA_BYTES = 2 * MB

# Minimum headroom for unrelated allocations and fragmentation.
MIN_GPU_MARGIN_BYTES = 64 * MB


class MemoryCheck(NamedTuple):
    """Capture-capacity result.

    Attributes:
        ok: Whether capture fits.
        reason: Result explanation.
        needed_gpu_bytes: Estimated requirement.
        free_gpu_bytes: Available GPU memory.
    """

    ok: bool
    reason: str
    needed_gpu_bytes: int
    free_gpu_bytes: int


def tensor_bytes(value: Any) -> int:
    """Sum tensor bytes recursively; non-tensor leaves contribute zero."""
    if isinstance(value, torch.Tensor):
        return value.element_size() * value.numel()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def estimate_capture_gpu_bytes(working_set_bytes: int) -> int:
    """Estimate capture residency plus metadata and safety margin.

    The working set includes static buffers and warmup-measured activations;
    without a measurement it covers static buffers only.
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
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # Do not block capture when host memory cannot be measured.
    return 1 << 62


def container_device(value: Any) -> torch.device | None:
    """Return the first tensor device found recursively, if any."""
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
    """Check GPU and host capacity for a capture.

    The GPU check uses the first input tensor's device or the current device.
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
