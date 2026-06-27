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
"""Unit tests for the CUDA-graph memory estimator/gate (memory.py).

These run on CPU: the driver/host queries are monkeypatched so the gate logic
is tested deterministically without depending on actual free memory.
"""
import torch

from tensorrt_bionemo._torch.graph_optimization import memory as gc_mem


def test_tensor_bytes_nested():
    container = {
        "a": torch.zeros(4, dtype=torch.float32),        # 16 bytes
        "b": [torch.zeros(2, dtype=torch.int64),          # 16 bytes
              (torch.zeros(3, dtype=torch.float16),)],    # 6 bytes
        "c": "not a tensor",
        "d": None,
    }
    assert gc_mem.tensor_bytes(container) == 16 + 16 + 6


def test_tensor_bytes_passthrough_non_tensors():
    assert gc_mem.tensor_bytes(None) == 0
    assert gc_mem.tensor_bytes(7) == 0
    assert gc_mem.tensor_bytes("x") == 0


def test_estimate_includes_metadata_and_margin():
    ws = 100 * gc_mem.MB
    est = gc_mem.estimate_capture_gpu_bytes(ws)
    # working set + metadata + max(min_margin, 10%)
    expected = ws + gc_mem.GRAPH_METADATA_BYTES + max(
        gc_mem.MIN_GPU_MARGIN_BYTES, int(0.10 * ws))
    assert est == expected
    assert est > ws  # always strictly larger than the bare working set


def test_gate_ok_when_plenty_free(monkeypatch):
    monkeypatch.setattr(gc_mem, "gpu_free_bytes", lambda device=None: 100 * gc_mem.MB * 1000)
    monkeypatch.setattr(gc_mem, "host_available_bytes", lambda: 1 << 40)
    check = gc_mem.check_capacity_for_capture(10 * gc_mem.MB)
    assert check.ok and check.reason == "ok"


def test_gate_blocks_on_low_gpu(monkeypatch):
    monkeypatch.setattr(gc_mem, "gpu_free_bytes", lambda device=None: 1 * gc_mem.MB)
    monkeypatch.setattr(gc_mem, "host_available_bytes", lambda: 1 << 40)
    check = gc_mem.check_capacity_for_capture(500 * gc_mem.MB)
    assert not check.ok
    assert "GPU" in check.reason
    assert check.needed_gpu_bytes > check.free_gpu_bytes


def test_gate_blocks_on_low_host(monkeypatch):
    monkeypatch.setattr(gc_mem, "gpu_free_bytes", lambda device=None: 1 << 40)
    monkeypatch.setattr(gc_mem, "host_available_bytes", lambda: 1 * gc_mem.MB)
    check = gc_mem.check_capacity_for_capture(10 * gc_mem.MB)
    assert not check.ok
    assert "host" in check.reason
