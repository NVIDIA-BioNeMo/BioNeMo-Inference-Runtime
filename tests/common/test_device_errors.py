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
"""Classification of unrecoverable CUDA errors."""

import pytest

from bionemo_ir.utils import is_device_fatal

# Messages CUDA reports for sticky faults, as they reach Python.
FATAL_MESSAGES = [
    "CUDA error: an illegal memory access was encountered",
    "cuLaunchKernel failed: <CUresult.CUDA_ERROR_ILLEGAL_ADDRESS: 700>",
    "CUDA error: misaligned address",
    "CUDA error: unspecified launch failure",
    "CUDA error: device-side assert triggered",
    "CUDA error: uncorrectable ECC error encountered",
    "Triton Error [CUDA]: an illegal memory access was encountered",
    "cuLaunchKernel failed: <CUresult.CUDA_ERROR_MISALIGNED_ADDRESS: 716>",
    "cuLaunchKernel failed: <CUresult.CUDA_ERROR_LAUNCH_FAILED: 719>",
]

RECOVERABLE_MESSAGES = [
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "size of tensor a (768) must match tensor b (128)",
    "CUDA error: operation not permitted when stream is capturing",
    "Expected all tensors to be on the same device",
    "forced capture failure",
]


@pytest.mark.parametrize("message", FATAL_MESSAGES)
def test_sticky_cuda_errors_are_fatal(message: str) -> None:
    assert is_device_fatal(RuntimeError(message))


@pytest.mark.parametrize("message", RECOVERABLE_MESSAGES)
def test_recoverable_errors_are_not_fatal(message: str) -> None:
    assert not is_device_fatal(RuntimeError(message))


def test_out_of_memory_stays_recoverable() -> None:
    """OOM must keep reverting to eager: the context is still usable."""
    assert not is_device_fatal(torch_oom())


def torch_oom() -> Exception:
    import torch

    return torch.cuda.OutOfMemoryError("CUDA out of memory.")
