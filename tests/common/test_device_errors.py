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
import torch

from bionemo_ir.utils import is_device_fatal

# PyTorch (c10/cuda/CUDAException) appends this hint to every CUDA error it raises.
TORCH_CUDA_ERROR_HINT = (
    "\nCUDA kernel errors might be asynchronously reported at some other API call, "
    "so the stacktrace below might be incorrect.\n"
    "For debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"
    "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
)

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


def test_torch_hint_alone_is_not_fatal() -> None:
    """The DSA hint on every torch CUDA error must not make the error fatal."""
    assert not is_device_fatal(RuntimeError(TORCH_CUDA_ERROR_HINT))


@pytest.mark.parametrize(
    ("error_enum", "cause", "fatal"),
    [
        ("cudaErrorStreamCaptureInvalidated", "operation failed due to a previous error during capture", False),
        ("cudaErrorStreamCaptureUnsupported", "operation not permitted when stream is capturing", False),
        ("cudaErrorAssert", "device-side assert triggered", True),
        ("cudaErrorIllegalAddress", "an illegal memory access was encountered", True),
    ],
)
def test_torch_cuda_errors_classify_by_cause(error_enum: str, cause: str, fatal: bool) -> None:
    """Full torch CUDA error messages classify by the error, not the trailing hint."""
    message = (
        f"CUDA error: {cause}\n"
        f"Search for `{error_enum}' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html "
        f"for more information.{TORCH_CUDA_ERROR_HINT}"
    )
    assert is_device_fatal(RuntimeError(message)) is fatal


@pytest.mark.parametrize(
    ("error_code", "fatal"),
    [
        pytest.param(700, True, id="cudaErrorIllegalAddress"),
        pytest.param(702, True, id="cudaErrorLaunchTimeout"),
        pytest.param(717, True, id="cudaErrorInvalidAddressSpace"),
        pytest.param(900, False, id="cudaErrorStreamCaptureUnsupported"),
        pytest.param(901, False, id="cudaErrorStreamCaptureInvalidated"),
    ],
)
def test_torch_cuda_errors_classify_by_code(error_code: int, fatal: bool) -> None:
    """PyTorch attaches the CUDA error code, which decides even when no marker names the fault."""
    exc = torch.AcceleratorError(f"CUDA error {error_code}{TORCH_CUDA_ERROR_HINT}")
    exc.error_code = error_code
    assert is_device_fatal(exc) is fatal


def test_out_of_memory_stays_recoverable() -> None:
    """OOM must keep reverting to eager: the context is still usable."""
    assert not is_device_fatal(torch_oom())


def torch_oom() -> Exception:
    return torch.cuda.OutOfMemoryError("CUDA out of memory.")
