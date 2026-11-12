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

"""Small shared utilities used across the framework."""

import os
from functools import cache

import torch

# Opt-in device-side input validation. Each check reads a device tensor back to
# the host, which drains the launch queue and is illegal under CUDA-graph
# capture, so production leaves it off.
DEBUG_ASSERTS: bool = os.getenv("BIOIR_DEBUG_ASSERTS", "0") == "1"

_STR_TO_TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "bool": torch.bool,
    "fp8": torch.float8_e4m3fn,
}
_TORCH_DTYPE_TO_STR = {value: key for key, value in _STR_TO_TORCH_DTYPE.items()}


def str_dtype_to_torch(dtype: str) -> torch.dtype:
    """Convert a supported dtype name to ``torch.dtype``."""
    result = _STR_TO_TORCH_DTYPE.get(dtype)
    assert result is not None, f"Unsupported dtype: {dtype}"
    return result


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a supported ``torch.dtype`` to its configuration name."""
    return _TORCH_DTYPE_TO_STR[dtype]


def get_sm_version(device: torch.device | int | None = None) -> int:
    """Return a CUDA device's compute capability, for example SM90 -> 90.

    Args:
        device: A CUDA device index, a CUDA ``torch.device``, or ``None``.
            ``None`` and a ``torch.device`` without an index use the current
            CUDA device.
    """
    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
    else:
        device_index = device
    return _get_sm_version(device_index)


@cache
def _get_sm_version(device_index: int) -> int:
    """Return one device's cached compute capability."""
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major * 10 + properties.minor


# CUDA marks these errors sticky: every later call on the context returns the same
# error, so no amount of retrying, falling back to eager, or emptying the cache can
# recover -- only tearing the process down can. driver_types.h documents each code
# as leaving the process in an inconsistent state, except uncorrectable ECC, which
# is fatal in practice. The runtime (cudaError_t) and driver (CUresult) enums share
# these values.
_STICKY_CUDA_ERROR_CODES = frozenset(
    {
        214,  # ECC uncorrectable
        226,  # contained
        700,  # illegal address
        702,  # launch timeout
        710,  # assert
        714,  # hardware stack error
        715,  # illegal instruction
        716,  # misaligned address
        717,  # invalid address space
        718,  # invalid program counter
        719,  # launch failure
        721,  # tensor memory leak
        810,  # MPS client terminated
        911,  # external device
    }
)

# Errors without a code, such as Triton's or a cuLaunchKernel failure's, are matched
# by message for the common sticky faults. Markers are matched against the message
# with underscores and hyphens flattened to spaces, so one entry covers both the
# runtime spelling ("an illegal memory access was encountered") and the driver enum
# ("CUDA_ERROR_ILLEGAL_ADDRESS") that a cuLaunchKernel failure reports. Each marker
# must be specific: PyTorch appends "Compile with `TORCH_USE_CUDA_DSA` to enable
# device-side assertions." to every CUDA error, and a re-raise that keeps only the
# message carries that hint too, so "device side assert" would match recoverable
# errors.
_STICKY_CUDA_ERROR_MARKERS = (
    "illegal memory access",
    "illegal address",
    "illegal instruction",
    "misaligned address",
    "invalid address space",
    "unspecified launch failure",
    "cuda error launch failed",
    "device side assert triggered",
    "cuda error assert",
    "uncorrectable ecc",
    "ecc uncorrectable",
    "hardware stack error",
)


def is_device_fatal(exc: BaseException) -> bool:
    """Return whether an exception means the CUDA context is unrecoverable.

    A sticky CUDA error poisons the context, so continuing masks the real fault and
    reports it at unrelated call sites later. Callers should propagate instead of
    retrying, and must not touch the device (``empty_cache`` included) first.

    PyTorch raises CUDA errors as ``torch.AcceleratorError`` with the error code
    attached, which decides; other errors are classified by their message.

    Args:
        exc: The exception to classify.
    """
    error_code = getattr(exc, "error_code", None)
    if isinstance(exc, torch.AcceleratorError) and error_code is not None:
        return error_code in _STICKY_CUDA_ERROR_CODES
    message = str(exc).lower().replace("_", " ").replace("-", " ")
    return any(marker in message for marker in _STICKY_CUDA_ERROR_MARKERS)


__all__ = ["get_sm_version", "is_device_fatal", "str_dtype_to_torch", "torch_dtype_to_str"]
