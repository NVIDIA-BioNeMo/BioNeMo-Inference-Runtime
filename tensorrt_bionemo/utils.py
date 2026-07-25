# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared utilities used across TensorRT-BioNeMo."""

from functools import lru_cache

import torch

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
_TORCH_DTYPE_TO_STR = {
    value: key
    for key, value in _STR_TO_TORCH_DTYPE.items()
}


def str_dtype_to_torch(dtype: str) -> torch.dtype:
    """Convert a supported dtype name to ``torch.dtype``."""
    result = _STR_TO_TORCH_DTYPE.get(dtype)
    assert result is not None, f"Unsupported dtype: {dtype}"
    return result


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a supported ``torch.dtype`` to its configuration name."""
    return _TORCH_DTYPE_TO_STR[dtype]


@lru_cache(maxsize=1)
def get_sm_version() -> int:
    """Return the compute capability as an integer, for example SM90 -> 90."""
    properties = torch.cuda.get_device_properties(0)
    return properties.major * 10 + properties.minor


__all__ = ["get_sm_version", "str_dtype_to_torch", "torch_dtype_to_str"]
