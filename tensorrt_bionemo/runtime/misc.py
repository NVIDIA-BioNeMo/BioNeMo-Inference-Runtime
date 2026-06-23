# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import numpy as np
import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


@torch.compiler.disable
def get_closest_n(s):
    return 2**int(np.ceil(np.log2(s)))


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


def query_sm_count(device: Optional[int] = None, default: int = 132) -> int:
    """Query the multiprocessor (SM) count of a CUDA device.

    Falls back to ``default`` if the query fails for any reason (e.g. no CUDA
    device available or a driver error).

    Args:
        device: CUDA device ordinal. Defaults to the current device.
        default: Value returned if the SM count cannot be queried.
    """
    try:
        if device is None:
            device = torch.cuda.current_device()
        err, sm_count = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, device)
        if err != cudart.cudaError_t.cudaSuccess:
            return default
        return sm_count
    except Exception:
        return default


def ensure_contiguous(func):
    """
    Decorator to ensure all torch.Tensor inputs are contiguous.
    Non-tensor inputs are passed through unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Process positional args
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                new_args.append(arg.contiguous())
            else:
                new_args.append(arg)

        # Process keyword args
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and not value.is_contiguous():
                new_kwargs[key] = value.contiguous()
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


@contextmanager
def dtype_context(
    expected_dtype: torch.dtype,
    original_dtype: Optional[torch.dtype] = None,
    skip_keys: Optional[Tuple[str]] = None,
):
    """
    Context manager to cast inputs to `expected_dtype` and restore outputs to `original_dtype`.

    Args:
        expected_dtype: Target dtype for inputs (e.g., torch.float32).
        original_dtype: If None, inferred from the first input tensor.
        skip_keys: Tuple of argument names to skip dtype conversion.
        verbose: Print conversion details.
    """
    skip_keys = skip_keys or ()

    def cast_tensor(x: Any, target_dtype: torch.dtype) -> Any:
        if isinstance(x, torch.Tensor) and x.dtype != target_dtype:
            return x.to(target_dtype)
        return x

    # Store original input tensors for restoration (if needed)
    input_tensors = {}

    def wrap_inputs(*args, **kwargs):
        nonlocal original_dtype
        # Infer original_dtype from the first floating-point tensor
        if original_dtype is None:
            for x in (*args, *kwargs.values()):
                if isinstance(x, torch.Tensor):
                    original_dtype = x.dtype
                    break

        # Cast inputs to expected_dtype (skip keys in skip_keys)
        new_args = []
        for i, arg in enumerate(args):
            input_tensors[f"arg_{i}"] = arg
            new_args.append(cast_tensor(arg, expected_dtype))

        new_kwargs = {}
        for k, v in kwargs.items():
            input_tensors[k] = v
            if k in skip_keys:
                new_kwargs[k] = v
            else:
                new_kwargs[k] = cast_tensor(v, expected_dtype)

        return new_args, new_kwargs

    def wrap_outputs(output: Any) -> Any:
        if isinstance(output, torch.Tensor) and output.dtype != original_dtype:
            return output.to(original_dtype)
        elif isinstance(output, (tuple, list)):
            return type(output)(wrap_outputs(x) for x in output)
        elif isinstance(output, dict):
            return {k: wrap_outputs(v) for k, v in output.items()}
        return output

    # Handle function calls inside the context
    def wrapped_func(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            new_args, new_kwargs = wrap_inputs(*args, **kwargs)
            output = func(*new_args, **new_kwargs)
            return wrap_outputs(output)

        return inner

    yield wrapped_func
