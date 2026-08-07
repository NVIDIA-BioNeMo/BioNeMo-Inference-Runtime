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
"""Shared runtime helpers for precompiled CuTeDSL kernel libraries.

The private ``_cutedsl_kernels`` extension exposes one submodule per kernel
family.  Family-specific adapters turn those launchers into callables matching
the corresponding CuTeDSL-compiled function.  This lets an existing
``_compiled_cache`` contain either implementation without changing its caller.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, MutableMapping
from types import ModuleType
from typing import Any, cast

import torch

_KERNEL_LIBRARY_MODULE = "tensorrt_bionemo.libs._cutedsl_kernels"
_kernel_library: ModuleType | None = None


class CuTeDSLKernelLibraryError(RuntimeError):
    """Base error for precompiled CuTeDSL kernel-library failures."""


class CuTeDSLKernelLibraryUnavailable(CuTeDSLKernelLibraryError):
    """The extension or requested kernel-family submodule is unavailable."""


class CuTeDSLKernelVariantUnavailable(CuTeDSLKernelLibraryError):
    """The extension cannot directly launch the requested kernel variant."""


class CuTeDSLKernelLibraryExecutable:
    """Marker base for callables backed by ``_cutedsl_kernels``.

    Unlike TVM-FFI executables, these callables pass the current CUDA stream
    directly to the C++ launcher.
    """


def _load_kernel_library() -> ModuleType:
    global _kernel_library
    if _kernel_library is not None:
        return _kernel_library
    try:
        module = importlib.import_module(_KERNEL_LIBRARY_MODULE)
    except ImportError as error:
        raise CuTeDSLKernelLibraryUnavailable(f"Cannot import {_KERNEL_LIBRARY_MODULE!r}") from error
    _kernel_library = module
    return module


def populate_compiled_cache_from_library[ExecutableT](
    cache: MutableMapping[tuple, Any],
    key: tuple,
    family: str,
    factory: Callable[[ModuleType, Any], ExecutableT],
) -> ExecutableT:
    """Create and cache a shared-library executable for ``family``.

    ``factory`` performs only family-specific config selection and argument
    packing. The extension lookup and cache mutation stay common to all
    CuTeDSL-backed operators.
    """
    cached = cache.get(key)
    if cached is not None:
        return cast(ExecutableT, cached)

    library = _load_kernel_library()
    family_module = getattr(library, family, None)
    if family_module is None:
        raise CuTeDSLKernelLibraryUnavailable(f"{_KERNEL_LIBRARY_MODULE!r} has no {family!r} kernel family")

    executable = factory(library, family_module)
    cache[key] = executable
    return executable


def launch_compiled_kernel(executable: Any, *args: Any) -> Any:
    """Launch a cached CUBIN executable or a CuTeDSL TVM-FFI executable."""
    if isinstance(executable, CuTeDSLKernelLibraryExecutable):
        return executable(*args)

    # Keep TVM-FFI optional for source-free distributions. It is imported only
    # when the cache entry came from the development-time compile path.
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        return executable(*args)


def tensor_s1_d0(library: ModuleType, tensor: torch.Tensor) -> Any:
    """Create a CuTe ``s1_d0`` view from a contiguous rank-1 tensor."""
    if tensor.ndim != 1 or tensor.stride(0) != 1:
        raise ValueError("s1_d0 tensor must be contiguous and rank 1")
    # get_device() yields the CUDA ordinal, or -1 to match the library's
    # UNKNOWN_DEVICE for a tensor that is not on a GPU.
    return library.Tensor1View(tensor.data_ptr(), (tensor.shape[0],), (), tensor.get_device())


def tensor_s3_d2(library: ModuleType, tensor: torch.Tensor) -> Any:
    """Create a CuTe ``s3_d2`` view using three extents and two strides."""
    if tensor.ndim < 3:
        raise ValueError("s3_d2 tensor must have at least 3 dimensions")
    return library.Tensor3View(
        tensor.data_ptr(),
        tuple(tensor.shape[:3]),
        tuple(tensor.stride()[:2]),
        tensor.get_device(),
    )


def tensor_s4_d3(library: ModuleType, tensor: torch.Tensor) -> Any:
    """Create a CuTe ``s4_d3`` view using four extents and three strides."""
    if tensor.ndim < 4:
        raise ValueError("s4_d3 tensor must have at least 4 dimensions")
    return library.Tensor4View(
        tensor.data_ptr(),
        tuple(tensor.shape[:4]),
        tuple(tensor.stride()[:3]),
        tensor.get_device(),
    )
