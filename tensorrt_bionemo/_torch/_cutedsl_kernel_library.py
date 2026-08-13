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
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch

_KERNEL_LIBRARY_MODULE = "tensorrt_bionemo.libs._cutedsl_kernels"
_KERNEL_LIBRARY_STEM = "_cutedsl_kernels"
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_KERNEL_LIBRARY_DIR = _PACKAGE_DIR / "libs"
_KERNEL_SOURCE_DIR = _PACKAGE_DIR / "dsl_kernels" / "cute"
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


def _extension_installed() -> bool:
    """Whether the compiled extension is present, without loading it.

    Named exactly as ``setup.py`` writes it. A file test rather than
    ``find_spec``, which imports the parent packages — this runs during
    ``tensorrt_bionemo``'s own import — and rather than an import, which would
    move any CUDA or driver problem into package import as a different failure.
    """
    return any((_KERNEL_LIBRARY_DIR / f"{_KERNEL_LIBRARY_STEM}{suffix}").is_file() for suffix in EXTENSION_SUFFIXES)


def _kernel_sources_installed() -> bool:
    """Whether any CuTeDSL kernel source survives to be compiled at runtime."""
    return any(path.name != "__init__.py" for path in _KERNEL_SOURCE_DIR.glob("*.py"))


def require_kernel_backend() -> None:
    """Refuse a build that can neither launch nor compile a CuTeDSL kernel.

    Backend selection picks on device and shape alone, so a build missing both
    paths installs cleanly and then dies on the first fused op, far from the
    cause. Raise the same error at import instead, where the message can name
    what is absent.

    Raises:
        CuTeDSLKernelLibraryUnavailable: Neither path is installed.
    """
    if _extension_installed() or _kernel_sources_installed():
        return
    raise CuTeDSLKernelLibraryUnavailable(
        "No CuTeDSL kernel backend is installed: this build has neither the "
        f"compiled {_KERNEL_LIBRARY_MODULE!r} extension nor kernel sources in "
        f"{_KERNEL_SOURCE_DIR}. Install a released wheel, or rebuild with "
        "TRTBNM_BUILD_CUTEDSL_KERNELS=1 from a checkout that carries the sources."
    )


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


def tensor_s2_d1(library: ModuleType, tensor: torch.Tensor, dynamic_stride_dim: int = 0) -> Any:
    """Create a CuTe ``s2_d1``/``s1_d1`` view from a rank-2 tensor.

    Both descriptors share this host view: they differ only in whether the
    compiled kernel left the inner extent dynamic, which the launcher resolves
    against its own spec rather than the caller's tensor.

    ``dynamic_stride_dim`` selects which dimension carries the dynamic stride;
    the other must be contiguous. Row-major operands use the default of 0.
    """
    if tensor.ndim != 2:
        raise ValueError("s2_d1 tensor must have rank 2")
    if dynamic_stride_dim not in (0, 1):
        raise ValueError("dynamic_stride_dim must be 0 or 1")
    contiguous_stride_dim = 1 - dynamic_stride_dim
    if tensor.stride(contiguous_stride_dim) != 1:
        raise ValueError(f"s2_d1 tensor stride for dimension {contiguous_stride_dim} must be 1")
    return library.Tensor2View(
        tensor.data_ptr(),
        tuple(tensor.shape),
        (tensor.stride(dynamic_stride_dim),),
        tensor.get_device(),
    )


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
