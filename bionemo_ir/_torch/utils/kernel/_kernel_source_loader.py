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
"""Import a kernel family's optional ``_source`` adapter, if present.

Source-free distributions omit these adapters, so every caller must tolerate
:class:`ImportError` and fall back to the packaged CUBIN path.
:class:`KernelSourceUnavailable` subclasses it, so a caller that already
catches :class:`ImportError` keeps working while anything that reaches the
exception directly gets a message naming the build rather than a bare
``ModuleNotFoundError`` for a private module the reader cannot see.

The module name is assembled at run time rather than written as a literal
``from ._source import ...``, so that a missing adapter surfaces as a
recoverable error at kernel-resolution time rather than an import error at
package-import time. For the same reason, source-path code converts dtypes to
CuTeDSL types with :func:`cute_numeric` rather than importing the compiler.

Every adapter imports the CuTeDSL compiler at module level, so a successful
load also means the compiler is installed. Callers rely on that: without the
compiler the load fails and they fall back to CUBINs, and capability checks
such as ``ships()`` treat a loaded adapter as able to compile.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import cutlass

type CutlassNumeric = type[cutlass.Numeric]

__all__ = ["CutlassNumeric", "KernelSourceUnavailable", "cute_numeric", "load_source_module"]

_SOURCE_SUFFIX = "_" + "source"


class KernelSourceUnavailable(ImportError):
    """A kernel family's source adapter is not present in this build."""


def load_source_module(package: str) -> ModuleType:
    """Return the package's source adapter, or raise :class:`KernelSourceUnavailable`."""
    module = f"{package}.{_SOURCE_SUFFIX}"
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        # Only the adapter's own absence is a source-free build. A
        # ModuleNotFoundError raised from *inside* a present adapter means its
        # own imports are broken; both still reach the callers' `except
        # ImportError` and fall back to CUBINs, but relabelling the second as a
        # stripped build would send a reader looking for the wrong cause.
        if error.name != module:
            raise
        raise KernelSourceUnavailable(
            f"{package} kernel sources are not available in this build; use the packaged CUBIN path",
            name=module,
        ) from error


def cute_numeric(dtype: torch.dtype) -> CutlassNumeric:
    """Return the CuTeDSL numeric type of a float16, bfloat16 or float32 ``dtype``.

    Imports the CuTeDSL compiler, so only source-path code calls it; the CUBIN
    path carries the ``torch.dtype``.

    Args:
        dtype: The PyTorch dtype to convert.

    Raises:
        TypeError: ``dtype`` has no CuTeDSL counterpart here.
    """
    import cutlass

    numeric = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    try:
        return numeric[dtype]
    except KeyError:
        raise TypeError(f"no CuTeDSL numeric type for {dtype}") from None
