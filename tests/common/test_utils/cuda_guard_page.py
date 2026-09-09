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
"""Device allocations flush against an unmapped guard page.

:class:`GuardedArena` places the operand at the end of a mapped page so an
over-read faults. Run in a throwaway subprocess: the fault is sticky, and the
arena is never unmapped.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import torch

# CuTeDSL declares ``assumed_align=16`` on kernel pointers, so only a size that
# is a multiple of 16 can be both flush against the page and legally aligned.
_REQUIRED_ALIGN = 16


def _chk(ret: Any) -> Any:
    """Unwrap a cuda-python return of ``(status,)`` or ``(status, value)``."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    status, rest = ret[0], ret[1:]
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def vmm_supported() -> bool:
    """Whether the device supports the virtual memory API the guard page needs."""
    return bool(
        _chk(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                torch.cuda.current_device(),
            )
        )
    )


class _CudaArrayInterface:
    """Minimal ``__cuda_array_interface__`` holder so torch can adopt a pointer."""

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


class GuardedArena:
    """A mapped page whose successor page is reserved but unmapped.

    Holds one operand: only one allocation can end at the guard page.
    """

    def __init__(self) -> None:
        device = torch.cuda.current_device()

        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device

        self._granularity = _chk(
            cuda.cuMemGetAllocationGranularity(
                prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            )
        )
        # Reserve two granules, map only the first: the second is the guard.
        self._base = int(_chk(cuda.cuMemAddressReserve(2 * self._granularity, self._granularity, 0, 0)))
        handle = _chk(cuda.cuMemCreate(self._granularity, prop, 0))
        _chk(cuda.cuMemMap(self._base, self._granularity, 0, handle, 0))

        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _chk(cuda.cuMemSetAccess(self._base, self._granularity, [access], 1))

        self._guard = self._base + self._granularity
        self._taken = False

    def tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Return a zeroed tensor whose last byte is the last mapped byte."""
        if self._taken:
            raise RuntimeError("this arena already holds an operand; build another")

        numel = 1
        for extent in shape:
            numel *= extent
        nbytes = numel * torch.empty((), dtype=dtype).element_size()

        if nbytes % _REQUIRED_ALIGN:
            raise ValueError(
                f"{nbytes} bytes cannot be both {_REQUIRED_ALIGN}-byte aligned and flush against the "
                f"guard page; pick a shape whose byte count is a multiple of {_REQUIRED_ALIGN} "
                f"(shape={shape}, dtype={dtype})"
            )
        if nbytes > self._granularity:
            raise ValueError(f"{nbytes} bytes exceeds the {self._granularity}-byte page")

        ptr = self._guard - nbytes
        # bfloat16 has no interface typestr; adopt as int16 and reinterpret.
        if dtype in (torch.bfloat16, torch.float16):
            flat = torch.as_tensor(_CudaArrayInterface(ptr, (numel,), "<i2"), device="cuda").view(dtype)
        elif dtype == torch.int32:
            flat = torch.as_tensor(_CudaArrayInterface(ptr, (numel,), "<i4"), device="cuda")
        else:
            raise ValueError(f"unsupported guarded dtype {dtype}")

        if flat.data_ptr() != ptr:
            raise RuntimeError(f"torch adopted {hex(flat.data_ptr())}, expected {hex(ptr)}")

        self._taken = True
        out = flat.view(shape)
        out.zero_()
        return out

    def describe(self, t: torch.Tensor) -> str:
        """One line of provenance for a failure report."""
        start = t.data_ptr()
        end = start + t.numel() * t.element_size()
        placement = "ends at guard" if end == self._guard else f"ENDS {self._guard - end} BYTES SHORT"
        return f"[{hex(start)}, {hex(end)}) {placement} {hex(self._guard)}"
