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
"""Outer-product-mean public operator selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from bionemo_ir.dsl_kernels.triton.dense_outer_product import dense_outer_product
from bionemo_ir.utils import get_sm_version

from ._config import _KERNEL_C, _KERNEL_D, _SUPPORTED_CZ

if TYPE_CHECKING:
    from .cutedsl import OuterProductMeanCuTe

# CuTe kernels written for each SM's own instructions. Elsewhere CuTe would run
# the Ampere kernel, so the Triton kernel goes first.
_CUTE_NATIVE = {
    80: (torch.float16, torch.bfloat16),
    86: (torch.float16, torch.bfloat16),
    89: (torch.float16, torch.bfloat16),
    90: (torch.bfloat16,),
}
# Triton's BF16 dot needs SM80.
_TRITON_MIN_SM = 80
_TRITON_DTYPES = (torch.float16, torch.bfloat16)


def _invoke_vanilla_opm(
    a: torch.Tensor,
    b: torch.Tensor,
    num_mask: torch.Tensor,
    W_o: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_before: bool = True,
) -> torch.Tensor:
    """PyTorch OPM reference with fp32 accumulation."""
    out_dtype = a.dtype
    B, S, I, C = a.shape
    J = b.shape[2]
    z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
    z = z.reshape(B, I, J, -1)
    nm = num_mask.to(torch.float32).reshape(B, I, J, 1)
    if norm_before:
        z = z / nm
    out = torch.nn.functional.linear(z, W_o.float(), bias.float() if bias is not None else None)
    if not norm_before:
        out = out / nm
    return out.to(out_dtype)


def _invoke_triton_opm(
    a: torch.Tensor,
    b: torch.Tensor,
    num_mask: torch.Tensor,
    W_o: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_before: bool = True,
) -> torch.Tensor:
    """Run the Triton backend."""
    return dense_outer_product(a, b, num_mask, W_o, bias, norm_before)


_OPM_CUTE: OuterProductMeanCuTe | None = None


def _get_cute_opm() -> OuterProductMeanCuTe:
    """Return the process-wide CuTe backend instance."""
    global _OPM_CUTE
    if _OPM_CUTE is None:
        # cutedsl imports this module for its PyTorch fallback.
        from .cutedsl import OuterProductMeanCuTe

        _OPM_CUTE = OuterProductMeanCuTe()
    return _OPM_CUTE


def _invoke_cute_opm(
    a: torch.Tensor,
    b: torch.Tensor,
    num_mask: torch.Tensor,
    W_o: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_before: bool = True,
) -> torch.Tensor:
    """Run the source-or-CUBIN CuTe backend."""
    return _get_cute_opm()(a, b, num_mask, W_o, bias, norm_before)


def get_outer_product_mean_op(dtype: torch.dtype, C: int, D: int, C_z: int) -> Callable:
    """Return the best backend for one dtype, shape, and device.

    CuTe runs where it has a native kernel for the SM and dtype; the Triton
    kernel serves every other SM80+ call, and everything else takes the
    PyTorch fallback.
    """
    if (C, D) != (_KERNEL_C, _KERNEL_D) or C_z not in _SUPPORTED_CZ:
        return _invoke_vanilla_opm

    sm = get_sm_version()
    if dtype in _CUTE_NATIVE.get(sm, ()):
        return _invoke_cute_opm
    if sm >= _TRITON_MIN_SM and dtype in _TRITON_DTYPES:
        return _invoke_triton_opm
    return _invoke_vanilla_opm
