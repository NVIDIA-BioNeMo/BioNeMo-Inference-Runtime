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
"""PyTorch fallback for the fused outer-product-mean op."""

from __future__ import annotations

from collections.abc import Callable

import torch

from bionemo_ir.utils import get_sm_version

from ._config import _KERNEL_C, _KERNEL_CZ, _KERNEL_D

_opm_cute_instance = None


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


def get_outer_product_mean_op(dtype: torch.dtype, C: int, D: int, C_z: int) -> Callable:
    """Return the CuTeDSL backend when a matching payload exists."""
    if (C, D, C_z) != (_KERNEL_C, _KERNEL_D, _KERNEL_CZ):
        return _invoke_vanilla_opm

    from .cutedsl import _DTYPE_STR, _SUPPORTED_SM, OuterProductMeanCuTe

    sm = get_sm_version()
    if sm in _SUPPORTED_SM and dtype in _DTYPE_STR:
        global _opm_cute_instance
        if _opm_cute_instance is None:
            _opm_cute_instance = OuterProductMeanCuTe()
        return _opm_cute_instance
    return _invoke_vanilla_opm
