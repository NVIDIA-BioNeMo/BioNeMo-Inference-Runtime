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

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from bionemo_ir.dsl_kernels.triton.dense_outer_product import dense_outer_product
from bionemo_ir.utils import get_sm_version

from ._config import _KERNEL_C, _KERNEL_D, _SM100_FAMILY, _SUPPORTED_CZ, sm100_a_chunks

if TYPE_CHECKING:
    from .cutedsl import OuterProductMeanCuTe

# CuTe kernels written for each SM's own instructions. Elsewhere CuTe would run
# the Ampere kernel, so the Triton kernel goes first.
_CUTE_NATIVE = {
    80: (torch.float16, torch.bfloat16),
    86: (torch.float16, torch.bfloat16),
    89: (torch.float16, torch.bfloat16),
    90: (torch.bfloat16,),
    100: (torch.float16, torch.bfloat16),
    103: (torch.float16, torch.bfloat16),
}
# Deepest MSA the SM100 kernel serves; past it the Triton kernel is faster or unmeasured. On B200, with
# sm100.json's cc = 8 streaming tiles and the prologue's sequence-last a, timed interleaved with Triton at either
# C_z: 1.06-1.24x up to S = 16384 from a pair side sqrt(I * J) of 320, and 1.01-1.15x up to S = 8192 below it,
# where N = 256 falls behind at S = 12288. SM103 keeps the depths per C_z it was measured at before those tiles;
# this has not been remeasured on B300.
_SM100_MAX_S_BY_SIDE = ((320, 16384), (0, 8192))
_SM103_MAX_S = {128: 1536, 256: 1024}
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
    """Run the source-or-CUBIN CuTe backend, or Triton for a call the SM100 kernel does not serve.

    A layer picks its op once, at construction, so S is checked per call.
    """
    I = a.shape[2] if a.dim() == 4 else num_mask.shape[1]
    if not _cute_serves_call(get_sm_version(), b.shape[1], W_o.shape[0], I, b.shape[2]):
        return _invoke_triton_opm(a, b, num_mask, W_o, bias, norm_before)
    return _get_cute_opm()(a, b, num_mask, W_o, bias, norm_before)


def _cute_runs_on(sm: int, dtype: torch.dtype) -> bool:
    """Whether the CuTe backend serves ``dtype`` on ``sm``."""
    return dtype in _CUTE_NATIVE.get(sm, ())


def _cute_serves_call(sm: int, S: int, C_z: int, I: int, J: int) -> bool:
    """Whether the CuTe backend serves one ``I x J`` call at MSA depth ``S`` (the SM100 kernel only up to its cap)."""
    if sm not in _SM100_FAMILY:
        return True
    if C_z not in _SUPPORTED_CZ:
        return False
    if sm == 103:
        return S <= _SM103_MAX_S[C_z]
    side = math.sqrt(float(I) * float(J))
    return S <= next(depth for min_side, depth in _SM100_MAX_S_BY_SIDE if side >= min_side)


def outer_product_mean_a_chunk(
    S: int, I: int, J: int, C_z: int, norm_before: bool, has_bias: bool, dtype: torch.dtype
) -> int | None:
    """Chunk ``cc`` when the selected backend takes ``a`` sequence-last (SM100/SM103 CuTe), else ``None``.

    A producer of ``a`` can then write ``[B, 32 / cc, ceil(I / T), cc, T, S]`` (``T = 128 / cc``, row pitch a
    multiple of 64) directly and hand it to the op from :func:`get_outer_product_mean_op`; every other
    backend takes ``a`` as ``[B, S, I, 32]``.
    """
    sm = get_sm_version()
    if not (_cute_runs_on(sm, dtype) and _cute_serves_call(sm, S, C_z, I, J)):
        return None
    return _get_cute_opm().a_chunk(S, I, J, C_z, norm_before, has_bias, dtype)


def outer_product_mean_a_chunks(C_z: int, dtype: torch.dtype) -> tuple[int, ...]:
    """Every chunk :func:`outer_product_mean_a_chunk` can return on this device for ``C_z`` and ``dtype``, at any
    shape, so a producer of ``a`` can prepare each sequence-last layout before its first call."""
    sm = get_sm_version()
    if sm not in _SM100_FAMILY or not _cute_runs_on(sm, dtype) or C_z not in _SUPPORTED_CZ:
        return ()
    return sm100_a_chunks(sm, C_z)


def get_outer_product_mean_op(dtype: torch.dtype, C: int, D: int, C_z: int) -> Callable:
    """Return the best backend for one dtype, shape, and device.

    CuTe runs where it has a native kernel for the SM and dtype; the Triton
    kernel serves every other SM80+ call, and everything else takes the
    PyTorch fallback.
    """
    if (C, D) != (_KERNEL_C, _KERNEL_D) or C_z not in _SUPPORTED_CZ:
        return _invoke_vanilla_opm

    sm = get_sm_version()
    if _cute_runs_on(sm, dtype):
        return _invoke_cute_opm
    if sm >= _TRITON_MIN_SM and dtype in _TRITON_DTYPES:
        return _invoke_triton_opm
    return _invoke_vanilla_opm
