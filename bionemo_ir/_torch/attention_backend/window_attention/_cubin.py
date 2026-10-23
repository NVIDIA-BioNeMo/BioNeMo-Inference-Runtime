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
"""CUBIN-backed executable adapter for the windowed attention kernels."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_s3_d2_static,
)


def _bias_view(library: ModuleType, bias: torch.Tensor) -> Any:
    """Describe the ``[B, T, H, n_queries, n_keys]`` bias by its five extents and four outer strides."""
    return library.Tensor5View(bias.data_ptr(), tuple(bias.shape), bias.stride()[:4], bias.get_device())


class WindowAttentionCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Adapt the source executable's call ABI to the C++ launcher.

    Both take ``(q, k, v, bias, out, softmax_scale_log2, softmax_scale)``: ``[S, N, H, D]`` row views with heads
    adjacent and a ``[B, T, H, n_queries, n_keys]`` bias, each with a unit-stride last axis. The launcher picks the
    image's kernel ABI, as the source build's config does.
    """

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        dtype_str: str,
        head_dim: int,
        n_queries: int,
        n_keys: int,
    ):
        dtype = launcher.DType.BFLOAT16 if dtype_str == "bf16" else launcher.DType.FLOAT16
        try:
            config = launcher.make_kernel_config(target_sm, head_dim, n_queries, n_keys, dtype)
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No window attention CUBIN for SM{target_sm}, {dtype_str}, D={head_dim}, windows {n_queries}x{n_keys}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        softmax_scale_log2: float,
        softmax_scale: float,
    ) -> None:
        if any(x.stride(-1) != 1 for x in (q, k, v, bias, out)):
            raise CuTeDSLKernelVariantUnavailable("window attention operands need a unit-stride last axis")
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.q = tensor_s3_d2_static(library, q)
        params.k = tensor_s3_d2_static(library, k)
        params.v = tensor_s3_d2_static(library, v)
        params.bias = _bias_view(library, bias)
        params.output = tensor_s3_d2_static(library, out)
        params.softmax_scale_log2 = softmax_scale_log2
        params.softmax_scale = softmax_scale
        params.stream = current_stream_handle(q)
        self._launcher.launch(self._config, params)
