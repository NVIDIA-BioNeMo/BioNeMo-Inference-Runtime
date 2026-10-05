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
"""CUBIN-backed executable adapter for the SM90 TriMul KF K1."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_flat,
    tensor_s1_d0,
)


class TrimulKFK1CubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        C: int,
        D: int,
        kernel_variant: str,
        ab_layout: str = "dense",
    ) -> None:
        try:
            config = launcher.make_kernel_config(
                target_sm,
                launcher.DType.BFLOAT16,
                C,
                D,
                int(kernel_variant.removeprefix("K1_")),
                padded=ab_layout == "padded",
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No trimul KF K1 CUBIN for SM{target_sm}, C={C}, D={D}, {kernel_variant}, {ab_layout} a/b"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._kernel_variant = kernel_variant

    def __call__(
        self,
        x: torch.Tensor,
        seqlen: torch.Tensor,
        w_in: torch.Tensor,
        w_gate_in: torch.Tensor | None,
        vec_in: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        stats: torch.Tensor | None,
        rows: int,
        n: int,
        nb: int,
        eps: float,
    ) -> None:
        if (w_gate_in is not None) != (self._kernel_variant == "K1_0") or (stats is not None) != (
            self._kernel_variant == "K1_2"
        ):
            raise CuTeDSLKernelVariantUnavailable(f"trimul KF K1 CUBIN operands do not match {self._kernel_variant}")
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.x = tensor_flat(library, x)
        params.seqlen = tensor_s1_d0(library, seqlen)
        params.w_in = tensor_s1_d0(library, w_in)
        if w_gate_in is not None:
            params.w_gate_in = tensor_s1_d0(library, w_gate_in)
        params.vec_in = tensor_s1_d0(library, vec_in)
        params.a = tensor_flat(library, a)
        params.b = tensor_flat(library, b)
        if stats is not None:
            params.stats = tensor_flat(library, stats)
        params.rows = rows
        params.n = n
        params.nb = nb
        params.eps = eps
        params.stream = current_stream_handle(x)
        self._launcher.launch(self._config, params)
