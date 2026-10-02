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
"""CUBIN-backed executable adapter for the SM90 TriMul KF K3."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_s1_d0,
)


class TrimulKFK3CubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        C: int,
        D: int,
        kernel_variant: str,
        residual: bool,
    ) -> None:
        try:
            config = launcher.make_kernel_config(
                target_sm, launcher.DType.BFLOAT16, C, D, int(kernel_variant.removeprefix("K3_")), residual
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No trimul KF K3 CUBIN for SM{target_sm}, C={C}, D={D}, {kernel_variant}, residual={residual}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._reads_stats = kernel_variant == "K3_1"
        self._residual = residual

    def __call__(
        self,
        prod: torch.Tensor,
        x: torch.Tensor,
        w_out: torch.Tensor,
        w_gate_out: torch.Tensor,
        vec_out: torch.Tensor,
        stats: torch.Tensor | None,
        seqlen: torch.Tensor | None,
        output: torch.Tensor,
        rows: int,
        n: int,
        nb: int,
        eps: float,
    ) -> None:
        if (stats is not None) != self._reads_stats or (seqlen is not None) != self._residual:
            raise CuTeDSLKernelVariantUnavailable("trimul KF K3 CUBIN operands do not match its variant")
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.prod = tensor_s1_d0(library, prod)
        params.x = tensor_s1_d0(library, x)
        params.w_out = tensor_s1_d0(library, w_out)
        params.w_gate_out = tensor_s1_d0(library, w_gate_out)
        params.vec_out = tensor_s1_d0(library, vec_out)
        if stats is not None:
            params.stats = tensor_s1_d0(library, stats)
        if seqlen is not None:
            params.seqlen = tensor_s1_d0(library, seqlen)
        params.output = tensor_s1_d0(library, output)
        params.rows = rows
        params.n = n
        params.nb = nb
        params.eps = eps
        params.stream = current_stream_handle(x)
        self._launcher.launch(self._config, params)
