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
"""CUBIN-backed executable adapter for the fused attention epilogue."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
)

# A tensor with the sizes and strides it takes as one kernel view, in the kernel's mode order.
type KernelOperand = tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]


def _operand(view: torch.Tensor) -> KernelOperand:
    """Describe a kernel view, whose mode 1 must be unit-stride."""
    if view.stride(1) != 1:
        raise CuTeDSLKernelVariantUnavailable("attention epilogue operands need a unit-stride inner mode")
    return view, tuple(view.shape), view.stride()


def _pair(library: ModuleType, operand: KernelOperand) -> Any:
    """Describe a ``(J, L, B*I)`` pair operand to the launcher as ``(B*I, J, L)``."""
    tensor, (columns, width, folded), (column_stride, _, row_stride) = operand
    return library.Tensor3View(
        tensor.data_ptr(), (folded, columns, width), (row_stride, column_stride), tensor.get_device()
    )


class AttnEpilogueCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Adapt the source executable's call ABI to the C++ launcher.

    Both take the kernel views ``(o, g, Wo, bias, destination, residual,
    output_gate)`` shaped ``(J, D, H, B*I)``, ``(J, H*D, B*I)``, ``(C, H*D)``,
    ``(C,)`` or ``None``, ``(J, C, B*I)`` twice and ``(J, C, B*I / mult)`` or
    ``None``; the launcher receives them reordered to ``(B*I, J, H, D)``,
    ``(B*I, J, H*D)``, ``(C, H*D)``, ``(C,)``, ``(B*I, J, C)`` and
    ``(B*I / mult, J, C)``. :meth:`launch` takes the same operands as sizes
    and strides, sparing the views on a call path that is otherwise host-bound.
    """

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        heads: int,
        head_dim: int,
        channels: int,
        has_bias: bool = False,
        has_output_gate: bool = False,
        rows: int = 0,
    ):
        try:
            # The launcher takes the image whose row anchor is nearest ``rows``.
            config = launcher.make_kernel_config(target_sm, heads, head_dim, channels, has_bias, has_output_gate, rows)
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No attention epilogue CUBIN for SM{target_sm}, heads={heads}, head_dim={head_dim}, "
                f"channels={channels}, has_bias={has_bias}, has_output_gate={has_output_gate}, rows={rows}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        destination: torch.Tensor,
        residual: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> None:
        if weight.stride(1) != 1:
            raise CuTeDSLKernelVariantUnavailable("attention epilogue operands need a unit-stride inner mode")
        self.launch(
            _operand(attention),
            _operand(gate),
            weight,
            bias,
            _operand(destination),
            _operand(residual),
            None if output_gate is None else _operand(output_gate),
        )

    def launch(
        self,
        attention: KernelOperand,
        gate: KernelOperand,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        destination: KernelOperand,
        residual: KernelOperand,
        output_gate: KernelOperand | None = None,
    ) -> None:
        """Launch on the current stream of the attention output's device.

        Args:
            attention: ``o`` as ``(J, D, H, B*I)``.
            gate: ``g`` as ``(J, H*D, B*I)``.
            weight: ``Wo`` as ``(C, H*D)`` with a unit inner stride.
            bias: The contiguous ``(C,)`` output bias, or ``None``.
            destination: The output as ``(J, C, B*I)``.
            residual: The residual as ``(J, C, B*I)``.
            output_gate: ``y`` as ``(J, C, B*I / mult)``, or ``None``.
        """
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        tensor, (columns, head_dim, heads, folded), (column_stride, _, head_stride, row_stride) = attention
        device = tensor.get_device()
        params.o = library.Tensor4View(
            tensor.data_ptr(), (folded, columns, heads, head_dim), (row_stride, column_stride, head_stride), device
        )
        params.g = _pair(library, gate)
        params.w = library.Tensor2View(
            weight.data_ptr(), (weight.shape[0], weight.shape[1]), (weight.stride(0),), weight.get_device()
        )
        if bias is not None:
            params.b = library.Tensor1View(bias.data_ptr(), (bias.shape[0],), (), bias.get_device())
        params.d = _pair(library, destination)
        params.z = _pair(library, residual)
        if output_gate is not None:
            params.y = _pair(library, output_gate)
        params.stream = current_stream_handle(tensor)
        self._launcher.launch(self._config, params)
