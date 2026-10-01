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
"""Operand layout, shape checks and backend selection for the fused attention epilogue."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

import torch

from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryError, launch_compiled_kernel

from ._config import OUTPUT_GATE_WIDTHS, STREAMED_WIDTHS, nearest_anchor, tuned_configs
from ._cubin import AttnEpilogueCubinExecutable, KernelOperand
from .cutedsl import AttnEpilogueCuTe

HEAD_DIMS = (32, 64, 128)
WIDTHS = (128, 256, 512)
CHANNELS = 128
SUPPORTED_SMS = (80, 86, 89, 90)
# The channels a STREAMED_WIDTHS layer projects to.
STREAMED_CHANNELS = (768,)
# Both kernels read the contiguous heads-inner output in slabs of this many
# channels, so they see H*D / SLAB heads of SLAB channels whatever the split.
SLAB = 64


def _pair_layout(pair: torch.Tensor) -> KernelOperand | None:
    """Lay a ``[B, I, J, L]`` pair tensor out as the kernel's ``(J, L, B*I)``, or ``None``.

    The ending node's ``transpose(1, 2)`` separates ``B`` from ``I``, so its
    two modes fold only when ``B`` is 1.
    """
    batch, rows, columns, width = pair.shape
    batch_stride, row_stride, column_stride, inner_stride = pair.stride()
    if inner_stride != 1:
        return None
    if batch == 1:
        folded = rows
    elif batch_stride == rows * row_stride:
        folded = batch * rows
    else:
        return None
    return pair, (columns, width, folded), (column_stride, 1, row_stride)


def _attention_layout(mha_o: torch.Tensor, heads: int, head_dim: int) -> KernelOperand | None:
    """Lay the heads-inner ``[..., J, H, D]`` attention output out as ``(J, D, H, B*I)``, or ``None``."""
    if not mha_o.is_contiguous():
        return None
    columns = mha_o.shape[-3]
    width = heads * head_dim
    return mha_o, (columns, head_dim, heads, mha_o.numel() // (columns * width)), (width, 1, head_dim, columns * width)


def _tma_ready(operand: KernelOperand) -> bool:
    """Whether TMA can address ``operand``: a 16-byte aligned base and strides."""
    tensor, sizes, strides = operand
    if tensor.data_ptr() % 16:
        return False
    element_size = tensor.element_size()
    return all(
        stride * element_size % 16 == 0 for stride, size in zip(strides, sizes, strict=True) if size > 1 and stride != 1
    )


def _untie(operand: KernelOperand) -> KernelOperand:
    """Give extent-1 dimensions distinct strides.

    An extent-1 stride is never used to index, but TMA still encodes it, and
    two dimensions sharing a stride make the descriptor invalid, as ``(J, C,
    B*I)`` does with ``J == B*I == 1``.
    """
    tensor, sizes, strides = operand
    if 1 not in sizes:
        return operand
    untied = list(strides)
    spare = max(stride * size for stride, size in zip(strides, sizes, strict=True))
    for index, size in enumerate(sizes):
        if size == 1:
            spare *= 2
            untied[index] = spare
    return tensor, sizes, tuple(untied)


def _as_view(operand: KernelOperand) -> torch.Tensor:
    """The kernel view of ``operand``, as the source executable takes it."""
    tensor, sizes, strides = operand
    return tensor.as_strided(sizes, strides, tensor.storage_offset())


class AttnEpilogue:
    """Gate, output projection and residual add as one kernel."""

    def __init__(
        self,
        backends: AttnEpilogueCuTe | Mapping[int, AttnEpilogueCuTe],
        heads: int,
        head_dim: int,
        channels: int,
        has_bias: bool = False,
        has_output_gate: bool = False,
        layer_heads: int | None = None,
        layer_head_dim: int | None = None,
    ) -> None:
        # One backend per tuning anchor; a call takes the anchor nearest its
        # folded rows. A lone backend serves every call.
        self._backends = dict(backends) if isinstance(backends, Mapping) else {0: backends}
        self._anchors = sorted(self._backends)
        # The kernel's head split; the layer's may differ for the streamed kernel.
        self._heads = heads
        self._head_dim = head_dim
        self._layer_heads = heads if layer_heads is None else layer_heads
        self._layer_head_dim = head_dim if layer_head_dim is None else layer_head_dim
        self._width = heads * head_dim
        self._channels = channels
        self._has_bias = has_bias
        self._has_output_gate = has_output_gate

    def _shapes_supported(
        self,
        mha_o: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor | None,
        bias: torch.Tensor | None,
        output_gate: torch.Tensor | None,
    ) -> bool:
        if (bias is not None) != self._has_bias or (output_gate is not None) != self._has_output_gate:
            return False
        optional = (output, bias, output_gate)
        operands = [mha_o, gate, weight, residual] + [tensor for tensor in optional if tensor is not None]
        if any(operand.dtype != torch.bfloat16 or not operand.is_cuda for operand in operands):
            return False
        if residual.ndim != 4 or residual.shape[-1] != self._channels:
            return False
        batch, rows, columns, _ = residual.shape
        # One pair row leaves the J and B*I modes both at extent 1, which
        # builds a TMA descriptor the kernel traps on.
        if batch * rows * columns <= 1:
            return False
        if output is not None and output.shape != residual.shape:
            return False
        if gate.shape != (batch, rows, columns, self._width):
            return False
        # Checked rather than assumed: a heads-outer buffer would otherwise be
        # read as heads-inner.
        if (
            mha_o.ndim < 3
            or tuple(mha_o.shape[-3:]) != (columns, self._layer_heads, self._layer_head_dim)
            or mha_o.numel() != gate.numel()
        ):
            return False
        if bias is not None and (bias.shape != (self._channels,) or bias.stride() != (1,) or bias.data_ptr() % 16):
            return False
        if output_gate is not None:
            gate_rows = output_gate.shape[1] if output_gate.ndim == 4 else 0
            if (
                gate_rows == 0
                or rows % gate_rows
                or tuple(output_gate.shape) != (batch, gate_rows, columns, self._channels)
            ):
                return False
        return weight.shape == (self._channels, self._width) and weight.stride() == (self._width, 1)

    def __call__(
        self,
        mha_o: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Write ``residual + o_proj(mha_o * sigmoid(gate))`` to ``output``.

        Args:
            mha_o: Attention output ``[..., J, H, D]``, heads-inner.
            gate: Unactivated gate projection ``[B, I, J, H*D]``.
            weight: Output projection weight ``[C, H*D]``.
            residual: Residual ``[B, I, J, C]``.
            output: Destination like ``residual``, possibly ``residual``
                itself; ``None`` allocates one.
            bias: Output projection bias ``[C]``, exactly when the op was
                built for one.
            output_gate: Unactivated output-gate logits ``[B, I_y, J, C]``,
                exactly when the op was built for them. ``I_y`` divides
                ``I``; residual row ``i`` reads gate row ``i // (I / I_y)``,
                which broadcasts the gate over a multiplicity.

        Returns:
            The destination, or ``None`` without launching when the kernel
            cannot take the operands without a copy.
        """
        if not self._shapes_supported(mha_o, gate, weight, residual, output, bias, output_gate):
            return None
        operands = [
            _attention_layout(mha_o, self._heads, self._head_dim),
            _pair_layout(gate),
            (weight, tuple(weight.shape), weight.stride()),
            _pair_layout(residual),
        ]
        if output_gate is not None:
            operands.append(_pair_layout(output_gate))
        if any(operand is None or not _tma_ready(operand) for operand in operands):
            return None
        if output is None:
            output = torch.empty_like(residual)
        destination = operands[3] if output is residual else _pair_layout(output)
        if destination is None or not _tma_ready(destination):
            return None
        attention, gate_layout, _, source, *gate_logits = (_untie(operand) for operand in operands)
        destination = _untie(destination)
        logits = gate_logits[0] if gate_logits else None
        backend = self._backends[nearest_anchor(self._anchors, residual.numel() // self._channels)]
        device_index = residual.get_device()
        executable = backend.executable(device_index)
        if isinstance(executable, AttnEpilogueCubinExecutable):
            executable.launch(attention, gate_layout, weight, bias, destination, source, logits)
        else:
            # A source-backed launch takes its stream from the current device,
            # not the operands'. Switch only when they differ: an unconditional
            # guard costs several microseconds a call.
            on_device = device_index == torch.cuda.current_device()
            with contextlib.nullcontext() if on_device else torch.cuda.device(device_index):
                launch_compiled_kernel(
                    executable,
                    _as_view(attention),
                    _as_view(gate_layout),
                    weight,
                    bias,
                    _as_view(destination),
                    _as_view(source),
                    None if logits is None else _as_view(logits),
                )
        return output


def get_attn_epilogue_op(
    dtype: torch.dtype | None,
    num_heads: int,
    head_dim: int,
    channels: int,
    has_bias: bool = False,
    has_output_gate: bool = False,
) -> AttnEpilogue | None:
    """Return the fused epilogue for a layer, or ``None`` when it cannot serve it.

    The kernels take bf16 layers on SM80, SM86, SM89 and SM90 whose ``H*D`` is
    128, 256 or 512 with ``D`` of 32, 64 or 128, projecting to 128 channels,
    and need a tuning for the exact shape and device. An output gate needs
    ``H*D`` of 128 or 256. Layers whose ``H*D`` and channels are both 768,
    such as the diffusion token transformers, take any head split, output
    gate included: on SM90 the streamed kernel or the channel-tiled SM80 one,
    and on SM80, SM86 and SM89 the channel-tiled one. Each call takes the
    tuning whose ``R=<rows>`` anchor is nearest its folded rows.
    """
    width = num_heads * head_dim
    if not torch.cuda.is_available() or dtype != torch.bfloat16:
        return None
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    if channels in STREAMED_CHANNELS:
        heads, dim = width // SLAB, SLAB
        supported = sm in SUPPORTED_SMS and width in STREAMED_WIDTHS
    else:
        heads, dim = num_heads, head_dim
        supported = (
            sm in SUPPORTED_SMS
            and head_dim in HEAD_DIMS
            and width in (OUTPUT_GATE_WIDTHS if has_output_gate else WIDTHS)
            and channels == CHANNELS
        )
    if not supported:
        return None
    backends = {}
    for tuning in tuned_configs(sm, heads, dim, channels):
        backend = AttnEpilogueCuTe(heads, dim, channels, has_bias, has_output_gate, anchor=tuning.rows)
        try:
            # Resolving here keeps JIT compilation out of the forward pass and
            # of any CUDA graph capture.
            backend.executable(device)
        except CuTeDSLKernelLibraryError:
            # Neither a tuned kernel source nor a packaged CUBIN serves this
            # anchor; its calls go to the nearest one that has either.
            continue
        backends[tuning.rows] = backend
    if not backends:
        return None
    return AttnEpilogue(backends, heads, dim, channels, has_bias, has_output_gate, num_heads, head_dim)
