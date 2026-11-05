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
from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache

from ._config import (
    BLACKWELL_SMS,
    OUTPUT_GATE_WIDTHS,
    SM100_OUTPUT_GATE_WIDTHS,
    SM100_TOKEN_TUNING,
    STREAMED_WIDTHS,
    TOKEN_WIDTH,
    nearest_anchor,
    tuned_configs,
)
from ._cubin import AttnEpilogueCubinExecutable, AttnEpilogueTokenCubinExecutable, KernelOperand
from .cutedsl import AttnEpilogueCuTe, AttnEpilogueTokenCuTe, max_token_clusters

HEAD_DIMS = (32, 64, 128)
WIDTHS = (128, 256, 512)
CHANNELS = 128
SUPPORTED_SMS = (80, 86, 89, 90)
# The channels a STREAMED_WIDTHS layer projects to.
STREAMED_CHANNELS = (256, 768)
# Both kernels read the contiguous heads-inner output in slabs of this many
# channels, so they see H*D / SLAB heads of SLAB channels whatever the split.
SLAB = 64
# Rows of one token-kernel tile.
_TOKEN_TILE_ROWS = 128


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
    """Gate, output projection and optional residual add as one kernel."""

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
        has_residual: bool = True,
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
        self._has_residual = has_residual

    def _shapes_supported(
        self,
        mha_o: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor | None,
        output: torch.Tensor | None,
        bias: torch.Tensor | None,
        output_gate: torch.Tensor | None,
    ) -> bool:
        if (bias is not None) != self._has_bias or (output_gate is not None) != self._has_output_gate:
            return False
        if (residual is not None) != self._has_residual:
            return False
        optional = (residual, output, bias, output_gate)
        operands = [mha_o, gate, weight] + [tensor for tensor in optional if tensor is not None]
        if any(operand.dtype != torch.bfloat16 or not operand.is_cuda for operand in operands):
            return False
        if gate.ndim != 4 or gate.shape[-1] != self._width:
            return False
        batch, rows, columns, _ = gate.shape
        # One pair row leaves the J and B*I modes both at extent 1, which
        # builds a TMA descriptor the kernel traps on.
        if batch * rows * columns <= 1:
            return False
        pair_shape = (batch, rows, columns, self._channels)
        if any(tensor is not None and tuple(tensor.shape) != pair_shape for tensor in (residual, output)):
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
        residual: torch.Tensor | None,
        output: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Write ``o_proj(mha_o * sigmoid(gate))``, plus ``residual`` when this op has one.

        Args:
            mha_o: Attention output ``[..., J, H, D]``, heads-inner.
            gate: Unactivated gate projection ``[B, I, J, H*D]``.
            weight: Output projection weight ``[C, H*D]``.
            residual: Residual ``[B, I, J, C]``, exactly when the op was built
                for one.
            output: Destination ``[B, I, J, C]``, possibly ``residual``
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
        layouts = {
            "o": _attention_layout(mha_o, self._heads, self._head_dim),
            "g": _pair_layout(gate),
            "w": (weight, tuple(weight.shape), weight.stride()),
        }
        if residual is not None:
            layouts["z"] = _pair_layout(residual)
        if output_gate is not None:
            layouts["y"] = _pair_layout(output_gate)
        if any(layout is None or not _tma_ready(layout) for layout in layouts.values()):
            return None
        if output is None:
            output = (
                torch.empty_like(residual)
                if residual is not None
                else gate.new_empty((*gate.shape[:-1], self._channels))
            )
        destination = layouts["z"] if output is residual else _pair_layout(output)
        if destination is None or not _tma_ready(destination):
            return None
        layouts = {name: _untie(layout) for name, layout in layouts.items()}
        destination = _untie(destination)
        attention, gate_layout = layouts["o"], layouts["g"]
        source, logits = layouts.get("z"), layouts.get("y")
        if not self._launch(attention, gate_layout, weight, bias, destination, source, logits):
            return None
        return output

    def _launch(
        self,
        attention: KernelOperand,
        gate_layout: KernelOperand,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        destination: KernelOperand,
        source: KernelOperand | None,
        logits: KernelOperand | None,
    ) -> bool:
        """Launch on the kernel views :meth:`__call__` laid out and checked; ``False`` declines the call."""
        # The gate has one H*D row per folded pair row, residual or not.
        columns, _, folded = gate_layout[1]
        backend = self._backends[nearest_anchor(self._anchors, columns * folded)]
        device_index = gate_layout[0].get_device()
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
                    None if source is None else _as_view(source),
                    None if logits is None else _as_view(logits),
                )
        return True


def _token_variant(rows: int, clusters: int) -> str:
    """Pick the token kernel variant for ``rows``: triples while one wave of ``clusters`` covers their tiles."""
    return "triples" if -(-rows // _TOKEN_TILE_ROWS) <= clusters else "pairs"


def _evenly_strided(operand: KernelOperand) -> bool:
    """Whether a ``(J, L, R)`` pair operand's rows follow on at its column stride, so it reads as ``(R * J, L)``."""
    _, (columns, _, folded), (column_stride, _, row_stride) = operand
    return folded == 1 or row_stride == columns * column_stride


def _flat(operand: KernelOperand) -> torch.Tensor:
    """An evenly strided ``(J, L, R)`` pair operand as its ``(R * J, L)`` rows."""
    tensor, (columns, width, folded), (column_stride, _, _) = operand
    return tensor.as_strided((folded * columns, width), (column_stride, 1), tensor.storage_offset())


class AttnEpilogueToken(AttnEpilogue):
    """The SM100 token transformers' epilogue: residual, output gate, optional bias, ``H*D = C = TOKEN_WIDTH``.

    It takes the operands :class:`AttnEpilogue` takes, under the same checks,
    and reads the gate, residual, destination and output gate as flat row
    views. It declines a call whose samples do not step evenly from one to the
    next, since those operands have no such view. A call nearer an anchor of
    ``small`` than ``anchor`` takes that anchor's SM80 kernel, which reads the
    operands as :class:`AttnEpilogue` does.

    Args:
        backends: One backend per variant of ``SM100_TOKEN_TUNING``.
        heads: Heads of the layer's attention output.
        head_dim: Channels per head of the layer's attention output.
        clusters: Clusters the backends' persistent grids launch at most.
        has_bias: Whether the backends add the output-projection bias.
        small: SM80 backends by ``R=<rows>`` anchor, for small calls.
        anchor: The token kernel's anchor, above every anchor of ``small``.
    """

    def __init__(
        self,
        backends: Mapping[str, AttnEpilogueTokenCuTe],
        heads: int,
        head_dim: int,
        clusters: int,
        has_bias: bool = False,
        small: Mapping[int, AttnEpilogueCuTe] | None = None,
        anchor: int = 0,
    ) -> None:
        small = small or {}
        super().__init__(
            small,
            TOKEN_WIDTH // SLAB,
            SLAB,
            TOKEN_WIDTH,
            has_bias=has_bias,
            has_output_gate=True,
            layer_heads=heads,
            layer_head_dim=head_dim,
        )
        self._variants = dict(backends)
        self._clusters = clusters
        # Rows nearest an SM80 anchor: up to the midpoint below the token kernel's, which a tie leaves to SM80.
        self._small_rows = (max(small) + anchor) // 2 if small else -1

    def _launch(
        self,
        attention: KernelOperand,
        gate_layout: KernelOperand,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        destination: KernelOperand,
        source: KernelOperand | None,
        logits: KernelOperand | None,
    ) -> bool:
        columns, _, folded = gate_layout[1]
        rows = columns * folded
        if rows <= self._small_rows:
            return super()._launch(attention, gate_layout, weight, bias, destination, source, logits)
        if not all(_evenly_strided(operand) for operand in (gate_layout, destination, source, logits)):
            return False
        device_index = weight.get_device()
        executable = self._variants[_token_variant(rows, self._clusters)].executable(device_index)
        on_device = device_index == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(device_index):
            if isinstance(executable, AttnEpilogueTokenCubinExecutable):
                executable.launch(attention, gate_layout, weight, bias, destination, source, logits, self._clusters)
            else:
                # The heads-inner attention output is contiguous: one H*D row per pair row. Output-gate row
                # ``i // gate_repeat`` serves pair row ``i``.
                launch_compiled_kernel(
                    executable,
                    attention[0].view(rows, self._width),
                    _flat(gate_layout),
                    weight,
                    bias,
                    _flat(destination),
                    _flat(source),
                    _flat(logits),
                    columns,
                    folded // logits[1][2],
                )
        return True


def get_attn_epilogue_op(
    dtype: torch.dtype | None,
    num_heads: int,
    head_dim: int,
    channels: int,
    has_bias: bool = False,
    has_output_gate: bool = False,
    has_residual: bool = True,
) -> AttnEpilogue | None:
    """Return the fused epilogue for a layer, or ``None`` when it cannot serve it.

    The kernels take bf16 layers on SM80, SM86, SM89 and SM90 whose ``H*D`` is
    128, 256 or 512 with ``D`` of 32, 64 or 128, projecting to 128 channels,
    and need a tuning for the exact shape and device. An output gate needs
    ``H*D`` of 128 or 256. Layers whose ``H*D`` and channels are both 256 or
    both 768, such as Protenix's triangle attention and the diffusion token
    transformers, take any head split, output gate included: on SM90 the streamed kernel or the channel-tiled SM80 one,
    and on SM80, SM86 and SM89 the channel-tiled one. Each call takes the
    tuning whose ``R=<rows>`` anchor is nearest its folded rows.

    SM100 and SM103 run their own kernels, at any head split: a 2-SM kernel
    for the residual epilogues whose ``H*D`` equals ``C`` of 128 or 256, with
    or without the output-projection bias and, at ``C`` of 128, with or
    without an output gate, and one for the token transformers', with a
    residual, an output gate and an optional bias, whose ``H*D`` and ``C``
    are both 768. Where a config file tunes the SM80 kernel for a width, as
    on SM100, calls nearer its small anchors take that kernel instead. Other
    layers there keep the unfused path.

    With ``has_residual=False`` the op writes the bare projection and takes no
    output gate; packaged CUBINs cover such layers at ``H*D`` in
    ``NO_RESIDUAL_WIDTHS``, and a source checkout compiles the rest.
    """
    width = num_heads * head_dim
    if not torch.cuda.is_available() or dtype != torch.bfloat16 or (has_output_gate and not has_residual):
        return None
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    if sm in BLACKWELL_SMS:
        if width == channels == TOKEN_WIDTH:
            # The token transformers' gated epilogue is the only one this width runs.
            return _token_op(sm, device, num_heads, head_dim, has_bias) if has_output_gate else None
        # The triangle kernel reads whole H*D rows, so any head split serves.
        heads, dim = width // SLAB, SLAB
        supported = channels == width and has_residual and (not has_output_gate or width in SM100_OUTPUT_GATE_WIDTHS)
    elif channels in STREAMED_CHANNELS:
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
        backend = AttnEpilogueCuTe(
            heads, dim, channels, has_bias, has_output_gate, anchor=tuning.rows, has_residual=has_residual
        )
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
    return AttnEpilogue(backends, heads, dim, channels, has_bias, has_output_gate, num_heads, head_dim, has_residual)


def _token_op(sm: int, device: int, num_heads: int, head_dim: int, has_bias: bool) -> AttnEpilogueToken | None:
    """Return the SM100 token transformer's epilogue, or ``None`` when neither its source nor its CUBINs serve."""
    heads, dim = TOKEN_WIDTH // SLAB, SLAB
    tunings = tuned_configs(sm, heads, dim, TOKEN_WIDTH)
    if not tunings:
        return None
    *small_tunings, native = tunings
    if native.kernel_abi != "sm100_token" or any(tuning.kernel_abi != "sm80" for tuning in small_tunings):
        # The op routes small calls by one threshold below the token kernel's anchor.
        raise ValueError(f"SM{sm} token-width tunings must name sm80 anchors below one sm100_token anchor")
    small = {}
    for tuning in small_tunings:
        backend = AttnEpilogueCuTe(heads, dim, TOKEN_WIDTH, has_bias, True, anchor=tuning.rows)
        try:
            # Resolving here keeps JIT compilation out of the forward pass and
            # of any CUDA graph capture.
            backend.executable(device)
        except CuTeDSLKernelLibraryError:
            # Its calls go to the nearest anchor that has a kernel.
            continue
        small[tuning.rows] = backend
    try:
        clusters = max_token_clusters(heads, dim, device, CuteKernelCache.force_cubin())
        if clusters < 1:
            # No six-CTA cluster fits, as on a small partition of the GPU.
            return None
        backends = {
            variant: AttnEpilogueTokenCuTe(heads, dim, variant, clusters, has_bias, anchor=native.rows)
            for variant in SM100_TOKEN_TUNING[sm]
        }
        for backend in backends.values():
            backend.executable(device)
    except CuTeDSLKernelLibraryError:
        return None
    return AttnEpilogueToken(backends, num_heads, head_dim, clusters, has_bias, small=small, anchor=native.rows)
