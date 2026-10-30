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
"""Weight pack, PyTorch reference and op for the SM90 transition with its LayerNorm."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..trimul_kf_k3.ops import centre_rows_bf16
from . import _config
from ._config import ACTIVATIONS
from .cutedsl import TransitionMlpWNormCuTe

__all__ = [
    "ACTIVATIONS",
    "MAX_BATCH",
    "MAX_ROWS",
    "TransitionMlpWNormOp",
    "TransitionMlpWNormPack",
    "get_transition_mlp_w_norm_op",
    "pack_weights",
    "transition_mlp_w_norm_reference",
]

_SUPPORTED_DTYPE = torch.bfloat16
#: The kernel builds its tile table for at most this many pair maps in one launch.
MAX_BATCH = 64
#: Rows are int32 in the kernel's tile arithmetic, with a tile of headroom: ``rows < 2**31 - 256``.
MAX_ROWS = 2**31 - 257
#: Per activation and width: hidden units per chunk and W2 slots per chunk of the kernel's weight stream, as the
#: kernel's ``TILING`` holds them; W1 takes as many slots per chunk for each of its row blocks.
IMAGE_LAYOUT = {
    "relu": {384: (64, 2), 256: (128, 4), 128: (128, 1), 64: (64, 1)},
    "silu_gate": {256: (64, 2), 128: (128, 1), 64: (64, 1)},
}
#: W1 row blocks per hidden unit, per activation: SwiGLU's ``value`` rows, then its ``gate`` rows.
_W1_BLOCKS = {"relu": 1, "silu_gate": 2}
#: Per activation, up to this width the kernel reads z from registers as GEMM 1's A fragments, and W1's columns
#: follow their order (:func:`_a_fragment_order`); the kernel's ``A_REGS_WIDTH`` is the same.
A_REGS_WIDTH = {"relu": 128}
# The kernel bulk-copies z and the weight image and stores the output 16 bytes at a time.
_ALIGNMENT_BYTES = 16
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_transition_mlp_w_norm_instances: dict[int, TransitionMlpWNormCuTe] = {}
# SM version per CUDA device index, read once: an op runs only on devices of its backend's SM.
_device_sms: dict[int, int] = {}


def _device_sm(device: torch.device) -> int:
    """The SM version of the CUDA ``device``."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm = _device_sms.get(index)
    if sm is None:
        major, minor = torch.cuda.get_device_capability(index)
        sm = _device_sms[index] = major * 10 + minor
    return sm


def _permuted_channel(n: torch.Tensor) -> torch.Tensor:
    """The output channel in row ``n`` of the packed W2.

    A WGMMA accumulator gives a thread the columns ``8 v + 2 (t % 4) + e`` of each 32-column group; in this order
    they are 8 consecutive channels, which the kernel reads and stores 16 bytes at a time.
    """
    return (n & ~31) | (((n >> 1) & 3) << 3) | (((n >> 3) & 3) << 1) | (n & 1)


def _kmajor_slabs(m: torch.Tensor) -> torch.Tensor:
    """``m`` ``[..., R, K]`` in WGMMA's K-major shared-memory layout, flattened over its last two dimensions.

    Slabs of ``S / 2`` columns, ``S`` the widest swizzle the columns allow: 128 bytes for whole 64-column slabs, else
    one slab of ``2 K`` bytes (64 or 32). Row ``n`` of a slab starts at ``n S``; its 16-byte unit ``c`` sits at
    ``c ^ ((n S / 128) mod (S / 16))``.
    """
    *lead, rows, cols = m.shape
    width = 128 if cols % 64 == 0 else 2 * cols
    per_slab, units = width // 2, width // 16
    device = m.device
    slab = torch.arange(cols // per_slab, device=device).view(-1, 1, 1, 1)
    n = torch.arange(rows, device=device).view(1, -1, 1, 1)
    pos = torch.arange(units, device=device).view(1, 1, -1, 1)
    e = torch.arange(8, device=device).view(1, 1, 1, -1)
    col = slab * per_slab + 8 * (pos ^ ((n * width >> 7) & (units - 1))) + e
    return m[..., n.expand_as(col), col].reshape(*lead, rows * cols)


def _a_fragment_order(width: int, device: torch.device) -> torch.Tensor:
    """The channel at each K position of GEMM 1 when its A operand comes from the threads' own channels.

    K-step ``2 u + h`` holds channels ``32 u + 8 q + 4 h + (0, 1)`` at fragment columns ``2 q + (0, 1)`` and
    ``32 u + 8 q + 4 h + (2, 3)`` at ``2 q + 8 + (0, 1)``, for each quad lane ``q``.
    """
    k = torch.arange(width, device=device)
    step, col = k // 16, k % 16
    return 32 * (step // 2) + 8 * ((col % 8) // 2) + 4 * (step % 2) + 2 * (col // 8) + col % 2


def _weight_image(w1_rows: torch.Tensor, w2: torch.Tensor, activation: str) -> torch.Tensor:
    """bf16 ``[(blocks + 1) * C * H]``: W1' and W2 in the order and shared-memory layout the kernel consumes them.

    ``w1_rows`` is ``[blocks * H, C]``, its row blocks one after the other. Per hidden chunk ``j`` of ``hc`` units
    (:data:`IMAGE_LAYOUT`), equal ring slots: the chunk's W1' rows (``hc`` of each block, block by block) split along
    C into ``blocks * w2_slots`` slots, then W2's permuted rows split along the chunk's hidden columns into
    ``w2_slots`` slots, each slot in :func:`_kmajor_slabs` layout.
    """
    width, hidden = w2.shape
    hidden_chunk, w2_slots = IMAGE_LAYOUT[activation][width]
    blocks = _W1_BLOCKS[activation]
    chunks, w1_slots = hidden // hidden_chunk, blocks * w2_slots
    if width <= A_REGS_WIDTH.get(activation, 0):
        w1_rows = w1_rows[:, _a_fragment_order(width, w1_rows.device)]
    w1_chunks = (
        w1_rows.view(blocks, chunks, hidden_chunk, width)
        .transpose(0, 1)
        .reshape(chunks, blocks * hidden_chunk, w1_slots, width // w1_slots)
        .transpose(1, 2)
    )
    channels = _permuted_channel(torch.arange(width, device=w2.device))
    w2_chunks = w2[channels].view(width, chunks, w2_slots, hidden_chunk // w2_slots).permute(1, 2, 0, 3)
    return torch.cat([_kmajor_slabs(w1_chunks), _kmajor_slabs(w2_chunks)], dim=1).reshape(-1).contiguous()


@dataclass(frozen=True)
class TransitionMlpWNormPack:
    """A transition's weights as the fused kernel reads them; see :func:`pack_weights`.

    Attributes:
        image: bf16 ``[(blocks + 1) * width * hidden]``: W1 with the LayerNorm folded in, every row centred, and W2
            with its rows permuted, in the kernel's consumption order and shared-memory layout. ``blocks`` is W1's
            rows over ``hidden``: 1 for ReLU, 2 for SwiGLU.
        vec: fp32 ``[blocks * hidden + width]``: ``b1 + W1 beta`` (SwiGLU: the value half, then the gate half
            halved), then ``b2``.
        width: Feature width ``C``.
        hidden: Hidden width ``H``.
        activation: ``"relu"`` or ``"silu_gate"``.
    """

    image: torch.Tensor
    vec: torch.Tensor
    width: int
    hidden: int
    activation: str = "relu"
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        if self.activation not in _config.W1_BLOCKS:
            raise ValueError(f"TransitionMlpWNormPack has no {self.activation!r} activation")
        width, hidden, activation = self.width, self.hidden, self.activation
        if (
            self.image.dtype != _SUPPORTED_DTYPE
            or self.image.shape != (_config.image_elements(width, hidden, activation),)
            or not self.image.is_contiguous()
            or self.vec.dtype != torch.float32
            or self.vec.shape != (_config.vec_elements(width, hidden, activation),)
            or not self.vec.is_contiguous()
            or self.vec.device != self.image.device
            or any(t.data_ptr() % _ALIGNMENT_BYTES for t in (self.image, self.vec))
        ):
            raise ValueError(
                "TransitionMlpWNormPack needs the 16-byte aligned contiguous bf16 image and fp32 vector of pack_weights"
            )
        object.__setattr__(self, "device", self.image.device)


@torch.no_grad()
def pack_weights(
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    *,
    activation: str = "relu",
) -> TransitionMlpWNormPack:
    """Fold the LayerNorm into the first linear and lay both linears out for the fused kernel.

    ``LayerNorm(z) W1^T + b1 = rstd (z W1'^T) + b1'`` with ``W1' = W1 diag(gamma)``, each row centred by
    :func:`~bionemo_ir._torch.custom_ops.trimul_kf_k3.ops.centre_rows_bf16` so that the bf16 row still sums to about
    zero and the mean of ``z`` drops out, and ``b1' = b1 + W1 beta``. SwiGLU folds both halves of W1 alike and then
    halves the gate rows and their bias, exactly in bf16 and fp32: the kernel takes ``g / 2`` for
    ``silu(g) = (g / 2) (1 + tanh(g / 2))``. The fold is fixed for fixed weights, keeps no autograd history and
    records no host synchronization, so it can run under graph capture. A missing bias is zero.

    Args:
        norm_weight: ``[C]`` LayerNorm weight ``gamma``.
        norm_bias: ``[C]`` LayerNorm bias ``beta``, or ``None``.
        w1: ``[H, C]`` first linear weight; SwiGLU's ``[2 H, C]`` holds the value rows, then the gate rows.
        b1: ``[H]`` (SwiGLU ``[2 H]``) first linear bias, or ``None``.
        w2: ``[C, H]`` second linear weight.
        b2: ``[C]`` second linear bias, or ``None``.
        activation: ``"relu"`` or ``"silu_gate"``; the kernel's weight order (:data:`IMAGE_LAYOUT`) depends on it
            and on ``C``.
    """
    width, hidden = w2.shape
    if (
        activation not in IMAGE_LAYOUT
        or width not in IMAGE_LAYOUT[activation]
        or hidden not in (2 * width, 4 * width)
        or w1.shape != (_W1_BLOCKS[activation] * hidden, width)
        or norm_weight.shape != (width,)
        or (norm_bias is not None and norm_bias.shape != (width,))
        or (b1 is not None and b1.shape != (w1.shape[0],))
        or (b2 is not None and b2.shape != (width,))
    ):
        raise ValueError(
            f"the fused transition packs W1 [H, C] and W2 [C, H] with C in {sorted(IMAGE_LAYOUT.get(activation, ()))} "
            f"and H = 2 C or 4 C ({activation}), not W1 {tuple(w1.shape)} and W2 {tuple(w2.shape)}"
        )
    device = w1.device
    w1d = w1.double()
    beta = torch.zeros(width, dtype=torch.float64, device=device) if norm_bias is None else norm_bias.double()
    bias1 = torch.zeros(w1.shape[0], dtype=torch.float64, device=device) if b1 is None else b1.double()
    bias2 = torch.zeros(width, dtype=torch.float32, device=device) if b2 is None else b2.float()
    folded_bias = (bias1 + w1d @ beta).float()
    folded = centre_rows_bf16(w1d * norm_weight.double())
    if activation == "silu_gate":
        folded = torch.cat([folded[:hidden], folded[hidden:] * 0.5])
        folded_bias = torch.cat([folded_bias[:hidden], folded_bias[hidden:] * 0.5])
    image = _weight_image(folded, w2.to(_SUPPORTED_DTYPE), activation)
    vec = torch.cat([folded_bias, bias2]).contiguous()
    return TransitionMlpWNormPack(image, vec, width, hidden, activation)


def transition_mlp_w_norm_reference(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    eps: float,
    *,
    activation: str = "relu",
    residual: bool = True,
) -> torch.Tensor:
    """fp32 ``[z +] mask * (act(LayerNorm(z) @ w1.T + b1) @ w2.T + b2)``.

    ``act`` is ReLU, or for ``"silu_gate"`` ``silu(gate) * value`` where ``w1``'s first ``H`` rows make ``value`` and
    the next ``H`` rows ``gate``, as in :class:`~bionemo_ir._torch.layers.transition.Transition`. ``mask`` holds one
    value per row; ``None`` keeps every row.
    """
    zf = z.float()

    def fp32(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.float()

    normed = torch.nn.functional.layer_norm(zf, (zf.shape[-1],), fp32(norm_weight), fp32(norm_bias), eps)
    projected = torch.nn.functional.linear(normed, w1.float(), fp32(b1))
    if activation == "silu_gate":
        value, gate = projected.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value
    else:
        hidden = torch.relu(projected)
    update = torch.nn.functional.linear(hidden, w2.float(), fp32(b2))
    if mask is not None:
        update = mask.reshape(*zf.shape[:-1], 1).float() * update
    return zf + update if residual else update


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two contiguous tensors share any byte."""
    return a.data_ptr() < b.data_ptr() + b.nbytes and b.data_ptr() < a.data_ptr() + a.nbytes


@dataclass(frozen=True)
class TransitionMlpWNormOp:
    """The fused transition bound to one ``(width, hidden, activation)``; see :func:`get_transition_mlp_w_norm_op`."""

    backend: TransitionMlpWNormCuTe
    width: int
    hidden: int
    activation: str = "relu"

    def accepts(self, z: torch.Tensor, mask: torch.Tensor | None, *, n_valid: torch.Tensor | None = None) -> bool:
        """Whether a call on ``z``, ``mask`` and ``n_valid`` can run.

        ``z`` is contiguous, 16-byte aligned bf16 ``[..., width]`` on a CUDA device of the backend's SM with between 1
        and :data:`MAX_ROWS` rows. ``mask`` is ``None``, or contiguous bf16 on ``z``'s device with one value per row,
        shaped ``z.shape[:-1]`` with or without a trailing 1. With ``n_valid``, ``z`` is a square pair map
        ``[B, N, N, width]`` with ``B <= MAX_BATCH`` and ``n_valid`` contiguous int32 ``[B]`` on its device.
        """
        if (
            z.dim() < 2
            or z.shape[-1] != self.width
            or z.dtype != _SUPPORTED_DTYPE
            or not z.is_cuda
            or _device_sm(z.device) != self.backend.sm_version
            or not z.is_contiguous()
            or z.data_ptr() % _ALIGNMENT_BYTES
            or not 0 < z.numel() // self.width <= MAX_ROWS
        ):
            return False
        if n_valid is not None and not (
            z.dim() == 4
            and z.shape[0] <= MAX_BATCH
            and z.shape[1] == z.shape[2]
            and n_valid.dtype == torch.int32
            and n_valid.device == z.device
            and n_valid.is_contiguous()
            and n_valid.shape == (z.shape[0],)
        ):
            return False
        return mask is None or (
            mask.dtype == _SUPPORTED_DTYPE
            and mask.device == z.device
            and mask.is_contiguous()
            and mask.shape in (z.shape[:-1], (*z.shape[:-1], 1))
        )

    def __call__(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        pack: TransitionMlpWNormPack,
        eps: float,
        *,
        residual: bool = True,
        n_valid: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        layout: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """``[z +] mask * (act(LayerNorm(z) @ W1.T + b1) @ W2.T + b2)`` over every row of ``z``, or a pair map's
        valid rows.

        A row whose mask is 0 comes out equal to ``z`` with the residual and 0 without it. A call too small to fill
        the GPU takes smaller row tiles, a hidden dimension split over a cluster of CTAs per tile, or both: the
        config's launch layouts by the call's waves on this device (:func:`._config.choose_layout`). A split sums its
        partials in a fixed order, so every layout's output is the same from call to call, and a split of 1 matches
        the full tile bit for bit.

        Args:
            z: bf16 ``[..., width]``, contiguous: flat rows (``[M, width]``, an MSA's ``[B, S, N, width]``) or, with
                ``n_valid``, pair maps ``[B, N, N, width]``; see :meth:`accepts`.
            mask: bf16 with one value per row, scaling the update; ``None`` keeps every row.
            pack: The weights :func:`pack_weights` packed for this op's shape and activation.
            eps: The LayerNorm's epsilon.
            residual: Whether to add ``z``.
            n_valid: int32 ``[B]``: only the rows ``(b, i, j)`` with ``i, j < n_valid[b]`` are computed (each length
                clamped to ``0 .. N``); the others are neither read nor written, so they keep what ``out`` held.
            out: Optional contiguous bf16 output shaped like ``z``: ``z`` itself (in place) or overlapping no
                operand. A new tensor by default.
            layout: ``(tile_wgs, split)`` to launch instead of the configured choice, one of :meth:`layouts`.
        """
        if not self.accepts(z, mask, n_valid=n_valid):
            raise ValueError(
                f"the fused transition needs contiguous bf16 z [..., {self.width}] on an SM{self.backend.sm_version} "
                f"device with at most {MAX_ROWS} rows, a "
                "contiguous bf16 mask with one value per row on its device or None, and, with n_valid, pair maps "
                f"[B, N, N, {self.width}] with B <= {MAX_BATCH} and contiguous int32 n_valid [B]"
            )
        if (pack.width, pack.hidden, pack.activation) != (self.width, self.hidden, self.activation) or (
            pack.device != z.device
        ):
            raise ValueError(
                f"the fused transition needs the width={self.width}, hidden={self.hidden}, "
                f"activation={self.activation!r} pack on z's device"
            )
        if out is None:
            out = torch.empty_like(z)
        else:
            if (
                out.shape != z.shape
                or out.dtype != z.dtype
                or out.device != z.device
                or not out.is_contiguous()
                or out.data_ptr() % _ALIGNMENT_BYTES
            ):
                raise ValueError("the fused transition's output must be a contiguous, 16-byte aligned bf16 z")
            if out.data_ptr() != z.data_ptr() and _overlaps(out, z):
                raise ValueError("the fused transition's output must be z itself or not overlap it")
            if any(
                operand is not None and _overlaps(out, operand) for operand in (mask, n_valid, pack.image, pack.vec)
            ):
                raise ValueError("the fused transition's output must not overlap the mask, n_valid or the pack")
        n, n_batch = (z.shape[1], z.shape[0]) if n_valid is not None else (0, 0)
        self.backend.run(
            z.view(-1),
            None if mask is None else mask.view(-1),
            pack.image,
            pack.vec,
            n_valid,
            z.numel() // self.width,
            n,
            n_batch,
            float(eps),
            residual,
            out.view(-1),
            self.width,
            self.hidden,
            self.activation,
            layout,
        )
        return out

    def layouts(self) -> tuple[tuple[int, int], ...]:
        """Every ``(tile_wgs, split)`` launch layout the kernel takes: ``tile_wgs`` of its warpgroups per row tile and
        ``split`` CTAs per tile."""
        _, num_wg, allowed = self.backend.geometry(self.width, self.hidden, self.activation)
        return tuple(
            (tile_wgs, split)
            for tile_wgs in (1, 2, 4)
            if tile_wgs <= num_wg
            for split in range(1, 9)
            if allowed >> _config.layout_bit(tile_wgs, split) & 1
        )


def get_transition_mlp_w_norm_op(
    dtype: torch.dtype | None, width: int, hidden: int, activation: str = "relu", *, candidates: bool = False
) -> TransitionMlpWNormOp | None:
    """Return the fused transition when this build ships it for ``(width, hidden, activation)`` on this GPU, else
    ``None``.

    A shape ships on SM90 in bf16 where a config entry declares it (see :mod:`._config`), from source in a private
    checkout and as a packaged CUBIN otherwise. ``candidates`` also returns the op for a shape whose entry is a
    candidate: built, but not yet shown to beat the path it would replace.
    """
    if dtype != _SUPPORTED_DTYPE or activation not in _config.W1_BLOCKS or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if not _config.ships(sm_version, width, hidden, activation, candidates=candidates):
        return None
    backend = _transition_mlp_w_norm_instances.get(sm_version)
    if backend is None:
        backend = _transition_mlp_w_norm_instances.setdefault(sm_version, TransitionMlpWNormCuTe(sm_version))
    if not backend.ships(width, hidden, activation, candidates=candidates):
        return None
    return TransitionMlpWNormOp(backend, width, hidden, activation)
