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
"""Variant selection, output weight fold and PyTorch reference for the SM90 TriMul KF K3."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..trimul_kf_k1.cutedsl import MAX_LAUNCH_ROWS, stats_rows
from ._config import PITCHED_VARIANTS, TrimulKFK3Selection
from .cutedsl import TrimulKFK3CuTe

_SUPPORTED_DTYPE = torch.bfloat16
#: Token counts must keep every channel-major row 16-byte aligned for TMA.
TOKEN_ALIGN = 8
#: Output columns per column tile of the centred weights; their rows are permuted within each tile.
CENTRED_TILE = 128
#: Rounding passes of :func:`centre_rows_bf16`: each takes the row sum of the rounded weights an order of
#: magnitude closer to zero; eight leave under 1e-6 of the row's absolute sum.
_CENTRE_PASSES = 8
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_trimul_kf_k3_instances: dict[int, TrimulKFK3CuTe] = {}


def _bf16_neighbour(r: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The bf16 value next to each entry of ``r``: towards +inf where ``up``, else towards -inf."""
    bits = r.view(torch.int16).to(torch.int32)
    # bf16 is sign-magnitude: one step of the magnitude bits moves away from zero
    away = torch.where(bits < 0, ~up, up)
    stepped = torch.where(away, bits + 1, bits - 1)
    zero = (bits & 0x7FFF) == 0
    stepped = torch.where(zero, torch.where(up, 1, -32767), stepped)
    return stepped.to(torch.int16).view(torch.bfloat16)


def centre_rows_bf16(w: torch.Tensor, passes: int = _CENTRE_PASSES) -> torch.Tensor:
    """Every row of ``w`` minus its mean, in bf16, rounded so the row still sums to about zero.

    ``a @ centred.T == (a - mean(a)) @ w.T`` for any row ``a``: a kernel multiplying by the centred rows
    needs no mean term of a LayerNorm in front. Rounding each entry to nearest leaves a row sum of about
    ``sqrt(K)`` rounding steps, which would multiply the input's mean; each pass moves entries one step
    against the remaining sum, the largest steps that fit in it first, never further than 1.5 steps from
    the exact value. The pass count is fixed: no data-dependent control flow, so a fold under graph
    capture records no host synchronization.
    """
    exact = w.double() - w.double().mean(-1, keepdim=True)
    r = exact.to(torch.bfloat16)
    for _ in range(passes):
        rd = r.double()
        rest = rd.sum(-1, keepdim=True)
        moved = _bf16_neighbour(r, (rest < 0).expand_as(r))
        md = moved.double()
        step = (md - rd).abs()
        # the moved entry's error in its own steps: 0.5 is as good as rounding to nearest; an entry is moved
        # only while it stays within 1.5 steps of its exact value
        cost = (md - exact).abs() / step.clamp_min(1e-300)
        fits = (step <= rest.abs()) & (cost <= 1.5)
        # the largest steps that fit first (like making change), the least costly among equal steps
        by_cost = torch.argsort(torch.where(fits, cost, torch.inf), dim=-1, stable=True)
        by_step = torch.argsort(torch.where(fits, -step, 0.0).gather(-1, by_cost), dim=-1, stable=True)
        order = by_cost.gather(-1, by_step)
        total = torch.where(fits, step, 0.0).gather(-1, order).cumsum(-1)
        take = (total <= rest.abs()) & fits.gather(-1, order)
        r = torch.where(torch.zeros_like(take).scatter(-1, order, take), moved, r)
    return r.contiguous()


def _centred_row_order(C: int, device: torch.device) -> torch.Tensor:
    """Row ``j`` of every 128-row tile of the centred weights is the folded row ``sig(j)`` of that tile.

    A WGMMA accumulator gives thread ``t`` of a warp columns ``8 v + 2 (t % 4) + e``; with this order its
    columns ``v = 4 Q + u`` are output columns ``32 Q + 8 (t % 4) + 2 u + e``: 8 consecutive ones per ``Q``,
    one 16-byte store.
    """
    j = torch.arange(CENTRED_TILE, device=device)
    v, t, e = j // 8, (j % 8) // 2, j % 2
    sig = 32 * (v // 4) + 8 * t + 2 * (v % 4) + e
    return (torch.arange(C // CENTRED_TILE, device=device)[:, None] * CENTRED_TILE + sig).reshape(-1)


@dataclass(frozen=True)
class TrimulKFOutputFold:
    """Both LayerNorms folded into K3's weights; see :func:`fold_output_weights`.

    Attributes:
        w_out: bf16 ``[C, D]`` output projection scaled by the output LayerNorm weight.
        w_gate: bf16 ``[C, C]`` gate projection scaled by the input LayerNorm weight.
        vec: fp32 ``[4C]``: for every output channel, the sum of its folded projection row, then
            ``W_out @ ln_out_bias + p_out_bias``, then the same two terms of the gate.
        centred: ``(w_out, w_gate)`` for the :data:`~._config.CENTRED_VARIANTS`: every row centred by
            :func:`centre_rows_bf16`, rows reordered within each 128-row tile by :func:`_centred_row_order`;
            ``None`` unless ``C`` is a multiple of 128.
    """

    w_out: torch.Tensor
    w_gate: torch.Tensor
    vec: torch.Tensor
    C: int = field(init=False)
    D: int = field(init=False)
    device: torch.device = field(init=False)
    centred: tuple[torch.Tensor, torch.Tensor] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        C, D = self.w_out.shape
        if (
            self.w_gate.shape != (C, C)
            or self.vec.shape != (4 * C,)
            or self.vec.dtype != torch.float32
            or any(w.dtype != _SUPPORTED_DTYPE or not w.is_contiguous() for w in (self.w_out, self.w_gate))
            or not self.vec.is_contiguous()
            or any(t.device != self.w_out.device for t in (self.w_gate, self.vec))
        ):
            raise ValueError("TrimulKFOutputFold needs the contiguous bf16 folds and fp32 terms of fold_output_weights")
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "device", self.w_out.device)
        centred = None
        if C % CENTRED_TILE == 0:
            order = _centred_row_order(C, self.w_out.device)
            centred = tuple(centre_rows_bf16(w)[order].contiguous() for w in (self.w_out, self.w_gate))
        object.__setattr__(self, "centred", centred)

    def weights(self, selection: TrimulKFK3Selection) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(w_out, w_gate, vec)`` as ``selection``'s variant reads them."""
        if not selection.centred:
            return self.w_out, self.w_gate, self.vec
        if self.centred is None:
            raise ValueError(f"trimul KF K3 {selection.kernel_variant} needs C % {CENTRED_TILE} == 0, got C={self.C}")
        return (*self.centred, self.vec)


def fold_output_weights(
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor | None,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    p_out_bias: torch.Tensor | None = None,
    g_out_bias: torch.Tensor | None = None,
) -> TrimulKFOutputFold:
    """Fold the output LayerNorm into the output projection and the input LayerNorm into the gate.

    The gate reads ``LayerNorm_in(x)``, so it takes the input LayerNorm's parameters. Any output
    biases fold into the bias terms. The fold is fixed for fixed weights. A missing LayerNorm
    bias is a zero beta.

    Args:
        norm_out_weight: ``[D]`` output LayerNorm weight.
        norm_out_bias: ``[D]`` output LayerNorm bias, or ``None`` when the norm learns none.
        norm_in_weight: ``[C]`` input LayerNorm weight.
        norm_in_bias: ``[C]`` input LayerNorm bias, or ``None`` when the norm learns none.
        p_out_weight: ``[C, D]`` output projection weight.
        g_out_weight: ``[C, C]`` gate weight.
        p_out_bias: Optional ``[C]`` output projection bias.
        g_out_bias: Optional ``[C]`` gate bias.
    """

    def fold(
        weight: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor | None, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = weight.float()
        gamma = gamma.float()
        beta = torch.zeros_like(gamma) if beta is None else beta.float()
        folded = (weight * gamma).to(torch.bfloat16)
        shift = weight @ beta
        if bias is not None:
            shift = shift + bias.float()
        return folded.contiguous(), folded.float().sum(-1), shift

    w_out, out_sum, out_shift = fold(p_out_weight, norm_out_weight, norm_out_bias, p_out_bias)
    w_gate, gate_sum, gate_shift = fold(g_out_weight, norm_in_weight, norm_in_bias, g_out_bias)
    vec = torch.cat((out_sum, out_shift, gate_sum, gate_shift)).contiguous()
    return TrimulKFOutputFold(w_out, w_gate, vec)


@dataclass(frozen=True)
class TrimulKFK3Op:
    """K3 bound to one ``(C, D)``; see :func:`get_trimul_kf_k3_op`."""

    backend: TrimulKFK3CuTe
    C: int
    D: int
    _selections: dict[int, TrimulKFK3Selection] = field(default_factory=dict, init=False, repr=False, compare=False)
    _pitched: dict[tuple[str, bool], bool] = field(default_factory=dict, init=False, repr=False, compare=False)

    def select(self, n: int) -> TrimulKFK3Selection:
        """The anchor and variant a call over ``n`` tokens runs, including whether it reads K1's statistics."""
        selection = self._selections.get(n)
        if selection is None:
            selection = self.backend.select(self.C, self.D, n)
            if selection is None:
                raise ValueError(f"trimul KF K3 ships no config for C={self.C}, D={self.D}")
            self._selections[n] = selection
        return selection

    def reads_pitch(self, n: int, residual: bool) -> bool:
        """Whether a call over ``n`` tokens takes K2's ``pitched`` product: its anchor's rule reads one at ``n``
        (``select(n).reads_pitch``) and this build ships that image with or without the residual."""
        if not self.select(n).reads_pitch:
            return False
        variant = self.select(n).kernel_variant
        key = (variant, residual)
        supported = self._pitched.get(key)
        if supported is None:
            supported = self._pitched[key] = self.backend.supports_pitch(self.C, self.D, variant, residual)
        return supported

    def accepts(self, prod: torch.Tensor, x: torch.Tensor) -> bool:
        """Whether a call on ``prod`` ``[B, D, N, N]`` and ``x`` ``[B, N, N, C]`` can run.

        ``prod`` is dense, or pitched as K2 writes it (``pitched``) when the variant for ``N`` can read a pitch
        (:data:`~._config.PITCHED_VARIANTS`).
        """
        if x.dim() != 4 or x.dtype != _SUPPORTED_DTYPE or not x.is_cuda or not x.is_contiguous():
            return False
        B, N, N_j, C = x.shape
        if not (
            B > 0
            and 0 < N == N_j
            and C == self.C
            and N % TOKEN_ALIGN == 0
            and N * N <= MAX_LAUNCH_ROWS
            and prod.dtype == _SUPPORTED_DTYPE
            and prod.is_cuda
            and prod.shape == (B, self.D, N, N)
        ):
            return False
        pitch, plane = prod.stride(2), prod.stride(1)
        if pitch != N and self.select(N).kernel_variant not in PITCHED_VARIANTS:
            return False
        # Rows pitch elements apart and channels N rows apart (B > 1: batch items D channels apart), every TMA
        # stride and the base a whole 16 bytes; a dense product has pitch N.
        return (
            prod.stride(3) == 1
            and pitch >= N
            and pitch % TOKEN_ALIGN == 0
            and plane == N * pitch
            and (B == 1 or prod.stride(0) == self.D * plane)
            and prod.data_ptr() % 16 == 0
        )

    def __call__(
        self,
        prod: torch.Tensor,
        x: torch.Tensor,
        fold: TrimulKFOutputFold,
        stats: torch.Tensor | None,
        eps: float,
        *,
        residual: bool = False,
        actual_seqlen: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The TriMul update of ``x``, or ``(x + update) * mask`` with ``residual``.

        Args:
            prod: bf16 ``[B, D, N, N]`` product from ``trimul_kf_k2``: dense, or written ``pitched`` for a
                variant that can read a pitch (the chain does where ``reads_pitch``).
            x: bf16 ``[B, N, N, C]`` input pair representation, ``N`` a multiple of 8.
            fold: The weights folded by :func:`fold_output_weights`.
            stats: The row statistics K1 returned; required exactly when ``select(N).reads_stats``.
            eps: The LayerNorms' epsilon.
            residual: Fuse ``(x + update) * mask``.
            actual_seqlen: int32 prefix lengths, one per ``(b, i)`` output row; required exactly
                with ``residual``.
            out: Optional bf16 output shaped like ``x``; it must not alias ``x``.
        """
        if not self.accepts(prod, x):
            raise ValueError(
                f"trimul KF K3 needs non-empty contiguous bf16 CUDA x [B, N, N, {self.C}] with N a multiple of {TOKEN_ALIGN} "
                f"and prod [B, {self.D}, N, N], dense or pitched as K2 writes it for a variant that reads a pitch"
            )
        B, N = x.shape[0], x.shape[1]
        selection = self.select(N)
        if (stats is not None) != selection.reads_stats:
            raise ValueError(
                f"trimul KF K3 {selection.kernel_variant} {'reads' if selection.reads_stats else 'takes no'} "
                "row statistics; pair it with the K1 variant selected for the same token count"
            )
        if stats is not None and (stats.dtype != torch.float32 or stats.shape != (B, stats_rows(N), 2)):
            raise ValueError("trimul KF K3 needs K1's fp32 row statistics for the same batch and token count")
        if residual != (actual_seqlen is not None):
            raise ValueError("trimul KF K3 masks the fused residual: pass actual_seqlen exactly with residual")
        if actual_seqlen is not None and (
            actual_seqlen.dtype != torch.int32 or not actual_seqlen.is_contiguous() or actual_seqlen.numel() != B * N
        ):
            raise ValueError("trimul KF K3 needs int32 actual_seqlen [B, N]")
        if fold.C != self.C or fold.D != self.D:
            raise ValueError(f"trimul KF K3 needs the C={self.C}, D={self.D} fold")
        if out is None:
            out = torch.empty_like(x)
        elif out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
            raise ValueError("trimul KF K3 output must be a contiguous bf16 tensor shaped like x")
        elif out.data_ptr() < x.data_ptr() + x.nbytes and x.data_ptr() < out.data_ptr() + out.nbytes:
            # K3 reads x while it writes out, so the two must not share any bytes.
            raise ValueError("trimul KF K3 output must not alias x")
        return self.backend.run(prod, x, *fold.weights(selection), stats, actual_seqlen, float(eps), selection, out)


def get_trimul_kf_k3_op(
    dtype: torch.dtype | None, dim: int, hidden_dim: int, residual: bool = True
) -> TrimulKFK3Op | None:
    """Return K3 when this build ships it for ``(dim, hidden_dim)`` and the residual setting, else ``None``."""
    if dtype != _SUPPORTED_DTYPE or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    backend = _trimul_kf_k3_instances.get(sm_version)
    if backend is None:
        backend = _trimul_kf_k3_instances.setdefault(sm_version, TrimulKFK3CuTe(sm_version))
    if not backend.ships(dim, hidden_dim, residual):
        return None
    return TrimulKFK3Op(backend, dim, hidden_dim)
