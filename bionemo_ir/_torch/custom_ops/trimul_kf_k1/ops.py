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
"""Variant selection, input weight fold and PyTorch reference for the SM90 TriMul KF K1."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ._config import TrimulKFK1Selection, ab_pitch
from .cutedsl import MAX_LAUNCH_ROWS, TrimulKFK1CuTe, TrimulKFK1Output

_SUPPORTED_DTYPE = torch.bfloat16
#: Token counts must keep every channel-major row 16-byte aligned for TMA.
TOKEN_ALIGN = 8
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_trimul_kf_k1_instances: dict[int, TrimulKFK1CuTe] = {}
#: Passes of :func:`_centred_rows_bf16`; one bf16 step of one entry per row each.
_CENTRE_PASSES = 128


@dataclass(frozen=True)
class TrimulKFInputFold:
    """The input LayerNorm folded into K1's projection weights; see :func:`fold_input_weights`.

    Attributes:
        proj: bf16 ``[2D, C]`` projection rows scaled by the LayerNorm weight, read by ``K1_0``.
        gate: bf16 ``[2D, C]`` gate rows scaled the same way, read by ``K1_0``.
        interleaved: bf16 ``[4D, C]``: ``proj`` and ``gate`` in alternating 8-row blocks, read by
            the ping-pong variants.
        vec: fp32 ``[8D]``: for every output row, the sum of its folded projection row, then
            ``W_p @ ln_bias + p_bias``, then the same two terms of the gate.
        centred: Whether every folded row is centred (sums to about zero), as a K1 that drops the
            LayerNorm mean term needs; any K1 reads a centred fold correctly.
    """

    proj: torch.Tensor
    gate: torch.Tensor
    interleaved: torch.Tensor
    vec: torch.Tensor
    centred: bool = False
    C: int = field(init=False)
    D: int = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        rows, C = self.proj.shape
        D = rows // 2
        if (
            rows != 2 * D
            or self.gate.shape != (rows, C)
            or self.interleaved.shape != (2 * rows, C)
            or self.vec.shape != (8 * D,)
            or self.vec.dtype != torch.float32
            or any(w.dtype != _SUPPORTED_DTYPE for w in (self.proj, self.gate, self.interleaved))
            or any(
                not t.is_contiguous() or t.device != self.proj.device for t in (self.gate, self.interleaved, self.vec)
            )
            or not self.proj.is_contiguous()
        ):
            raise ValueError("TrimulKFInputFold needs the contiguous bf16 folds and fp32 terms of fold_input_weights")
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "device", self.proj.device)


def _centred_rows_bf16(w: torch.Tensor, passes: int = _CENTRE_PASSES) -> torch.Tensor:
    """Every row of ``w`` minus its mean, in bf16, rounded so the row still sums to about zero.

    ``x @ centred.T == (x - mean(x)) @ w.T`` for any row ``x``, so a K1 reading the centred rows needs
    no LayerNorm mean term. Rounding each entry to nearest leaves a row sum of about ``sqrt(C)`` half
    steps, which the input's mean would multiply. Each pass moves one entry per row by one bf16 step
    against the remaining sum: the entry adding the least rounding error per unit of sum it removes
    (entries near a rounding midpoint first), and only a step under twice the sum. The pass count is
    fixed: no data-dependent control flow, so a fold under graph capture records no host synchronization.
    """
    exact = w.double() - w.double().mean(-1, keepdim=True)
    r = exact.to(torch.bfloat16)
    for _ in range(passes):
        rd = r.double()
        rest = rd.sum(-1, keepdim=True)
        # bf16 is sign-magnitude: one step of the bits moves away from zero; this lowers |rest|
        bits = r.view(torch.int16).to(torch.int32)
        moved = (bits + torch.where((rd > 0) == (rest > 0), -1, 1)).to(torch.int16).view(torch.bfloat16)
        md = moved.double()
        step = md - rd
        fits = (rd != 0) & torch.isfinite(md) & (step * rest < 0) & (step.abs() < 2 * rest.abs())
        cost = ((md - exact).abs() - (rd - exact).abs()) / step.abs().clamp_min(1e-300)
        best = torch.where(fits, cost, torch.inf).argmin(-1, keepdim=True)
        r = torch.where(fits.gather(-1, best), r.scatter(-1, best, moved.gather(-1, best)), r)
    return r.contiguous()


def fold_input_weights(
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor | None,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    p_in_bias: torch.Tensor | None = None,
    g_in_bias: torch.Tensor | None = None,
    centred: bool = False,
) -> TrimulKFInputFold:
    """Fold the input LayerNorm, and any projection biases, into K1's weights.

    ``LayerNorm(x) @ W.T + bias == rstd * (x @ (W * gamma).T - mean * rowsum(W * gamma)) + W @ beta + bias``,
    so K1 applies the LayerNorm from per-row statistics on raw ``x``. The fold is fixed for fixed
    weights. A missing LayerNorm bias is a zero beta. A centred fold also subtracts every folded row's
    mean, so ``rowsum`` is about zero and the mean term drops out: the fold a K1 whose op is
    :attr:`TrimulKFK1Op.centred` reads.

    Args:
        norm_in_weight: ``[C]`` LayerNorm weight ``gamma``.
        norm_in_bias: ``[C]`` LayerNorm bias ``beta``, or ``None`` when the norm learns none.
        p_in_weight: ``[2D, C]`` projection weight; rows ``[0, D)`` make ``a``, ``[D, 2D)`` make ``b``.
        g_in_weight: ``[2D, C]`` gate weight, rows matching ``p_in_weight``.
        p_in_bias: Optional ``[2D]`` projection bias.
        g_in_bias: Optional ``[2D]`` gate bias.
        centred: Centre every folded row (:func:`_centred_rows_bf16`).
    """
    gamma = norm_in_weight.float()
    beta = torch.zeros_like(gamma) if norm_in_bias is None else norm_in_bias.float()
    rows, C = p_in_weight.shape
    # one pass over the projection and gate rows together: centring costs the same for twice the rows
    scaled = torch.cat((p_in_weight, g_in_weight)).float() * gamma
    folded = _centred_rows_bf16(scaled) if centred else scaled.to(torch.bfloat16)

    def fold(
        folded: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift = weight.float() @ beta
        if bias is not None:
            shift = shift + bias.float()
        return folded.contiguous(), folded.float().sum(-1), shift

    proj, proj_sum, proj_shift = fold(folded[:rows], p_in_weight, p_in_bias)
    gate, gate_sum, gate_shift = fold(folded[rows:], g_in_weight, g_in_bias)
    interleaved = torch.stack((proj.view(-1, 8, C), gate.view(-1, 8, C)), dim=1).reshape(2 * rows, C)
    vec = torch.cat((proj_sum, proj_shift, gate_sum, gate_shift)).contiguous()
    return TrimulKFInputFold(proj, gate, interleaved, vec, centred)


@dataclass(frozen=True)
class TrimulKFK1Op:
    """K1 bound to one ``(C, D)``; see :func:`get_trimul_kf_k1_op`."""

    backend: TrimulKFK1CuTe
    C: int
    D: int
    _selections: dict[int, TrimulKFK1Selection] = field(default_factory=dict, init=False, repr=False, compare=False)

    def select(self, n: int) -> TrimulKFK1Selection:
        """The anchor, variant and a/b layout a call over ``n`` tokens runs: which fold it reads, whether it
        writes stats. A layout this build cannot write falls back to dense planes."""
        selection = self._selections.get(n)
        if selection is None:
            selection = self.backend.select(self.C, self.D, n)
            if selection is None:
                raise ValueError(f"trimul KF K1 ships no config for C={self.C}, D={self.D}")
            if not self.backend.supports(self.C, self.D, selection.kernel_variant, selection.ab_layout):
                selection = selection._replace(ab_layout="dense")
            self._selections[n] = selection
        return selection

    @property
    def centred(self) -> bool:
        """Whether a variant this op runs drops the LayerNorm mean term, so its fold must be centred
        (``fold_input_weights(..., centred=True)``)."""
        return self.backend.centred(self.C, self.D)

    def accepts(self, x: torch.Tensor, actual_seqlen: torch.Tensor) -> bool:
        """Whether a call on ``x`` ``[B, N, N, C]`` can run."""
        if x.dim() != 4 or x.dtype != _SUPPORTED_DTYPE or not x.is_cuda or not x.is_contiguous():
            return False
        B, N, N_j, C = x.shape
        return (
            B > 0
            and 0 < N == N_j
            and C == self.C
            and N % TOKEN_ALIGN == 0
            and N * ab_pitch(N, "padded") <= MAX_LAUNCH_ROWS
            and actual_seqlen.dtype == torch.int32
            and actual_seqlen.is_cuda
            and actual_seqlen.is_contiguous()
            and actual_seqlen.numel() == B * N
        )

    def __call__(
        self, x: torch.Tensor, actual_seqlen: torch.Tensor, fold: TrimulKFInputFold, eps: float
    ) -> TrimulKFK1Output:
        """``a``, ``b`` and the row statistics K3 reads, for the variant tuned for ``x``'s token count.

        Args:
            x: bf16 ``[B, N, N, C]`` pair representation, ``N`` a multiple of 8.
            actual_seqlen: int32 prefix lengths, one per ``(b, i)`` row of ``x``.
            fold: The input weights folded by :func:`fold_input_weights`, centred when :attr:`centred`.
            eps: The input LayerNorm's epsilon.

        Returns:
            Channel-major bf16 ``a`` and ``b`` ``[B, D, N, N]`` (views of ``[B, D, P, P]`` planes, ``P``
            the selection's row pitch), and fp32 row statistics when the variant hands them to K3
            (``select(N).writes_stats``), else ``None``.
        """
        if not self.accepts(x, actual_seqlen):
            raise ValueError(
                f"trimul KF K1 needs a non-empty contiguous bf16 CUDA x [B, N, N, {self.C}] with N a multiple of "
                f"{TOKEN_ALIGN} and int32 actual_seqlen [B, N] beside it"
            )
        if fold.C != self.C or fold.D != self.D:
            raise ValueError(f"trimul KF K1 needs the C={self.C}, D={self.D} fold")
        selection = self.select(x.shape[1])
        if selection.centred and not fold.centred:
            raise ValueError(f"trimul KF K1 {selection.kernel_variant} needs fold_input_weights(..., centred=True)")
        w_in, w_gate_in = (fold.interleaved, None) if selection.interleaved else (fold.proj, fold.gate)
        return self.backend.run(x, actual_seqlen, w_in, w_gate_in, fold.vec, float(eps), selection)


def get_trimul_kf_k1_op(dtype: torch.dtype | None, dim: int, hidden_dim: int) -> TrimulKFK1Op | None:
    """Return K1 when this build ships it for ``(dim, hidden_dim)`` on the current GPU, else ``None``.

    The kernels run on SM90 in bf16, from source in a private checkout and from packaged CUBINs
    otherwise.
    """
    if dtype != _SUPPORTED_DTYPE or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    backend = _trimul_kf_k1_instances.get(sm_version)
    if backend is None:
        backend = _trimul_kf_k1_instances.setdefault(sm_version, TrimulKFK1CuTe(sm_version))
    if not backend.ships(dim, hidden_dim):
        return None
    return TrimulKFK1Op(backend, dim, hidden_dim)
