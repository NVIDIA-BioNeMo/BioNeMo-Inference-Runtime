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

from ..trimul_kf_k1.cutedsl import stats_rows
from ._config import TrimulKFK3Selection
from .cutedsl import TrimulKFK3CuTe

_SUPPORTED_DTYPE = torch.bfloat16
#: Token counts must keep every channel-major row 16-byte aligned for TMA.
TOKEN_ALIGN = 8
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_trimul_kf_k3_instances: dict[int, TrimulKFK3CuTe] = {}


@dataclass(frozen=True)
class TrimulKFOutputFold:
    """Both LayerNorms folded into K3's weights; see :func:`fold_output_weights`.

    Attributes:
        w_out: bf16 ``[C, D]`` output projection scaled by the output LayerNorm weight.
        w_gate: bf16 ``[C, C]`` gate projection scaled by the input LayerNorm weight.
        vec: fp32 ``[4C]``: for every output channel, the sum of its folded projection row, then
            ``W_out @ ln_out_bias + p_out_bias``, then the same two terms of the gate.
    """

    w_out: torch.Tensor
    w_gate: torch.Tensor
    vec: torch.Tensor
    C: int = field(init=False)
    D: int = field(init=False)
    device: torch.device = field(init=False)

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


def fold_output_weights(
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    p_out_bias: torch.Tensor | None = None,
    g_out_bias: torch.Tensor | None = None,
) -> TrimulKFOutputFold:
    """Fold the output LayerNorm into the output projection and the input LayerNorm into the gate.

    The gate reads ``LayerNorm_in(x)``, so it takes the input LayerNorm's parameters. Any output
    biases fold into the bias terms. The fold is fixed for fixed weights.

    Args:
        norm_out_weight: ``[D]`` output LayerNorm weight.
        norm_out_bias: ``[D]`` output LayerNorm bias.
        norm_in_weight: ``[C]`` input LayerNorm weight.
        norm_in_bias: ``[C]`` input LayerNorm bias.
        p_out_weight: ``[C, D]`` output projection weight.
        g_out_weight: ``[C, C]`` gate weight.
        p_out_bias: Optional ``[C]`` output projection bias.
        g_out_bias: Optional ``[C]`` gate bias.
    """

    def fold(
        weight: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = weight.float()
        folded = (weight * gamma.float()).to(torch.bfloat16)
        shift = weight @ beta.float()
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

    def select(self, n: int) -> TrimulKFK3Selection:
        """The anchor and variant a call over ``n`` tokens runs, including whether it reads K1's statistics."""
        selection = self._selections.get(n)
        if selection is None:
            selection = self.backend.select(self.C, self.D, n)
            if selection is None:
                raise ValueError(f"trimul KF K3 ships no config for C={self.C}, D={self.D}")
            self._selections[n] = selection
        return selection

    def accepts(self, prod: torch.Tensor, x: torch.Tensor) -> bool:
        """Whether a call on ``prod`` ``[B, D, N, N]`` and ``x`` ``[B, N, N, C]`` can run."""
        return (
            x.dim() == 4
            and x.numel() > 0
            and prod.dim() == 4
            and x.is_cuda
            and prod.device == x.device
            and x.dtype == prod.dtype == _SUPPORTED_DTYPE
            and x.is_contiguous()
            and prod.is_contiguous()
            and x.shape[1] == x.shape[2]
            and x.shape[1] % TOKEN_ALIGN == 0
            and x.shape[3] == self.C
            and prod.shape == (x.shape[0], self.D, x.shape[1], x.shape[2])
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
            prod: bf16 ``[B, D, N, N]`` product from ``trimul_kf_k2``.
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
                f"and prod [B, {self.D}, N, N]"
            )
        B, N = x.shape[0], x.shape[1]
        selection = self.select(N)
        if (stats is not None) != selection.reads_stats:
            raise ValueError(
                f"trimul KF K3 {selection.kernel_variant} {'reads' if selection.reads_stats else 'takes no'} "
                "row statistics; pair it with the K1 variant selected for the same token count"
            )
        if stats is not None and (
            stats.shape != (B, stats_rows(N), 2) or stats.dtype != torch.float32 or stats.device != x.device
        ):
            raise ValueError("trimul KF K3 needs K1's fp32 row statistics for the same batch and token count")
        if residual != (actual_seqlen is not None):
            raise ValueError("trimul KF K3 masks the fused residual: pass actual_seqlen exactly with residual")
        if actual_seqlen is not None and (
            actual_seqlen.dtype != torch.int32
            or actual_seqlen.device != x.device
            or not actual_seqlen.is_contiguous()
            or actual_seqlen.numel() != B * N
        ):
            raise ValueError("trimul KF K3 needs int32 actual_seqlen [B, N] on the input's device")
        if fold.C != self.C or fold.D != self.D or fold.device != x.device:
            raise ValueError(f"trimul KF K3 needs the C={self.C}, D={self.D} fold on {x.device}")
        if out is None:
            out = torch.empty_like(x)
        elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
            raise ValueError("trimul KF K3 output must be a contiguous bf16 tensor shaped like x")
        elif out.data_ptr() < x.data_ptr() + x.nbytes and x.data_ptr() < out.data_ptr() + out.nbytes:
            # K3 reads x while it writes out, so the two must not share any bytes.
            raise ValueError("trimul KF K3 output must not alias x")
        return self.backend.run(
            prod, x, fold.w_out, fold.w_gate, fold.vec, stats, actual_seqlen, float(eps), selection, out
        )


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
