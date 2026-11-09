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
"""Triton kernel: the outer-product-mean operands in one pass over the MSA.

The outer-product-mean layer normalizes the MSA, projects it to ``a`` and ``b`` and masks both::

    m [B, S, I, K] -> LayerNorm -> m W^T + bias -> [B, S, I, 2C] -> split -> a, b (each * mask)

Eagerly that is a LayerNorm, a GEMM and two strided multiplies, each a full HBM round trip; PyTorch's
LayerNorm also runs one block per ``K``-wide row, which for ``K = 64`` costs as much as the OPM itself at
deep MSAs. This kernel reads ``m`` once and writes ``a`` and ``b`` once. The rounding points match the eager
chain: the normalized row and the projection are rounded to the input dtype, and the mask product is
rounded once more.

``b`` is written ``[B, S, I, C]``. ``a`` is written either the same way or sequence-last (see
:func:`empty_sequence_last`), the layout the SM100 outer-product-mean kernel loads with one TMA box per
chunk; in the layer this kernel is its only producer.
"""

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

__all__ = ["OpmPrologue"]

# Token tile (BLOCK_S sequences x BLOCK_I tokens) and warps per layout, picked on B200 at
# P S173 N960 and S1024 N768. A sequence-last store wants many sequences of one token; a
# plain store wants many tokens of one sequence.
_TILES = {False: (1, 128, 4), True: (256, 1, 4)}


@triton.jit(do_not_specialize=["S", "I", "I_PAD", "PITCH", "eps"])
def _opm_prologue_kernel(
    X,
    LN_W,
    LN_B,
    W,
    BIAS,
    MASK,
    A,
    B_OUT,
    S,
    I,
    I_PAD,
    PITCH,
    eps,
    K: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SEQUENCE_LAST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    BLOCK_M: tl.constexpr = BLOCK_S * BLOCK_I
    batch = tl.program_id(2).to(tl.int64)
    r = tl.arange(0, BLOCK_M)
    s = tl.program_id(0) * BLOCK_S + r // BLOCK_I
    i = tl.program_id(1) * BLOCK_I + r % BLOCK_I
    valid = (s < S) & (i < I)
    token = batch * S * I + s.to(tl.int64) * I + i
    k = tl.arange(0, K)
    x = tl.load(X + token[:, None] * K + k[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    x = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / K + eps)
    x = x * rstd[:, None] * tl.load(LN_W + k).to(tl.float32)[None, :] + tl.load(LN_B + k).to(tl.float32)[None, :]
    x = x.to(X.dtype.element_ty)
    n = tl.arange(0, C)
    # W is [2C, K]: rows [0, C) project a, rows [C, 2C) project b.
    ya = tl.dot(x, tl.load(W + n[None, :] * K + k[:, None]))
    yb = tl.dot(x, tl.load(W + (n[None, :] + C) * K + k[:, None]))
    if HAS_BIAS:
        ya += tl.load(BIAS + n).to(tl.float32)[None, :]
        yb += tl.load(BIAS + C + n).to(tl.float32)[None, :]
    m = tl.load(MASK + token, mask=valid, other=0.0).to(tl.float32)[:, None]
    a = (ya.to(X.dtype.element_ty).to(tl.float32) * m).to(X.dtype.element_ty)
    b = (yb.to(X.dtype.element_ty).to(tl.float32) * m).to(X.dtype.element_ty)
    tl.store(B_OUT + token[:, None] * C + n[None, :], b, mask=valid[:, None])
    if SEQUENCE_LAST:
        # Channel n of token i -> row ((n // CG * I_PAD / T + i // T) * CG + n % CG) * T + i % T.
        row = (((n[None, :] // CG) * (I_PAD // T) + i[:, None] // T) * CG + n[None, :] % CG) * T + i[:, None] % T
        tl.store(A + batch * I_PAD * C * PITCH + row.to(tl.int64) * PITCH + s[:, None], a, mask=valid[:, None])
    else:
        tl.store(A + token[:, None] * C + n[None, :], a, mask=valid[:, None])


#: Row pitch of the sequence-last ``a`` in elements: 128 bytes of 16-bit values. The SM100 kernel needs a multiple
#: of 8.
SEQUENCE_LAST_PITCH = 64


def empty_sequence_last(
    B: int, S: int, I: int, C: int, cc: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """An unwritten sequence-last ``a``: ``[B, C / cc, ceil(I / T), cc, T, S]`` with ``T = 128 / cc``.

    ``s`` is contiguous, a chunk's ``cc`` channels of ``T`` tokens are consecutive rows, and rows start every
    ``ceil(S / 64) * 64`` elements. Pad tokens and the columns past ``S`` are never read as data.
    """
    T = 128 // cc
    pitch = -(-S // SEQUENCE_LAST_PITCH) * SEQUENCE_LAST_PITCH
    return torch.empty(B, C // cc, -(-I // T), cc, T, pitch, dtype=dtype, device=device)[..., :S]


def _groups(C: int, chunk: int | None) -> tuple[int, int]:
    """``(CG, T)`` of ``a``'s layout: channels per chunk and tokens per row group (plain: all ``C`` channels, 1)."""
    return (C, 1) if chunk is None else (chunk, 128 // chunk)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


class OpmPrologue(TritonKernelCache):
    """LayerNorm, the fused ``a``/``b`` projection, the mask and the split, as one kernel."""

    _global_cache: dict[tuple, CachedKernel] = {}

    @staticmethod
    def serves(dtype: torch.dtype, K: int, c_hidden: int) -> bool:
        """Whether the kernel serves a 16-bit ``m`` of ``K`` channels (a power of two) projected to ``c_hidden``."""
        return (
            dtype in (torch.float16, torch.bfloat16)
            and 16 <= K <= 256
            and K & (K - 1) == 0
            and c_hidden in (16, 32, 64)
        )

    @staticmethod
    def supports(m: torch.Tensor, mask: torch.Tensor, c_hidden: int) -> bool:
        """Whether the kernel serves ``m [B, S, I, K]`` and ``mask [B, S, I]`` (16-bit, power-of-two ``K``)."""
        return (
            m.is_cuda
            and m.dim() == 4
            and mask.dtype == m.dtype
            and mask.shape == m.shape[:-1]
            and OpmPrologue.serves(m.dtype, m.shape[-1], c_hidden)
        )

    def prepare(self, dtype: torch.dtype, K: int, c_hidden: int, has_bias: bool, chunks: tuple[int, ...] = ()) -> None:
        """Compile the variant that writes ``a`` plain and one per sequence-last ``chunk`` now, not at a first call.

        The layout a call takes follows the backend's routing for its shape, so a model that prepares every layout
        it can be routed to compiles nothing mid-inference. Does nothing without CUDA or for unserved operands.
        """
        if torch.cuda.is_available() and self.serves(dtype, K, c_hidden):
            for chunk in (None, *chunks):
                self._kernel(dtype, K, c_hidden, has_bias, chunk)

    def _kernel(self, dtype: torch.dtype, K: int, C: int, has_bias: bool, chunk: int | None):
        CG, T = _groups(C, chunk)
        seq_last = chunk is not None
        key = (torch.cuda.current_device(), dtype, K, C, CG, T, has_bias, seq_last)
        kernel = OpmPrologue._global_cache.get(key)
        if kernel is None:
            block_s, block_i, warps = _TILES[seq_last]
            kernel = self.compile_for_dtypes(
                _opm_prologue_kernel,
                dtypes=[dtype],
                make_dummy_args=lambda dt: (
                    *(torch.empty(1, dtype=dt, device="cuda") for _ in range(8)),
                    2,
                    2,
                    2,
                    2,
                    1e-5,
                ),
                grid=(0,),
                K=K,
                C=C,
                CG=CG,
                T=T,
                HAS_BIAS=has_bias,
                SEQUENCE_LAST=seq_last,
                BLOCK_S=block_s,
                BLOCK_I=block_i,
                num_warps=warps,
            )[dtype]
            OpmPrologue._global_cache[key] = kernel
        return kernel

    def __call__(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize, project, mask and split.

        Args:
            m: MSA ``[B, S, I, K]``, FP16 or BF16.
            mask: ``[B, S, I]`` in ``m``'s dtype.
            ln_weight: LayerNorm scale ``[K]``.
            ln_bias: LayerNorm shift ``[K]``.
            eps: LayerNorm epsilon.
            weight: Fused projection ``[2C, K]``; rows ``[0, C)`` give ``a``.
            bias: Optional projection bias ``[2C]``.
            chunk: ``None`` writes ``a`` as ``[B, S, I, C]``. Otherwise ``a`` is written sequence-last with
                ``chunk`` channels per chunk, as :func:`empty_sequence_last` lays it out.

        Returns:
            ``(a, b)``; ``b`` is ``[B, S, I, C]``.
        """
        if not self.supports(m, mask, weight.shape[0] // 2):
            raise ValueError("OpmPrologue takes FP16/BF16 m [B, S, I, K] (K a power of two) and a matching mask")
        B, S, I, K = m.shape
        C = weight.shape[0] // 2
        if weight.shape != (2 * C, K) or (bias is not None and bias.shape != (2 * C,)):
            raise ValueError(f"OpmPrologue projection must be [2C, {K}] with a [2C] bias")
        if chunk is not None and (C % chunk or 128 % chunk):
            raise ValueError("OpmPrologue chunk must divide C and 128")
        # The cached kernel assumes 16-byte aligned, contiguous operands.
        x, ln_w, ln_b, w, mk = (_aligned(t.to(m.dtype)) for t in (m, ln_weight, ln_bias, weight, mask))
        has_bias = bias is not None
        bv = _aligned(bias.to(m.dtype)) if has_bias else w
        b = torch.empty(B, S, I, C, dtype=m.dtype, device=m.device)
        CG, T = _groups(C, chunk)
        if chunk is None:
            a = torch.empty_like(b)
            i_pad, pitch = I, 1
        else:
            a = empty_sequence_last(B, S, I, C, chunk, m.dtype, m.device)
            i_pad, pitch = a.shape[2] * T, a.stride(-2)
        if b.numel():
            block_s, block_i, _ = _TILES[chunk is not None]
            grid = (triton.cdiv(S, block_s), triton.cdiv(I, block_i), B)
            with torch.cuda.device(m.device):
                kernel = self._kernel(m.dtype, K, C, has_bias, chunk)
                ptrs = (x, ln_w, ln_b, w, bv, mk, a, b)
                if kernel.driver is not None:
                    kernel.driver.launch_with((*(t.data_ptr() for t in ptrs), S, I, i_pad, pitch, eps), *grid)
                else:
                    kernel.launch(
                        grid, *ptrs, S, I, i_pad, pitch, eps, K, C, CG, T, has_bias, chunk is not None, block_s, block_i
                    )
        return a, b
