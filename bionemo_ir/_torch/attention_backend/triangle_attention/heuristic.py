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
"""Shape-based router for optimized triangle-attention backends."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bionemo_ir.utils import DEBUG_ASSERTS

from ..interface import AttentionBackend
from .claude_kit import (
    ClaudeKitTriangleAttentionMetadata,
    ClaudeKitTriangleAttentionSM90D32,
)
from .cuequiv import CuEquivAttention, cuequiv_supports
from .cutedsl import TriangleAttentionCuTeLeftMask

_KERNEL_LIBRARY_MODULE = "bionemo_ir.libs._cutedsl_kernels"
# ClaudeKit's fixed per-call cost pays off only past this many attention scores,
# batch * i_dim * tokens**2 * num_heads. Keep in sync with kClaudeKitMinScores.
_CLAUDE_KIT_MIN_SCORES = 23_000_000
_CLAUDE_KIT = "ClaudeKit"
_CUTEDSL = "CuTeDSL"
_CUEQUIV = "CUEQUIV"
_SDPA = "SDPA"
_UNSUPPORTED = "unsupported"
# Backends that take an additive row mask and an unpadded pair bias instead of per-row lengths.
_DENSE_MASK_BACKENDS = frozenset({_CUEQUIV, _SDPA})
# The default pair-mask precompute masks with -1e9; halving the fp16 maximum keeps a row with no key finite.
_MASK_INF = 1e9
# Byte cap on the summed row mask and pair bias of one SDPA row chunk.
_SDPA_MASK_BYTES = 1 << 30


class HeuristicTriangleAttentionMetadata(ClaudeKitTriangleAttentionMetadata):
    """Metadata accepted by every optimized delegate."""


@dataclass(frozen=True)
class _RouteKey:
    device_index: int
    dtype: torch.dtype
    batch: int
    tokens: int
    i_dim: int
    has_lse: bool


def _cutedsl_supports(sm_version: int, dtype: torch.dtype, head_dim: int, tokens: int, i_dim: int) -> bool:
    if dtype not in (torch.float16, torch.bfloat16) or tokens <= 0 or i_dim <= 0:
        return False
    if sm_version not in (80, 86, 89, 90, 100, 103):
        return False
    padded_head_dim = ((head_dim + 31) // 32) * 32
    supported_head_dims = {32, 64, 128} if sm_version in (100, 103) else {32, 64, 128, 256}
    return head_dim > 0 and padded_head_dim in supported_head_dims


def _python_policy(
    sm_version: int,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    batch: int,
    tokens: int,
    i_dim: int,
) -> str:
    if (
        sm_version == 90
        and dtype == torch.bfloat16
        and head_dim == 32
        and min(num_heads, batch, tokens, i_dim) > 0
        and batch * i_dim * tokens * tokens * num_heads >= _CLAUDE_KIT_MIN_SCORES
    ):
        return _CLAUDE_KIT
    if _cutedsl_supports(sm_version, dtype, head_dim, tokens, i_dim):
        return _CUTEDSL
    return _UNSUPPORTED


def _native_policy(
    sm_version: int,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    batch: int,
    tokens: int,
    i_dim: int,
    has_lse: bool,
) -> str | None:
    """Ask the optional native policy module, or return ``None`` if absent."""
    try:
        library = importlib.import_module(_KERNEL_LIBRARY_MODULE)
    except (ImportError, OSError):
        return None
    policy = getattr(library, "heuristic", None)
    if policy is None:
        return None

    dtype_map = {
        torch.float16: policy.DType.FLOAT16,
        torch.bfloat16: policy.DType.BFLOAT16,
        torch.float32: policy.DType.FLOAT32,
    }
    policy_dtype = dtype_map.get(dtype)
    if policy_dtype is None:
        return _UNSUPPORTED

    implementation = policy.select_triangle_attention(
        target_sm=sm_version,
        head_dim=head_dim,
        num_heads=num_heads,
        batch=batch,
        tokens=tokens,
        i_dim=i_dim,
        dtype=policy_dtype,
        has_lse=has_lse,
    )
    implementations = policy.TriangleAttentionImplementation
    if implementation == implementations.CLAUDE_KIT:
        return _CLAUDE_KIT
    if implementation == implementations.CUTEDSL:
        return _CUTEDSL
    return _UNSUPPORTED


def _logical_triangle_shape(q: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(batch, tokens, i_dim)`` of ``[B, I, J, H*D]`` or ``[B, I, J, H, D]`` Q."""
    if q.ndim not in (4, 5):
        raise ValueError(f"Q must be [B, I, J, H*D] or [B, I, J, H, D]; got ndim={q.ndim}")
    return q.shape[0], q.shape[2], q.shape[1]


def _normalize_left_mask_biases(biases: list[torch.Tensor] | None) -> list[torch.Tensor] | None:
    """Convert legacy additive row masks to left-mask lengths."""
    if biases is None or not biases:
        return biases
    mask = biases[0]
    if mask.ndim <= 2 or not mask.is_floating_point():
        return biases

    if mask.ndim == 3:
        valid = mask > 0.5
    else:
        valid = mask >= 0
    # A row's length is only right for a ``1...1 0...0`` row. The check reads the mask back to the
    # host, which capture forbids, so like the CuTeDSL pair-mask precompute it is debug-only
    # (``BIOIR_DEBUG_ASSERTS=1``).
    if DEBUG_ASSERTS and not (mask.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not torch.all(valid[..., :-1] >= valid[..., 1:]):
            raise ValueError(
                "Heuristic triangle attention requires a left-aligned (``1...1 0...0``) row mask; got interior zeros"
            )
    actual_s_kv = valid.sum(dim=-1, dtype=torch.int32)
    while actual_s_kv.ndim > 2 and actual_s_kv.shape[-1] == 1:
        actual_s_kv = actual_s_kv.squeeze(-1)
    return [actual_s_kv.contiguous(), *biases[1:]]


def _dense_mask_biases(
    biases: list[torch.Tensor] | None, batch: int, tokens: int, i_dim: int, dtype: torch.dtype
) -> list[torch.Tensor]:
    """Rebuild the ``[B, I, 1, 1, J]`` additive row mask and the ``[B, H, J, J]`` pair bias.

    The dense-mask backends take these where the left-mask kernels take per-row
    lengths and a pair bias padded along the key axis.
    """
    if biases is None or len(biases) < 2 or biases[1] is None:
        raise ValueError("Heuristic triangle attention expects biases=[actual_s_kv, pair_bias]")
    lengths, pair_bias = _normalize_left_mask_biases(biases)[:2]
    lengths = lengths.reshape(batch, -1).expand(batch, i_dim)
    masked = torch.arange(tokens, device=lengths.device) >= lengths[..., None]
    mask_bias = masked.to(dtype).mul_(-min(_MASK_INF, torch.finfo(dtype).max / 2))[..., None, None, :]
    return [mask_bias, pair_bias[..., :tokens]]


class _RowChunkedSDPA(AttentionBackend):
    """SDPA over chunks of query rows with 4-D operands.

    One SDPA call over ``[B, I, H, J, D]`` runs the math kernel, which holds the
    summed ``[B, I, H, J, J]`` mask and bias plus fp32 scores. Per chunk of rows,
    4-D operands let SDPA pick a fused kernel, and the chunk's mask stays under
    ``_SDPA_MASK_BYTES``; a row larger than that is split along its query
    positions.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: HeuristicTriangleAttentionMetadata | None = None,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attend ``[B, I, J, H*D]`` operands under ``biases=[[B, I, 1, 1, J] mask, [B, H, J, J] bias]``.

        Returns:
            ``[B, I, J, H, D]``, written to ``output`` when given.
        """
        mask_bias, pair_bias = biases
        batch, i_dim, tokens, _ = q.shape
        shape = (batch, i_dim, tokens, self.num_heads, self.head_dim)
        result = torch.empty(shape, dtype=q.dtype, device=q.device) if output is None else output.view(shape)
        # [B, I, J, H*D] -> [B, I, H, J, D]
        q, k, v = (operand.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) for operand in (q, k, v))
        query_bytes = max(1, self.num_heads * tokens * q.element_size())  # One query position's mask.
        rows = max(1, _SDPA_MASK_BYTES // (query_bytes * max(1, tokens)))
        queries = max(1, tokens if rows > 1 else min(tokens, _SDPA_MASK_BYTES // query_bytes))
        for b in range(batch):
            for start in range(0, i_dim, rows):
                chunk = slice(start, start + rows)
                for query_start in range(0, tokens, queries):
                    positions = slice(query_start, query_start + queries)
                    attn_mask = mask_bias[b, chunk] + pair_bias[b, :, positions]
                    attended = F.scaled_dot_product_attention(
                        q[b, chunk, :, positions], k[b, chunk], v[b, chunk], attn_mask=attn_mask
                    )
                    result[b, chunk, positions] = attended.transpose(-3, -2)
        return result


class HeuristicTriangleAttention(AttentionBackend[HeuristicTriangleAttentionMetadata]):
    """Route triangle attention to Claude-kit or CuTeDSL by static shape.

    SM90 BF16 D=32 calls of at least 23M attention scores
    (``batch * i_dim * tokens**2 * num_heads``) use ``ClaudeKit`` when its
    native launcher is installed; every other CuTeDSL-supported call uses
    ``CuTeDSL``. A call neither kernel serves, such as a head dimension above
    128 on SM100 and SM103, runs on ``CUEQUIV`` where its kernels take the
    head dimension and key count, else on SDPA over row chunks, and raises a
    ``RuntimeWarning`` once per layer and route. That route skips the LSE
    store. Routes are chosen on the first eager call per shape and reused
    during CUDA graph capture.

    Args:
        layer_idx: The index of the attention layer.
        num_heads: The number of attention heads.
        head_dim: The dimension of each attention head.
        num_kv_heads: The number of key-value heads; must equal ``num_heads``.
    """

    Metadata = HeuristicTriangleAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        if self.num_heads != self.num_kv_heads:
            raise ValueError("num_heads must be equal to num_kv_heads")
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        self._sm_version = capability[0] * 10 + capability[1]
        self._route_cache: dict[_RouteKey, str] = {}
        self._delegates: dict[str, AttentionBackend] = {}
        self._last_backend_name: str | None = None
        self._warned_fallbacks: set[str] = set()

    @property
    def last_backend_name(self) -> str | None:
        """Return the delegate selected by the most recent call."""
        return self._last_backend_name

    def _create_delegate(self, backend_name: str) -> AttentionBackend:
        if backend_name == _CLAUDE_KIT:
            backend_cls = ClaudeKitTriangleAttentionSM90D32
        elif backend_name == _CUTEDSL:
            backend_cls = TriangleAttentionCuTeLeftMask
        elif backend_name == _CUEQUIV:
            backend_cls = CuEquivAttention
        elif backend_name == _SDPA:
            backend_cls = _RowChunkedSDPA
        else:
            raise ValueError(f"Unsupported heuristic triangle-attention delegate {backend_name!r}")
        return backend_cls(self.layer_idx, self.num_heads, self.head_dim, self.num_kv_heads)

    def _select_backend_name(self, key: _RouteKey) -> str:
        native_choice = _native_policy(
            self._sm_version,
            key.dtype,
            self.head_dim,
            self.num_heads,
            key.batch,
            key.tokens,
            key.i_dim,
            key.has_lse,
        )
        if native_choice is not None:
            choice = native_choice
        else:
            choice = _python_policy(
                self._sm_version, key.dtype, self.head_dim, self.num_heads, key.batch, key.tokens, key.i_dim
            )

        cutedsl_supported = _cutedsl_supports(
            self._sm_version,
            key.dtype,
            self.head_dim,
            key.tokens,
            key.i_dim,
        )
        if choice == _CLAUDE_KIT:
            if ClaudeKitTriangleAttentionSM90D32.native_available():
                return choice
            if cutedsl_supported:
                return _CUTEDSL
        elif choice == _CUTEDSL and cutedsl_supported:
            return choice

        # Keep CuTeDSL's padded head dimensions when the native policy only knows exact ones.
        if cutedsl_supported:
            return _CUTEDSL
        fallback = (
            _CUEQUIV
            if cuequiv_supports(key.dtype, self.head_dim, sm_version=self._sm_version, keys=key.tokens)
            else _SDPA
        )
        if fallback not in self._warned_fallbacks:
            self._warned_fallbacks.add(fallback)
            warnings.warn(
                f"Heuristic triangle attention: no ClaudeKit or CuTeDSL kernel for SM{self._sm_version}, "
                f"dtype={key.dtype}, head_dim={self.head_dim}; layer {self.layer_idx} runs on {fallback}",
                RuntimeWarning,
                stacklevel=2,
            )
        return fallback

    def _delegate(self, q: torch.Tensor, output_lse: torch.Tensor | None) -> tuple[str, AttentionBackend]:
        """Return this call's route and its delegate; concurrent calls may change ``last_backend_name``."""
        batch, tokens, i_dim = _logical_triangle_shape(q)
        key = _RouteKey(
            device_index=q.device.index if q.device.index is not None else -1,
            dtype=q.dtype,
            batch=batch,
            tokens=tokens,
            i_dim=i_dim,
            has_lse=output_lse is not None,
        )
        backend_name = self._route_cache.get(key)
        if backend_name is None:
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Heuristic triangle-attention route was not cached before CUDA capture; "
                    "run one eager warmup call for this shape first"
                )
            backend_name = self._select_backend_name(key)
            delegate = self._delegates.get(backend_name)
            if delegate is None:
                delegate = self._create_delegate(backend_name)
                self._delegates[backend_name] = delegate
            self._route_cache[key] = backend_name
        else:
            delegate = self._delegates[backend_name]
        self._last_backend_name = backend_name
        return backend_name, delegate

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: HeuristicTriangleAttentionMetadata | None = None,
        output: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the cached delegate.

        An optimized delegate gets the inputs and buffers unchanged. A
        dense-mask delegate gets the row mask and pair bias it expects and its
        default metadata, and its result is written to ``output`` when given.
        Heuristic masks are left-aligned, so cuEquivariance always gets
        ``use_kv_lengths=True``.
        """
        backend_name, delegate = self._delegate(q, output_lse)
        if backend_name in _DENSE_MASK_BACKENDS:
            batch, tokens, i_dim = _logical_triangle_shape(q)
            dense_biases = _dense_mask_biases(biases, batch, tokens, i_dim, q.dtype)
            if q.ndim == 5:  # [B, I, J, H, D] -> [B, I, J, H*D]
                q, k, v = (operand.flatten(-2) for operand in (q, k, v))
            if backend_name == _SDPA:
                return delegate.forward(q, k, v, biases=dense_biases, output=output)
            result = delegate.forward(q, k, v, biases=dense_biases, **{**kwargs, "use_kv_lengths": True})
            if output is None:
                return result
            return output.view(result.shape).copy_(result)
        return delegate.forward(
            q,
            k,
            v,
            biases=_normalize_left_mask_biases(biases),
            metadata=metadata,
            output=output,
            output_lse=output_lse,
            **kwargs,
        )
