# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable
from enum import IntEnum
from functools import lru_cache
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from bionemo_ir._torch.utils import CHUNK_REGISTRY, TRIANGLE_ATTENTION, ChunkPolicy, chunk_apply
from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import layer_norm_transpose
from bionemo_ir.runtime.buffers import PreallocatedBuffers
from bionemo_ir.utils import get_sm_version

from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import precompute_pair_masks
from ..custom_ops.dual_gemm_x0_x1 import get_dual_gemm_x0_x1_op
from ..custom_ops.dual_gemm_x_x import get_dual_gemm_x_x_op
from .attention import TriangleAttention


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


_CUEQ_TRIMUL_PAIR_DIM = 384
_CUEQ_TRIMUL_HIDDEN_DIM = 256
_CUEQ_TRIMUL_SEQUENCE_THRESHOLD = 256


@lru_cache(maxsize=1)
def _get_cueq_trimul_api() -> tuple[Callable[..., torch.Tensor], Callable[..., bool]] | None:
    """Return the optional internal cuEquivariance TriMul API."""
    try:
        module = import_module("cuequivariance_ops_torch")
    except (ImportError, OSError):
        return None
    operation = getattr(module, "triangle_multiplicative_update", None)
    is_supported = getattr(module, "triangle_multiplicative_update_is_supported", None)
    if not callable(operation) or not callable(is_supported):
        return None
    return operation, is_supported


class TriangleAttentionNodeType(IntEnum):
    STARTING = 0
    ENDING = 1


class TriangleMultiplicationNodeType(IntEnum):
    INCOMING = 0
    OUTGOING = 1


class TriangleAttentionNode(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        num_heads: int,
        node_type: TriangleAttentionNodeType = TriangleAttentionNodeType.STARTING,
        inf: float = 1e9,
        layer_idx: int = 0,
        dtype: torch.dtype = None,
        chunk_policy: ChunkPolicy | None = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
        mha_bias_flags: dict[str, bool] | None = None,
        pair_mask_left_aligned: bool = True,
        transposed_bias: bool = False,
    ):
        """
        Args:
            c_in (int): input channel dimension
            c_hidden (int): hidden channel dimension
            num_heads (int): number of attention heads
            node_type (TriangleAttentionNodeType): whether this is the starting node
            inf (float): infinity value
            dtype (torch.dtype): data type
            chunk_policy (ChunkPolicy | None): query-row chunking policy; ``None`` uses the
                shared ``triangle_attention`` policy from ``CHUNK_REGISTRY``.
            skip_create_weights (bool): whether to skip creating weights
            attn_backend (str): attention backend
            pair_mask_left_aligned: Whether the mask is prefix-shaped along
                both pair axes.
            transposed_bias: Build the shared triangle bias from the transposed
                pair representation. OpenFold-3 v0.5.0 adopted this for the
                ending node; AlphaFold-2, Boltz and Protenix do not.
                See https://github.com/aqlaboratory/openfold-3/commit/1baf2c71.
        """
        super().__init__()
        if mha_bias_flags is None:
            mha_bias_flags = {"q": False, "k": False, "v": False, "g": False, "o": False}
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.transposed_bias = transposed_bias
        self.inf = inf
        self.dtype = dtype
        self.attn_backend = attn_backend
        self.pair_mask_left_aligned = pair_mask_left_aligned
        # Query-row chunking policy (registry default unless overridden). Attention within each row
        # is independent, so row-chunking is numerically identical; bounds the [chunk, J, H, ...]
        # attention temporaries at large N.
        self.chunk_policy = chunk_policy if chunk_policy is not None else CHUNK_REGISTRY.get(TRIANGLE_ATTENTION)
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.linear = Linear(
            self.c_in,
            self.num_heads,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.mha = TriangleAttention(
            layer_idx=layer_idx,
            hidden_size=self.c_in,
            head_dim=c_hidden,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            gating=True,
            bias_flags=mha_bias_flags,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )

        self.J_padded_multiple = -1
        if self.attn_backend == "CuTeDSL":
            self.J_padded_multiple = 8
        self._ln_proj_moveaxis_pad = LNProjMoveaxisPad(
            D=self.c_in,
            H=self.num_heads,
            dtype=dtype or torch.bfloat16,
        )

    @staticmethod
    def _ensure_contiguous(x: torch.Tensor, mask_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention kernels require contiguous inputs"""
        if not x.is_contiguous():
            x = x.contiguous()
        if not mask_bias.is_contiguous():
            mask_bias = mask_bias.contiguous()
        return x, mask_bias

    def _ensure_dtype(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure the dtype of the input and mask"""
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if mask.dtype != self.dtype:
            mask = mask.to(self.dtype)
        return x, mask

    def _prep_bias(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute triangle bias and apply LayerNorm to x.

        Args:
            x: input tensor [B, I, J, c_in] (already transposed for ending node)

        Returns:
            (x_normed, triangle_bias):
                x_normed: [B, I, J, c_in] after LayerNorm
                triangle_bias: [B, H, I, J_padded] projected and transposed
        """
        x = self.layer_norm(x)
        # Transpose before projecting, not after: the pad below lands on the
        # key axis, so swapping the token axes afterwards would misplace it.
        bias_src = x.transpose(1, 2).contiguous() if self.transposed_bias else x
        triangle_bias = self._ln_proj_moveaxis_pad(
            bias_src,
            ln_weight=None,
            ln_bias=None,
            proj_weight=self.linear.weight,
            pad_multiple=self.J_padded_multiple,
            proj_z=self.linear,
        )
        return x, triangle_bias

    def _mha_slice(
        self,
        x: torch.Tensor,
        mask_bias: torch.Tensor,
        triangle_bias: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """Run MHA for a (possibly row-chunked) slice of ``x``; ``triangle_bias`` is shared."""
        return self.mha(
            x,
            biases=[mask_bias, triangle_bias],
            attn_metadata=attn_metadata,
            buffers=buffers,
            use_kv_lengths=self.pair_mask_left_aligned,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_bias: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the triangle attention node. Currently supports only
        batch_size = 1.

        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor | None): mask tensor [B, I, J].
                Ignored when *mask_bias* is provided.
            mask_bias (torch.Tensor | None): precomputed additive mask bias
                ([B, I, 1, 1, J] for starting, [B, J, 1, 1, I] for ending).
                When supplied, the per-layer mask->bias computation is skipped.
            attn_metadata (AttentionMetadata | None): attention metadata
            buffers: Shared pre-allocated buffer dict.
        """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if self.node_type == TriangleAttentionNodeType.ENDING:
            x = x.transpose(1, 2)

        if mask_bias is None:
            if mask is None:
                mask = x.new_ones(x.shape[:-1])
            if self.node_type == TriangleAttentionNodeType.ENDING:
                mask = mask.transpose(1, 2)
            precomputed = precompute_pair_masks(self.attn_backend, mask, inf=self.inf, dtype=self.dtype)
            mask_bias = precomputed.mask_bias

        x, triangle_bias = self._prep_bias(x)

        x, mask_bias = self._ensure_contiguous(x, mask_bias)
        # Row-chunk the query dim (mask_bias slices in lockstep; triangle_bias is shared across
        # rows so it passes through). ``chunk_apply`` falls back to a single dense call below the
        # policy threshold, so small N is unaffected.
        if self.chunk_policy is not None:
            output = chunk_apply(
                self._mha_slice,
                x,
                mask_bias,
                policy=self.chunk_policy,
                cat_dim=1,
                triangle_bias=triangle_bias,
                attn_metadata=attn_metadata,
                buffers=buffers,
            )
        else:
            output = self._mha_slice(x, mask_bias, triangle_bias, attn_metadata=attn_metadata, buffers=buffers)
        if self.node_type == TriangleAttentionNodeType.ENDING:
            output = output.transpose(2, 1)
        return output


class TriangleAttentionStartingNode(TriangleAttentionNode):
    def __init__(self, *args, **kwargs):
        kwargs["node_type"] = TriangleAttentionNodeType.STARTING
        super().__init__(*args, **kwargs)


class TriangleAttentionEndingNode(TriangleAttentionNode):
    def __init__(self, *args, **kwargs):
        kwargs["node_type"] = TriangleAttentionNodeType.ENDING
        super().__init__(*args, **kwargs)


class TriangleMultiplicationNode(nn.Module):
    def __init__(
        self,
        layer_idx: int = 0,
        dim: int = 128,
        hidden_dim: int | None = None,
        eps: float = 1e-5,
        multiplication_type: TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.OUTGOING,
        bias_flags: dict[str, bool] | None = None,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        high_precision: bool = True,
        mean_normalization: bool = False,
        pair_mask_left_aligned: bool = True,
    ):
        """Triangle multiplication node.

        The internal cuEquivariance 384x256 TriMul owner is selected only for
        BF16 ``dim=384`` / ``hidden_dim=256`` inference on SM90 when both its
        operation and support-query APIs are installed. Runtime dispatch also
        requires a supported input with more than 256 residues. Public
        cuEquivariance releases without the support-query API retain BioIR.

        Args:
            pair_mask_left_aligned: Whether the runtime ``mask`` passed to
                ``forward`` is guaranteed to be left-aligned along its
                masked axis (``1...1 0...0``). The CuTe ``dual_gemm_x_x``
                LM kernel masks via a per-row ``actual_seqlen`` prefix
                count, so it only produces correct outputs under that
                invariant. Default ``True`` for typical outer-product
                ``pair_mask = seq[..., None] * seq[..., None, :]``; set
                ``False`` for bipartite / interior-zero masks such as
                Boltz-2 affinity ``cross_pair_mask`` -- the dispatcher
                then routes the x_x dual GEMM around the CuTe path
                (cuEquiv / vanilla both consume ``mask`` directly
                without the prefix assumption).
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        if bias_flags is None:
            bias_flags = {"p_in": False, "g_in": False, "p_out": False, "g_out": False}
        self.dtype = dtype
        self.high_precision = high_precision
        self.mean_normalization = mean_normalization
        self.pair_mask_left_aligned = pair_mask_left_aligned
        self.eps = eps

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiplication_type = multiplication_type
        self.norm_in = nn.LayerNorm(self.dim, dtype=dtype, eps=eps)
        self.p_in = Linear(
            self.dim,
            2 * self.hidden_dim,
            bias=bias_flags["p_in"],
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.g_in = Linear(
            self.dim,
            2 * self.hidden_dim,
            bias=bias_flags["g_in"],
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        # Use float32 for the output layers
        if self.high_precision:
            self.high_precision_dtype = torch.float32
        else:
            self.high_precision_dtype = dtype
        self.norm_out = nn.LayerNorm(self.hidden_dim, dtype=self.high_precision_dtype, eps=eps)
        self.p_out = Linear(
            self.hidden_dim,
            self.dim,
            bias=bias_flags["p_out"],
            dtype=self.high_precision_dtype,
            skip_create_weights=skip_create_weights,
        )
        self.g_out = Linear(
            self.dim,
            self.dim,
            bias=bias_flags["g_out"],
            dtype=self.high_precision_dtype,
            skip_create_weights=skip_create_weights,
        )

        # Fused half-precision gates require K divisible by 8. The 196-wide
        # operand is padded to 200; LayerNorm emits the zero tail.
        k_align = 8 if self.high_precision_dtype in (torch.float16, torch.bfloat16) else 1
        self._k_align_or_off = k_align if k_align > 1 else -1
        self._x0_k = _round_up(self.dim, k_align)
        self._x1_k = _round_up(self.hidden_dim, k_align)
        self._k_pad_cache: dict[str, tuple[tuple, torch.Tensor]] = {}
        self._dual_gemm_x0_x1_op = get_dual_gemm_x0_x1_op(
            self.high_precision_dtype,
            transpose_out=False,
            N=self.dim,
            K0=self._x0_k,
            K1=self._x1_k,
        )
        # Route interior-zero masks around CuTe's prefix-mask kernel.
        self._dual_gemm_x_x_op_transpose = get_dual_gemm_x_x_op(
            self.dtype,
            transpose_out=True,
            N=2 * self.hidden_dim,
            K=self.dim,
            pair_mask_left_aligned=self.pair_mask_left_aligned,
        )
        self._cueq_trimul_api: tuple[Callable[..., torch.Tensor], Callable[..., bool]] | None = None
        if (
            not skip_create_weights
            and self.dim == _CUEQ_TRIMUL_PAIR_DIM
            and self.hidden_dim == _CUEQ_TRIMUL_HIDDEN_DIM
            and self.dtype == torch.bfloat16
            and not self.high_precision
            and not self.mean_normalization
            and self.p_in.bias is not None
            and self.g_in.bias is not None
            and self.p_out.bias is not None
            and self.g_out.bias is not None
        ):
            cueq_trimul_api = _get_cueq_trimul_api()
            if cueq_trimul_api is not None and torch.cuda.is_available() and get_sm_version() == 90:
                self._cueq_trimul_api = cueq_trimul_api

    def _einsum_compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
            return torch.einsum("bikd,bjkd->bijd", a, b)
        return torch.einsum("bkid,bkjd->bijd", a, b)

    def _contract_projection(self, projected: torch.Tensor, mask: torch.Tensor, *, transposed: bool) -> torch.Tensor:
        """Contract a dual-GEMM projection in a lifetime-bounded scope.

        ``projected`` and its two views are O(N²) temporaries. Keeping them
        local to this helper releases their references as soon as the
        contraction is returned, before the caller allocates output
        normalization and gating tensors.
        """
        split_dim = 0 if transposed else -1
        a, b = torch.chunk(projected, 2, dim=split_dim)
        if self.mean_normalization:
            n_valid = mask[:, 0, :].sum(dim=-1)
            denominator = n_valid[None, :, None, None] if transposed else n_valid[:, None, None, None]
            b = b / (denominator + 1e-3)

        if not transposed:
            return self._einsum_compute(a, b)
        if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
            return torch.einsum("dbik,dbjk->dbij", a, b)
        return torch.einsum("dbki,dbkj->dbij", a, b)

    def _k_padded_weight(self, slot: str, weight: torch.Tensor, k_padded: int) -> torch.Tensor:
        """Zero-extend a ``[N, K]`` weight to match a padded K operand.

        Cached across calls and rebuilt whenever the source weight is
        rewritten, which covers ``load_state_dict``, an in-place ``copy_``
        and a ``.data`` reassignment.
        """
        if weight.shape[-1] >= k_padded:
            return weight
        key = (weight.data_ptr(), weight._version, weight.dtype)
        cached = self._k_pad_cache.get(slot)
        if cached is None or cached[0] != key:
            padded = weight.new_zeros((*weight.shape[:-1], k_padded))
            padded[..., : weight.shape[-1]] = weight
            cached = (key, padded)
            self._k_pad_cache[slot] = cached
        return cached[1]

    @staticmethod
    def _zero_extend_k(x: torch.Tensor, k_padded: int) -> torch.Tensor:
        """Widen the last dim to ``k_padded``, leaving wider inputs alone."""
        pad = k_padded - x.shape[-1]
        return x if pad <= 0 else F.pad(x, (0, pad))

    def _output_gate(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Run the fused output gate, zero-extending K where required."""
        return self._dual_gemm_x0_x1_op(
            self._zero_extend_k(x0, self._x0_k),
            self._zero_extend_k(x1, self._x1_k),
            self._k_padded_weight("g_out", self.g_out.weight, self._x0_k),
            self._k_padded_weight("p_out", self.p_out.weight, self._x1_k),
            self.g_out.bias,
            self.p_out.bias,
        )

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the dtype of the input"""
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x

    def _cueq_forward_if_supported(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Run the optional internal SM90 384x256 TriMul owner."""
        if self._cueq_trimul_api is None or x.shape[-2] <= _CUEQ_TRIMUL_SEQUENCE_THRESHOLD or torch.is_grad_enabled():
            return None

        x = self._ensure_dtype(x)
        mask_value = mask.to(dtype=x.dtype).contiguous()
        operation, is_supported = self._cueq_trimul_api
        direction = "outgoing" if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else "incoming"
        if not is_supported(
            x,
            direction=direction,
            mask=mask_value,
            c_hidden=self.hidden_dim,
        ):
            return None
        return operation(
            x,
            direction=direction,
            mask=mask_value,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,
            p_in_bias=self.p_in.bias,
            g_in_weight=self.g_in.weight,
            g_in_bias=self.g_in.bias,
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,
            p_out_bias=self.p_out.bias,
            g_out_weight=self.g_out.weight,
            g_out_bias=self.g_out.bias,
            eps=self.eps,
        )

    def _forward_impl(
        self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run triangle multiplication with feature-major intermediates.

        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            actual_seqlen (torch.Tensor | None): precomputed ``int32[B, I]``
                per-row valid-J count for the CuTe dual_gemm_x_x backend
                (same as ``actual_s_kv`` from CuTeDSL precompute).
        """
        x = self._ensure_dtype(x)
        x = layer_norm_transpose(x, self.norm_in.weight, self.norm_in.bias, eps=self.eps, layout="bijd->bijd")
        x_in = x
        # Gated dual gemm
        x = self._contract_projection(
            self._dual_gemm_x_x_op_transpose(
                x,
                self.g_in.weight,
                self.p_in.weight,
                self.g_in.bias,
                self.p_in.bias,
                mask,
                transpose_out=True,
                actual_seqlen=actual_seqlen,
            ),
            mask,
            transposed=True,
        )

        # Output normalization. The gate wants a K that is a multiple of 8,
        # and this norm can emit the zero tail without an extra pass.
        x = layer_norm_transpose(
            x,
            self.norm_out.weight,
            self.norm_out.bias,
            eps=self.eps,
            layout="dbij->bijd",
            pad_multiple=self._k_align_or_off,
        )

        # Output gating
        return self._output_gate(x_in.to(self.high_precision_dtype), x.to(self.high_precision_dtype))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: input pair tensor ``[B, I, J, c_in]``.
            mask: pair mask ``[B, I, J]`` (1 = valid, 0 = padded).
            actual_seqlen: optional precomputed ``int32[B, I]`` per-row
                valid-J count consumed by the CuTe dual_gemm_x_x backend
                (the LM kernel treats each ``(b, i)`` row as a separate
                kernel batch). When supplied, it is forwarded to the
                gated GEMM so the wrapper can skip its internal
                ``mask.sum(-1)`` reduction (which would otherwise repeat
                at every layer). When threading from
                :class:`~bionemo_ir._torch.attention_backend.utils.PrecomputedPairMasks`,
                ``tri_mul_out`` (``OUTGOING``) should be passed
                ``precomputed_masks.mask_bias`` (which for the CuTeDSL
                backend is ``actual_s_kv``, the per-row ``[B, I]`` int32
                valid-J count); ``tri_mul_in`` (``INCOMING``) the
                analogous ``precomputed_masks.mask_bias_transposed``
                (``actual_s_kv_t``, ``[B, J]``).  Only meaningful when
                the precompute came from the CuTeDSL backend; default
                backends store an additive bias in those fields and
                callers must pass ``None`` instead.
        """
        cueq_output = self._cueq_forward_if_supported(x, mask)
        if cueq_output is not None:
            return cueq_output
        return self._forward_impl(x, mask, actual_seqlen=actual_seqlen)
