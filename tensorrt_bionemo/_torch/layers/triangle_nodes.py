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

from enum import IntEnum

import torch
import torch.nn as nn
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

from tensorrt_bionemo._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from tensorrt_bionemo._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers

from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import precompute_pair_masks
from ..auto_chunk import CHUNK_REGISTRY, TRIANGLE_ATTENTION, ChunkPolicy, chunk_apply
from ..custom_ops.dual_gemm_x0_x1 import get_dual_gemm_x0_x1_op
from ..custom_ops.dual_gemm_x_x import get_dual_gemm_x_x_op
from .attention import TriangleAttention


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
    ):
        """
        Args:
            c_in (int): input channel dimension
            c_hidden (int): hidden channel dimension
            num_heads (int): number of attention heads
            node_type (TriangleAttentionNodeType): whether this is the starting node
            inf (float): infinity value
            dtype (torch.dtype): data type
            chunk_policy (Optional[ChunkPolicy]): query-row chunking policy; ``None`` uses the
                shared ``triangle_attention`` policy from ``CHUNK_REGISTRY``.
            skip_create_weights (bool): whether to skip creating weights
            attn_backend (str): attention backend
        """
        super().__init__()
        if mha_bias_flags is None:
            mha_bias_flags = {"q": False, "k": False, "v": False, "g": False, "o": False}
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        self.dtype = dtype
        self.attn_backend = attn_backend
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
        self._ln_proj_moveaxis_pad = LNProjMoveaxisPad(D=self.c_in, H=self.num_heads, dtype=dtype or torch.bfloat16)

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
        triangle_bias = self._ln_proj_moveaxis_pad(
            x,
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
        return self.mha(x, biases=[mask_bias, triangle_bias], attn_metadata=attn_metadata, buffers=buffers)

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
            mask (Optional[torch.Tensor]): mask tensor [B, I, J].
                Ignored when *mask_bias* is provided.
            mask_bias (Optional[torch.Tensor]): precomputed additive mask bias
                ([B, I, 1, 1, J] for starting, [B, J, 1, 1, I] for ending).
                When supplied, the per-layer mask->bias computation is skipped.
            attn_metadata (Optional[AttentionMetadata]): attention metadata
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

        # TODO: Make this threshold configurable
        self._forward_impl_v2_threshold = 384
        # Dedicated x0_x1 dispatcher: routes to CuTe (SM 80/86/89/90) or
        # cuEquiv / vanilla otherwise, based on
        # ``(high_precision_dtype, N=self.dim, K=self.hidden_dim)``. Note
        # that ``high_precision=True`` -> fp32 -> vanilla fallback (the
        # CuTe / cuEquiv paths only accept fp16 / bf16).
        self._dual_gemm_x0_x1_op = get_dual_gemm_x0_x1_op(
            self.high_precision_dtype,
            transpose_out=False,
            N=self.dim,
            K=self.hidden_dim,
        )
        # ``pair_mask_left_aligned`` must propagate so a bipartite /
        # interior-zero pair mask routes around the CuTe LM kernel (which
        # masks via a per-row prefix count and is silently wrong otherwise).
        self._dual_gemm_x_x_op = get_dual_gemm_x_x_op(
            self.dtype,
            transpose_out=False,
            N=2 * self.hidden_dim,
            K=self.dim,
            pair_mask_left_aligned=self.pair_mask_left_aligned,
        )
        self._dual_gemm_x_x_op_transpose = get_dual_gemm_x_x_op(
            self.dtype,
            transpose_out=True,
            N=2 * self.hidden_dim,
            K=self.dim,
            pair_mask_left_aligned=self.pair_mask_left_aligned,
        )

    def _einsum_compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
            return torch.einsum("bikd,bjkd->bijd", a, b)
        return torch.einsum("bkid,bkjd->bijd", a, b)

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the dtype of the input"""
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x

    def _forward_impl_v1(
        self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None
    ) -> torch.Tensor:
        """This version is used for short sequences in eager mode
        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            actual_seqlen (Optional[torch.Tensor]): precomputed ``int32[B, I]``
                per-row valid-J count for the CuTe dual_gemm_x_x backend
                (same as ``actual_s_kv`` from CuTeDSL precompute).
        """
        x = self._ensure_dtype(x)
        x = self.norm_in(x)

        dg_actual_seqlen = actual_seqlen
        x_in = x
        x = self._dual_gemm_x_x_op(
            x, self.g_in.weight, self.p_in.weight, self.g_in.bias, self.p_in.bias, mask, actual_seqlen=dg_actual_seqlen
        )
        x = x.to(self.high_precision_dtype)

        a, b = x.split([self.dim, self.dim], dim=-1)
        if self.mean_normalization:
            # Divide right branch by number of valid tokens (mean over contraction axis).
            # mask is [B, I, J] where 1.0=valid; any row gives the valid count.
            n_valid = mask[:, 0, :].sum(dim=-1)  # [B]
            b = b / (n_valid[:, None, None, None] + 1e-3)
        x = self._einsum_compute(a, b)
        x_0_out = self.norm_out(x)
        x_1_out = x_in.to(self.high_precision_dtype)

        x = self._dual_gemm_x0_x1_op(
            x_1_out, x_0_out, self.g_out.weight, self.p_out.weight, self.g_out.bias, self.p_out.bias
        )
        x = self._ensure_dtype(x)
        return x

    def _forward_impl_v2(
        self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None
    ) -> torch.Tensor:
        """This version is used for long sequences and in the compile mode.
        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            actual_seqlen (Optional[torch.Tensor]): precomputed ``int32[B, I]``
                per-row valid-J count for the CuTe dual_gemm_x_x backend
                (same as ``actual_s_kv`` from CuTeDSL precompute).
        """
        x = self._ensure_dtype(x)
        x = layer_norm_transpose(x, self.norm_in.weight, self.norm_in.bias, eps=self.eps, layout="bijd->bijd")

        x_in = x
        # Gated dual gemm
        ab = self._dual_gemm_x_x_op_transpose(
            x,
            self.g_in.weight,
            self.p_in.weight,
            self.g_in.bias,
            self.p_in.bias,
            mask,
            transpose_out=True,
            actual_seqlen=actual_seqlen,
        )

        a, b = torch.chunk(ab, 2, dim=0)
        if self.mean_normalization:
            # Divide right branch by number of valid tokens (mean over contraction axis).
            # mask is [B, I, J]; b is [d, B, I, K] (transposed layout).
            n_valid = mask[:, 0, :].sum(dim=-1)  # [B]
            b = b / (n_valid[None, :, None, None] + 1e-3)
        # Triangular projection
        if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
            x = torch.einsum("dbik,dbjk->dbij", a, b)
        else:
            x = torch.einsum("dbki,dbkj->dbij", a, b)

        # Output normalization
        x_out = layer_norm_transpose(x, self.norm_out.weight, self.norm_out.bias, eps=self.eps, layout="dbij->bijd")

        # Output gating
        x_out = x_out.to(self.high_precision_dtype)
        x_in = x_in.to(self.high_precision_dtype)
        x = self._dual_gemm_x0_x1_op(
            x_in, x_out, self.g_out.weight, self.p_out.weight, self.g_out.bias, self.p_out.bias
        )
        return x

    def _eager_mode_forward(
        self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len < self._forward_impl_v2_threshold:
            return self._forward_impl_v1(x, mask, actual_seqlen=actual_seqlen)
        return self._forward_impl_v2(x, mask, actual_seqlen=actual_seqlen)

    def _compile_mode_forward(
        self, x: torch.Tensor, mask: torch.Tensor, actual_seqlen: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._forward_impl_v2(x, mask, actual_seqlen=actual_seqlen)

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
                :class:`~tensorrt_bionemo._torch.attention_backend.utils.PrecomputedPairMasks`,
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
        if not torch.compiler.is_compiling():
            return self._eager_mode_forward(x, mask, actual_seqlen=actual_seqlen)
        return self._compile_mode_forward(x, mask, actual_seqlen=actual_seqlen)
