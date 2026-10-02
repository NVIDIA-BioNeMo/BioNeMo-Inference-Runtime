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

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from bionemo_ir._torch.utils import CHUNK_REGISTRY, TRIANGLE_ATTENTION, ChunkPolicy
from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import layer_norm_transpose
from bionemo_ir.runtime.buffers import PreallocatedBuffers

from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import precompute_pair_masks
from ..custom_ops.dual_gemm_x0_x1 import get_cute_dual_gemm_x0_x1_residual_op, get_dual_gemm_x0_x1_op
from ..custom_ops.dual_gemm_x_x import get_cute_dual_gemm_x_x_op, get_dual_gemm_x_x_op
from ..custom_ops.trimul_kf_k1 import TrimulKFInputFold, TrimulKFK1Op, fold_input_weights, get_trimul_kf_k1_op
from ..custom_ops.trimul_kf_k2 import TrimulKFK2Op, get_trimul_kf_k2_op
from ..custom_ops.trimul_kf_k3 import TrimulKFK3Op, TrimulKFOutputFold, fold_output_weights, get_trimul_kf_k3_op
from .attention import TriangleAttention


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


#: Token multiple that keeps the contraction on cuBLAS's SM90 bf16 kernels.
_GEMM_TOKEN_ALIGN = 8
#: Widest ``dim == hidden_dim`` the SM90 TriMul KF chain ships.
_KF_MAX_DIM = 256


class TriangleAttentionNodeType(IntEnum):
    STARTING = 0
    ENDING = 1


class TriangleMultiplicationNodeType(IntEnum):
    INCOMING = 0
    OUTGOING = 1


@dataclass(frozen=True)
class TriangleMultiplicationMetadata:
    """Token alignment and row lengths shared by a TriMul stack."""

    token_pad_multiple: int = -1
    outgoing_actual_seqlen: torch.Tensor | None = None
    incoming_actual_seqlen: torch.Tensor | None = None
    padded_outgoing_actual_seqlen: torch.Tensor | None = None
    padded_incoming_actual_seqlen: torch.Tensor | None = None


def _pad_actual_seqlen(actual_seqlen: torch.Tensor, rows: int, rows_padded: int) -> torch.Tensor:
    counts = actual_seqlen.reshape(-1, rows)
    padded = counts.new_zeros((counts.shape[0], rows_padded))
    padded[:, :rows] = counts
    return padded


def precompute_trimul_metadata(
    x: torch.Tensor,
    outgoing_actual_seqlen: torch.Tensor | None,
    incoming_actual_seqlen: torch.Tensor | None,
    *,
    enabled: bool = True,
) -> TriangleMultiplicationMetadata:
    """Prepare row lengths and optional token padding once per stack."""
    if not enabled:
        outgoing_actual_seqlen = incoming_actual_seqlen = None
    token_pad = -1
    padded_outgoing = padded_incoming = None
    can_pad = (
        (outgoing_actual_seqlen is not None or incoming_actual_seqlen is not None)
        and x.device.type == "cuda"
        and x.dtype in (torch.bfloat16, torch.float16)
        and (x.shape[1] % _GEMM_TOKEN_ALIGN != 0 or x.shape[2] % _GEMM_TOKEN_ALIGN != 0)
    )
    if can_pad:
        token_pad = _GEMM_TOKEN_ALIGN
        if outgoing_actual_seqlen is not None:
            padded_outgoing = _pad_actual_seqlen(
                outgoing_actual_seqlen,
                x.shape[1],
                _round_up(x.shape[1], token_pad),
            )
        if incoming_actual_seqlen is not None:
            padded_incoming = _pad_actual_seqlen(
                incoming_actual_seqlen,
                x.shape[2],
                _round_up(x.shape[2], token_pad),
            )
    return TriangleMultiplicationMetadata(
        token_pad_multiple=token_pad,
        outgoing_actual_seqlen=outgoing_actual_seqlen,
        incoming_actual_seqlen=incoming_actual_seqlen,
        padded_outgoing_actual_seqlen=padded_outgoing,
        padded_incoming_actual_seqlen=padded_incoming,
    )


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
        self.inf = inf
        self.dtype = dtype
        self.attn_backend = attn_backend
        self.pair_mask_left_aligned = pair_mask_left_aligned
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.mha = TriangleAttention(
            layer_idx=layer_idx,
            hidden_size=self.c_in,
            head_dim=c_hidden,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            gating=True,
            bias_flags=mha_bias_flags,
            bias_proj=True,
            transposed_bias=transposed_bias,
            # Row chunks bound the [chunk, J, H, ...] attention temporaries at large N.
            chunk_policy=chunk_policy if chunk_policy is not None else CHUNK_REGISTRY.get(TRIANGLE_ATTENTION),
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_bias: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        *,
        residual: bool = False,
        inplace_residual: bool = False,
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
            residual: Return ``x + update`` instead of the update; on SM90 the
                output projection's epilogue adds ``x``.
            inplace_residual: Accumulate into ``x``, which the caller must own.
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

        output = self.mha(
            self.layer_norm(x),
            mask_bias,
            attn_metadata=attn_metadata,
            buffers=buffers,
            use_kv_lengths=self.pair_mask_left_aligned,
            residual=x if residual else None,
            inplace_residual=inplace_residual,
        )
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
        align_contraction_tokens: bool = True,
        pad_tokens: bool = False,
    ):
        """Triangle multiplication node.

        Supported SM90 BF16 shapes may dispatch to the SM90 TriMul KF chain
        when the model pads tokens.

        Args:
            pair_mask_left_aligned: Whether each mask row is ``1...1 0...0``,
                as required by CuTe's prefix-length masking. ``False`` disables
                the CuTe dual GEMM.
            align_contraction_tokens: Pad both token axes to multiples of 8 for
                fast SM90 GEMMs. Requires CuTe and ``actual_seqlen``.
            pad_tokens: Whether the owning model pads token axes to multiples
                of 8 before this node runs (its ``enable_token_pad``). Enables
                the SM90 TriMul KF chain for ``dim == hidden_dim <= 256``; see
                :meth:`set_token_padding`.
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
        self.align_contraction_tokens = align_contraction_tokens
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

        # Select a fused output gate only when the native widths are tuned.
        self._dual_gemm_x0_x1_op = get_dual_gemm_x0_x1_op(
            self.high_precision_dtype,
            transpose_out=False,
            N=self.dim,
            K0=self.dim,
            K1=self.hidden_dim,
        )
        self._dual_gemm_x0_x1_residual_op = (
            get_cute_dual_gemm_x0_x1_residual_op(
                self.high_precision_dtype,
                N=self.dim,
                K0=self.dim,
                K1=self.hidden_dim,
            )
            if self.pair_mask_left_aligned
            else None
        )
        # Route interior-zero masks around CuTe's prefix-mask kernel.
        self._dual_gemm_x_x_op_transpose = get_dual_gemm_x_x_op(
            self.dtype,
            transpose_out=True,
            N=2 * self.hidden_dim,
            K=self.dim,
            pair_mask_left_aligned=self.pair_mask_left_aligned,
        )
        # Only the CuTe backend takes the token extent from ``actual_seqlen``.
        # The others read it off ``mask``, which stays unpadded, so a padded
        # operand would not line up.
        self._token_align_backend = self.pair_mask_left_aligned and (
            get_cute_dual_gemm_x_x_op(self.dtype, N=2 * self.hidden_dim, K=self.dim, gate="sigmoid") is not None
        )
        self._kf_ops: tuple[TrimulKFK1Op, TrimulKFK2Op, TrimulKFK3Op] | None = None
        self._kf_folds: tuple[TrimulKFInputFold, TrimulKFOutputFold] | None = None
        self.set_token_padding(pad_tokens)

    def set_token_padding(self, enabled: bool) -> None:
        """Set whether the owning model pads token axes to multiples of 8.

        Padding enables the SM90 TriMul KF chain ``trimul_kf_k1`` ->
        ``trimul_kf_k2`` -> ``trimul_kf_k3`` when :meth:`_get_kf_ops` finds all
        three for this node.
        """
        self.pad_tokens = enabled
        self._kf_folds = None
        self._kf_ops = self._get_kf_ops() if enabled else None

    def _get_kf_ops(self) -> tuple[TrimulKFK1Op, TrimulKFK2Op, TrimulKFK3Op] | None:
        """The KF K1, K2 and K3 ops for this node, or ``None`` when any does not ship.

        The chain covers ``dim == hidden_dim <= 256`` (N = K0 = K1) with bf16
        output projections, plain sums and prefix-shaped masks, so
        ``high_precision``, ``mean_normalization`` and non-left-aligned masks
        keep the regular path.
        """
        if (
            self.dim != self.hidden_dim
            or self.dim > _KF_MAX_DIM
            or self.high_precision
            or self.mean_normalization
            or not self.pair_mask_left_aligned
        ):
            return None
        outgoing = self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING
        ops = (
            get_trimul_kf_k1_op(self.dtype, self.dim, self.hidden_dim),
            get_trimul_kf_k2_op(self.dtype, self.hidden_dim, outgoing),
            get_trimul_kf_k3_op(self.dtype, self.dim, self.hidden_dim),
        )
        return None if any(op is None for op in ops) else ops

    def _prepare_kf_weights(self) -> None:
        """Fold the LayerNorms and biases into the KF chain's weights.

        The weights hold placeholders until ``load_weights`` fills them in
        place, so the node folds on its first KF call rather than at
        construction, and :meth:`post_load_weights` drops the folds a reload
        makes stale.
        """
        self._kf_folds = (
            fold_input_weights(
                self.norm_in.weight,
                self.norm_in.bias,
                self.p_in.weight,
                self.g_in.weight,
                self.p_in.bias,
                self.g_in.bias,
            ),
            fold_output_weights(
                self.norm_out.weight,
                self.norm_out.bias,
                self.norm_in.weight,
                self.norm_in.bias,
                self.p_out.weight,
                self.g_out.weight,
                self.p_out.bias,
                self.g_out.bias,
            ),
        )

    def post_load_weights(self) -> None:
        """Drop the folded KF weights, so the next KF call folds the loaded ones."""
        self._kf_folds = None

    def _kf_forward_if_supported(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        trimul_metadata: TriangleMultiplicationMetadata,
        residual: bool,
    ) -> torch.Tensor | None:
        """Run the KF chain K1 -> K2 -> K3, or return ``None`` for the regular path.

        A call takes it when its token axes are 8-aligned, as model-level
        padding guarantees. The row lengths are those the dual GEMMs use: the
        node direction's lengths for K1 and the outgoing ones for the output,
        or the mask's row counts for both when the metadata has none.
        """
        if self._kf_ops is None:
            return None
        k1, k2, k3 = self._kf_ops
        outgoing = self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING
        out_seqlen = trimul_metadata.outgoing_actual_seqlen
        k1_seqlen = out_seqlen if outgoing else trimul_metadata.incoming_actual_seqlen
        if k1_seqlen is None or out_seqlen is None:
            k1_seqlen = out_seqlen = (mask > 0).sum(-1, dtype=torch.int32)
        if not k1.accepts(x, k1_seqlen):
            return None
        if self._kf_folds is None or self._kf_folds[0].device != x.device:
            self._prepare_kf_weights()
        fold_in, fold_out = self._kf_folds
        a, b, stats = k1(x, k1_seqlen, fold_in, self.eps)
        return k3(
            k2(a, b),
            x,
            fold_out,
            stats,
            self.eps,
            residual=residual,
            actual_seqlen=out_seqlen if residual else None,
        )

    def _einsum_compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
            return torch.einsum("bikd,bjkd->bijd", a, b)
        return torch.einsum("bkid,bkjd->bijd", a, b)

    def _contract_projection(self, projected: torch.Tensor, mask: torch.Tensor, *, transposed: bool) -> torch.Tensor:
        """Contract a dual-GEMM projection while limiting temporary lifetimes."""
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

    def can_fuse_residual(self, x: torch.Tensor) -> bool:
        """Whether the residual matches the output gate's aligned layout."""
        return (
            self._dual_gemm_x0_x1_residual_op is not None
            and x.shape[1] % _GEMM_TOKEN_ALIGN == 0
            and x.shape[2] % _GEMM_TOKEN_ALIGN == 0
        )

    def _output_gate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        actual_seqlen: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the output gate, optionally fusing the masked residual."""
        operation = self._dual_gemm_x0_x1_op
        if residual is not None:
            if self._dual_gemm_x0_x1_residual_op is None or actual_seqlen is None:
                raise ValueError("fused triangle residual requires a supported CuTe kernel and actual_seqlen")
            operation = self._dual_gemm_x0_x1_residual_op
        return operation(
            x0,
            x1,
            self.g_out.weight,
            self.p_out.weight,
            self.g_out.bias,
            self.p_out.bias,
            actual_seqlen=actual_seqlen,
            residual=residual,
        )

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the dtype of the input"""
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        trimul_metadata: TriangleMultiplicationMetadata,
        *,
        residual: bool = False,
    ) -> torch.Tensor:
        """Run triangle multiplication with feature-major intermediates.

        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            trimul_metadata: precomputed token padding and row lengths.
        """
        residual_pair = x
        x = self._ensure_dtype(x)
        tokens_i, tokens_j = x.shape[1], x.shape[2]
        outgoing = trimul_metadata.outgoing_actual_seqlen
        incoming = trimul_metadata.incoming_actual_seqlen
        padded_outgoing = trimul_metadata.padded_outgoing_actual_seqlen
        padded_incoming = trimul_metadata.padded_incoming_actual_seqlen
        selected = outgoing if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else incoming
        padded_selected = (
            padded_outgoing if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else padded_incoming
        )
        use_token_pad = (
            self.align_contraction_tokens
            and self._token_align_backend
            and trimul_metadata.token_pad_multiple > 0
            and padded_selected is not None
        )
        token_pad = trimul_metadata.token_pad_multiple if use_token_pad else -1
        actual_seqlen = padded_selected if use_token_pad else selected
        output_actual_seqlen = padded_outgoing if use_token_pad else outgoing
        x = layer_norm_transpose(
            x,
            self.norm_in.weight,
            self.norm_in.bias,
            eps=self.eps,
            layout="bijd->bijd",
            token_pad_multiple=token_pad,
        )
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

        # Output normalization.
        x = layer_norm_transpose(
            x,
            self.norm_out.weight,
            self.norm_out.bias,
            eps=self.eps,
            layout="dbij->bijd",
        )

        # Output gating
        fuse_residual = residual and self.can_fuse_residual(residual_pair) and output_actual_seqlen is not None
        out = self._output_gate(
            x_in.to(self.high_precision_dtype),
            x.to(self.high_precision_dtype),
            actual_seqlen=output_actual_seqlen if fuse_residual else None,
            residual=residual_pair if fuse_residual else None,
        )
        if token_pad > 0:
            # Hand back the caller's token extents as a view, so the padding
            # costs no copy at all.
            out = out[:, :tokens_i, :tokens_j]
        if residual and not fuse_residual:
            out = residual_pair + out
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        trimul_metadata: TriangleMultiplicationMetadata,
        *,
        residual: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: input pair tensor ``[B, I, J, c_in]``.
            mask: pair mask ``[B, I, J]`` (1 = valid, 0 = padded).
            trimul_metadata: precomputed token padding and row lengths.
            residual: fuse ``x + update`` and the output mask when supported.
        """
        kf_output = self._kf_forward_if_supported(x, mask, trimul_metadata, residual)
        if kf_output is not None:
            return kf_output
        return self._forward_impl(
            x,
            mask,
            trimul_metadata=trimul_metadata,
            residual=residual,
        )


def set_trimul_token_padding(module: nn.Module, enabled: bool) -> None:
    """Propagate a model's token-padding config to every triangle multiplication under ``module``.

    The owning model pads token axes to multiples of 8 before running
    ``module``, which lets each :class:`TriangleMultiplicationNode` take the
    SM90 TriMul KF chain. A node still falls back on any call that arrives
    unaligned.
    """
    for submodule in module.modules():
        if isinstance(submodule, TriangleMultiplicationNode):
            submodule.set_token_padding(enabled)
