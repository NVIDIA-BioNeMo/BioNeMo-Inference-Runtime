# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import torch
import torch.nn as nn
from cuequivariance_ops_torch.fused_layer_norm_torch import \
    layer_norm_transpose

from tensorrt_bionemo._torch.custom_ops.fused_ln_proj_moveaxis_pad import \
    LNProjMoveaxisPad
from tensorrt_bionemo._torch.distributed import (
    AllReduceParams, get_default_dcp_group_coordinator,
    get_default_tp_group_coordinator)
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._trt.layers.triangle_nodes import (
    TriangleAttentionNodeType, TriangleMultiplicationNodeType)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers

from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import precompute_pair_masks
from ..custom_ops.dual_gemm_x0_x1 import get_dual_gemm_x0_x1_op
from ..custom_ops.dual_gemm_x_x import get_dual_gemm_x_x_op
from .attention import TriangleAttention


class TriangleAttentionNode(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        num_heads: int,
        node_type: TriangleAttentionNodeType = TriangleAttentionNodeType.
        STARTING,
        inf: float = 1e9,
        layer_idx: int = 0,
        dtype: torch.dtype = None,
        chunk_size: int = 0,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
        mha_bias_flags: dict[str, bool] = {
            "q": False,
            "k": False,
            "v": False,
            "g": False,
            "o": False
        }):
        """
        Args:
            c_in (int): input channel dimension
            c_hidden (int): hidden channel dimension
            num_heads (int): number of attention heads
            node_type (TriangleAttentionNodeType): whether this is the starting node
            inf (float): infinity value
            dtype (torch.dtype): data type
            chunk_size (int): chunk size
            mapping (Mapping): mapping
            skip_create_weights (bool): whether to skip creating weights
            attn_backend (str): attention backend
        """
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        self.mapping = mapping or Mapping()
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node
        self.dtype = dtype
        self.attn_backend = attn_backend

        assert self.num_heads % self.tp_size == 0
        self.num_heads = self.num_heads // self.tp_size
        self.chunk_size = chunk_size

        if self.chunk_size > 0:
            assert self.chunk_size % self.dcp_size == 0
            self.chunk_size = self.chunk_size // self.dcp_size
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.linear = Linear(
            self.c_in,
            self.tp_size * self.num_heads,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
        )

        self.mha = TriangleAttention(
            layer_idx=layer_idx,
            hidden_size=self.c_in,
            head_dim=c_hidden,
            num_attention_heads=self.num_heads * self.tp_size,
            num_key_value_heads=self.num_heads * self.tp_size,
            gating=True,
            bias_flags=mha_bias_flags,
            dtype=dtype,
            mapping=self.mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )

        self.J_padded_multiple = -1
        if self.attn_backend == "CuTeDSL":
            self.J_padded_multiple = 8
        self._ln_proj_moveaxis_pad = LNProjMoveaxisPad(
            D=self.c_in,
            H=self.tp_size * self.num_heads,
            dtype=dtype or torch.bfloat16)
        self.dcp_group_comm = None
        if self.dcp_size > 1:
            self.dcp_group_comm = get_default_dcp_group_coordinator()
            assert self.dcp_group_comm(
            ) is not None, "DP group coordinator is not initialized"

    def _dcp_slice(
            self, x: torch.Tensor,
            mask_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Deal with the dcp size > 1 """
        seq_len = x.shape[1]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            start = self.dcp_rank * seq_len
            end = (self.dcp_rank + 1) * seq_len
            x = x[:, start:end, ...]
            mask_bias = mask_bias[:, start:end, ...]
        if not x.is_contiguous():
            x = x.contiguous()
        if not mask_bias.is_contiguous():
            mask_bias = mask_bias.contiguous()
        return x, mask_bias

    def _dcp_gather(self, output: torch.Tensor) -> torch.Tensor:
        """ Gather the input by dcp size """
        if self.dcp_size > 1:
            output = self.dcp_group_comm().all_gather(output, dim=1)
        return output

    def _ensure_dtype(self, x: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Ensure the dtype of the input and mask """
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_bias: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        buffers: Optional[PreallocatedBuffers] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the triangle attention node. If dcp_size > 1 and chunk_size,
        make sure the sequence length is a multiple of chunk_size*dcp_size. Currently,
        supports only batch_size = 1

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
            precomputed = precompute_pair_masks(self.attn_backend,
                                                mask,
                                                inf=self.inf,
                                                dtype=self.dtype)
            mask_bias = precomputed.mask_bias

        x, triangle_bias = self._prep_bias(x)

        seq_len = x.shape[1]
        x, mask_bias = self._dcp_slice(x, mask_bias)
        if self.chunk_size > 0:
            niters = (seq_len + self.chunk_size - 1) // self.chunk_size
            outputs = []
            for i in range(niters):
                start = i * self.chunk_size
                end = start + self.chunk_size
                x_chunk = x[:, start:end, ...]
                chunk_mask_bias = mask_bias[:, start:end, ...]
                biases = [chunk_mask_bias, triangle_bias]
                chunk_output = self.mha(x_chunk,
                                        biases=biases,
                                        attn_metadata=attn_metadata,
                                        all_reduce_params=all_reduce_params,
                                        buffers=buffers)
                outputs.append(chunk_output)
            output = torch.cat(outputs, dim=1)
        else:
            biases = [mask_bias, triangle_bias]
            output = self.mha(x,
                              biases=biases,
                              attn_metadata=attn_metadata,
                              all_reduce_params=all_reduce_params,
                              buffers=buffers)
        output = self._dcp_gather(output)
        if self.node_type == TriangleAttentionNodeType.ENDING:
            output = output.transpose(2, 1)
        return output


class TriangleAttentionStartingNode(TriangleAttentionNode):

    def __init__(self, *args, **kwargs):
        kwargs['node_type'] = TriangleAttentionNodeType.STARTING
        super().__init__(*args, **kwargs)


class TriangleAttentionEndingNode(TriangleAttentionNode):

    def __init__(self, *args, **kwargs):
        kwargs['node_type'] = TriangleAttentionNodeType.ENDING
        super().__init__(*args, **kwargs)


class TriangleMultiplicationNode(nn.Module):

    def __init__(
            self,
            layer_idx: int = 0,
            dim: int = 128,
            hidden_dim: Optional[int] = None,
            eps: float = 1e-5,
            multiplication_type:
        TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.
        OUTGOING,
            bias_flags: dict[str, bool] = {
                "p_in": False,
                "g_in": False,
                "p_out": False,
                "g_out": False
            },
            dtype: torch.dtype = None,
            mapping: Optional[Mapping] = None,
            skip_create_weights: bool = False,
            max_tri_mul_tp_size: bool = False,
            high_precision: bool = True,
            mean_normalization: bool = False,
            pair_mask_left_aligned: bool = True):
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
                (cuEquiv / CUTLASS / vanilla all consume ``mask``
                directly without the prefix assumption).
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.mapping = mapping or Mapping()
        if max_tri_mul_tp_size:
            self.mapping = create_max_tp_mapping(self.mapping, dim)
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node
        self.dtype = dtype
        self.high_precision = high_precision
        self.mean_normalization = mean_normalization
        self.pair_mask_left_aligned = pair_mask_left_aligned
        self.eps = eps

        self.dim = dim // self.tp_size
        self.hidden_dim = hidden_dim // self.tp_size
        self.multiplication_type = multiplication_type
        self.norm_in = nn.LayerNorm(self.dim * self.tp_size,
                                    dtype=dtype,
                                    eps=eps)
        self.p_in = Linear(self.dim * self.tp_size,
                           2 * self.hidden_dim * self.tp_size,
                           bias=bias_flags["p_in"],
                           dtype=dtype,
                           mapping=self.mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=False,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=skip_create_weights)
        self.g_in = Linear(self.dim * self.tp_size,
                           2 * self.hidden_dim * self.tp_size,
                           bias=bias_flags["g_in"],
                           dtype=dtype,
                           mapping=self.mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=False,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=skip_create_weights)
        # Use float32 for the output layers
        if self.high_precision:
            self.high_precision_dtype = torch.float32
        else:
            self.high_precision_dtype = dtype
        self.norm_out = nn.LayerNorm(self.hidden_dim * self.tp_size,
                                     dtype=self.high_precision_dtype,
                                     eps=eps)
        self.p_out = Linear(self.hidden_dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=bias_flags["p_out"],
                            dtype=self.high_precision_dtype,
                            mapping=self.mapping,
                            tensor_parallel_mode=TensorParallelMode.COLUMN,
                            gather_output=True,
                            skip_create_weights=skip_create_weights)
        self.g_out = Linear(self.dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=bias_flags["g_out"],
                            dtype=self.high_precision_dtype,
                            mapping=self.mapping,
                            tensor_parallel_mode=TensorParallelMode.COLUMN,
                            gather_output=True,
                            skip_create_weights=skip_create_weights)

        self.tp_group_comm = None
        self.dcp_group_comm = None
        if self.tp_size > 1:
            self.tp_group_comm = get_default_tp_group_coordinator()
            assert self.tp_group_comm(
            ) is not None, "TP group coordinator is not initialized"
        if self.dcp_size > 1:
            self.dcp_group_comm = get_default_dcp_group_coordinator()
            assert self.dcp_group_comm(
            ) is not None, "DP group coordinator is not initialized"

        # TODO: Make this threshold configurable
        self._forward_impl_v2_threshold = 384
        # Dedicated x0_x1 dispatcher: routes to CuTe (SM 80/86/89), legacy
        # CUTLASS (SM 90), or cuEquiv / vanilla otherwise based on
        # ``(high_precision_dtype, N=self.dim, K=self.hidden_dim)``. Note
        # that ``high_precision=True`` -> fp32 -> vanilla fallback (the
        # CuTe / CUTLASS / cuEquiv paths only accept fp16 / bf16).
        self._dual_gemm_x0_x1_op = get_dual_gemm_x0_x1_op(
            self.high_precision_dtype,
            transpose_out=False,
            N=self.dim,
            K=self.hidden_dim,
        )
        self._dual_gemm_x_x_op = get_dual_gemm_x_x_op(self.dtype,
                                                      transpose_out=False,
                                                      N=2*self.hidden_dim,
                                                      K=self.dim)
        self._dual_gemm_x_x_op_transpose = get_dual_gemm_x_x_op(
            self.dtype, 
            transpose_out=True, 
            N=2*self.hidden_dim, 
            K=self.dim)

    def _dcp_slice(self, x: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Slice the input by dcp size """
        seq_len = x.shape[1]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            st = self.dcp_rank * seq_len
            et = (self.dcp_rank + 1) * seq_len
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                x = x[:, st:et, ...]
                mask = mask[:, st:et, ...]
            elif self.multiplication_type == TriangleMultiplicationNodeType.INCOMING:
                x = x[:, :, st:et, ...]
                mask = mask[:, :, st:et]
            x = x.contiguous()
            mask = mask.contiguous()
        return x, mask

    def _dcp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """ Gather the input by dcp size """
        if self.dcp_size > 1:
            gather_dim = 1 if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else 2
            self.dcp_group_comm().barrier()
            x = self.dcp_group_comm().all_gather(x, dim=gather_dim)
        return x

    def _tp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """ Gather the input by tp size """
        if self.tp_size > 1:
            self.tp_group_comm().barrier()
            x = self.tp_group_comm().all_gather(x)
        return x

    def _ring_einsum_compute(self, a: torch.Tensor,
                             b: torch.Tensor) -> torch.Tensor:
        """ Compute the enisum operation in a ring manner """

        def _einsum_compute(a_, b_):
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                return torch.einsum("bikd,bjkd->bijd", a_, b_)
            else:
                return torch.einsum("bkid,bkjd->bijd", a_, b_)

        # Ring communication
        if self.dcp_size > 1:
            a = a.contiguous()
            b = b.contiguous()
            enisum_results = [
                None,
            ] * self.dcp_size
            enisum_results[self.dcp_rank] = _einsum_compute(a, b)
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                b_recv = torch.zeros_like(b)
                buffers = [b, b_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dcp_group_comm().batch_isend_irecv(
                        buffers[send_idx], buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _einsum_compute(
                                       a, buffers[recv_idx])
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=2)
            else:
                a_recv = torch.zeros_like(a)
                buffers = [a, a_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dcp_group_comm().batch_isend_irecv(
                        buffers[send_idx], buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _einsum_compute(
                                       buffers[recv_idx], b)
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=1)
        else:
            x = _einsum_compute(a, b)
        return x

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """ Ensure the dtype of the input """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x

    def _forward_impl_v1(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            actual_seqlen: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ This version is used for short sequences in eager mode
        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            actual_seqlen (Optional[torch.Tensor]): precomputed ``int32[B, I]``
                per-row valid-J count for the CuTe dual_gemm_x_x backend
                (same as ``actual_s_kv`` from CuTeDSL precompute).
                Under DCP: OUTGOING slices it on I to match the sliced
                rows; INCOMING (slices on J) drops it and lets the wrapper
                recompute from the sliced ``mask``.
        """
        x = self._ensure_dtype(x)
        x = self.norm_in(x)

        x, mask = self._dcp_slice(x, mask)
        # ``actual_seqlen[b, i] = mask[b, i, :].sum()``; align it with the
        # DCP-sliced mask:
        #   * OUTGOING slices on I -> slice ``actual_seqlen`` on dim 1
        #     (per-row counts on the kept I rows are unchanged).
        #   * INCOMING slices on J -> per-(b, i) J counts shrink, so the
        #     precomputed value no longer matches; let the dual_gemm
        #     wrapper recompute from the sliced ``mask``.
        dg_actual_seqlen = actual_seqlen
        if dg_actual_seqlen is not None and self.dcp_size > 1:
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                seq_len_full = dg_actual_seqlen.shape[1]
                shard = seq_len_full // self.dcp_size
                st = self.dcp_rank * shard
                et = (self.dcp_rank + 1) * shard
                dg_actual_seqlen = dg_actual_seqlen[:, st:et].contiguous()
            else:
                dg_actual_seqlen = None
        x_in = x
        x = self._dual_gemm_x_x_op(x,
                                   self.g_in.weight,
                                   self.p_in.weight,
                                   self.g_in.bias,
                                   self.p_in.bias,
                                   mask,
                                   actual_seqlen=dg_actual_seqlen)
        x = x.to(self.high_precision_dtype)

        a, b = x.split([self.dim, self.dim], dim=-1)
        if self.mean_normalization:
            # Divide right branch by number of valid tokens (mean over contraction axis).
            # mask is [B, I, J] where 1.0=valid; any row gives the valid count.
            n_valid = mask[:, 0, :].sum(dim=-1)  # [B]
            b = b / (n_valid[:, None, None, None] + 1e-3)
        x = self._ring_einsum_compute(a, b)
        # need to gather here for LayerNorm

        x = self._tp_gather(x)
        x_0_out = self.norm_out(x)
        x_1_out = x_in.to(self.high_precision_dtype)

        x = self._dual_gemm_x0_x1_op(x_1_out, x_0_out, self.g_out.weight,
                                     self.p_out.weight, self.g_out.bias,
                                     self.p_out.bias)
        x = self._dcp_gather(x)
        x = self._ensure_dtype(x)
        return x

    def _forward_impl_v2(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            actual_seqlen: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ This version is used for long sequences and int the compile mode.
        Distributed is not supported yet.
        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
            actual_seqlen (Optional[torch.Tensor]): precomputed ``int32[B, I]``
                per-row valid-J count for the CuTe dual_gemm_x_x backend
                (same as ``actual_s_kv`` from CuTeDSL precompute). v2 does
                not slice for DCP (distributed unsupported), so the value
                is always safe to forward when supplied.
        """
        x = self._ensure_dtype(x)
        x = layer_norm_transpose(x,
                                 self.norm_in.weight,
                                 self.norm_in.bias,
                                 eps=self.eps,
                                 layout="bijd->bijd")

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
        x_out = layer_norm_transpose(x,
                                     self.norm_out.weight,
                                     self.norm_out.bias,
                                     eps=self.eps,
                                     layout="dbij->bijd")

        # Output gating
        x_out = x_out.to(self.high_precision_dtype)
        x_in = x_in.to(self.high_precision_dtype)
        x = self._dual_gemm_x0_x1_op(x_in, x_out, self.g_out.weight,
                                     self.p_out.weight, self.g_out.bias,
                                     self.p_out.bias)
        return x

    def _eager_mode_forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            actual_seqlen: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        is_distributed = self.dcp_size > 1 or self.tp_size > 1
        if seq_len < self._forward_impl_v2_threshold or is_distributed:
            return self._forward_impl_v1(x, mask, actual_seqlen=actual_seqlen)
        return self._forward_impl_v2(x, mask, actual_seqlen=actual_seqlen)

    def _compile_mode_forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            actual_seqlen: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._forward_impl_v2(x, mask, actual_seqlen=actual_seqlen)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                actual_seqlen: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            return self._eager_mode_forward(x,
                                            mask,
                                            actual_seqlen=actual_seqlen)
        return self._compile_mode_forward(x, mask, actual_seqlen=actual_seqlen)
