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

import torch
import torch.nn as nn

from bionemo_ir._torch.custom_ops.dual_gemm_x_x import get_cute_dual_gemm_x_x_op
from bionemo_ir._torch.custom_ops.gated_sigmoid import get_gated_sigmoid_op
from bionemo_ir._torch.graph_optimization.cudnn_graph import (
    CudnnGraphModule,
    can_use_cudnn_graph,
    cudnn_linear_mask,
    cudnn_linear_mask_residual,
    cudnn_linear_relu,
    cudnn_linear_residual,
    prepare_cudnn_linear_mask,
    prepare_cudnn_linear_relu,
)
from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from bionemo_ir._torch.layers.normalization import AdaLN, AdaLNNormType, FusedLayerNorm
from bionemo_ir._torch.utils import ChunkPolicy, chunk_apply
from bionemo_ir.dsl_kernels.triton.fused_swiglu import FusedSwiGLU
from bionemo_ir.runtime.buffers import PreallocatedBuffers

type _CudnnPlan = Callable[..., torch.Tensor | None]


def _run_cudnn_linear(
    prepared: _CudnnPlan | None,
    static: _CudnnPlan,
    *inputs: torch.Tensor,
) -> torch.Tensor | None:
    """Run a held dynamic-shape plan when there is one, else the static plan.

    ``prepared`` is only populated under ``cudnn_dynamic_shapes``, and it
    rejects row counts cuDNN will not override, so the static plan is both the
    default and the fallback.
    """
    if prepared is not None:
        output = prepared(*inputs)
        if output is not None:
            return output
    return static(*inputs)


def _get_silu_projection_op(dtype: torch.dtype | None, K: int, N: int):
    """Resolve the no-intermediate SwiGLU projection when it ships."""
    resolved_dtype = dtype or torch.get_default_dtype()
    if resolved_dtype not in (torch.float16, torch.bfloat16) or not torch.cuda.is_available():
        return None
    return get_cute_dual_gemm_x_x_op(resolved_dtype, K=K, N=N, gate="silu")


class Transition(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        out_dim: int | None = None,
        layer_idx: int = 0,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        auto_chunk_policy: ChunkPolicy | None = None,
        normalize: bool = True,
    ):
        """SwiGLU feed-forward transition.

        Args:
            normalize: If True (default), apply LayerNorm before the FFN.
                Set this to False when AdaLN (or no pre-norm)
                already ran outside the block.
        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.auto_chunk_policy = auto_chunk_policy
        self.normalize = normalize

        self.dtype = dtype
        self.hidden = hidden
        self.norm = nn.LayerNorm(dim, eps=eps, dtype=dtype) if normalize else None

        self.fused_fc2_fc1 = Linear(
            dim,
            2 * hidden,
            dtype=dtype,
            bias=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self._swiglu = FusedSwiGLU(d=self.hidden, three_way=False, dtype=dtype or torch.bfloat16)
        self._dual_gemm_silu_op = _get_silu_projection_op(dtype, K=dim, N=hidden)
        self.fc3 = Linear(hidden, out_dim, dtype=dtype, bias=False, skip_create_weights=skip_create_weights)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Chunk position-wise FFNs when configured; small inputs stay dense.
        if self.auto_chunk_policy is not None:
            return chunk_apply(self._forward_impl, x, mask, policy=self.auto_chunk_policy)
        return self._forward_impl(x, mask)

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        if self._dual_gemm_silu_op is not None:
            weight = self.fused_fc2_fc1.weight
            x = self._dual_gemm_silu_op(
                x,
                weight[self.hidden :],
                weight[: self.hidden],
                gate="silu",
            )
        else:
            x = self._swiglu(self.fused_fc2_fc1(x))
        x = self.fc3(x)

        if mask is not None:
            if mask.ndim == x.ndim - 1:
                mask = mask.unsqueeze(-1)
            x = x * mask.to(dtype=x.dtype)
        return x


class ConditionedTransitionBlock(nn.Module):
    def __init__(
        self,
        dim_single: int,
        dim_single_cond: int,
        expansion_factor: int = 2,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        using_silu: bool = False,
        norm_type: AdaLNNormType = "layer_norm",
        cond_norm_bias: bool = False,
        output_gate_bias_init: float | None = None,
    ):
        """Conditioned SwiGLU transition with AdaLN and gated output.

        Args:
            using_silu: If True, use 2-way SwiGLU; otherwise use 3-way.
            norm_type: AdaLN primary/condition norm (``layer_norm`` or
                ``rms_norm``).
            cond_norm_bias: Give the AdaLN condition norm a bias.
            output_gate_bias_init: If set, zero the output-gate weight and
                fill its bias (AdaLN-zero variants use ``-2.0``).
        """
        super().__init__()
        self.dtype = dtype

        self.dim_single = dim_single
        self.dim_single_cond = dim_single_cond
        self.expansion_factor = expansion_factor

        self.adaln = AdaLN(
            dim_single, dim_single_cond, eps=eps, dtype=dtype, norm_type=norm_type, cond_norm_bias=cond_norm_bias
        )
        self.dim_inner = int(dim_single * expansion_factor)
        # Fused swiglu_gate linear and a_to_b
        self.using_silu = using_silu
        self._swiglu = FusedSwiGLU(d=self.dim_inner, three_way=not using_silu, dtype=dtype or torch.bfloat16)
        if not using_silu:
            self.fused_swl_a_to_b = Linear(
                self.dim_single,
                3 * self.dim_inner,
                bias=False,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
                weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            )
        else:
            self.fused_swl_a_to_b = Linear(
                self.dim_single,
                2 * self.dim_inner,
                bias=False,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
                weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            )
        self._dual_gemm_silu_op = (
            _get_silu_projection_op(dtype, K=self.dim_single, N=self.dim_inner) if using_silu else None
        )

        self.b_to_a = Linear(
            self.dim_inner, self.dim_single, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
        )

        self.output_projection = Linear(
            self.dim_single_cond, self.dim_single, bias=True, dtype=dtype, skip_create_weights=skip_create_weights
        )
        if output_gate_bias_init is not None and not skip_create_weights:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.constant_(self.output_projection.bias, output_gate_bias_init)
        self._gated_sigmoid_op = get_gated_sigmoid_op(
            dtype or torch.get_default_dtype(),
            N=self.dim_single,
            K=self.dim_single_cond,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        buffers: PreallocatedBuffers | None = None,
        buffer_key: str = "cond_trans_adaln",
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]
            buffers: optional preallocated buffer dict, forwarded to AdaLN.
            buffer_key: key into ``buffers`` for the AdaLN output tensor.
            mask: optional binary mask ``[B, I]`` applied after AdaLN and
                after the output gate.

        Returns:
            a: [B, I, d]
        """
        a = self.adaln(a, s, buffers=buffers, buffer_key=buffer_key, mask=mask)
        if self._dual_gemm_silu_op is not None:
            weight = self.fused_swl_a_to_b.weight
            b = self._dual_gemm_silu_op(
                a,
                weight[self.dim_inner :],
                weight[: self.dim_inner],
                gate="silu",
            )
        else:
            b = self._swiglu(self.fused_swl_a_to_b(a))
        a = self.b_to_a(b)

        # The gated-sigmoid op broadcasts `s` (gate) across the multiplicity
        # dim of `a` when their leading shapes differ, falling back to torch
        # internally for unsupported patterns. Reuse the AdaLN output buffer —
        # fused_swl_a_to_b consumed it above, same shape as the gated_sigmoid
        # output.
        a = self._gated_sigmoid_op(
            s,
            self.output_projection.weight,
            a,
            self.output_projection.bias,
            output=buffers.get(buffer_key) if buffers is not None else None,
        )

        if mask is not None:
            if mask.ndim == a.ndim - 1:
                mask = mask.unsqueeze(-1)
            a = a * mask.to(dtype=a.dtype)
        return a


class PairTransition(CudnnGraphModule):
    def __init__(
        self,
        c_z: int,
        n: int,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        auto_chunk_policy: ChunkPolicy | None = None,
        enable_cudnn_graph: bool = False,
        cudnn_dynamic_shapes: bool = False,
    ):
        super().__init__()
        self.auto_chunk_policy = auto_chunk_policy
        self.enable_cudnn_graph = enable_cudnn_graph
        self.cudnn_dynamic_shapes = cudnn_dynamic_shapes
        self.dtype = dtype
        self.c_z = c_z
        self.n = n
        self._cudnn_graph_plans: dict[str, _CudnnPlan] = {}

        self.layer_norm = FusedLayerNorm(c_z, eps=eps, dtype=dtype)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.linear_2 = Linear(n * c_z, c_z, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.relu = nn.ReLU()
        self._prepare_cudnn_graphs()

    def _prepare_cudnn_graphs(self) -> None:
        self._cudnn_graph_plans.clear()
        if not self.cudnn_dynamic_shapes:
            return
        if not self.linear_1._weights_created or not self.linear_2._weights_created:
            return
        if not can_use_cudnn_graph(self.linear_1.weight, enabled=self.enable_cudnn_graph):
            return
        hidden_dim = self.n * self.c_z
        linear_relu = prepare_cudnn_linear_relu(
            self.linear_1.weight.device,
            self.linear_1.weight.dtype,
            self.c_z,
            hidden_dim,
        )
        linear_mask = prepare_cudnn_linear_mask(
            self.linear_2.weight.device,
            self.linear_2.weight.dtype,
            hidden_dim,
            self.c_z,
        )
        if linear_relu is not None:
            self._cudnn_graph_plans["linear_relu"] = linear_relu
        if linear_mask is not None:
            self._cudnn_graph_plans["linear_mask"] = linear_mask

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None, *, residual: bool = False):
        if self.auto_chunk_policy is not None and self.auto_chunk_policy.should_chunk(z):
            return chunk_apply(self._forward_impl, z, mask, policy=self.auto_chunk_policy, residual=residual)
        if can_use_cudnn_graph(z, enabled=self.enable_cudnn_graph):
            output = self._forward_cudnn(z, mask, residual=residual)
            if output is not None:
                return output
        return self._forward_impl(z, mask, residual=residual)

    def _forward_cudnn(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        residual: bool,
    ) -> torch.Tensor | None:
        old_z = z
        z = self.layer_norm(z)
        rows = z.numel() // self.c_z
        hidden_dim = self.n * self.c_z
        z_flat = z.reshape(1, rows, self.c_z)
        weight_1_t = self.linear_1.weight.unsqueeze(0).transpose(-1, -2)
        bias_1 = self.linear_1.bias.reshape(1, 1, hidden_dim)
        hidden = _run_cudnn_linear(
            self._cudnn_graph_plans.get("linear_relu"),
            cudnn_linear_relu,
            z_flat,
            weight_1_t,
            bias_1,
        )
        if hidden is None:
            return None
        if mask is None and not residual:
            return self.linear_2(hidden).view_as(z)

        weight_2_t = self.linear_2.weight.unsqueeze(0).transpose(-1, -2)
        bias_2 = self.linear_2.bias.reshape(1, 1, self.c_z)
        if residual:
            residual_flat = old_z.reshape(1, rows, self.c_z)
            if mask is None:
                output = cudnn_linear_residual(hidden, weight_2_t, bias_2, residual_flat)
            else:
                mask_flat = mask.reshape(1, rows, 1).to(dtype=z.dtype)
                output = cudnn_linear_mask_residual(hidden, weight_2_t, bias_2, mask_flat, residual_flat)
        else:
            mask_flat = mask.reshape(1, rows, 1).to(dtype=z.dtype)
            output = _run_cudnn_linear(
                self._cudnn_graph_plans.get("linear_mask"),
                cudnn_linear_mask,
                hidden,
                weight_2_t,
                bias_2,
                mask_flat,
            )
        return None if output is None else output.view_as(z)

    def _forward_impl(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        residual: bool = False,
    ) -> torch.Tensor:
        old_z = z
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        if mask is not None:
            if mask.ndim == z.ndim - 1:
                mask = mask.unsqueeze(-1)
            z = z * mask.to(dtype=z.dtype)
        if residual:
            z = z + old_z

        return z


class MSATransition(CudnnGraphModule):
    def __init__(
        self,
        c_m: int,
        n: int,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        enable_cudnn_graph: bool = False,
        cudnn_dynamic_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.enable_cudnn_graph = enable_cudnn_graph
        self.cudnn_dynamic_shapes = cudnn_dynamic_shapes
        self.dtype = dtype
        self.c_m = c_m
        self.n = n
        self._cudnn_graph_plans: dict[str, _CudnnPlan] = {}

        self.layer_norm = nn.LayerNorm(c_m, eps=eps, dtype=dtype)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.linear_2 = Linear(n * c_m, c_m, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.relu = nn.ReLU()
        self._prepare_cudnn_graphs()

    def _prepare_cudnn_graphs(self) -> None:
        self._cudnn_graph_plans.clear()
        if not self.cudnn_dynamic_shapes:
            return
        if not self.linear_1._weights_created or not self.linear_2._weights_created:
            return
        if not can_use_cudnn_graph(self.linear_1.weight, enabled=self.enable_cudnn_graph):
            return
        hidden_dim = self.n * self.c_m
        linear_relu = prepare_cudnn_linear_relu(
            self.linear_1.weight.device,
            self.linear_1.weight.dtype,
            self.c_m,
            hidden_dim,
        )
        linear_mask = prepare_cudnn_linear_mask(
            self.linear_2.weight.device,
            self.linear_2.weight.dtype,
            hidden_dim,
            self.c_m,
        )
        if linear_relu is not None:
            self._cudnn_graph_plans["linear_relu"] = linear_relu
        if linear_mask is not None:
            self._cudnn_graph_plans["linear_mask"] = linear_mask

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if can_use_cudnn_graph(m, enabled=self.enable_cudnn_graph):
            output = self._forward_cudnn(m, mask)
            if output is not None:
                return output
        return self._forward_impl(m, mask)

    def _forward_cudnn(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        m = self.layer_norm(m)
        rows = m.numel() // self.c_m
        hidden_dim = self.n * self.c_m
        m_flat = m.reshape(1, rows, self.c_m).contiguous()
        mask_flat = mask.reshape(1, rows, 1).to(dtype=m.dtype).contiguous()
        weight_1_t = self.linear_1.weight.unsqueeze(0).transpose(-1, -2)
        bias_1 = self.linear_1.bias.reshape(1, 1, hidden_dim)
        hidden = _run_cudnn_linear(
            self._cudnn_graph_plans.get("linear_relu"),
            cudnn_linear_relu,
            m_flat,
            weight_1_t,
            bias_1,
        )
        if hidden is None:
            return None
        weight_2_t = self.linear_2.weight.unsqueeze(0).transpose(-1, -2)
        bias_2 = self.linear_2.bias.reshape(1, 1, self.c_m)
        output = _run_cudnn_linear(
            self._cudnn_graph_plans.get("linear_mask"),
            cudnn_linear_mask,
            hidden,
            weight_2_t,
            bias_2,
            mask_flat,
        )
        return None if output is None else output.view_as(m)

    def _forward_impl(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)
        m = m * mask

        return m
