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
"""Transition (feed-forward) layers.

Each layer runs a call on the first path that accepts it:

1. ``transition_mlp_w_norm``: the LayerNorm, both linears, the mask and the residual in one kernel
   (:class:`_LayerNormFusedPath`), where it ships for the layer's shape.
2. The LayerNorm, then ``transition_mlp``: both linears, the mask and the residual in one kernel with the hidden
   activation on chip.
3. cuDNN graphs (ReLU layers, when enabled).
4. The PyTorch modules.
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from bionemo_ir._torch.custom_ops.dual_gemm_x_x import get_cute_dual_gemm_x_x_min_rows, get_cute_dual_gemm_x_x_op
from bionemo_ir._torch.custom_ops.gated_sigmoid import get_gated_sigmoid_op
from bionemo_ir._torch.custom_ops.transition_mlp import get_transition_mlp_op
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import (
    TransitionMlpWNormOp,
    TransitionMlpWNormPack,
    get_transition_mlp_w_norm_op,
    pack_weights,
)
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
from bionemo_ir._torch.utils import ChunkPolicy, chunk_apply, iter_chunks
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


def _refold_after_load_state_dict(layer: "_LayerNormFusedPath", *args: object) -> None:
    """Refold the weights ``load_state_dict`` loaded, as ``recursive_calling_load_weights`` does through the layer."""
    layer.post_load_weights()


def _row_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """``mask`` as the LayerNorm-fused kernel reads it: one value per row of ``x``, ``[*x.shape[:-1]]`` in its dtype.

    The unfused paths scale each row's update by ``mask``, with or without a trailing axis of size 1, broadcast over
    the rows. ``None`` for a mask that does not hold one value per row.
    """
    rows = x.shape[:-1]
    if mask.ndim == x.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != len(rows) or any(size not in (1, count) for size, count in zip(mask.shape, rows, strict=True)):
        return None
    return mask.expand(rows).to(x.dtype).contiguous()


class _LayerNormFusedPath:
    """A transition layer's LayerNorm and both linears as one ``transition_mlp_w_norm`` kernel, where it ships.

    The kernel reads the input once and writes ``[x +] mask * update`` once; the LayerNorm folds into the first
    linear's weights (``pack_weights``). A layer on the kernel folds them whenever its weights exist: at construction,
    where they hold placeholders until ``load_weights`` or ``load_state_dict`` fills them in place, again in
    :meth:`post_load_weights`, which ``load_state_dict`` also runs, and for a call from another device or after any
    weight it folded changed (a load into one child, an in-place edit). The kernel holds no activation beyond its
    output, so it takes a whole input that the layer's chunk policy would split.
    """

    _w_norm_op: TransitionMlpWNormOp | None
    _w_norm_pack: TransitionMlpWNormPack | None
    _w_norm_state: tuple[tuple[int, int], ...] | None
    _w_norm_activation: str

    def _w_norm_parts(self) -> tuple[nn.LayerNorm, Linear, Linear]:
        """The LayerNorm and the two linears the kernel runs."""
        raise NotImplementedError

    def _init_w_norm(self, op: TransitionMlpWNormOp | None, activation: str) -> None:
        """Take the kernel's op, ``None`` when it does not ship for the layer, and fold the weights for it."""
        self._w_norm_op = op
        self._w_norm_activation = activation
        self._w_norm_pack = self._fold_w_norm_weights()
        self._w_norm_state = self._w_norm_weight_state()
        self.register_load_state_dict_post_hook(_refold_after_load_state_dict)

    def _w_norm_weight_state(self) -> tuple[tuple[int, int], ...] | None:
        """Each folded weight's address and in-place write count, or ``None`` where versions are not tracked.

        A load into one child module, an in-place edit or a replaced parameter changes it. Inference tensors keep no
        version counter; for them the pack refolds only as :meth:`post_load_weights` and device moves do.
        """
        if self._w_norm_op is None:
            return None
        state = []
        try:
            for module in self._w_norm_parts():
                parameters = module._parameters  # Module.__getattr__ costs about as much as the rest of the check
                for name in ("weight", "bias"):
                    weight = parameters.get(name)
                    if weight is not None:
                        state.append((weight.data_ptr(), weight._version))
        except RuntimeError:
            return None
        return tuple(state)

    @torch.no_grad()
    def _fold_w_norm_weights(self, device: torch.device | None = None) -> TransitionMlpWNormPack | None:
        """The weights folded on ``device`` (default: the first linear's), or ``None`` off the kernel or without them.

        The LayerNorm may still be on the CPU while the linears are on the GPU; the fold reads every weight on
        ``device``.
        """
        norm, linear_1, linear_2 = self._w_norm_parts()
        if self._w_norm_op is None or not (linear_1._weights_created and linear_2._weights_created):
            return None
        if device is None:
            device = linear_1.weight.device

        def on(weight: torch.Tensor | None) -> torch.Tensor | None:
            return None if weight is None else weight.to(device)

        return pack_weights(
            on(norm.weight),
            on(norm.bias),
            on(linear_1.weight),
            on(linear_1.bias),
            on(linear_2.weight),
            on(linear_2.bias),
            activation=self._w_norm_activation,
        )

    def post_load_weights(self) -> None:
        """Refold the LayerNorm-fused kernel's weights from the loaded ones."""
        self._w_norm_pack = self._fold_w_norm_weights()
        self._w_norm_state = self._w_norm_weight_state()

    def _forward_w_norm(
        self, x: torch.Tensor, mask: torch.Tensor | None, *, residual: bool, inplace: bool = False
    ) -> torch.Tensor | None:
        """``mask * update``, or ``x`` plus it with ``residual``, from the kernel; ``None`` if it declines the call.

        With ``residual`` and ``inplace`` the kernel writes the sum into ``x``.
        """
        op = self._w_norm_op
        if op is None:
            return None
        row_mask = None if mask is None else _row_mask(x, mask)
        if (mask is not None and row_mask is None) or not op.accepts(x, row_mask):
            return None
        pack = self._w_norm_pack
        state = self._w_norm_weight_state()
        if pack is None or pack.device != x.device or state != self._w_norm_state:
            pack = self._fold_w_norm_weights(x.device)
            if pack is None:
                return None
            # A fold under graph capture is filled only when the graph replays: this call uses it, later ones refold.
            if not torch.cuda.is_current_stream_capturing():
                self._w_norm_pack, self._w_norm_state = pack, state
        eps = self._w_norm_parts()[0].eps
        return op(x, row_mask, pack, eps, residual=residual, out=x if residual and inplace else None)


class Transition(_LayerNormFusedPath, nn.Module):
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

        A layer with its LayerNorm and an output as wide as its input runs as one ``transition_mlp_w_norm`` kernel
        where it ships for ``(dim, hidden)``; see :class:`_LayerNormFusedPath`.

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
        self.fc3 = Linear(hidden, out_dim, dtype=dtype, bias=False, skip_create_weights=skip_create_weights)
        # Both fused kernels write an output as wide as their input.
        op_dtype = dtype or torch.get_default_dtype()
        self._fused_mlp_ops = {
            (has_mask, has_residual): get_transition_mlp_op(
                op_dtype,
                width=dim,
                hidden=hidden,
                activation="silu_gate",
                has_bias=False,
                has_mask=has_mask,
                has_residual=has_residual,
            )
            for has_mask in (False, True)
            for has_residual in (False, True)
            if out_dim == dim
        }
        w_norm_op = None
        if normalize and out_dim == dim:
            w_norm_op = get_transition_mlp_w_norm_op(op_dtype, width=dim, hidden=hidden, activation="silu_gate")
        self._init_w_norm(w_norm_op, "silu_gate")

    def _w_norm_parts(self) -> tuple[nn.LayerNorm, Linear, Linear]:
        return self.norm, self.fused_fc2_fc1, self.fc3

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        residual: bool = False,
        inplace: bool = False,
    ) -> torch.Tensor:
        """The SwiGLU update, or ``x`` plus it when ``residual`` is set.

        Args:
            inplace: With ``residual``, write the sum into ``x`` as ``x += update`` would.
        """
        output = self._forward_w_norm(x, mask, residual=residual, inplace=inplace)
        if output is not None:
            return output
        policy = self.auto_chunk_policy
        if policy is None or not policy.should_chunk(x):
            return self._forward_slice(x, mask, residual=residual, inplace=inplace)
        if residual and inplace:
            # Write each slice back directly; chunk_apply would allocate another full output.
            for start, length in iter_chunks(x.shape[policy.dim], policy.chunk_size):
                part = x.narrow(policy.dim, start, length)
                part_mask = mask
                if mask is not None and mask.ndim > policy.dim and mask.shape[policy.dim] == x.shape[policy.dim]:
                    part_mask = mask.narrow(policy.dim, start, length)
                self._forward_slice(part, part_mask, residual=True, inplace=True)
            return x
        if self._fused_mlp_ops.get((mask is not None, residual)) is None:
            # Keep one full residual add when every slice stays unfused.
            update = chunk_apply(self._forward_impl, x, mask, policy=policy)
            return x + update if residual else update
        return chunk_apply(self._forward_slice, x, mask, policy=policy, residual=residual, inplace=inplace)

    def _forward_slice(
        self, x: torch.Tensor, mask: torch.Tensor | None, *, residual: bool, inplace: bool
    ) -> torch.Tensor:
        """Run one dense slice, including its residual, without rechecking the chunk policy."""
        output = self._forward_fused(x, mask, residual=residual, inplace=inplace)
        if output is not None:
            return output
        update = self._forward_impl(x, mask)
        if not residual:
            return update
        return x.add_(update) if inplace else x + update

    def _forward_fused(
        self, x: torch.Tensor, mask: torch.Tensor | None, *, residual: bool, inplace: bool
    ) -> torch.Tensor | None:
        """Normalize, then one kernel for the SwiGLU, ``fc3``, the mask and any residual.

        The hidden activation stays on chip. Returns ``None`` when the call needs the unfused path.
        """
        op = self._fused_mlp_ops.get((mask is not None, residual))
        if op is None:
            return None
        weights = (self.fused_fc2_fc1.weight, None, self.fc3.weight, None)
        if not op.accepts(x, mask, *weights):
            return None
        normed = x if self.norm is None else self.norm(x)
        if not residual:
            return op(normed, *weights, None, mask)
        output = op(normed, *weights, x, mask, out=x if inplace and x.is_contiguous() else None)
        if output is not None and inplace and output is not x:
            x.copy_(output)
            return x
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc3(self._swiglu(self.fused_fc2_fc1(x)))

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
        self.using_silu = using_silu
        # Value rows, then gate rows, then (3-way only) the second value's rows.
        self.fused_swl_a_to_b = Linear(
            self.dim_single,
            (2 if using_silu else 3) * self.dim_inner,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR if using_silu else WeightMode.FUSED_QKV_LINEAR
            ),
        )
        self._swiglu = FusedSwiGLU(d=self.dim_inner, three_way=not using_silu, dtype=dtype or torch.bfloat16)
        self.b_to_a = Linear(
            self.dim_inner, self.dim_single, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
        )
        # The SwiGLU MLP runs on one kernel path: the fused transition op, which keeps the hidden
        # activation on chip where a config ships for this width, or else, for a 2-way SwiGLU, the
        # dual GEMM for the gated projection ahead of b_to_a. Below the dual GEMM's tuned row count,
        # where its fixed cost dominates, the separate projection and SwiGLU pass run instead.
        self._fused_mlp_op = get_transition_mlp_op(
            dtype or torch.get_default_dtype(),
            width=self.dim_single,
            hidden=self.dim_inner,
            activation="silu_gate" if using_silu else "silu_gate_3way",
            has_bias=False,
            has_mask=False,
            has_residual=False,
        )
        self._dual_gemm_silu_op = None
        self._dual_gemm_min_rows = 0
        if using_silu:
            self._dual_gemm_silu_op = _get_silu_projection_op(dtype, K=self.dim_single, N=self.dim_inner)
            if self._dual_gemm_silu_op is not None:
                self._dual_gemm_min_rows = get_cute_dual_gemm_x_x_min_rows(K=self.dim_single, N=self.dim_inner)

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
        residual: torch.Tensor | None = None,
        excluded: torch.Tensor | None = None,
        mask_value: torch.Tensor | None = None,
        projected_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]
            buffers: optional preallocated buffer dict, forwarded to AdaLN.
            buffer_key: key into ``buffers`` for the AdaLN output tensor.
            mask: optional binary mask ``[B, I]`` applied after AdaLN and
                after the output gate.
            residual: optional tensor shaped like ``a``, added to the output.
            excluded: optional ``~mask`` as a bool ``[B, I, 1]`` for the
                AdaLN, precomputed by a caller that masks many layers.
            mask_value: optional ``mask`` as ``[B, I, 1]`` in ``a``'s dtype,
                precomputed likewise; the output gate then skips its cast.
            projected_condition: the AdaLN's ``project_condition(s)``,
                precomputed.

        Returns:
            a: [B, I, d]
        """
        a = self.adaln(
            a,
            s,
            buffers=buffers,
            buffer_key=buffer_key,
            mask=mask,
            excluded=excluded,
            projected_condition=projected_condition,
        )
        a = self._swiglu_mlp(a)

        # The gated-sigmoid op broadcasts `s` (gate) across the multiplicity
        # dim of `a` when their leading shapes differ, and applies the mask and
        # residual in the same pass. Without a residual, reuse the AdaLN output
        # buffer: the MLP consumed it above. A residual sum outlives the next
        # block's AdaLN, so it gets its own tensor.
        output = buffers.get(buffer_key) if buffers is not None and residual is None else None
        return self._gated_sigmoid_op(
            s,
            self.output_projection.weight,
            a,
            self.output_projection.bias,
            output=output,
            residual=residual,
            mask=mask if mask_value is None else mask_value,
        )

    def _swiglu_mlp(self, a: torch.Tensor) -> torch.Tensor:
        """``b_to_a(swiglu(fused_swl_a_to_b(a)))`` on the chosen kernel path, else unfused."""
        weight = self.fused_swl_a_to_b.weight
        if self._fused_mlp_op is not None:
            weights = (weight, None, self.b_to_a.weight, None)
            if self._fused_mlp_op.accepts(a, None, *weights):
                output = self._fused_mlp_op(a, *weights, None, None)
                if output is not None:
                    return output
        if self._dual_gemm_silu_op is not None and a.numel() >= self._dual_gemm_min_rows * self.dim_single:
            # The dual GEMM gates its first weight; the projection holds value rows, then gate rows.
            b = self._dual_gemm_silu_op(a, weight[self.dim_inner :], weight[: self.dim_inner], gate="silu")
        else:
            b = self._swiglu(self.fused_swl_a_to_b(a))
        return self.b_to_a(b)


class _ReluTransition(_LayerNormFusedPath, CudnnGraphModule):
    """``linear_2(relu(linear_1(LayerNorm(x))))``, both linears with biases, over ``[..., width]``.

    The body :class:`PairTransition` and :class:`MSATransition` share. It runs as one ``transition_mlp_w_norm`` kernel
    where it ships for ``(width, n * width)``; see :class:`_LayerNormFusedPath`.
    """

    def __init__(
        self,
        width: int,
        n: int,
        layer_norm: nn.Module,
        *,
        dtype: torch.dtype | None,
        skip_create_weights: bool,
        enable_cudnn_graph: bool,
        cudnn_dynamic_shapes: bool,
    ) -> None:
        super().__init__()
        self.enable_cudnn_graph = enable_cudnn_graph
        self.cudnn_dynamic_shapes = cudnn_dynamic_shapes
        self.dtype = dtype
        self.n = n
        self._width = width
        self._cudnn_graph_plans: dict[str, _CudnnPlan] = {}

        self.layer_norm = layer_norm
        self.linear_1 = Linear(width, n * width, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.linear_2 = Linear(n * width, width, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
        self.relu = nn.ReLU()
        op_dtype = dtype or torch.get_default_dtype()
        self._fused_mlp_ops = {
            has_residual: get_transition_mlp_op(op_dtype, width=width, hidden=n * width, has_residual=has_residual)
            for has_residual in (False, True)
        }
        self._init_w_norm(
            get_transition_mlp_w_norm_op(op_dtype, width=width, hidden=n * width, activation="relu"), "relu"
        )
        self._prepare_cudnn_graphs()

    def _w_norm_parts(self) -> tuple[nn.LayerNorm, Linear, Linear]:
        return self.layer_norm, self.linear_1, self.linear_2

    def _prepare_cudnn_graphs(self) -> None:
        self._cudnn_graph_plans.clear()
        if not self.cudnn_dynamic_shapes:
            return
        if not self.linear_1._weights_created or not self.linear_2._weights_created:
            return
        if not can_use_cudnn_graph(self.linear_1.weight, enabled=self.enable_cudnn_graph):
            return
        hidden_dim = self.n * self._width
        linear_relu = prepare_cudnn_linear_relu(
            self.linear_1.weight.device,
            self.linear_1.weight.dtype,
            self._width,
            hidden_dim,
        )
        linear_mask = prepare_cudnn_linear_mask(
            self.linear_2.weight.device,
            self.linear_2.weight.dtype,
            hidden_dim,
            self._width,
        )
        if linear_relu is not None:
            self._cudnn_graph_plans["linear_relu"] = linear_relu
        if linear_mask is not None:
            self._cudnn_graph_plans["linear_mask"] = linear_mask

    def _forward_fused(self, x: torch.Tensor, mask: torch.Tensor | None, *, residual: bool) -> torch.Tensor | None:
        """LayerNorm, then one ``transition_mlp`` kernel for both linears, the mask and any residual; ``None`` if
        declined.

        The hidden activation stays on chip, which saves most of the layer's DRAM traffic. The op takes masked calls
        only.
        """
        op = self._fused_mlp_ops[residual]
        weights = (self.linear_1.weight, self.linear_1.bias, self.linear_2.weight, self.linear_2.bias)
        if op is None or mask is None or not op.accepts(x, mask, *weights):
            return None
        return op(self.layer_norm(x), *weights, x if residual else None, mask)

    def _cudnn_hidden(self, normed: torch.Tensor) -> torch.Tensor | None:
        """``relu(linear_1(normed))`` as ``[1, rows, n * width]`` from cuDNN, or ``None`` if it declines."""
        rows = normed.numel() // self._width
        return _run_cudnn_linear(
            self._cudnn_graph_plans.get("linear_relu"),
            cudnn_linear_relu,
            normed.reshape(1, rows, self._width).contiguous(),
            self.linear_1.weight.unsqueeze(0).transpose(-1, -2),
            self.linear_1.bias.reshape(1, 1, self.n * self._width),
        )

    def _cudnn_linear_2(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``linear_2``'s weight and bias in the layout the cuDNN graphs take."""
        return self.linear_2.weight.unsqueeze(0).transpose(-1, -2), self.linear_2.bias.reshape(1, 1, self._width)


class PairTransition(_ReluTransition):
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
        """ReLU pair transition ``linear_2(relu(linear_1(LayerNorm(z))))`` over ``[*, N, N, c_z]``.

        It runs as one ``transition_mlp_w_norm`` kernel where it ships for ``(c_z, n * c_z)``; see
        :class:`_LayerNormFusedPath`.
        """
        super().__init__(
            c_z,
            n,
            FusedLayerNorm(c_z, eps=eps, dtype=dtype),
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            enable_cudnn_graph=enable_cudnn_graph,
            cudnn_dynamic_shapes=cudnn_dynamic_shapes,
        )
        self.c_z = c_z
        self.auto_chunk_policy = auto_chunk_policy

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None, *, residual: bool = False):
        output = self._forward_w_norm(z, mask, residual=residual)
        if output is not None:
            return output
        if self.auto_chunk_policy is not None and self.auto_chunk_policy.should_chunk(z):
            return chunk_apply(self._forward_slice, z, mask, policy=self.auto_chunk_policy, residual=residual)
        return self._forward_slice(z, mask, residual=residual)

    def _forward_slice(self, z: torch.Tensor, mask: torch.Tensor | None, *, residual: bool) -> torch.Tensor:
        """Run a dense slice through fused, cuDNN, or ordinary linears."""
        output = self._forward_fused(z, mask, residual=residual)
        if output is not None:
            return output
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
        normed = self.layer_norm(z)
        hidden = self._cudnn_hidden(normed)
        if hidden is None:
            return None
        if mask is None and not residual:
            return self.linear_2(hidden).view_as(normed)

        rows = normed.numel() // self.c_z
        weight_2_t, bias_2 = self._cudnn_linear_2()
        if residual:
            residual_flat = z.reshape(1, rows, self.c_z)
            if mask is None:
                output = cudnn_linear_residual(hidden, weight_2_t, bias_2, residual_flat)
            else:
                mask_flat = mask.reshape(1, rows, 1).to(dtype=normed.dtype)
                output = cudnn_linear_mask_residual(hidden, weight_2_t, bias_2, mask_flat, residual_flat)
        else:
            mask_flat = mask.reshape(1, rows, 1).to(dtype=normed.dtype)
            output = _run_cudnn_linear(
                self._cudnn_graph_plans.get("linear_mask"),
                cudnn_linear_mask,
                hidden,
                weight_2_t,
                bias_2,
                mask_flat,
            )
        return None if output is None else output.view_as(normed)

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


class MSATransition(_ReluTransition):
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
        """ReLU MSA transition ``linear_2(relu(linear_1(LayerNorm(m))))`` over ``[*, S, N, c_m]``.

        It runs as one ``transition_mlp_w_norm`` kernel where it ships for ``(c_m, n * c_m)``; see
        :class:`_LayerNormFusedPath`.
        """
        super().__init__(
            c_m,
            n,
            nn.LayerNorm(c_m, eps=eps, dtype=dtype),
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            enable_cudnn_graph=enable_cudnn_graph,
            cudnn_dynamic_shapes=cudnn_dynamic_shapes,
        )
        self.c_m = c_m

    def forward(self, m: torch.Tensor, mask: torch.Tensor, *, residual: bool = False) -> torch.Tensor:
        """The masked MLP update, or ``m`` plus it when ``residual`` is set."""
        output = self._forward_w_norm(m, mask, residual=residual)
        if output is None:
            output = self._forward_fused(m, mask, residual=residual)
        if output is not None:
            return output
        if can_use_cudnn_graph(m, enabled=self.enable_cudnn_graph):
            output = self._forward_cudnn(m, mask)
        if output is None:
            output = self._forward_impl(m, mask)
        return m + output if residual else output

    def _forward_cudnn(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        normed = self.layer_norm(m)
        rows = normed.numel() // self.c_m
        mask_flat = mask.reshape(1, rows, 1).to(dtype=normed.dtype).contiguous()
        hidden = self._cudnn_hidden(normed)
        if hidden is None:
            return None
        weight_2_t, bias_2 = self._cudnn_linear_2()
        output = _run_cudnn_linear(
            self._cudnn_graph_plans.get("linear_mask"),
            cudnn_linear_mask,
            hidden,
            weight_2_t,
            bias_2,
            mask_flat,
        )
        return None if output is None else output.view_as(normed)

    def _forward_impl(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)
        m = m * mask

        return m
