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

"""Reusable inference blocks accepting caller-defined tensor masks."""

from typing import Literal

import torch
import torch.nn as nn

from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import PrecomputedPairMasks, PrecomputedSingleMasks
from bionemo_ir._torch.custom_ops.fused_masked_residual import fused_masked_residual
from bionemo_ir._torch.custom_ops.fused_pair_injection import PairInjection
from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock, PairTransition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleMultiplicationMetadata,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
    precompute_trimul_metadata,
)
from bionemo_ir._torch.utils import CHUNK_REGISTRY, PAIR_TRANSITION
from bionemo_ir.runtime.buffers import PreallocatedBuffers

__all__ = [
    "ConditionedTransformerBlock",
    "PairUpdateBlock",
    "TriangleMultiplicationMetadata",
    "precompute_trimul_metadata",
    "with_cuda_graph",
]


def _apply_masked_residual(
    value: torch.Tensor,
    update: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask_value is None:
        mask_value = mask.unsqueeze(-1).to(value.dtype)
    output = fused_masked_residual(value, update, mask_value)
    if output is not None:
        return output
    return (value + update).masked_fill(~mask_value.bool(), 0)


class ConditionedTransformerBlock(nn.Module):
    """Inference-only conditioned attention and transition with masked residuals.

    The caller supplies a boolean valid-token mask; no token semantics are
    inferred. ``sanitize_masked_inputs`` removes non-finite values from invalid
    positions. Callers that already maintain exact-zero padding may disable it
    to avoid copying the pair representation on every layer.
    """

    def __init__(
        self,
        layer_idx: int,
        token_dim: int,
        pair_dim: int,
        condition_dim: int,
        num_heads: int,
        transition_expansion: int,
        backend: str,
        mask_inf: float,
        use_qk_norm: bool,
        eps: float,
        dtype: torch.dtype,
        norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
        cond_norm_bias: bool = False,
        use_pair_bias: bool = True,
        mask_left_aligned: bool = False,
        sanitize_masked_inputs: bool = True,
        require_inference_mode: bool = True,
    ) -> None:
        super().__init__()
        if backend == "CuTeDSL" and not use_pair_bias:
            raise ValueError("CuTeDSL attention requires pair bias; select SDPA for mask-only attention")
        self.backend = backend
        self.layer_idx = layer_idx
        self.use_pair_bias = use_pair_bias
        self.sanitize_masked_inputs = sanitize_masked_inputs
        self.require_inference_mode = require_inference_mode
        self.adaln = AdaLN(
            token_dim, condition_dim, eps=eps, dtype=dtype, norm_type=norm_type, cond_norm_bias=cond_norm_bias
        )
        self.mha = AttentionPairBias(
            layer_idx=layer_idx,
            c_s=token_dim,
            c_z=pair_dim,
            num_heads=num_heads,
            initial_norm=True,
            norm_type=norm_type,
            bias_proj=use_pair_bias,
            pair_norm=True,
            pair_norm_type=norm_type,
            use_qk_norm=use_qk_norm,
            kv_bias=True,
            out_bias=True,
            output_gate_dim=condition_dim,
            use_ada_layer_norm=False,
            inf=mask_inf,
            eps=eps,
            dtype=dtype,
            attn_backend=backend,
            mask_left_aligned=mask_left_aligned,
        )
        # The attention's output gate is the adaLN-zero gate, so the epilogue applies it with the residual add.
        nn.init.zeros_(self.mha.output_projection.weight)
        nn.init.constant_(self.mha.output_projection.bias, -2.0)
        self.transition = ConditionedTransitionBlock(
            dim_single=token_dim,
            dim_single_cond=condition_dim,
            expansion_factor=transition_expansion,
            eps=eps,
            dtype=dtype,
            using_silu=True,
            norm_type=norm_type,
            cond_norm_bias=cond_norm_bias,
            output_gate_bias_init=-2.0,
        )

    def project_condition(self, condition: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``condition`` ``[B, N, condition_dim]`` for the attention and transition AdaLNs.

        A trunk that runs this block on several passes under one condition projects it once
        and passes the pair as ``forward(projected_condition=...)``.
        """
        if self.sanitize_masked_inputs:
            condition = condition.masked_fill(~mask.unsqueeze(-1), 0)
        return self.adaln.project_condition(condition), self.transition.adaln.project_condition(condition)

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor | None,
        condition: torch.Tensor,
        mask: torch.Tensor,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        projected_condition: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run conditioned attention, then the conditioned transition, each with a masked residual.

        ``precomputed_single_masks`` may carry ``excluded`` and ``value_mask`` forms of
        ``mask``; ``projected_condition`` is :meth:`project_condition` of ``condition``.
        A trunk computes both once instead of in every layer.
        """
        if self.require_inference_mode and torch.is_grad_enabled():
            raise RuntimeError("BioIR blocks require no_grad or inference_mode; use native blocks for gradients")
        if self.sanitize_masked_inputs:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            condition = condition.masked_fill(~mask.unsqueeze(-1), 0)
            if pair is not None:
                pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                pair = pair.masked_fill(~pair_mask.unsqueeze(-1), 0)
        excluded = value_mask = mask_bias = None
        if precomputed_single_masks is not None:
            excluded = precomputed_single_masks.excluded
            value_mask = precomputed_single_masks.value_mask
            mask_bias = precomputed_single_masks.mask_bias
        attention_condition, transition_condition = projected_condition or (None, None)
        residual = x
        attended = self.adaln(x, condition, mask=mask, excluded=excluded, projected_condition=attention_condition)
        # Where tuned, one epilogue kernel gates the attention output, projects it, applies the adaLN-zero gate
        # and adds the residual. Masked rows hold residual plus update until the transition's masked residual
        # zeroes them.
        x = self.mha(
            attended,
            pair,
            mask,
            single_embedding=condition,
            attn_metadata=metadata,
            mask_bias=mask_bias,
            buffers=buffers,
            residual=residual,
        )

        residual = x
        x = self.transition(
            x,
            condition,
            mask=mask,
            excluded=excluded,
            mask_value=value_mask,
            projected_condition=transition_condition,
        )
        return _apply_masked_residual(x, residual, mask, value_mask)


class PairUpdateBlock(nn.Module):
    """Sequence-to-pair injection, triangle multiplication, and pair transition.

    A :class:`PrecomputedPairMasks` bundle supplies the outgoing and incoming
    trimul row lengths so every pair-update layer avoids repeating mask
    reductions.
    """

    def __init__(
        self,
        layer_idx: int,
        token_dim: int,
        pair_dim: int,
        trimul_hidden_dim: int,
        pair_transition_expansion: int,
        pair_mask_left_aligned: bool,
        trimul_high_precision: bool,
        eps: float,
        dtype: torch.dtype,
        pair_write: Literal["sum", "hadamard"] = "sum",
        enable_cudnn_graph: bool = True,
        use_tri_mult: bool = True,
        trimul_bias: bool = True,
    ) -> None:
        super().__init__()
        if pair_write not in ("sum", "hadamard"):
            raise ValueError(f"pair_write must be 'sum' or 'hadamard', got {pair_write!r}")
        self.pair_write = pair_write
        self.pair_mask_left_aligned = pair_mask_left_aligned
        self.enable_cudnn_graph = enable_cudnn_graph
        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = nn.LayerNorm(token_dim, eps=eps, dtype=dtype)
        self.linear_x = nn.Linear(token_dim, 2 * pair_dim, bias=False, dtype=dtype)
        trimul_kwargs = {
            "layer_idx": layer_idx,
            "dim": pair_dim,
            "hidden_dim": trimul_hidden_dim,
            "eps": eps,
            "bias_flags": {
                "p_in": trimul_bias,
                "g_in": trimul_bias,
                "p_out": trimul_bias,
                "g_out": trimul_bias,
            },
            "norm_bias": trimul_bias,
            "dtype": dtype,
            "high_precision": trimul_high_precision,
            "mean_normalization": False,
            "pair_mask_left_aligned": pair_mask_left_aligned,
        }
        if use_tri_mult:
            self.tri_mult_out = TriangleMultiplicationNode(
                multiplication_type=TriangleMultiplicationNodeType.OUTGOING, **trimul_kwargs
            )
            self.tri_mult_in = TriangleMultiplicationNode(
                multiplication_type=TriangleMultiplicationNodeType.INCOMING, **trimul_kwargs
            )
        self.transition_out = PairTransition(
            c_z=pair_dim,
            n=pair_transition_expansion,
            eps=eps,
            dtype=dtype,
            auto_chunk_policy=CHUNK_REGISTRY.get(PAIR_TRANSITION),
            enable_cudnn_graph=enable_cudnn_graph,
        )
        self.pair_injection = PairInjection(
            pair_dim=pair_dim,
            pair_write=pair_write,
            dtype=dtype or torch.bfloat16,
        )

    def _masked_residual(
        self,
        pair: torch.Tensor,
        update: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate ``update`` into ``pair`` and re-apply ``mask``.

        The Triton kernel accepts the strided ``update`` a token-padded trimul
        returns. ``pair`` must be storage this update owns, because both the
        fused path and the eager fallback accumulate in place.
        """
        fused = fused_masked_residual(pair, update, mask)
        if fused is not None:
            return fused
        return pair.add_(update).masked_fill_(~mask.bool(), 0)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
        trimul_metadata: TriangleMultiplicationMetadata,
        precomputed_masks: PrecomputedPairMasks | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
    ) -> torch.Tensor:
        """Inject the single track into the pair, then run triangle multiplication and the pair transition.

        ``precomputed_masks.value_mask`` and ``precomputed_single_masks.excluded``, when a trunk
        computed them once, replace this block's own mask conversions.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("BioIR blocks require no_grad or inference_mode; use native blocks for gradients")
        pair_mask = (
            precomputed_masks.pair_mask if precomputed_masks is not None else mask.unsqueeze(-1) & mask.unsqueeze(-2)
        )
        pair_mask = pair_mask.to(torch.bool)

        excluded = precomputed_single_masks.excluded if precomputed_single_masks is not None else None
        single = single.masked_fill(~mask.unsqueeze(-1) if excluded is None else excluded, 0)
        row, column = self.linear_x(self.layer_norm_in(single)).chunk(2, dim=-1)
        pair_mask_value = precomputed_masks.value_mask if precomputed_masks is not None else None
        if pair_mask_value is None or pair_mask_value.dtype != pair.dtype:
            pair_mask_value = pair_mask.unsqueeze(-1).to(pair.dtype)

        # Returns storage this update owns, so the residuals below accumulate in
        # place without mutating the caller's pair representation.
        pair = self.pair_injection(pair, row, column, pair_mask_value)

        if self.use_tri_mult:
            if self.tri_mult_out.can_fuse_residual(pair):
                pair = self.tri_mult_out(
                    pair,
                    pair_mask,
                    trimul_metadata=trimul_metadata,
                    residual=True,
                )
            else:
                update = self.tri_mult_out(pair, pair_mask, trimul_metadata=trimul_metadata)
                pair = self._masked_residual(pair, update, pair_mask_value)
            if self.tri_mult_in.can_fuse_residual(pair):
                pair = self.tri_mult_in(
                    pair,
                    pair_mask,
                    trimul_metadata=trimul_metadata,
                    residual=True,
                )
            else:
                update = self.tri_mult_in(pair, pair_mask, trimul_metadata=trimul_metadata)
                pair = self._masked_residual(pair, update, pair_mask_value)
        return self.transition_out(pair, mask=pair_mask_value, residual=True)


def with_cuda_graph(block: nn.Module, *, max_graphs: int = 1) -> CUDAGraphOptimizationTracker:
    """Warm, verify and replay an inference block using bounded exact-shape graphs.

    Tensor values (including masks) are refreshed on every call. Outputs own their
    storage. Capture failures use the same eager block and are logged by the runtime.
    Call ``reset()`` before changing weights or moving an already-captured module.
    """
    if max_graphs < 1:
        raise ValueError("max_graphs must be positive")
    return CUDAGraphOptimizationTracker(
        CUDAGraphOptimizationConfig(
            graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            num_graphs_max_for_this_module=max_graphs,
            verify_capture=True,
        ),
        inner_module=block,
    )
