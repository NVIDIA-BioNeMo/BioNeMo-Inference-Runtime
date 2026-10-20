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

import torch
import torch.nn as nn

from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import (
    PrecomputedPairMasks,
    PrecomputedSingleMasks,
    precompute_pair_masks,
    precompute_single_masks,
)
from bionemo_ir._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from bionemo_ir._torch.graph_optimization.region import GraphRegion
from bionemo_ir._torch.layers.token_padding import pad_trunk_tokens, unpad_trunk_tokens
from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock
from bionemo_ir._torch.layers.transformers.diffusion_transformer import build_bias_mega_weight
from bionemo_ir._torch.layers.transition import Transition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleMultiplicationMetadata,
    precompute_trimul_metadata,
    set_trimul_token_padding,
)
from bionemo_ir.configs import BaseConfig
from bionemo_ir.runtime.buffers import PreallocatedBuffers


class RFProteinaTrunkLayer(ConditionedTransformerBlock):
    """Compatibility name for the RFProteína conditioned transformer block."""


class RFProteinaPairUpdate(PairUpdateBlock):
    """Compatibility name for the RFProteína pair-update block."""


# The streaming pair-bias kernel projects at most this many heads in one pass.
_MAX_SHARED_PAIR_BIAS_HEADS = 64


def _shared_pair_bias_groups(
    pair_update_layers: nn.ModuleList, num_layers: int, max_layers: int
) -> list[tuple[int, ...]]:
    """Runs of consecutive layers that read one pair tensor, at most ``max_layers`` long."""
    groups, current = [], [0]
    for i in range(1, num_layers):
        updated = i - 1 < len(pair_update_layers) and pair_update_layers[i - 1] is not None
        if updated or len(current) == max_layers:
            groups.append(tuple(current))
            current = []
        current.append(i)
    groups.append(tuple(current))
    return groups


class LocalLatentsTransformer(nn.Module):
    """RFProteína velocity trunk operating on precomputed representations.

    Feature construction remains an external boundary for the model-first
    integration. ``single``, ``pair``, and ``condition`` correspond to the
    outputs of RFProteina's sequence, pair, and conditioning factories.

    As in Pairformer, backend-specific pair and single mask payloads are
    computed once before the layer loop and shared by every consumer. With
    ``enable_token_pad``, the layer loop runs on token dims zero-padded to a
    multiple of 8, and the heads read the trunk output sliced back to ``N``.
    """

    def __init__(self, config: BaseConfig) -> None:
        """
        Args:
            config: a :class:`~bionemo_ir.configs.BaseConfig` carrying the
                RFProteína trunk fields (see
                ``bionemo_ir.models.rfproteina.config.RFProteinaTrunkConfig``):
                ``token_dim``, ``pair_dim``, ``condition_dim``, ``latent_dim``,
                ``num_layers``, ``num_heads``, the transition / pair-update
                expansions, ``use_qk_norm`` / ``use_insertion_head`` /
                ``enable_token_pad`` toggles, ``pairwise_attention_backend``,
                and ``dtype`` / ``norm_epsilon``.
        """
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.token_dim = config.token_dim

        cond_hidden = int(config.condition_dim * config.condition_transition_expansion)
        self.transition_c_1 = Transition(
            config.condition_dim,
            cond_hidden,
            dtype=self.dtype,
            normalize=False,
        )
        self.transition_c_2 = Transition(
            config.condition_dim,
            cond_hidden,
            dtype=self.dtype,
            normalize=False,
        )
        self.transformer_layers = nn.ModuleList(
            [
                RFProteinaTrunkLayer(
                    layer_idx=i,
                    token_dim=config.token_dim,
                    pair_dim=config.pair_dim,
                    condition_dim=config.condition_dim,
                    num_heads=config.num_heads,
                    transition_expansion=config.transition_expansion,
                    backend=config.pairwise_attention_backend,
                    mask_inf=config.mask_inf,
                    use_qk_norm=config.use_qk_norm,
                    eps=config.norm_epsilon,
                    dtype=self.dtype,
                    mask_left_aligned=config.pair_mask_left_aligned,
                    sanitize_masked_inputs=False,
                    require_inference_mode=False,
                )
                for i in range(config.num_layers)
            ]
        )
        self.pair_update_layers = nn.ModuleList(
            [
                RFProteinaPairUpdate(
                    layer_idx=i,
                    token_dim=config.token_dim,
                    pair_dim=config.pair_dim,
                    trimul_hidden_dim=config.trimul_hidden_dim,
                    pair_transition_expansion=config.pair_transition_expansion,
                    pair_write=config.pair_write,
                    pair_mask_left_aligned=config.pair_mask_left_aligned,
                    trimul_high_precision=config.trimul_high_precision,
                    eps=config.norm_epsilon,
                    dtype=self.dtype,
                    enable_cudnn_graph=config.enable_cudnn_graph,
                    trimul_bias=config.trimul_bias,
                )
                if i % config.pair_update_every == 0
                else None
                for i in range(config.num_layers - 1)
            ]
        )
        # Layers between two pair updates read one pair tensor, so one streaming pass projects all of their pair
        # biases. Each layer's RMSNorm gamma is folded into its projection in FP32, which reproduces the
        # per-layer kernel's own fold bit for bit.
        self._pair_bias_groups = _shared_pair_bias_groups(
            self.pair_update_layers, config.num_layers, max(1, _MAX_SHARED_PAIR_BIAS_HEADS // config.num_heads)
        )
        self._shared_pair_bias: dict[int, LNProjMoveaxisPad] = {}
        for size in {len(group) for group in self._pair_bias_groups if len(group) > 1}:
            projection = LNProjMoveaxisPad(
                D=config.pair_dim,
                H=size * config.num_heads,
                dtype=self.dtype,
                rms_norm=True,
                eps=config.norm_epsilon,
                fp32_projection=True,
            )
            if projection.streams(self.dtype):
                self._shared_pair_bias[size] = projection
        self._pair_bias_pad_multiple = 8 if config.pairwise_attention_backend == "CuTeDSL" else -1

        self.recycle_enabled = config.recycle_enabled
        if self.recycle_enabled:
            self.s_cycle_norm = nn.RMSNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype)
            self.z_cycle_norm = nn.RMSNorm(config.pair_dim, eps=config.norm_epsilon, dtype=self.dtype)
            self.s_cycle_proj = nn.Linear(config.token_dim, config.token_dim, bias=False, dtype=self.dtype)
            self.z_cycle_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False, dtype=self.dtype)

        self.local_latents_linear = nn.Sequential(
            nn.RMSNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
            nn.Linear(config.token_dim, config.latent_dim, bias=False, dtype=self.dtype),
        )
        self.ca_linear = nn.Sequential(
            nn.RMSNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
            nn.Linear(config.token_dim, 3, bias=False, dtype=self.dtype),
        )
        self.use_sequence_readout = config.use_sequence_readout
        if self.use_sequence_readout:
            self.sequence_readout = nn.Sequential(
                nn.RMSNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
                nn.Linear(config.token_dim, config.sequence_readout_classes, bias=False, dtype=self.dtype),
            )
        self.use_insertion_head = config.use_insertion_head
        if self.use_insertion_head:
            self.insertion_q_proj = nn.Linear(config.token_dim, config.token_dim, bias=False, dtype=self.dtype)
            self.insertion_k_proj = nn.Linear(config.token_dim, config.token_dim, bias=False, dtype=self.dtype)
            self._insertion_head_scale = config.token_dim**-0.5

        self.enable_token_pad = config.enable_token_pad
        self.token_pad_spec = config.token_pad_spec
        # forward() pads the tokens before any layer runs.
        set_trimul_token_padding(self, self.enable_token_pad)
        # One capture serves every recycle depth, because the repeated computation is the step
        # rather than the whole forward. ``configure_graphs`` supplies the policy.
        self.trunk_graph = GraphRegion(self, "trunk_step", None)

    def _fold_shared_pair_bias_weights(self) -> dict[int, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]]:
        """Each shared group, its FP32 folded projection and a unit gamma, keyed by the group's first layer.

        Rebuilt per call so weight loading and in-place edits are observed, as the decoder's bias precompute is;
        one fold covers every shared layer, and each group takes its rows.
        """
        groups = [group for group in self._pair_bias_groups if len(group) in self._shared_pair_bias]
        if not groups:
            return {}
        layers = [self.transformer_layers[i] for group in groups for i in group]
        folded = build_bias_mega_weight(layers, attn_attr="mha", dtype=torch.float32)  # [layers * heads, pair_dim]
        unit_gamma = torch.ones(folded.shape[-1], device=folded.device, dtype=torch.float32)
        shared, start = {}, 0
        for group in groups:
            rows = len(group) * self.config.num_heads
            shared[group[0]] = (group, folded[start : start + rows], unit_gamma)
            start += rows
        return shared

    def _project_shared_pair_biases(
        self, pair: torch.Tensor, group: tuple[int, ...], weight: torch.Tensor, unit_gamma: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Project a group's pair biases in one pass; an empty dict leaves each layer to project its own.

        Args:
            pair: ``[B, N, N, pair_dim]`` pair representation that the group's layers read.
            group: Indices of consecutive layers that read ``pair``.
            weight: The group's FP32 folded projection ``[len(group) * num_heads, pair_dim]``.
            unit_gamma: FP32 ones ``[pair_dim]``; the gammas are already in ``weight``.

        Returns:
            Each layer's ``[B, num_heads, N, N_padded]`` bias, a view of one projection.
        """
        if not (pair.is_cuda and pair.is_contiguous() and pair.dtype == self.dtype):
            return {}
        bias = self._shared_pair_bias[len(group)](
            pair, unit_gamma, None, weight, pad_multiple=self._pair_bias_pad_multiple
        )
        heads = self.config.num_heads
        return {layer: bias.narrow(-3, k * heads, heads) for k, layer in enumerate(group)}

    def trunk_step(
        self,
        *,
        single: torch.Tensor,
        pair: torch.Tensor,
        single_init: torch.Tensor,
        pair_init: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        trimul_metadata: TriangleMultiplicationMetadata,
        precomputed_masks: PrecomputedPairMasks | None,
        precomputed_single_masks: PrecomputedSingleMasks,
        metadata: AttentionMetadata | None,
        buffers: PreallocatedBuffers | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recycle the running state into the initial one, then run one pass of the layer stack.

        This is the repeated computation: every recycle of every denoising step runs it, over the
        per-step tensors the caller resolves once. A recycling trunk folds the running state in
        first; the zero state it starts from contributes exactly zero through the bias-free norm
        and projection, so the first pass needs no separate form.
        """
        if self.recycle_enabled:
            single = single_init + self.s_cycle_proj(self.s_cycle_norm(single))
            pair = (pair_init + self.z_cycle_proj(self.z_cycle_norm(pair))).contiguous()
        shared_bias_weights = self._fold_shared_pair_bias_weights()
        pair_biases: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.transformer_layers):
            if i in shared_bias_weights:
                pair_biases = self._project_shared_pair_biases(pair, *shared_bias_weights[i])
            pair_bias = pair_biases.pop(i, None)
            # As the decoder's bias precompute does, the layer takes its projected bias in place of the pair.
            if pair_bias is not None:
                layer.mha.bias_proj = False
            try:
                single = layer(
                    single,
                    pair if pair_bias is None else pair_bias,
                    condition,
                    padding_mask,
                    metadata=metadata,
                    buffers=buffers,
                    precomputed_single_masks=precomputed_single_masks,
                )
            finally:
                if pair_bias is not None:
                    layer.mha.bias_proj = True
            if i < len(self.pair_update_layers):
                pair_update = self.pair_update_layers[i]
                if pair_update is not None:
                    pair = pair_update(
                        single,
                        pair,
                        padding_mask,
                        trimul_metadata,
                        precomputed_masks=precomputed_masks,
                        precomputed_single_masks=precomputed_single_masks,
                    )
        return single, pair

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        recycle_depth: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Predict product-space velocities.

        Args:
            single: Initial token representation ``[B, N, token_dim]``.
            pair: Initial pair representation ``[B, N, N, pair_dim]``.
            condition: Initial condition representation
                ``[B, N, condition_dim]``, or one row per trunk pass
                ``[recycle_depth, B, N, condition_dim]`` when the checkpoint
                conditions on the recycle index. A single condition is shared
                by every pass.
            padding_mask: Boolean valid-token mask ``[B, N]``.
            metadata: Optional pairwise-attention metadata.
            buffers: Optional preallocated Torch buffers.
            recycle_depth: Number of trunk passes for this denoising step.
        """
        if recycle_depth < 0:
            raise ValueError(f"recycle_depth must be non-negative, got {recycle_depth}")
        if not self.recycle_enabled and recycle_depth > 1:
            raise ValueError("recycle_depth > 1 requires a recycling checkpoint")
        padding_mask = padding_mask.to(torch.bool)
        n_true = padding_mask.shape[-1]
        if self.enable_token_pad:
            padded, _, n_true = pad_trunk_tokens(
                {"single": single, "pair": pair, "condition": condition, "padding_mask": padding_mask},
                n_true,
                self.token_pad_spec,
            )
            single, pair, condition, padding_mask = (
                padded["single"],
                padded["pair"],
                padded["condition"],
                padded["padding_mask"],
            )
            del padded
        # Every layer masks with these, so each form is computed once per call.
        pair_mask = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)
        precomputed_masks = None
        if self.config.pair_mask_left_aligned:
            precomputed_masks = precompute_pair_masks(
                "CuTeDSL",
                pair_mask,
                inf=self.config.mask_inf,
                dtype=self.dtype,
                activation_dtype=self.dtype,
            )
        precomputed_single_masks = precompute_single_masks(
            self.config.pairwise_attention_backend,
            padding_mask,
            inf=self.config.mask_inf,
            activation_dtype=self.dtype,
        )

        depth = max(1, recycle_depth)
        # A leading pass axis supplies one condition per trunk pass. A single
        # condition is shared, which is every checkpoint that has no recycle feature.
        if condition.ndim == 4:
            if condition.shape[0] != depth:
                raise ValueError(f"condition has {condition.shape[0]} recycle passes, but recycle_depth is {depth}")
        elif condition.ndim != 3:
            raise ValueError(f"condition must be [B, N, C] or [recycle_depth, B, N, C], got {tuple(condition.shape)}")

        def prepare_condition(raw: torch.Tensor) -> torch.Tensor:
            prepared = self.transition_c_1(raw, padding_mask)
            return self.transition_c_2(prepared, padding_mask)

        if condition.ndim == 4:
            conditions = [prepare_condition(condition[cycle]) for cycle in range(depth)]
        else:
            conditions = [prepare_condition(condition)]
        single_init = single * padding_mask.unsqueeze(-1).to(single.dtype)
        pair_init = pair * pair_mask.unsqueeze(-1).to(pair.dtype)
        # A padded pair is a copy only the masked init reads; holding it would
        # keep a second pair tensor alive through every layer.
        del single, pair
        trimul_metadata = precompute_trimul_metadata(
            pair_init,
            precomputed_masks.mask_bias if precomputed_masks is not None else None,
            precomputed_masks.mask_bias_transposed if precomputed_masks is not None else None,
            enabled=self.config.pair_mask_left_aligned,
        )

        step = {
            "single_init": single_init,
            "pair_init": pair_init,
            "padding_mask": padding_mask,
            "trimul_metadata": trimul_metadata,
            "precomputed_masks": precomputed_masks,
            "precomputed_single_masks": precomputed_single_masks,
            "metadata": metadata,
            "buffers": buffers,
        }
        # Recycling the zero state contributes exactly zero, so the first pass takes the same form
        # as the rest and one captured step serves every recycle depth.
        single = torch.zeros_like(single_init) if self.recycle_enabled else single_init
        pair = torch.zeros_like(pair_init) if self.recycle_enabled else pair_init
        for cycle in range(depth if self.recycle_enabled else 1):
            single, pair = self.trunk_graph(
                single=single,
                pair=pair,
                condition=conditions[cycle if condition.ndim == 4 else 0],
                **step,
            )

        single, padding_mask = unpad_trunk_tokens(single, padding_mask, n_true=n_true, kinds=("single", "single_last"))
        valid = padding_mask.unsqueeze(-1).to(single.dtype)
        output = {
            "bb_ca": {"v": self.ca_linear(single) * valid},
            "local_latents": {"v": self.local_latents_linear(single) * valid},
        }
        if self.use_sequence_readout:
            # Upstream trains this probe on a detached track; detaching keeps
            # the inference graph identical to it.
            output["sequence"] = {"logits": self.sequence_readout(single.detach()) * valid}
        if self.use_insertion_head:
            insertion_single = single.detach()
            q = self.insertion_q_proj(insertion_single)
            k = self.insertion_k_proj(insertion_single)
            logits = torch.bmm(q, k.transpose(1, 2))
            output["insertion"] = {"logits": logits * self._insertion_head_scale}
        return output
