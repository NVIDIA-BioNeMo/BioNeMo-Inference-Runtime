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

import dataclasses
from typing import NamedTuple

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
from bionemo_ir._torch.layers.token_padding import (
    pad_single_token_dim,
    pad_trunk_tokens,
    round_up_tokens,
    unpad_trunk_tokens,
)
from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock
from bionemo_ir._torch.layers.transformers.diffusion_transformer import build_bias_mega_weight
from bionemo_ir._torch.layers.transition import Transition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleMultiplicationMetadata,
    precompute_trimul_metadata,
    set_trimul_token_padding,
)
from bionemo_ir.configs import BaseConfig
from bionemo_ir.dsl_kernels.triton.rms_linear_add import rms_linear_add
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


class TrunkMasks(NamedTuple):
    """A token mask padded as the trunk's layers run it, and the masks its passes derive from it."""

    padding_mask: torch.Tensor
    precomputed_masks: PrecomputedPairMasks | None
    precomputed_single_masks: PrecomputedSingleMasks
    trimul_metadata: TriangleMultiplicationMetadata


def _masked_pair(
    pair: torch.Tensor,
    padding_mask: torch.Tensor,
    tokens: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``pair`` zeroed outside its pair mask and zero-padded to ``tokens`` along both token axes, in one pass.

    Equals padding first and masking the padded pair after, which reads and writes the pair twice.

    Args:
        pair: ``[B, N, N, C]`` pair representation.
        padding_mask: ``[B, N]`` boolean mask of real tokens.
        tokens: Padded token count, at least ``N``.
        out: Optional two ``[B, tokens, tokens, C]`` buffers, which both receive the result.

    Returns:
        ``[B, tokens, tokens, C]`` masked pair, the first of ``out`` when given.
    """
    mask = (padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)).unsqueeze(-1).to(pair.dtype)
    batch, n = padding_mask.shape
    if out is None and tokens == n:
        return pair * mask
    first = pair.new_empty((batch, tokens, tokens, pair.shape[-1])) if out is None else out[0]
    torch.mul(pair, mask, out=first[:, :n, :n])
    first[:, n:].zero_()
    first[:, :n, n:].zero_()
    if out is not None:
        out[1].copy_(first)
    return first


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
        groups = _shared_pair_bias_groups(
            self.pair_update_layers, config.num_layers, max(1, _MAX_SHARED_PAIR_BIAS_HEADS // config.num_heads)
        )
        # The last group reads the pair a pass returns. Its projection also leaves each row's sum of squares, from
        # which the next pass's recycle norm takes its statistics, so a recycling trunk projects that group here
        # even when it is one layer.
        self._shared_pair_bias_groups = [
            group for group in groups if len(group) > 1 or (config.recycle_enabled and group == groups[-1])
        ]
        self._shared_pair_bias: dict[int, LNProjMoveaxisPad] = {}
        for size in {len(group) for group in self._shared_pair_bias_groups}:
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
            # Per-pass flags for the pair recycle: a step's last pass has no next pass to recycle for. The kernel
            # reads the flag on the device, so one captured pass serves both.
            self.register_buffer("_recycle_next", torch.ones(1, dtype=torch.int32), persistent=False)
            self.register_buffer("_recycle_none", torch.zeros(1, dtype=torch.int32), persistent=False)
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
        # forward() pads the pair while it masks it, and prepare_masks() the token mask, so the shared padding
        # pads the rest.
        spec = config.token_pad_spec
        self._pad_spec_per_call = dataclasses.replace(
            spec,
            pair_channel=tuple(k for k in spec.pair_channel if k != "pair"),
            single_last=tuple(k for k in spec.single_last if k != "padding_mask"),
        )
        # forward() pads the tokens before any layer runs.
        set_trimul_token_padding(self, self.enable_token_pad)
        # One capture serves every recycle depth, because the repeated computation is the step
        # rather than the whole forward. ``configure_graphs`` supplies the policy.
        self.trunk_graph = GraphRegion(self, "trunk_step", None)

    def _padded_tokens(self, tokens: int) -> int:
        """The token count the layers run at: ``tokens`` rounded up to the TriMul alignment when padding is on."""
        return round_up_tokens(tokens) if self.enable_token_pad else tokens

    def initial_pair_buffers(
        self, batch: int, tokens: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The trunk's buffers for a step's initial pair when its passes would run as CUDA graphs, else ``None``.

        Both are ``[batch, padded, padded, pair_dim]``, ``padded`` being the trunk's token padding of ``tokens``:
        the first holds the step's ``pair_init`` and the second the first pass's ``pair``, which the passes then
        recycle in place. The captured passes borrow both, so a producer that writes the masked pair into both,
        with zeros outside it, spares the graph its input copies; see ``pair_in_buffers`` of :meth:`forward`. The
        graph region keeps them as long as it keeps graphs.
        """
        padded = self._padded_tokens(tokens)
        if not self.trunk_graph.captures(num_tokens=padded):
            return None
        shape = (batch, padded, padded, self.config.pair_dim)
        return self.trunk_graph.borrowed_buffers(
            ("initial_pair", shape, device),
            lambda: tuple(torch.zeros(shape, device=device, dtype=self.dtype) for _ in range(2)),
        )

    def prepare_masks(self, padding_mask: torch.Tensor) -> TrunkMasks:
        """The token mask padded as the layers run it, and the masks the passes derive from it.

        A rollout's steps share their mask, so a rollout builds these once and hands them to every
        :meth:`forward`; the captured passes then find these stable inputs unchanged and copy none of them in.

        Args:
            padding_mask: ``[B, N]`` mask of real tokens.

        Returns:
            The padded ``[B, P]`` boolean mask, the pair masks, the single masks and the TriMul metadata.
        """
        padding_mask = padding_mask.to(torch.bool)
        batch, tokens = padding_mask.shape
        padded = self._padded_tokens(tokens)
        if padded != tokens and "padding_mask" in self.token_pad_spec.single_last:
            padding_mask = pad_single_token_dim(padding_mask, padded - tokens, has_channel=False)
        # Every layer masks with these, so each form is computed once.
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
        # The TriMul metadata reads only the padded pair's layout.
        layout = torch.empty((batch, padded, padded, 0), device=padding_mask.device, dtype=self.dtype)
        return TrunkMasks(
            padding_mask=padding_mask,
            precomputed_masks=precomputed_masks,
            precomputed_single_masks=precompute_single_masks(
                self.config.pairwise_attention_backend,
                padding_mask,
                inf=self.config.mask_inf,
                activation_dtype=self.dtype,
            ),
            trimul_metadata=precompute_trimul_metadata(
                layout,
                precomputed_masks.mask_bias if precomputed_masks is not None else None,
                precomputed_masks.mask_bias_transposed if precomputed_masks is not None else None,
                enabled=self.config.pair_mask_left_aligned,
            ),
        )

    def _fold_shared_pair_bias_weights(self) -> dict[int, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]]:
        """Each shared group, its FP32 folded projection and a unit gamma, keyed by the group's first layer.

        Rebuilt per call so weight loading and in-place edits are observed, as the decoder's bias precompute is;
        one fold covers every shared layer, and each group takes its rows.
        """
        groups = [group for group in self._shared_pair_bias_groups if len(group) in self._shared_pair_bias]
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
        self,
        pair: torch.Tensor,
        group: tuple[int, ...],
        weight: torch.Tensor,
        unit_gamma: torch.Tensor,
        with_row_sumsq: bool = False,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """Project a group's pair biases in one pass; an empty dict leaves each layer to project its own.

        Args:
            pair: ``[B, N, N, pair_dim]`` pair representation that the group's layers read.
            group: Indices of consecutive layers that read ``pair``.
            weight: The group's FP32 folded projection ``[len(group) * num_heads, pair_dim]``.
            unit_gamma: FP32 ones ``[pair_dim]``; the gammas are already in ``weight``.
            with_row_sumsq: Also return each pair row's sum of squares, which the projection takes as it reads
                ``pair``.

        Returns:
            Each layer's ``[B, num_heads, N, N_padded]`` bias, a view of one projection, and the FP32 ``[B, N, N]``
            row sums when asked for and projected here, else ``None``.
        """
        if not (pair.is_cuda and pair.is_contiguous() and pair.dtype == self.dtype):
            return {}, None
        row_sumsq = pair.new_empty(pair.shape[:-1], dtype=torch.float32) if with_row_sumsq else None
        bias = self._shared_pair_bias[len(group)](
            pair, unit_gamma, None, weight, pad_multiple=self._pair_bias_pad_multiple, row_sumsq=row_sumsq
        )
        heads = self.config.num_heads
        return {layer: bias.narrow(-3, k * heads, heads) for k, layer in enumerate(group)}, row_sumsq

    def _recycles_pair_from_sums(self) -> bool:
        """Whether the pair recycle takes its norm statistics from the last pair-bias projection's row sums."""
        return self.recycle_enabled and len(self._shared_pair_bias_groups[-1]) in self._shared_pair_bias

    def _recycle_pair(
        self,
        pair: torch.Tensor,
        pair_init: torch.Tensor,
        pair_sumsq: torch.Tensor | None,
        out: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``pair_init + z_cycle_proj(z_cycle_norm(pair))``, the next pass's pair.

        Given each row's sum of squares, the norm needs no pass of its own: its gamma folds into the projection
        weight, and one GEMM reads the raw pair, scales each row of its FP32 product by the row's rstd and adds
        ``pair_init`` before rounding (:func:`rms_linear_add`). That rounds the folded weight to the pair's dtype
        where the plain form rounds the normalized pair and the product.

        Args:
            pair: ``[B, N, N, pair_dim]`` pair a pass ends with.
            pair_init: ``[B, N, N, pair_dim]`` initial pair representation.
            pair_sumsq: FP32 ``[B, N, N]`` sums of squares of ``pair``'s rows, or ``None`` to normalize ``pair``.
            out: Optional output, which may share ``pair``'s storage.
            active: Optional ``[1]`` int32 device flag; zero skips the fused GEMM and leaves the result undefined.
                The plain form always computes it.

        Returns:
            ``out``, or a new contiguous ``[B, N, N, pair_dim]`` tensor.
        """
        if pair_sumsq is None:
            update = self.z_cycle_proj(self.z_cycle_norm(pair))
            return (pair_init + update).contiguous() if out is None else torch.add(pair_init, update, out=out)
        weight = (self.z_cycle_proj.weight * self.z_cycle_norm.weight).to(pair.dtype)
        return rms_linear_add(pair_init, pair, weight, pair_sumsq, eps=self.z_cycle_norm.eps, out=out, active=active)

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
        recycles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one pass of the layer stack between the recycles of its single and its pair.

        This is the repeated computation: every recycle of every denoising step runs it, over the
        per-step tensors the caller resolves once. A recycling trunk folds the running single state
        in first; the zero state it starts from contributes exactly zero through the bias-free norm
        and projection, so the first pass needs no separate form.

        The pair recycles at the end of the pass instead (:meth:`_recycle_pair`), into ``pair``,
        which nothing reads after the first pair update, so a captured step hands that static
        buffer back without a copy. A first pass given ``pair_init`` itself, which every pass reads
        again, writes a new tensor instead. ``recycles`` is a device flag that is zero for a step's last pass,
        whose recycle no pass would read.

        Returns:
            ``(single, pair)``: the pass's single representation and the next pass's pair, or the
            pass's own pair without recycling.
        """
        if self.recycle_enabled:
            single = single_init + self.s_cycle_proj(self.s_cycle_norm(single))
        carried = pair
        shared_bias_weights = self._fold_shared_pair_bias_weights()
        # The last shared projection reads the pair the pass ends with, so its row sums are the recycle's.
        sums_layer = self._shared_pair_bias_groups[-1][0] if self._recycles_pair_from_sums() else None
        pair_sumsq = None
        pair_biases: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.transformer_layers):
            if i in shared_bias_weights:
                pair_biases, pair_sumsq = self._project_shared_pair_biases(
                    pair, *shared_bias_weights[i], with_row_sumsq=i == sums_layer
                )
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
        if self.recycle_enabled:
            initial = carried.untyped_storage().data_ptr() == pair_init.untyped_storage().data_ptr()
            pair = self._recycle_pair(pair, pair_init, pair_sumsq, out=None if initial else carried, active=recycles)
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
        pair_in_buffers: bool = False,
        masks: TrunkMasks | None = None,
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
            pair_in_buffers: ``pair`` is the first of :meth:`initial_pair_buffers` cut to ``[B, N, N, C]``, and
                both buffers already hold the masked pair, zero-padded.
            masks: :meth:`prepare_masks` of ``padding_mask``, which a rollout builds once; built here when omitted.
        """
        if recycle_depth < 0:
            raise ValueError(f"recycle_depth must be non-negative, got {recycle_depth}")
        if not self.recycle_enabled and recycle_depth > 1:
            raise ValueError("recycle_depth > 1 requires a recycling checkpoint")
        padding_mask = padding_mask.to(torch.bool)
        if masks is None:
            masks = self.prepare_masks(padding_mask)
        n_true = padding_mask.shape[-1]
        padded = self._padded_tokens(n_true)
        pair_buffers = self.initial_pair_buffers(padding_mask.shape[0], n_true, padding_mask.device)
        if pair_in_buffers:
            if pair_buffers is None or pair.data_ptr() != pair_buffers[0].data_ptr():
                raise ValueError("pair_in_buffers needs pair to be the first of the trunk's initial-pair buffers")
            pair_init, first_pair = pair_buffers
        else:
            if pair.dtype != self.dtype:
                pair_buffers = None
            pair_init = _masked_pair(pair, padding_mask, padded, out=pair_buffers)
            first_pair = pair_init if pair_buffers is None else pair_buffers[1]
        # The caller's pair is only read once, by the masked init.
        del pair
        if self.enable_token_pad:
            tracks, _, n_true = pad_trunk_tokens(
                {"single": single, "condition": condition}, n_true, self._pad_spec_per_call
            )
            single, condition = tracks["single"], tracks["condition"]
            del tracks
        padding_mask, precomputed_masks, precomputed_single_masks, trimul_metadata = masks

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
        del single

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
        # as the rest and one captured step serves every recycle depth. The pair recycles at the end
        # of each pass, so the first pass reads the initial pair: ``pair_init`` itself, or the
        # second initial-pair buffer, which holds the same values.
        single = torch.zeros_like(single_init) if self.recycle_enabled else single_init
        pair = first_pair
        passes = depth if self.recycle_enabled else 1
        for cycle in range(passes):
            recycles = None
            if self.recycle_enabled:
                recycles = self._recycle_next if cycle + 1 < passes else self._recycle_none
            single, pair = self.trunk_graph(
                single=single,
                pair=pair,
                condition=conditions[cycle if condition.ndim == 4 else 0],
                recycles=recycles,
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
