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

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from bionemo_ir._torch.attention_backend import auto_select_pairwise_attention_backend
from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.layers.normalization import replace_with_fused_layernorm, replace_with_fused_rmsnorm
from bionemo_ir._torch.layers.transformers.local_latents_decoder import LEGACY_RESIDUE_TYPES, RFProteinaLatentDecoder
from bionemo_ir._torch.layers.transformers.local_latents_encoder import RFProteinaLatentEncoder
from bionemo_ir._torch.layers.transformers.local_latents_transformer import LocalLatentsTransformer, TrunkMasks
from bionemo_ir._torch.modules.rfproteina.atom_utils import finite_atom_mask
from bionemo_ir._torch.modules.rfproteina.contracts import (
    RFProteinaCleanStateEncoding,
    RFProteinaConversionReport,
    RFProteinaDecodedOutput,
    RFProteinaPredictFn,
    RFProteinaRepresentations,
    RFProteinaSimulationOutput,
    RFProteinaState,
)
from bionemo_ir._torch.modules.rfproteina.groups import extract_groups
from bionemo_ir._torch.modules.rfproteina.representation_builder import (
    ATOM37_SIZE,
    ATOMIZED_TOKEN_TYPE,
    CONDITION_COORDINATE,
    CONDITION_COORDINATE_MASK,
    CONDITION_INDEX_MASK,
    DNA_TOKEN_TYPE,
    RNA_TOKEN_TYPE,
    RFProteinaRepresentationBuilder,
    StaticTrackParts,
)
from bionemo_ir._torch.modules.rfproteina.simulator import RFProteinaProductSpaceSimulator
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import FlowMatchingSchedule
from bionemo_ir.configs import AcceleratedConfig, BackendType, BaseConfig
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint
from bionemo_ir.hubs.support_matrix import GenerativeSupportMatrix as SupMat
from bionemo_ir.models.rfproteina.config import (
    PRETRAINED_CONFIG_REGISTRY,
    RFProteinaConfig,
    RFProteinaSimulationConfig,
    trunk_graph_config,
)
from bionemo_ir.models.rfproteina.convert import (
    RELEASED_LORA_ALPHA,
    load_rfproteina_checkpoint_state,
    load_rfproteina_decoder_checkpoint,
    load_rfproteina_encoder_checkpoint,
    load_rfproteina_representation_builder_checkpoint,
    load_rfproteina_trunk_checkpoint,
    merge_rfproteina_lora,
    reject_unused_source,
    rfproteina_decoder_config_from_weights,
    rfproteina_encoder_config_from_checkpoint,
    rfproteina_encoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_weights,
    validate_reference_config,
)
from bionemo_ir.runtime.buffers import PreallocatedBuffers
from bionemo_ir.utils import str_dtype_to_torch

from ..optimize_module_setter import OptimizedModuleSetterMixin

MODEL_NAME = "RFProteína"

#: Tokens whose backbone the flow generates, ``[B, N]``. A token outside it only conditions: the trunk
#: attends to it, but it holds zero state, takes no step and takes no part in centering.
FLOW_BACKBONE_MASK = "flow_backbone_mask"


def _normalize_token_mask(name: str, value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Convert a model-boundary token mask to bool and clear padding."""
    if value.shape != padding_mask.shape:
        raise ValueError(f"{name} must have shape {tuple(padding_mask.shape)}, got {tuple(value.shape)}")
    return value.to(device=padding_mask.device, dtype=torch.bool) & padding_mask


def _resolve_rollout_mask(
    static_features: Mapping[str, torch.Tensor] | None, padding_mask: torch.Tensor
) -> torch.Tensor:
    """Tokens a rollout generates: every real token unless the features name a narrower set.

    An unindexed motif residue is a real token the trunk attends to and not one the flow moves. The
    checkpoint was trained with such tokens held out of the flow, so a rollout that lets them evolve
    hands the trunk coordinates it never saw. The sampler takes this mask where it takes the padding
    mask; the velocity network still sees every token.
    """
    if static_features is None or FLOW_BACKBONE_MASK not in static_features:
        return padding_mask
    return _normalize_token_mask(FLOW_BACKBONE_MASK, static_features[FLOW_BACKBONE_MASK], padding_mask)


def _first_feature(
    static_features: Mapping[str, torch.Tensor], device: torch.device, *names: str
) -> torch.Tensor | None:
    """The first of ``names`` the batch carries, on ``device``, or ``None``.

    The encoder of a checkpoint trained with per-atom chemistry reads each atom's element, charge and incident
    bond types. Upstream's batch names the element ``ground_truth_element``, which is the structure's own and so
    the one an encoder is given.
    """
    for name in names:
        value = static_features.get(name)
        if value is not None:
            return value.to(device=device)
    return None


def _drop_undecodable_tokens(
    can_be_encoded: torch.Tensor,
    token_type: torch.Tensor,
    padding_mask: torch.Tensor,
    decoder_num_residue_types: int,
) -> torch.Tensor:
    """Remove from ``can_be_encoded`` the tokens the decoder cannot emit.

    Atomized tokens never decode. A decoder with the 20 protein classes cannot name an RNA or DNA residue
    either. The narrowing is out of place: a bool mask is its own ``.to(torch.bool)``, so ``&=`` would write
    through to the caller's tensor.
    """
    if token_type.shape != padding_mask.shape:
        raise ValueError(f"token_type must have shape {tuple(padding_mask.shape)}, got {tuple(token_type.shape)}")
    token_type = token_type.to(device=padding_mask.device)
    can_be_encoded = can_be_encoded & (token_type != ATOMIZED_TOKEN_TYPE)
    if decoder_num_residue_types == LEGACY_RESIDUE_TYPES:
        can_be_encoded = can_be_encoded & (token_type != RNA_TOKEN_TYPE) & (token_type != DNA_TOKEN_TYPE)
    return can_be_encoded


def _resolve_runtime_masks(
    static_features: Mapping[str, torch.Tensor],
    padding_mask: torch.Tensor,
    can_be_encoded: torch.Tensor | None,
    coordinate_condition_mask: torch.Tensor | None,
    decoder_num_residue_types: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Resolve sampler masks from an already-tensorized feature batch."""
    padding_mask = padding_mask.to(torch.bool)
    can_be_encoded = _normalize_token_mask(
        "can_be_encoded",
        static_features.get("can_be_encoded", padding_mask) if can_be_encoded is None else can_be_encoded,
        padding_mask,
    )
    has_known_index = static_features.get("has_known_index")
    if has_known_index is None and CONDITION_INDEX_MASK in static_features:
        index_mask = static_features[CONDITION_INDEX_MASK]
        atom37_mask = static_features.get("atom37_mask")
        if index_mask.ndim == 3:
            if atom37_mask is None or atom37_mask.shape != index_mask.shape:
                raise ValueError(f"atom37_mask must match {CONDITION_INDEX_MASK} shape {tuple(index_mask.shape)}")
            has_known_index = (index_mask.to(torch.bool) | ~atom37_mask.to(torch.bool)).all(dim=-1)
        else:
            has_known_index = index_mask
    if has_known_index is not None:
        can_be_encoded &= _normalize_token_mask(
            "has_known_index",
            has_known_index,
            padding_mask,
        )
    if FLOW_BACKBONE_MASK in static_features:
        # Local latents are emitted only where the backbone is.
        can_be_encoded &= _normalize_token_mask(FLOW_BACKBONE_MASK, static_features[FLOW_BACKBONE_MASK], padding_mask)
    token_type = static_features.get("token_type")
    if token_type is not None:
        can_be_encoded = _drop_undecodable_tokens(can_be_encoded, token_type, padding_mask, decoder_num_residue_types)

    if coordinate_condition_mask is None:
        coordinate_condition_mask = static_features.get("coord_cond_mask")
    if coordinate_condition_mask is None:
        atom_mask = static_features.get(CONDITION_COORDINATE_MASK)
        if atom_mask is not None:
            expected = (*padding_mask.shape, ATOM37_SIZE)
            if atom_mask.shape != expected:
                raise ValueError(
                    f"{CONDITION_COORDINATE_MASK} must have shape {expected}, got {tuple(atom_mask.shape)}"
                )
            coordinate_condition_mask = atom_mask.to(torch.bool).any(dim=-1)
    if coordinate_condition_mask is None:
        coordinates = static_features.get(CONDITION_COORDINATE)
        if coordinates is not None:
            expected = (*padding_mask.shape, ATOM37_SIZE, 3)
            if coordinates.shape != expected:
                raise ValueError(f"{CONDITION_COORDINATE} must have shape {expected}, got {tuple(coordinates.shape)}")
            coordinate_condition_mask = finite_atom_mask(coordinates).any(dim=-1)
    if coordinate_condition_mask is not None:
        coordinate_condition_mask = _normalize_token_mask(
            "coordinate_condition_mask",
            coordinate_condition_mask,
            padding_mask,
        )
    return can_be_encoded, coordinate_condition_mask


class RFProteina(nn.Module, OptimizedModuleSetterMixin):
    """RFProteína encoder, velocity network, numerical integrator and decoder."""

    # Only the velocity trunk is graph-optimized. It runs once per sampling
    # step with stable shapes; the one-shot decoder is intentionally excluded.
    # The trunk's repeated computation is one recycle step, not the whole
    # forward, so a shape captures once instead of once per recycle depth.
    GRAPH_REGIONS = {"velocity_model": "velocity_model.trunk_graph"}

    @classmethod
    def from_reference_state_dicts(
        cls,
        flow_state: Mapping[str, torch.Tensor],
        autoencoder_state: Mapping[str, torch.Tensor],
        *,
        simulation_config: RFProteinaSimulationConfig,
        device: torch.device | str | None = None,
        attention_backend: str | None = None,
        noise_align_chirality: bool = True,
        noise_align_chirality_translate: bool = False,
        lora_alpha: float | None = RELEASED_LORA_ALPHA,
        heads: torch.nn.Module | None = None,
        reference_config: Mapping | None = None,
    ) -> "RFProteina":
        """Build a model from weights a host already holds, inferring every width.

        An embedding application that has loaded and selected its own weights —
        applying an EMA, say — can hand them straight over instead of naming a
        file the model would reopen. Both autoencoder halves load, so redesign
        and inverse folding need no second pass.

        Args:
            flow_state: Released score-network weights under the ``nn.`` prefix.
            heads: Caller-owned PyTorch readouts; ``requires_pair`` declares pair use.
                ``forward(single, pair, padding_mask, emit_masks)`` returns the prediction mapping. A rollout
                passes the masks it integrates with: ``bb_ca`` holds the generated tokens and ``local_latents``
                the tokens whose latents it emits. The engine loads their weights under ``nn.heads.`` in the
                trunk dtype.
            reference_config: Training network settings for feature-order and semantic compatibility checks.
                When supplied, this config also determines chirality alignment instead of the defaults below.
            autoencoder_state: Released autoencoder weights under the
                ``encoder.`` / ``decoder.`` prefixes.
            simulation_config: Numerical step and reference-distribution parameters
                from the host's own sampler configuration. Required: see
                :class:`RFProteinaSimulationConfig` for why its defaults are no fallback.
            device: Where to place the built model.
            attention_backend: Pairwise attention backend for the trunk and,
                when it has a pair track, the autoencoder. ``None`` selects the
                best one available for the execution dtype.
            noise_align_chirality: Whether the network was trained with its
                chirality vectors rotated onto the noisy state, a checkpoint's
                ``nn.noise_align_chirality``. Defaults to ``True`` for the
                released vocab-147 checkpoint. The weights do not record it;
                pass the training setting explicitly for other checkpoints.
            noise_align_chirality_translate: Likewise for
                ``nn.noise_align_chirality_translate``. Defaults to ``False``
                for the released vocab-147 checkpoint.
            lora_alpha: ``lora_alpha`` used to fold PEFT adapters that
                ``flow_state`` still carries, as the released vocab-147 flow
                checkpoint does. The default is that release's value; a vocab-33
                checkpoint has no adapters, so it is unused. ``None`` refuses
                adapters instead of assuming a scale.

        Returns:
            A model whose four conversion reports cover the given weights.
        """
        flow = load_rfproteina_checkpoint_state(flow_state)
        if lora_alpha is not None:
            flow = merge_rfproteina_lora(flow, lora_alpha=lora_alpha)
        autoencoder = load_rfproteina_checkpoint_state(autoencoder_state)
        reference = cls.get_pretrained_config()
        if attention_backend is not None:
            reference.trunk.set_pairwise_attention_backend(attention_backend)
            if reference.decoder.use_pair_bias:
                reference.decoder.set_pairwise_attention_backend(attention_backend)
        decoder = rfproteina_decoder_config_from_weights(
            autoencoder,
            dtype=reference.decoder.dtype,
            transformer_dtype=reference.decoder.transformer_dtype,
            pairwise_attention_backend=reference.decoder.pairwise_attention_backend,
        )
        trunk = rfproteina_trunk_config_from_weights(
            flow,
            latent_dim=decoder.latent_dim,
            dtype=reference.trunk.dtype,
            pairwise_attention_backend=reference.trunk.pairwise_attention_backend,
        )
        if reference_config is not None:
            noise_align_chirality = bool(reference_config.get("noise_align_chirality", False))
            noise_align_chirality_translate = bool(reference_config.get("noise_align_chirality_translate", False))
        config = RFProteinaConfig(
            dtype=trunk.dtype,
            trunk=trunk,
            representation=rfproteina_representation_config_from_weights(
                flow,
                latent_dim=trunk.latent_dim,
                dtype=reference.representation.dtype,
                pair_adaln_dtype=reference.representation.pair_adaln_dtype,
                noise_align_chirality=noise_align_chirality,
                noise_align_chirality_translate=noise_align_chirality_translate,
            ),
            decoder=decoder,
            encoder=rfproteina_encoder_config_from_weights(
                autoencoder,
                dtype=decoder.dtype,
                transformer_dtype=decoder.transformer_dtype,
                pairwise_attention_backend=decoder.pairwise_attention_backend,
            ),
            simulation=simulation_config,
        )
        if reference_config is not None:
            validate_reference_config(reference_config, config.trunk, config.representation)
        model = cls(config, heads=heads)
        model.load_weights(flow)
        model.load_decoder_weights(autoencoder)
        model.load_encoder_weights(autoencoder)
        model.eval()
        return model if device is None else model.to(device)

    def configure_graphs(
        self,
        *,
        capacity: int = 1,
        verify_capture: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        """Replay the velocity trunk's recycle step from exact-shape CUDA graphs.

        The step is what repeats, so each distinct ``(batch, tokens)`` the
        rollout visits takes one cache slot whatever recycle depths its
        schedule uses, and capturing it records one trunk pass. Size
        ``capacity`` for the shapes a request will run: too small and the
        cache evicts and recaptures instead of replaying.

        Args:
            capacity: Input shapes this trunk retains. Pair-dependent heads need two graphs
                per shape; other heads need one. Each graph holds its own static buffers.
            verify_capture: Compare the first replay against eager output.
            max_tokens: Refuse to capture past this token count, running eager
                there instead. ``None`` keeps the trunk's declared limit.
        """
        if capacity < 1:
            raise ValueError(f"CUDA graph capacity must be positive, got {capacity}")
        # Release the previous policy's graphs: they replay the buffers they captured.
        self.reset_graphs()
        config = trunk_graph_config() if max_tokens is None else trunk_graph_config(max_tokens)
        pair_heads = self.velocity_model.heads is not None and self.velocity_model.heads.requires_pair
        config.num_graphs_max_for_this_module = capacity * (2 if pair_heads else 1)
        config.verify_capture = verify_capture
        self.optimize(
            {
                "velocity_model": AcceleratedConfig(
                    backend=BackendType.TORCH,
                    default=BaseConfig(graph_optimization_config=config),
                )
            }
        )

    def _graph_tracker(self) -> CUDAGraphOptimizationTracker | None:
        """The trunk step's tracker, which exists once the region has captured."""
        return self.velocity_model.trunk_graph.tracker

    def graph_status(self) -> dict[str, int | bool]:
        """Report the velocity trunk's graph cache occupancy.

        ``fallback`` counts keys that failed capture and are pinned to eager;
        it stays non-zero until :meth:`reset_graphs`.
        """
        region = self.velocity_model.trunk_graph
        tracker = self._graph_tracker()
        if tracker is None:
            capacity = 0
            if region.enabled and region.policy is not None:
                capacity = region.policy.num_graphs_max_for_this_module
            return {
                "enabled": region.enabled,
                "capacity": capacity,
                "active": 0,
                "captured": 0,
                "verified": 0,
                "fallback": 0,
            }
        states = tracker.graph_state_by_key.values()
        captured = sum(
            state.preparation_state
            in (CUDAGraphPreparationState.GRAPH_CAPTURED, CUDAGraphPreparationState.GRAPH_VERIFIED)
            for state in states
        )
        verified = sum(state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED for state in states)
        return {
            "enabled": True,
            "capacity": tracker.graph_optimization_config.num_graphs_max_for_this_module,
            "active": len(tracker.graph_state_by_key),
            "captured": captured,
            "verified": verified,
            "fallback": sum(tracker.fallback_to_eager_by_key.values()),
        }

    def reset_graphs(self) -> None:
        """Release every captured graph and restore the eager trunk.

        Call this before the weights, shapes, or sampling policy change: a
        retained graph replays the buffers and kernels it captured, not the
        new ones.
        """
        region = self.velocity_model.trunk_graph
        region.reset()
        region.enabled = False
        region.policy = None

    @staticmethod
    def get_pretrained_config(model_name: str = SupMat.RFProteina) -> RFProteinaConfig:
        """Create the optimized default inference configuration.

        The architecture is the latest released checkpoint's. Older ones need
        :func:`~...convert.rfproteina_trunk_config_from_checkpoint`.
        """
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(f"RFProteína pretrained config not found for model name: {model_name}")
        # Passing the dtype in lets the config propagate it to the trunk.
        config = config_class(dtype="bfloat16")
        # The mixed-precision boundary: the velocity, decoder, and lazy encoder
        # transformer stacks plus terminal pair AdaLN run in BF16. Feature
        # construction and the autoencoder output heads stay FP32. The lazy
        # encoder inherits the decoder precision/backend in load_encoder_weights.
        config.representation.pair_adaln_dtype = "bfloat16"
        config.decoder.set_dtype(torch.float32)
        config.decoder.transformer_dtype = "bfloat16"
        pair_backend = auto_select_pairwise_attention_backend(torch.bfloat16)
        config.trunk.set_pairwise_attention_backend(pair_backend)
        # CuTeDSL's pairwise kernel requires a pair bias. The unified AE has
        # no pair track, so its mask-only attention remains on SDPA.
        if config.decoder.use_pair_bias:
            config.decoder.set_pairwise_attention_backend(pair_backend)
        return config

    def __init__(
        self,
        config: RFProteinaConfig | None = None,
        checkpoint_path: str | Path | None = None,
        include_load_weights: bool = False,
        autoencoder_checkpoint_path: str | Path | None = None,
        lora_alpha: float | None = RELEASED_LORA_ALPHA,
        heads: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_name = MODEL_NAME
        self.config = config or self.get_pretrained_config()
        resolved_checkpoint = resolve_rfproteina_checkpoint(
            checkpoint_path if checkpoint_path is not None else self.config.checkpoint_path
        )
        resolved_autoencoder_checkpoint = resolve_rfproteina_autoencoder_checkpoint(
            autoencoder_checkpoint_path
            if autoencoder_checkpoint_path is not None
            else self.config.autoencoder_checkpoint_path
        )
        self._autoencoder_checkpoint_path = resolved_autoencoder_checkpoint

        if include_load_weights and resolved_checkpoint is None:
            raise ValueError("RFProteína checkpoint path is required when include_load_weights=True")
        if include_load_weights and resolved_autoencoder_checkpoint is None:
            raise ValueError("RFProteína autoencoder checkpoint path is required when include_load_weights=True")

        # Unlike the trunk/representation widths, which the config settles,
        # this one means two incompatible checkpoints rather than a config that
        # needs normalizing, so it stays an error.
        if self.config.trunk.latent_dim != self.config.decoder.latent_dim:
            raise ValueError(
                "RFProteína trunk and decoder latent dimensions must match, "
                f"got {self.config.trunk.latent_dim} and "
                f"{self.config.decoder.latent_dim}"
            )
        self.dtype = self.config.trunk.torch_dtype
        self.state_dtype = str_dtype_to_torch(self.config.simulation.state_dtype)
        self.representation_builder = RFProteinaRepresentationBuilder(self.config.representation, self.config.trunk)
        self.velocity_model = LocalLatentsTransformer(self.config.trunk, heads=heads)
        self.decoder = RFProteinaLatentDecoder(self.config.decoder)
        self.encoder = None if self.config.encoder is None else RFProteinaLatentEncoder(self.config.encoder)
        self.simulator = RFProteinaProductSpaceSimulator(
            self.config.simulation,
            latent_dim=self.config.trunk.latent_dim,
        )
        self.conversion_report: RFProteinaConversionReport | None = None
        self.representation_conversion_report: RFProteinaConversionReport | None = None
        self.decoder_conversion_report: RFProteinaConversionReport | None = None
        self.encoder_conversion_report: RFProteinaConversionReport | None = None

        if include_load_weights:
            self.load_weights(resolved_checkpoint, lora_alpha=lora_alpha)
            self.load_decoder_weights(resolved_autoencoder_checkpoint)
            if self.encoder is not None:
                self.load_encoder_weights(resolved_autoencoder_checkpoint)

        for module in (self.velocity_model, self.representation_builder, self.decoder, self.encoder):
            if module is not None:
                replace_with_fused_layernorm(module)
                replace_with_fused_rmsnorm(module)

    def load_weights(
        self, checkpoint_path: str | Path, lora_alpha: float | None = RELEASED_LORA_ALPHA
    ) -> RFProteinaConversionReport:
        """Load the released score network and representation-builder weights.

        The trunk goes first because its loader validates the architecture
        against the checkpoint before touching tensors. Both modules are sized
        from ``config.trunk``, so leading with it turns a config/checkpoint
        mismatch into one message naming the offending field instead of a wall
        of per-tensor size errors from the builder.

        Args:
            checkpoint_path: Released score-network checkpoint.
            lora_alpha: ``lora_alpha`` used to fold PEFT adapters the checkpoint
                still carries, as the released vocab-147 flow checkpoint does.
                The default is that release's value; a checkpoint without
                adapters ignores it. ``None`` refuses adapters instead of
                assuming a scale.
        """
        self.conversion_report = load_rfproteina_trunk_checkpoint(
            self.velocity_model,
            checkpoint_path,
            strict=True,
            lora_alpha=lora_alpha,
        )
        self.representation_conversion_report = load_rfproteina_representation_builder_checkpoint(
            self.representation_builder,
            checkpoint_path,
            strict=True,
            lora_alpha=lora_alpha,
        )
        reject_unused_source(self.conversion_report, self.representation_conversion_report, component="score network")
        return self.conversion_report

    def load_decoder_weights(self, checkpoint_path: str | Path) -> RFProteinaConversionReport:
        """Load the released partial-autoencoder decoder weights."""
        self.decoder_conversion_report = load_rfproteina_decoder_checkpoint(
            self.decoder,
            checkpoint_path,
            strict=True,
        )
        reject_unused_source(self.decoder_conversion_report, component="decoder")
        return self.decoder_conversion_report

    def load_encoder_weights(self, checkpoint_path: str | Path) -> RFProteinaConversionReport:
        """Build when needed and strictly load the autoencoder encoder."""
        self._autoencoder_checkpoint_path = checkpoint_path
        if self.encoder is None:
            encoder_config = rfproteina_encoder_config_from_checkpoint(
                checkpoint_path,
                dtype=self.config.decoder.dtype,
                transformer_dtype=self.config.decoder.transformer_dtype,
                pairwise_attention_backend=(self.config.decoder.pairwise_attention_backend),
            )
            if encoder_config.latent_dim != self.config.trunk.latent_dim:
                raise ValueError(
                    "RFProteína encoder and trunk latent dimensions must "
                    f"match, got {encoder_config.latent_dim} and "
                    f"{self.config.trunk.latent_dim}"
                )
            self.config.encoder = encoder_config
            self.encoder = RFProteinaLatentEncoder(encoder_config)
        self.encoder_conversion_report = load_rfproteina_encoder_checkpoint(
            self.encoder,
            checkpoint_path,
            strict=True,
        )
        reject_unused_source(self.encoder_conversion_report, component="encoder")
        # ``encoder`` may be constructed lazily after ``__init__`` performed
        # the model-wide rewrite.
        replace_with_fused_layernorm(self.encoder)
        replace_with_fused_rmsnorm(self.encoder)
        return self.encoder_conversion_report

    def _ensure_encoder(self, device: torch.device) -> RFProteinaLatentEncoder:
        """Lazily load the encoder so full generation pays no memory cost."""
        if self.encoder is None:
            if self._autoencoder_checkpoint_path is None:
                raise ValueError("an RFProteína autoencoder checkpoint is required to encode an existing structure")
            self.load_encoder_weights(self._autoencoder_checkpoint_path)
        if self.encoder.latent_dim != self.config.trunk.latent_dim:
            raise ValueError(
                "RFProteína encoder and trunk latent dimensions must match, "
                f"got {self.encoder.latent_dim} and "
                f"{self.config.trunk.latent_dim}"
            )
        self.encoder.to(device=device)
        self.encoder.eval()
        return self.encoder

    @torch.no_grad()
    def encode(
        self,
        static_features: Mapping[str, torch.Tensor],
        coordinates: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        residue_type: torch.Tensor | None = None,
        can_be_encoded: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sample_latent: bool = True,
    ) -> RFProteinaCleanStateEncoding:
        """Encode an existing atom37 protein into a clean product-space state.

        Coordinates use model units (nanometres). ``coordinates`` may be
        omitted when ``static_features`` carries ``coords_nm``.
        """
        if "padding_mask" not in static_features:
            raise ValueError("static_features must carry 'padding_mask' for encoding")
        if coordinates is None:
            coordinates = static_features.get("coords_nm")
        if coordinates is None:
            raise ValueError("coordinates in nanometres are required (argument or static_features['coords_nm'])")
        if atom_mask is None:
            atom_mask = static_features.get("atom37_mask")
        if atom_mask is None:
            raise ValueError("atom_mask or static_features['atom37_mask'] is required for encoding")
        if residue_type is None:
            residue_type = static_features.get("residue_type")
        if residue_type is None:
            raise ValueError("residue_type or static_features['residue_type'] is required for encoding")

        padding_mask = static_features["padding_mask"].to(device=coordinates.device, dtype=torch.bool)
        can_be_encoded, _ = _resolve_runtime_masks(
            static_features,
            padding_mask,
            can_be_encoded,
            coordinate_condition_mask=None,
            decoder_num_residue_types=self.decoder.num_residue_types,
        )
        encoder = self._ensure_encoder(coordinates.device)
        encoder_residue_type = residue_type.to(device=coordinates.device)
        if encoder.num_residue_types == LEGACY_RESIDUE_TYPES:
            # Model inputs use the pipeline's unified residue space
            # (<M>=0, ALA=1, ..., TYR=20), while the released protein-only
            # autoencoder was trained on ALA=0, ..., TYR=19.
            encoder_residue_type = torch.where(
                can_be_encoded,
                encoder_residue_type - 1,
                torch.zeros_like(encoder_residue_type),
            )
        output = encoder(
            coordinates=coordinates.to(dtype=encoder.dtype),
            atom_mask=atom_mask.to(device=coordinates.device),
            residue_type=encoder_residue_type,
            mask=padding_mask,
            can_be_encoded=can_be_encoded,
            token_type=(
                None
                if "token_type" not in static_features
                else static_features["token_type"].to(device=coordinates.device)
            ),
            token_index=(
                None
                if (
                    "within_index_group_res_idx" not in static_features
                    or "condition_index-group_1_token" not in static_features
                )
                else static_features["within_index_group_res_idx"].to(device=coordinates.device)
            ),
            residue_index=(
                None
                if "residue_pdb_idx" not in static_features
                else static_features["residue_pdb_idx"].to(device=coordinates.device)
            ),
            chain_breaks=(
                None
                if "chain_breaks_per_residue" not in static_features
                else static_features["chain_breaks_per_residue"].to(device=coordinates.device)
            ),
            chain_iid=(
                None
                if "chain_iid" not in static_features
                else static_features["chain_iid"].to(device=coordinates.device)
            ),
            atom37_element=_first_feature(
                static_features, coordinates.device, "atom37_element", "ground_truth_element"
            ),
            atom37_charge=_first_feature(static_features, coordinates.device, "atom37_charge", "charge"),
            incident_bond_type_counts=_first_feature(static_features, coordinates.device, "incident_bond_type_counts"),
            generator=generator,
            sample=sample_latent,
        )
        valid = padding_mask.unsqueeze(-1).to(self.state_dtype)
        clean_state = {
            "bb_ca": coordinates[:, :, 1].to(self.state_dtype) * valid,
            "local_latents": output["z_latent"].to(self.state_dtype),
        }
        return RFProteinaCleanStateEncoding(
            clean_state=clean_state,
            mean=output["mean"],
            log_scale=output["log_scale"],
            encoding_mask=output["encoding_mask"],
        )

    def forward(
        self,
        representations: RFProteinaRepresentations,
        padding_mask: torch.Tensor,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        recycle_depth: int = 1,
        pair_in_buffers: bool = False,
        masks: TrunkMasks | None = None,
        emit_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run one velocity-network evaluation.

        The sampler calls this once per step. It is public so a harness can
        compare a single denoiser evaluation without driving a whole rollout.

        Args:
            representations: Trunk inputs, ``single`` ``[B, N, token_dim]``,
                ``pair`` ``[B, N, N, pair_dim]``, and ``condition``
                ``[B, N, condition_dim]``.
            padding_mask: ``[B, N]`` mask of real tokens.
            pair_in_buffers: The pair is the trunk's first initial-pair buffer, and both buffers already hold it;
                see ``LocalLatentsTransformer.initial_pair_buffers``.
            masks: The trunk's ``prepare_masks`` of ``padding_mask``, which a rollout builds once.

        Returns:
            Per-modality velocities, ``bb_ca.v`` ``[B, N, 3]`` and
            ``local_latents.v`` ``[B, N, latent_dim]``, plus
            ``insertion.logits`` ``[B, N, N]`` when the trunk has an insertion
            head.
        """
        return self.velocity_model(
            representations.single,
            representations.pair,
            representations.condition,
            padding_mask,
            metadata=metadata,
            buffers=buffers,
            recycle_depth=recycle_depth,
            pair_in_buffers=pair_in_buffers,
            masks=masks,
            emit_masks=emit_masks,
        )

    def _make_predict_fn(
        self,
        *,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        emit_masks: dict[str, torch.Tensor],
    ) -> RFProteinaPredictFn:
        """Bind static representations and velocity evaluation for integration.

        ``emit_masks`` are the masks the sampler integrates with, which the readout receives as given.
        """
        parts: StaticTrackParts | None = None
        masks: TrunkMasks | None = None

        def predict(
            state: RFProteinaState,
            times: dict[str, torch.Tensor],
            self_conditioning: RFProteinaState,
            self_conditioning_active: bool,
            recycle_depth: int,
        ) -> dict[str, dict[str, torch.Tensor]]:
            nonlocal parts, masks
            # The first step builds what the static features and the mask alone fix, and the rest reuse it.
            if masks is None:
                masks = self.velocity_model.prepare_masks(padding_mask)
            pair_out = None
            if parts is None:
                parts = self.representation_builder.prepare(static_features, padding_mask)
            # The builder writes the pair straight into the trunk's captured inputs when its dtype allows.
            if self.representation_builder.output_dtype == self.representation_builder.pair_adaln_dtype:
                pair_out = self.velocity_model.initial_pair_buffers(*padding_mask.shape, padding_mask.device)
            representations = self.representation_builder(
                static_features=static_features,
                state=state,
                times=times,
                self_conditioning=self_conditioning,
                self_conditioning_active=self_conditioning_active,
                padding_mask=padding_mask,
                parts=parts,
                recycle_depth=recycle_depth,
                pair_out=pair_out,
            )
            return self(
                representations,
                padding_mask,
                recycle_depth=recycle_depth,
                pair_in_buffers=pair_out is not None,
                masks=masks,
                emit_masks=emit_masks,
            )

        return predict

    @torch.no_grad()
    def partial_simulation(
        self,
        batch: Mapping[str, torch.Tensor],
        x: RFProteinaState,
        *,
        start_step: int = 0,
        end_step: int | None = None,
        initial_clean: RFProteinaState | None = None,
        carry_self_condition: bool = True,
        ts: Mapping[str, torch.Tensor],
        gt: Mapping[str, torch.Tensor],
        recycle_depths: Sequence[int],
        context: SamplingContext | None = None,
        seed: int | None = None,
        target_backbone: torch.Tensor | None = None,
        save_trajectory_every: int = 0,
    ) -> RFProteinaSimulationOutput:
        """Advance tensor state over absolute schedule intervals ``[start_step, end_step)``.

        The caller owns branching, scoring, selection, stopping, and persistence.
        Inputs are not modified. Reuse ``context`` across intervals to continue
        the same random stream; pass the previous ``clean_prediction`` to carry
        self-conditioning. Set ``carry_self_condition=False`` to restart it.

        Args:
            batch: Model inputs, including boolean ``padding_mask``
                ``[B, N]``. Conditioning tokens remain visible to the trunk.
            x: ``bb_ca`` ``[B, N, 3]`` in nanometres and ``local_latents``
                ``[B, N, latent_dim]`` at ``start_step``.
            start_step: Inclusive index into the full schedule.
            end_step: Exclusive index into the full schedule.
            initial_clean: Previous clean prediction, required when resuming
                with self-conditioning.
            carry_self_condition: Continue self-conditioning across the boundary.
            ts: Full per-modality time grids, with one more value than steps.
            gt: Per-modality noise rates, one per interval.
            recycle_depths: Explicit depths for every interval of the full schedule.
            context: Caller-owned random stream; takes precedence over ``seed``.
            seed: Seed for an independent stream when ``context`` is absent.
            target_backbone: Fixed clean backbone for inverse folding, in nm.
            save_trajectory_every: Snapshot interval, or zero to disable.

        Returns:
            State and clean prediction at the interval end, logits, and snapshots.
        """
        padding_mask = batch["padding_mask"].to(torch.bool)
        rollout_mask = _resolve_rollout_mask(batch, padding_mask)
        can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
            batch, padding_mask, None, None, self.decoder.num_residue_types
        )
        predict = self._make_predict_fn(
            static_features=batch,
            padding_mask=padding_mask,
            emit_masks={"bb_ca": rollout_mask, "local_latents": can_be_encoded},
        )
        return self.simulator.partial_simulation(
            predict=predict,
            padding_mask=rollout_mask,
            dtype=self.state_dtype,
            start_step=start_step,
            end_step=end_step,
            initial_state=x,
            initial_clean=initial_clean,
            carry_self_condition=carry_self_condition,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=extract_groups(batch) if self.simulator.group_keys else None,
            schedules={mode: FlowMatchingSchedule(times=times, noise_rate=gt[mode]) for mode, times in ts.items()},
            recycle_depths=recycle_depths,
            target_backbone=target_backbone,
            save_trajectory_every=save_trajectory_every,
            context=context,
            seed=seed,
        )

    @torch.no_grad()
    def decode(
        self,
        state: RFProteinaState,
        padding_mask: torch.Tensor,
        token_type: torch.Tensor | None = None,
        can_be_encoded: torch.Tensor | None = None,
        conditional_residue_type: torch.Tensor | None = None,
        conditional_residue_mask: torch.Tensor | None = None,
    ) -> RFProteinaDecodedOutput:
        """Return numerical decoder predictions in the original token layout.

        State holds ``bb_ca`` [B, N, 3] in nm and ``local_latents``
        [B, N, latent_dim]. Masks and residue conditions have shape [B, N].
        Sequence conditions are network inputs, not output overrides.
        Coordinates are [B, N, 37, 3] in nm; sequence and optional atom-presence
        logits are returned without atom selection or coordinate restoration.
        Undecoded tokens have zero coordinates, residue ID -1 and presence -inf.
        """
        padding_mask = padding_mask.to(device=state["bb_ca"].device, dtype=torch.bool)
        if can_be_encoded is None:
            can_be_encoded = padding_mask
        else:
            can_be_encoded = _normalize_token_mask("can_be_encoded", can_be_encoded, padding_mask)
        if token_type is not None:
            can_be_encoded = _drop_undecodable_tokens(
                can_be_encoded, token_type, padding_mask, self.decoder.num_residue_types
            )
            token_type = token_type.to(device=padding_mask.device)
        if (conditional_residue_type is None) != (conditional_residue_mask is None):
            raise ValueError("conditional_residue_type and conditional_residue_mask must be supplied together")
        if conditional_residue_type is not None:
            assert conditional_residue_mask is not None
            if (
                conditional_residue_type.shape != padding_mask.shape
                or conditional_residue_mask.shape != padding_mask.shape
            ):
                raise ValueError("conditional residue tensors must have the same shape as padding_mask")
            conditional_residue_type = conditional_residue_type.to(device=padding_mask.device)
            conditional_residue_mask = conditional_residue_mask.to(device=padding_mask.device)
        if torch.equal(can_be_encoded, padding_mask):
            decoded = self.decoder(
                backbone_ca=state["bb_ca"],
                local_latents=state["local_latents"],
                mask=padding_mask,
                token_type=token_type,
                conditional_residue_type=conditional_residue_type,
                conditional_residue_mask=conditional_residue_mask,
            )
            if "atom_presence_logits" in decoded:
                decoded["atom_presence_logits"] = decoded["atom_presence_logits"].masked_fill(
                    ~padding_mask.unsqueeze(-1), -torch.inf
                )
            decoded["coords"] = decoded["coords"].masked_fill(~padding_mask[..., None, None], 0)
            decoded["residue_type"] = decoded["residue_type"].masked_fill(~padding_mask, -1)
            return decoded

        batch, tokens = padding_mask.shape
        coordinates = torch.zeros(batch, tokens, ATOM37_SIZE, 3, device=padding_mask.device, dtype=self.decoder.dtype)
        sequence_logits = torch.zeros(
            batch, tokens, self.decoder.num_residue_types, device=padding_mask.device, dtype=self.decoder.dtype
        )
        residue_types = torch.full((batch, tokens), -1, device=padding_mask.device, dtype=torch.long)
        presence = (
            torch.full(
                (*padding_mask.shape, ATOM37_SIZE), -torch.inf, device=padding_mask.device, dtype=self.decoder.dtype
            )
            if self.config.decoder.use_atom_presence
            else None
        )
        for batch_index in range(batch):
            encoded = can_be_encoded[batch_index]
            if torch.any(encoded):
                encoded_count = int(encoded.sum())
                decoded = self.decoder(
                    backbone_ca=state["bb_ca"][batch_index, encoded].unsqueeze(0),
                    local_latents=state["local_latents"][batch_index, encoded].unsqueeze(0),
                    mask=torch.ones(1, encoded_count, device=padding_mask.device, dtype=torch.bool),
                    token_type=(None if token_type is None else token_type[batch_index, encoded].unsqueeze(0)),
                    conditional_residue_type=(
                        None
                        if conditional_residue_type is None
                        else conditional_residue_type[batch_index, encoded].unsqueeze(0)
                    ),
                    conditional_residue_mask=(
                        None
                        if conditional_residue_mask is None
                        else conditional_residue_mask[batch_index, encoded].unsqueeze(0)
                    ),
                )
                coordinates[batch_index, encoded] = decoded["coords"][0]
                sequence_logits[batch_index, encoded] = decoded["sequence_logits"][0]
                residue_types[batch_index, encoded] = decoded["residue_type"][0]
                if presence is not None:
                    presence[batch_index, encoded] = decoded["atom_presence_logits"][0]
        output: RFProteinaDecodedOutput = {
            "coords": coordinates,
            "sequence_logits": sequence_logits,
            "residue_type": residue_types,
        }
        if presence is not None:
            output["atom_presence_logits"] = presence
        return output
