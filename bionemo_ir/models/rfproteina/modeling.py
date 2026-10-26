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
from dataclasses import replace
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
from bionemo_ir._torch.layers.transformers.local_latents_decoder import get_residue_atom_mask
from bionemo_ir._torch.layers.transformers.local_latents_transformer import LocalLatentsTransformer
from bionemo_ir._torch.modules.rfproteina import (
    ATOMIZED_TOKEN_TYPE,
    CONDITION_INDEX_MASK,
    DNA_TOKEN_TYPE,
    RNA_TOKEN_TYPE,
    RFProteinaCleanStateEncoding,
    RFProteinaConversionReport,
    RFProteinaCorruptionOutput,
    RFProteinaLatentDecoder,
    RFProteinaLatentEncoder,
    RFProteinaPartialRolloutSetup,
    RFProteinaPartialSamplingOutput,
    RFProteinaPredictFn,
    RFProteinaPredictionOutput,
    RFProteinaProductSpaceSampler,
    RFProteinaRepresentationBuilder,
    RFProteinaRepresentationBuilderFn,
    RFProteinaRepresentations,
    RFProteinaSamplingOutput,
    RFProteinaSearchOutput,
    RFProteinaState,
    apply_insertion_assignment,
    assign_unindexed_tokens,
    create_fraction_partial_schedules,
    create_modality_schedule,
    extract_groups,
    materialize_atomized_output,
    materialize_nucleic_acid_output,
    materialize_partial_redesign_output,
    partial_num_steps,
    resolve_partial_times,
    sample_reference_state,
    validate_nucleic_acid_contract,
)
from bionemo_ir._torch.modules.rfproteina.atom_utils import finite_atom_mask
from bionemo_ir._torch.modules.rfproteina.groups import CONDITION_COORDINATE_GROUP
from bionemo_ir._torch.modules.rfproteina.metrics import RFProteinaLocalMetricEvaluator
from bionemo_ir._torch.modules.rfproteina.representation_builder import (
    ATOM37_SIZE,
    CONDITION_COORDINATE,
    CONDITION_COORDINATE_MASK,
    CONDITION_SEQUENCE,
    CONDITION_SEQUENCE_MASK,
    UNSPECIFIED_RESIDUE_TOKEN,
)
from bionemo_ir._torch.modules.rfproteina.search import run_rfproteina_beam_search
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import FlowMatchingSchedule
from bionemo_ir._torch.search import CompositeMetricEvaluator, MetricEvaluatorLike, derive_search_seed
from bionemo_ir.configs import AcceleratedConfig, BackendType, BaseConfig
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint
from bionemo_ir.hubs.support_matrix import GenerativeSupportMatrix as SupMat
from bionemo_ir.models.rfproteina.config import (
    PRETRAINED_CONFIG_REGISTRY,
    RFProteinaBeamSearchConfig,
    RFProteinaConfig,
    RFProteinaPartialSimulationConfig,
    RFProteinaSamplingConfig,
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
    rfproteina_decoder_config_from_weights,
    rfproteina_encoder_config_from_checkpoint,
    rfproteina_encoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_weights,
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
        if token_type.shape != padding_mask.shape:
            raise ValueError(f"token_type must have shape {tuple(padding_mask.shape)}, got {tuple(token_type.shape)}")
        can_be_encoded &= token_type.to(device=padding_mask.device) != ATOMIZED_TOKEN_TYPE
        if decoder_num_residue_types == 20:
            token_type = token_type.to(device=padding_mask.device)
            can_be_encoded &= (token_type != RNA_TOKEN_TYPE) & (token_type != DNA_TOKEN_TYPE)

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


def _repeat_state(state: RFProteinaState, repeats: int) -> RFProteinaState:
    """Tile a product-space state along the batch axis."""
    return {name: tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1)) for name, tensor in state.items()}


class RFProteina(nn.Module, OptimizedModuleSetterMixin):
    """Torch-only RFProteína model with a model-local VF-TSR sampler."""

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
        sampling_config: RFProteinaSamplingConfig | None = None,
        device: torch.device | str | None = None,
        attention_backend: str | None = None,
        noise_align_chirality: bool = True,
        noise_align_chirality_translate: bool = False,
        lora_alpha: float | None = RELEASED_LORA_ALPHA,
    ) -> "RFProteina":
        """Build a model from weights a host already holds, inferring every width.

        An embedding application that has loaded and selected its own weights —
        applying an EMA, say — can hand them straight over instead of naming a
        file the model would reopen. Both autoencoder halves load, so redesign
        and inverse folding need no second pass.

        Args:
            flow_state: Released score-network weights under the ``nn.`` prefix.
            autoencoder_state: Released autoencoder weights under the
                ``encoder.`` / ``decoder.`` prefixes.
            sampling_config: Inference policy. Defaults to the released one.
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
        trunk = rfproteina_trunk_config_from_weights(
            flow,
            dtype=reference.trunk.dtype,
            pairwise_attention_backend=reference.trunk.pairwise_attention_backend,
        )
        decoder = rfproteina_decoder_config_from_weights(
            autoencoder,
            dtype=reference.decoder.dtype,
            transformer_dtype=reference.decoder.transformer_dtype,
            pairwise_attention_backend=reference.decoder.pairwise_attention_backend,
        )
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
            sampling=sampling_config or reference.sampling,
        )
        model = cls(config)
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
            capacity: Graphs this trunk retains. Each one holds its own static
                buffers, so this trades GPU memory for avoided recapture.
            verify_capture: Compare the first replay against eager output.
            max_tokens: Refuse to capture past this token count, running eager
                there instead. ``None`` keeps the trunk's declared limit.
        """
        if capacity < 1:
            raise ValueError(f"CUDA graph capacity must be positive, got {capacity}")
        # Release the previous policy's graphs: they replay the buffers they captured.
        self.reset_graphs()
        config = trunk_graph_config() if max_tokens is None else trunk_graph_config(max_tokens)
        config.num_graphs_max_for_this_module = capacity
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
        self.state_dtype = str_dtype_to_torch(self.config.sampling.state_dtype)
        self.representation_builder = RFProteinaRepresentationBuilder(self.config.representation, self.config.trunk)
        self.velocity_model = LocalLatentsTransformer(self.config.trunk)
        self.decoder = RFProteinaLatentDecoder(self.config.decoder)
        self.encoder = None if self.config.encoder is None else RFProteinaLatentEncoder(self.config.encoder)
        self.sampler = RFProteinaProductSpaceSampler(
            self.config.sampling,
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
        return self.conversion_report

    def load_decoder_weights(self, checkpoint_path: str | Path) -> RFProteinaConversionReport:
        """Load the released partial-autoencoder decoder weights."""
        self.decoder_conversion_report = load_rfproteina_decoder_checkpoint(
            self.decoder,
            checkpoint_path,
            strict=True,
        )
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
    def encode_clean_state(
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
        if encoder.num_residue_types == 20:
            # Model inputs use the pipeline's unified residue space
            # (<M>=0, ALA=1, ..., TYR=20), while the released protein-only
            # autoencoder was trained on ALA=0, ..., TYR=19.
            encoder_residue_type = torch.where(
                can_be_encoded,
                encoder_residue_type - 1,
                torch.zeros_like(encoder_residue_type),
            )
        output = encoder(
            sampling_stream=self.config.sampling.match_reference_rng,
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

    @torch.no_grad()
    def forward(
        self,
        static_features: Mapping[str, torch.Tensor],
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        initial_state: RFProteinaState | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
    ) -> RFProteinaPredictionOutput:
        """Generate a design: draw a reference state, integrate it, decode it.

        This is the whole inference path, so ``model(batch, seed=0)`` is a
        complete call: the batch describes the tokens and everything else is
        drawn or built internally. ``initial_state`` and
        ``representation_builder`` exist for harnesses that need to pin one
        stage against a reference implementation, not for normal use.

        Shapes use ``B`` for batch and ``N`` for tokens; coordinates are in
        nanometres.

        Args:
            static_features: Batch of per-token features, each leading with
                ``[B, N, ...]``; :class:`RFProteinaRepresentationBuilder`
                documents the keys it reads. Must carry ``padding_mask``
                ``[B, N]``, marking real tokens. Lower-level :meth:`sample`
                takes that mask separately because it can run from a
                representation builder alone, with no batch at all. Tensorized
                motif conditioning uses ``condition_coordinate_1_atom``. Finite
                values mark the conditioned atoms when
                ``mask_coordinate_1_atom`` is omitted. Internal coordinates use
                ``condition_InternalCoordinate_1_atom`` /
                ``mask_InternalCoordinate_1_atom``,
                ``condition_sequence_1_residue`` /
                ``mask_sequence_1_residue``, and
                ``condition_index-group_1_token``. When
                ``mask_index_1_residue`` is present, ``chain_iid`` enables
                same-chain Hungarian assignment of unindexed tokens. Optional
                ``can_be_encoded`` and ``coord_cond_mask`` provide token-level
                sampler masks. ``flow_backbone_mask`` ``[B, N]`` names the
                tokens whose backbone the flow generates; a token outside it
                only conditions, as an unindexed motif residue does, so it holds
                zero state, takes no step, and is left out of centering while the
                trunk still attends to it. It defaults to every real token, and
                narrows ``can_be_encoded`` to the same tokens. A checkpoint
                trained with noise-aligned chirality
                (``config.representation.noise_align_chirality``) also reads
                ``chiral_frame_atoms`` ``[B, N, S, 2 + K]`` and
                ``chiral_slot_owner`` ``[B, N, S]`` beside ``chirality``.
                Atomized small-molecule atoms use
                ``token_type == 3``, ``can_be_encoded == false``, exactly one
                occupied ``atom37_mask`` slot (slot 1), unified
                ``residue_type == 32``, and optional chemical/bond features.
                RNA and DNA use ``token_type`` 1 and 2 plus unified residue
                classes 22–26 and 27–31. A 33-class decoder can generate them;
                a legacy 20-class decoder requires complete coordinate and
                sequence conditions for every nucleic-acid token.
            initial_state: Product-space state to start from instead of a fresh
                reference draw: ``bb_ca`` ``[B, N, 3]`` and ``local_latents``
                ``[B, N, latent_dim]``.
            can_be_encoded: ``[B, N]``, false on tokens the autoencoder cannot
                encode. Their latents are pinned to zero for the whole rollout,
                so reference noise never reaches the trunk on those tokens.
                Atomized tokens and DNA/RNA tokens used with the legacy
                20-class decoder are forced false from ``token_type``.
            coordinate_condition_mask: ``[B, N]``, true on tokens conditioned
                on absolute coordinates. Stochastic centering leaves them at
                the origin rather than translating them with the rest. When
                omitted, it is read from ``coord_cond_mask``, reduced with
                ``any`` over ``mask_coordinate_1_atom``, or taken from finite
                values in ``condition_coordinate_1_atom``.

        Returns:
            A :class:`RFProteinaPredictionOutput`: decoded ``coordinates``
            ``[B, N, 37, 3]``, ``sequence_logits``
            ``[B, N, num_residue_types]``, ``residue_types`` ``[B, N]``, and
            the :class:`RFProteinaSamplingOutput` the decode ran on. When index
            and chain metadata are supplied, ``insertion_assignment [B, N]``
            contains the generated token selected for each unindexed token;
            ``coordinates_with_insertion`` and
            ``residue_types_with_insertion`` materialize motif conditions
            without changing the raw decoder outputs. For atomized tokens,
            ``atomized`` preserves identity, chemistry, masks, and bond
            topology; ``complex_coordinates`` combines sampled ligand atoms
            with decoded polymers and restores fixed-ligand coordinate
            conditions. ``nucleic_acid`` exposes canonical DNA/RNA rows with
            fixed coordinate and sequence conditions restored;
            ``complex_residue_types`` and ``complex_atom_mask`` provide the
            matching unified identities and atom37 occupancy.

        Raises:
            ValueError: If the batch does not carry ``padding_mask``.
        """
        if "padding_mask" not in static_features:
            raise ValueError(
                "static_features must carry 'padding_mask'; it defines which "
                "tokens are real and there is nothing to generate without it"
            )
        padding_mask = static_features["padding_mask"]
        can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
            static_features,
            padding_mask,
            can_be_encoded,
            coordinate_condition_mask,
            self.decoder.num_residue_types,
        )
        token_type = static_features.get("token_type")
        if token_type is not None and torch.any((token_type == 1) | (token_type == 2)):
            validate_nucleic_acid_contract(
                static_features,
                can_be_encoded,
                self.decoder.num_residue_types,
                supports_na_secondary_structure=self.config.representation.supports_nucleic_acid_secondary_structure,
            )
        sampling = self.sample(
            padding_mask=padding_mask,
            representation_builder=representation_builder,
            initial_state=initial_state,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            generator=generator,
            save_trajectory_every=save_trajectory_every,
            metadata=metadata,
            buffers=buffers,
            seed=seed,
            static_features=static_features,
        )
        return self._prediction_from_sampling(sampling, static_features, padding_mask, can_be_encoded)

    def _condition_on_predicted_sequence(
        self,
        trunk_sequence_logits: torch.Tensor | None,
        padding_mask: torch.Tensor,
        can_be_encoded: torch.Tensor,
        token_type: torch.Tensor | None,
        conditional_residue_type: torch.Tensor | None,
        conditional_residue_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill unspecified decoded tokens with the trunk readout's argmax.

        Caller-supplied sequence conditions win; only tokens the autoencoder
        decodes and the caller left unspecified take a predicted class.
        """
        if trunk_sequence_logits is None:
            raise ValueError(
                "condition_decoder_on_sequence_prediction needs the trunk sequence readout; "
                "the loaded checkpoint has none (trunk.use_sequence_readout is false)"
            )
        if conditional_residue_type is None:
            conditional_residue_type = torch.full_like(padding_mask, UNSPECIFIED_RESIDUE_TOKEN, dtype=torch.long)
            conditional_residue_mask = torch.zeros_like(padding_mask)
        assert conditional_residue_mask is not None
        logits = trunk_sequence_logits.to(device=padding_mask.device)
        if logits.shape[-1] != self.decoder.num_residue_types:
            raise ValueError(
                f"trunk sequence readout emits {logits.shape[-1]} classes, but the decoder "
                f"has {self.decoder.num_residue_types}"
            )
        allowed = self.decoder.allowed_residue_types(logits.device)
        if allowed is not None and token_type is not None:
            logits = logits.masked_fill(
                ~allowed[token_type.clamp(min=0, max=allowed.shape[0] - 1).long()],
                torch.finfo(logits.dtype).min,
            )
        predicted = logits.argmax(dim=-1)
        fill = can_be_encoded & ~conditional_residue_mask
        return (
            torch.where(fill, predicted.to(conditional_residue_type.dtype), conditional_residue_type),
            conditional_residue_mask | fill,
        )

    def _prediction_from_sampling(
        self,
        sampling: RFProteinaSamplingOutput,
        static_features: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        can_be_encoded: torch.Tensor,
    ) -> RFProteinaPredictionOutput:
        """Decode and materialize all model-level output views."""
        token_type = static_features.get("token_type")
        conditional_residue_type = static_features.get(CONDITION_SEQUENCE)
        conditional_residue_mask = static_features.get(CONDITION_SEQUENCE_MASK)
        if conditional_residue_type is not None and conditional_residue_mask is None:
            conditional_residue_mask = conditional_residue_type != UNSPECIFIED_RESIDUE_TOKEN
        if self.config.sampling.condition_decoder_on_sequence_prediction:
            conditional_residue_type, conditional_residue_mask = self._condition_on_predicted_sequence(
                sampling.trunk_sequence_logits,
                padding_mask,
                can_be_encoded,
                token_type,
                conditional_residue_type,
                conditional_residue_mask,
            )
        coordinate_group = static_features.get(CONDITION_COORDINATE_GROUP)
        decoded = self.decode(
            sampling.state,
            padding_mask,
            token_type=token_type,
            can_be_encoded=can_be_encoded,
            conditional_residue_type=conditional_residue_type,
            conditional_residue_mask=conditional_residue_mask,
            required_atom_mask=None if coordinate_group is None else coordinate_group >= 0,
        )
        atomized = materialize_atomized_output(sampling.state["bb_ca"], static_features)
        nucleic_acid = None
        if token_type is not None:
            nucleic_acid = materialize_nucleic_acid_output(
                decoded["coords"],
                decoded["residue_type"],
                decoded["atom_mask"],
                static_features,
                self.decoder.num_residue_types,
            )
        insertion = None
        if CONDITION_INDEX_MASK in static_features:
            if "chain_iid" not in static_features:
                raise ValueError(f"static_features must carry 'chain_iid' when '{CONDITION_INDEX_MASK}' is supplied")
            if sampling.insertion_logits is None:
                insertion_assignment = torch.full_like(padding_mask, -1, dtype=torch.long)
            else:
                insertion_assignment = assign_unindexed_tokens(
                    sampling.insertion_logits,
                    padding_mask,
                    static_features[CONDITION_INDEX_MASK],
                    static_features["chain_iid"],
                    atom37_mask=static_features.get("atom37_mask"),
                )
            insertion = apply_insertion_assignment(
                decoded["coords"],
                decoded["residue_type"],
                insertion_assignment,
                static_features,
                condition_sequence_offset=(-1 if self.decoder.num_residue_types == 20 else 0),
            )
        return RFProteinaPredictionOutput(
            sampling=sampling,
            coordinates=decoded["coords"],
            sequence_logits=decoded["sequence_logits"],
            residue_types=decoded["residue_type"],
            atom_mask=decoded["atom_mask"],
            atomized=atomized,
            nucleic_acid=nucleic_acid,
            insertion=insertion,
            residue_type_offset=(1 if self.decoder.num_residue_types == 20 else 0),
        )

    def velocity(
        self,
        representations: RFProteinaRepresentations,
        padding_mask: torch.Tensor,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        recycle_depth: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run one velocity-network evaluation.

        The sampler calls this once per step. It is public so a harness can
        compare a single denoiser evaluation without driving a whole rollout.

        Args:
            representations: Trunk inputs, ``single`` ``[B, N, token_dim]``,
                ``pair`` ``[B, N, N, pair_dim]``, and ``condition``
                ``[B, N, condition_dim]``.
            padding_mask: ``[B, N]`` mask of real tokens.

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
        )

    def _make_predict_fn(
        self,
        *,
        static_features: Mapping[str, torch.Tensor] | None,
        padding_mask: torch.Tensor,
        representation_builder: RFProteinaRepresentationBuilderFn | None,
        metadata: AttentionMetadata | None,
        buffers: PreallocatedBuffers | None,
    ) -> RFProteinaPredictFn:
        """Bind representation construction and velocity evaluation for a rollout."""
        if representation_builder is None:
            if static_features is None:
                raise ValueError("static_features are required when no custom representation_builder is supplied")

            def resolved_builder(
                state: RFProteinaState,
                times: dict[str, torch.Tensor],
                self_conditioning: RFProteinaState,
                self_conditioning_active: bool,
            ) -> RFProteinaRepresentations:
                return self.representation_builder(
                    static_features=static_features,
                    state=state,
                    times=times,
                    self_conditioning=self_conditioning,
                    self_conditioning_active=self_conditioning_active,
                    padding_mask=padding_mask,
                )

        else:
            resolved_builder = representation_builder

        def predict(
            state: RFProteinaState,
            times: dict[str, torch.Tensor],
            self_conditioning: RFProteinaState,
            self_conditioning_active: bool,
            recycle_depth: int,
        ) -> dict[str, dict[str, torch.Tensor]]:
            representations = resolved_builder(state, times, self_conditioning, self_conditioning_active)
            representations = self.bind_recycle_conditions(
                representations,
                static_features,
                state,
                times,
                padding_mask,
                recycle_depth,
                expand=representation_builder is None,
            )
            return self.velocity(
                representations,
                padding_mask,
                metadata=metadata,
                buffers=buffers,
                recycle_depth=recycle_depth,
            )

        return predict

    def bind_recycle_conditions(
        self,
        representations: RFProteinaRepresentations,
        static_features: Mapping[str, torch.Tensor] | None,
        state: RFProteinaState,
        times: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        recycle_depth: int,
        *,
        expand: bool = True,
    ) -> RFProteinaRepresentations:
        """Attach one condition per trunk pass when the checkpoint has a recycle feature.

        The builder's own condition is trunk pass 1. Later passes differ only
        in that feature. A caller-supplied representation builder keeps the
        condition it returned.
        """
        if (
            not expand
            or recycle_depth <= 1
            or static_features is None
            or not self.representation_builder.conditions_per_recycle_pass
        ):
            return representations
        return replace(
            representations,
            condition=self.representation_builder.condition_stack(
                static_features,
                state,
                times,
                padding_mask,
                recycle_depth,
            ),
        )

    def sample(
        self,
        padding_mask: torch.Tensor,
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        initial_state: RFProteinaState | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
        static_features: Mapping[str, torch.Tensor] | None = None,
    ) -> RFProteinaSamplingOutput:
        """Sample product space using native or caller-supplied representations.

        The rollout without the decode, for callers that want the latent state
        itself. See :meth:`forward` for the tensor arguments it shares.

        Args:
            padding_mask: ``[B, N]`` mask of real tokens, which fixes ``B`` and
                ``N`` for every tensor the rollout draws.
            save_trajectory_every: Snapshot the state every this many steps
                (plus the last), or 0 for none.

        Returns:
            A :class:`RFProteinaSamplingOutput`: the final ``state`` and the
            network's ``clean_prediction``, both ``bb_ca`` ``[B, N, 3]`` and
            ``local_latents`` ``[B, N, latent_dim]``, optional
            ``insertion_logits`` ``[B, N, N]``, and the requested
            ``trajectory`` snapshots.
        """
        padding_mask = padding_mask.to(torch.bool)
        if static_features is not None:
            can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
                static_features,
                padding_mask,
                can_be_encoded,
                coordinate_condition_mask,
                self.decoder.num_residue_types,
            )

        predict = self._make_predict_fn(
            static_features=static_features,
            padding_mask=padding_mask,
            representation_builder=representation_builder,
            metadata=metadata,
            buffers=buffers,
        )
        # Only per-group stochastic centering reads these, so pay for the
        # extraction only when the configuration asks for it.
        groups = extract_groups(static_features) if static_features is not None and self.sampler.group_keys else None
        return self.sampler.sample(
            predict=predict,
            padding_mask=_resolve_rollout_mask(static_features, padding_mask),
            dtype=self.state_dtype,
            initial_state=initial_state,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            generator=generator,
            save_trajectory_every=save_trajectory_every,
            seed=seed,
        )

    @torch.no_grad()
    def sample_partial(
        self,
        clean_state: RFProteinaState,
        start_times: dict[str, float],
        num_steps: int,
        padding_mask: torch.Tensor,
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
        static_features: Mapping[str, torch.Tensor] | None = None,
        resample_target_backbone: bool = False,
        target_backbone: torch.Tensor | None = None,
        schedules: Mapping[str, FlowMatchingSchedule] | None = None,
    ) -> RFProteinaPartialSamplingOutput:
        """Corrupt a clean endpoint and sample over partial schedule tails.

        ``schedules`` replaces the tails otherwise rebuilt from ``start_times``.
        """
        padding_mask = padding_mask.to(torch.bool)
        if static_features is not None:
            can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
                static_features,
                padding_mask,
                can_be_encoded,
                coordinate_condition_mask,
                self.decoder.num_residue_types,
            )

        predict = self._make_predict_fn(
            static_features=static_features,
            padding_mask=padding_mask,
            representation_builder=representation_builder,
            metadata=metadata,
            buffers=buffers,
        )
        groups = extract_groups(static_features) if static_features is not None and self.sampler.group_keys else None
        return self.sampler.sample_partial(
            predict=predict,
            clean_state=clean_state,
            start_times=start_times,
            num_steps=num_steps,
            padding_mask=_resolve_rollout_mask(static_features, padding_mask),
            dtype=self.state_dtype,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            generator=generator,
            save_trajectory_every=save_trajectory_every,
            seed=seed,
            resample_target_backbone=resample_target_backbone,
            target_backbone=target_backbone,
            schedules=schedules,
        )

    @torch.no_grad()
    def redesign(
        self,
        static_features: Mapping[str, torch.Tensor],
        partial_config: RFProteinaPartialSimulationConfig,
        coordinates: torch.Tensor | None = None,
        clean_state: RFProteinaState | None = None,
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sample_latent: bool = True,
        save_trajectory_every: int = 0,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
    ) -> RFProteinaPredictionOutput:
        """Redesign an existing structure from an encoded or supplied endpoint.

        The default path encodes ``coordinates`` (nanometres). Advanced callers
        may instead supply ``clean_state``. Fixed coordinate conditions affect
        centering and final materialization, but are not per-step clamps.
        Inverse folding (``partial_config.inverse_folding``) keeps the encoded
        backbone and redesigns only local latents.
        """
        if "padding_mask" not in static_features:
            raise ValueError("static_features must carry 'padding_mask' for redesign")
        if coordinates is not None and clean_state is not None:
            raise ValueError("provide either coordinates or clean_state, not both")
        padding_mask = static_features["padding_mask"].to(torch.bool)
        can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
            static_features,
            padding_mask,
            can_be_encoded,
            coordinate_condition_mask,
            self.decoder.num_residue_types,
        )
        start_times = resolve_partial_times(partial_config, self.config.sampling, padding_mask.device)
        if not start_times:
            raise ValueError("redesign requires partial start times or partial_trajectory_fraction_steps")
        num_steps = partial_num_steps(partial_config, self.config.sampling.num_steps)
        fraction_schedules = (
            None
            if partial_config.partial_trajectory_fraction_steps is None
            else create_fraction_partial_schedules(self.config.sampling, num_steps, padding_mask.device)
        )

        def run(active_generator: torch.Generator) -> RFProteinaPredictionOutput:
            resolved_clean_state = clean_state
            if resolved_clean_state is None:
                encoding = self.encode_clean_state(
                    static_features,
                    coordinates=coordinates,
                    can_be_encoded=can_be_encoded,
                    generator=active_generator,
                    sample_latent=sample_latent,
                )
                resolved_clean_state = encoding.clean_state
            partial_sampling = self.sample_partial(
                clean_state=resolved_clean_state,
                start_times=start_times,
                num_steps=num_steps,
                padding_mask=padding_mask,
                representation_builder=representation_builder,
                can_be_encoded=can_be_encoded,
                coordinate_condition_mask=coordinate_condition_mask,
                generator=active_generator,
                save_trajectory_every=save_trajectory_every,
                metadata=metadata,
                buffers=buffers,
                static_features=static_features,
                resample_target_backbone=partial_config.inverse_folding,
                target_backbone=resolved_clean_state["bb_ca"] if partial_config.inverse_folding else None,
                schedules=fraction_schedules,
            )
            prediction = self._prediction_from_sampling(
                partial_sampling.sampling,
                static_features,
                padding_mask,
                can_be_encoded,
            )
            partial_output = materialize_partial_redesign_output(
                prediction.coordinates,
                static_features,
                partial_sampling,
            )
            return replace(prediction, partial_redesign=partial_output)

        if generator is not None:
            return run(generator)
        with SamplingContext.graph_safe(padding_mask.device, seed) as context:
            return run(context.generator)

    @torch.no_grad()
    def inverse_fold(
        self,
        static_features: Mapping[str, torch.Tensor],
        coordinates: torch.Tensor | None = None,
        clean_state: RFProteinaState | None = None,
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sample_latent: bool = True,
        save_trajectory_every: int = 0,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
    ) -> RFProteinaPredictionOutput:
        """Redesign sequence latents while holding the supplied backbone fixed.

        This is upstream ``generation=inverse_folding``: full-noise start times
        on both modalities, the backbone redrawn from the target's forward
        marginal every step, and the clean backbone restored at the end.
        """
        return self.redesign(
            static_features,
            RFProteinaPartialSimulationConfig(inverse_folding=True),
            coordinates=coordinates,
            clean_state=clean_state,
            representation_builder=representation_builder,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            generator=generator,
            sample_latent=sample_latent,
            save_trajectory_every=save_trajectory_every,
            metadata=metadata,
            buffers=buffers,
            seed=seed,
        )

    @torch.no_grad()
    def search(
        self,
        static_features: Mapping[str, torch.Tensor],
        search_config: RFProteinaBeamSearchConfig,
        *,
        evaluators: Sequence[MetricEvaluatorLike] | None = None,
        representation_builder: RFProteinaRepresentationBuilderFn | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        partial_config: RFProteinaPartialSimulationConfig | None = None,
        coordinates: torch.Tensor | None = None,
        clean_state: RFProteinaState | None = None,
        sample_latent: bool = True,
    ) -> RFProteinaSearchOutput:
        """Run metric-weighted beam search over product-space rollouts.

        Existing ``sample``, ``redesign``, and ``inverse_fold`` paths are
        unchanged. Search is tensor-in / tensor-out: featurization, CIF I/O,
        and oracle execution stay with the caller. Pass ``evaluators`` to
        combine local, custom, and oracle-backed metrics.

        Args:
            static_features: Native feature batch. Must include ``padding_mask``.
            search_config: Checkpoints, beam width, branching, and reward weights.
            evaluators: Ordered plain or contextual metric evaluators. Their
                per-row metric mappings are merged and keys must not overlap.
                Contextual evaluators receive the checkpoint, request IDs, and
                ``reward_metrics_tags``. ``None`` uses the local clash, motif
                RMSD, sequence, and ligand-contact evaluator. An explicit
                sequence replaces that default.
            partial_config: Optional redesign / inverse-fold start-time contract.
                When set, the beam starts from a corrupted clean endpoint.
            seed: Root seed. Child replica/branch/checkpoint streams are derived
                from it and do not depend on chunk processing order.
            generator: If given, only its ``initial_seed()`` is used as the root.

        Returns:
            Intermediates (if a save threshold is set) followed by the final
            beam, plus a cost report.
        """
        if "padding_mask" not in static_features:
            raise ValueError("static_features must carry 'padding_mask' for search")
        if coordinates is not None and clean_state is not None:
            raise ValueError("provide either coordinates or clean_state, not both")
        resolved_evaluators = (RFProteinaLocalMetricEvaluator(),) if evaluators is None else tuple(evaluators)
        evaluator = CompositeMetricEvaluator(resolved_evaluators)
        padding_mask = static_features["padding_mask"].to(torch.bool)
        can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
            static_features,
            padding_mask,
            can_be_encoded,
            coordinate_condition_mask,
            self.decoder.num_residue_types,
        )
        token_type = static_features.get("token_type")
        if token_type is not None and torch.any((token_type == 1) | (token_type == 2)):
            validate_nucleic_acid_contract(
                static_features,
                can_be_encoded,
                self.decoder.num_residue_types,
                supports_na_secondary_structure=self.config.representation.supports_nucleic_acid_secondary_structure,
            )
        device = padding_mask.device
        rollout_mask = _resolve_rollout_mask(static_features, padding_mask)
        root_seed = (
            int(generator.initial_seed()) if generator is not None else SamplingContext.create(device, seed).seed
        )
        groups = extract_groups(static_features) if self.sampler.group_keys else None
        predict = self._make_predict_fn(
            static_features=static_features,
            padding_mask=padding_mask,
            representation_builder=representation_builder,
            metadata=metadata,
            buffers=buffers,
        )

        start_times: dict[str, float] = {}
        base_corruption: RFProteinaCorruptionOutput | None = None
        target_backbone = None
        if partial_config is not None:
            start_times = resolve_partial_times(partial_config, self.config.sampling, device)
            if not start_times:
                raise ValueError("search with partial_config requires partial start times or inverse_folding")
            num_steps = partial_num_steps(partial_config, self.config.sampling.num_steps)
            resolved_clean = clean_state
            if resolved_clean is None:
                encoding = self.encode_clean_state(
                    static_features,
                    coordinates=coordinates,
                    can_be_encoded=can_be_encoded,
                    generator=SamplingContext.create(device, derive_search_seed(root_seed, 3)).generator,
                    sample_latent=sample_latent,
                )
                resolved_clean = encoding.clean_state
            partial_schedules: Mapping[str, FlowMatchingSchedule] | None = None
            fraction_schedules = (
                None
                if partial_config.partial_trajectory_fraction_steps is None
                else create_fraction_partial_schedules(self.config.sampling, num_steps, device)
            )
            initial_slots = []
            for replica in range(search_config.beam_width):
                replica_context = SamplingContext.create(device, derive_search_seed(root_seed, 0, replica))
                setup: RFProteinaPartialRolloutSetup = self.sampler.prepare_partial_rollout(
                    resolved_clean,
                    start_times,
                    num_steps,
                    rollout_mask,
                    self.state_dtype,
                    replica_context.generator,
                    can_be_encoded=can_be_encoded,
                    coordinate_condition_mask=coordinate_condition_mask,
                    groups=groups,
                    schedules=partial_schedules if partial_schedules is not None else fraction_schedules,
                    resample_target_backbone=partial_config.inverse_folding,
                )
                if partial_schedules is None:
                    base_corruption = setup.corruption
                    partial_schedules = setup.schedules
                    target_backbone = setup.target_backbone
                initial_slots.append(setup.corruption.corrupted_state)
            if partial_schedules is None:
                raise RuntimeError("beam_width must produce at least one partial rollout")
            schedules = partial_schedules
        else:
            if coordinates is not None or clean_state is not None:
                raise ValueError("coordinates and clean_state require partial_config")
            num_steps = self.config.sampling.num_steps
            schedules = {
                "bb_ca": create_modality_schedule(self.config.sampling.backbone, num_steps, device),
                "local_latents": create_modality_schedule(self.config.sampling.local_latents, num_steps, device),
            }
            initial_slots = []
            for replica in range(search_config.beam_width):
                replica_context = SamplingContext.create(device, derive_search_seed(root_seed, 0, replica))
                initial_slots.append(
                    sample_reference_state(
                        rollout_mask,
                        self.config.trunk.latent_dim,
                        self.state_dtype,
                        self.config.sampling,
                        generator=replica_context.generator,
                        can_be_encoded=can_be_encoded,
                        coordinate_condition_mask=coordinate_condition_mask,
                        groups=groups,
                    )
                )

        def decode_prediction(
            sampling: RFProteinaSamplingOutput,
            features: Mapping[str, torch.Tensor],
        ) -> RFProteinaPredictionOutput:
            batch_padding = features["padding_mask"].to(torch.bool)
            encoded, _ = _resolve_runtime_masks(
                features,
                batch_padding,
                None,
                None,
                self.decoder.num_residue_types,
            )
            prediction = self._prediction_from_sampling(sampling, features, batch_padding, encoded)
            if base_corruption is None or not start_times:
                return prediction
            repeats = batch_padding.shape[0] // padding_mask.shape[0]
            tiled_corruption = RFProteinaCorruptionOutput(
                clean_state=_repeat_state(base_corruption.clean_state, repeats),
                reference_state=_repeat_state(base_corruption.reference_state, repeats),
                corrupted_state=_repeat_state(base_corruption.corrupted_state, repeats),
                times={
                    name: time.repeat(repeats) if time.ndim == 1 and time.shape[0] == padding_mask.shape[0] else time
                    for name, time in base_corruption.times.items()
                },
            )
            partial_sampling = RFProteinaPartialSamplingOutput(
                sampling=sampling,
                corruption=tiled_corruption,
                start_times=dict(start_times),
                num_steps=num_steps,
            )
            return replace(
                prediction,
                partial_redesign=materialize_partial_redesign_output(
                    prediction.coordinates,
                    features,
                    partial_sampling,
                ),
            )

        return run_rfproteina_beam_search(
            sampler=self.sampler,
            predict=predict,
            decode_prediction=decode_prediction,
            evaluator=evaluator,
            objective=search_config.rewards.to_weighted_objective(),
            static_features=static_features,
            padding_mask=rollout_mask,
            dtype=self.state_dtype,
            initial_slots=initial_slots,
            num_steps=num_steps,
            step_checkpoints=search_config.step_checkpoints,
            n_branch=search_config.n_branch,
            beam_width=search_config.beam_width,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            schedules=schedules,
            target_backbone=target_backbone,
            scoring_chunk_size=search_config.scoring_chunk_size,
            intermediate_save_min_reward=search_config.intermediate_save_min_reward,
            root_seed=root_seed,
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
        required_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode a final product-space state into atom37 coordinates/sequence.

        Args:
            state: Product-space state, ``bb_ca`` ``[B, N, 3]`` in nanometres
                and ``local_latents`` ``[B, N, latent_dim]``.
            padding_mask: ``[B, N]`` mask of real tokens.
            token_type: Optional token-kind IDs ``[B, N]``. The unified
                33-class decoder constrains sequence classes by token kind.
            can_be_encoded: Optional ``[B, N]`` mask selecting tokens decoded
                by the protein autoencoder. Other valid tokens retain their
                product-space ``bb_ca`` in atom37 slot 1 and residue type -1.
                Atomized tokens and legacy-decoder DNA/RNA are always excluded.
            conditional_residue_type: Optional conditioned residue classes
                ``[B, N]`` fed to the decoder. With ``conditional_residue_mask``,
                a decoder with an atom-presence head also outputs them in place
                of its own prediction.
            conditional_residue_mask: Which tokens carry a residue condition.
            required_atom_mask: Optional atoms ``[B, N, 37]`` the caller
                conditions on. A decoder with an atom-presence head keeps those
                its residue can have, whatever the head predicts.

        Returns:
            Dict with ``coords`` ``[B, N, 37, 3]`` in nanometres,
            ``sequence_logits`` ``[B, N, num_residue_types]``, the argmax
            ``residue_type`` ``[B, N]``, and the ``atom_mask`` ``[B, N, 37]``.
        """
        padding_mask = padding_mask.to(device=state["bb_ca"].device, dtype=torch.bool)
        if can_be_encoded is None:
            can_be_encoded = padding_mask
        else:
            can_be_encoded = _normalize_token_mask("can_be_encoded", can_be_encoded, padding_mask)
        if token_type is not None:
            if token_type.shape != padding_mask.shape:
                raise ValueError(
                    f"token_type must have shape {tuple(padding_mask.shape)}, got {tuple(token_type.shape)}"
                )
            token_type = token_type.to(device=padding_mask.device)
            # Narrow out of place: a bool mask is its own ``.to(torch.bool)``, so
            # ``&=`` here would write through to the caller's tensor.
            can_be_encoded = can_be_encoded & (token_type != ATOMIZED_TOKEN_TYPE)
            if self.decoder.num_residue_types == 20:
                can_be_encoded = can_be_encoded & (token_type != RNA_TOKEN_TYPE) & (token_type != DNA_TOKEN_TYPE)
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
        if required_atom_mask is not None:
            if required_atom_mask.shape != (*padding_mask.shape, ATOM37_SIZE):
                raise ValueError(
                    f"required_atom_mask must have shape {(*padding_mask.shape, ATOM37_SIZE)}, "
                    f"got {tuple(required_atom_mask.shape)}"
                )
            required_atom_mask = required_atom_mask.to(device=padding_mask.device, dtype=torch.bool)

        if torch.equal(can_be_encoded, padding_mask):
            decoded = self.decoder(
                backbone_ca=state["bb_ca"],
                local_latents=state["local_latents"],
                mask=padding_mask,
                token_type=token_type,
                conditional_residue_type=conditional_residue_type,
                conditional_residue_mask=conditional_residue_mask,
                required_atom_mask=required_atom_mask,
            )
            if "atom_mask" not in decoded:
                decoded["atom_mask"] = get_residue_atom_mask(
                    decoded["residue_type"],
                    self.decoder.num_residue_types,
                    padding_mask,
                )
            return decoded

        batch, tokens = padding_mask.shape
        coordinates = torch.zeros(batch, tokens, ATOM37_SIZE, 3, device=padding_mask.device, dtype=self.decoder.dtype)
        sequence_logits = torch.zeros(
            batch, tokens, self.decoder.num_residue_types, device=padding_mask.device, dtype=self.decoder.dtype
        )
        residue_types = torch.full((batch, tokens), -1, device=padding_mask.device, dtype=torch.long)
        atom_mask = torch.zeros(batch, tokens, ATOM37_SIZE, device=padding_mask.device, dtype=torch.bool)
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
                    required_atom_mask=(
                        None if required_atom_mask is None else required_atom_mask[batch_index, encoded].unsqueeze(0)
                    ),
                )
                coordinates[batch_index, encoded] = decoded["coords"][0]
                sequence_logits[batch_index, encoded] = decoded["sequence_logits"][0]
                residue_types[batch_index, encoded] = decoded["residue_type"][0]
                decoded_atom_mask = decoded.get("atom_mask")
                if decoded_atom_mask is None:
                    decoded_atom_mask = get_residue_atom_mask(
                        decoded["residue_type"],
                        self.decoder.num_residue_types,
                        torch.ones_like(decoded["residue_type"], dtype=torch.bool),
                    )
                atom_mask[batch_index, encoded] = decoded_atom_mask[0]

            no_latent = padding_mask[batch_index] & ~encoded
            coordinates[batch_index, no_latent, 1] = state["bb_ca"][batch_index, no_latent].to(coordinates.dtype)
            atom_mask[batch_index, no_latent, 1] = True

        return {
            "coords": coordinates,
            "sequence_logits": sequence_logits,
            "residue_type": residue_types,
            "atom_mask": atom_mask,
        }
