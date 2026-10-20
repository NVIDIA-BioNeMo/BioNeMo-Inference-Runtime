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

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig
from bionemo_ir._torch.graph_optimization.decorator import NamedDimTies
from bionemo_ir._torch.graph_optimization.graph_policy import exact_graph_config
from bionemo_ir._torch.modules.rfproteina.groups import GROUP_KEYS
from bionemo_ir._torch.search import WeightedObjective, build_weighted_objective
from bionemo_ir.configs import BaseConfig, TrunkPadSpec
from bionemo_ir.hubs.support_matrix import GenerativeSupportMatrix as SupMat

#: Tokens the captured trunk step accepts; a longer request runs the trunk eagerly.
TRUNK_GRAPH_MAX_TOKENS = 1024


def trunk_graph_config(max_tokens: int = TRUNK_GRAPH_MAX_TOKENS) -> CUDAGraphOptimizationConfig:
    """Exact policy for one trunk recycle step.

    The step is what repeats, within a denoising step and across them, so a shape captures once
    and recycle depth never reaches the cache key. Exact keying with no padded dims: padding the
    token axes here would undo the trunk's own 8-alignment and its fused TriMul residual.
    """
    return exact_graph_config(
        named_dims=(
            NamedDimTies(
                name="num_tokens",
                input_dims=(
                    ("single", (-2,)),
                    ("pair", (-2, -3)),
                    ("condition", (-2,)),
                    ("padding_mask", (-1,)),
                ),
            ),
        ),
        max_tokens=max_tokens,
        repeated=True,
        static_args=("metadata",),
        workspace_kwargs=("buffers",),
        # A denoising step resolves these once and every recycle of that step is handed the same
        # tensors, so only the running state and the pass's condition are copied in after the
        # first. The runtime still compares each against what it last copied.
        stable_kwargs=(
            "single_init",
            "pair_init",
            "padding_mask",
            "trimul_metadata",
            "precomputed_masks",
            "precomputed_single_masks",
        ),
    )


class RFProteinaScheduleConfig(BaseModel):
    """Time-grid configuration for one product-space modality."""

    mode: str
    power: float = 2.0
    #: Sample the final ``tail_steps`` intervals at ``tail_step_multiplier``
    #: times the density, holding the total step count fixed. Zero keeps the
    #: plain grid.
    tail_steps: int = Field(default=0, ge=0)
    tail_step_multiplier: int = 1

    @model_validator(mode="after")
    def _validate_tail(self) -> "RFProteinaScheduleConfig":
        if self.tail_steps and self.tail_step_multiplier < 2:
            raise ValueError("tail_step_multiplier must be an integer >= 2 when tail_steps is set")
        return self


class RFProteinaNoiseScheduleConfig(BaseModel):
    """Stochastic forcing schedule for one modality."""

    mode: str
    power: float = 1.0
    clamp: float | None = None


class RFProteinaVFStepConfig(BaseModel):
    """Velocity-flow integration parameters."""

    sampling_mode: str = "vf_tsr"
    score_scale: float = 1.0
    noise_scale: float = 0.1
    ode_upper: float = 0.98
    ode_lower: float = 0.02
    center_every_step: bool = False
    tsr_k: float = 4.0
    tsr_sigma: float = 8.0
    #: Scale the injected noise to the score the TSR ratio already applied,
    #: ``score_scale / ratio``, instead of the configured ``noise_scale``.
    tsr_matched_noise: bool = False
    #: Drive the Langevin term with the TSR-scaled score (``score * ratio``) and
    #: inject noise of variance ``2 * g(t) * noise_scale * dt``; ``score_scale``
    #: is then unused. This is the stochastic VF-TSR update RFD4-Proteina uses,
    #: and it divides the noise schedule ``g(t)`` by the TSR ratio squared
    #: where the other noise laws divide it by the ratio once.
    tsr_scaled_score_drift: bool = False

    @model_validator(mode="after")
    def _validate_noise_laws(self) -> "RFProteinaVFStepConfig":
        if self.tsr_scaled_score_drift and self.tsr_matched_noise:
            raise ValueError("tsr_scaled_score_drift and tsr_matched_noise are mutually exclusive")
        return self


class RFProteinaReferenceConfig(BaseModel):
    """Reference (``t = 0``) distribution for one product-space modality.

    Mirrors one entry of upstream's ``product_flowmatcher`` config. No
    dimensionality field: the backbone is 3-vector and the latent width comes
    from the checkpoint, so repeating it could disagree with the weights.
    """

    #: Project the draw onto the zero-centre-of-mass subspace.
    zero_com_noise: bool = False
    #: Only ``x_0`` perturbs the reference draw; the rest act on training-time
    #: interpolation and are inference no-ops, accepted so configs load as-is.
    stochastic_centering_mode: Literal["none", "x_0", "x_1", "x_t"] = "none"
    stochastic_centering_scale: float | None = None
    stochastic_centering_scope: Literal["all", "without_absolute_coords", "per_group"] = "without_absolute_coords"
    #: Group tensors whose unique combinations each get an independent offset.
    #: Required by, and only read for, the ``per_group`` scope.
    stochastic_centering_group_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_scope(self) -> "RFProteinaReferenceConfig":
        if self.stochastic_centering_scope == "per_group" and not self.stochastic_centering_group_keys:
            raise ValueError("stochastic_centering_scope='per_group' needs a non-empty stochastic_centering_group_keys")
        invalid = set(self.stochastic_centering_group_keys) - GROUP_KEYS
        if invalid:
            raise ValueError(
                f"unknown stochastic_centering_group_keys {sorted(invalid)}; expected a subset of {sorted(GROUP_KEYS)}"
            )
        return self


class RFProteinaModalitySamplingConfig(BaseModel):
    """Reference distribution, schedules, and integrator for one modality."""

    schedule: RFProteinaScheduleConfig
    noise_schedule: RFProteinaNoiseScheduleConfig
    step: RFProteinaVFStepConfig
    reference: RFProteinaReferenceConfig = Field(default_factory=RFProteinaReferenceConfig)


def _backbone_sampling_config() -> RFProteinaModalitySamplingConfig:
    return RFProteinaModalitySamplingConfig(
        schedule=RFProteinaScheduleConfig(mode="log", power=2.0),
        noise_schedule=RFProteinaNoiseScheduleConfig(mode="1/t", power=1.0),
        step=RFProteinaVFStepConfig(noise_scale=0.3, tsr_k=16.0, tsr_sigma=3.0),
        reference=RFProteinaReferenceConfig(
            zero_com_noise=True, stochastic_centering_mode="x_0", stochastic_centering_scale=0.2
        ),
    )


def _latent_sampling_config() -> RFProteinaModalitySamplingConfig:
    return RFProteinaModalitySamplingConfig(
        schedule=RFProteinaScheduleConfig(mode="power", power=2.0),
        noise_schedule=RFProteinaNoiseScheduleConfig(mode="tan", power=1.0),
        step=RFProteinaVFStepConfig(noise_scale=0.1, tsr_k=4.0, tsr_sigma=8.0),
        reference=RFProteinaReferenceConfig(),
    )


#: The released recycle strategies, as ``(steps, recycle depth)`` bands over a 60-step
#: trajectory, which ``recycle_strategy`` selects upstream. Each peaks where the one above
#: does, over the ten steps that decide the fold, and they differ in how much thinking they
#: spend there: 110, 185 and 295 trunk passes against the 170 of the 50-step default.
RECYCLE_SCHEDULES: dict[str, list[tuple[int, int]]] = {
    "thinking_low": [(5, 1), (5, 1), (5, 2), (5, 3), (10, 3), (5, 3), (5, 2), (10, 1), (10, 1)],
    "thinking_medium": [(5, 1), (5, 2), (5, 3), (5, 5), (10, 6), (5, 5), (5, 3), (10, 2), (10, 1)],
    "thinking_high": [(5, 2), (5, 3), (5, 5), (5, 8), (10, 9), (5, 8), (5, 5), (10, 3), (10, 2)],
}


def recycle_schedule(name: str) -> list[tuple[int, int]]:
    """Return a released recycle schedule's bands by name.

    Args:
        name: One of :data:`RECYCLE_SCHEDULES`, as ``recycle_strategy`` selects upstream.

    Returns:
        A copy of the bands, safe for the caller to edit.

    Raises:
        KeyError: The name is not a released schedule.
    """
    if name not in RECYCLE_SCHEDULES:
        raise KeyError(f"unknown recycle schedule {name!r}; expected one of {sorted(RECYCLE_SCHEDULES)}")
    return list(RECYCLE_SCHEDULES[name])


class RFProteinaSamplingConfig(BaseModel):
    """RFProteína product-space sampling and inference-recycling policy."""

    num_steps: int = 50
    self_condition: bool = False
    #: Decode against the trunk readout's predicted sequence on tokens the
    #: caller left unspecified. Requires ``trunk.use_sequence_readout``.
    condition_decoder_on_sequence_prediction: bool = False
    n_recycle: int = Field(default=0, ge=0)
    n_recycle_schedule: list[tuple[int, int]] | None = None
    state_dtype: str = "float32"
    #: Consume random draws in the reference implementation's order, so a fixed
    #: seed reproduces its noise and two backends' designs compare one to one.
    #: The reference encodes the input before it draws the reference noise --
    #: even for unconditional design, where the encoding goes unused -- and the
    #: encoder's posterior sample takes one draw from the same stream; this mode
    #: takes that draw too and routes the encoder's own sample through the
    #: sampling stream. Off by default: production draws with plain
    #: ``torch.randn`` and skips a draw nothing reads.
    match_reference_rng: bool = False
    backbone: RFProteinaModalitySamplingConfig = Field(default_factory=_backbone_sampling_config)
    local_latents: RFProteinaModalitySamplingConfig = Field(default_factory=_latent_sampling_config)

    @model_validator(mode="after")
    def _validate_recycle_schedule(self) -> "RFProteinaSamplingConfig":
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.n_recycle_schedule is None:
            return self
        if any(width <= 0 or depth < 0 for width, depth in self.n_recycle_schedule):
            raise ValueError("n_recycle_schedule entries require a positive width and non-negative depth")
        scheduled_steps = sum(width for width, _ in self.n_recycle_schedule)
        if scheduled_steps != self.num_steps:
            raise ValueError(
                f"n_recycle_schedule expands to {scheduled_steps} steps, but num_steps is {self.num_steps}"
            )
        return self

    def recycle_depths(self, num_steps: int | None = None) -> tuple[int, ...]:
        """Expand the full schedule, taking its tail for a partial trajectory."""
        actual_steps = self.num_steps if num_steps is None else num_steps
        if not 0 < actual_steps <= self.num_steps:
            raise ValueError(f"num_steps must be in [1, {self.num_steps}], got {actual_steps}")
        if self.n_recycle_schedule is None:
            return (self.n_recycle,) * actual_steps
        depths = tuple(depth for width, depth in self.n_recycle_schedule for _ in range(width))
        if len(depths) != self.num_steps:
            raise ValueError(f"n_recycle_schedule expands to {len(depths)} steps, but num_steps is {self.num_steps}")
        return depths[-actual_steps:]


#: The released sampler densifies the last ten intervals of each modality.
RELEASED_TAIL_STEPS, RELEASED_TAIL_STEP_MULTIPLIER = 10, 2


def _latest_sampling_config() -> RFProteinaSamplingConfig:
    """The released inference policy: ``thinking_medium`` over the densified tail.

    This is what ``rfd4-generate`` runs by default, down to the tail grid and the decoder
    conditioning, so a design taken straight from this config is the served one.
    """
    bands = recycle_schedule("thinking_medium")

    def modality(build: "Callable[[], RFProteinaModalitySamplingConfig]") -> RFProteinaModalitySamplingConfig:
        config = build()
        config.schedule.tail_steps = RELEASED_TAIL_STEPS
        config.schedule.tail_step_multiplier = RELEASED_TAIL_STEP_MULTIPLIER
        return config

    return RFProteinaSamplingConfig(
        num_steps=sum(width for width, _ in bands),
        n_recycle_schedule=bands,
        condition_decoder_on_sequence_prediction=True,
        backbone=modality(_backbone_sampling_config),
        local_latents=modality(_latent_sampling_config),
    )


#: Upstream inverse-folding method: resample the target backbone from its
#: forward marginal every step, then restore the clean backbone at the end.
RESAMPLED_FORWARD_MARGINAL = "resampled_forward_marginal"
INVERSE_FOLDING_METHODS = frozenset({RESAMPLED_FORWARD_MARGINAL})


class RFProteinaPartialSimulationConfig(BaseModel):
    """Runtime start-time contract for existing-structure redesign.

    Explicit modality times run the configured full step count over shorter
    schedule tails. Trajectory-fraction mode restarts on the configured
    schedule itself, at the index that leaves ``round(fraction * num_steps)``
    steps, and integrates that tail. Inverse folding is the special case that
    keeps the backbone and redesigns only sequence latents.
    """

    partial_t_backbone: float | None = None
    partial_t_latent: float | None = None
    couple_partial_t: bool = False
    partial_trajectory_fraction_steps: float | None = None
    inverse_folding: bool = False
    inverse_folding_method: str = RESAMPLED_FORWARD_MARGINAL
    #: Resolution of the dense schedule that explicit and coupled start times are read from.
    reference_num_steps: int = 10000

    @model_validator(mode="after")
    def _validate_partial_times(self) -> "RFProteinaPartialSimulationConfig":
        backbone = self.partial_t_backbone
        latent = self.partial_t_latent
        fraction = self.partial_trajectory_fraction_steps
        if self.reference_num_steps <= 0:
            raise ValueError("reference_num_steps must be positive")
        if self.inverse_folding:
            if backbone is not None or latent is not None or self.couple_partial_t or fraction is not None:
                raise ValueError(
                    "inverse_folding is mutually exclusive with partial_t_backbone, "
                    "partial_t_latent, couple_partial_t, and "
                    "partial_trajectory_fraction_steps"
                )
            if self.inverse_folding_method not in INVERSE_FOLDING_METHODS:
                raise ValueError(
                    "inverse_folding_method must be one of "
                    f"{sorted(INVERSE_FOLDING_METHODS)}, got {self.inverse_folding_method!r}"
                )
            return self
        if fraction is not None:
            if backbone is not None or latent is not None or self.couple_partial_t:
                raise ValueError(
                    "partial_trajectory_fraction_steps is mutually exclusive "
                    "with partial_t_backbone, partial_t_latent, and "
                    "couple_partial_t"
                )
            if not 0.0 < fraction <= 1.0:
                raise ValueError("partial_trajectory_fraction_steps must be in (0, 1]")
            return self

        for name, value in (("partial_t_backbone", backbone), ("partial_t_latent", latent)):
            if value is not None and not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        count = int(backbone is not None) + int(latent is not None)
        if count == 2 and self.couple_partial_t:
            raise ValueError("couple_partial_t=true requires exactly one explicit modality time")
        if count == 1 and not self.couple_partial_t:
            raise ValueError("one explicit modality time requires couple_partial_t=true; otherwise provide both times")
        return self


class RFProteinaMetricClipConfig(BaseModel):
    """Bounds applied to a metric before its search weight."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, description="Lower clip applied before weighting.")
    max: float | None = Field(default=None, description="Upper clip applied before weighting.")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RFProteinaMetricClipConfig":
        if self.min is None and self.max is None:
            raise ValueError("clip_values entries must set min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"clip has min={self.min} > max={self.max}")
        return self


class RFProteinaSearchRewardOverrideConfig(BaseModel):
    """Per-checkpoint reward fields; each named field replaces the base wholesale."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] | None = Field(
        default=None,
        description="Replacement metric weights at this checkpoint. Negative weights minimize.",
    )
    clip_values: dict[str, RFProteinaMetricClipConfig] | None = Field(
        default=None,
        description="Replacement clips at this checkpoint, keyed like weights.",
    )
    reward_metrics_tags: list[str] | None = Field(
        default=None,
        description="Replacement tags for an injected metric manager. Local metrics ignore tags.",
    )


class RFProteinaSearchRewardConfig(BaseModel):
    """Weighted local or injected metrics used to rank beam candidates."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(
        description=(
            "Maps a metric key such as clash_metrics.frac_atoms_clashing_1p5A to a weight. Negative minimizes."
        )
    )
    clip_values: dict[str, RFProteinaMetricClipConfig] = Field(
        default_factory=dict,
        description="Optional per-key min/max applied before weighting. The reported metric stays raw.",
    )
    reward_metrics_tags: list[str] = Field(
        default_factory=list,
        description="Tags forwarded to an injected OSS metric manager. The local evaluator ignores them.",
    )
    checkpoint_overrides: dict[int, RFProteinaSearchRewardOverrideConfig] = Field(
        default_factory=dict,
        description="Maps a scored checkpoint step to replacement weights, clips, and/or tags.",
    )

    @field_validator("checkpoint_overrides", mode="before")
    @classmethod
    def _coerce_checkpoint_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        coerced: dict[int, object] = {}
        for key, spec in value.items():
            if isinstance(key, bool) or not (isinstance(key, int) or (isinstance(key, str) and key.isdigit())):
                raise ValueError(f"checkpoint_overrides keys must be integer steps; got {key!r}")
            coerced[int(key)] = spec
        return coerced

    def to_weighted_objective(self) -> WeightedObjective:
        """Build the runtime objective, resolving per-checkpoint replacements."""
        overrides: dict[int, dict[str, object]] = {}
        for step, spec in self.checkpoint_overrides.items():
            replacement: dict[str, object] = {}
            if spec.weights is not None:
                replacement["weights"] = spec.weights
            if spec.clip_values is not None:
                replacement["clip_values"] = {key: value.model_dump() for key, value in spec.clip_values.items()}
            if spec.reward_metrics_tags is not None:
                replacement["reward_metrics_tags"] = spec.reward_metrics_tags
            overrides[step] = replacement
        return build_weighted_objective(
            weights=self.weights,
            clip_values={key: value.model_dump() for key, value in self.clip_values.items()},
            reward_metrics_tags=self.reward_metrics_tags,
            checkpoint_overrides=overrides,
        )


class RFProteinaBeamSearchConfig(BaseModel):
    """Opt-in metric-weighted beam search over RFProteína product-space rollouts."""

    model_config = ConfigDict(extra="forbid")

    step_checkpoints: list[int] = Field(
        description=(
            "Inclusive global steps to branch, score, and prune. Must start at 0, increase, and end at num_steps."
        )
    )
    n_branch: int = Field(default=4, ge=1, description="Stochastic continuations spawned from each survivor.")
    beam_width: int = Field(default=4, ge=1, description="Candidates kept per input after each prune.")
    scoring_chunk_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum decoded rows per scoring chunk. None scores the full candidate tensor.",
    )
    intermediate_save_min_reward: float | None = Field(
        default=None,
        description="Keep non-survivor completions whose search.reward is at least this value. None keeps none.",
    )
    rewards: RFProteinaSearchRewardConfig = Field(description="Weighted objective used at each scored checkpoint.")

    @model_validator(mode="after")
    def _validate_checkpoints(self) -> "RFProteinaBeamSearchConfig":
        checkpoints = self.step_checkpoints
        if len(checkpoints) < 2:
            raise ValueError("step_checkpoints must have at least 2 entries (start and end)")
        if checkpoints[0] != 0:
            raise ValueError("step_checkpoints must begin at 0")
        if any(later <= earlier for earlier, later in zip(checkpoints, checkpoints[1:], strict=False)):
            raise ValueError("step_checkpoints must be strictly increasing")
        return self


class RFProteinaTrunkConfig(BaseConfig):
    """Configuration for the released local-latents velocity network.

    Defaults are the September 2026 z12 checkpoint's
    (``2026_09_04_release_mgnify_z12_600k_140k-ema``). Earlier ones differ,
    notably in latent and triangle widths, so build theirs with
    :func:`~...convert.rfproteina_trunk_config_from_checkpoint`; a mismatch on
    load raises and names the field.
    """

    token_dim: int = 768
    pair_dim: int = 384
    condition_dim: int = 256
    latent_dim: int = 12
    num_layers: int = 14
    num_heads: int = 12
    condition_transition_expansion: int = 2
    transition_expansion: int = 4
    pair_update_every: int = 2
    trimul_hidden_dim: int = 256
    # The September z12 release learns triangle projection and LayerNorm biases.
    # The October 2026 340k checkpoint does not; its converter sets this false.
    trimul_bias: bool = True
    pair_transition_expansion: int = 2
    pair_write: Literal["sum", "hadamard"] = "hadamard"
    recycle_enabled: bool = True
    use_qk_norm: bool = True
    use_insertion_head: bool = True
    # Per-token sequence-class probe on the trunk's final single track. The
    # October 2026 340k checkpoint carries one; earlier releases do not, so it
    # stays off unless the converter finds its weights.
    use_sequence_readout: bool = False
    sequence_readout_classes: int = 33
    pair_mask_left_aligned: bool = True
    # Pad the token dims to a multiple of 8 once before the recycling loop
    # (bionemo_ir._torch.layers.token_padding). TriangleMultiplicationNode's
    # fused residual epilogue (triangle_nodes.can_fuse_residual) requires
    # 8-aligned pair dims; otherwise every pair update falls back to an
    # unfused residual add.
    enable_token_pad: bool = True
    token_pad_spec: TrunkPadSpec = TrunkPadSpec(
        single_channel=("single", "condition"),
        pair_channel=("pair",),
        single_last=("padding_mask",),
    )
    # Use cached cuDNN operation graphs for validated BF16 pair operations.
    # Unsupported devices, dtypes, batch sizes, and active chunking retain the
    # vanilla Torch path.
    enable_cudnn_graph: bool = True
    pairwise_attention_backend: str = "SDPA"
    mask_inf: float = 1e4
    trimul_high_precision: bool = False
    dtype: str = "bfloat16"

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "RFProteinaTrunkConfig":
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.pair_update_every <= 0:
            raise ValueError("pair_update_every must be positive")
        return self


class RFProteinaRepresentationConfig(BaseConfig):
    """Tensor-only feature construction for the released velocity network.

    Covers the learned embeddings/projections that turn pipeline annotations
    plus the current state into single, pair, and condition tracks. Producing
    the annotations stays with the data pipeline.
    """

    input_schema: Literal["legacy_self_conditioned", "latest", "sequence_probability"] = "latest"
    latent_dim: int = 12
    time_embedding_dim: int = 256
    token_type_embedding_dim: int = 32
    #: Token kinds the embedding table covers. The unified release adds two to the
    #: five the earlier networks use; the count comes from the checkpoint.
    num_token_kinds: int = 5
    num_conditional_residue_types: int = 33
    cath_embedding_dim: int = 32
    secondary_structure_embedding_dim: int = 16
    charge_embedding_dim: int = 16
    num_charge_classes: int = 11
    charge_offset: int = 4
    element_embedding_dim: int = 16
    max_atomic_number: int = 128
    chirality_dim: int = 18
    coordinate_group_dim: int = 37
    # Standard polymer tokens in unified-encoding order, excluding unknown, mask,
    # atomized, and gap. Each contributes a probability and an explicit-mask channel.
    num_sequence_probability_channels: int = 28
    atomized_token_type: int = 3
    pair_distance_bins: int = 30
    pair_distance_min: float = 0.1
    pair_distance_max: float = 3.0
    token_bond_path_bins: int = 64
    num_bond_orders: int = 5
    # FP32 except the terminal pair AdaLN; the outputs are then cast to the
    # velocity-transformer dtype.
    pair_adaln_dtype: str = "float32"
    # The 340k score network uses bias-free RMSNorm for its feature-factory
    # output norms. Earlier releases use LayerNorm, including the bias.
    factory_norm: Literal["layer_norm", "rms_norm"] = "layer_norm"
    #: Recycle feature the condition track carries, named as a checkpoint's
    #: ``nn.feats_cond_seq`` names it. Unset takes the feature the input schema
    #: implies: the sequence-probability network conditions on the pass index,
    #: and earlier schemas carry no recycle feature.
    recycle_conditioning: Literal["none", "recycle_index", "recycle_remaining"] | None = None
    #: Width of the ``recycle_remaining`` one-hot, a checkpoint's
    #: ``nn.recycle.conditioning_dim``. Remaining passes saturate at the last slot.
    recycle_remaining_dim: int = 2
    #: Rotate each stereocentre's chirality vectors onto its current noisy neighbours before the
    #: single projection, as a checkpoint's ``nn.noise_align_chirality`` does. Nothing in the weights
    #: records it, so the host that holds the training configuration sets it.
    noise_align_chirality: bool = False
    #: Also move them to the frame's noisy position, a checkpoint's ``nn.noise_align_chirality_translate``.
    noise_align_chirality_translate: bool = False
    dtype: str = "float32"

    @property
    def single_input_dim(self) -> int:
        if self.input_schema == "sequence_probability":
            return (
                3
                + self.latent_dim
                + self.token_type_embedding_dim
                + 2 * self.num_sequence_probability_channels
                + 2 * 37 * 4
                + 37
                + self.secondary_structure_embedding_dim
                + 2 * 37
                + 37
                + 2 * 37
                + 37
                + 37
                + 37
                + 2
                + 37
                + self.charge_embedding_dim
                + self.element_embedding_dim
                + self.coordinate_group_dim
                + self.chirality_dim
                + 3
            )
        if self.input_schema == "latest":
            return (
                3
                + self.latent_dim
                + self.token_type_embedding_dim
                + self.num_conditional_residue_types
                + 2 * 37 * 4
                + 37
                + self.secondary_structure_embedding_dim
                + 2 * 37
                + 37
                + 2 * 37
                + 2
                + 37
                + 1
                + self.charge_embedding_dim
                + self.element_embedding_dim
                + self.coordinate_group_dim
                + self.chirality_dim
                + 3
            )
        return (
            3
            + self.latent_dim
            + 3
            + self.latent_dim
            + self.token_type_embedding_dim
            + self.num_conditional_residue_types
            + 2 * 37 * 4
            + 37
            + self.secondary_structure_embedding_dim
            + 37
            + 37
            + 1
            + self.charge_embedding_dim
            + self.element_embedding_dim
            + self.coordinate_group_dim
            + self.chirality_dim
            + 3
        )

    @property
    def recycle_feature(self) -> str | None:
        """Recycle feature in the condition track, or ``None`` when it carries none."""
        if self.recycle_conditioning is not None:
            return None if self.recycle_conditioning == "none" else self.recycle_conditioning
        return "recycle_index" if self.input_schema == "sequence_probability" else None

    @property
    def condition_base_dim(self) -> int:
        """Condition width before the recycle feature: both times and the CATH code."""
        return 2 * self.time_embedding_dim + 3 * self.cath_embedding_dim

    @property
    def condition_input_dim(self) -> int:
        feature = self.recycle_feature
        if feature == "recycle_index":
            return self.condition_base_dim + self.time_embedding_dim
        if feature == "recycle_remaining":
            return self.condition_base_dim + self.recycle_remaining_dim
        return self.condition_base_dim

    @property
    def supports_nucleic_acid_secondary_structure(self) -> bool:
        return self.input_schema != "legacy_self_conditioned"

    @property
    def pair_input_dim(self) -> int:
        if self.input_schema != "legacy_self_conditioned":
            # xt distance, bond/aromatic flags, index/molecule/entity equality,
            # bond-path one-hot, canonical ordering, and NA pair value + mask.
            return (
                self.pair_distance_bins + 1 + self.num_bond_orders + 1 + 1 + 1 + 1 + self.token_bond_path_bins + 1 + 2
            )
        return 2 * self.pair_distance_bins + 1 + self.num_bond_orders + 1 + 1 + 1 + self.token_bond_path_bins + 1

    @property
    def pair_condition_input_dim(self) -> int:
        return 2 * self.time_embedding_dim

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "RFProteinaRepresentationConfig":
        if self.time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be at least 2")
        if self.pair_distance_bins < 2:
            raise ValueError("pair_distance_bins must be at least 2")
        if self.num_conditional_residue_types <= 0:
            raise ValueError("num_conditional_residue_types must be positive")
        if self.num_sequence_probability_channels <= 0:
            raise ValueError("num_sequence_probability_channels must be positive")
        if self.recycle_remaining_dim < 1:
            raise ValueError("recycle_remaining_dim must be positive")
        if self.num_token_kinds < 1:
            raise ValueError("num_token_kinds must be positive")
        return self


class RFProteinaEncoderConfig(BaseConfig):
    """Configuration for the released partial-autoencoder encoder.

    The default unified release encodes protein and nucleic-acid polymers into
    12-dimensional local latents without pair-biased attention. Set
    ``input_schema="legacy"`` for the older protein-only autoencoder.
    """

    input_schema: Literal["legacy", "unified"] = "unified"
    token_dim: int = 768
    pair_dim: int = 256
    condition_dim: int = 128
    latent_dim: int = 12
    num_layers: int = 12
    num_heads: int = 12
    condition_transition_expansion: int = 2
    transition_expansion: int = 4
    num_residue_types: int = 33
    token_type_embedding_dim: int = 32
    #: Token kinds the embedding table covers. The unified release adds two to
    #: the five the earlier encoders use; the count comes from the checkpoint.
    num_token_kinds: int = 5
    #: Read per-atom chemistry beside the coordinates: how many bonds of each
    #: type reach every atom37 slot, and what element and formal charge fills
    #: it. Released encoders before the unified one carry neither.
    atom_chemistry: bool = False
    #: Biotite bond types counted per atom37 slot.
    incident_bond_type_bins: int = 10
    num_elements: int = 119
    element_embedding_dim: int = 8
    #: Formal charges -4..+6 map to classes 0..10, saturating at both ends.
    num_atom_charges: int = 11
    atom_charge_offset: int = 4
    atom_charge_embedding_dim: int = 16
    sequence_separation_bins: int = 64
    distance_bins: int = 21
    orientation_bins: int = 21
    use_pair_bias: bool = False
    use_qk_norm: bool = True
    pairwise_attention_backend: str = "SDPA"
    mask_inf: float = 1e4
    #: Project the static pair representation for all transformer layers with
    #: one mega-GEMM. This is faster for long inputs but keeps all
    #: ``num_layers * num_heads * N^2`` biases live through the layer stack.
    precompute_bias: bool = False
    transformer_dtype: str = "float32"
    dtype: str = "float32"

    @property
    def single_input_dim(self) -> int:
        if self.input_schema == "unified":
            # token kind + absolute/CA-relative atom37 coordinates + chain id.
            width = self.token_type_embedding_dim + 2 * (37 * 4) + 1
            if self.atom_chemistry:
                # Incident bond counts, then element and charge, per atom37 slot.
                width += 37 * self.incident_bond_type_bins
                width += 37 * (self.element_embedding_dim + self.atom_charge_embedding_dim)
            return width
        # chain break + residue one-hot + absolute/relative atom37 +
        # three backbone torsions + four side-chain torsions and their masks.
        return 1 + self.num_residue_types + 2 * (37 * 4) + 3 * 21 + (4 * 21 + 4)

    @property
    def chemistry_input_dim(self) -> int:
        """Width of the element and charge columns the released projection reads, per request.

        A converted encoder never projects them. Conversion folds each slot's columns into lookup
        tables that land in ``token_dim`` directly, and the projection keeps :attr:`dense_input_dim`.
        """
        if self.input_schema != "unified" or not self.atom_chemistry:
            return 0
        return 37 * (self.element_embedding_dim + self.atom_charge_embedding_dim)

    @property
    def dense_input_dim(self) -> int:
        """Width of the features a converted encoder projects: :attr:`single_input_dim` less the folded columns."""
        return self.single_input_dim - self.chemistry_input_dim

    @property
    def pair_input_dim(self) -> int:
        if not self.use_pair_bias:
            return 0
        # sequence separation + four backbone distances + five orientations.
        return self.sequence_separation_bins + 4 * self.distance_bins + 5 * self.orientation_bins

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "RFProteinaEncoderConfig":
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.input_schema == "legacy" and self.sequence_separation_bins % 2 == 0:
            raise ValueError("sequence_separation_bins must be odd")
        if self.num_residue_types < 1:
            raise ValueError("num_residue_types must be positive")
        # The unified encoder embeds the token kind and never one-hots the residue
        # class, so its input width does not depend on the vocabulary size and a
        # larger vocabulary needs no change here. The legacy encoder does one-hot
        # it, and ``single_input_dim`` already carries that dependency.
        if self.input_schema == "legacy" and (self.distance_bins != 21 or self.orientation_bins != 21):
            raise ValueError("the released encoder uses 21 distance and orientation bins")
        if not self.use_pair_bias and self.precompute_bias:
            raise ValueError("precompute_bias requires use_pair_bias=True")
        if not self.use_pair_bias and self.pairwise_attention_backend == "CuTeDSL":
            raise ValueError("CuTeDSL pairwise attention requires use_pair_bias=True")
        return self


class RFProteinaDecoderConfig(BaseConfig):
    """Configuration for the released partial-autoencoder decoder.

    Defaults match ``2026_09_04_release_ae_z12_600k-ema.ckpt`` and keep the
    autoencoder feature projections/output heads in float32; optimized
    inference runs only the transformer in bfloat16.
    """

    input_schema: Literal["legacy", "unified"] = "unified"
    token_dim: int = 768
    pair_dim: int = 256
    condition_dim: int = 128
    latent_dim: int = 12
    num_layers: int = 12
    num_heads: int = 12
    condition_transition_expansion: int = 2
    transition_expansion: int = 4
    num_residue_types: int = 33
    token_type_embedding_dim: int = 32
    #: Token kinds the embedding table covers, read from the checkpoint.
    num_token_kinds: int = 5
    #: Intersect the static occupancy table with a learned per-slot presence
    #: head, which the unified release adds so a decoded residue can suppress
    #: atoms its class would otherwise carry. The OSS autoencoder that has the
    #: head also keeps a conditioned residue over the argmax and every
    #: conditioned atom over the head, and so does the decoder.
    use_atom_presence: bool = False
    #: Occupied atom37 slots per residue class, for a vocabulary wider than the
    #: table the decoder carries. Resolved from
    #: :mod:`bionemo_ir.data.schemas.vocabulary` when weights are converted.
    atom37_occupancy: tuple[tuple[int, ...], ...] | None = None
    #: Residue classes each token kind may decode as, for a vocabulary wider
    #: than the table the decoder carries. Resolved with the occupancy.
    allowed_residue_classes: tuple[tuple[int, ...], ...] | None = None
    num_conditional_residue_types: int = 33
    sequence_separation_bins: int = 64
    pair_distance_bins: int = 30
    pair_distance_min: float = 0.1
    pair_distance_max: float = 3.0
    absolute_coordinates: bool = False
    use_pair_bias: bool = False
    use_qk_norm: bool = True
    pairwise_attention_backend: str = "SDPA"
    mask_inf: float = 1e4
    #: Project all layers' pair biases up front with one mega-GEMM, which the
    #: decoder can do because it never updates its pair representation. On an
    #: A100: break-even to ~512 tokens (+21% peak memory), 1.5-1.6x from 1024
    #: up (+30-34%, all ``num_layers * num_heads * N^2`` biases live at once).
    #: Turn off when memory-bound.
    precompute_bias: bool = False
    transformer_dtype: str = "float32"
    dtype: str = "float32"

    @property
    def single_input_dim(self) -> int:
        width = 3 + self.latent_dim
        if self.input_schema == "unified":
            width += self.token_type_embedding_dim + self.num_conditional_residue_types
        return width

    @property
    def pair_input_dim(self) -> int:
        if not self.use_pair_bias:
            return 0
        return self.sequence_separation_bins + self.pair_distance_bins

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "RFProteinaDecoderConfig":
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.input_schema == "legacy" and self.sequence_separation_bins % 2 == 0:
            raise ValueError("sequence_separation_bins must be odd")
        if self.num_residue_types < 1:
            raise ValueError("num_residue_types must be positive")
        if self.num_conditional_residue_types < 1:
            raise ValueError("num_conditional_residue_types must be positive")
        # Vocabulary size is read from the checkpoint rather than enumerated here:
        # the readout width and the conditional embedding carry it, and the atom37
        # occupancy table reports separately when it has no rows for a class.
        if not self.use_pair_bias and self.precompute_bias:
            raise ValueError("precompute_bias requires use_pair_bias=True")
        if not self.use_pair_bias and self.pairwise_attention_backend == "CuTeDSL":
            raise ValueError("CuTeDSL pairwise attention requires use_pair_bias=True")
        return self


class RFProteinaConfig(BaseConfig):
    """Torch-only model and sampler configuration."""

    trunk: RFProteinaTrunkConfig = Field(default_factory=RFProteinaTrunkConfig)
    representation: RFProteinaRepresentationConfig = Field(default_factory=RFProteinaRepresentationConfig)
    # Lazily derived from the AE checkpoint on the first encode call when
    # omitted, keeping unconditional-generation memory unchanged.
    encoder: RFProteinaEncoderConfig | None = None
    decoder: RFProteinaDecoderConfig = Field(default_factory=RFProteinaDecoderConfig)
    sampling: RFProteinaSamplingConfig = Field(default_factory=_latest_sampling_config)
    checkpoint_path: str | None = None
    autoencoder_checkpoint_path: str | None = None
    dtype: str = "bfloat16"

    @model_validator(mode="after")
    def _propagate_trunk_fields(self) -> "RFProteinaConfig":
        """Settle the fields the sub-configs cannot agree on by themselves.

        The trunk runs at the top-level ``dtype``, and owns the latent width
        because that comes from the checkpoint; the representation follows.
        """
        self.trunk.set_dtype(self.dtype)
        self.representation.latent_dim = self.trunk.latent_dim
        return self


PRETRAINED_CONFIG_REGISTRY = {
    SupMat.RFProteina: RFProteinaConfig,
}
