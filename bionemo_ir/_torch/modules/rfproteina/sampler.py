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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import FlowMatchingRolloutPlan, FlowMatchingSchedule, FlowMatchingState
from bionemo_ir._torch.sampling.random import randn
from bionemo_ir._torch.sampling.runner import GenerativeRunner

from .contracts import (
    RFProteinaCorruptionOutput,
    RFProteinaPartialSamplingOutput,
    RFProteinaPredictFn,
    RFProteinaSamplingOutput,
    RFProteinaState,
)
from .flow_matcher import PRODUCT_SPACE_MODALITIES, RFProteinaProductSpaceFlowMatcher
from .rollout import RFProteinaFlowMatchingIntegrator
from .schedules import create_modality_schedule, create_partial_schedules

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaSamplingConfig


def _zero_unencodable_latents(
    state: RFProteinaState,
    padding_mask: torch.Tensor,
    can_be_encoded: torch.Tensor | None,
) -> RFProteinaState:
    """Clear latents on tokens the autoencoder cannot encode.

    The reference distribution itself must not do this -- upstream masks these
    tokens inside its sampling loop, and the integrator does the same on
    every step. Repeating it before the rollout starts keeps atomized tokens
    from reaching the trunk carrying reference noise on the first velocity call.
    """
    if can_be_encoded is None:
        return state
    latents = state["local_latents"]
    latent_mask = padding_mask.to(torch.bool) & can_be_encoded.to(torch.bool)
    return {
        **state,
        "local_latents": latents * latent_mask.unsqueeze(-1).to(latents.dtype),
    }


def sample_reference_state(
    padding_mask: torch.Tensor,
    latent_dim: int,
    dtype: torch.dtype,
    config: RFProteinaSamplingConfig,
    generator: torch.Generator | None = None,
    can_be_encoded: torch.Tensor | None = None,
    coordinate_condition_mask: torch.Tensor | None = None,
    groups: Mapping[str, torch.Tensor | None] | None = None,
) -> RFProteinaState:
    """Draw the product-space state a rollout starts from.

    Standalone entry point for callers that want the reference draw without a
    sampler; :class:`RFProteinaProductSpaceSampler` uses the same two steps.
    """
    state = RFProteinaProductSpaceFlowMatcher(config, latent_dim).sample_noise(
        padding_mask,
        dtype=dtype,
        generator=generator,
        coordinate_condition_mask=coordinate_condition_mask,
        groups=groups,
    )
    return _zero_unencodable_latents(state, padding_mask, can_be_encoded)


def sample_reference_beam(
    padding_mask: torch.Tensor,
    latent_dim: int,
    dtype: torch.dtype,
    config: RFProteinaSamplingConfig,
    beam_width: int,
    generator: torch.Generator | None = None,
    can_be_encoded: torch.Tensor | None = None,
    coordinate_condition_mask: torch.Tensor | None = None,
    groups: Mapping[str, torch.Tensor | None] | None = None,
) -> list[RFProteinaState]:
    """Draw a beam's initial states as the reference does: one draw for every row of every replica.

    The reference stacks the replicas, ``padding_mask`` repeated ``beam_width`` times, and draws all
    ``beam_width * B`` rows at once, so a replica's rows are the stream's draws at its own offset. Drawing the
    replicas one after another would give each of them the rows the stream starts with.

    Args:
        padding_mask: ``[B, N]`` mask of the tokens each replica generates.
        latent_dim: Width of the local latents.
        dtype: Precision of the returned states.
        config: Sampling configuration, which supplies each modality's reference-noise settings.
        beam_width: Number of replicas.
        generator: Random stream for the draw; ``None`` uses the default device stream. A stream installed with
            :func:`~bionemo_ir._torch.sampling.random.normal_draws` takes its place.
        can_be_encoded: ``[B, N]`` mask of the tokens the autoencoder can encode. Latents elsewhere start at zero;
            ``None`` leaves every latent as drawn.
        coordinate_condition_mask: ``[B, N]``, true for tokens pinned to absolute coordinates. Every stochastic
            centering scope except ``all`` leaves these at the origin.
        groups: Token-level group tensors from :func:`~.groups.extract_groups`, read only by the ``per_group``
            stochastic centering scope.

    Returns:
        One ``[B, N, ...]`` state per replica, in replica order.
    """

    def tile(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.repeat(beam_width, *([1] * (value.ndim - 1)))

    state = sample_reference_state(
        tile(padding_mask),
        latent_dim,
        dtype,
        config,
        generator=generator,
        can_be_encoded=tile(can_be_encoded),
        coordinate_condition_mask=tile(coordinate_condition_mask),
        groups=None if groups is None else {name: tile(value) for name, value in groups.items()},
    )
    rows = padding_mask.shape[0]
    return [
        {modality: tensor[replica * rows : (replica + 1) * rows] for modality, tensor in state.items()}
        for replica in range(beam_width)
    ]


@dataclass(frozen=True)
class RFProteinaPartialRolloutSetup:
    """Forward-marginal initialization and schedules for a partial rollout."""

    corruption: RFProteinaCorruptionOutput
    schedules: Mapping[str, FlowMatchingSchedule]
    target_backbone: torch.Tensor | None


class RFProteinaProductSpaceSampler:
    """Model-local product-space sampler for the released VF-TSR path.

    Owns only what is specific to RFProteína -- the per-modality schedules and
    the shape of the velocity-network call -- and delegates the ``t = 0`` draw
    to :class:`~.flow_matcher.RFProteinaProductSpaceFlowMatcher`.
    Iteration, random state, and step ordering come from the shared sampling
    runtime (:class:`~bionemo_ir._torch.sampling.runner.GenerativeRunner`
    driving a
    :class:`~.rollout.RFProteinaFlowMatchingIntegrator`).
    """

    def __init__(self, config: RFProteinaSamplingConfig, latent_dim: int) -> None:
        self.config = config
        self.flow_matcher = RFProteinaProductSpaceFlowMatcher(config, latent_dim)

    @property
    def group_keys(self) -> frozenset[str]:
        """Group tensors this configuration needs, empty when none do."""
        return self.flow_matcher.group_keys

    def _schedules(self, device: torch.device) -> dict[str, FlowMatchingSchedule]:
        return {
            "bb_ca": create_modality_schedule(self.config.backbone, self.config.num_steps, device),
            "local_latents": create_modality_schedule(self.config.local_latents, self.config.num_steps, device),
        }

    @torch.no_grad()
    def sample(
        self,
        predict: RFProteinaPredictFn,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        initial_state: RFProteinaState | None = None,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        seed: int | None = None,
    ) -> RFProteinaSamplingOutput:
        """Run the model prediction callback for every product-space step.

        Args:
            padding_mask: ``[B, N]`` mask of the tokens the rollout generates. A
                token outside it holds zero state, takes no step, and is left
                out of centering, however many tokens ``predict`` attends to.
                Callers with no conditioning-only tokens pass the real-token mask.
            groups: Token-level group tensors from
                :func:`~.groups.extract_groups`, needed only when a modality
                uses per-group stochastic centering and ``initial_state`` is
                not supplied.
            generator: Explicit random stream, adopted as-is; takes precedence
                over ``seed``.
            seed: Seed for the rollout's private stream. With neither, the
                stream is cloned from the default device generator so capture
                of a CUDA graph does not perturb it.
        """
        padding_mask = padding_mask.to(torch.bool)
        forward = {
            "predict": predict,
            "padding_mask": padding_mask,
            "dtype": dtype,
            "initial_state": initial_state,
            "can_be_encoded": can_be_encoded,
            "coordinate_condition_mask": coordinate_condition_mask,
            "groups": groups,
            "save_trajectory_every": save_trajectory_every,
        }
        if generator is not None:
            return self._sample(**forward, context=SamplingContext(generator=generator, seed=generator.initial_seed()))
        with SamplingContext.graph_safe(padding_mask.device, seed) as context:
            return self._sample(**forward, context=context)

    def _sample(
        self,
        predict: RFProteinaPredictFn,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        initial_state: RFProteinaState | None,
        can_be_encoded: torch.Tensor | None,
        coordinate_condition_mask: torch.Tensor | None,
        groups: Mapping[str, torch.Tensor | None] | None,
        save_trajectory_every: int,
        context: SamplingContext,
    ) -> RFProteinaSamplingOutput:
        """Drive the shared runner once the rollout context is resolved."""
        if initial_state is None:
            if self.config.match_reference_rng:
                # The reference encodes its input before drawing x_0, and the
                # encoder's posterior sample takes one draw of this shape. Taking
                # it keeps every later draw where the reference's is; the values
                # are never read.
                batch, tokens = padding_mask.shape
                randn(
                    (batch, tokens, self.flow_matcher.modalities["local_latents"].dim),
                    device=padding_mask.device,
                    dtype=dtype,
                    generator=context.generator,
                )
            initial_value = _zero_unencodable_latents(
                self.flow_matcher.sample_noise(
                    padding_mask,
                    dtype=dtype,
                    generator=context.generator,
                    coordinate_condition_mask=coordinate_condition_mask,
                    groups=groups,
                ),
                padding_mask,
                can_be_encoded,
            )
        else:
            initial_value = {modality: initial_state[modality].clone() for modality in PRODUCT_SPACE_MODALITIES}

        return self._run(
            predict=predict,
            padding_mask=padding_mask,
            dtype=dtype,
            initial_value=initial_value,
            schedules=self._schedules(padding_mask.device),
            can_be_encoded=can_be_encoded,
            save_trajectory_every=save_trajectory_every,
            context=context,
        )

    def _run(
        self,
        predict: RFProteinaPredictFn,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        initial_value: RFProteinaState,
        schedules: Mapping[str, FlowMatchingSchedule],
        can_be_encoded: torch.Tensor | None,
        save_trajectory_every: int,
        context: SamplingContext,
        target_backbone: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        start_step: int = 0,
        end_step: int | None = None,
        initial_clean: RFProteinaState | None = None,
        carry_self_condition: bool = False,
    ) -> RFProteinaSamplingOutput:
        """Run a resolved initial state and set of modality schedules."""
        plan = FlowMatchingRolloutPlan(
            schedule=schedules,
            device=padding_mask.device,
            dtype=dtype,
        )
        recycle_depths = self.config.recycle_depths(plan.num_steps)
        resume_state = None
        if start_step > 0 or initial_clean is not None:
            clean = (
                {modality: tensor.clone() for modality, tensor in initial_clean.items()}
                if initial_clean is not None
                else {modality: torch.zeros_like(tensor) for modality, tensor in initial_value.items()}
            )
            resume_state = FlowMatchingState(
                value={modality: tensor.clone() for modality, tensor in initial_value.items()},
                clean=clean,
            )
        integrator = RFProteinaFlowMatchingIntegrator(
            config=self.config,
            initial_value=initial_value,
            padding_mask=padding_mask,
            can_be_encoded=can_be_encoded,
            save_trajectory_every=save_trajectory_every,
            target_backbone=target_backbone,
            flow_matcher=None if target_backbone is None else self.flow_matcher,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            resume_state=resume_state,
            segment_start_step=start_step,
            carry_self_condition=carry_self_condition,
        )

        insertion_logits = None
        trunk_sequence_logits = None
        current_step = start_step

        def predict_velocities(model_input, time) -> dict[str, torch.Tensor]:
            nonlocal current_step, insertion_logits, trunk_sequence_logits
            prediction = predict(
                model_input.state,
                time,
                model_input.self_conditioning,
                model_input.use_self_conditioning,
                recycle_depths[current_step],
            )
            current_step += 1
            insertion = prediction.get("insertion")
            if insertion is not None:
                insertion_logits = insertion.get("logits")
            sequence = prediction.get("sequence")
            if sequence is not None:
                trunk_sequence_logits = sequence.get("logits")
            return {modality: prediction[modality]["v"] for modality in PRODUCT_SPACE_MODALITIES}

        result = GenerativeRunner().run_segment(
            integrator,
            plan,
            predict_velocities,
            context,
            start_step=start_step,
            end_step=end_step,
            initial_state=resume_state,
        )
        return RFProteinaSamplingOutput(
            state=result.final_state.value,
            clean_prediction=result.final_state.clean,
            insertion_logits=insertion_logits,
            trunk_sequence_logits=trunk_sequence_logits,
            trajectory=integrator.trajectory,
            seed=result.seed,
        )

    @torch.no_grad()
    def sample_segment(
        self,
        predict: RFProteinaPredictFn,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        start_step: int,
        end_step: int,
        initial_state: RFProteinaState,
        initial_clean: RFProteinaState | None = None,
        carry_self_condition: bool = False,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        schedules: Mapping[str, FlowMatchingSchedule] | None = None,
        target_backbone: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        seed: int | None = None,
        context: SamplingContext | None = None,
    ) -> RFProteinaSamplingOutput:
        """Continue a product-space rollout over global steps ``[start_step, end_step)``.

        Uses the full modality schedules so times and last-step inverse-folding
        logic stay correct. ``carry_self_condition`` feeds ``initial_clean``
        into the first step of the segment; scoring rollouts leave it false
        and start from a zero-filled clean prediction.

        Args:
            predict: Velocity-network callback.
            padding_mask: ``[B, N]`` mask of real tokens.
            dtype: Product-space dtype.
            start_step: Inclusive global step index.
            end_step: Exclusive global step index.
            initial_state: ``x_t`` at ``start_step``.
            initial_clean: Parent ``x_1`` prediction, or zeros when omitted.
            carry_self_condition: If true, the first step of this segment uses
                ``initial_clean`` when ``start_step > 0``.
            schedules: Full rollout schedules. Defaults to this sampler's
                configured grids.
            context: Explicit rollout context. Takes precedence over
                ``generator`` and ``seed``.
        """
        padding_mask = padding_mask.to(torch.bool)
        initial_value = {modality: initial_state[modality].clone() for modality in PRODUCT_SPACE_MODALITIES}
        resolved_schedules = self._schedules(padding_mask.device) if schedules is None else schedules
        forward = {
            "predict": predict,
            "padding_mask": padding_mask,
            "dtype": dtype,
            "initial_value": initial_value,
            "schedules": resolved_schedules,
            "can_be_encoded": can_be_encoded,
            "save_trajectory_every": save_trajectory_every,
            "target_backbone": None if target_backbone is None else target_backbone.clone(),
            "coordinate_condition_mask": coordinate_condition_mask,
            "groups": groups,
            "start_step": start_step,
            "end_step": end_step,
            "initial_clean": None
            if initial_clean is None
            else {modality: initial_clean[modality].clone() for modality in PRODUCT_SPACE_MODALITIES},
            "carry_self_condition": carry_self_condition,
        }
        if context is not None:
            return self._run(**forward, context=context)
        if generator is not None:
            return self._run(
                **forward,
                context=SamplingContext(generator=generator, seed=generator.initial_seed()),
            )
        with SamplingContext.graph_safe(padding_mask.device, seed) as resolved_context:
            return self._run(**forward, context=resolved_context)

    @torch.no_grad()
    def prepare_partial_rollout(
        self,
        clean_state: RFProteinaState,
        start_times: dict[str, float],
        num_steps: int,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        generator: torch.Generator,
        *,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        schedules: Mapping[str, FlowMatchingSchedule] | None = None,
        resample_target_backbone: bool = False,
        target_backbone: torch.Tensor | None = None,
    ) -> RFProteinaPartialRolloutSetup:
        """Prepare the corruption, schedules, and inverse-fold target."""
        padding_mask = padding_mask.to(torch.bool)
        if resample_target_backbone and target_backbone is None:
            target_backbone = clean_state["bb_ca"]
        if target_backbone is not None and not resample_target_backbone:
            raise ValueError("target_backbone requires resample_target_backbone=True")
        corruption = self.flow_matcher.corrupt_state(
            clean_state,
            start_times,
            padding_mask,
            dtype=dtype,
            generator=generator,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
        )
        resolved_schedules = (
            create_partial_schedules(
                self.config,
                start_times,
                num_steps,
                padding_mask.device,
            )
            if schedules is None
            else schedules
        )
        return RFProteinaPartialRolloutSetup(
            corruption=corruption,
            schedules=resolved_schedules,
            target_backbone=None if target_backbone is None else target_backbone.clone(),
        )

    @torch.no_grad()
    def sample_partial(
        self,
        predict: RFProteinaPredictFn,
        clean_state: RFProteinaState,
        start_times: dict[str, float],
        num_steps: int,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        generator: torch.Generator | None = None,
        save_trajectory_every: int = 0,
        seed: int | None = None,
        resample_target_backbone: bool = False,
        target_backbone: torch.Tensor | None = None,
        schedules: Mapping[str, FlowMatchingSchedule] | None = None,
    ) -> RFProteinaPartialSamplingOutput:
        """Corrupt a clean endpoint and integrate modality schedule tails.

        ``schedules`` replaces the tails otherwise rebuilt from ``start_times``; their first times must be
        the ``start_times`` the state is corrupted to.
        """
        padding_mask = padding_mask.to(torch.bool)
        forward = {
            "predict": predict,
            "clean_state": clean_state,
            "start_times": start_times,
            "num_steps": num_steps,
            "padding_mask": padding_mask,
            "dtype": dtype,
            "can_be_encoded": can_be_encoded,
            "coordinate_condition_mask": coordinate_condition_mask,
            "groups": groups,
            "save_trajectory_every": save_trajectory_every,
            "resample_target_backbone": resample_target_backbone,
            "target_backbone": target_backbone,
            "schedules": schedules,
        }
        if generator is not None:
            return self._sample_partial(
                **forward,
                context=SamplingContext(generator=generator, seed=generator.initial_seed()),
            )
        with SamplingContext.graph_safe(padding_mask.device, seed) as context:
            return self._sample_partial(**forward, context=context)

    def _sample_partial(
        self,
        predict: RFProteinaPredictFn,
        clean_state: RFProteinaState,
        start_times: dict[str, float],
        num_steps: int,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        can_be_encoded: torch.Tensor | None,
        coordinate_condition_mask: torch.Tensor | None,
        groups: Mapping[str, torch.Tensor | None] | None,
        save_trajectory_every: int,
        resample_target_backbone: bool,
        target_backbone: torch.Tensor | None,
        schedules: Mapping[str, FlowMatchingSchedule] | None,
        context: SamplingContext,
    ) -> RFProteinaPartialSamplingOutput:
        setup = self.prepare_partial_rollout(
            clean_state,
            start_times,
            num_steps,
            padding_mask,
            dtype,
            context.generator,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            schedules=schedules,
            resample_target_backbone=resample_target_backbone,
            target_backbone=target_backbone,
        )
        sampling = self._run(
            predict=predict,
            padding_mask=padding_mask,
            dtype=dtype,
            initial_value=setup.corruption.corrupted_state,
            schedules=setup.schedules,
            can_be_encoded=can_be_encoded,
            save_trajectory_every=save_trajectory_every,
            context=context,
            target_backbone=setup.target_backbone,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
        )
        return RFProteinaPartialSamplingOutput(
            sampling=sampling,
            corruption=setup.corruption,
            start_times=dict(start_times),
            num_steps=num_steps,
        )
