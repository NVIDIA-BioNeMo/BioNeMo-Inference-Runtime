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
"""Checkpointed metric-weighted beam search for RFProteína."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass

import torch

from bionemo_ir._torch.modules.rfproteina.contracts import (
    RFProteinaPredictFn,
    RFProteinaPredictionOutput,
    RFProteinaSamplingOutput,
    RFProteinaSearchCandidate,
    RFProteinaSearchCost,
    RFProteinaSearchOutput,
    RFProteinaState,
)
from bionemo_ir._torch.modules.rfproteina.flow_matcher import PRODUCT_SPACE_MODALITIES
from bionemo_ir._torch.modules.rfproteina.sampler import RFProteinaProductSpaceSampler
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import FlowMatchingSchedule
from bionemo_ir._torch.search import (
    SEARCH_REWARD_KEY,
    MetricEvaluationContext,
    MetricEvaluatorLike,
    WeightedObjective,
    derive_search_seed,
    evaluate_metrics,
    select_beam_indices,
    validate_step_checkpoints,
)

DecodeFn = Callable[[RFProteinaSamplingOutput, Mapping[str, torch.Tensor]], RFProteinaPredictionOutput]


def _clone_state(state: RFProteinaState) -> RFProteinaState:
    return {modality: state[modality].clone() for modality in PRODUCT_SPACE_MODALITIES}


def _cat_states(states: Sequence[RFProteinaState]) -> RFProteinaState:
    return {modality: torch.cat([state[modality] for state in states], dim=0) for modality in PRODUCT_SPACE_MODALITIES}


def _slice_state(state: RFProteinaState, start: int, end: int) -> RFProteinaState:
    return {modality: state[modality][start:end] for modality in PRODUCT_SPACE_MODALITIES}


def _index_state(state: RFProteinaState, indices: torch.Tensor) -> RFProteinaState:
    return {modality: state[modality][indices] for modality in PRODUCT_SPACE_MODALITIES}


def _index_value(value: object, index: int) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        return value if value.ndim == 0 else value[index : index + 1]
    if isinstance(value, Mapping):
        return {key: _index_value(item, index) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: _index_value(getattr(value, field.name), index) for field in fields(value)})
    return value


def index_prediction(prediction: RFProteinaPredictionOutput, index: int) -> RFProteinaPredictionOutput:
    """Slice a decoded batch down to a single-request prediction."""
    indexed = _index_value(prediction, index)
    if not isinstance(indexed, RFProteinaPredictionOutput):
        raise TypeError("indexed prediction must remain RFProteinaPredictionOutput")
    return indexed


def repeat_feature_batch(features: Mapping[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    """Tile a request batch ``repeats`` times along the batch axis."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    tiled: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if torch.is_tensor(value) and value.ndim >= 1:
            tiled[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
        else:
            tiled[key] = value
    return tiled


def slice_feature_batch(
    features: Mapping[str, torch.Tensor],
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Slice tensors whose leading dimension matches ``batch_size``."""
    sliced: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _normalize_metrics(
    result: Mapping[str, float] | Sequence[Mapping[str, float]],
    batch: int,
) -> list[dict[str, float]]:
    if isinstance(result, Mapping):
        if batch != 1:
            raise ValueError(f"metric evaluator returned one mapping for a batch of {batch}")
        return [dict(result)]
    rows = [dict(row) for row in result]
    if len(rows) != batch:
        raise ValueError(f"metric evaluator returned {len(rows)} rows for a batch of {batch}")
    return rows


def _candidate_layout_index(branch: int, replica: int, request: int, beam_width: int, num_requests: int) -> int:
    return branch * beam_width * num_requests + replica * num_requests + request


def run_rfproteina_beam_search(
    *,
    sampler: RFProteinaProductSpaceSampler,
    predict: RFProteinaPredictFn,
    decode_prediction: DecodeFn,
    evaluator: MetricEvaluatorLike,
    objective: WeightedObjective,
    static_features: Mapping[str, torch.Tensor],
    padding_mask: torch.Tensor,
    dtype: torch.dtype,
    initial_slots: Sequence[RFProteinaState],
    num_steps: int,
    step_checkpoints: Sequence[int],
    n_branch: int,
    beam_width: int,
    can_be_encoded: torch.Tensor | None = None,
    coordinate_condition_mask: torch.Tensor | None = None,
    groups: Mapping[str, torch.Tensor | None] | None = None,
    schedules: Mapping[str, FlowMatchingSchedule],
    target_backbone: torch.Tensor | None = None,
    scoring_chunk_size: int | None = None,
    intermediate_save_min_reward: float | None = None,
    root_seed: int,
) -> RFProteinaSearchOutput:
    """Run checkpointed branching, completion scoring, and stable per-request pruning.

    Candidates are stacked branch-major, then replica, then request. Scoring
    rollouts start from a zero-filled clean prediction. Branching rollouts
    carry the parent's clean prediction after the first interval. The beam
    keeps noisy ``x_t`` at each checkpoint; last-checkpoint survivors are
    already complete and are not decoded a second time.
    """
    padding_mask = padding_mask.to(torch.bool)
    num_requests = padding_mask.shape[0]
    checkpoints = validate_step_checkpoints(step_checkpoints, num_steps)
    objective.validate_checkpoint_steps(checkpoints[1:])
    if len(initial_slots) != beam_width:
        raise ValueError(f"initial_slots must contain beam_width={beam_width} states, got {len(initial_slots)}")
    resolved_schedules = schedules
    chunk_rows = num_requests * n_branch * beam_width if scoring_chunk_size is None else scoring_chunk_size
    if chunk_rows < 1:
        raise ValueError("scoring_chunk_size must be positive")

    slot_states = [_clone_state(state) for state in initial_slots]
    slot_clean: list[RFProteinaState | None] = [None] * beam_width
    slot_parents: list[list[str | None]] = [[None] * num_requests for _ in range(beam_width)]
    device = padding_mask.device
    intermediates: list[RFProteinaSearchCandidate] = []
    velocity_steps = 0
    completion_rollouts = 0
    last_select = torch.empty(0, dtype=torch.long, device=device)
    last_candidates: list[RFProteinaSearchCandidate] = []

    def _run_segment(
        start_step: int,
        end_step: int,
        state: RFProteinaState,
        clean: RFProteinaState | None,
        carry_self_condition: bool,
        seed: int,
        logits_fallback: torch.Tensor | None = None,
        sequence_fallback: torch.Tensor | None = None,
    ) -> RFProteinaSamplingOutput:
        nonlocal velocity_steps
        context = SamplingContext.create(device, seed)
        if end_step == start_step:
            zeros = {modality: torch.zeros_like(tensor) for modality, tensor in state.items()}
            return RFProteinaSamplingOutput(
                state=_clone_state(state),
                clean_prediction=zeros if clean is None else _clone_state(clean),
                insertion_logits=None if logits_fallback is None else logits_fallback.clone(),
                trunk_sequence_logits=None if sequence_fallback is None else sequence_fallback.clone(),
                trajectory=[],
                seed=seed,
            )
        sampling = sampler.sample_segment(
            predict,
            padding_mask,
            dtype,
            start_step,
            end_step,
            initial_state=state,
            initial_clean=clean,
            carry_self_condition=carry_self_condition,
            can_be_encoded=can_be_encoded,
            coordinate_condition_mask=coordinate_condition_mask,
            groups=groups,
            schedules=resolved_schedules,
            target_backbone=target_backbone,
            context=context,
        )
        velocity_steps += (end_step - start_step) * next(iter(state.values())).shape[0]
        if sampling.insertion_logits is None and logits_fallback is not None:
            sampling.insertion_logits = logits_fallback
        if sampling.trunk_sequence_logits is None and sequence_fallback is not None:
            sampling.trunk_sequence_logits = sequence_fallback
        return sampling

    def _score_completed(
        completed: RFProteinaSamplingOutput,
        checkpoint: int,
        row_seeds: Sequence[int],
        row_parents: Sequence[str | None],
        row_ids: Sequence[str],
        scoring_objective: WeightedObjective,
    ) -> tuple[torch.Tensor, list[RFProteinaSearchCandidate]]:
        total = completed.state["bb_ca"].shape[0]
        tiled = repeat_feature_batch(static_features, total // num_requests)
        candidates: list[RFProteinaSearchCandidate] = []
        rewards: list[float] = []
        for start in range(0, total, chunk_rows):
            end = min(start + chunk_rows, total)
            chunk_sampling = RFProteinaSamplingOutput(
                state=_slice_state(completed.state, start, end),
                clean_prediction=_slice_state(completed.clean_prediction, start, end),
                insertion_logits=(
                    None if completed.insertion_logits is None else completed.insertion_logits[start:end]
                ),
                trunk_sequence_logits=(
                    None if completed.trunk_sequence_logits is None else completed.trunk_sequence_logits[start:end]
                ),
                trajectory=[],
                seed=completed.seed,
            )
            chunk_features = slice_feature_batch(tiled, start, end, total)
            prediction = decode_prediction(chunk_sampling, chunk_features)
            evaluation_context = MetricEvaluationContext(
                checkpoint=checkpoint,
                tags=scoring_objective.reward_metrics_tags,
                request_ids=tuple(row_ids[start:end]),
            )
            metric_rows = _normalize_metrics(
                evaluate_metrics(evaluator, prediction, chunk_features, evaluation_context),
                end - start,
            )
            for local, metrics in enumerate(metric_rows):
                global_index = start + local
                reward = scoring_objective.score(metrics)
                published = dict(metrics)
                published[SEARCH_REWARD_KEY] = reward
                request_index = global_index % num_requests
                inner = global_index // num_requests
                replica = inner % beam_width
                rank = replica
                candidates.append(
                    RFProteinaSearchCandidate(
                        prediction=index_prediction(prediction, local),
                        metrics=published,
                        reward=reward,
                        rank=rank,
                        request_index=request_index,
                        parent_id=row_parents[global_index],
                        candidate_id=row_ids[global_index],
                        checkpoint=checkpoint,
                        seed=row_seeds[global_index],
                        is_intermediate=False,
                    )
                )
                rewards.append(reward)
        return torch.tensor(rewards, device=device, dtype=torch.float32), candidates

    for interval, (start_step, end_step) in enumerate(zip(checkpoints, checkpoints[1:], strict=False)):
        branched_states: list[RFProteinaState] = []
        branched_clean: list[RFProteinaState] = []
        branched_logits: list[torch.Tensor | None] = []
        branched_sequence: list[torch.Tensor | None] = []
        row_seeds: list[int] = []
        row_parents: list[str | None] = []
        row_ids: list[str] = []
        carry = start_step > 0
        for branch in range(n_branch):
            for replica in range(beam_width):
                seed = derive_search_seed(root_seed, 1, replica, branch, start_step)
                sampling = _run_segment(
                    start_step,
                    end_step,
                    slot_states[replica],
                    slot_clean[replica],
                    carry,
                    seed,
                )
                branched_states.append(sampling.state)
                branched_clean.append(sampling.clean_prediction)
                branched_logits.append(sampling.insertion_logits)
                branched_sequence.append(sampling.trunk_sequence_logits)
                for request in range(num_requests):
                    row_seeds.append(seed)
                    row_parents.append(slot_parents[replica][request])
                    row_ids.append(f"req{request}_step{end_step}_b{branch}_r{replica}")

        xt = _cat_states(branched_states)
        x1 = _cat_states(branched_clean)
        if all(logits is not None for logits in branched_logits):
            insertion_logits = torch.cat(branched_logits, dim=0)
        else:
            insertion_logits = None
        if all(logits is not None for logits in branched_sequence):
            sequence_logits = torch.cat(branched_sequence, dim=0)
        else:
            sequence_logits = None

        scoring_state = xt
        scoring_clean = None
        if end_step < num_steps:
            scored_states: list[RFProteinaState] = []
            scored_clean: list[RFProteinaState] = []
            scored_logits: list[torch.Tensor | None] = []
            scored_sequence: list[torch.Tensor | None] = []
            for branch in range(n_branch):
                for replica in range(beam_width):
                    start = _candidate_layout_index(branch, replica, 0, beam_width, num_requests)
                    end = start + num_requests
                    score_seed = derive_search_seed(root_seed, 2, replica, branch, end_step)
                    fallback = None if insertion_logits is None else insertion_logits[start:end]
                    sequence_slice = None if sequence_logits is None else sequence_logits[start:end]
                    completed = _run_segment(
                        end_step,
                        num_steps,
                        _slice_state(xt, start, end),
                        None,
                        False,
                        score_seed,
                        logits_fallback=fallback,
                        sequence_fallback=sequence_slice,
                    )
                    scored_states.append(completed.state)
                    scored_clean.append(completed.clean_prediction)
                    scored_logits.append(completed.insertion_logits)
                    scored_sequence.append(completed.trunk_sequence_logits)
                    completion_rollouts += num_requests
            scoring_state = _cat_states(scored_states)
            scoring_clean = _cat_states(scored_clean)
            if all(logits is not None for logits in scored_logits):
                insertion_logits = torch.cat(scored_logits, dim=0)
            if all(logits is not None for logits in scored_sequence):
                sequence_logits = torch.cat(scored_sequence, dim=0)
        else:
            completion_rollouts += n_branch * beam_width * num_requests
            scoring_clean = x1

        completed = RFProteinaSamplingOutput(
            state=scoring_state,
            clean_prediction=scoring_clean,
            insertion_logits=insertion_logits,
            trunk_sequence_logits=sequence_logits,
            trajectory=[],
            seed=root_seed,
        )
        checkpoint_objective = objective.at_step(end_step)
        rewards, scored_candidates = _score_completed(
            completed,
            end_step,
            row_seeds,
            row_parents,
            row_ids,
            checkpoint_objective,
        )
        try:
            select = select_beam_indices(rewards, n_branch, beam_width, num_requests)
        except ValueError as error:
            raise ValueError(f"every candidate at step {end_step} scored -inf: {error}") from error

        last_interval = interval == len(checkpoints) - 2
        survivor_mask = torch.zeros(rewards.shape[0], dtype=torch.bool, device=device)
        if last_interval:
            survivor_mask[select] = True
        if intermediate_save_min_reward is not None:
            keep = rewards >= intermediate_save_min_reward
            keep &= ~survivor_mask
            for index in keep.nonzero(as_tuple=False).flatten().tolist():
                candidate = scored_candidates[index]
                intermediates.append(
                    RFProteinaSearchCandidate(
                        prediction=candidate.prediction,
                        metrics=candidate.metrics,
                        reward=candidate.reward,
                        rank=candidate.rank,
                        request_index=candidate.request_index,
                        parent_id=candidate.parent_id,
                        candidate_id=candidate.candidate_id,
                        checkpoint=candidate.checkpoint,
                        seed=candidate.seed,
                        is_intermediate=True,
                    )
                )

        xt = _index_state(xt, select)
        x1 = _index_state(x1, select)
        if insertion_logits is not None:
            insertion_logits = insertion_logits[select]
        selected_ids = [row_ids[int(index)] for index in select.tolist()]
        for replica in range(beam_width):
            start = replica * num_requests
            end = start + num_requests
            slot_states[replica] = _slice_state(xt, start, end)
            slot_clean[replica] = _slice_state(x1, start, end)
            slot_parents[replica] = selected_ids[start:end]
        last_select = select
        last_candidates = scored_candidates

    finals: list[RFProteinaSearchCandidate] = []
    for rank, index in enumerate(last_select.tolist()):
        replica = rank // num_requests
        request = rank % num_requests
        candidate = last_candidates[index]
        finals.append(
            RFProteinaSearchCandidate(
                prediction=candidate.prediction,
                metrics=candidate.metrics,
                reward=candidate.reward,
                rank=replica,
                request_index=request,
                parent_id=candidate.parent_id,
                candidate_id=candidate.candidate_id,
                checkpoint=candidate.checkpoint,
                seed=candidate.seed,
                is_intermediate=False,
            )
        )

    return RFProteinaSearchOutput(
        candidates=tuple(intermediates + finals),
        cost=RFProteinaSearchCost(
            num_requests=num_requests,
            beam_width=beam_width,
            n_branch=n_branch,
            num_checkpoints=len(checkpoints),
            completion_rollouts=completion_rollouts,
            velocity_steps=velocity_steps,
        ),
    )
