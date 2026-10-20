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

import math
from collections.abc import Mapping

import pytest
import torch
from pydantic import ValidationError

from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingRolloutPlan,
    FlowMatchingSchedule,
)
from bionemo_ir._torch.sampling.runner import GenerativeRunner, SamplingCancelled
from bionemo_ir._torch.search import (
    SEARCH_REWARD_KEY,
    CompositeMetricEvaluator,
    MetricClip,
    MetricEvaluationContext,
    build_weighted_objective,
    derive_search_seed,
    score_metrics,
    select_beam_indices,
    validate_step_checkpoints,
)
from bionemo_ir.models.rfproteina.config import (
    RFProteinaBeamSearchConfig,
    RFProteinaSearchRewardConfig,
)


def _linear_plan(num_steps: int, device: torch.device | None = None) -> FlowMatchingRolloutPlan:
    device = torch.device("cpu") if device is None else device
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    return FlowMatchingRolloutPlan(
        schedule={"x": FlowMatchingSchedule(times=times)},
        device=device,
        dtype=torch.float32,
    )


def test_validate_step_checkpoints_requires_full_cover() -> None:
    assert validate_step_checkpoints([0, 2, 4], 4) == (0, 2, 4)
    with pytest.raises(ValueError, match="begin at 0"):
        validate_step_checkpoints([1, 4], 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_step_checkpoints([0, 2, 2, 4], 4)
    with pytest.raises(ValueError, match="must equal num_steps"):
        validate_step_checkpoints([0, 3], 4)


def test_derive_search_seed_is_identity_keyed_not_order_keyed() -> None:
    forward = {
        (replica, branch): derive_search_seed(7, 1, replica, branch, 2) for branch in range(3) for replica in range(2)
    }
    reverse = {
        (replica, branch): derive_search_seed(7, 1, replica, branch, 2)
        for branch in reversed(range(3))
        for replica in reversed(range(2))
    }
    assert forward == reverse
    assert derive_search_seed(7, 1, 0, 1, 2) != derive_search_seed(7, 1, 1, 0, 2)
    assert derive_search_seed(7, 1, 0, 0, 2) != 0


def test_weighted_objective_clips_and_invalidates() -> None:
    objective = build_weighted_objective(
        weights={"clash": -1.0, "rmsd": 2.0},
        clip_values={"clash": {"max": 0.5}},
    )
    assert objective.score({"clash": 0.8, "rmsd": 1.0}) == pytest.approx(-0.5 + 2.0)
    assert objective.score({"clash": 0.1, "rmsd": 1.0}) == pytest.approx(-0.1 + 2.0)
    assert objective.score({"rmsd": 1.0}) == float("-inf")
    assert objective.score({"clash": 0.1, "rmsd": float("nan")}) == float("-inf")
    published = {"clash": 0.1, "rmsd": 1.0, SEARCH_REWARD_KEY: 0.0}
    assert score_metrics(published, objective) == pytest.approx(-0.1 + 2.0)


def test_weighted_objective_override_replaces_weights() -> None:
    objective = build_weighted_objective(
        weights={"clash": -1.0},
        checkpoint_overrides={4: {"weights": {"rmsd": -2.0}}},
    )
    assert objective.at_step(2).score({"clash": 0.25}) == pytest.approx(-0.25)
    assert objective.at_step(4).score({"clash": 0.25}) == float("-inf")
    assert objective.at_step(4).score({"rmsd": 0.5}) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="never scores"):
        objective.validate_checkpoint_steps([2, 3])


def test_select_beam_indices_is_per_request_and_stable_on_ties() -> None:
    # branch-major, replica, request. Two requests, two replicas, two branches.
    rewards = torch.tensor(
        [
            1.0,
            0.0,  # b0 r0
            3.0,
            5.0,  # b0 r1
            1.0,
            4.0,  # b1 r0
            2.0,
            5.0,  # b1 r1  request 1 ties 5.0 at lower index b0 r1
        ]
    )
    selected = select_beam_indices(rewards, n_branch=2, beam_width=2, num_requests=2)
    assert selected.tolist() == [2, 3, 6, 7]
    assert rewards[selected[0]].item() == 3.0
    assert rewards[selected[1]].item() == 5.0
    assert int(selected[1]) == 3


def test_select_beam_indices_all_invalid_raises() -> None:
    rewards = torch.full((8,), float("-inf"))
    with pytest.raises(ValueError, match="-inf"):
        select_beam_indices(rewards, n_branch=2, beam_width=2, num_requests=2)


def test_beam_search_config_rejects_unknown_fields_and_bad_checkpoints() -> None:
    with pytest.raises(ValidationError):
        RFProteinaBeamSearchConfig(
            step_checkpoints=[0, 4],
            rewards=RFProteinaSearchRewardConfig(weights={"clash": -1.0}),
            unknown=1,
        )
    with pytest.raises(ValidationError, match="begin at 0"):
        RFProteinaBeamSearchConfig(
            step_checkpoints=[1, 4],
            rewards=RFProteinaSearchRewardConfig(weights={"clash": -1.0}),
        )
    config = RFProteinaBeamSearchConfig(
        step_checkpoints=[0, 2, 4],
        n_branch=2,
        beam_width=3,
        rewards=RFProteinaSearchRewardConfig(
            weights={"clash_metrics.frac_atoms_clashing_1p5A": -1.0},
            checkpoint_overrides={"4": {"weights": {"design_motif_allatom_rmsd.rmsd": -1.0}}},
        ),
    )
    assert 4 in config.rewards.checkpoint_overrides
    objective = config.rewards.to_weighted_objective()
    assert "design_motif_allatom_rmsd.rmsd" in objective.at_step(4).weights


def test_metric_clip_requires_a_bound() -> None:
    with pytest.raises(ValueError, match="min and/or max"):
        MetricClip()
    assert MetricClip(max=1.0).apply(2.0) == 1.0


def test_composite_metric_evaluator_merges_plain_and_contextual_metrics() -> None:
    contexts: list[MetricEvaluationContext] = []

    def plain_evaluator(
        prediction: object,
        static_features: Mapping[str, object],
    ) -> list[dict[str, float]]:
        del static_features
        if not isinstance(prediction, int):
            raise TypeError("prediction must be a batch size")
        return [{"plain": float(index)} for index in range(prediction)]

    class ContextEvaluator:
        def evaluate(
            self,
            prediction: object,
            static_features: Mapping[str, object],
            context: MetricEvaluationContext,
        ) -> list[dict[str, float]]:
            del static_features
            if not isinstance(prediction, int):
                raise TypeError("prediction must be a batch size")
            contexts.append(context)
            return [{"contextual": float(context.checkpoint or 0)} for _ in range(prediction)]

    context = MetricEvaluationContext(checkpoint=4, tags=("oracle",), request_ids=("a", "b"))
    evaluator = CompositeMetricEvaluator([plain_evaluator, ContextEvaluator()])

    assert evaluator.evaluate(2, {}, context) == [
        {"plain": 0.0, "contextual": 4.0},
        {"plain": 1.0, "contextual": 4.0},
    ]
    assert contexts == [context]


def test_composite_metric_evaluator_rejects_invalid_composition() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeMetricEvaluator([])

    context = MetricEvaluationContext()
    collision = CompositeMetricEvaluator(
        [
            lambda _prediction, _features: {"score": 1.0},
            lambda _prediction, _features: {"score": 2.0},
        ]
    )
    with pytest.raises(ValueError, match="collides"):
        collision.evaluate(object(), {}, context)

    mismatched_rows = CompositeMetricEvaluator(
        [
            lambda _prediction, _features: [{"first": 1.0}],
            lambda _prediction, _features: [{"second": 1.0}, {"second": 2.0}],
        ]
    )
    with pytest.raises(ValueError, match="returned 2 rows"):
        mismatched_rows.evaluate(object(), {}, context)


def test_generative_runner_segment_matches_full_rollout() -> None:
    start = torch.zeros(1, 2, 3)
    velocity = torch.ones(1, 2, 3)
    plan = _linear_plan(num_steps=4)

    def predict(_model_input, _time):
        return {"x": velocity}

    full = GenerativeRunner().run(
        FlowMatchingIntegrator(FlowMatchingIntegratorConfig(), initial_value={"x": start}),
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
    )
    first = GenerativeRunner().run_segment(
        FlowMatchingIntegrator(FlowMatchingIntegratorConfig(), initial_value={"x": start}),
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
        end_step=2,
    )
    second = GenerativeRunner().run_segment(
        FlowMatchingIntegrator(FlowMatchingIntegratorConfig(), initial_value={"x": start}),
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
        start_step=2,
        initial_state=first.final_state.clone(),
    )
    torch.testing.assert_close(second.final_state.value["x"], full.final_state.value["x"])
    torch.testing.assert_close(second.final_state.clean["x"], full.final_state.clean["x"])
    assert first.num_steps == 2
    assert second.num_steps == 2
    assert full.num_steps == 4


def test_generative_runner_prepares_continuation_without_advancing_generator() -> None:
    class ContinuationIntegrator:
        def __init__(self) -> None:
            self.initialize_calls = 0

        def num_steps(self, plan: object) -> int:
            del plan
            return 2

        def initialize(self, plan: object, context: SamplingContext) -> torch.Tensor:
            del plan
            self.initialize_calls += 1
            return torch.randn((), generator=context.generator)

        def step(
            self,
            step_index: int,
            state: torch.Tensor,
            plan: object,
            predict: object,
            context: SamplingContext,
        ) -> torch.Tensor:
            del step_index, plan, predict, context
            return state + 1

    integrator = ContinuationIntegrator()
    context = SamplingContext.create(torch.device("cpu"), seed=0)
    generator_state = context.generator.get_state().clone()
    result = GenerativeRunner().run_segment(
        integrator,
        object(),
        lambda model_input, time: (model_input, time),
        context,
        start_step=1,
        initial_state=torch.tensor(3),
    )
    assert result.final_state.item() == 4
    assert integrator.initialize_calls == 1
    assert torch.equal(context.generator.get_state(), generator_state)


def test_generative_runner_cancellation_observes_resumed_step() -> None:
    class Integrator:
        def num_steps(self, _plan: object) -> int:
            return 4

        def initialize(self, _plan: object, _context: SamplingContext) -> object:
            return object()

        def step(self, *_args: object) -> object:
            raise AssertionError("cancelled segment must not execute a step")

    context = SamplingContext.create(torch.device("cpu"), seed=0)
    observed_steps: list[int] = []

    def is_cancelled() -> bool:
        observed_steps.append(context.step_index)
        return True

    context.is_cancelled = is_cancelled
    with pytest.raises(SamplingCancelled, match="before step 2"):
        GenerativeRunner().run_segment(
            Integrator(),
            object(),
            lambda model_input, time: (model_input, time),
            context,
            start_step=2,
            initial_state=object(),
        )
    assert observed_steps == [2]


def test_segment_self_conditioning_can_carry_parent_clean() -> None:
    start = torch.zeros(1, 2, 3)
    velocity = torch.ones(1, 2, 3)
    plan = _linear_plan(num_steps=4)
    seen: list[bool] = []
    seen_clean: list[torch.Tensor] = []

    def predict(model_input, time):
        seen.append(model_input.use_self_conditioning)
        seen_clean.append(model_input.self_conditioning["x"].clone())
        return {"x": velocity}

    noncarry_integrator = FlowMatchingIntegrator(
        FlowMatchingIntegratorConfig(self_condition=True),
        initial_value={"x": start},
    )
    first = GenerativeRunner().run_segment(
        noncarry_integrator,
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
        end_step=2,
    )
    assert seen == [False, True]
    seen.clear()
    seen_clean.clear()
    GenerativeRunner().run_segment(
        FlowMatchingIntegrator(
            FlowMatchingIntegratorConfig(self_condition=True),
            initial_value={"x": start},
            carry_self_condition=True,
        ),
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
        start_step=2,
        initial_state=first.final_state.clone(),
    )
    assert seen == [True, True]
    assert torch.count_nonzero(seen_clean[0]) > 0
    seen.clear()
    seen_clean.clear()
    GenerativeRunner().run_segment(
        noncarry_integrator,
        plan,
        predict,
        SamplingContext.create(torch.device("cpu"), seed=0),
        start_step=2,
        initial_state=first.final_state.clone(),
    )
    assert seen == [False, True]
    assert torch.count_nonzero(seen_clean[0]) == 0


def test_resume_without_state_raises() -> None:
    plan = _linear_plan(2)
    with pytest.raises(ValueError, match="initial_state"):
        GenerativeRunner().run_segment(
            FlowMatchingIntegrator(FlowMatchingIntegratorConfig(), initial_value={"x": torch.zeros(1, 2, 3)}),
            plan,
            lambda model_input, time: {"x": torch.zeros(1, 2, 3)},
            SamplingContext.create(torch.device("cpu"), seed=0),
            start_step=1,
        )


def test_score_metrics_zero_weight_skips_missing_key() -> None:
    objective = build_weighted_objective(weights={"clash": 0.0, "rmsd": 1.0})
    assert objective.score({"rmsd": 2.0}) == 2.0
    assert math.isfinite(objective.score({"rmsd": 2.0}))
