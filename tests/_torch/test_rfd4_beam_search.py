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

import os
from collections.abc import Sequence
from typing import Any, NoReturn

import pytest
import torch

from bionemo_ir._torch.layers.transformers.local_latents_decoder import get_residue_atom_mask
from bionemo_ir._torch.modules.rfproteina import create_modality_schedule, sample_reference_state
from bionemo_ir._torch.modules.rfproteina.metrics import (
    CLASH_METRIC_KEY,
    MOTIF_RMSD_METRIC_KEY,
    SEQUENCE_ADHERENCE_METRIC_KEY,
)
from bionemo_ir._torch.modules.rfproteina.representation_builder import SEQUENCE_PROBABILITY_TOKENS
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.random import normal_draws
from bionemo_ir._torch.search import SEARCH_REWARD_KEY, MetricEvaluationContext
from bionemo_ir.data.schemas import (
    AtomIdentity,
    AtomType,
    OracleResponse,
    OracleStatus,
    PolymerType,
    ResidueIdentity,
    ResTypes,
)
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint
from bionemo_ir.models.rfproteina import (
    RFProteina,
    RFProteinaBeamSearchConfig,
    RFProteinaConfig,
    RFProteinaLocalMetricEvaluator,
    RFProteinaPartialSimulationConfig,
    RFProteinaRepresentations,
    RFProteinaSearchRewardConfig,
    load_rfproteina_checkpoint_state,
    modeling,
)
from bionemo_ir.models.rfproteina.modeling import _resolve_runtime_masks
from bionemo_ir.oracles import (
    CallableAtomTypeResolver,
    CallableOracleProvider,
    CallableResidueTypeResolver,
)
from bionemo_ir.oracles.adapters import OracleMetricEvaluator, RFProteinaOracleRequestAdapter
from tests._torch import init_module_weights
from tests._torch.test_rfd4_sampling import (
    _deterministic_sampling_config,
    _native_static_features,
    _small_decoder_config,
    _small_encoder_config,
    _small_trunk_config,
)


def _tiny_model(num_steps: int = 4, *, self_condition: bool = True) -> RFProteina:
    sampling = _deterministic_sampling_config(num_steps=num_steps)
    sampling.self_condition = self_condition
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            encoder=_small_encoder_config(),
            sampling=sampling,
        )
    ).cuda()
    init_module_weights(model)
    return model


def _clash_search_config(num_steps: int, **updates) -> RFProteinaBeamSearchConfig:
    values = {
        "step_checkpoints": [0, num_steps // 2, num_steps],
        "n_branch": 2,
        "beam_width": 2,
        "rewards": RFProteinaSearchRewardConfig(weights={CLASH_METRIC_KEY: -1.0}),
    }
    values.update(updates)
    return RFProteinaBeamSearchConfig(**values)


def _velocity_predict(model: RFProteina, static_features: dict[str, torch.Tensor], padding_mask: torch.Tensor):
    def predict(state, times, self_conditioning, active, recycle_depth):
        representations = model.representation_builder(
            static_features=static_features,
            state=state,
            times=times,
            self_conditioning=self_conditioning,
            self_conditioning_active=active,
            padding_mask=padding_mask,
        )
        return model.velocity(representations, padding_mask, recycle_depth=recycle_depth)

    return predict


def test_rfproteina_segment_matches_uninterrupted_rollout() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.tensor([[True] * 6 + [False] * 2], device="cuda")
    static_features = _native_static_features(mask)
    can_be_encoded, coordinate_condition_mask = _resolve_runtime_masks(
        static_features, mask, None, None, model.decoder.num_residue_types
    )
    predict = _velocity_predict(model, static_features, mask)
    kwargs = {
        "can_be_encoded": can_be_encoded,
        "coordinate_condition_mask": coordinate_condition_mask,
    }
    full_generator = torch.Generator(device="cuda")
    full_generator.manual_seed(11)
    full = model.sampler.sample(predict, mask, model.state_dtype, generator=full_generator, **kwargs)

    segment_generator = torch.Generator(device="cuda")
    segment_generator.manual_seed(11)
    initial = sample_reference_state(
        mask,
        model.config.trunk.latent_dim,
        model.state_dtype,
        model.config.sampling,
        generator=segment_generator,
        **kwargs,
    )
    context = SamplingContext(generator=segment_generator, seed=11)
    first = model.sampler.sample_segment(
        predict, mask, model.state_dtype, 0, 2, initial_state=initial, context=context, **kwargs
    )
    second = model.sampler.sample_segment(
        predict,
        mask,
        model.state_dtype,
        2,
        4,
        initial_state=first.state,
        initial_clean=first.clean_prediction,
        carry_self_condition=True,
        context=context,
        **kwargs,
    )
    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(second.state[modality], full.state[modality], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            second.clean_prediction[modality], full.clean_prediction[modality], atol=1e-5, rtol=1e-5
        )


def test_rfproteina_search_is_seed_reproducible_and_chunk_order_independent() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.tensor([[True] * 6 + [False] * 2], device="cuda")
    static_features = _native_static_features(mask)
    config = _clash_search_config(4)
    first = model.search(static_features, config, seed=5)
    second = model.search(static_features, config, seed=5)
    other = model.search(static_features, config, seed=6)
    chunked = model.search(static_features, _clash_search_config(4, scoring_chunk_size=1), seed=5)
    assert len(first.final_candidates) == 2
    assert first.cost.num_requests == 1
    assert first.cost.beam_width == 2
    assert first.cost.n_branch == 2
    assert first.cost.completion_rollouts == 8
    assert first.cost.velocity_steps > 0
    for left, right in zip(first.final_candidates, second.final_candidates, strict=True):
        torch.testing.assert_close(left.prediction.coordinates, right.prediction.coordinates)
        assert left.reward == right.reward
        assert SEARCH_REWARD_KEY in left.metrics
        assert left.candidate_id == right.candidate_id
    for left, right in zip(first.final_candidates, chunked.final_candidates, strict=True):
        torch.testing.assert_close(left.prediction.coordinates, right.prediction.coordinates)
        assert left.reward == right.reward
    assert any(
        not torch.allclose(left.prediction.coordinates, right.prediction.coordinates)
        for left, right in zip(first.final_candidates, other.final_candidates, strict=True)
    )


class _BeamCaptured(Exception):
    """Ends a search once its beam driver has been handed the initial states."""


def _initial_beam(
    model: RFProteina,
    static_features: dict[str, torch.Tensor],
    config: RFProteinaBeamSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
    stream_seed: int,
) -> dict[str, Any]:
    """The ``initial_slots`` and driver arguments one search hands its beam driver, with draws from one hosted stream."""
    captured: dict[str, Any] = {}

    def driver(**kwargs: Any) -> NoReturn:
        captured.update(kwargs)
        raise _BeamCaptured

    monkeypatch.setattr(modeling, "run_rfproteina_beam_search", driver)
    generator = torch.Generator(device="cuda").manual_seed(stream_seed)

    def draw(shape: Sequence[int], *, device: torch.device | str, dtype: torch.dtype | None) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    with normal_draws(draw), pytest.raises(_BeamCaptured):
        model.search(static_features, config, seed=5)
    return captured


@pytest.mark.parametrize("matched", [True, False])
def test_search_draws_the_reference_beam_only_when_it_matches_the_reference(
    matched: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference draws every replica's initial state in one call over the stacked rows, so a replica's rows are
    the stream's draws at its own offset. Matched noise reproduces that; otherwise each replica draws on its own."""
    model = _tiny_model(num_steps=4)
    model.config.sampling.match_reference_rng = matched
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 8], device="cuda")
    static_features = _native_static_features(mask)
    config = _clash_search_config(4)
    captured = _initial_beam(model, static_features, config, monkeypatch, stream_seed=7)
    slots = captured["initial_slots"]

    def tile(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.repeat(config.beam_width, *([1] * (value.ndim - 1)))

    generator = torch.Generator(device="cuda").manual_seed(7)
    with normal_draws(
        lambda shape, *, device, dtype: torch.randn(shape, device=device, dtype=dtype, generator=generator)
    ):
        reference = sample_reference_state(
            tile(captured["padding_mask"]),
            model.config.trunk.latent_dim,
            captured["dtype"],
            model.config.sampling,
            can_be_encoded=tile(captured["can_be_encoded"]),
            coordinate_condition_mask=tile(captured["coordinate_condition_mask"]),
            groups=None if captured["groups"] is None else {k: tile(v) for k, v in captured["groups"].items()},
        )
    rows = mask.shape[0]
    assert len(slots) == config.beam_width
    for replica, slot in enumerate(slots):
        assert not torch.equal(slot["bb_ca"], slots[0]["bb_ca"]) or replica == 0, "replicas must not repeat a draw"
    same = [
        all(torch.equal(slot[name], reference[name][replica * rows : (replica + 1) * rows]) for name in slot)
        for replica, slot in enumerate(slots)
    ]
    assert all(same) is matched


def test_rfproteina_search_top_k_is_per_request() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(2, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)

    def evaluator(prediction, _features):
        batch = prediction.coordinates.shape[0]
        return [{"score": float(index + 1)} for index in range(batch)]

    config = RFProteinaBeamSearchConfig(
        step_checkpoints=[0, 2, 4],
        n_branch=2,
        beam_width=2,
        rewards=RFProteinaSearchRewardConfig(weights={"score": 1.0}),
    )
    result = model.search(static_features, config, evaluators=[evaluator], seed=3)
    finals = result.final_candidates
    assert len(finals) == 4
    by_request = {0: [], 1: []}
    for candidate in finals:
        by_request[candidate.request_index].append(candidate)
        assert candidate.prediction.coordinates.shape[0] == 1
    assert len(by_request[0]) == 2
    assert len(by_request[1]) == 2
    assert by_request[0][0].rank == 0
    assert by_request[0][0].reward >= by_request[0][1].reward
    assert by_request[1][0].request_index == 1


def test_rfproteina_search_missing_metric_raises() -> None:
    model = _tiny_model(num_steps=2)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    config = RFProteinaBeamSearchConfig(
        step_checkpoints=[0, 2],
        n_branch=2,
        beam_width=2,
        rewards=RFProteinaSearchRewardConfig(weights={"not_a_metric": 1.0}),
    )
    with pytest.raises(ValueError, match="-inf"):
        model.search(static_features, config, seed=0)


def test_rfproteina_search_keeps_threshold_intermediates_once() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    result = model.search(static_features, _clash_search_config(4, intermediate_save_min_reward=float("-inf")), seed=1)
    finals = result.final_candidates
    intermediates = result.intermediate_candidates
    assert intermediates
    final_ids = {candidate.candidate_id for candidate in finals}
    assert final_ids.isdisjoint({candidate.candidate_id for candidate in intermediates})
    assert all(candidate.is_intermediate for candidate in intermediates)


def test_rfproteina_search_last_checkpoint_does_not_add_scoring_steps() -> None:
    model = _tiny_model(num_steps=2)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    result = model.search(
        static_features, _clash_search_config(2, step_checkpoints=[0, 2], n_branch=2, beam_width=2), seed=2
    )
    assert result.cost.completion_rollouts == 4
    assert result.cost.velocity_steps == 8


def test_rfproteina_search_partial_redesign_and_inverse_fold() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    legacy_residue_type = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda")
    atom_mask = get_residue_atom_mask(legacy_residue_type, 20, mask)
    coordinates = torch.randn(1, 6, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    static_features.update(
        {
            "residue_type": legacy_residue_type + 1,
            "atom37_mask": atom_mask,
            "can_be_encoded": mask.clone(),
            "coords_nm": coordinates,
        }
    )
    redesigned = model.search(
        static_features,
        _clash_search_config(4),
        partial_config=RFProteinaPartialSimulationConfig(partial_t_backbone=0.2, partial_t_latent=0.2),
        coordinates=coordinates,
        seed=4,
    )
    assert redesigned.final_candidates
    assert redesigned.final_candidates[0].prediction.partial_redesign is not None
    folded = model.search(
        static_features,
        _clash_search_config(4, step_checkpoints=[0, 4], n_branch=1, beam_width=2),
        partial_config=RFProteinaPartialSimulationConfig(inverse_folding=True),
        coordinates=coordinates,
        seed=4,
    )
    backbones = [candidate.prediction.sampling.state["bb_ca"] for candidate in folded.final_candidates]
    assert len(backbones) == 2
    torch.testing.assert_close(backbones[0], backbones[1])


def test_rfproteina_search_by_fraction_integrates_the_tail_of_the_configured_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    legacy_residue_type = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda")
    atom_mask = get_residue_atom_mask(legacy_residue_type, 20, mask)
    coordinates = torch.randn(1, 6, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    static_features.update(
        {
            "residue_type": legacy_residue_type + 1,
            "atom37_mask": atom_mask,
            "can_be_encoded": mask.clone(),
            "coords_nm": coordinates,
        }
    )
    for modality_config in (model.config.sampling.backbone, model.config.sampling.local_latents):
        modality_config.schedule.tail_steps = 2
        modality_config.schedule.tail_step_multiplier = 2
    prepared = []
    prepare = model.sampler.prepare_partial_rollout

    def record(*args, **kwargs):
        prepared.append(kwargs.get("schedules"))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(model.sampler, "prepare_partial_rollout", record)

    result = model.search(
        static_features,
        _clash_search_config(2),
        partial_config=RFProteinaPartialSimulationConfig(partial_trajectory_fraction_steps=0.5),
        coordinates=coordinates,
        seed=4,
    )

    assert result.final_candidates
    assert len(prepared) == 2 and all(schedules is not None for schedules in prepared)
    for modality, modality_config in (
        ("bb_ca", model.config.sampling.backbone),
        ("local_latents", model.config.sampling.local_latents),
    ):
        full = create_modality_schedule(modality_config, 4, torch.device("cuda"))
        torch.testing.assert_close(prepared[0][modality].times, full.times[2:], rtol=0, atol=0)


def test_rfproteina_local_metrics_on_conditioned_batch() -> None:
    model = _tiny_model(num_steps=2)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    atom_mask = torch.zeros(1, 6, 37, dtype=torch.bool, device="cuda")
    atom_mask[:, :, 1] = True
    coordinates = torch.zeros(1, 6, 37, 3, device="cuda")
    coordinates[0, :, 1, 0] = torch.arange(6, device="cuda", dtype=torch.float32)
    static_features["atom37_mask"] = atom_mask
    static_features["mask_coordinate_1_atom"] = atom_mask
    static_features["condition_coordinate_1_atom"] = coordinates * 10.0
    seq_mask = torch.zeros_like(mask)
    seq_mask[0, :3] = True
    static_features["mask_sequence_1_residue"] = seq_mask
    static_features["condition_sequence_1_residue"] = torch.ones(1, 6, dtype=torch.long, device="cuda")
    prediction = model(static_features, seed=0)
    metrics = RFProteinaLocalMetricEvaluator()(prediction, static_features)
    assert CLASH_METRIC_KEY in metrics[0]
    assert MOTIF_RMSD_METRIC_KEY in metrics[0]
    assert SEQUENCE_ADHERENCE_METRIC_KEY in metrics[0]
    assert metrics[0][CLASH_METRIC_KEY] >= 0.0


def test_rfproteina_search_scoring_resets_self_conditioning() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    flags: list[bool] = []
    single = torch.randn(1, 6, model.config.trunk.token_dim, device="cuda")
    pair = torch.randn(1, 6, 6, model.config.trunk.pair_dim, device="cuda")
    condition = torch.randn(1, 6, model.config.trunk.condition_dim, device="cuda")

    def build(_state, _times, _self_conditioning, active):
        flags.append(active)
        return RFProteinaRepresentations(single=single, pair=pair, condition=condition)

    model.search(
        static_features,
        _clash_search_config(4, n_branch=1, beam_width=1),
        representation_builder=build,
        seed=0,
    )
    assert flags[:2] == [False, True]
    assert False in flags[2:]


def test_rfproteina_search_passes_checkpoint_metric_context() -> None:
    model = _tiny_model(num_steps=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    contexts: list[MetricEvaluationContext] = []

    class ContextEvaluator:
        def evaluate(self, prediction, _features, context):
            contexts.append(context)
            return [{"score": 1.0} for _ in range(prediction.final_coordinates.shape[0])]

    config = RFProteinaBeamSearchConfig(
        step_checkpoints=[0, 2, 4],
        n_branch=1,
        beam_width=1,
        rewards=RFProteinaSearchRewardConfig(
            weights={"score": 1.0},
            reward_metrics_tags=["early"],
            checkpoint_overrides={4: {"reward_metrics_tags": ["final"]}},
        ),
    )
    model.search(static_features, config, evaluators=[ContextEvaluator()], seed=0)
    assert [context.checkpoint for context in contexts] == [2, 4]
    assert [context.tags for context in contexts] == [("early",), ("final",)]
    assert all(len(context.request_ids) == 1 for context in contexts)


def test_rfproteina_search_runs_custom_oracle_provider() -> None:
    model = _tiny_model(num_steps=2)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    seen_sequences: list[str] = []

    def oracle(requests):
        seen_sequences.extend(request.structure.sequence for request in requests)
        return [
            OracleResponse(
                request.request_id,
                OracleStatus.OK,
                {"custom_oracle.sequence_length": float(len(request.structure.sequence))},
            )
            for request in requests
        ]

    evaluator = OracleMetricEvaluator(
        CallableOracleProvider("custom", oracle),
        RFProteinaOracleRequestAdapter(
            CallableResidueTypeResolver(
                lambda structure, token: ResidueIdentity(
                    int(structure.residue_type_indices[token]),
                    ResTypes.X,
                    "X",
                    PolymerType.PROTEIN,
                )
            ),
            CallableAtomTypeResolver(
                lambda structure, token, slot, residue: AtomIdentity(
                    int(structure.atom_type_indices[token, slot]),
                    AtomType(name=f"ATOM{int(structure.atom_type_indices[token, slot])}"),
                )
            ),
        ),
        required_tags_any={"custom_oracle"},
    )
    result = model.search(
        static_features,
        RFProteinaBeamSearchConfig(
            step_checkpoints=[0, 2],
            n_branch=1,
            beam_width=1,
            rewards=RFProteinaSearchRewardConfig(
                weights={"custom_oracle.sequence_length": 1.0},
                reward_metrics_tags=["custom_oracle"],
            ),
        ),
        evaluators=[RFProteinaLocalMetricEvaluator(), evaluator],
        seed=0,
    )
    assert seen_sequences == ["XXXXXX"]
    assert result.final_candidates[0].metrics["custom_oracle.sequence_length"] == 6.0
    assert CLASH_METRIC_KEY in result.final_candidates[0].metrics


@pytest.mark.skipif(
    os.environ.get("BIOIR_TEST_RFPROTEINA_SEARCH_PARITY") != "1",
    reason="set BIOIR_TEST_RFPROTEINA_SEARCH_PARITY=1 to run released-checkpoint search smoke",
)
def test_rfproteina_released_checkpoint_search_smoke() -> None:
    checkpoint = resolve_rfproteina_checkpoint()
    autoencoder = resolve_rfproteina_autoencoder_checkpoint()
    if checkpoint is None or autoencoder is None:
        pytest.skip("released RFProteína checkpoints are not available")
    sampling = _deterministic_sampling_config(num_steps=2)
    # The released flow checkpoint carries unmerged LoRA adapters; this entry point folds them and sizes the model.
    model = RFProteina.from_reference_state_dicts(
        load_rfproteina_checkpoint_state(checkpoint),
        load_rfproteina_checkpoint_state(autoencoder),
        sampling_config=sampling,
    )
    model.sampler = type(model.sampler)(sampling, latent_dim=model.config.trunk.latent_dim)
    model = model.cuda().eval()
    mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    # The released network reads per-residue sequence probabilities; NaN leaves a residue unspecified.
    for token in SEQUENCE_PROBABILITY_TOKENS:
        key = f"condition_sequence-probability-{token}_1_residue"
        static_features[key] = torch.full(mask.shape, torch.nan, device=mask.device)
    result = model.search(
        static_features,
        _clash_search_config(2, step_checkpoints=[0, 2], n_branch=1, beam_width=1),
        seed=0,
    )
    assert result.final_candidates
    assert all(candidate.reward == candidate.reward for candidate in result.final_candidates)
