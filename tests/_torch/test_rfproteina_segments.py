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

from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.modules.rfproteina.simulator import RFProteinaProductSpaceSimulator
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir.models.rfproteina import RFProteina, RFProteinaSimulationConfig


class NumericalModel(RFProteina):
    """Exercise the real model boundary and integrator with a known velocity field."""

    def __init__(self, stochastic: bool) -> None:
        torch.nn.Module.__init__(self)
        config = RFProteinaSimulationConfig(self_condition=True)
        for modality in (config.bb_ca, config.local_latents):
            modality.step.sampling_mode = "vf_tsr" if stochastic else "vf"
            modality.step.t_lim_ode_below = 0.0
            modality.step.t_lim_ode = 1.0
        self.config = SimpleNamespace(simulation=config)
        self.simulator = RFProteinaProductSpaceSimulator(config, latent_dim=4)
        self.decoder = SimpleNamespace(num_residue_types=33)
        self.state_dtype = torch.float32
        self.grids = {
            "ts": {key: torch.linspace(0.1, 1.0, 7, device="cuda") for key in ("bb_ca", "local_latents")},
            "gt": {
                key: torch.full((6,), 0.15 if stochastic else 0.0, device="cuda") for key in ("bb_ca", "local_latents")
            },
            "recycle_depths": [1, 1, 2, 2, 3, 3],
        }
        self.depths = []
        self.visible_masks = []

    def _make_predict_fn(self, *, static_features, padding_mask, **kwargs):
        self.visible_masks.append(padding_mask.clone())

        def predict(state, times, clean, active, recycle_depth):
            self.depths.append(recycle_depth)
            output = {key: {"v": 0.1 * value + (0.05 * clean[key] if active else 0.0)} for key, value in state.items()}
            output["sequence"] = {"logits": state["local_latents"].clone()}
            return output

        return predict


def inputs():
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]], device="cuda")
    features = {"padding_mask": mask}
    state = {
        key: torch.randn(2, 4, width, device="cuda") * mask[..., None]
        for key, width in (("bb_ca", 3), ("local_latents", 4))
    }
    return features, state


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("self_condition", [False, True])
def test_segment_resume_matches_uninterrupted_sampling(stochastic: bool, self_condition: bool) -> None:
    model = NumericalModel(stochastic)
    model.config.simulation.self_condition = self_condition
    features, state = inputs()
    original = {key: tensor.clone() for key, tensor in state.items()}
    expected = model.partial_simulation(features, state, **model.grids, start_step=0, end_step=6, seed=42)
    model.depths.clear()
    with SamplingContext.graph_safe(torch.device("cuda"), 42) as context:
        first = model.partial_simulation(features, state, **model.grids, start_step=0, end_step=2, context=context)
        actual = model.partial_simulation(
            features,
            first.state,
            **model.grids,
            start_step=2,
            end_step=6,
            initial_clean=first.clean_prediction if self_condition else None,
            context=context,
        )
    for key in state:
        torch.testing.assert_close(actual.state[key], expected.state[key], rtol=0, atol=0)
        torch.testing.assert_close(actual.clean_prediction[key], expected.clean_prediction[key], rtol=0, atol=0)
        torch.testing.assert_close(state[key], original[key], rtol=0, atol=0)
    assert model.depths == [1, 1, 2, 2, 3, 3]


def test_caller_can_select_rows_and_restart_a_branch() -> None:
    model = NumericalModel(False)
    features, state = inputs()
    first = model.partial_simulation(features, state, **model.grids, start_step=0, end_step=2, seed=42)
    indices = torch.tensor([1, 0, 1], device="cuda")
    selected = {key: value[indices] for key, value in first.state.items()}
    branch_features = {key: value[indices] for key, value in features.items()}
    before = {key: value.clone() for key, value in first.state.items()}
    actual = model.partial_simulation(
        branch_features,
        selected,
        start_step=2,
        end_step=6,
        carry_self_condition=False,
        **{**model.grids, "recycle_depths": [2] * 6},
        seed=84,
    )
    for key in state:
        torch.testing.assert_close(actual.state[key][0], actual.state[key][2])
        torch.testing.assert_close(first.state[key], before[key], rtol=0, atol=0)
    assert model.depths[-4:] == [2, 2, 2, 2]


def test_conditioning_tokens_stay_visible_but_do_not_flow() -> None:
    model = NumericalModel(False)
    features, state = inputs()
    features["flow_backbone_mask"] = features["padding_mask"].clone()
    features["flow_backbone_mask"][:, 1] = False
    result = model.partial_simulation(features, state, **model.grids, start_step=0, end_step=6, seed=42)
    assert model.visible_masks[0][:, 1].all()
    assert not result.state["bb_ca"][:, 1].any()
    assert features["padding_mask"][:, 1].all()


def test_resume_requires_clean_prediction_and_valid_depths() -> None:
    model = NumericalModel(False)
    features, state = inputs()
    with pytest.raises(ValueError, match="initial_clean"):
        model.partial_simulation(features, state, **model.grids, start_step=2, end_step=6)
    with pytest.raises(ValueError, match="recycle_depths"):
        model.partial_simulation(features, state, start_step=0, end_step=6, **{**model.grids, "recycle_depths": [1]})


def test_trajectory_readouts_belong_to_their_own_step() -> None:
    model = NumericalModel(False)
    features, state = inputs()
    result = model.partial_simulation(
        features, state, **model.grids, start_step=0, end_step=6, seed=42, save_trajectory_every=1
    )
    assert len(result.trajectory) == 6
    torch.testing.assert_close(result.trajectory[0]["sequence_logits"], state["local_latents"])
    for previous, frame in zip(result.trajectory, result.trajectory[1:], strict=False):
        torch.testing.assert_close(frame["sequence_logits"], previous["x_t"]["local_latents"])
    torch.testing.assert_close(result.trunk_sequence_logits, result.trajectory[-1]["sequence_logits"])
    assert result.trajectory[0]["sequence_logits"].data_ptr() != state["local_latents"].data_ptr()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_context_matches_native_draw_order_and_commits_stream(device: str) -> None:
    from bionemo_ir._torch.sampling.random import randn, randperm

    device = torch.device(device)
    get_state = torch.get_rng_state if device.type == "cpu" else lambda: torch.cuda.get_rng_state(device)
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(123)
        expected = [torch.randn(2, 4, device=device), torch.randperm(7, device=device), torch.randn(3, device=device)]
        expected_state = get_state()
        torch.manual_seed(123)
        with SamplingContext.resolve(device) as context:
            actual = [randn((2, 4), device=device), randperm(7, device=device)]
            # Reusing the context must neither reseed nor rewind between segments.
            with SamplingContext.resolve(device, context=context, seed=999):
                actual.append(randn((3,), device=device))
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert torch.equal(get_state(), expected_state)


def test_context_isolates_seeded_permutations_and_restores_scope_on_failure() -> None:
    from bionemo_ir._torch.sampling.random import randn, randperm

    device = torch.device("cpu")
    original = torch.get_rng_state()
    expected = torch.Generator().manual_seed(5)
    with SamplingContext.resolve(device, seed=5):
        first = randperm(9, device=device)
        with pytest.raises(RuntimeError, match="prediction failed"):
            with SamplingContext.resolve(device, seed=7):
                randn((4,), device=device)
                raise RuntimeError("prediction failed")
        second = randn((4,), device=device)
    torch.testing.assert_close(first, torch.randperm(9, generator=expected), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.randn(4, generator=expected), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), original)
