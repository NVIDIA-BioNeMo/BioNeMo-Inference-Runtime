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
import pytest
import torch

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.modules.protenix import ProtenixDiffusionConditioning, ProtenixDiffusionSampler
from bionemo_ir.configs import BaseConfig
from bionemo_ir.models.protenix.config import DiffusionConditioningConfig, RelativePositionEncodingConfig

B, N, S = 1, 32, 5


@pytest.fixture(scope="module")
def model():
    cfg = DiffusionConditioningConfig(
        c_s=32,
        c_z=32,
        c_s_inputs=16,
        c_noise_embedding=32,
        relpe_config=RelativePositionEncodingConfig(c_z=32),
        dtype="float32",
        z_pair_dtype="float32",
    )
    m = ProtenixDiffusionConditioning(cfg).cuda().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(std=0.1)
    return m


@pytest.fixture(scope="module")
def inputs(model):
    c_s = model.linear_no_bias_n.out_features
    s_in = torch.randn(B, N, model.linear_no_bias_s.in_features - c_s, device="cuda")
    s_tr = torch.randn(B, N, c_s, device="cuda")
    return s_in, s_tr


def test_stride0_dedupe_matches_stock(model, inputs):
    """Stride-0 t_hat computes one sample row and expands it; values match stock."""
    t = torch.tensor(1.5, device="cuda").reshape(B, 1).expand(B, S)
    with torch.inference_mode():
        s_ded = model.forward_single(t, *inputs)
        s_stk = model.forward_single(t.contiguous(), *inputs)
    assert s_ded.shape == s_stk.shape == (B, S, N, model.linear_no_bias_n.out_features)
    assert s_ded.stride(1) == 0
    torch.testing.assert_close(s_ded, s_stk, atol=1e-4, rtol=1e-4)


def test_per_sample_t_uses_stock_path(model, inputs):
    """Distinct per-sample noise levels produce distinct sample rows."""
    t = torch.rand(B, S, device="cuda") * 10 + 0.1
    with torch.inference_mode():
        s = model.forward_single(t, *inputs)
    assert s.stride(1) != 0
    assert not torch.allclose(s[:, 0], s[:, 1])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_singleton_noise_matches_dense_samples(model, inputs, batch_size: int) -> None:
    """Singleton noise preserves distinct batches and matches dense sample noise."""
    s_in, s_tr = (x.repeat(batch_size, 1, 1) for x in inputs)
    t = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.float32).unsqueeze(-1)
    with torch.inference_mode():
        shared = model.forward_single(t, s_in, s_tr)
        dense = model.forward_single(t.expand(batch_size, S).contiguous(), s_in, s_tr)
    assert shared.shape == (batch_size, 1, N, model.linear_no_bias_n.out_features)
    torch.testing.assert_close(shared.expand_as(dense), dense, atol=1e-4, rtol=1e-4)


class _ConditioningDenoiser(torch.nn.Module):
    """Exercise sampler noise routing through the real graph tracker and conditioner."""

    def __init__(self, conditioning: ProtenixDiffusionConditioning) -> None:
        super().__init__()
        self.config = BaseConfig(
            graph_optimization_config=CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH, verify_capture=True
            )
        )
        self.conditioning = conditioning
        self.noise_shapes: list[tuple[int, ...]] = []

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        self.noise_shapes.append(tuple(t_hat_noise_level.shape))
        single = self.conditioning.forward_single(t_hat_noise_level, s_inputs, s_trunk)
        return single.expand(x_noisy.shape[0], x_noisy.shape[1], *single.shape[-2:]).contiguous()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_sampler_shared_noise_survives_graph_capture_and_replay(model, inputs, batch_size: int) -> None:
    """Cloned inputs retain one noise row; replay updates sigma and stays equivalent."""
    denoiser = _ConditioningDenoiser(model).eval()
    sampler = ProtenixDiffusionSampler(denoiser).eval()
    s_in, s_tr = (x.repeat(batch_size, 1, 1) for x in inputs)
    x_noisy = torch.randn(batch_size, S, N, 3, device="cuda")
    kwargs = {
        "batch_shape": (batch_size,),
        "n_sample": S,
        "input_feature_dict": {},
        "s_inputs": s_in,
        "s_trunk": s_tr,
        "z_trunk": torch.empty(batch_size, N, N, 32, device="cuda"),
        "attn_metadata": None,
        "cache": None,
    }
    # Different dtype forces a conversion, which used to lose the stride-0 hint.
    with torch.inference_mode():
        for _ in range(4):
            sampler.denoise(x_noisy, torch.tensor(1.5, device="cuda", dtype=torch.float64), **kwargs)
        tracker = sampler.graph.tracker
        assert tracker is not None
        states = list(tracker.graph_state_by_key.values())
        assert states and all(s.preparation_state is CUDAGraphPreparationState.GRAPH_VERIFIED for s in states)
        assert not any(tracker.fallback_to_eager_by_key.values())
        assert denoiser.noise_shapes and set(denoiser.noise_shapes) == {(batch_size, 1)}
        for sigma in (0.5, 3.0):
            actual = sampler.denoise(x_noisy, torch.tensor(sigma, device="cuda", dtype=torch.float64), **kwargs)
            dense_t = torch.full((batch_size, S), sigma, device="cuda")
            expected = model.forward_single(dense_t, s_in, s_tr)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
