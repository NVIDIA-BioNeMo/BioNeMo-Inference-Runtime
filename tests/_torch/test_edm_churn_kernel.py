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
"""Verify churn fusion across levels and fallback paths."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bionemo_ir._torch.sampling.edm import edm_churn


def reference(x, a, b, noise, scale):
    return x + scale * torch.sqrt(b.square() - a.square()) * noise


@pytest.mark.parametrize("size", [0, 1, 17, 45000, 120000])
@pytest.mark.parametrize("sigma", [0.0, 0.01, 1.0, 80.0, 2560.0])
def test_churn_exact(size, sigma):
    state, noise = [torch.randn(size, device="cuda") for _ in range(2)]
    levels = torch.tensor([99.0, sigma, sigma * 1.8], device="cuda")
    rng = torch.cuda.get_rng_state()
    for scale in (0.0, 0.1, 1.003, 2.3):
        args = (state, levels[1], levels[2], noise, scale)
        torch.testing.assert_close(edm_churn(*args), reference(*args), atol=0, rtol=0)
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_churn_fallback(dtype, device):
    x, noise = [torch.randn(8, 3, device=device, dtype=dtype).T for _ in range(2)]
    a, b = torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(2.0, device=device, dtype=dtype)
    torch.testing.assert_close(edm_churn(x, a, b, noise, 1.003), reference(x, a, b, noise, 1.003), atol=0, rtol=0)


def test_churn_gradients():
    tensors = [torch.randn(17, device="cuda", requires_grad=True)]
    tensors += [torch.tensor(v, device="cuda", requires_grad=True) for v in (1.0, 2.0)]
    tensors += [torch.randn(17, device="cuda", requires_grad=True)]
    expected = torch.autograd.grad(reference(*tensors, 1.003).sum(), tensors)
    actual = torch.autograd.grad(edm_churn(*tensors, 1.003).sum(), tensors)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("fallback", [False, True], ids=["driver", "triton"])
def test_churn_stream_capture(monkeypatch: pytest.MonkeyPatch, fallback: bool) -> None:
    from bionemo_ir.dsl_kernels.cache_base import DriverLauncher
    from bionemo_ir.dsl_kernels.triton.edm import _churn_kernel

    kernel = _churn_kernel(torch.cuda.current_device()).kernel
    if fallback:
        monkeypatch.setattr(kernel, "_driver", None)
    elif kernel.driver is None:
        pytest.skip("CUDA driver launcher unavailable")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        state, noise = [torch.randn(257, device="cuda") for _ in range(2)]
        levels = torch.tensor([99.0, 1.0, 1.8], device="cuda")
        args = (state, levels[1], levels[2], noise, 1.003)
        with (
            patch.object(type(kernel), "launch", autospec=True, side_effect=type(kernel).launch) as launch,
            patch.object(
                DriverLauncher, "launch_with", autospec=True, side_effect=DriverLauncher.launch_with
            ) as driver_launch,
        ):
            actual = edm_churn(*args)
            if fallback:
                launch.assert_called_once()
                driver_launch.assert_not_called()
            else:
                driver_launch.assert_called_once()
                launch.assert_not_called()
        expected = reference(*args)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = edm_churn(*args)
    for _ in range(3):
        state.normal_()
        noise.normal_()
        levels[1:].mul_(0.5)
        graph.replay()
        torch.testing.assert_close(actual, reference(*args), atol=0, rtol=0)


@pytest.mark.parametrize("index", range(4))
def test_churn_negative_view(index: int) -> None:
    args = [
        torch.randn(17, device="cuda"),
        torch.tensor(1.0, device="cuda"),
        torch.tensor(2.0, device="cuda"),
        torch.randn(17, device="cuda"),
    ]
    args[index] = args[index]._neg_view()
    torch.testing.assert_close(edm_churn(*args, 1.003), reference(*args, 1.003), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("size", [1, 255, 256, 257, 1025])
def test_churn_offsets(offset: int, size: int) -> None:
    state, noise = [torch.randn(size + offset, device="cuda")[offset:] for _ in range(2)]
    levels = torch.tensor([0.0, 1.0, 1.8, 2.0, 3.0, 4.0], device="cuda")
    args = (state, levels[offset], levels[offset + 1], noise, 1.003)
    torch.testing.assert_close(edm_churn(*args), reference(*args), atol=0, rtol=0)


@pytest.mark.parametrize("levels", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (1e-20, 2e-20), (1e20, 2e20)])
def test_churn_extreme_levels(levels: tuple[float, float]) -> None:
    state, noise = [torch.randn(257, device="cuda") for _ in range(2)]
    a, b = torch.tensor(levels, device="cuda")
    for scale in (0.0, 1.003):
        args = (state, a, b, noise, scale)
        torch.testing.assert_close(edm_churn(*args), reference(*args), atol=0, rtol=0, equal_nan=True)


def test_churn_compile() -> None:
    state, noise = [torch.randn(17, device="cuda") for _ in range(2)]
    a, b = torch.tensor([1.0, 1.8], device="cuda")
    compiled = torch.compile(edm_churn, backend="eager", fullgraph=True)
    args = (state, a, b, noise, 1.003)
    torch.testing.assert_close(compiled(*args), reference(*args), atol=0, rtol=0)


def test_churn_broadcast() -> None:
    state = torch.randn(2, 17, 3, device="cuda")
    noise = torch.randn(17, 3, device="cuda")
    a = torch.ones(2, 1, 1, device="cuda")
    args = (state, a, a * 1.8, noise, 1.003)
    torch.testing.assert_close(edm_churn(*args), reference(*args), atol=0, rtol=0)


@pytest.mark.parametrize("final", ["keep", "zero", "append_zero"])
@pytest.mark.parametrize("augment", [False, True])
def test_churn_rollout(monkeypatch: pytest.MonkeyPatch, final: str, augment: bool) -> None:
    import bionemo_ir._torch.sampling as sampling
    import bionemo_ir._torch.sampling.edm as edm

    device = torch.device("cuda", torch.cuda.current_device())
    schedule = sampling.create_edm_schedule(8, 16.0, 160.0, 0.0004, 7.0, device=device, final=final)
    plan = sampling.EDMRolloutPlan(schedule, (2, 17, 3), device, torch.float32, augment_coordinates=augment)
    integrator = sampling.AF3EDMIntegrator(sampling.EDMIntegratorConfig())
    context = sampling.SamplingContext.create(device, seed=123)
    expected_context = sampling.SamplingContext.create(device, seed=123)
    runner = sampling.GenerativeRunner()

    def denoise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x * 0.7 - 0.1

    actual = runner.run(integrator, plan, denoise, context).final_state
    monkeypatch.setattr(edm, "supports_churn", lambda *args: False)
    expected = runner.run(integrator, plan, denoise, expected_context).final_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(context.generator.get_state(), expected_context.generator.get_state())


def test_churn_cold_cache(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    cache_dir = str(tmp_path / "triton")
    env = {**os.environ, "BIOIR_TRITON_CACHE_DIR": cache_dir, "TRITON_CACHE_DIR": cache_dir}
    script = """
from pathlib import Path
import torch
import bionemo_ir.dsl_kernels.triton_cache as cache
from bionemo_ir.dsl_kernels.triton.edm import _ChurnKernel
original = cache._compile_in_subprocess
warmed = set()
def warm(*args, **kwargs):
    original(*args, **kwargs)
    warmed.update(Path(cache.TRITON_CACHE_DIR).rglob('*.cubin'))
cache._compile_in_subprocess = warm
_ChurnKernel()
assert warmed
assert set(Path(cache.TRITON_CACHE_DIR).rglob('*.cubin')) == warmed
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=120, capture_output=True, text=True)
