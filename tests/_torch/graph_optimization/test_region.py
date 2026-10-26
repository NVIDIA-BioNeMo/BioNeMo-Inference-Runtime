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
"""Graph regions with a policy capture their owner's computation by default."""

import copy
import gc
import inspect
import pickle
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bionemo_ir._torch.graph_optimization import GraphRegion, eager_graphs, graph_policy
from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputRoutingConfig,
    NamedDimTies,
)
from bionemo_ir._torch.graph_optimization.graph_policy import _release_unused_graphs, exact_graph_config
from bionemo_ir._torch.graph_optimization.region import _validate_routing_against_signature
from bionemo_ir.configs import AcceleratedConfig, BaseConfig
from bionemo_ir.models.openfold2.modeling import OpenFold2
from bionemo_ir.models.optimize_module_setter import OptimizedModuleSetterMixin


def _policy(max_tokens: int):
    return exact_graph_config(
        named_dims=(NamedDimTies(name="num_tokens", input_dims=(("x", (-2,)),)),),
        max_tokens=max_tokens,
        repeated=True,
    )


class _Model(nn.Module, OptimizedModuleSetterMixin):
    GRAPH_REGIONS = {"region": "graph"}

    def __init__(self, max_tokens: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, 5, dtype=torch.float32))
        self.graph = GraphRegion(self, "step", _policy(max_tokens))

    def step(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return x * self.weight * scale

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return self.graph(x, scale=scale)


class _Parent(_Model):
    def __init__(self) -> None:
        super().__init__(max_tokens=4)
        self.child = _Model(max_tokens=8)

    def step(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return self.child(x, scale=scale) + self.weight


def test_repeated_regions_verify_only_under_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    script = "from bionemo_ir._torch.graph_optimization import graph_policy as p; print(p._VERIFY_REPEATED_CAPTURES)"
    served = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert served.stdout.strip() == "False"
    assert _policy(8).verify_capture
    monkeypatch.setattr(graph_policy, "_VERIFY_REPEATED_CAPTURES", False)
    assert not _policy(8).verify_capture
    monkeypatch.setattr(graph_policy, "_VERIFY_REPEATED_CAPTURES", True)
    single = exact_graph_config(named_dims=(NamedDimTies(name="num_tokens", input_dims=(("x", (-2,)),)),), max_tokens=8)
    assert not single.verify_capture


def test_regions_capture_by_default_and_optimize_enables_selected_ones() -> None:
    model = _Model()
    model.plain = GraphRegion(model, "step", None)
    assert model.graph.enabled and not model.plain.enabled
    model.graph.enabled = False
    model.optimize({})
    assert not model.graph.enabled
    assert model.optimize() is model
    assert model.graph.enabled
    assert not model.plain.enabled
    for key in ("region", "graph"):
        model, override = _Model(), _policy(4)
        model.optimize(
            {key: AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config=override))}
        )
        assert model.graph.enabled
        assert model.graph.policy is override
    with pytest.raises(ValueError, match="torch backend"):
        _Model().optimize({"region": AcceleratedConfig(backend="tensorrt")})
    model = _Model()
    model.graph.enabled = False
    with pytest.raises(ValueError, match=r"No graph region named \['regoin'\]"):
        model.optimize({"region": AcceleratedConfig(backend="torch"), "regoin": AcceleratedConfig(backend="torch")})
    # A misspelled keyword is an error, not a call that re-enables every region.
    with pytest.raises(TypeError, match="accelerated_config"):
        model.optimize(accelerated_config={"region": AcceleratedConfig(backend="torch")})
    # A region that would be left without a policy rejects the whole request up front.
    model.plain = GraphRegion(model, "step", None)
    override = _policy(4)
    with pytest.raises(ValueError, match="'plain' has no CUDA-graph policy"):
        model.optimize(
            {
                "region": AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config=override)),
                "plain": AcceleratedConfig(backend="torch"),
            }
        )
    assert model.graph.policy is not override
    # A rejected request enables nothing.
    assert not model.graph.enabled


def test_optimize_converts_mapping_policies_and_rejects_other_values() -> None:
    model = _Model()
    mapping = _policy(4).model_dump()
    model.optimize(
        {"region": AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config=mapping))}
    )
    assert isinstance(model.graph.policy, CUDAGraphOptimizationConfig)
    assert model.graph.policy == _policy(4)
    policy = model.graph.policy
    model.graph.enabled = False
    with pytest.raises(TypeError, match="needs a CUDAGraphOptimizationConfig policy, got str"):
        model.optimize(
            {"region": AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config="x"))}
        )
    assert model.graph.policy is policy
    assert not model.graph.enabled


def test_cpu_inputs_and_autograd_run_eagerly() -> None:
    model = _Model().eval().optimize()
    x = torch.ones(1, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model.step(x))
    assert model.graph.tracker is None
    if torch.cuda.is_available():
        model.cuda()
        x = x.cuda().requires_grad_()
        model(x).sum().backward()
        torch.testing.assert_close(x.grad, model.weight.detach().expand_as(x))
        assert not model.graph.tracker.execution_counts


@torch.inference_mode()
def test_eager_scopes_and_parent_capture_run_child_regions_eagerly() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _Parent().cuda().eval().optimize()
    x = torch.ones(1, 4, 4, device="cuda")
    with eager_graphs(), eager_graphs(False):
        expected = model(x)
    with pytest.raises(RuntimeError, match="leave scope"), eager_graphs():
        raise RuntimeError("leave scope")
    parent, child = model.graph.tracker.execution_counts, model.child.graph.tracker.execution_counts
    assert not parent and not child
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert parent["capture"] == 1
    assert not child
    # The parent rejects five tokens and runs eagerly, so the child captures its own graph.
    result = model(torch.ones(1, 5, 4, device="cuda"))
    assert parent["out_of_range"] == 1
    assert child["capture"] == 1
    torch.testing.assert_close(result, torch.full_like(result, 2) * model.weight, rtol=0, atol=0)


@pytest.fixture
def cuda_model() -> _Model:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return _Model().cuda().eval().optimize()


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({"named_dims": (NamedDimTies(name="num_tokens", input_dims=(("y", (-2,)),)),)}, "input-tie tensor 'y'"),
        ({"named_dims": (), "stable_kwargs": ("missing",)}, "stable kwarg 'missing'"),
    ],
)
def test_routed_names_must_be_parameters_of_the_region_target(cuda_model: _Model, routing, message) -> None:
    cuda_model.graph.enable(exact_graph_config(max_tokens=8, **routing))
    with pytest.raises(ValueError, match=f"{message} is not a parameter of forward"):
        cuda_model(torch.ones(1, 4, 4, device="cuda"))


@pytest.mark.parametrize(
    ("target", "name", "reaches"),
    [
        (lambda x, *rest: x, "arg0", False),  # position 0 routes as ``x``
        (lambda x, *rest: x, "arg1", True),  # the first ``*rest`` element
        (lambda x, y: x, "arg1", False),  # no ``*args`` takes a position past ``y``
        (lambda x, **kwargs: x, "arg0", True),  # a keyword named ``arg0``
    ],
)
def test_positional_routing_names_reach_only_star_args_positions(target, name, reaches) -> None:
    routing = InputRoutingConfig(named_dim_ties=[NamedDimTies(name="num_tokens", input_dims=((name, (-1,)),))])
    if reaches:
        _validate_routing_against_signature(inspect.signature(target), routing, "GraphRegion")
    else:
        with pytest.raises(ValueError, match=f"'{name}' is not a parameter of forward"):
            _validate_routing_against_signature(inspect.signature(target), routing, "GraphRegion")


@torch.inference_mode()
def test_first_call_captures_then_replays_new_values(cuda_model: _Model) -> None:
    x = torch.ones(1, 4, 4, device="cuda")
    first = cuda_model(x)
    expected_first = first.clone()
    counts = cuda_model.graph.tracker.execution_counts
    assert (counts["capture"], counts["replay"]) == (1, 0)
    second = cuda_model(x + 3)
    torch.testing.assert_close(second, cuda_model.step(x + 3), rtol=0, atol=0)
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
    assert (counts["capture"], counts["replay"]) == (1, 1)


@torch.inference_mode()
def test_scalar_values_key_graphs_and_long_inputs_run_eagerly(cuda_model: _Model) -> None:
    x = torch.ones(1, 4, 4, device="cuda")
    for scale in (1.0, 2.0):
        torch.testing.assert_close(cuda_model(x, scale=scale), cuda_model.step(x, scale=scale), rtol=0, atol=0)
    counts = cuda_model.graph.tracker.execution_counts
    assert counts["capture"] == 2
    large = torch.ones(1, 9, 4, device="cuda")
    torch.testing.assert_close(cuda_model(large), cuda_model.step(large), rtol=0, atol=0)
    assert counts["out_of_range"] == 1
    assert counts["capture"] == 2


def test_weight_loading_and_dtype_changes_drop_graphs(cuda_model: _Model) -> None:
    x = torch.ones(1, 4, 4, device="cuda")
    with torch.inference_mode():
        cuda_model(x)
    counts = cuda_model.graph.tracker.execution_counts
    # Assigning replaces the parameter storage the graph captured.
    cuda_model.load_state_dict({"weight": torch.full((4,), 7.0, device="cuda")}, assign=True)
    with torch.inference_mode():
        torch.testing.assert_close(cuda_model(x), cuda_model.step(x), rtol=0, atol=0)
    assert (counts["capture"], counts["replay"]) == (2, 0)
    cuda_model.double().float()
    with torch.inference_mode():
        torch.testing.assert_close(cuda_model(x), cuda_model.step(x), rtol=0, atol=0)
    assert (counts["capture"], counts["replay"]) == (3, 0)
    assert list(cuda_model.state_dict()) == ["weight"]


@torch.inference_mode()
def test_release_unused_graphs_keeps_in_range_caches() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    short = _Model(max_tokens=4).cuda().eval().optimize()
    longer = _Model(max_tokens=8).cuda().eval().optimize()
    x = torch.ones(1, 4, 4, device="cuda")
    held = short(x)
    longer(x)
    _release_unused_graphs((short.graph, longer.graph), num_tokens=5)
    short_counts, longer_counts = short.graph.tracker.execution_counts, longer.graph.tracker.execution_counts
    assert (short_counts["eviction"], longer_counts["eviction"]) == (1, 0)
    torch.testing.assert_close(held, short.step(x), rtol=0, atol=0)
    torch.testing.assert_close(longer(x), longer.step(x), rtol=0, atol=0)
    assert longer_counts["replay"] == 1
    torch.testing.assert_close(short(x), short.step(x), rtol=0, atol=0)
    assert short_counts["capture"] == 2


@pytest.mark.parametrize("num_tokens", [8, 9])
@torch.inference_mode()
def test_openfold2_releases_graphs_before_recycle_allocation(
    cuda_model: _Model, monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    cuda_model(torch.ones(1, 4, 4, device="cuda"))
    tracker = cuda_model.graph.tracker
    tracker.fallback_to_eager_by_key["failed"] = True
    probe = SimpleNamespace(
        parameters=cuda_model.parameters,
        trunk_graph=cuda_model.graph,
        config=SimpleNamespace(input_embedder=SimpleNamespace(c_m=4, torch_dtype=torch.float32)),
    )
    feed_dict = {
        "target_feat": torch.empty(num_tokens, 1, 1),
        "msa_feat": torch.empty(1, num_tokens, 1, 1),
        "aatype": torch.empty(num_tokens, 1),
    }

    def stop_allocation(*args: object, **kwargs: object) -> None:
        raise RuntimeError("recycle state allocation")

    # Abort at the first state allocation so eviction must happen before it.
    monkeypatch.setattr(torch, "zeros", stop_allocation)
    with pytest.raises(RuntimeError, match="recycle state allocation"):
        OpenFold2.forward(probe, feed_dict)

    assert len(tracker.graph_state_by_key) == int(num_tokens <= 8)
    assert tracker.execution_counts["eviction"] == int(num_tokens > 8)
    assert tracker.fallback_to_eager_by_key == {"failed": True}


def test_copied_and_unpickled_models_call_their_own_owner() -> None:
    model = _Model()
    for copied in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        with torch.no_grad():
            copied.weight.fill_(10.0)
        assert torch.equal(copied(torch.ones(4)), torch.full((4,), 10.0))
    assert torch.equal(model(torch.ones(4)), torch.arange(1, 5, dtype=torch.float32))


@torch.inference_mode()
def test_dropped_model_frees_its_graphs_without_garbage_collection() -> None:
    # A graph freed by a later collection could land inside another capture and invalidate it.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _Model().cuda().eval()
    model(torch.ones(1, 4, 4, device="cuda"))
    tracker = weakref.ref(model.graph.tracker)
    gc.disable()
    try:
        del model
        assert tracker() is None
    finally:
        gc.enable()


def test_region_target_must_be_a_callable_attribute_of_its_owner() -> None:
    with pytest.raises(AttributeError, match="no callable 'missing'"):
        GraphRegion(_Model(), "missing", None)
    with pytest.raises(AttributeError, match="no callable 'weight'"):
        GraphRegion(_Model(), "weight", None)
