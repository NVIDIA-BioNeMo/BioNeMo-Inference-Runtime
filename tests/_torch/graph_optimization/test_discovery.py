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
"""Generic ``named_modules()`` discovery replaces per-model registries.

Constructs OpenFold3 without weights (cheap, no GPU) and asserts the
``DiscoveredModuleRegistry`` behaves like the hand-written registry it replaced:

* each role alias resolves to the expected decorated module *on the
  forward-reachable path* (not a dedup-first duplicate path),
* a configured child is dropped when its parent is also configured
  (parent/child conflict via qualified-path prefix),
* unconfigured modules are never selected, and a qualified-path key works
  directly.

Each folding model's graph regions capture by default, without changing its
weights or config, and ``disable_cuda_graphs()`` builds them eager.
"""

import pytest
import torch.nn as nn

from bionemo_ir._torch.graph_optimization import GraphRegion
from bionemo_ir._torch.graph_optimization.config import GraphOptimizationMode, InputKeyMethod, NamedDimTies
from bionemo_ir._torch.graph_optimization.decorator import support_graph_optimization
from bionemo_ir._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer
from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir.configs import AcceleratedConfig, BackendType, BaseConfig
from bionemo_ir.models.optimize_module_setter import DiscoveredModuleRegistry
from bionemo_ir.registry import get_model_class


@pytest.fixture(scope="module")
def of3_model():
    cls = get_model_class("openfold3")
    return cls(config=cls.get_pretrained_config("openfold3"), include_load_weights=False)


def _torch_cfg():
    return AcceleratedConfig(backend=BackendType.TORCH)


def test_role_aliases_resolve_to_expected_modules(of3_model):
    reg = of3_model.get_optimized_modules({})
    reg.get_accelerated_modules()  # populates name->path
    expected = {
        "structure_pairformer": ("pairformer_stack", PairformerModule),
        "token_transformer": (
            "diffusion_sampler.diffusion_module.diffusion_transformer",
            OpenFold3DiffusionTransformer,
        ),
    }
    for role, (path, cls) in expected.items():
        assert reg._name_to_path[role] == path
        assert isinstance(of3_model.get_submodule(path), cls)


def test_parent_child_conflict_drops_child():
    @support_graph_optimization(
        named_dims=(NamedDimTies(name="num_tokens", input_dims=(("x", (-1,)),)),),
        graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
        input_key_method=InputKeyMethod.EXACT,
    )
    class Block(nn.Module):
        def __init__(self, child: nn.Module | None = None) -> None:
            super().__init__()
            self.child = child

        def forward(self, x):
            return x if self.child is None else self.child(x)

    reg = DiscoveredModuleRegistry(nn.Sequential(Block(Block())), {"0": _torch_cfg(), "0.child": _torch_cfg()})
    assert reg.get_module_names() == ["0"]


def test_only_configured_modules_selected(of3_model):
    reg = of3_model.get_optimized_modules({"structure_pairformer": _torch_cfg()})
    assert reg.get_module_names() == ["structure_pairformer"]


def test_qualified_path_key_resolves_directly(of3_model):
    # A caller may key by qualified path instead of a role alias.
    reg = of3_model.get_optimized_modules({"pairformer_stack": _torch_cfg()})
    assert "pairformer_stack" in reg.get_module_names()


def test_alias_and_canonical_path_deduplicated(of3_model):
    """Configuring both a role alias and its canonical qualified path
    targets the same submodule twice; the duplicate is dropped so it is not
    wrapped twice."""
    reg = of3_model.get_optimized_modules(
        {
            "structure_pairformer": _torch_cfg(),  # alias -> pairformer_stack
            "pairformer_stack": _torch_cfg(),  # canonical path (same module)
        }
    )
    names = reg.get_module_names()
    assert len(names) == 1
    assert reg._name_to_path[names[0]] == "pairformer_stack"


def test_missing_configured_path_is_config_error(of3_model):
    """A configured key with no decorated target raises, not silently
    skips."""
    with pytest.raises(ValueError, match="no decorated, discoverable target"):
        of3_model.get_optimized_modules({"not.a.real.module": _torch_cfg()})
    # A real but *undecorated* path is likewise rejected (it is not a candidate).
    with pytest.raises(ValueError, match="no decorated, discoverable target"):
        of3_model.get_optimized_modules({"input_embedder": _torch_cfg()})


def test_optimize_falls_back_to_decorator_default():
    """When the model config supplies no ``graph_optimization_config``, the
    tracker is built from the module's decorator-declared ``graph_opt_default``
    (set by ``@support_graph_optimization``) — the identical config object.

    Uses a fresh model because ``optimize`` swaps submodules in place.
    """
    from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker

    cls = get_model_class("openfold3")
    model = cls(config=cls.get_pretrained_config("openfold3"), include_load_weights=False)
    # AcceleratedConfig(backend=TORCH).default is None -> no explicit config.
    model.optimize({"structure_pairformer": _torch_cfg()})

    wrapped = model.get_submodule("pairformer_stack")
    assert isinstance(wrapped, CUDAGraphOptimizationTracker)
    assert wrapped.graph_optimization_config is PairformerModule.graph_opt_default


def test_explicit_policy_overrides_region_policy():
    model = get_model_class("openfold3")(include_load_weights=False).eval()
    policy = model.config.trunk.graph_optimization_config
    override = policy.model_copy(deep=True)
    model.optimize(
        {"trunk": AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config=override))}
    )
    assert model.trunk_graph.enabled
    assert model.trunk_graph.policy is override
    assert model.config.trunk.graph_optimization_config is policy


def test_decorated_child_remains_reachable_inside_enabled_region():
    from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker

    model = get_model_class("openfold3")(include_load_weights=False).eval()
    owner = model.diffusion_sampler.diffusion_module
    model.optimize({"diffusion_module": _torch_cfg(), "token_transformer": _torch_cfg()})
    assert model.diffusion_sampler.graph.enabled
    assert model.diffusion_sampler.diffusion_module is owner
    assert isinstance(owner.diffusion_transformer, CUDAGraphOptimizationTracker)


@pytest.mark.parametrize(
    "model_name", ["boltz-1", "boltz-2", "boltz-2-affinity", "openfold3", "protenix-v2", "alphafold2_1"]
)
def test_graph_regions_capture_by_default(model_name):
    model = get_model_class(model_name)(model_name=model_name, include_load_weights=False).eval()
    regions = {role: model.get_submodule(path) for role, path in model.GRAPH_REGIONS.items()}
    with_policy = {path for path, m in model.named_modules() if isinstance(m, GraphRegion) and m.policy is not None}
    assert regions and set(model.GRAPH_REGIONS.values()) == with_policy
    assert all(region.enabled and not region.state_dict() for region in regions.values())
    weights = dict(model.named_parameters())
    keys = list(model.state_dict())
    config = model.config.model_dump()
    model.optimize()
    assert all(region.enabled for region in regions.values())
    assert list(model.state_dict()) == keys
    assert all(model.get_parameter(name) is weight for name, weight in weights.items())
    assert model.config.model_dump() == config


def test_disable_cuda_graphs_builds_eager_regions():
    model_class = get_model_class("boltz-2")
    config = model_class.get_pretrained_config("boltz-2")
    config.disable_cuda_graphs()
    model = model_class(config=config, model_name="boltz-2", include_load_weights=False)
    regions = [m for m in model.modules() if isinstance(m, GraphRegion)]
    assert regions and not any(region.enabled or region.policy is not None for region in regions)
