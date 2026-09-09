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
"""

import pytest

from bionemo_ir._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer
from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir._torch.modules.openfold3.diffusion_module import DiffusionModule as OF3DiffusionModule
from bionemo_ir.configs import AcceleratedConfig, BackendType
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
        "diffusion_module": ("diffusion_sampler.diffusion_module", OF3DiffusionModule),
    }
    for role, (path, cls) in expected.items():
        assert reg._name_to_path[role] == path
        assert isinstance(of3_model.get_submodule(path), cls)


def test_parent_child_conflict_drops_child(of3_model):
    reg = of3_model.get_optimized_modules(
        {
            "diffusion_module": _torch_cfg(),  # parent
            "token_transformer": _torch_cfg(),  # child (nested inside it)
        }
    )
    names = reg.get_module_names()
    assert "diffusion_module" in names
    assert "token_transformer" not in names


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
