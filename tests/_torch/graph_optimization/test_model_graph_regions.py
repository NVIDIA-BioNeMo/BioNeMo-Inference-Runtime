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
"""Folding models' graph regions.

Each model's graph regions capture by default, without changing its weights or
config, and ``disable_cuda_graphs()`` builds them eager. ``optimize()`` keys name
regions only.
"""

import os
from collections.abc import Callable

import pytest
import torch.nn as nn

from bionemo_ir._torch.graph_optimization import GraphRegion
from bionemo_ir.configs import AcceleratedConfig, BackendType, BaseConfig
from bionemo_ir.registry import get_model_class


def _torch_cfg():
    return AcceleratedConfig(backend=BackendType.TORCH)


@pytest.mark.parametrize(
    "key", ["token_transformer", "structure_pairformer", "pairformer_stack", "input_embedder", "not.a.real.module"]
)
def test_optimize_rejects_keys_that_name_no_region(key, structural_model):
    model = structural_model(get_model_class("openfold3"), include_load_weights=False).eval()
    model.trunk_graph.enabled = False
    with pytest.raises(ValueError, match="No graph region named"):
        model.optimize({"trunk": _torch_cfg(), key: _torch_cfg()})
    assert not model.trunk_graph.enabled


def test_explicit_policy_overrides_region_policy(structural_model):
    model = structural_model(get_model_class("openfold3"), include_load_weights=False).eval()
    policy = model.config.trunk.graph_optimization_config
    override = policy.model_copy(deep=True)
    model.optimize(
        {"trunk": AcceleratedConfig(backend="torch", default=BaseConfig(graph_optimization_config=override))}
    )
    assert model.trunk_graph.enabled
    assert model.trunk_graph.policy is override
    assert model.config.trunk.graph_optimization_config is policy


@pytest.mark.parametrize(
    "model_name", ["boltz-1", "boltz-2", "boltz-2-affinity", "openfold3", "protenix-v2", "alphafold2_1"]
)
def test_graph_regions_capture_by_default(model_name, structural_model):
    model = structural_model(get_model_class(model_name), model_name=model_name, include_load_weights=False).eval()
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


def test_disable_cuda_graphs_builds_eager_regions(structural_model):
    model_class = get_model_class("boltz-2")
    config = model_class.get_pretrained_config("boltz-2")
    config.disable_cuda_graphs()
    model = structural_model(model_class, config=config, model_name="boltz-2", include_load_weights=False)
    regions = [m for m in model.modules() if isinstance(m, GraphRegion)]
    assert regions and not any(region.enabled or region.policy is not None for region in regions)


def test_structural_model_preserves_dispatch(structural_model: Callable[..., nn.Module]) -> None:
    from bionemo_ir._torch.custom_ops.attn_epilogue.cutedsl import AttnEpilogueCuTe

    cache = AttnEpilogueCuTe._compiled_cache
    contents = cache.copy()
    mode = os.environ.get("CUTEDSL_FORCE_CUBIN")
    cls = get_model_class("openfold3")
    structural_model(cls, include_load_weights=False)
    assert AttnEpilogueCuTe._compiled_cache is cache
    assert cache == contents
    assert os.environ.get("CUTEDSL_FORCE_CUBIN") == mode
