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
"""Boltz2 graph-cache limits land on copies of the region policies."""

import pytest

from bionemo_ir._torch.graph_optimization.config import InputKeyMethod
from bionemo_ir.models.boltz2.config import Boltz2Config, Boltz2GraphCacheConfig

_CACHE = Boltz2GraphCacheConfig(max_tokens=2048, max_graphs=4, budget_bytes=16 << 30)


def _config() -> Boltz2Config:
    return Boltz2Config(min_dist=2.0, max_dist=22.0)


def test_cache_limits_land_on_copies_of_region_policies() -> None:
    original = _config()
    before = original.model_dump()
    configured = original.with_graph_cache(_CACHE)
    assert original.model_dump() == before
    for owner in (configured.trunk, configured.structure_module.score_model, configured.confidence_module):
        policy = owner.graph_optimization_config
        assert policy.num_graphs_max_for_this_module == 4
        assert policy.graph_cache_budget_bytes == 16 << 30
        limits = [(dim.name, dim.dim_len_max) for dim in policy.input_routing_config.input_acceptance_dims]
        assert limits == [("num_tokens", 2048)]
    original.confidence_module.graph_optimization_config = None
    assert original.with_graph_cache(_CACHE).confidence_module.graph_optimization_config is None


def test_cache_rejects_inexact_routing() -> None:
    config = _config()
    config.trunk.graph_optimization_config.input_key_method = InputKeyMethod.BUCKETED_SHAPES
    with pytest.raises(ValueError, match="exact unpadded routing"):
        config.with_graph_cache(_CACHE)
