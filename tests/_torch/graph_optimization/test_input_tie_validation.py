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
"""Item 1.2.4/1.2.5: the tracker validates input tie points against the first
representative call — a tie to an absent tensor or out-of-range axis is a
configuration error, not a silently-ignored rule."""
import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig, InputKeyMethod, InputRoutingConfigFactory,
    NamedDimTies)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import \
    CUDAGraphOptimizationTracker


def _tracker(factory: InputRoutingConfigFactory) -> CUDAGraphOptimizationTracker:
    cfg = CUDAGraphOptimizationConfig(
        input_key_method=InputKeyMethod.BUCKETED_SHAPES,
        input_routing_config=factory.export_config())
    return CUDAGraphOptimizationTracker(cfg, inner_module=nn.Identity())


def _acceptance_factory() -> InputRoutingConfigFactory:
    f = InputRoutingConfigFactory()
    f.set_named_dim_ties([
        NamedDimTies(name="num_tokens", input_dims=(("s", (-2,)),)),
    ])
    f.set_input_acceptance_dim("num_tokens", 100)
    return f


def _shape(*dims: int) -> torch.Tensor:
    return torch.tensor(dims, dtype=torch.int32)


def test_valid_ties_pass_and_run_once():
    tracker = _tracker(_acceptance_factory())
    tracker.validate_input_ties({"s_shape": _shape(1, 50, 64)})  # 3-D, axis -2 ok
    assert tracker._input_ties_validated is True
    # Guarded: a later bad call is not re-validated (hot-path one-shot).
    tracker.validate_input_ties({"x_shape": _shape(1, 2)})  # would fail if checked


def test_tie_to_absent_tensor_raises():
    tracker = _tracker(_acceptance_factory())
    with pytest.raises(ValueError, match="no such tensor is present"):
        tracker.validate_input_ties({"x_shape": _shape(1, 50, 64)})  # no s_shape


def test_tie_to_out_of_range_axis_raises():
    tracker = _tracker(_acceptance_factory())
    with pytest.raises(ValueError, match="only 1 dimension"):
        tracker.validate_input_ties({"s_shape": _shape(50)})  # 1-D, axis -2 invalid


def test_padded_ties_are_also_validated():
    f = InputRoutingConfigFactory()
    f.set_named_dim_ties([
        NamedDimTies(name="num_tokens", input_dims=(("z", (-3,)),)),
    ])
    f.set_padded_dim("num_tokens", dim_len_min=4, dim_len_max=16,
                     num_intervals=4, multiple_of=1)
    tracker = _tracker(f)
    with pytest.raises(ValueError, match="no such tensor is present"):
        tracker.validate_input_ties({"s_shape": _shape(1, 8, 8, 32)})  # no z_shape
