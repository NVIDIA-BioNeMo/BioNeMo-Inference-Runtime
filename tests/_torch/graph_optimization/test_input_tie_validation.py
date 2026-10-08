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
"""Every new signature must resolve tied axes with consistent lengths."""

import pytest
import torch.nn as nn

from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputKeyMethod,
    InputRoutingConfigFactory,
    NamedDimTies,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker


def _tracker(factory: InputRoutingConfigFactory) -> CUDAGraphOptimizationTracker:
    cfg = CUDAGraphOptimizationConfig(
        input_key_method=InputKeyMethod.BUCKETED_SHAPES, input_routing_config=factory.export_config()
    )
    return CUDAGraphOptimizationTracker(cfg, inner_module=nn.Identity())


def _acceptance_factory() -> InputRoutingConfigFactory:
    f = InputRoutingConfigFactory()
    f.set_named_dim_ties(
        [
            NamedDimTies(name="num_tokens", input_dims=(("s", (-2,)),)),
        ]
    )
    f.set_input_acceptance_dim("num_tokens", 100)
    return f


def _shape(*dims: int) -> tuple[int, ...]:
    return dims


def test_new_signatures_are_validated():
    tracker = _tracker(_acceptance_factory())
    tracker.validate_input_ties({"s_shape": _shape(1, 50, 64)})  # 3-D, axis -2 ok
    with pytest.raises(ValueError, match="no such tensor is present"):
        tracker.validate_input_ties({"x_shape": _shape(1, 2)})


def test_tied_lengths_must_match_across_inputs_and_pair_axes():
    factory = _acceptance_factory()
    factory.set_named_dim_ties([NamedDimTies(name="num_tokens", input_dims=(("s", (-2,)), ("z", (-2, -3))))])
    tracker = _tracker(factory)
    tracker.validate_input_ties({"s_shape": _shape(1, 8, 64), "z_shape": _shape(1, 8, 8, 32)})
    with pytest.raises(ValueError, match="unequal lengths"):
        tracker.validate_input_ties({"s_shape": _shape(1, 8, 64), "z_shape": _shape(1, 9, 8, 32)})
    with pytest.raises(ValueError, match="unequal lengths"):
        tracker.validate_input_ties({"s_shape": _shape(1, 8, 64), "z_shape": _shape(1, 9, 9, 32)})


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
    f.set_named_dim_ties(
        [
            NamedDimTies(name="num_tokens", input_dims=(("z", (-3,)),)),
        ]
    )
    f.set_padded_dim("num_tokens", dim_len_min=4, dim_len_max=16, num_intervals=4, multiple_of=1)
    tracker = _tracker(f)
    with pytest.raises(ValueError, match="no such tensor is present"):
        tracker.validate_input_ties({"s_shape": _shape(1, 8, 8, 32)})  # no z_shape
