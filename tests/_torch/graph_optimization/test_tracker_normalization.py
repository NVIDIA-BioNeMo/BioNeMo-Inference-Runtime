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
"""Tracker consumption of decorator-declared data:
* positional args normalized to parameter names (``Signature.bind``).
* workspace kwargs read from the ``InputRoutingConfig``.
"""
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig, InputKeyMethod, InputRoutingConfigFactory)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import \
    CUDAGraphOptimizationTracker


class _NamedStub(nn.Module):
    """Wrapped module whose forward names the pairformer-style params."""

    def forward(self, s, z, mask=None, buffers=None, **kwargs):
        return s, z


def _tracker(inner, input_routing_config=None) -> CUDAGraphOptimizationTracker:
    cfg_kwargs = {"input_key_method": InputKeyMethod.EXACT}
    if input_routing_config is not None:
        cfg_kwargs["input_routing_config"] = input_routing_config
    cfg = CUDAGraphOptimizationConfig(**cfg_kwargs)
    return CUDAGraphOptimizationTracker(cfg, inner_module=inner)


# --- positional -> parameter name normalization ----------------------------
def test_positional_args_normalized_to_param_names():
    tracker = _tracker(_NamedStub())
    s = torch.zeros(1, 10, 4)
    z = torch.zeros(1, 10, 10, 4)
    positional = tracker._extract_tensor_container_shapes((s, z), {})
    keyword = tracker._extract_tensor_container_shapes((), {"s": s, "z": z})
    # positional s, z key on the parameter names, matching a keyword call.
    assert set(positional) == {"s_shape", "z_shape"}
    assert set(positional) == set(keyword)


def test_input_key_independent_of_call_spelling():
    tracker = _tracker(_NamedStub())
    s = torch.zeros(1, 10, 4)
    z = torch.zeros(1, 10, 10, 4)
    assert (tracker.input_key_for_this_call(s, z)
            == tracker.input_key_for_this_call(s=s, z=z))


def test_positional_overflow_falls_back_to_argi():
    class _VarArgs(nn.Module):
        def forward(self, s, *args):
            return s

    tracker = _tracker(_VarArgs())
    a, b, c = (torch.zeros(1, 4) for _ in range(3))
    shapes = tracker._extract_tensor_container_shapes((a, b, c), {})
    # first positional -> "s"; the *args overflow -> arg1, arg2.
    assert set(shapes) == {"s_shape", "arg1_shape", "arg2_shape"}


# --- workspace kwargs sourced from the config ------------------------------
def test_workspace_kwargs_read_from_config():
    factory = InputRoutingConfigFactory()
    factory.set_internal_workspace_kwargs(["scratch"])
    tracker = _tracker(_NamedStub(), factory.export_config())
    assert tracker._effective_workspace_kwargs() == frozenset({"scratch"})
    kept = tracker._graph_input_kwargs({"s": 1, "scratch": 2})
    assert "scratch" not in kept and "s" in kept


def test_workspace_kwargs_fall_back_to_builtin_default():
    # No config (or a config that declares none) -> built-in {"buffers"}.
    tracker = _tracker(_NamedStub(), None)
    assert tracker._effective_workspace_kwargs() == frozenset({"buffers"})
    kept = tracker._graph_input_kwargs({"s": 1, "buffers": 2})
    assert "buffers" not in kept and "s" in kept
