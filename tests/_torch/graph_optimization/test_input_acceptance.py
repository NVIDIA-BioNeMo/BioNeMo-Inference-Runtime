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
"""``GraphOptimizationTracker.input_accepted`` — the acceptance-range check.

An acceptance rule caps a named dimension: an input axis tied to that dimension
is accepted when its length does not exceed the configured ``dim_len_max`` (an
inclusive maximum), and rejected only when strictly greater. The
rules are carried as plain data on the :class:`InputRoutingConfig` snapshot
(``input_acceptance_dims`` + ``input_dims_with_assigned_padded_dim``), so the
predicate is rebuilt from the config — including after (de)serialization — with
no closure involved. These are pure shape checks, so no CUDA is needed.
"""
import pickle

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig, InputKeyMethod)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker)
from tensorrt_bionemo._torch.graph_optimization.config import (
    InputRoutingConfig, InputRoutingConfigFactory)

DIM_LEN = 8  # feature dim of the stand-in token tensor: [B, n_tokens, DIM]


def _factory_with_rule(dim_len_max: int) -> InputRoutingConfigFactory:
    """A factory tying ``num_tokens`` to arg0 axis -2, capped at ``dim_len_max``."""
    factory = InputRoutingConfigFactory()
    factory.set_input_acceptance_dim("num_tokens", dim_len_max)
    factory.input_dim_is_acceptance("arg0", -2, "num_tokens")
    return factory


def _tracker(input_routing_config=None) -> CUDAGraphOptimizationTracker:
    cfg_kwargs = {"input_key_method": InputKeyMethod.BUCKETED_SHAPES}
    if input_routing_config is not None:
        cfg_kwargs["input_routing_config"] = input_routing_config
    cfg = CUDAGraphOptimizationConfig(**cfg_kwargs)
    return CUDAGraphOptimizationTracker(cfg, inner_module=nn.Identity())


def _accepted(tracker: CUDAGraphOptimizationTracker, *args, **kwargs) -> bool:
    """Derive the ``tensor_container_shapes`` map the way ``forward`` does, then
    run the acceptance check on it (``input_accepted`` keys off shapes, not the
    live tensors)."""
    shapes = tracker._extract_tensor_container_shapes(args, kwargs)
    return tracker.input_accepted(shapes)


@pytest.mark.parametrize("n_tokens, accepted", [
    (50, True),     # well under the limit
    (100, True),    # equal to the limit is accepted (inclusive bound)
    (101, False),   # just over (strict >)
    (150, False),   # well over the limit
])
def test_input_accepted_respects_inclusive_limit(n_tokens, accepted):
    tracker = _tracker(_factory_with_rule(100).export_config())
    assert _accepted(tracker, torch.zeros(1, n_tokens, DIM_LEN)) is accepted


def test_no_acceptance_rule_accepts_any_size():
    """A padded dim with no acceptance rule imposes no cap."""
    factory = InputRoutingConfigFactory()
    factory.set_padded_dim("num_tokens", 4, 2048, 8)
    factory.input_dim_is_padded("arg0", -2, "num_tokens")
    tracker = _tracker(factory.export_config())
    assert _accepted(tracker, torch.zeros(1, 9999, DIM_LEN)) is True


def test_no_routing_config_accepts_everything():
    """A module without an ``input_routing_config`` accepts every call."""
    tracker = _tracker(input_routing_config=None)
    assert _accepted(tracker, torch.zeros(1, 9999, DIM_LEN)) is True


def test_survives_json_and_pickle_roundtrip():
    """The predicate is rebuilt from the serialized snapshot, so acceptance is
    identical after a JSON or pickle round-trip of the config."""
    cfg = _factory_with_rule(100).export_config()

    for restored in (InputRoutingConfig.model_validate_json(cfg.model_dump_json()),
                     pickle.loads(pickle.dumps(cfg))):
        tracker = _tracker(restored)
        assert _accepted(tracker, torch.zeros(1, 100, DIM_LEN)) is True
        assert _accepted(tracker, torch.zeros(1, 150, DIM_LEN)) is False


# ---------------------------------------------------------------------------
# Multiple inputs tied to one acceptance dim, across positional + keyword args
# and across several axes of the same tensor. Models a boltz-2-style pairformer
# call ``pairformer_module(s, z, mask=..., pair_mask=...)`` where the token axis
# ``num_tokens`` rides: s -> arg0 (-2); z -> arg1 (-2 and -3); mask (-1);
# pair_mask (-1 and -2). ``input_accepted`` must return True only when *every*
# tied axis is within the cap (at most ``dim_len_max``), and False as soon as
# *any one* exceeds it.
# ---------------------------------------------------------------------------
def _factory_pairformer_rule(dim_len_max: int) -> InputRoutingConfigFactory:
    factory = InputRoutingConfigFactory()
    factory.set_input_acceptance_dim("num_tokens", dim_len_max)
    factory.input_dim_is_acceptance("arg0", -2, "num_tokens")       # s
    factory.input_dim_is_acceptance("arg1", -2, "num_tokens")       # z
    factory.input_dim_is_acceptance("arg1", -3, "num_tokens")       # z
    factory.input_dim_is_acceptance("mask", -1, "num_tokens")       # mask
    factory.input_dim_is_acceptance("pair_mask", -1, "num_tokens")  # pair_mask
    factory.input_dim_is_acceptance("pair_mask", -2, "num_tokens")  # pair_mask
    return factory


def _pairformer_inputs(n_tokens: int, *, pair_mask_tokens: int | None = None):
    """(args, kwargs) for the pairformer call with a given token count.

    ``s``/``z`` are passed positionally (-> ``arg0``/``arg1``); ``mask``/
    ``pair_mask`` by keyword. ``pair_mask_tokens`` lets one input's token axes be
    sized independently, to show a single over-limit input rejects the call.
    """
    n = n_tokens
    m = n if pair_mask_tokens is None else pair_mask_tokens
    s = torch.zeros(1, n, DIM_LEN)
    z = torch.zeros(1, n, n, DIM_LEN)
    mask = torch.zeros(1, n)
    pair_mask = torch.zeros(1, m, m)
    return (s, z), {"mask": mask, "pair_mask": pair_mask}


def test_all_tied_axes_under_limit_accepted():
    """Every tied axis (positional + keyword, multi-axis) under the cap -> True."""
    tracker = _tracker(_factory_pairformer_rule(1024).export_config())
    args, kwargs = _pairformer_inputs(512)
    assert _accepted(tracker, *args, **kwargs) is True


def test_all_tied_axes_at_limit_accepted():
    """Every tied axis exactly at the inclusive cap -> accepted."""
    tracker = _tracker(_factory_pairformer_rule(1024).export_config())
    args, kwargs = _pairformer_inputs(1024)
    assert _accepted(tracker, *args, **kwargs) is True


def test_rejected_when_any_single_input_over_limit():
    """All inputs within the cap except ``pair_mask``, whose token axes exceed
    it; a single offending input is enough to reject the call."""
    tracker = _tracker(_factory_pairformer_rule(1024).export_config())
    args, kwargs = _pairformer_inputs(512, pair_mask_tokens=1025)
    assert _accepted(tracker, *args, **kwargs) is False
