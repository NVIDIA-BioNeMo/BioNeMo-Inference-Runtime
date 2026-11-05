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
"""The conditioned transition MLP op: each config against the PyTorch reference from source and CUBINs, the config
choice, and the calls the op declines."""

from __future__ import annotations

import pytest
import torch

from bionemo_ir._torch.custom_ops.conditioned_transition_mlp import (
    EPS,
    _config,
    conditioned_transition_mlp_reference,
    get_conditioned_transition_mlp_op,
)
from bionemo_ir._torch.custom_ops.conditioned_transition_mlp import cutedsl as conditioned_cutedsl
from bionemo_ir._torch.custom_ops.conditioned_transition_mlp.cutedsl import ConditionedTransitionMlpCuTe
from bionemo_ir._torch.custom_ops.conditioned_transition_mlp.ops import ConditionedTransitionMlpOp
from tests._torch import cutedsl_test_modes, require_cubin_family, run_cutedsl_test_mode

_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.conditioned_transition_mlp._source"
_WIDTH, _HIDDEN, _SAMPLES = 128, 256, 5


def _require_kernel_sm() -> int:
    if not torch.cuda.is_available():
        pytest.skip("the conditioned transition needs CUDA")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in _config.KERNEL_ABIS:
        pytest.skip("the conditioned transition runs on SM100 and SM103 only")
    return sm


def _pairs() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count // 2


def _shape(tile: str, pairs: int) -> tuple[int, int]:
    """``(rows, atoms)`` of a call that runs ``tile`` on ``pairs`` CTA pairs."""
    pair_rows = 2 * _config.CTA_ROWS
    atoms = {
        "base": pairs * pair_rows + 8,
        "staged": pairs * pair_rows,
        # One sample-major tile per pair, the last clipped at the atom count.
        "sample_major": pairs * 2 * _config.sample_atoms(_SAMPLES) - 5,
        "sample_stream": 224,
    }[tile]
    rows = atoms * _SAMPLES if tile.startswith("sample_") else atoms
    return rows, atoms


def _cases() -> list[tuple[str, str]]:
    """Every ``(activation, tile)`` the SM100 bundle ships."""
    return [
        (activation, tile)
        for activation in _config.ACTIVATIONS
        for tile in _config.configs(100, _WIDTH, _HIDDEN, activation)
    ]


def _operands(rows: int, atoms: int, activation: str = "silu_gate", seed: int = 0) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(rows, _WIDTH, device="cuda", dtype=torch.bfloat16, generator=gen)
    blocks = _config.W1_BLOCKS[activation]
    w1 = torch.randn(blocks * _HIDDEN, _WIDTH, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    w2 = torch.randn(_WIDTH, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    cond = torch.rand(atoms, 3 * _WIDTH, device="cuda", dtype=torch.bfloat16, generator=gen)
    return a, w1, w2, cond


@pytest.mark.parametrize("mode", cutedsl_test_modes(_SOURCE_MODULE))
@pytest.mark.parametrize(("activation", "tile"), _cases())
def test_each_config_matches_the_reference(
    mode: str, activation: str, tile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every config computes ``a + sigmoid(g) * MLP(sigmoid(gamma) * LayerNorm(a) + beta)`` from source and from
    the packaged CUBIN, for the 2-way and the 3-way SwiGLU."""
    sm = _require_kernel_sm()
    if mode == "cubin":
        require_cubin_family("conditioned_transition_mlp")
    tiles = _config.configs(sm, _WIDTH, _HIDDEN, activation)
    rows, atoms = _shape(tile, _pairs())
    assert _config.choose_config(rows, atoms, _pairs(), _SAMPLES, tiles) == tile
    a, w1, w2, cond = _operands(rows, atoms, activation)
    op = ConditionedTransitionMlpOp(ConditionedTransitionMlpCuTe(sm), _WIDTH, _HIDDEN, activation)
    out = run_cutedsl_test_mode(
        mode, monkeypatch, ConditionedTransitionMlpCuTe, conditioned_cutedsl, lambda: op(a, w1, w2, cond)
    )
    assert out is not None
    expected = conditioned_transition_mlp_reference(a, w1, w2, cond, EPS, activation)
    torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("atoms", "config"),
    [
        (1776, "sample_stream"),
        (1777, "sample_major"),
        (3552, "sample_major"),
        (3553, "staged"),
        (7104, "sample_stream"),
        (7552, "base"),
        (7616, "sample_stream"),
        (11360, "base"),
        (13472, "sample_stream"),
    ],
)
def test_config_choice_on_74_pairs(atoms: int, config: str) -> None:
    """On B200's 74 CTA pairs, staged sample-major tiles run only when one round fills over half the pairs, and
    streamed ones only when their 120-row tiles add no round over the base config's 128-row tiles."""
    assert _config.choose_config(atoms * _SAMPLES, atoms, 74, _SAMPLES) == config


@pytest.mark.parametrize(("rows", "config"), [(74 * 256, "staged"), (74 * 256 + 1, "base")])
def test_config_choice_without_five_samples(rows: int, config: str) -> None:
    """Rows that are not five samples of the atoms take the row-tiled configs."""
    assert _config.choose_config(rows, rows, 74, _SAMPLES) == config


@pytest.mark.parametrize(("atoms", "config"), [(1777, "sample_stream"), (3553, "base"), (7104, "sample_stream")])
def test_config_choice_among_the_tiles_an_activation_ships(atoms: int, config: str) -> None:
    """The 3-way SwiGLU ships no staged tiles: calls that would stage their conditioning stream it."""
    assert _config.choose_config(atoms * _SAMPLES, atoms, 74, _SAMPLES, ("base", "sample_stream")) == config


def test_every_bundle_ships_both_activations() -> None:
    """The SM100 and SM103 bundles ship ``(128, 256)`` with every tile of the 2-way SwiGLU and the 3-way SwiGLU's
    unstaged tiles, its larger W1 leaving no room to stage; SM103's copy SM100's."""
    for sm in _config.KERNEL_ABIS:
        assert _config.shipped_shapes(sm) == {(_WIDTH, _HIDDEN, activation) for activation in _config.ACTIVATIONS}
        assert tuple(_config.configs(sm, _WIDTH, _HIDDEN, "silu_gate")) == _config.TILES
        assert tuple(_config.configs(sm, _WIDTH, _HIDDEN, "silu_gate_3way")) == ("base", "sample_stream")
        for activation in _config.ACTIVATIONS:
            assert _config.configs(sm, _WIDTH, _HIDDEN, activation) == _config.configs(100, _WIDTH, _HIDDEN, activation)


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        ({"base": {}}, "invalid conditioned transition config key"),
        ({"act=silu_gate|tile=sample_stream": {"sample_major": 5}}, "no base tile"),
        ({"act=silu_gate|tile=base": {"cond_stage": 1}}, "cond_stage"),
        ({"act=silu_gate|tile=base": {"x_stages": True}}, "non-negative integers"),
        (
            {
                "act=silu_gate|tile=base": {},
                "act=silu_gate|tile=sample_stream": {"sample_major": 5},
                "act=silu_gate|tile=sample_major": {"sample_major": 4, "cond_stage": 1},
            },
            "same samples",
        ),
    ],
)
def test_malformed_bundles_are_rejected(configs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config.validate_config("sm100", None, configs, "CT128_H256_sm100.json")


def test_op_serves_only_the_shipped_block() -> None:
    """The op serves a bf16 2-way or 3-way SwiGLU block with a LayerNorm AdaLN of its epsilon, at a shipped
    shape."""
    sm = _require_kernel_sm()
    assert get_conditioned_transition_mlp_op(torch.float16, _WIDTH, _HIDDEN, "silu_gate", EPS, "layer_norm") is None
    assert get_conditioned_transition_mlp_op(torch.bfloat16, 768, 1536, "silu_gate", EPS, "layer_norm") is None
    assert get_conditioned_transition_mlp_op(torch.bfloat16, _WIDTH, _HIDDEN, "relu", EPS, "layer_norm") is None
    assert get_conditioned_transition_mlp_op(torch.bfloat16, _WIDTH, _HIDDEN, "silu_gate", 1e-6, "layer_norm") is None
    assert get_conditioned_transition_mlp_op(torch.bfloat16, _WIDTH, _HIDDEN, "silu_gate", EPS, "rms_norm") is None
    op = ConditionedTransitionMlpOp(ConditionedTransitionMlpCuTe(sm), _WIDTH, _HIDDEN)
    a, w1, w2, cond = _operands(10, 3)
    # Rows that are not a whole number of the conditioning's rows, a non-contiguous input and a wrong dtype.
    assert op(a, w1, w2, cond) is None
    a, w1, w2, cond = _operands(12, 3)
    assert op(a.t().contiguous().t(), w1, w2, cond) is None
    assert op(a.float(), w1, w2, cond) is None
