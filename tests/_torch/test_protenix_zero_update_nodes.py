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
"""Protenix truncates pairformer nodes whose checkpoint parameters are all ~0 (convert.py and modeling.py)."""

import pytest
import torch

from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir.configs import PairformerConfig
from bionemo_ir.models.protenix import modeling
from bionemo_ir.models.protenix.convert import drop_zero_update_nodes, zero_update_nodes

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="BioIR Linear allocates on CUDA")


def test_zero_update_nodes_reads_the_checkpoint_and_drops_their_keys():
    tiny, live = torch.full((4, 4), 1e-37), torch.ones(4, 4)
    weights = {
        "pairformer_stack.blocks.0.attention_pair_bias.attention.linear_q.weight": tiny,
        "pairformer_stack.blocks.0.attention_pair_bias.layernorm_a.weight": tiny[0],
        "pairformer_stack.blocks.0.single_transition.linear_no_bias.weight": live,
        "pairformer_stack.blocks.1.tri_att_end.linear.weight": tiny,
        "pairformer_stack.blocks.1.pair_transition.linear.weight": torch.zeros(4, 4),
        "pairformer_stack.blocks.1.tri_mul_in.linear.weight": tiny,
        "pairformer_stack.blocks.1.tri_mul_in.layer_norm_in.weight": live[0],  # one live parameter keeps the node
        "pairformer_stack.blocks.1.tri_att_start.linear.weight": torch.tensor([1e-37, float("nan")]),  # so does a NaN
        "other.blocks.0.attention_pair_bias.weight": tiny,
    }
    dead = zero_update_nodes(weights, "pairformer_stack", num_blocks=2)
    assert dead == {0: frozenset({"attention"}), 1: frozenset({"tri_attn_end", "transition_z"})}
    state = {
        "pairformer_stack.layers.0.attention.proj_o.weight": tiny,
        "pairformer_stack.layers.0.transition_s.fc3.weight": live,
        "pairformer_stack.layers.1.tri_attn_end.linear.weight": tiny,
        "layernorm_s.weight": live[0],
    }
    assert set(drop_zero_update_nodes(state, "pairformer_stack", dead)) == {
        "pairformer_stack.layers.0.transition_s.fc3.weight",
        "layernorm_s.weight",
    }


def test_truncated_nodes_move_and_cast_with_the_model():
    """A reload restores the truncated nodes, so they follow the model's .to() outside its module tree."""
    model = modeling.Protenix.__new__(modeling.Protenix)
    torch.nn.Module.__init__(model)
    model.kept = torch.nn.Linear(2, 2)
    truncated = torch.nn.Linear(2, 2)
    model._truncated_nodes = [(torch.nn.Module(), "node", truncated), (torch.nn.Module(), "no_update_s", False)]
    model.to(torch.float64)
    assert model.kept.weight.dtype == truncated.weight.dtype == torch.float64


def _stack() -> PairformerModule:
    config = PairformerConfig(
        token_s=64,
        token_z=32,
        num_blocks=2,
        num_heads=4,
        pairwise_head_width=16,
        pairwise_num_heads=2,
        no_update_s=False,
        attention_initial_norm=True,
        version="v1",
        dtype="float32",
        triangle_attention_backend="VANILLA",
        pairwise_attention_backend="SDPA",
    )
    torch.manual_seed(0)
    stack = PairformerModule(config).cuda().eval()
    with torch.no_grad():
        for parameter in stack.parameters():
            parameter.normal_(0, 0.1)
    return stack


def _run(stack: PairformerModule, s: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return stack(
            s, z, torch.ones(1, s.shape[1], device="cuda"), torch.ones(1, s.shape[1], s.shape[1], device="cuda")
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, 12, 64, generator=generator).cuda(), torch.randn(1, 12, 12, 32, generator=generator).cuda()


def _make_dead(stack: PairformerModule, dead: dict[int, frozenset[str]]) -> None:
    with torch.no_grad():
        for index, nodes in dead.items():
            for name in nodes:
                for parameter in getattr(stack.layers[index], name).parameters():
                    parameter.fill_(1e-37)


@cuda
def test_truncated_nodes_change_no_output_bit_and_never_write_the_inputs():
    """Every pair node before the in-place pair transition is dead: the first stub copies, the caller's z survives."""
    stack = _stack()
    dead = {0: frozenset({"attention", "tri_mul_out", "tri_mul_in", "tri_attn_start", "tri_attn_end"})}
    _make_dead(stack, dead)
    s, z = _inputs()
    expected = _run(stack, s, z)
    s_in, z_in = s.clone(), z.clone()
    replaced = modeling._truncate_zero_update_nodes(stack, dead)
    assert not any(True for name in dead[0] for _ in getattr(stack.layers[0], name).parameters())
    actual = _run(stack, s, z)
    for want, got in zip(expected, actual, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(s, s_in) and torch.equal(z, z_in)
    modeling._restore(replaced)
    assert sum(1 for _ in stack.layers[0].tri_mul_out.parameters()) > 0


@cuda
def test_a_dead_single_track_turns_off_the_layers_single_update():
    stack = _stack()
    dead = {1: frozenset({"attention", "transition_s"})}
    _make_dead(stack, dead)
    s, z = _inputs()
    expected = _run(stack, s, z)
    replaced = modeling._truncate_zero_update_nodes(stack, dead)
    assert stack.layers[1].no_update_s
    actual = _run(stack, s, z)
    for want, got in zip(expected, actual, strict=True):
        assert torch.equal(got, want)
    modeling._restore(replaced)
    assert not stack.layers[1].no_update_s
