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
"""Boltz's atom transformer runs each 3-way transition as one kernel from tables projected once per rollout."""

from unittest.mock import patch

import pytest
import torch

from bionemo_ir._torch.layers.transformers.atom import AtomTransformer
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    BoltzDiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock
from bionemo_ir.configs import DiffusionTransformerConfig

WINDOWS, QUERIES, KEYS, DIM, PAIR, HEADS = 7, 32, 128, 128, 16, 4


def _model(windowed: bool = False) -> AtomTransformer:
    """Boltz-2's atom transformer at two layers, in bf16; ``windowed`` keeps its windowed attention kernel."""
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=HEADS,
        dim=DIM,
        dim_single_cond=DIM,
        dim_pairwise=PAIR,
        bias_proj=False,
        conditioned_transition_using_silu=False,
        expansion_factor=2,
        dtype="bfloat16",
        pairwise_attention_backend="SDPA",
        version="v2",
    )
    model = AtomTransformer(
        attn_window_queries=QUERIES,
        attn_window_keys=KEYS,
        diffusion_transformer_config=config,
        diffusion_transformer_cls=BoltzDiffusionTransformer,
    )
    model = model.cuda().eval()
    if not windowed:
        model.diffusion_transformer.disable_window_attention()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.05)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    return model


@pytest.mark.parametrize("windowed", [False, True], ids=["attention-backend", "windowed-attention"])
@pytest.mark.parametrize("samples", [1, 5])
def test_prepared_transition_conds_run_each_transition_fused(samples: int, windowed: bool) -> None:
    """The tables feed the transitions only: with the windowed attention kernel, which takes prepared pair biases,
    as with the attention backend, every transition runs fused and matches the unfused layers."""
    torch.manual_seed(3)
    model = _model(windowed)
    if model.diffusion_transformer.layers[0].transition._conditioned_op is None:
        pytest.skip("the fused conditioned transition does not ship for this GPU and build")
    atoms = WINDOWS * QUERIES
    td = torch.bfloat16
    q = torch.randn(1, samples, atoms, DIM, device="cuda", dtype=td)
    c = torch.randn(1, atoms, DIM, device="cuda", dtype=td)
    bias = torch.randn(1, atoms, KEYS, 2 * HEADS, device="cuda", dtype=td)
    mask = torch.ones(1, atoms, device="cuda", dtype=td)
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(WINDOWS, QUERIES, KEYS, torch.device("cuda"))
    fused = []
    run_fused = ConditionedTransitionBlock.run_fused

    def spy(block: ConditionedTransitionBlock, a: torch.Tensor, table: torch.Tensor) -> torch.Tensor | None:
        out = run_fused(block, a, table)
        fused.append(out is not None)
        return out

    with torch.inference_mode():
        biases = model.prepare_pair_biases(bias, mask, metadata) if windowed else None
        expected = model(q, c, bias, mask, metadata, prepared_pair_biases=biases).clone()
        conds = model.prepare_transition_conds(c)
        assert conds is not None and [cond.shape for cond in conds] == [(atoms, 3 * DIM)] * 2
        with patch.object(ConditionedTransitionBlock, "run_fused", spy):
            actual = model(q, c, bias, mask, metadata, prepared_pair_biases=biases, prepared_transition_conds=conds)
    assert fused == [True, True]
    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=5e-2)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 1e-2, error


def test_prepared_transition_conds_need_one_batch_item() -> None:
    """The table broadcasts over the samples of one batch item only, so two items keep the unfused transition."""
    model = _model()
    c = torch.randn(2, WINDOWS * QUERIES, DIM, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert model.prepare_transition_conds(c) is None
