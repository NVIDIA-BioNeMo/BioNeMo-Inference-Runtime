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

from functools import partial

import pytest
import torch
from torch import nn

from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.modules.boltz import confidence as boltz_confidence
from bionemo_ir._torch.modules.boltz import confidence_utils as boltz_utils
from bionemo_ir._torch.modules.boltz.confidence import Boltz1ConfidenceHeads, Boltz2ConfidenceHeads
from bionemo_ir._torch.modules.openfold2.confidence import AuxiliaryHeads
from bionemo_ir._torch.utils.confidence import pair_expectations
from bionemo_ir.models.boltz1.config import ConfidenceHeadsConfig as Boltz1Config
from bionemo_ir.models.boltz2.config import ConfidenceHeadsConfig as Boltz2Config
from bionemo_ir.models.openfold2.config import ConfidenceModuleConfig
from tests._torch.test_confidence_utils import _frame_feats

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def initialize_heads(module: nn.Module, device: str, dtype: torch.dtype) -> nn.Module:
    for child in module.modules():
        if isinstance(child, Linear):
            child.weight = nn.Parameter(torch.randn(child.out_features, child.in_features, dtype=dtype) * 0.1)
            child.register_parameter(
                "bias", nn.Parameter(torch.zeros(child.out_features, dtype=dtype)) if child.has_bias else None
            )
            child._weights_created = True
    return module.to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("symmetric", [False, True])
def test_pair_expectations_bound_projection(device: str, dtype: torch.dtype, symmetric: bool) -> None:
    torch.manual_seed(71)
    pair = torch.randn(2, 3, 11, 11, 8, device=device, dtype=dtype)
    projection = nn.Linear(8, 64, bias=False, device=device, dtype=dtype)
    centers = torch.arange(0.25, 32, 0.5, device=device)
    weights = (centers, torch.rand(2, 3, 1, 1, 64, device=device))
    seen = []

    def project(block: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        seen.append((start, stop))
        assert block.shape[-3] <= 3
        return projection(block)

    with torch.inference_mode():
        logits = projection(pair + pair.transpose(-3, -2) if symmetric else pair)
        probabilities = logits.softmax(-1)
        expected = tuple((probabilities * weight).sum(-1) for weight in weights)
        actual = pair_expectations(pair, project, weights, symmetric=symmetric, rows=3)
    assert seen == [(0, 3), (3, 6), (6, 9), (9, 11)]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("multimer", [False, True])
@pytest.mark.parametrize("tm_enabled", [False, True])
def test_of2_compact_matches_raw(device: str, dtype: torch.dtype, multimer: bool, tm_enabled: bool) -> None:
    torch.manual_seed(93)
    config = ConfidenceModuleConfig(skip_create_weights=True)
    config.set_dtype(dtype)
    config.per_residue_lddt.c_in = config.per_residue_lddt.c_hidden = 8
    config.distogram.c_z = config.tm.c_z = config.masked_msa.c_m = config.experimentally_resolved.c_s = 8
    config.tm.enabled = tm_enabled
    module = initialize_heads(AuxiliaryHeads(config), device, dtype)
    inputs = {
        "single": torch.randn(1, 13, 8, device=device, dtype=dtype),
        "pair": torch.randn(1, 13, 13, 8, device=device, dtype=dtype),
        "msa": torch.randn(1, 3, 13, 8, device=device, dtype=dtype),
    }
    if multimer:
        inputs["asym_id"] = (torch.arange(13, device=device) % 2).unsqueeze(0)
    with torch.inference_mode():
        raw = module(dict(inputs))
        module.config.compact_output = True
        compact = module(dict(inputs))
    assert not any("logits" in name or "probs" in name for name in compact)
    for name, value in compact.items():
        torch.testing.assert_close(value, raw[name], atol=2e-5, rtol=1e-6)
    assert ("ptm_score" in compact) == tm_enabled
    assert ("iptm_score" in compact) == (tm_enabled and multimer)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("case", ["multichain", "monomer", "no-frame"])
def test_boltz_compact_preserves_metrics(
    device: str, dtype: torch.dtype, version: int, separate: bool, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(41)
    if version == 1:
        config = Boltz1Config(token_s=8, token_z=8, skip_create_weights=True)
        config.set_dtype(dtype)
        module = Boltz1ConfidenceHeads(config)
    else:
        config = Boltz2Config(token_s=8, token_z=8, use_separate_heads=separate, skip_create_weights=True)
        module = Boltz2ConfidenceHeads(config, dtype=dtype, skip_create_weights=True)
    module = initialize_heads(module, device, dtype)
    feats, frames = _frame_feats(torch.device(device))
    if case == "monomer":
        feats["asym_id"].zero_()
        feats["mol_type"].zero_()
    feats["frames_idx"] = frames
    coordinates = torch.randn(1, 3, 24, 3, device=device)
    if case == "no-frame":
        coordinates.zero_()
    pair = torch.randn(1, 3, 12, 12, 8, device=device, dtype=dtype)
    same = (feats["asym_id"][:, None, :, None] == feats["asym_id"][:, None, None, :]).expand(1, 3, 12, 12)

    def run_heads() -> tuple[dict, torch.Tensor]:
        if version == 1:
            return module._compute_pae_outputs(pair, coordinates, feats), module._compute_pde(pair)
        return (
            module._compute_pae_outputs(pair, coordinates, feats, 3, same, ~same),
            module._compute_pde(pair, same, ~same),
        )

    with torch.inference_mode():
        raw = run_heads()
        blocked = partial(pair_expectations, rows=5)
        monkeypatch.setattr(boltz_confidence, "pair_expectations", blocked)
        monkeypatch.setattr(boltz_utils, "pair_expectations", blocked)
        if version == 1:
            module.config.compact_output = True
        else:
            module.compact_output = True
        compact = run_heads()
    torch.testing.assert_close(compact, raw, atol=2e-5, rtol=1e-6)
