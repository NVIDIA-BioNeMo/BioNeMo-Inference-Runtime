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

"""Indexed pLDDT preserves inference values and fallback contracts."""

import pytest
import torch

from bionemo_ir._torch.modules.protenix.confidence import ProtenixConfidenceHead
from bionemo_ir.dsl_kernels.triton.indexed_projection import (
    _cached_projection,
    indexed_projection,
    prepare_indexed_rows,
)
from bionemo_ir.models.protenix.config import ConfidenceHeadConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("atoms,batch", [(1, 1), (33, 2), (517, 1), (2048, 2)])
@pytest.mark.parametrize("single_group", [False, True])
@torch.no_grad()
def test_indexed_projection(atoms: int, batch: int, single_group: bool) -> None:
    torch.manual_seed(42)
    x = torch.randn(batch, atoms, 384, device="cuda")
    weight = torch.randn(24, 384, 50, device="cuda") * 0.05
    index = (
        torch.zeros(atoms, device="cuda", dtype=torch.long)
        if single_group
        else torch.randint(-24, 24, (atoms * 2,), device="cuda")[::2]
    )
    rows = prepare_indexed_rows(index, 24)
    assert rows is not None
    expected = torch.einsum("bac,aco->bao", x, weight[index])
    actual = indexed_projection(x, weight, rows)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = indexed_projection(x, weight, rows)
    x.add_(0.1)
    graph.replay()
    torch.testing.assert_close(captured, torch.einsum("bac,aco->bao", x, weight[index]), atol=1e-5, rtol=1e-5)


def test_invalid_layout() -> None:
    assert prepare_indexed_rows(torch.tensor([0]), 24) is None
    for values in ([], [-25], [24]):
        assert prepare_indexed_rows(torch.tensor(values, device="cuda", dtype=torch.long), 24) is None


class IdentityPairformer(torch.nn.Module):
    def forward(self, s: torch.Tensor, z: torch.Tensor, *masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return s, z


@torch.no_grad()
def test_confidence_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bionemo_ir._torch.modules.protenix.confidence.PairformerModule", lambda config: IdentityPairformer()
    )
    # The stub Pairformer's signature does not match the confidence graph policy.
    config = ConfidenceHeadConfig(graph_optimization_config=None)
    head = ProtenixConfidenceHead(config).cuda().eval()
    for parameter in head.parameters():
        parameter.normal_(std=0.05)
    tokens, atoms = 64, 16384
    owner = torch.arange(atoms, device="cuda") % tokens
    slots = torch.arange(atoms, device="cuda") % 8
    batch = {
        "atom_to_token_idx": owner,
        "atom_to_tokatom_idx": slots,
        "distogram_rep_atom_mask": torch.arange(atoms, device="cuda") < tokens,
    }
    si = torch.randn(1, tokens, config.c_s_inputs, device="cuda")
    s = torch.randn(1, tokens, config.c_s, device="cuda")
    z = torch.randn(1, tokens, tokens, config.c_z, device="cuda")
    ctx = head.prepare(batch, si, s, z)
    assert ctx["plddt_rows"] is not None
    x = torch.randn(1, atoms, 3, device="cuda")
    actual = head.per_sample_logits(ctx, x)
    reference_ctx = dict(ctx, plddt_rows=None)
    expected = head.per_sample_logits(reference_ctx, x)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    with torch.enable_grad():
        assert head.prepare(batch, si, s, z)["plddt_rows"] is None
        head.per_sample_logits(ctx, x)[0].sum().backward()
    assert head.plddt_weight.grad is not None


@pytest.mark.parametrize("launch", ["driver", "compiled", "unaligned"])
@torch.no_grad()
def test_cached_launch(launch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = _cached_projection(torch.cuda.current_device(), 384, 50)
    if launch == "driver" and kernel.driver is None:
        pytest.skip("CUDA driver launcher unavailable")
    if launch == "compiled":
        monkeypatch.setattr(kernel, "_driver", None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for atoms in (1, 35):
            offset = int(launch == "unaligned")
            x = torch.randn(atoms * 384 + offset, device="cuda")[offset:].view(1, atoms, 384)
            weight = torch.randn(24, 384, 50, device="cuda") * 0.05
            index = torch.arange(atoms, device="cuda") % 24
            rows = prepare_indexed_rows(index, 24)
            expected = torch.einsum("bac,aco->bao", x, weight[index])
            actual = indexed_projection(x, weight, rows)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    stream.synchronize()


@pytest.mark.parametrize("layout", ["transposed", "sliced", "expanded", "weights"])
@torch.no_grad()
def test_reject_noncontiguous(layout: str) -> None:
    features = torch.empty(2, 35, 384, device="cuda")
    weights = torch.empty(24, 384, 50, device="cuda")
    if layout == "transposed":
        features = torch.empty(2, 384, 35, device="cuda").transpose(-1, -2)
    elif layout == "sliced":
        features = torch.empty(2, 70, 384, device="cuda")[:, ::2]
    elif layout == "expanded":
        features = features[:1].expand(2, -1, -1)
    else:
        weights = torch.empty(24, 50, 384, device="cuda").transpose(-1, -2)
    rows = prepare_indexed_rows(torch.arange(35, device="cuda") % 24, 24)
    with pytest.raises(ValueError, match="features and weights must be contiguous"):
        indexed_projection(features, weights, rows)
