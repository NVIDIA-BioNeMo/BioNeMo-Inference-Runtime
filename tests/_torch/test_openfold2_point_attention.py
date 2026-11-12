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


"""FP32 IPA score fusion and per-call pair projection reuse."""

import math
from unittest.mock import patch

import pytest
import torch

from bionemo_ir._torch.modules.openfold2.point_attention import InvariantPointAttention
from bionemo_ir._torch.modules.openfold2.structure import StructureModule
from bionemo_ir._torch.modules.openfold2.utils.geometry.rigid_matrix_vector import Rigid3Array
from bionemo_ir._torch.modules.openfold2.utils.rigid_utils import Rigid
from bionemo_ir.dsl_kernels.triton.ipa_scores import _scores_kernel, ipa_scores
from bionemo_ir.models.openfold2.config import StructureModuleConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_scores(
    scalar: torch.Tensor,
    bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    channels: int,
    inf: float,
) -> torch.Tensor:
    a = scalar * math.sqrt(1.0 / (3 * channels))
    a = a + math.sqrt(1.0 / 3) * bias.movedim(-1, -3)
    distance = (q.unsqueeze(-4) - k.unsqueeze(-5)) ** 2
    distance = sum(torch.unbind(distance, dim=-1))
    distance = distance * weight[:, None]
    distance = torch.sum(distance, dim=-1) * (-0.5)
    square = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    return torch.softmax(a + distance.movedim(-1, -3) + (inf * (square - 1)).unsqueeze(-3), dim=-1)


@pytest.mark.parametrize(
    "n,batch,points",
    [(1, (), 4), (31, (2,), 4), (65, (2, 2), 8), (373, (), 4), (920, (), 4), (2048, (), 4)],
)
@pytest.mark.parametrize("masked", [False, True])
def test_ipa_scores(n: int, batch: tuple[int, ...], points: int, masked: bool) -> None:
    torch.manual_seed(42)
    h, c = 12, 16
    a = torch.randn((*batch, h, n, n), device="cuda")
    b = torch.randn((*batch, n, n, h), device="cuda")
    q = torch.randn((*batch, n, h, points, 3), device="cuda")
    # A split of the combined key/value projection is non-contiguous.
    k = torch.randn((*batch, n, h, points + 8, 3), device="cuda")[..., :points, :]
    w = torch.rand(h, device="cuda")
    mask = torch.ones((*batch, n), device="cuda")
    if masked:
        mask[..., ::3] = 0
    expected = _reference_scores(a, b, q, k, w, mask, c, 1e5)
    actual = ipa_scores(a, b, q, k, w, mask, c, 1e5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)), atol=1e-6, rtol=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = ipa_scores(a, b, q, k, w, mask, c, 1e5)
    graph.replay()
    torch.testing.assert_close(replayed, actual, atol=0, rtol=0)


@pytest.mark.parametrize("multimer", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_ipa_forward(multimer: bool, shared: bool) -> None:
    torch.manual_seed(5)
    module = InvariantPointAttention(384, 128, 16, 12, 4, 8, is_multimer=multimer).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
        s = torch.randn(2, 65, 384, device="cuda")
        z = torch.randn(2, 65, 65, 128, device="cuda")
        r = (
            Rigid3Array.identity(s.shape[:-1], s.device)
            if multimer
            else Rigid.identity(s.shape[:-1], s.dtype, s.device, False)
        )
        mask = torch.ones(2, 65, device="cuda")
        mask[:, -5:] = 0
        if shared:
            z = z[0]
            mask = mask[0]
        with patch("bionemo_ir._torch.modules.openfold2.point_attention.ipa_scores", _reference_scores):
            expected = module(s, z, r, mask)
        with patch("bionemo_ir._torch.modules.openfold2.point_attention.ipa_scores", wraps=ipa_scores) as fused:
            actual = module(s, z, r, mask)
        # Without the fused call, both sides run eager scores and the comparison proves nothing.
        assert fused.called == (torch.cuda.get_device_capability(s.device) in ((10, 0), (10, 3)))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_ipa_scores_empty() -> None:
    a = torch.empty(0, 12, 31, 31, device="cuda")
    b = torch.empty(0, 31, 31, 12, device="cuda")
    q = torch.empty(0, 31, 12, 4, 3, device="cuda")
    mask = torch.empty(0, 31, device="cuda")
    weight = torch.rand(12, device="cuda")
    assert ipa_scores(a, b, q, q.clone(), weight, mask, 16, 1e5).shape == (0, 12, 31, 31)


def test_ipa_scores_compile_once_per_block() -> None:
    """Lengths that share a power-of-two block reuse one compiled kernel."""
    torch.manual_seed(3)
    _scores_kernel.cache_clear()
    for n in (33, 47, 64):
        a = torch.randn(12, n, n, device="cuda")
        b = torch.randn(n, n, 12, device="cuda")
        q = torch.randn(n, 12, 4, 3, device="cuda")
        k = torch.randn_like(q)
        w = torch.rand(12, device="cuda")
        mask = torch.ones(n, device="cuda")
        mask[::5] = 0
        expected = _reference_scores(a, b, q, k, w, mask, 16, 1e5)
        torch.testing.assert_close(ipa_scores(a, b, q, k, w, mask, 16, 1e5), expected, atol=2e-6, rtol=2e-5)
    assert _scores_kernel.cache_info().currsize == 1


def test_ipa_scores_broadcast() -> None:
    torch.manual_seed(12)
    a = torch.randn(2, 12, 31, 31, device="cuda")
    b = torch.randn(31, 31, 12, device="cuda")
    q = torch.randn(31, 12, 4, 3, device="cuda")
    k = torch.randn_like(q)
    weight = torch.rand(12, device="cuda")
    mask = torch.ones(31, device="cuda")
    expected = _reference_scores(a, b, q, k, weight, mask, 16, 1e5)
    actual = ipa_scores(a, b, q, k, weight, mask, 16, 1e5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("negative_view", [False, True])
def test_ipa_scores_storage_views(negative_view: bool) -> None:
    torch.manual_seed(17)

    def offset(shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.randn(math.prod(shape) + 1, device="cuda")[1:].view(shape)
        return torch._neg_view(-tensor) if negative_view else tensor

    a = offset((12, 512, 512))
    b = offset((512, 512, 12))
    q = offset((512, 12, 4, 3))
    k = offset((512, 12, 4, 3))
    w = offset((12,)).abs()
    mask = torch.ones(513, device="cuda")[1:]
    mask[::3] = 0
    if negative_view:
        mask = torch._neg_view(-mask)
    expected = _reference_scores(a, b, q, k, w, mask, 16, 1e5)
    actual = ipa_scores(a, b, q, k, w, mask, 16, 1e5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["integer_mask", "double_mask", "negative_view", "double_inputs"])
def test_ipa_eager_fallback(mode: str) -> None:
    torch.manual_seed(13)
    module = InvariantPointAttention(384, 128, 16, 12, 4, 8).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
    s = torch.randn(17, 384, device="cuda")
    z = torch.randn(17, 17, 128, device="cuda")
    mask = torch.ones(17, device="cuda")
    if mode == "integer_mask":
        mask = mask.long()
    elif mode == "double_mask":
        mask = mask.double()
    elif mode == "negative_view":
        mask = torch._neg_view(-mask)
    elif mode == "double_inputs":
        module = module.double()
        s, z = s.double(), z.double()
    r = Rigid.identity(s.shape[:-1], s.dtype, s.device, False)
    with patch(
        "bionemo_ir._torch.modules.openfold2.point_attention.ipa_scores",
        side_effect=AssertionError("unsupported tensors must use eager IPA"),
    ):
        with torch.enable_grad():
            expected = module(s, z, r, mask)
        with torch.no_grad():
            actual = module(s, z, r, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("multimer", [False, True])
def test_structure_pair_projection_is_per_call(multimer: bool) -> None:
    torch.manual_seed(7)
    config = StructureModuleConfig(is_multimer=multimer, no_blocks=3, dtype="float32")
    module = StructureModule(config).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
        s = torch.randn(17, 384, device="cuda")
        z = torch.randn(17, 17, 128, device="cuda")
        aatype = torch.zeros(17, device="cuda", dtype=torch.long)
        with patch.object(module.ipa.linear_b, "forward", wraps=module.ipa.linear_b.forward) as projection:
            actual = module(s, z, aatype)
            assert projection.call_count == 1
            module(s, z + 0.25, aatype)
            assert projection.call_count == 2
        original = module.ipa.forward

        def recompute(
            single: torch.Tensor,
            pair: torch.Tensor,
            rigids: Rigid | Rigid3Array,
            mask: torch.Tensor,
            pair_bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return original(single, pair, rigids, mask)

        with patch.object(module.ipa, "forward", side_effect=recompute):
            expected = module(s, z, aatype)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
