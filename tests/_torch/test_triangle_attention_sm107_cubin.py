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
"""SM107 native scheduler, tail and graph parity."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests._torch import SM_VERSION

pytestmark = pytest.mark.skipif(SM_VERSION != 107, reason="requires SM107")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("head_dim", "heads", "tokens", "logical_dim"),
    [
        (32, 2, 320, 32),
        (32, 3, 384, 32),
        (32, 2, 385, 32),
        (32, 8, 520, 32),
        (32, 3, 640, 32),
        (32, 4, 520, 16),
        (64, 2, 129, 64),
        (128, 3, 80, 128),
        (128, 2, 129, 128),
    ],
)
def test_sm107_native_parity(
    dtype: torch.dtype, packed: bool, head_dim: int, heads: int, tokens: int, logical_dim: int
) -> None:
    from bionemo_ir._torch.attention_backend.triangle_attention._cubin import TriangleAttentionCubinExecutable
    from bionemo_ir.libs import _cutedsl_kernels as library

    torch.manual_seed(tokens + heads)
    rows = 7
    if packed:
        projection = torch.randn(rows, tokens, 3 * heads * head_dim, device="cuda", dtype=dtype)
        q, k, v = (x.view(rows, tokens, heads, head_dim) for x in projection.chunk(3, dim=-1))
    else:
        q, k, v = (torch.randn(rows, tokens, heads, head_dim, device="cuda", dtype=dtype) for _ in range(3))
    for tensor in (q, k, v):
        tensor[..., logical_dim:] = 0
    bias = torch.randn(1, heads, tokens, tokens, device="cuda", dtype=dtype)
    padded_bias = F.pad(bias, (0, (-tokens) % 8))
    lengths = torch.tensor([tokens, tokens - 1, min(tokens, 128), 1, 0, 31, tokens], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    lse = torch.empty(rows, tokens, heads, 1, device="cuda", dtype=torch.float32)
    bucket = {32: 240, 64: 208, 128: 80}[head_dim]
    executable = TriangleAttentionCubinExecutable(
        library, library.triangle_attention, 107, head_dim, bucket, dtype, packed
    )
    scale = logical_dim**-0.5

    def run() -> None:
        executable(q, k, v, padded_bias, lengths, output, lse, scale * math.log2(math.e), scale, rows)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    scores = torch.einsum("iqhd,ikhd->ihqk", q.float(), k.float()) * scale + bias[0].float()[None]
    keys = torch.arange(tokens, device="cuda")
    scores.masked_fill_(keys[None, None, None, :] >= lengths[:, None, None, None], -torch.inf)
    expected = torch.einsum("ihqk,ikhd->iqhd", scores.softmax(dim=-1), v.float())
    expected_lse = scores.logsumexp(dim=-1).permute(0, 2, 1)
    live = lengths > 0
    torch.testing.assert_close(output[live].float(), expected[live], atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(lse[live, :, :, 0], expected_lse[live], atol=1e-3, rtol=1e-4)
    assert output[~live].count_nonzero().item() == 0
    assert torch.isneginf(lse[~live]).all()
