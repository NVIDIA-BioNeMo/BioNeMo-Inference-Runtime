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
"""SM100 triangle attention at D=32: split MMA issue and the paired-tail launch.

The source kernel splits a call whose last 256-row m_block holds at most 128 rows into a
regular launch and a tail launch. Above 384 keys the tail pairs heads 2g and 2g+1 as one
two-tile item (an odd head count keeps the single-tile kernel); at 384 keys and below the
call keeps the single-issuer kernel. These cases check the output and the LSE of every head
against an fp32 reference on both sides of both boundaries.
"""

import math

import pytest
import torch

from tests._torch import SM_VERSION, source_module_available

_SOURCE_MODULE = "bionemo_ir._torch.attention_backend.triangle_attention._source"
_HEAD_DIM = 32

pytestmark = pytest.mark.skipif(
    SM_VERSION not in (100, 103) or not source_module_available(_SOURCE_MODULE),
    reason="the SM100 split-issue and paired-tail launches exist in the source kernel on SM100/SM103",
)


def _rows_for_sm100_bucket(tokens: int) -> int:
    """Rows I that put S = sqrt(I * J) on the SM100 anchor (S >= 193) with few rows."""
    return max(8, math.ceil(200**2 / tokens))


def _lengths(i_dim: int, tokens: int) -> torch.Tensor:
    """Full rows plus boundary cases: one key short, a whole number of key tiles, one key, none."""
    lengths = torch.full((1, i_dim), tokens, dtype=torch.int32)
    lengths[0, 1] = tokens - 1
    lengths[0, 2] = max(128, (tokens // 128) * 128)
    lengths[0, 3] = 1
    lengths[0, 4] = 0
    return lengths.cuda()


def _reference(q, k, v, bias, lengths, heads):
    """fp32 output [I, J, H, D] and natural-log LSE [I, J, H] of a left-masked triangle attention."""
    i_dim, tokens = q.shape[1], q.shape[2]
    qf, kf, vf = (t.float().view(i_dim, tokens, heads, _HEAD_DIM) for t in (q, k, v))
    scores = torch.einsum("iqhd,ikhd->ihqk", qf, kf) * _HEAD_DIM**-0.5 + bias[0].float()[None]
    keys = torch.arange(tokens, device=q.device)
    valid = keys[None, :] < lengths[0][:, None]  # [I, J]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)
    out = torch.einsum("ihqk,ikhd->iqhd", torch.softmax(scores, dim=-1), vf)
    return out, lse


@pytest.mark.parametrize("heads", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "tokens",
    [320, 384, 385, 520, 640, 641],
    ids=["single-issuer-tail64", "single-issuer-tail128", "regular129", "tail8", "tail128", "regular129-6blocks"],
)
def test_sm100_tail_launch_output_and_lse_per_head(heads: int, tokens: int):
    from bionemo_ir._torch.attention_backend.triangle_attention import _config
    from bionemo_ir._torch.attention_backend.triangle_attention.cutedsl import TriangleAttentionCuTeLeftMask

    torch.manual_seed(tokens * 16 + heads)
    i_dim = _rows_for_sm100_bucket(tokens)
    bucket = _config.get_nearest_bucket(SM_VERSION, _HEAD_DIM, round(math.sqrt(i_dim * tokens)))
    assert _config.get_kernel_config(SM_VERSION, _HEAD_DIM, bucket).arch == "sm100"

    width = heads * _HEAD_DIM
    proj = torch.randn(1, i_dim, tokens, 3 * width, dtype=torch.bfloat16, device="cuda")
    q, k, v = proj.split(width, dim=-1)
    bias = torch.randn(1, heads, tokens, tokens, dtype=torch.bfloat16, device="cuda")
    lengths = _lengths(i_dim, tokens)
    lse = torch.full((i_dim, tokens, heads, 1), float("nan"), dtype=torch.float32, device="cuda")

    attn = TriangleAttentionCuTeLeftMask(0, heads, _HEAD_DIM, num_kv_heads=heads)
    out = attn.forward(q, k, v, biases=[lengths, bias], output_lse=lse)
    torch.cuda.synchronize()

    expected, expected_lse = _reference(q, k, v, bias, lengths, heads)
    got = out.view(i_dim, tokens, heads, _HEAD_DIM).float()
    live = lengths[0] > 0
    torch.testing.assert_close(got[live], expected[live], atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(lse[live, :, :, 0], expected_lse[live], atol=1e-3, rtol=1e-4)
    # A row with no valid key outputs zeros and an LSE of -inf.
    assert torch.count_nonzero(got[~live]).item() == 0
    assert torch.isneginf(lse[~live]).all()
