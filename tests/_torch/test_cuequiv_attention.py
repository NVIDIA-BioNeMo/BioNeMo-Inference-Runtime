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

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend.cuequiv import CuEquivAttention, CuEquivAttentionMetadata
from tensorrt_bionemo._torch.layers.triangle_nodes import TriangleAttentionNode
from tests._torch import SM_VERSION


def _inputs(
    n_valid: torch.Tensor,
    *,
    seq_len: int = 8,
    num_heads: int = 2,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_rows = n_valid.shape
    shape = (batch_size, num_rows, seq_len, num_heads * head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    valid = torch.arange(seq_len, device=device) < n_valid.to(device=device).unsqueeze(-1)
    mask_bias = (valid.to(dtype) - 1) * 1e9
    mask_bias = mask_bias[..., None, None, :].contiguous()
    pair_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=dtype, device=device)
    return q, k, v, mask_bias, pair_bias


@pytest.mark.parametrize(
    ("use_kv_lengths", "sm_version", "expect_lengths"),
    [
        (False, 100, False),
        (True, 90, False),
        (True, 100, True),
    ],
    ids=["dense-mask", "sm90-fallback", "sm100f-kv-lengths"],
)
def test_cuequiv_routes_mask_representation(monkeypatch, use_kv_lengths: bool, sm_version: int, expect_lengths: bool):
    n_valid = torch.tensor([[8, 5, 0], [4, 3, 1]], dtype=torch.int32)
    q, k, v, mask_bias, pair_bias = _inputs(
        n_valid,
        head_dim=8,
        dtype=torch.float16,
    )
    call = {}

    def _fake_kernel(q, k, v, bias, mask, actual_s_kv, scale=None):
        call.update(mask=mask, actual_s_kv=actual_s_kv, sm_scale=scale, q_contiguous=q.is_contiguous())
        aux = q.new_empty(q.shape[:-1], dtype=torch.float32)
        return torch.ones_like(q), aux, aux

    monkeypatch.setattr(torch.ops.cuequivariance, "triangle_attention", _fake_kernel)
    attention = CuEquivAttention(layer_idx=0, num_heads=2, head_dim=8, num_kv_heads=2)
    attention._sm_version = sm_version
    output = attention.forward(
        q,
        k,
        v,
        biases=[mask_bias, pair_bias],
        metadata=CuEquivAttentionMetadata(),
        use_kv_lengths=use_kv_lengths,
    )

    assert output.shape == (2, 3, 8, 2, 8)
    assert call["sm_scale"] == pytest.approx(8**-0.5)
    if expect_lengths:
        assert not call["q_contiguous"]
        assert call["mask"] is None
        assert call["actual_s_kv"].dtype == torch.int32
        assert call["actual_s_kv"].is_contiguous()
        torch.testing.assert_close(
            call["actual_s_kv"],
            n_valid,
        )
        assert torch.count_nonzero(output[0, 2]) == 0
        assert torch.all(output[0, :2] == 1)
    else:
        assert call["q_contiguous"]
        assert call["actual_s_kv"] is None
        expected_mask = (torch.arange(8) < n_valid.unsqueeze(-1))[..., None, None, :]
        torch.testing.assert_close(call["mask"], expected_mask)


@pytest.mark.parametrize("pair_mask_left_aligned", [False, True])
def test_triangle_node_forwards_mask_contract(pair_mask_left_aligned: bool):

    class _AttentionSpy(nn.Module):
        def forward(self, x, **kwargs):
            self.kwargs = kwargs
            return x

    node = TriangleAttentionNode(
        c_in=8,
        c_hidden=4,
        num_heads=2,
        dtype=torch.float32,
        skip_create_weights=True,
        pair_mask_left_aligned=pair_mask_left_aligned,
    )
    spy = _AttentionSpy()
    node.mha = spy
    x = torch.zeros(1, 2, 3, 8)
    node._mha_slice(
        x,
        mask_bias=torch.zeros(1, 2, 1, 1, 3),
        triangle_bias=torch.zeros(1, 2, 3, 3),
    )

    assert spy.kwargs["use_kv_lengths"] is pair_mask_left_aligned


@pytest.mark.skipif(
    SM_VERSION not in (100, 103),
    reason=f"cuEquivariance kv_lengths fast path requires SM100f (current SM{SM_VERSION})",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuequiv_sm100f_kv_lengths_matches_dense_mask(dtype: torch.dtype):
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 128
    num_heads = 4
    head_dim = 32
    n_valid = (seq_len - torch.arange(seq_len, device="cuda") % 32).reshape(batch_size, seq_len)
    q, k, v, mask_bias, pair_bias = _inputs(
        n_valid,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
    )
    attention = CuEquivAttention(layer_idx=0, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_heads)
    metadata = CuEquivAttentionMetadata()

    with torch.inference_mode():
        dense_output = attention.forward(
            q,
            k,
            v,
            biases=[mask_bias, pair_bias],
            metadata=metadata,
            use_kv_lengths=False,
        )
        lengths_output = attention.forward(
            q,
            k,
            v,
            biases=[mask_bias, pair_bias],
            metadata=metadata,
            use_kv_lengths=True,
        )

    torch.testing.assert_close(lengths_output, dense_output, atol=5e-2, rtol=5e-2)
