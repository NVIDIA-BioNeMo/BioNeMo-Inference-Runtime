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
"""A CuTeDSL ``AttentionPairBias`` pads its heads when its weights load, never in forward.

The checks stop at the projections, so they need a CUDA device but no attention kernel.
"""

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.layers.attention import AttentionPairBias
from tests._torch import init_module_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HEADS = 4
HEAD_DIM = 24  # padded to 32 for the kernel
PADDED_DIM = 32
C_S = HEADS * HEAD_DIM


def _layer(backend: str = "CuTeDSL", c_s: int = C_S) -> AttentionPairBias:
    layer = AttentionPairBias(
        layer_idx=0, c_s=c_s, c_z=16, num_heads=HEADS, dtype=torch.float32, attn_backend=backend
    ).cuda()
    init_module_weights(layer, std=0.5)
    layer.in_proj.bias.normal_()
    return layer


def _per_head(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, head_dim: int) -> torch.Tensor:
    """``in_proj`` applied to *x*, as ``[B, N, 4, H, head_dim]`` for the q, gate, k and v segments."""
    return F.linear(x, weight, bias).unflatten(-1, (4, HEADS, head_dim))


def test_post_load_weights_zero_pads_each_head() -> None:
    layer = _layer()
    assert layer._padded_in_weight is None
    layer.post_load_weights()

    x = torch.randn(2, 5, C_S, device="cuda")
    padded = _per_head(x, layer._padded_in_weight, layer._padded_in_bias, PADDED_DIM)
    plain = _per_head(x, layer.in_proj.weight, layer.in_proj.bias, HEAD_DIM)
    torch.testing.assert_close(padded[..., :HEAD_DIM], plain)
    assert not padded[..., HEAD_DIM:].any()

    heads = torch.randn(2, 5, HEADS, HEAD_DIM, device="cuda")
    projected = F.linear(F.pad(heads, (0, PADDED_DIM - HEAD_DIM)).flatten(-2), layer._padded_out_weight)
    torch.testing.assert_close(projected, F.linear(heads.flatten(-2), layer.proj_o.weight))


def test_load_state_dict_pads_and_keeps_the_checkpoint_layout() -> None:
    source, target = _layer(), _layer()
    target.load_state_dict(source.state_dict(), strict=True)
    source.post_load_weights()

    for name in ("_padded_in_weight", "_padded_in_bias", "_padded_out_weight"):
        torch.testing.assert_close(getattr(target, name), getattr(source, name), atol=0, rtol=0)
    assert target.state_dict().keys() == _layer("SDPA").state_dict().keys()


def test_a_reload_rebuilds_the_copies() -> None:
    layer = _layer()
    layer.post_load_weights()
    before = layer._padded_in_weight.clone()

    layer.in_proj.weight.mul_(2)
    layer.post_load_weights()
    torch.testing.assert_close(layer._padded_in_weight, 2 * before)


def test_the_copies_follow_the_layer_across_dtypes() -> None:
    layer = _layer()
    layer.post_load_weights()
    layer.to(torch.bfloat16)
    assert {layer._padded_in_weight.dtype, layer._padded_in_bias.dtype, layer._padded_out_weight.dtype} == {
        torch.bfloat16
    }


def test_forward_before_post_load_weights_fails_loudly() -> None:
    layer = _layer()
    s = torch.randn(1, 8, C_S, device="cuda")
    pair_bias = torch.randn(1, HEADS, 8, 8, device="cuda")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="post_load_weights"):
        layer(s, pair_bias, torch.ones(1, 8, device="cuda"))


@pytest.mark.parametrize(("backend", "c_s"), [("CuTeDSL", HEADS * 32), ("SDPA", C_S)], ids=["aligned", "sdpa"])
def test_layers_without_padded_heads_keep_no_copies(backend: str, c_s: int) -> None:
    layer = _layer(backend, c_s)
    layer.post_load_weights()
    assert not list(layer.buffers())
