# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_self_pairwise_attention_weights,
    load_self_pairwise_attention_weights_torch,
)
from test_utils.boltz.ref_attn import RefPairwiseSelfAttention

from bionemo_ir._torch.attention_backend import AttentionType, get_attention_backend
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir.utils import str_dtype_to_torch


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 32
    c_s: int = 384
    c_z: int = 128
    chunk_size: int = None
    chunk_dim: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(backend="VANILLA"),
        Scenario(backend="VANILLA", torch_dtype="bfloat16"),
        Scenario(backend="SDPA"),
        Scenario(backend="SDPA", torch_dtype="bfloat16"),
    ],
)
def test_pairwise_attention_backend(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(sc.backend, AttentionType.PAIRWISE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_attn = RefPairwiseSelfAttention.load_weights()
    ref_attn = ref_attn.to(device)

    weights_and_biases = create_self_pairwise_attention_weights(from_ref=ref_attn)

    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=ref_attn.num_heads,
        dtype=dtype,
        bias_proj=True,
        initial_norm=True,
        attn_backend=sc.backend,
    )
    load_self_pairwise_attention_weights_torch(attn, weights_and_biases, dtype=dtype)
    attn.to(device)

    attn_metadata = metadata_cls()
    s = torch.randn(bs, sc.seq_len, sc.c_s).to(device)
    z = torch.randn(bs, sc.seq_len, sc.seq_len, sc.c_z).to(device)
    mask = torch.randn(bs, sc.seq_len).to(device)

    with torch.inference_mode():
        ref_output_float = ref_attn(s, z, mask)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_attn = ref_attn.to(dtype)

        ref_output = ref_attn(s, z, mask)
        output = attn.forward(s, z, mask, attn_metadata)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-2, rtol=1e-3)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("keys", [37, 40])
def test_sdpa_sums_the_mask_into_aligned_rows(monkeypatch: pytest.MonkeyPatch, keys: int, batch: int) -> None:
    """cuDNN slows down on mask rows that are not 16-byte aligned, so the summed mask keeps aligned rows.

    With two batches of five samples, folding the mask to rank 4 copies it; the copy keeps them too.
    """
    from bionemo_ir._torch.attention_backend.pairwise_attention import SDPAPairwiseAttention, VanillaPairwiseAttention

    torch.manual_seed(0)
    heads, head_dim, mult = 4, 32, 5
    q, k, v = (torch.randn(batch, mult, keys, heads * head_dim, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    valid = torch.arange(keys, device="cuda")[None, :] < torch.tensor([[keys - 3], [keys - 9]], device="cuda")[:batch]
    mask_bias = ((~valid).float() * -1e9)[:, None, None, None, :]
    pair_bias = torch.randn(batch, 1, heads, keys, keys, dtype=torch.bfloat16, device="cuda")
    seen: list[torch.Tensor | None] = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(*args: object, attn_mask: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor:
        seen.append(attn_mask)
        return sdpa(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    actual = SDPAPairwiseAttention(0, heads, head_dim, heads).forward(q, k, v, biases=[mask_bias, pair_bias])

    (mask,) = seen
    assert mask.stride(-1) == 1
    assert all(stride % 8 == 0 for stride in mask.stride()[:-1] if stride)
    expected = VanillaPairwiseAttention(0, heads, head_dim, heads).forward(
        q.float(), k.float(), v.float(), biases=[mask_bias, pair_bias.float()]
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=1e-2)
