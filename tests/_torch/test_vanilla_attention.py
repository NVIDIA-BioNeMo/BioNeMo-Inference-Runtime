# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from test_utils.boltz.ref_attn import plain_mha

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.pairwise_attention_cute import (
    PairwiseAttentionCuTe, PairwiseAttentionCuTeMetadata)
from tensorrt_bionemo._torch.attention_backend.triangle_attention_cute import (
    TriangleAttentionCuTe, TriangleAttentionCuTeMetadata)
from tensorrt_bionemo._torch.attention_backend.vanilla import (
    VanillaPairwiseAttention, VanillaTriangleAttention)
from tests._torch import skip_if_no_cutedsl


def _pw_meta(kv_packed: bool) -> PairwiseAttentionCuTeMetadata:
    m = PairwiseAttentionCuTeMetadata()
    m.kv_packed = kv_packed
    return m


def _tri_meta(qkv_packed: bool) -> TriangleAttentionCuTeMetadata:
    m = TriangleAttentionCuTeMetadata()
    m.qkv_packed = qkv_packed
    return m


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_triangle(seq_len, dtype):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    bs = 1

    q = torch.randn(bs, seq_len, seq_len,
                    num_heads * head_dim).cuda().to(dtype)
    k = torch.randn(bs, seq_len, seq_len,
                    num_heads * head_dim).cuda().to(dtype)
    v = torch.randn(bs, seq_len, seq_len,
                    num_heads * head_dim).cuda().to(dtype)

    vanilla_attn = VanillaTriangleAttention(layer_idx,
                                            num_heads,
                                            head_dim,
                                            num_kv_heads=num_heads)

    biases = [
        torch.randn(bs, seq_len, 1, 1, seq_len).cuda().to(dtype),
        torch.randn(bs, 1, num_heads, seq_len, seq_len).cuda().to(dtype)
    ]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(
        q, k, v, biases=[biases[0], biases[1].squeeze(1)], metadata=metadata)
    assert vanilla_out.shape == (bs, seq_len, seq_len, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape
    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out,
                                   plain_out,
                                   atol=5e-2,
                                   rtol=1e-4)


@pytest.mark.parametrize("batch_size", [16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_pairwise(batch_size, dtype):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    q_size = 32
    kv_size = 128

    q = torch.randn(batch_size, q_size, num_heads * head_dim).to(dtype)
    k = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)
    v = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)

    vanilla_attn = VanillaPairwiseAttention(layer_idx,
                                            num_heads,
                                            head_dim,
                                            num_kv_heads=num_heads)

    biases = [
        torch.randn(batch_size, 1, 1, kv_size),
        torch.randn(batch_size, num_heads, q_size, kv_size)
    ]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(q,
                                       k,
                                       v,
                                       biases=biases,
                                       metadata=metadata)
    assert vanilla_out.shape == (batch_size, q_size, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape

    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out,
                                   plain_out,
                                   atol=5e-2,
                                   rtol=1e-4)


# ---------------------------------------------------------------------------
# CuTeDSL pairwise attention vs Vanilla — self & cross attention, with mult
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,q_size,kv_size,num_heads,head_dim,mult", [
    (1, 32, 32, 4, 32, 1),
    (1, 32, 32, 4, 32, 5),
    (1, 32, 128, 4, 32, 1),
    (1, 32, 128, 4, 32, 5),
    (1, 117, 117, 16, 48, 1),
    (1, 117, 117, 16, 48, 5),
    (29, 32, 128, 4, 32, 1),
    (29, 32, 128, 4, 32, 5),
],
                         ids=[
                             "self-B1-S32-H4D32-m1",
                             "self-B1-S32-H4D32-m5",
                             "cross-B1-Q32K128-H4D32-m1",
                             "cross-B1-Q32K128-H4D32-m5",
                             "self-B1-S117-H16D48-m1",
                             "self-B1-S117-H16D48-m5",
                             "cross-B29-Q32K128-H4D32-m1",
                             "cross-B29-Q32K128-H4D32-m5",
                         ])
@pytest.mark.parametrize("kv_packed", [False, True],
                         ids=["separate", "packed"])
def test_cutedsl_vs_vanilla_pairwise(batch_size, q_size, kv_size, num_heads,
                                     head_dim, mult, kv_packed):
    """Compare CuTeDSL pairwise attention against vanilla reference."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = torch.bfloat16
    device = torch.device('cuda')

    B = batch_size
    B_flat = B * mult
    Sq, Sk, H, D = q_size, kv_size, num_heads, head_dim

    q = torch.randn(B_flat, Sq, H * D, dtype=dtype, device=device)

    if kv_packed:
        kv = torch.randn(B_flat, Sk, 2, H * D, dtype=dtype, device=device)
        k = kv[..., 0, :]
        v = kv[..., 1, :]
    else:
        k = torch.randn(B_flat, Sk, H * D, dtype=dtype, device=device)
        v = torch.randn(B_flat, Sk, H * D, dtype=dtype, device=device)

    binary_mask = torch.randint(0,
                                2, (B, Sk),
                                dtype=torch.float32,
                                device=device)
    binary_mask[:, 0] = 1.0
    pair_bias = torch.randn(B, H, Sq, Sk, dtype=dtype, device=device)

    additive_mask = (1.0 - binary_mask) * -1e9
    if mult > 1:
        additive_mask_expanded = additive_mask.unsqueeze(1).unsqueeze(
            2).unsqueeze(3).expand(B, mult, 1, 1,
                                   Sk).reshape(B_flat, 1, 1, Sk)
        pair_bias_expanded = pair_bias.unsqueeze(1).expand(
            B, mult, H, Sq, Sk).reshape(B_flat, H, Sq, Sk)
    else:
        additive_mask_expanded = additive_mask[:, None, None, :]
        pair_bias_expanded = pair_bias

    vanilla_attn = VanillaPairwiseAttention(0, H, D, num_kv_heads=H)
    vanilla_out = vanilla_attn.forward(
        q,
        k.contiguous(),
        v.contiguous(),
        biases=[additive_mask_expanded, pair_bias_expanded],
        metadata=AttentionMetadata())

    cute_attn = PairwiseAttentionCuTe(0, H, D, num_kv_heads=H)
    cute_out = cute_attn.forward(q,
                                 k,
                                 v,
                                 biases=[binary_mask, pair_bias],
                                 metadata=_pw_meta(kv_packed))

    assert cute_out.shape == vanilla_out.shape, (
        f"Shape mismatch: cute={cute_out.shape}, vanilla={vanilla_out.shape}")
    diff_max = torch.max(torch.abs(cute_out.float() - vanilla_out.float()))
    diff_mean = torch.mean(torch.abs(cute_out.float() - vanilla_out.float()))
    assert diff_max < 1e-1, f"Max diff {diff_max} too large"
    assert diff_mean < 1e-2, f"Mean diff {diff_mean} too large"


# ---------------------------------------------------------------------------
# CuTeDSL triangle attention vs Vanilla — non-trivial J (J != J_padded)
# ---------------------------------------------------------------------------


# bf16 alignment = 128 bits / 16 bits = 8 elements.
# J values that are NOT multiples of 8 exercise the J_padded != J path.
@pytest.mark.parametrize("bs,I,J,num_heads,head_dim", [
    (1, 32, 13, 4, 32),
    (1, 32, 29, 4, 32),
    (1, 13, 13, 4, 32),
    (1, 29, 29, 4, 32),
    (1, 16, 33, 4, 32),
    (1, 16, 65, 4, 32),
    (1, 32, 32, 4, 32),
    (1, 64, 64, 4, 32),
    (1, 17, 117, 4, 32),
],
                         ids=[
                             "B1-I32-J13-H4D32",
                             "B1-I32-J29-H4D32",
                             "B1-I13-J13-H4D32",
                             "B1-I29-J29-H4D32",
                             "B1-I16-J33-H4D32",
                             "B1-I16-J65-H4D32",
                             "B1-I32-J32-H4D32-aligned",
                             "B1-I64-J64-H4D32-aligned",
                             "B1-I17-J117-H4D32",
                         ])
@pytest.mark.parametrize("qkv_packed", [False, True],
                         ids=["separate", "packed"])
def test_cutedsl_vs_vanilla_triangle(bs, I, J, num_heads, head_dim,
                                     qkv_packed):
    """Compare CuTeDSL triangle attention against vanilla reference.

    Non-multiple-of-8 J values force the kernel to pad (J_padded != J),
    exercising the pad/unpad paths in the backend.
    Tests both qkv_packed=True (non-contiguous slices from a fused buffer)
    and qkv_packed=False (independent contiguous tensors).
    """
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = torch.bfloat16
    device = torch.device('cuda')

    H, D = num_heads, head_dim

    if qkv_packed:
        qkv = torch.randn(bs, I, J, 3, H * D, dtype=dtype, device=device)
        q = qkv[..., 0, :]
        k = qkv[..., 1, :]
        v = qkv[..., 2, :]
    else:
        q = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)
        k = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)
        v = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)

    binary_mask = torch.randint(0,
                                2, (bs, I, J),
                                dtype=torch.float32,
                                device=device)
    binary_mask[..., 0] = 1.0
    pair_bias = torch.randn(bs, H, J, J, dtype=dtype, device=device)

    additive_mask = (1.0 - binary_mask) * -1e9
    additive_mask_vanilla = additive_mask.unsqueeze(-2).unsqueeze(-2)

    vanilla_attn = VanillaTriangleAttention(0, H, D, num_kv_heads=H)
    vanilla_out = vanilla_attn.forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        biases=[additive_mask_vanilla, pair_bias],
        metadata=AttentionMetadata())

    cute_attn = TriangleAttentionCuTe(0, H, D, num_kv_heads=H)
    cute_out = cute_attn.forward(q,
                                 k,
                                 v,
                                 biases=[binary_mask, pair_bias],
                                 metadata=_tri_meta(qkv_packed))

    assert cute_out.shape == vanilla_out.shape, (
        f"Shape mismatch: cute={cute_out.shape}, vanilla={vanilla_out.shape}")
    diff_max = torch.max(torch.abs(cute_out.float() - vanilla_out.float()))
    diff_mean = torch.mean(torch.abs(cute_out.float() - vanilla_out.float()))
    assert diff_max < 1e-1, f"Max diff {diff_max} too large"
    assert diff_mean < 1e-2, f"Mean diff {diff_mean} too large"
