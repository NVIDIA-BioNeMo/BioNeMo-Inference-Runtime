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
    create_msa_layer_weights, create_msa_module_weights,
    load_msa_layer_weights_torch, load_msa_module_weights_torch)
from test_utils.boltz.ref_layers import RefMSALayer, RefMSAModule

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.attention_backend.utils import \
    precompute_pair_masks
from tensorrt_bionemo._torch.modules.boltz.trunk import MSALayer, MSAModule
from tensorrt_bionemo.configs import MSAModuleConfig
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV"),
    Scenario(triangle_attn_backend="CuTeDSL", torch_dtype="bfloat16"),
])
def test_msa_layer(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_mod = RefMSALayer.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_layer_weights(from_ref=ref_mod)

    msa_layer = MSALayer(msa_s=ref_mod.msa_s,
                         token_z=ref_mod.token_z,
                         pairwise_head_width=ref_mod.pairwise_head_width,
                         pairwise_num_heads=ref_mod.pairwise_num_heads,
                         triangle_attn_backend=sc.triangle_attn_backend,
                         dtype=dtype)
    load_msa_layer_weights_torch(msa_layer, weights_and_biases, dtype=dtype)
    msa_layer.to(device)

    z = torch.randn(bs, 64, 64, ref_mod.token_z, dtype=torch.float32).cuda()
    m = torch.randn(bs, 32, 64, ref_mod.msa_s, dtype=torch.float32).cuda()
    # Real left-aligned mask with n_valid in [N/2, N]: keeps the masking
    # path exercised while avoiding pathologically tight masks (n_valid = 1
    # would leave 63/64 rows fully masked, where the PyTorch reference
    # collapses to NaN via softmax-of-all-(-inf) and the CuTeDSL left-mask
    # kernel runs an ``actual_s_kv = 1`` no-op tile of garbage).
    seq_mask = make_left_aligned_mask(bs,
                                      64,
                                      dtype=torch.float32,
                                      device=device,
                                      min_valid=32)
    token_mask = seq_mask[..., None] * seq_mask[..., None, :]
    msa_mask = torch.randint(0, 2, (bs, 32, 64),
                             dtype=torch.float32).to(device)

    triangle_metadata_cls = get_attention_backend(
        sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata
    with torch.inference_mode():
        ref_z_float, ref_m_float = ref_mod(z, m, token_mask, msa_mask)
        z = z.to(dtype)
        m = m.to(dtype)
        token_mask = token_mask.to(dtype)
        msa_mask = msa_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z, ref_m = ref_mod(z, m, token_mask, msa_mask)
        output_z, output_m = msa_layer(z,
                                       m,
                                       token_mask,
                                       msa_mask,
                                       attn_metadata=triangle_metadata_cls())

    assert ref_z.shape == output_z.shape
    assert ref_m.shape == output_m.shape

    # With a real left-aligned mask, padded query rows in
    # ``pair_weighted_averaging`` softmax over fully-masked keys -> NaN in
    # the PyTorch reference; the CuTeDSL left-mask kernel sees
    # ``actual_s_kv`` clamped to >= 1 and emits arithmetic garbage there.
    # Both implementations agree on the *valid* sub-block; only compare
    # there.
    z_keep = token_mask.float().unsqueeze(-1)  # [B, N, N, 1]
    m_keep = seq_mask.float()[:, None, :, None]  # [B, 1, N, 1]

    def _masked(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0,
                                neginf=0.0) * keep

    if dtype == torch.float32:
        torch.testing.assert_close(_masked(ref_z, z_keep),
                                   _masked(output_z, z_keep),
                                   atol=1e-3,
                                   rtol=1e-4)
        torch.testing.assert_close(_masked(ref_m, m_keep),
                                   _masked(output_m, m_keep),
                                   atol=1e-3,
                                   rtol=1e-4)
    else:
        # Asymmetric tolerance: ``ours`` must be no more than ``tol_mult``×
        # worse than the bf16 reference's distance to the fp32 ground
        # truth.  This passes when ``ours`` is *more* accurate than the ref
        # (which happens for the CuTeDSL left-mask path: the kernel keeps
        # internal fp32 accumulation while the bf16 reference module's
        # chain of bf16 ops loses precision).
        tol_mult = 2.0
        for name, out_v, ref_v, ref_f32_v, keep in [
            ("m", output_m, ref_m, ref_m_float, m_keep),
            ("z", output_z, ref_z, ref_z_float, z_keep),
        ]:
            d_out = _masked(out_v, keep) - _masked(ref_f32_v, keep)
            d_ref = _masked(ref_v, keep) - _masked(ref_f32_v, keep)
            d_out_max = torch.max(torch.abs(d_out))
            d_ref_max = torch.max(torch.abs(d_ref))
            assert d_out_max <= tol_mult * d_ref_max + 1e-3, (
                f"{name} max: ours_diff={d_out_max.item()}, "
                f"ref_diff={d_ref_max.item()}")
            d_out_mean = torch.mean(torch.abs(d_out))
            d_ref_mean = torch.mean(torch.abs(d_ref))
            assert d_out_mean <= tol_mult * d_ref_mean + 1e-3, (
                f"{name} mean: ours_diff={d_out_mean.item()}, "
                f"ref_diff={d_ref_mean.item()}")


@pytest.mark.parametrize("sc", [
    Scenario(),
])
def test_msa_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_mod = RefMSAModule.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_module_weights(from_ref=ref_mod)

    config = MSAModuleConfig(architecture="msa_module",
                             msa_s=ref_mod.msa_s,
                             token_z=ref_mod.token_z,
                             token_s=ref_mod.token_s,
                             msa_blocks=ref_mod.msa_blocks,
                             num_tokens=ref_mod.num_tokens,
                             pairwise_head_width=ref_mod.pairwise_head_width,
                             pairwise_num_heads=ref_mod.pairwise_num_heads,
                             version="v2",
                             dtype=sc.torch_dtype)
    msa_module = MSAModule(config)
    load_msa_module_weights_torch(msa_module, weights_and_biases, dtype=dtype)
    msa_module.to(device)

    B = 1
    N = 117
    N_msa = 83
    token_z = ref_mod.token_z
    token_s = ref_mod.token_s

    z = torch.randn(B, N, N, token_z, dtype=torch.float32).cuda()
    emb = torch.randn(B, N, token_s, dtype=torch.float32).cuda()
    msa = torch.randint(0, 33, (B, N_msa, N), dtype=torch.int64).cuda()
    has_deletion = torch.randint(0, 2, (B, N_msa, N),
                                 dtype=torch.float32).cuda()
    deletion_value = torch.randn(B, N_msa, N, dtype=torch.float32).cuda()
    msa_paired = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    msa_mask = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    token_pad_mask = make_left_aligned_mask(B,
                                            N,
                                            dtype=torch.float32,
                                            device="cuda")
    pair_mask = token_pad_mask[:, :, None] * token_pad_mask[:, None, :]

    triangle_metadata_cls = get_attention_backend(
        "VANILLA", AttentionType.TRIANGLE).Metadata
    with torch.inference_mode():
        ref_z_float = ref_mod(z, emb, msa, has_deletion, deletion_value,
                              msa_paired, msa_mask, token_pad_mask)
        z = z.to(dtype)
        emb = emb.to(dtype)
        has_deletion = has_deletion.to(dtype)
        deletion_value = deletion_value.to(dtype)
        msa_paired = msa_paired.to(dtype)
        msa_mask = msa_mask.to(dtype)
        token_pad_mask = token_pad_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z = ref_mod(z, emb, msa, has_deletion, deletion_value, msa_paired,
                        msa_mask, token_pad_mask)
        output_z = msa_module(z, emb, msa, has_deletion, deletion_value,
                              msa_paired, msa_mask, pair_mask)

    assert ref_z.shape == output_z.shape
    torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests for precomputed pair masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV"),
    Scenario(triangle_attn_backend="CUEQUIV", torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL", torch_dtype="bfloat16"),
])
def test_msa_layer_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_mod = RefMSALayer.load_weights()
    ref_mod = ref_mod.to(device)
    weights_and_biases = create_msa_layer_weights(from_ref=ref_mod)

    msa_layer = MSALayer(msa_s=ref_mod.msa_s,
                         token_z=ref_mod.token_z,
                         pairwise_head_width=ref_mod.pairwise_head_width,
                         pairwise_num_heads=ref_mod.pairwise_num_heads,
                         triangle_attn_backend=sc.triangle_attn_backend,
                         dtype=dtype)
    load_msa_layer_weights_torch(msa_layer, weights_and_biases, dtype=dtype)
    msa_layer.to(device)

    z = torch.randn(bs, 64, 64, ref_mod.token_z, dtype=dtype, device=device)
    m = torch.randn(bs, 32, 64, ref_mod.msa_s, dtype=dtype, device=device)
    # Build ``token_mask`` as the outer product of a left-aligned 1D seq
    # mask. The CuTeDSL precompute (and the underlying left-mask kernel)
    # require this layout; for other backends a left-aligned mask is still
    # a valid input, so this works for every scenario.
    seq_mask = make_left_aligned_mask(bs, 64, dtype=dtype, device=device)
    token_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(dtype)
    msa_mask = torch.randint(0,
                             2, (bs, 32, 64),
                             dtype=torch.float32,
                             device=device).to(dtype)

    triangle_metadata_cls = get_attention_backend(
        sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata

    precomputed = precompute_pair_masks(sc.triangle_attn_backend,
                                        token_mask,
                                        inf=msa_layer.inf,
                                        dtype=dtype)

    # MSALayer accumulates into z/m *in place* (memory opt), so it consumes its inputs. Give each
    # call its own copy, otherwise the second call would run on the first call's mutated z/m rather
    # than the same inputs -- the two mask paths must be compared on identical inputs.
    with torch.inference_mode():
        out_z, out_m = msa_layer(z.clone(),
                                 m.clone(),
                                 token_mask,
                                 msa_mask,
                                 attn_metadata=triangle_metadata_cls())
        out_z_pre, out_m_pre = msa_layer(z.clone(),
                                         m.clone(),
                                         token_mask,
                                         msa_mask,
                                         attn_metadata=triangle_metadata_cls(),
                                         precomputed_masks=precomputed)

    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)
    torch.testing.assert_close(out_m_pre, out_m, atol=0, rtol=0)
