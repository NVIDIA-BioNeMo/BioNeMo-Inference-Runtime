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
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_torch)
from test_utils.boltz.ref_layers import RefPairformerLayer

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.attention_backend.utils import (
    PrecomputedPairMasks, precompute_pair_masks, precompute_single_masks)
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerLayerV1
from tensorrt_bionemo.mapping import Mapping
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl_single


def _skip_if_cutedsl(*backend_names: str):
    for name in backend_names:
        _skip_if_cutedsl_single(name)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    triangle_attn_backend: str
    pairwise_attn_backend: str
    seq_len: int = 32
    chunk_size: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="VANILLA",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="VANILLA",
             pairwise_attn_backend="CuTeDSL",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL",
             pairwise_attn_backend="CuTeDSL",
             torch_dtype="bfloat16"),
])
def test_pairformer_layer(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend, sc.pairwise_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    pairwise_metadata_cls = get_attention_backend(
        sc.pairwise_attn_backend, AttentionType.PAIRWISE).Metadata
    triangle_metadata_cls = get_attention_backend(
        sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_layer = RefPairformerLayer.load_weights()
    ref_layer = ref_layer.to(device)

    weights_and_biases = create_pairformer_layer_weights(from_ref=ref_layer)

    layer = PairformerLayerV1(
        layer_idx=0,
        token_s=ref_layer.token_s,
        token_z=ref_layer.token_z,
        num_heads=ref_layer.num_heads,
        pairwise_head_width=ref_layer.pairwise_head_width,
        pairwise_num_heads=ref_layer.pairwise_num_heads,
        dtype=dtype,
        triangle_attn_backend=sc.triangle_attn_backend,
        pairwise_attn_backend=sc.pairwise_attn_backend,
        skip_create_weights=False,
        attention_initial_norm=True  # Pairformer v1 uses attention_initial_norm
    )
    layer.to(device)
    load_pairformer_layer_weights_torch(layer, weights_and_biases, dtype)

    s = torch.randn(bs, sc.seq_len, ref_layer.token_s).to(device)
    z = torch.randn(bs, sc.seq_len, sc.seq_len, ref_layer.token_z).to(device)
    # Use a real left-aligned mask with substantial padding (~half of the
    # rows): keeps the bf16 ratio check meaningful (mask cuts down softmax
    # accumulation noise) while staying compatible with the CuTeDSL
    # left-mask kernel's ``actual_s_kv`` contract.  Fully-padded query rows
    # exist; the kernel handles them (early-exit work tile) and the
    # PyTorch reference produces NaN there, which we filter out below.
    mask = make_left_aligned_mask(bs,
                                  sc.seq_len,
                                  dtype=torch.float32,
                                  device=device,
                                  min_valid=sc.seq_len // 2)
    pair_mask = mask[..., None] * mask[..., None, :]

    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=Mapping()),
        "pairwise_attn": pairwise_metadata_cls(mapping=Mapping()),
    }

    with torch.inference_mode():
        ref_s_float, ref_z_float = ref_layer(s, z, mask, pair_mask)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        pair_mask = pair_mask.to(dtype)
        # ref_layer = ref_layer.to(dtype)
        # cast all modules to dtype, except norm_out in tri_mul
        for name, module in ref_layer.named_modules():
            if name.startswith("tri_attn_start.") or \
                name.startswith("tri_attn_end.") or \
                name.startswith("transition_s.") or \
                name.startswith("transition_z.") or \
                name.startswith("attention."):
                module.to(dtype)
            elif name.startswith("tri_mul_out.") or name.startswith(
                    "tri_mul_in."):
                if not "norm_out" in name and not "p_out" in name and not "g_out" in name:
                    module.to(dtype)
                else:
                    module.float()

        ref_s, ref_z = ref_layer(s, z, mask, pair_mask)
        output_s, output_z = layer(s,
                                   z,
                                   mask,
                                   pair_mask,
                                   attn_metadatas=attn_metadatas)

    assert ref_s.shape == output_s.shape
    assert ref_z.shape == output_z.shape

    # Filter out fully-padded query rows: the PyTorch reference's softmax
    # produces NaN at those rows, and the CuTeDSL left-mask kernel emits an
    # early-exit (zero/garbage) tile.  Both implementations agree on the
    # valid sub-block; only compare there.
    s_keep = mask.float().unsqueeze(-1)  # [B, N, 1]
    z_keep = pair_mask.float().unsqueeze(-1)  # [B, N, N, 1]

    def _masked(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0,
                                neginf=0.0) * keep

    if dtype == torch.float32:
        torch.testing.assert_close(_masked(ref_s, s_keep),
                                   _masked(output_s, s_keep),
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(_masked(ref_z, z_keep),
                                   _masked(output_z, z_keep),
                                   atol=1e-3,
                                   rtol=1e-3)
    else:
        # Asymmetric tolerance: ``ours`` must be no more than ``tol_mult``×
        # worse than the bf16 reference's distance to the fp32 ground
        # truth.  The previous ``|d0-d1| / min(d0, d1) <= 0.5`` ratio
        # check was symmetric and would *fail* when ``ours`` is
        # significantly *more* accurate than the reference (which happens
        # when the CuTeDSL left-mask kernel preserves higher-precision
        # accumulation than the eager bf16 reference's chain of bf16 ops).
        # See test_openfold2_evoformer_block / test_boltz_msa_module which
        # already use this asymmetric form.
        tol_mult = 2.0
        for name, out_v, ref_v, ref_f32_v, keep in [
            ("s", output_s, ref_s, ref_s_float, s_keep),
            ("z", output_z, ref_z, ref_z_float, z_keep),
        ]:
            diff_ours = torch.max(
                torch.abs(_masked(out_v, keep) - _masked(ref_f32_v, keep)))
            diff_ref = torch.max(
                torch.abs(_masked(ref_v, keep) - _masked(ref_f32_v, keep)))
            assert diff_ours <= tol_mult * diff_ref + 1e-3, (
                f"{name}: ours_diff={diff_ours.item()}, "
                f"ref_diff={diff_ref.item()}")


# ---------------------------------------------------------------------------
# Tests for precomputed pair masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", ["VANILLA", "CUEQUIV", "TRIFAST"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_precompute_pair_masks(backend_name: str, dtype: torch.dtype):
    """Verify shapes, dtypes, and values of the default precomputed mask tensors."""
    device = torch.device("cuda")
    B, I, J = 2, 16, 16
    inf_val = 1e9

    pair_mask = torch.randint(0,
                              2, (B, I, J),
                              dtype=torch.float32,
                              device=device)
    precomputed = precompute_pair_masks(backend_name,
                                        pair_mask,
                                        inf=inf_val,
                                        dtype=dtype)

    assert isinstance(precomputed, PrecomputedPairMasks)
    assert precomputed.pair_mask is pair_mask
    assert precomputed.mask_bias.shape == (B, I, 1, 1, J)
    assert precomputed.mask_bias_transposed.shape == (B, J, 1, 1, I)
    assert precomputed.mask_bias.dtype == dtype
    assert precomputed.mask_bias_transposed.dtype == dtype

    mask_typed = pair_mask.to(dtype)
    expected_bias = (inf_val * (mask_typed - 1))[..., :, None, None, :]
    expected_bias_t = (inf_val * (mask_typed.transpose(-2, -1) - 1))[..., :,
                                                                     None,
                                                                     None, :]
    torch.testing.assert_close(precomputed.mask_bias, expected_bias)
    torch.testing.assert_close(precomputed.mask_bias_transposed,
                               expected_bias_t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [16, 13])
def test_precompute_pair_masks_cutedsl_left_aligned(dtype: torch.dtype,
                                                    seq_len: int):
    """CuTeDSL precompute returns int32 ``actual_s_kv`` per row in ``mask_bias``.

    Left-aligned ``pair_mask = seq_mask[..., None] * seq_mask[..., None, :]``
    case: ``mask_bias[b, i] == n_valid[b]`` for valid rows and ``0`` for
    padded rows.
    """
    device = torch.device("cuda")
    B, I, J = 2, seq_len, seq_len
    inf_val = 1e9

    n_valid = torch.tensor([seq_len, max(1, seq_len - 3)],
                           dtype=torch.int64,
                           device=device)
    seq_mask = (torch.arange(seq_len, device=device).view(1, seq_len)
                < n_valid.view(B, 1)).to(torch.float32)
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

    precomputed = precompute_pair_masks("CuTeDSL",
                                        pair_mask,
                                        inf=inf_val,
                                        dtype=dtype)

    assert isinstance(precomputed, PrecomputedPairMasks)
    assert precomputed.pair_mask is pair_mask

    assert precomputed.mask_bias.dtype == torch.int32
    assert precomputed.mask_bias_transposed.dtype == torch.int32

    assert precomputed.mask_bias.shape == (B, I)
    assert precomputed.mask_bias_transposed.shape == (B, J)

    expected = torch.where(seq_mask.bool(),
                           n_valid.view(B, 1).to(torch.int32),
                           torch.zeros_like(seq_mask, dtype=torch.int32))
    torch.testing.assert_close(precomputed.mask_bias, expected)
    torch.testing.assert_close(precomputed.mask_bias_transposed, expected)


@pytest.mark.parametrize("seq_len", [16, 13])
def test_precompute_pair_masks_cutedsl_rejects_non_left_aligned(seq_len: int):
    """CuTeDSL precompute requires a left-aligned ``pair_mask``.

    The underlying left-mask kernel interprets ``actual_s_kv = sum(>0.5)``
    as the count of leading 1s per row (i.e. the row is ``1...1 0...0``).
    Feeding a non-left-aligned mask would silently produce wrong attention
    masking, so the precompute now rejects such inputs with an explicit
    AssertionError instead of swallowing the bug.
    """
    device = torch.device("cuda")
    B, I, J = 2, seq_len, seq_len

    # Random 0/1 mask: with high probability it is not non-increasing along
    # at least one axis, so the precompute must reject it.
    torch.manual_seed(0)
    pair_mask = torch.randint(0,
                              2, (B, I, J),
                              dtype=torch.float32,
                              device=device)
    # Force an interior zero so the input is guaranteed non-left-aligned
    # regardless of the random draw.
    pair_mask[0, 0, 0] = 1.0
    pair_mask[0, 0, 1] = 0.0
    pair_mask[0, 0, 2] = 1.0

    with pytest.raises(AssertionError, match="left-aligned"):
        precompute_pair_masks("CuTeDSL", pair_mask)


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="VANILLA",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="VANILLA",
             pairwise_attn_backend="CuTeDSL",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL",
             pairwise_attn_backend="CuTeDSL",
             torch_dtype="bfloat16"),
])
def test_pairformer_layer_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend, sc.pairwise_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    triangle_metadata_cls = get_attention_backend(
        sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata
    pairwise_metadata_cls = get_attention_backend(
        sc.pairwise_attn_backend, AttentionType.PAIRWISE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_layer = RefPairformerLayer.load_weights()
    ref_layer = ref_layer.to(device)
    weights_and_biases = create_pairformer_layer_weights(from_ref=ref_layer)

    layer = PairformerLayerV1(
        layer_idx=0,
        token_s=ref_layer.token_s,
        token_z=ref_layer.token_z,
        num_heads=ref_layer.num_heads,
        pairwise_head_width=ref_layer.pairwise_head_width,
        pairwise_num_heads=ref_layer.pairwise_num_heads,
        dtype=dtype,
        triangle_attn_backend=sc.triangle_attn_backend,
        pairwise_attn_backend=sc.pairwise_attn_backend,
        skip_create_weights=False,
        attention_initial_norm=True,
    )
    layer.to(device)
    load_pairformer_layer_weights_torch(layer, weights_and_biases, dtype)

    s = torch.randn(bs,
                    sc.seq_len,
                    ref_layer.token_s,
                    dtype=dtype,
                    device=device)
    z = torch.randn(bs,
                    sc.seq_len,
                    sc.seq_len,
                    ref_layer.token_z,
                    dtype=dtype,
                    device=device)
    mask = make_left_aligned_mask(bs, sc.seq_len, dtype=dtype, device=device)
    pair_mask = (mask[..., None] * mask[..., None, :]).to(dtype)

    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=Mapping()),
        "pairwise_attn": pairwise_metadata_cls(mapping=Mapping()),
    }

    precomputed = precompute_pair_masks(sc.triangle_attn_backend,
                                        pair_mask,
                                        inf=1e9,
                                        dtype=dtype)
    precomputed_single = precompute_single_masks(sc.pairwise_attn_backend,
                                                 mask,
                                                 inf=1e9)

    with torch.inference_mode():
        out_s, out_z = layer(s,
                             z,
                             mask,
                             pair_mask,
                             attn_metadatas=attn_metadatas)

        out_s_pre, out_z_pre = layer(
            s,
            z,
            mask,
            pair_mask,
            attn_metadatas=attn_metadatas,
            precomputed_masks=precomputed,
            precomputed_single_masks=precomputed_single)

    torch.testing.assert_close(out_s_pre, out_s, atol=0, rtol=0)
    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)
