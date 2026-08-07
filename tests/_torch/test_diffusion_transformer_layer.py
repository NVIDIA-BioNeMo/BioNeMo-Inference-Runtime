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
    create_diffusion_transformer_layer_weights,
    load_diffusion_transformer_layer_weights_torch,
)
from test_utils.boltz.ref_layers import RefDiffusionTransformerLayer as BoltzRefDiffusionTransformerLayer
from test_utils.openfold3.ref_layers import Openfold3RefDiffusionTransformerLayer

from tensorrt_bionemo._torch.attention_backend import AttentionType, get_attention_backend
from tensorrt_bionemo._torch.attention_backend.utils import precompute_single_masks
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import DiffusionTransformerLayer
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    torch_dtype: str = "float32"
    seq_len: int = 128
    num_heads: int = 16
    dim_pairwise: int = 128
    num_samples: int = 1
    test_with_openfold3: bool = False
    conditioned_transition_using_silu: bool = False
    backend: str = "VANILLA"


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim=768, dim_single_cond=768),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5),
        Scenario(dim=768, dim_single_cond=768, num_samples=10, torch_dtype="bfloat16"),
        Scenario(
            dim=768,
            dim_single_cond=384,
            num_samples=10,
            test_with_openfold3=True,
            conditioned_transition_using_silu=True,
        ),
        Scenario(dim=768, dim_single_cond=768, backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5, backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, num_samples=10, torch_dtype="bfloat16", backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", backend="CuTeDSL"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5, torch_dtype="bfloat16", backend="CuTeDSL"),
        Scenario(
            dim=768,
            dim_single_cond=384,
            num_samples=10,
            torch_dtype="bfloat16",
            test_with_openfold3=True,
            conditioned_transition_using_silu=True,
            backend="CuTeDSL",
        ),
    ],
    ids=[
        "boltz-single-float32",
        "boltz-single-bfloat16",
        "boltz-samples5-float32",
        "boltz-samples10-bfloat16",
        "openfold3-samples10-silu-float32",
        "boltz-single-float32-sdpa",
        "boltz-single-bfloat16-sdpa",
        "boltz-samples5-float32-sdpa",
        "boltz-samples10-bfloat16-sdpa",
        "boltz-single-bfloat16-cutedsl",
        "boltz-samples5-bfloat16-cutedsl",
        "openfold3-samples10-bfloat16-cutedsl",
    ],
)
def test_diffusion_transformer_layer(sc: Scenario, monkeypatch: pytest.MonkeyPatch):
    _skip_if_cutedsl(sc.backend)
    torch.manual_seed(42)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    # Load reference module based on whether it's OpenFold3 or Boltz
    if sc.test_with_openfold3:
        if os.environ.get("OPENFOLD3_CKPT") is None:
            pytest.skip("OPENFOLD3_CKPT environment variable is not set")
        ref_module = Openfold3RefDiffusionTransformerLayer.load_weights()
    else:
        ref_module = BoltzRefDiffusionTransformerLayer.load_weights()

    ref_module = ref_module.to(device)

    weights_and_biases = create_diffusion_transformer_layer_weights(from_ref=ref_module)

    attn_pairwise_metadata_cls = get_attention_backend(sc.backend, AttentionType.PAIRWISE).Metadata

    module = DiffusionTransformerLayer(
        layer_idx=0,
        num_heads=ref_module.pair_bias_attn.num_heads,
        dim=sc.dim,
        dim_single_cond=sc.dim_single_cond,
        dim_pairwise=sc.dim_pairwise,
        bias_proj=True,
        dtype=dtype,
        conditioned_transition_using_silu=sc.conditioned_transition_using_silu,
        attn_backend=sc.backend,
    )

    load_diffusion_transformer_layer_weights_torch(module, weights_and_biases, dtype=dtype)

    module.to(device)

    # Handle both single and multi-sample cases. ``mask`` is a real
    # left-aligned 0/1 mask so the additive-bias path (VANILLA/SDPA) and the
    # ``actual_s_kv`` path (CuTeDSL left-mask kernel) produce the same
    # masked attention.  Use ``min_valid=seq_len`` (no fully-padded rows) to
    # keep tensor-wise comparison against the PyTorch reference clean.
    if sc.num_samples == 1:
        a = torch.randn(bs, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
        s = torch.randn(bs, sc.seq_len, sc.dim_single_cond, dtype=torch.float32).cuda()
    else:
        a = torch.randn(bs, sc.num_samples, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
        s = torch.randn(bs, 1, sc.seq_len, sc.dim_single_cond, dtype=torch.float32).cuda()
    mask = make_left_aligned_mask(bs, sc.seq_len, dtype=torch.float32, device="cuda", min_valid=sc.seq_len)

    z = torch.randn(bs, sc.seq_len, sc.seq_len, sc.dim_pairwise, dtype=torch.float32).cuda()
    with torch.inference_mode():
        ref_output_float = ref_module(a, s, z, mask)

        a = a.to(dtype)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)

        ref_module = ref_module.to(dtype)
        ref_output = ref_module(a, s, z, mask)
        output = module.forward(a, s, z, mask, attn_metadata=attn_pairwise_metadata_cls(bias_cache={}))

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        # OF3 uses dim_single_cond=384 (vs 768 for boltz) and 10 samples;
        # the fused LN+proj + FusedSwiGLU cascade accumulates ~1e-2 fp32
        # roundoff vs the unfused PyTorch reference (relative error stays
        # ~3e-6 against output magnitude ~4e3).  Use a slightly looser
        # absolute tolerance for the OF3 path; boltz keeps the tight one.
        atol = 2e-2 if sc.test_with_openfold3 else 1e-3
        torch.testing.assert_close(ref_output, output, atol=atol, rtol=1e-4)
    else:
        # Asymmetric tolerance: ``ours`` must be no more than ``tol_mult``×
        # worse than the bf16 reference's distance to the fp32 ground
        # truth.  Symmetric ratio checks (``|d0-d1|/min<=0.5``) fail when
        # ``ours`` is significantly *more* accurate than the bf16 reference,
        # which happens for the CuTeDSL left-mask path with binary
        # left-aligned masks (kernel keeps internal fp32 accumulation while
        # the bf16 ref module's chain of bf16 ops loses precision).
        tol_mult = 2.0
        diff_ours_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff_ref_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        assert diff_ours_max <= tol_mult * diff_ref_max + 1e-3, (
            f"max: ours_diff={diff_ours_max.item()}, ref_diff={diff_ref_max.item()}"
        )
        diff_ours_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff_ref_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))
        assert diff_ours_mean <= tol_mult * diff_ref_mean + 1e-3, (
            f"mean: ours_diff={diff_ours_mean.item()}, ref_diff={diff_ref_mean.item()}"
        )


# ---------------------------------------------------------------------------
# Tests for precomputed single masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim=768, dim_single_cond=768),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5),
        Scenario(dim=768, dim_single_cond=768, backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5, backend="SDPA"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", backend="CuTeDSL"),
        Scenario(dim=768, dim_single_cond=768, num_samples=5, torch_dtype="bfloat16", backend="CuTeDSL"),
    ],
    ids=[
        "vanilla-float32",
        "vanilla-bfloat16",
        "vanilla-samples5-float32",
        "sdpa-float32",
        "sdpa-bfloat16",
        "sdpa-samples5-float32",
        "cutedsl-bfloat16",
        "cutedsl-samples5-bfloat16",
    ],
)
def test_diffusion_transformer_layer_precomputed_masks(sc: Scenario, monkeypatch: pytest.MonkeyPatch):
    """Outputs with precomputed single masks must exactly match the original path."""
    _skip_if_cutedsl(sc.backend)
    torch.manual_seed(42)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_module = BoltzRefDiffusionTransformerLayer.load_weights()
    ref_module = ref_module.to(device)
    weights_and_biases = create_diffusion_transformer_layer_weights(from_ref=ref_module)

    attn_pairwise_metadata_cls = get_attention_backend(sc.backend, AttentionType.PAIRWISE).Metadata

    module = DiffusionTransformerLayer(
        layer_idx=0,
        num_heads=ref_module.pair_bias_attn.num_heads,
        dim=sc.dim,
        dim_single_cond=sc.dim_single_cond,
        dim_pairwise=sc.dim_pairwise,
        bias_proj=True,
        dtype=dtype,
        attn_backend=sc.backend,
    )
    load_diffusion_transformer_layer_weights_torch(module, weights_and_biases, dtype=dtype)
    module.to(device)

    if sc.num_samples == 1:
        a = torch.randn(bs, sc.seq_len, sc.dim, dtype=dtype, device=device)
        s = torch.randn(bs, sc.seq_len, sc.dim_single_cond, dtype=dtype, device=device)
        mask = torch.randint(0, 2, (bs, sc.seq_len), dtype=dtype, device=device)
    else:
        a = torch.randn(bs, sc.num_samples, sc.seq_len, sc.dim, dtype=dtype, device=device)
        s = torch.randn(bs, 1, sc.seq_len, sc.dim_single_cond, dtype=dtype, device=device)
        mask = torch.randint(0, 2, (bs, sc.seq_len), dtype=dtype, device=device)

    z = torch.randn(bs, sc.seq_len, sc.seq_len, sc.dim_pairwise, dtype=dtype, device=device)

    precomputed = precompute_single_masks(sc.backend, mask, inf=1e9)
    attn_metadata = attn_pairwise_metadata_cls(bias_cache={})

    with torch.inference_mode():
        out = module.forward(a, s, z, mask, attn_metadata=attn_metadata)
        out_pre = module.forward(a, s, z, mask, attn_metadata=attn_metadata, mask_bias=precomputed.mask_bias)

    torch.testing.assert_close(out_pre, out, atol=0, rtol=0)
