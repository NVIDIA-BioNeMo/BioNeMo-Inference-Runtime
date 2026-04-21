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
"""Tests for the mega-GEMM precomputed bias path in OpenFold3DiffusionTransformer.

The mega-GEMM fuses per-layer LN γ into projection weights (W_mega), then
computes ``W_mega @ layer_norm(z).T`` in one GEMM producing [NH, BIJ] with
J contiguous.  For B=1, slicing yields zero-copy [1, H, I, J_pad] views.

These tests verify that ``precompute_bias=True`` (mega-GEMM) produces the
same output as ``precompute_bias=False`` (per-layer LN+Proj inside each
layer's AttentionPairBias).
"""
from dataclasses import dataclass

import pytest
import torch

from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    OpenFold3DiffusionTransformer
from tensorrt_bionemo.configs.modules import DiffusionTransformerConfig
from tests._torch import skip_if_cutedsl

SEED = 42


def _init_small(module: torch.nn.Module, scale: float = 0.02):
    """Small uniform init for all parameters to avoid NaN / overflow."""
    for p in module.parameters():
        p.data.uniform_(-scale, scale)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    num_layers: int = 4
    dim: int = 256
    dim_single_cond: int = 256
    dim_pairwise: int = 64
    num_heads: int = 8
    seq_len: int = 32
    dtype: str = "bfloat16"
    backend: str = "CuTeDSL"


def _build_models(sc: Scenario, device: torch.device):
    """Build two identical OF3 DiT models: one with mega-GEMM, one without."""
    dtype_str = sc.dtype
    cfg_kwargs = dict(
        num_blocks=sc.num_layers,
        num_heads=sc.num_heads,
        dim=sc.dim,
        dim_single_cond=sc.dim_single_cond,
        dim_pairwise=sc.dim_pairwise,
        bias_proj=True,
        attention_initial_norm=True,
        post_layer_norm=False,
        version="v1",
        dtype=dtype_str,
        conditioned_transition_using_silu=True,
        pairwise_attention_backend=sc.backend,
        initial_norm=True,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        attn_output_gate=True,
        transition_expansion_factor=2,
    )

    cfg_mega = DiffusionTransformerConfig(**cfg_kwargs, precompute_bias=True)
    cfg_base = DiffusionTransformerConfig(**cfg_kwargs, precompute_bias=False)

    torch.manual_seed(SEED)
    model_mega = OpenFold3DiffusionTransformer(cfg_mega)
    _init_small(model_mega)

    torch.manual_seed(SEED)
    model_base = OpenFold3DiffusionTransformer(cfg_base)
    _init_small(model_base)

    # Copy weights from mega to base to ensure identical weights
    model_base.load_state_dict(model_mega.state_dict())

    model_mega = model_mega.to(device).eval()
    model_base = model_base.to(device).eval()

    return model_mega, model_base


@pytest.mark.parametrize("sc", [
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=32),
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=33),
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=64),
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=127),
    Scenario(backend="VANILLA", dtype="bfloat16", seq_len=32),
    Scenario(backend="VANILLA", dtype="float32", seq_len=32),
    Scenario(backend="SDPA", dtype="bfloat16", seq_len=32),
    Scenario(backend="SDPA", dtype="bfloat16", seq_len=33),
],
                         ids=[
                             "cutedsl-bf16-aligned",
                             "cutedsl-bf16-odd",
                             "cutedsl-bf16-64",
                             "cutedsl-bf16-127",
                             "vanilla-bf16",
                             "vanilla-fp32",
                             "sdpa-bf16-aligned",
                             "sdpa-bf16-odd",
                         ])
def test_mega_gemm_vs_per_layer(sc: Scenario):
    """Mega-GEMM precomputed bias must match per-layer bias projection."""
    skip_if_cutedsl(sc.backend)
    device = torch.device("cuda")
    dtype = torch.float32 if sc.dtype == "float32" else torch.bfloat16

    model_mega, model_base = _build_models(sc, device)

    torch.manual_seed(SEED + 1)
    B = 1
    a = torch.randn(B, sc.seq_len, sc.dim, device=device, dtype=dtype)
    s = torch.randn(B,
                    sc.seq_len,
                    sc.dim_single_cond,
                    device=device,
                    dtype=dtype)
    z = torch.randn(B,
                    sc.seq_len,
                    sc.seq_len,
                    sc.dim_pairwise,
                    device=device,
                    dtype=dtype)
    mask = torch.ones(B, sc.seq_len, device=device, dtype=dtype)

    with torch.inference_mode():
        out_mega = model_mega(a=a.clone(), s=s, z=z, mask=mask)
        out_base = model_base(a=a.clone(), s=s, z=z, mask=mask)

    assert out_mega.shape == out_base.shape, (
        f"Shape mismatch: {out_mega.shape} vs {out_base.shape}")

    if dtype == torch.float32:
        torch.testing.assert_close(out_mega, out_base, atol=1e-4, rtol=1e-4)
    else:
        torch.testing.assert_close(out_mega, out_base, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("sc", [
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=32),
    Scenario(backend="CuTeDSL", dtype="bfloat16", seq_len=33),
],
                         ids=[
                             "cutedsl-aligned",
                             "cutedsl-odd",
                         ])
def test_mega_gemm_bias_correctness(sc: Scenario):
    """Verify the fused W_mega produces the same per-layer biases as the
    original LN + Linear projection."""
    skip_if_cutedsl(sc.backend)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(SEED)
    cfg = DiffusionTransformerConfig(
        num_blocks=sc.num_layers,
        num_heads=sc.num_heads,
        dim=sc.dim,
        dim_single_cond=sc.dim_single_cond,
        dim_pairwise=sc.dim_pairwise,
        bias_proj=True,
        attention_initial_norm=True,
        post_layer_norm=False,
        version="v1",
        dtype="bfloat16",
        conditioned_transition_using_silu=True,
        pairwise_attention_backend=sc.backend,
        initial_norm=True,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        attn_output_gate=True,
        transition_expansion_factor=2,
        precompute_bias=True,
    )
    model = OpenFold3DiffusionTransformer(cfg)
    _init_small(model)
    model = model.to(device).eval()

    B = 1
    torch.manual_seed(SEED + 1)
    z = torch.randn(B,
                    sc.seq_len,
                    sc.seq_len,
                    sc.dim_pairwise,
                    device=device,
                    dtype=dtype)

    with torch.inference_mode():
        # Mega-GEMM path
        if hasattr(model, 'layer_norm_z'):
            z_normed = model.layer_norm_z(z)
        else:
            z_normed = z
        mega_biases = model._precompute_all_biases(z_normed)

        # Per-layer reference path
        import torch.nn.functional as F
        D = sc.dim_pairwise
        sc.num_heads
        pad_align = 8 if sc.backend == "CuTeDSL" else -1
        J = sc.seq_len
        J_pad = ((J + pad_align - 1) //
                 pad_align) * pad_align if pad_align > 0 else J

        ref_biases = []
        for layer in model.layers:
            proj_z = layer.pair_bias_attn.proj_z
            ln = proj_z[0] if len(proj_z) > 1 else None
            proj = proj_z[-1]

            if ln is not None:
                z_ln = F.layer_norm(z_normed, [D], ln.weight, ln.bias, ln.eps)
            else:
                z_ln = z_normed
            bias = F.linear(z_ln, proj.weight)  # [B, I, J, H]
            bias = bias.permute(0, 3, 1, 2)  # [B, H, I, J]
            if J_pad != J:
                bias = F.pad(bias, (0, J_pad - J))
            ref_biases.append(bias)

    for i, (mega_b, ref_b) in enumerate(zip(mega_biases, ref_biases)):
        assert mega_b.shape == ref_b.shape, (
            f"Layer {i}: shape {mega_b.shape} vs {ref_b.shape}")
        torch.testing.assert_close(mega_b,
                                   ref_b,
                                   atol=1e-2,
                                   rtol=1e-2,
                                   msg=f"Layer {i} bias mismatch")


def test_mega_gemm_weight_invalidation():
    """W_mega must be rebuilt when underlying weights change."""
    device = torch.device("cuda")

    torch.manual_seed(SEED)
    cfg = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=4,
        dim=64,
        dim_single_cond=64,
        dim_pairwise=32,
        bias_proj=True,
        attention_initial_norm=True,
        post_layer_norm=False,
        version="v1",
        dtype="bfloat16",
        conditioned_transition_using_silu=True,
        pairwise_attention_backend="VANILLA",
        initial_norm=True,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        attn_output_gate=True,
        transition_expansion_factor=2,
        precompute_bias=True,
    )
    model = OpenFold3DiffusionTransformer(cfg)
    _init_small(model)
    model = model.to(device).eval()

    z = torch.randn(1, 8, 8, 32, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 8, device=device, dtype=torch.bfloat16)
    a = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)
    s = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        out1 = model(a=a.clone(), s=s, z=z, mask=mask)

    assert model._W_mega is not None, "W_mega should be built after first forward"
    W_mega_before = model._W_mega.clone()

    # Reload via state_dict (the standard PyTorch path)
    state = model.state_dict()
    model.load_state_dict(state)
    # Manually invalidate (load_weights does this; load_state_dict does not)
    model._W_mega = None

    assert model._W_mega is None, "W_mega should be invalidated"

    with torch.inference_mode():
        out2 = model(a=a.clone(), s=s, z=z, mask=mask)

    assert model._W_mega is not None, "W_mega should be rebuilt after forward"
    torch.testing.assert_close(model._W_mega, W_mega_before)
    torch.testing.assert_close(out2, out1, atol=0, rtol=0)

    # Verify mutating a proj weight and rebuilding gives a DIFFERENT W_mega
    with torch.no_grad():
        model.layers[0].pair_bias_attn.proj_z[-1].weight.fill_(0.0)
    model._W_mega = None
    model._build_mega_weight()
    assert not torch.equal(model._W_mega, W_mega_before), \
        "W_mega should differ after weight mutation"


def test_precompute_bias_disabled():
    """With precompute_bias=False the model must still work (per-layer path)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(SEED)
    cfg = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=4,
        dim=64,
        dim_single_cond=64,
        dim_pairwise=32,
        bias_proj=True,
        attention_initial_norm=True,
        post_layer_norm=False,
        version="v1",
        dtype="bfloat16",
        conditioned_transition_using_silu=True,
        pairwise_attention_backend="VANILLA",
        initial_norm=True,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        attn_output_gate=True,
        transition_expansion_factor=2,
        precompute_bias=False,
    )
    model = OpenFold3DiffusionTransformer(cfg)
    _init_small(model)
    model = model.to(device).eval()

    assert model._precompute_bias is False

    z = torch.randn(1, 8, 8, 32, device=device, dtype=dtype)
    mask = torch.ones(1, 8, device=device, dtype=dtype)
    a = torch.randn(1, 8, 64, device=device, dtype=dtype)
    s = torch.randn(1, 8, 64, device=device, dtype=dtype)

    with torch.inference_mode():
        out = model(a=a, s=s, z=z, mask=mask)

    assert out.shape == a.shape
    assert torch.isfinite(out).all(), "Output should be finite"
