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
    create_msa_layer_weights,
    create_msa_module_weights,
    load_msa_layer_weights_torch,
    load_msa_module_weights_torch,
)
from test_utils.boltz.ref_layers import RefMSALayer, RefMSAModule

from bionemo_ir._torch.attention_backend import AttentionMetadata, AttentionType, get_attention_backend
from bionemo_ir._torch.attention_backend.utils import precompute_pair_masks
from bionemo_ir._torch.graph_optimization.graph_policy import trunk_graph_config
from bionemo_ir._torch.layers.triangle_nodes import precompute_trimul_metadata
from bionemo_ir._torch.modules.boltz.trunk import MSALayer, MSAModule, Trunk
from bionemo_ir.models.boltz2.config import MSAModuleConfig, TrunkConfig
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        Scenario(torch_dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="CuTeDSL", torch_dtype="bfloat16"),
    ],
)
def test_msa_layer(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_mod = RefMSALayer.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_layer_weights(from_ref=ref_mod)

    msa_layer = MSALayer(
        msa_s=ref_mod.msa_s,
        token_z=ref_mod.token_z,
        pairwise_head_width=ref_mod.pairwise_head_width,
        pairwise_num_heads=ref_mod.pairwise_num_heads,
        triangle_attn_backend=sc.triangle_attn_backend,
        dtype=dtype,
    )
    load_msa_layer_weights_torch(msa_layer, weights_and_biases, dtype=dtype)
    msa_layer.to(device)

    z = torch.randn(bs, 64, 64, ref_mod.token_z, dtype=torch.float32).cuda()
    m = torch.randn(bs, 32, 64, ref_mod.msa_s, dtype=torch.float32).cuda()
    # Real left-aligned mask with n_valid in [N/2, N]: keeps the masking
    # path exercised while avoiding pathologically tight masks (n_valid = 1
    # would leave 63/64 rows fully masked, where the PyTorch reference
    # collapses to NaN via softmax-of-all-(-inf) and the CuTeDSL left-mask
    # kernel runs an ``actual_s_kv = 1`` no-op tile of garbage).
    seq_mask = make_left_aligned_mask(bs, 64, dtype=torch.float32, device=device, min_valid=32)
    token_mask = seq_mask[..., None] * seq_mask[..., None, :]
    msa_mask = torch.randint(0, 2, (bs, 32, 64), dtype=torch.float32).to(device)

    triangle_metadata_cls = get_attention_backend(sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata
    with torch.inference_mode():
        ref_z_float, ref_m_float = ref_mod(z, m, token_mask, msa_mask)
        z = z.to(dtype)
        m = m.to(dtype)
        token_mask = token_mask.to(dtype)
        msa_mask = msa_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z, ref_m = ref_mod(z, m, token_mask, msa_mask)
        z_before = z.clone()
        output_z, output_m = msa_layer(
            z,
            m,
            token_mask,
            msa_mask,
            precompute_trimul_metadata(z, None, None),
            attn_metadata=triangle_metadata_cls(),
        )

    assert ref_z.shape == output_z.shape
    assert ref_m.shape == output_m.shape
    # MSALayer accumulates z into the fresh OPM output instead of mutating its input z.
    torch.testing.assert_close(z, z_before, atol=0, rtol=0)

    # With a real left-aligned mask, padded query rows in
    # ``pair_weighted_averaging`` softmax over fully-masked keys -> NaN in
    # the PyTorch reference; the CuTeDSL left-mask kernel sees
    # ``actual_s_kv`` clamped to >= 1 and emits arithmetic garbage there.
    # Both implementations agree on the *valid* sub-block; only compare
    # there.
    z_keep = token_mask.float().unsqueeze(-1)  # [B, N, N, 1]
    m_keep = seq_mask.float()[:, None, :, None]  # [B, 1, N, 1]

    def _masked(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0) * keep

    if dtype == torch.float32:
        torch.testing.assert_close(_masked(ref_z, z_keep), _masked(output_z, z_keep), atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(_masked(ref_m, m_keep), _masked(output_m, m_keep), atol=1e-3, rtol=1e-4)
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
                f"{name} max: ours_diff={d_out_max.item()}, ref_diff={d_ref_max.item()}"
            )
            d_out_mean = torch.mean(torch.abs(d_out))
            d_ref_mean = torch.mean(torch.abs(d_ref))
            assert d_out_mean <= tol_mult * d_ref_mean + 1e-3, (
                f"{name} mean: ours_diff={d_out_mean.item()}, ref_diff={d_ref_mean.item()}"
            )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
    ],
)
def test_msa_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_mod = RefMSAModule.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_module_weights(from_ref=ref_mod)

    config = MSAModuleConfig(
        architecture="msa_module",
        msa_s=ref_mod.msa_s,
        token_z=ref_mod.token_z,
        token_s=ref_mod.token_s,
        msa_blocks=ref_mod.msa_blocks,
        num_tokens=ref_mod.num_tokens,
        pairwise_head_width=ref_mod.pairwise_head_width,
        pairwise_num_heads=ref_mod.pairwise_num_heads,
        version="v2",
        dtype=sc.torch_dtype,
    )
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
    has_deletion = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    deletion_value = torch.randn(B, N_msa, N, dtype=torch.float32).cuda()
    msa_paired = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    msa_mask = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    token_pad_mask = make_left_aligned_mask(B, N, dtype=torch.float32, device="cuda")
    pair_mask = token_pad_mask[:, :, None] * token_pad_mask[:, None, :]

    with torch.inference_mode():
        z = z.to(dtype)
        z_before = z.clone()
        emb = emb.to(dtype)
        has_deletion = has_deletion.to(dtype)
        deletion_value = deletion_value.to(dtype)
        msa_paired = msa_paired.to(dtype)
        msa_mask = msa_mask.to(dtype)
        token_pad_mask = token_pad_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z = ref_mod(z, emb, msa, has_deletion, deletion_value, msa_paired, msa_mask, token_pad_mask)
        output_z = msa_module(z, emb, msa, has_deletion, deletion_value, msa_paired, msa_mask, pair_mask)

    assert ref_z.shape == output_z.shape
    torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)
    # The trunk computes ``z = z + msa_module(z, ...)``: the module must not modify the z it is given.
    torch.testing.assert_close(z, z_before, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Tests for precomputed pair masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        Scenario(torch_dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="CUEQUIV", torch_dtype="bfloat16"),
        Scenario(triangle_attn_backend="CuTeDSL", torch_dtype="bfloat16"),
    ],
)
def test_msa_layer_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_mod = RefMSALayer.load_weights()
    ref_mod = ref_mod.to(device)
    weights_and_biases = create_msa_layer_weights(from_ref=ref_mod)

    msa_layer = MSALayer(
        msa_s=ref_mod.msa_s,
        token_z=ref_mod.token_z,
        pairwise_head_width=ref_mod.pairwise_head_width,
        pairwise_num_heads=ref_mod.pairwise_num_heads,
        triangle_attn_backend=sc.triangle_attn_backend,
        dtype=dtype,
    )
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
    msa_mask = torch.randint(0, 2, (bs, 32, 64), dtype=torch.float32, device=device).to(dtype)

    triangle_metadata_cls = get_attention_backend(sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata

    precomputed = precompute_pair_masks(sc.triangle_attn_backend, token_mask, inf=msa_layer.inf, dtype=dtype)
    trimul_metadata = precompute_trimul_metadata(
        z,
        precomputed.mask_bias if precomputed.mask_bias.dtype == torch.int32 else None,
        precomputed.mask_bias_transposed if precomputed.mask_bias_transposed.dtype == torch.int32 else None,
    )

    # MSALayer accumulates into m *in place* (memory opt), while preserving its input z.
    # Give each call its own copy of m so both mask paths run on identical inputs.
    z_before = z.clone()
    with torch.inference_mode():
        out_z, out_m = msa_layer(
            z,
            m.clone(),
            token_mask,
            msa_mask,
            trimul_metadata,
            attn_metadata=triangle_metadata_cls(),
        )
        out_z_pre, out_m_pre = msa_layer(
            z,
            m.clone(),
            token_mask,
            msa_mask,
            trimul_metadata,
            attn_metadata=triangle_metadata_cls(),
            precomputed_masks=precomputed,
        )

    torch.testing.assert_close(z, z_before, atol=0, rtol=0)
    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)
    torch.testing.assert_close(out_m_pre, out_m, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Regression tests for Trunk._recycle_step residual preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", ["float32", "bfloat16"])
def test_trunk_recycle_step_preserves_pre_msa_residual(dtype_str: str) -> None:
    """Trunk._recycle_step must feed pairformer ``z_pre_msa + msa_out`` without mutating ``z_init``."""
    torch.manual_seed(0)
    dtype = str_dtype_to_torch(dtype_str)
    device = torch.device("cuda")

    B, N, N_msa = 1, 16, 8
    config = TrunkConfig(dtype=dtype_str, graph_optimization_config=None)
    config.msa_module.dtype = dtype_str
    config.msa_module.msa_blocks = 2
    config.msa_module.triangle_attention_backend = "VANILLA"
    config.pairformer.dtype = dtype_str
    config.pairformer.s_path_dtype = dtype_str
    config.pairformer.num_blocks = 1
    config.pairformer.enable_token_pad = False
    config.pairformer.triangle_attention_backend = "VANILLA"
    config.pairformer.pairwise_attention_backend = "VANILLA"

    trunk = Trunk(config).to(device).eval()
    with torch.no_grad():
        for p in trunk.parameters():
            p.normal_(0.0, 0.02)

    s_init = torch.randn(B, N, config.pairformer.token_s, dtype=dtype, device=device)
    z_init = torch.randn(B, N, N, config.pairformer.token_z, dtype=dtype, device=device)
    s_inputs = torch.randn(B, N, config.pairformer.token_s, dtype=dtype, device=device)
    msa = torch.randint(0, config.msa_module.num_tokens, (B, N_msa, N), dtype=torch.int64, device=device)
    has_deletion = torch.randint(0, 2, (B, N_msa, N), dtype=dtype, device=device)
    deletion_value = torch.randn(B, N_msa, N, dtype=dtype, device=device)
    msa_paired = torch.randint(0, 2, (B, N_msa, N), dtype=dtype, device=device)
    msa_mask = torch.ones(B, N_msa, N, dtype=dtype, device=device)
    token_pad_mask = make_left_aligned_mask(B, N, dtype=dtype, device=device, min_valid=12)

    s_init_before = s_init.clone()
    z_init_before = z_init.clone()

    # Capture the exact z entering pairformer_module on each recycle and compare against an explicit
    # out-of-place computation of z_pre_msa + msa_module(z_pre_msa, ...).
    observed_pf_z: list[torch.Tensor] = []
    expected_pf_z: list[torch.Tensor] = []
    orig_msa_forward = trunk.msa_module.forward
    orig_pf_forward = trunk.pairformer_module.forward

    def spy_msa(
        z_in: torch.Tensor,
        emb: torch.Tensor,
        msa_in: torch.Tensor,
        has_deletion_in: torch.Tensor,
        deletion_value_in: torch.Tensor,
        msa_paired_in: torch.Tensor,
        msa_mask: torch.Tensor,
        token_pad_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        z_snapshot = z_in.clone()
        msa_out = orig_msa_forward(
            z_in,
            emb,
            msa_in,
            has_deletion_in,
            deletion_value_in,
            msa_paired_in,
            msa_mask=msa_mask,
            token_pad_mask=token_pad_mask,
            attn_metadata=attn_metadata,
        )
        torch.testing.assert_close(z_in, z_snapshot, atol=0, rtol=0)
        expected_pf_z.append(z_snapshot + msa_out)
        return msa_out

    def spy_pf(
        s_in: torch.Tensor,
        z_in: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_pf_z.append(z_in.clone())
        return orig_pf_forward(s_in, z_in, mask=mask, pair_mask=pair_mask, attn_metadata=attn_metadata)

    trunk.msa_module.forward = spy_msa
    trunk.pairformer_module.forward = spy_pf

    with torch.inference_mode():
        s_eager, z_eager = trunk(
            s_init=s_init,
            z_init=z_init,
            s_inputs=s_inputs,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_paired=msa_paired,
            msa_mask=msa_mask,
            token_pad_mask=token_pad_mask,
            recycling_steps=2,
        )

    assert len(observed_pf_z) == 3
    for step, (obs, exp) in enumerate(zip(observed_pf_z, expected_pf_z, strict=True)):
        torch.testing.assert_close(obs, exp, atol=0, rtol=0, msg=f"recycle {step} mismatch")
    torch.testing.assert_close(s_init, s_init_before, atol=0, rtol=0)
    torch.testing.assert_close(z_init, z_init_before, atol=0, rtol=0)

    # Verify CUDA-graph replay matches eager output identically.
    trunk.msa_module.forward = orig_msa_forward
    trunk.pairformer_module.forward = orig_pf_forward
    graph_config = TrunkConfig(
        dtype=dtype_str,
        graph_optimization_config=trunk_graph_config(),
    )
    graph_config.msa_module = config.msa_module.model_copy()
    graph_config.pairformer = config.pairformer.model_copy()
    trunk_graph = Trunk(graph_config).to(device).eval()
    trunk_graph.load_state_dict(trunk.state_dict())

    with torch.inference_mode():
        s_graph, z_graph = trunk_graph(
            s_init=s_init,
            z_init=z_init,
            s_inputs=s_inputs,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_paired=msa_paired,
            msa_mask=msa_mask,
            token_pad_mask=token_pad_mask,
            recycling_steps=2,
        )

    torch.testing.assert_close(s_graph, s_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(z_graph, z_eager, atol=1e-4, rtol=1e-4)
