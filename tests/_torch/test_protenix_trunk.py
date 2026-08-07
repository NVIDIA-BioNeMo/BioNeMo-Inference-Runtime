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
"""Smoke and precision tests for the Protenix recycling trunk."""

import os

import torch

from tensorrt_bionemo._torch.modules.protenix import ProtenixTrunk
from tensorrt_bionemo.configs import PairformerConfig
from tensorrt_bionemo.models.protenix.config import (
    ProtenixMSAModuleConfig,
    TemplateEmbedderConfig,
    TrunkConfig,
    _pairformer_config,
)


def _small_trunk_config(c_s: int, c_z: int, c_s_inputs: int, pair_state_dtype: str = "float32") -> TrunkConfig:
    # Keep random-weight assembly stable in fp32.
    template = TemplateEmbedderConfig(
        c=32, c_z=c_z, n_blocks=1, pairwise_head_width=16, pairwise_num_heads=2, pairformer_dtype="float32"
    )
    msa = ProtenixMSAModuleConfig(
        c_m=64,
        c_z=c_z,
        c_hidden_mul=c_z,
        c_hidden_pair_att=32,
        no_heads_pair=c_z // 32,
        no_blocks=2,
        c_s_inputs=c_s_inputs,
        dtype="float32",
    )
    pairformer = PairformerConfig(
        token_s=c_s,
        token_z=c_z,
        num_blocks=2,
        num_heads=4,
        pairwise_head_width=32,
        pairwise_num_heads=c_z // 32,
        no_update_s=False,
        attention_initial_norm=True,
        version="v1",
        dtype="float32",
    )
    return TrunkConfig(
        c_s=c_s,
        c_z=c_z,
        n_cycle=1,
        pair_state_dtype=pair_state_dtype,
        use_template=True,  # exercise the template path in-trunk
        template_embedder_config=template,
        msa_module_config=msa,
        pairformer_config=pairformer,
    )


def test_protenix_trunk_smoke():
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    B, N, S = 1, 16, 6
    c_s, c_z, c_s_inputs = 64, 64, 449

    trunk = ProtenixTrunk(_small_trunk_config(c_s, c_z, c_s_inputs)).to(device).eval()

    feat = {
        "asym_id": (torch.arange(N) // 8).unsqueeze(0).to(device),
        "msa": torch.randint(0, 32, (B, S, N), device=device),
        "has_deletion": (torch.randn(B, S, N, device=device) > 0).float(),
        "deletion_value": torch.rand(B, S, N, device=device),
        "template_distogram": torch.randn(B, 1, N, N, 39, device=device),
        "template_pseudo_beta_mask": (torch.randn(B, 1, N, N, device=device) > 0).float(),
        "template_aatype": torch.randint(0, 32, (B, 1, N), device=device),
        "template_unit_vector": torch.randn(B, 1, N, N, 3, device=device),
        "template_backbone_frame_mask": (torch.randn(B, 1, N, N, device=device) > 0).float(),
    }
    # Small random inputs keep uninitialized recycling paths stable.
    s_inputs = 0.1 * torch.randn(B, N, c_s_inputs, device=device)
    s_init = 0.1 * torch.randn(B, N, c_s, device=device)
    z_init = 0.1 * torch.randn(B, N, N, c_z, device=device)

    with torch.inference_mode():
        s, z = trunk(feat, s_inputs, s_init, z_init)

    # Random untrained recycles may be non-finite; only validate assembly shape.
    assert s.shape == (B, N, c_s)
    assert z.shape == (B, N, N, c_z)


def test_protenix_default_pairformer_config():
    """Check production pairformer defaults."""
    cfg = _pairformer_config()
    assert cfg.num_blocks == 48
    assert cfg.token_s == 384 and cfg.token_z == 256
    assert cfg.num_heads == 16
    assert cfg.pairwise_num_heads == 256 // 32
    assert cfg.no_update_s is False
    assert cfg.torch_dtype == torch.bfloat16
    assert TrunkConfig().pair_state_dtype == "float32"


def test_protenix_bf16_pair_state_parity_and_input_safety():
    """Persistent bf16 z stays close to fp32 state and never aliases z_init."""
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    B, N = 1, 8
    c_s, c_z, c_s_inputs = 64, 64, 449

    trunk_fp32 = ProtenixTrunk(_small_trunk_config(c_s, c_z, c_s_inputs, pair_state_dtype="float32")).to(device).eval()
    # Initialize matrix weights while preserving LayerNorm defaults.
    with torch.no_grad():
        for name, p in trunk_fp32.named_parameters():
            if p.ndim >= 2:
                p.normal_(mean=0.0, std=0.01)
            elif name.endswith("weight"):
                p.fill_(1)
            else:
                p.zero_()

    trunk_bf16 = ProtenixTrunk(_small_trunk_config(c_s, c_z, c_s_inputs, pair_state_dtype="bfloat16")).to(device).eval()
    trunk_bf16.load_state_dict(trunk_fp32.state_dict())

    s_inputs = 0.01 * torch.randn(B, N, c_s_inputs, device=device)
    s_init = 0.01 * torch.randn(B, N, c_s, device=device)
    z_init = 0.01 * torch.randn(B, N, N, c_z, device=device)
    s_init_before = s_init.clone()
    z_init_before = z_init.clone()

    # Empty features isolate recycled-state storage.
    with torch.inference_mode():
        s_fp32, z_fp32 = trunk_fp32({}, s_inputs, s_init, z_init, num_cycles=10)
        s_bf16, z_bf16 = trunk_bf16({}, s_inputs, s_init, z_init, num_cycles=10)

    assert z_fp32.dtype == torch.float32
    assert z_bf16.dtype == torch.bfloat16
    assert s_fp32.dtype == s_bf16.dtype == torch.float32
    torch.testing.assert_close(s_init, s_init_before)
    torch.testing.assert_close(z_init, z_init_before)
    assert torch.isfinite(s_bf16).all() and torch.isfinite(z_bf16).all()

    def rmse_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
        actual, expected = actual.float(), expected.float()
        return (torch.mean((actual - expected) ** 2).sqrt() / (torch.mean(expected**2).sqrt() + 1e-8)).item()

    assert rmse_ratio(s_bf16, s_fp32) < 5e-2
    assert rmse_ratio(z_bf16, z_fp32) < 5e-2
