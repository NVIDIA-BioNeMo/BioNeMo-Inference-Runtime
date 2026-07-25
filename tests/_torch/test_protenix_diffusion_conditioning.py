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
"""OSS equivalence tests for Protenix diffusion conditioning (AF3 Algorithm 21)."""
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.auto_chunk import (CHUNK_REGISTRY,
                                                DIFFUSION_PAIR_TRANSITION,
                                                ChunkPolicy)
from tensorrt_bionemo._torch.modules.protenix import \
    ProtenixDiffusionConditioning
from tensorrt_bionemo.models.protenix.config import (
    DiffusionConditioningConfig, RelativePositionEncodingConfig)
from tensorrt_bionemo.models.protenix.convert import \
    convert_diffusion_conditioning_torch
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixDiffusionConditioningFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 8
    n_sample: int = 2
    c_z: int = 64
    c_s: int = 32
    c_s_inputs: int = 16
    c_noise: int = 32
    batch_size: int = 1
    dtype: str = "float32"
    z_pair_dtype: str = "bfloat16"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32", z_pair_dtype="float32"),
    Scenario(dtype="float32", z_pair_dtype="bfloat16"),
    Scenario(dtype="bfloat16"),
],
                         ids=["fp32", "fp32_bf16_pair", "bf16"])
def test_protenix_diffusion_conditioning(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    z_pair_dtype = str_dtype_to_torch(sc.z_pair_dtype)
    B, N, S = sc.batch_size, sc.n_token, sc.n_sample

    # Randomize zero-initialized transition outputs.
    ref = RefProtenixDiffusionConditioningFromOSS.build(
        c_z=sc.c_z,
        c_s=sc.c_s,
        c_s_inputs=sc.c_s_inputs,
        c_noise_embedding=sc.c_noise).to(device=device,
                                         dtype=torch.float32).eval()
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)

    config = DiffusionConditioningConfig(
        c_s=sc.c_s,
        c_z=sc.c_z,
        c_s_inputs=sc.c_s_inputs,
        c_noise_embedding=sc.c_noise,
        relpe_config=RelativePositionEncodingConfig(c_z=sc.c_z),
        z_pair_dtype=sc.z_pair_dtype,
        dtype=sc.dtype)
    model = ProtenixDiffusionConditioning(config).to(device).eval()
    converted = convert_diffusion_conditioning_torch(config,
                                                     ref.state_dict(),
                                                     prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    # Build relp with the OSS buckets.
    tok = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    zero = torch.zeros(B, N, dtype=torch.long, device=device)
    feat = {
        "asym_id": zero,
        "residue_index": tok.long(),
        "entity_id": zero,
        "token_index": tok.long(),
        "sym_id": zero,
    }
    relp = ref.relpe.generate_relp(feat)["relp"].to(torch.float32)

    t = torch.rand(B, S, device=device) * 20.0 + 0.1
    s_inputs = torch.randn(B, N, sc.c_s_inputs, device=device)
    s_trunk = torch.randn(B, N, sc.c_s, device=device)
    z_trunk = torch.randn(B, N, N, sc.c_z, device=device)

    with torch.inference_mode():
        exp_s, exp_z = ref(t, relp, s_inputs, s_trunk, z_trunk, pair_z=None)
        act_s, act_z = model(t.to(torch_dtype), relp.to(torch_dtype),
                             s_inputs.to(torch_dtype), s_trunk.to(torch_dtype),
                             z_trunk.to(torch_dtype))

    assert act_s.dtype == torch_dtype
    assert act_z.dtype == z_pair_dtype
    # Pair transitions run in z_pair_dtype; single transitions stay module dtype.
    assert all(p.dtype == z_pair_dtype for layer in model.transition_z
               for p in layer.parameters())
    assert all(p.dtype == torch_dtype for layer in model.transition_s
               for p in layer.parameters())
    assert all(layer.auto_chunk_policy is CHUNK_REGISTRY.get(
        DIFFUSION_PAIR_TRANSITION) for layer in model.transition_z)
    assert all(layer.auto_chunk_policy is None for layer in model.transition_s)

    any_bf16 = torch.bfloat16 in (torch_dtype, z_pair_dtype)
    tol = 5e-2 if any_bf16 else 3e-3
    r_s = _rmse_ratio(act_s, exp_s)
    r_z = _rmse_ratio(act_z, exp_z)
    assert r_s < tol, f"single rmse_ratio={r_s:.3e} exceeds {tol:.0e} ({sc.dtype})"
    assert r_z < tol, f"pair rmse_ratio={r_z:.3e} exceeds {tol:.0e} ({sc.dtype})"


@pytest.mark.parametrize("z_pair_dtype", ["float32", "bfloat16"])
def test_protenix_diffusion_pair_projection_without_concat(z_pair_dtype: str):
    """Joint-statistics projection matches dense LayerNorm(cat)+Linear."""
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    c_z, n = 64, 64

    config = DiffusionConditioningConfig(
        c_z=c_z,
        c_s=32,
        c_s_inputs=16,
        c_noise_embedding=32,
        relpe_config=RelativePositionEncodingConfig(c_z=c_z),
        z_pair_dtype=z_pair_dtype,
        dtype="float32",
    )
    model = ProtenixDiffusionConditioning(config).to(device).eval()
    with torch.no_grad():
        model.relpe.linear.weight.normal_(mean=0.0, std=0.1)
        model.layernorm_z.weight.uniform_(0.5, 1.5)
        model.linear_no_bias_z.weight.normal_(mean=0.0, std=0.1)

    relp = torch.randn(1,
                       n,
                       n,
                       model.relpe.linear.in_features,
                       dtype=torch.float32,
                       device=device)
    z_trunk = torch.randn(1, n, n, c_z, dtype=torch.float32, device=device)

    with torch.inference_mode():
        relpe_z = model.relpe(relp=relp)
        expected = model.linear_no_bias_z(
            model.layernorm_z(torch.cat([z_trunk, relpe_z], dim=-1))).to(
                str_dtype_to_torch(z_pair_dtype))
        actual = model._joint_layernorm_linear_z(z_trunk, relpe_z).to(
            str_dtype_to_torch(z_pair_dtype))

    tol = (dict(atol=5e-5, rtol=5e-5)
           if z_pair_dtype == "float32" else dict(atol=2e-2, rtol=2e-2))
    torch.testing.assert_close(actual, expected, **tol)


@pytest.mark.parametrize("torch_dtype", ["float32", "bfloat16"])
def test_protenix_diffusion_pair_transition_auto_chunk(torch_dtype: str):
    """Match row-chunked ``transition_z`` to the dense path."""
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(torch_dtype)
    c_z, n = 128, 64
    policy = ChunkPolicy(chunk_size=32, min_size=1, dim=1, min_rank=4)

    config = DiffusionConditioningConfig(
        c_z=c_z,
        c_s=32,
        c_s_inputs=16,
        c_noise=32,
        relpe_config=RelativePositionEncodingConfig(c_z=c_z),
        z_pair_dtype=torch_dtype,
    )
    config.set_dtype(torch_dtype)
    model = ProtenixDiffusionConditioning(config).to(device).eval()
    for layer in model.transition_z:
        layer.auto_chunk_policy = policy
        with torch.no_grad():
            for p in layer.parameters():
                p.normal_(mean=0.0, std=0.1)

    tol = (dict(atol=1e-4, rtol=1e-4)
           if dtype == torch.float32 else dict(atol=3e-3, rtol=3e-3))
    z = torch.randn(1, n, n, c_z, dtype=dtype, device=device)
    with torch.inference_mode():
        for layer in model.transition_z:
            dense = layer._forward_impl(z)
            chunked = layer(z)
            torch.testing.assert_close(chunked, dense, **tol)
            z = z + chunked
