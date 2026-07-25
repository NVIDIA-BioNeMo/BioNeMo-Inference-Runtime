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
"""OSS equivalence tests for the Protenix template embedder."""
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from tensorrt_bionemo._torch.modules.protenix import ProtenixTemplateEmbedder
from tensorrt_bionemo.models.protenix.config import TemplateEmbedderConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_template_embedder_torch
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import skip_if_cutedsl
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixTemplateEmbedderFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 24
    n_templ: int = 2
    n_chains: int = 2
    # Four triangle heads satisfy CuTeDSL alignment.
    c: int = 128
    c_z: int = 256
    n_blocks: int = 2
    dtype: str = "float32"
    pairformer_dtype: str = "float32"
    tri_backend: str = "VANILLA"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _make_features(sc: Scenario, device: torch.device) -> dict:
    """Unbatched OSS-format template features (leading template axis)."""
    N, T = sc.n_token, sc.n_templ
    g = torch.Generator(device="cpu").manual_seed(7)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g).to(device)

    asym = (torch.arange(N) //
            max(1, N // sc.n_chains)).clamp(max=sc.n_chains - 1).to(device)
    return {
        "asym_id": asym,  # [N]
        "template_distogram": rnd(T, N, N, 39),
        "template_pseudo_beta_mask": (rnd(T, N, N) > 0).float(),
        "template_aatype": torch.randint(0, 32, (T, N),
                                         generator=g).to(device),
        "template_unit_vector": rnd(T, N, N, 3),
        "template_backbone_frame_mask": (rnd(T, N, N) > 0).float(),
    }


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32", pairformer_dtype="float32"),
    Scenario(dtype="bfloat16", pairformer_dtype="bfloat16"),
    Scenario(dtype="float32", pairformer_dtype="bfloat16"),
    Scenario(
        dtype="float32", pairformer_dtype="bfloat16", tri_backend="CuTeDSL"),
],
                         ids=[
                             "fp32", "bf16", "fp32-pairformer-bf16", "cutedsl"
                         ])
def test_protenix_template_embedder(sc: Scenario):
    skip_if_cutedsl(sc.tri_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    ref = RefProtenixTemplateEmbedderFromOSS.build(
        n_blocks=sc.n_blocks, c=sc.c,
        c_z=sc.c_z).to(device=device, dtype=torch.float32).eval()

    config = TemplateEmbedderConfig(c=sc.c,
                                    c_z=sc.c_z,
                                    n_blocks=sc.n_blocks,
                                    pairwise_head_width=32,
                                    pairwise_num_heads=sc.c // 32,
                                    dtype=sc.dtype,
                                    pairformer_dtype=sc.pairformer_dtype,
                                    triangle_attention_backend=sc.tri_backend)
    model = ProtenixTemplateEmbedder(config).to(device).eval()
    converted = convert_template_embedder_torch(config,
                                                ref.state_dict(),
                                                prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    feat = _make_features(sc, device)
    z = torch.randn(sc.n_token, sc.n_token, sc.c_z, device=device)
    feat_batched = {k: v.unsqueeze(0) for k, v in feat.items()}

    with torch.inference_mode():
        exp = ref(feat, z)  # unbatched [N, N, c_z]
        act = model(feat_batched, z.unsqueeze(0).to(torch_dtype)).squeeze(0)

    r = _rmse_ratio(act, exp)
    # Pair-stack precision sets the tolerance.
    all_fp32 = sc.dtype == "float32" and sc.pairformer_dtype == "float32"
    tol = 5e-3 if all_fp32 else 8e-2
    assert r < tol, (f"rmse_ratio={r:.3e} exceeds {tol:.0e} "
                     f"(dtype={sc.dtype}, pairformer={sc.pairformer_dtype})")


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_protenix_template_split_projection_matches_dense(dtype_name: str):
    """Weight-sliced projection matches the original 108-channel concat."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(dtype_name)
    sc = Scenario(n_token=12,
                  n_templ=1,
                  c=64,
                  c_z=64,
                  n_blocks=0,
                  dtype=dtype_name,
                  pairformer_dtype=dtype_name)
    config = TemplateEmbedderConfig(c=sc.c,
                                    c_z=sc.c_z,
                                    n_blocks=0,
                                    pairwise_head_width=32,
                                    pairwise_num_heads=sc.c // 32,
                                    dtype=dtype_name,
                                    pairformer_dtype=dtype_name)
    model = ProtenixTemplateEmbedder(config).to(device).eval()
    with torch.no_grad():
        model.linear_no_bias_a.weight.normal_(mean=0.0, std=0.02)

    feat = _make_features(sc, device)
    batched = {name: value.unsqueeze(0) for name, value in feat.items()}
    pair_mask = (torch.rand(1, sc.n_token, sc.n_token, device=device)
                 > 0.2).to(dtype)
    multichain_mask = (
        batched["asym_id"][..., :,
                           None] == batched["asym_id"][..., None, :]).to(dtype)
    masked_by = pair_mask * multichain_mask

    with torch.inference_mode():
        dgram = batched["template_distogram"][:, 0].to(
            dtype) * masked_by.unsqueeze(-1)
        pseudo_beta = (batched["template_pseudo_beta_mask"][:, 0].to(dtype) *
                       masked_by).unsqueeze(-1)
        aatype = F.one_hot(batched["template_aatype"][:, 0],
                           num_classes=32).to(dtype)
        aatype_i = aatype.unsqueeze(1).expand(-1, sc.n_token, -1, -1)
        aatype_j = aatype.unsqueeze(2).expand(-1, -1, sc.n_token, -1)
        unit_vector = batched["template_unit_vector"][:, 0].to(
            dtype) * masked_by.unsqueeze(-1)
        backbone = (batched["template_backbone_frame_mask"][:, 0].to(dtype) *
                    masked_by).unsqueeze(-1)
        dense = torch.cat([
            dgram,
            pseudo_beta,
            aatype_i,
            aatype_j,
            unit_vector,
            backbone,
        ],
                          dim=-1)
        expected = model.linear_no_bias_a(dense)
        actual = model._project_single_template_features(batched, 0, masked_by)

    tol = (dict(atol=2e-5, rtol=2e-5)
           if dtype == torch.float32 else dict(atol=5e-3, rtol=5e-2))
    torch.testing.assert_close(actual, expected, **tol)
