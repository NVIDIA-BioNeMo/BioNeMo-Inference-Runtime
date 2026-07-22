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
"""OSS equivalence tests for the Protenix MSA module (AF3 Algorithm 8)."""
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from tensorrt_llm_lite._utils import str_dtype_to_torch

from tensorrt_bionemo._torch.modules.protenix import ProtenixMSAModule
from tensorrt_bionemo.models.protenix.config import ProtenixMSAModuleConfig
from tensorrt_bionemo.models.protenix.convert import convert_msa_module_torch
from tests._torch import skip_if_cutedsl
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixMSAModuleFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 16
    n_msa: int = 8
    c_m: int = 128
    # Four pair heads satisfy CuTeDSL alignment.
    c_z: int = 128
    c_s_inputs: int = 449
    n_blocks: int = 2
    batch_size: int = 1
    dtype: str = "float32"
    tri_backend: str = "VANILLA"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _config(sc: Scenario) -> ProtenixMSAModuleConfig:
    return ProtenixMSAModuleConfig(c_m=sc.c_m,
                                   c_z=sc.c_z,
                                   c_hidden_mul=sc.c_z,
                                   c_hidden_pair_att=32,
                                   no_heads_pair=sc.c_z // 32,
                                   no_blocks=sc.n_blocks,
                                   c_s_inputs=sc.c_s_inputs,
                                   dtype=sc.dtype,
                                   triangle_attention_backend=sc.tri_backend)


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32"),
    Scenario(dtype="bfloat16"),
    Scenario(dtype="bfloat16", tri_backend="CuTeDSL"),
],
                         ids=["fp32", "bf16", "cutedsl"])
def test_protenix_msa_module(sc: Scenario):
    skip_if_cutedsl(sc.tri_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    ref = RefProtenixMSAModuleFromOSS.build(n_blocks=sc.n_blocks,
                                            c_m=sc.c_m,
                                            c_z=sc.c_z,
                                            c_s_inputs=sc.c_s_inputs).to(
                                                device=device,
                                                dtype=torch.float32).eval()

    config = _config(sc)
    model = ProtenixMSAModule(config).to(device).eval()
    converted = convert_msa_module_torch(config, ref.state_dict(), prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    bs, n, s_msa = sc.batch_size, sc.n_token, sc.n_msa
    feat = {
        "msa": torch.randint(0, 32, (bs, s_msa, n), device=device),
        "has_deletion": (torch.randn(bs, s_msa, n, device=device) > 0).float(),
        "deletion_value": torch.rand(bs, s_msa, n, device=device),
    }
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    s_inputs = torch.randn(bs, n, sc.c_s_inputs, device=device)

    with torch.inference_mode():
        ref_z = ref(feat, z, s_inputs, pair_mask=None)
        act_z = model(
            {
                k: v.to(torch_dtype) if v.is_floating_point() else v
                for k, v in feat.items()
            }, z.to(torch_dtype), s_inputs.to(torch_dtype))

    r = _rmse_ratio(act_z, ref_z)
    tol = 1e-2 if torch_dtype == torch.float32 else 8e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_protenix_msa_embedding_gather_matches_dense(dtype_name: str):
    """Embedding + scalar columns matches the original one-hot concat Linear."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(dtype_name)
    sc = Scenario(dtype=dtype_name, n_token=12, n_msa=5)
    model = ProtenixMSAModule(_config(sc)).to(device).eval()
    with torch.no_grad():
        model.linear_no_bias_m.weight.normal_(mean=0.0, std=0.02)
        model.linear_no_bias_s.weight.normal_(mean=0.0, std=0.02)

    feat = {
        "msa":
        torch.randint(0,
                      32, (sc.batch_size, sc.n_msa, sc.n_token),
                      device=device),
        "has_deletion": (torch.randn(sc.batch_size,
                                     sc.n_msa,
                                     sc.n_token,
                                     device=device) > 0).to(dtype),
        "deletion_value":
        torch.rand(sc.batch_size,
                   sc.n_msa,
                   sc.n_token,
                   dtype=dtype,
                   device=device),
    }
    s_inputs = torch.randn(sc.batch_size,
                           sc.n_token,
                           sc.c_s_inputs,
                           dtype=dtype,
                           device=device)

    with torch.inference_mode():
        dense_input = torch.cat([
            F.one_hot(feat["msa"], num_classes=32).to(dtype),
            feat["has_deletion"].unsqueeze(-1),
            feat["deletion_value"].unsqueeze(-1),
        ],
                                dim=-1)
        expected = model.linear_no_bias_m(dense_input)
        expected = expected + model.linear_no_bias_s(s_inputs).unsqueeze(1)
        actual = model._embed_msa(feat, s_inputs)

    tol = (dict(atol=1e-5, rtol=1e-5)
           if dtype == torch.float32 else dict(atol=3e-3, rtol=3e-2))
    torch.testing.assert_close(actual, expected, **tol)


def test_protenix_msa_module_precomputed_masks():
    """Precomputed and inline pair masks are bit-identical."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    sc = Scenario(dtype="float32")
    model = ProtenixMSAModule(_config(sc)).to(device).eval()

    bs, n, s_msa = sc.batch_size, sc.n_token, sc.n_msa
    feat = {
        "msa": torch.randint(0, 32, (bs, s_msa, n), device=device),
        "has_deletion": (torch.randn(bs, s_msa, n, device=device) > 0).float(),
        "deletion_value": torch.rand(bs, s_msa, n, device=device),
    }
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    s_inputs = torch.randn(bs, n, sc.c_s_inputs, device=device)
    pair_mask = torch.ones(bs, n, n, device=device)

    with torch.inference_mode():
        out_default = model(feat, z, s_inputs, pair_mask=pair_mask)
        precomputed = model.build_pair_masks(pair_mask)
        out_precomputed = model(feat,
                                z,
                                s_inputs,
                                pair_mask=pair_mask,
                                precomputed_masks=precomputed)
    assert torch.equal(out_default, out_precomputed)
