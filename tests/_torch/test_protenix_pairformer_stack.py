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
"""OSS equivalence tests for the Protenix pairformer stack."""
import os
from dataclasses import dataclass

import pytest
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch

from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerModule
from tensorrt_bionemo.configs import PairformerConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_pairformer_stack_torch
from tests._torch import skip_if_cutedsl
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixPairformerStackFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 24
    # Four triangle heads satisfy CuTeDSL alignment.
    c_s: int = 128
    c_z: int = 128
    n_blocks: int = 2
    n_heads: int = 4
    head_width: int = 32
    batch_size: int = 1
    dtype: str = "float32"
    tri_backend: str = "VANILLA"
    pair_backend: str = "SDPA"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32"),
    Scenario(dtype="bfloat16"),
    Scenario(dtype="bfloat16", tri_backend="CuTeDSL", pair_backend="CuTeDSL"),
],
                         ids=["fp32", "bf16", "cutedsl"])
def test_protenix_pairformer_stack(sc: Scenario):
    skip_if_cutedsl(sc.tri_backend)
    skip_if_cutedsl(sc.pair_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    ref = RefProtenixPairformerStackFromOSS.build(
        n_blocks=sc.n_blocks, n_heads=sc.n_heads, c_z=sc.c_z,
        c_s=sc.c_s).to(device=device, dtype=torch.float32).eval()

    config = PairformerConfig(
        token_s=sc.c_s,
        token_z=sc.c_z,
        num_blocks=sc.n_blocks,
        num_heads=sc.n_heads,
        pairwise_head_width=sc.head_width,
        pairwise_num_heads=sc.c_z // sc.head_width,
        no_update_s=False,
        attention_initial_norm=True,
        version="v1",
        dtype=sc.dtype,
        triangle_attention_backend=sc.tri_backend,
        pairwise_attention_backend=sc.pair_backend,
    )
    model = PairformerModule(config).to(device).eval()
    converted = convert_pairformer_stack_torch(config,
                                               ref.state_dict(),
                                               prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    bs, n = sc.batch_size, sc.n_token
    s = torch.randn(bs, n, sc.c_s, device=device)
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    mask = torch.ones(bs, n, device=device)
    pair_mask = torch.ones(bs, n, n, device=device)

    with torch.inference_mode():
        ref_s, ref_z = ref(s, z, pair_mask=None)
        out_s, out_z = model(s.to(torch_dtype), z.to(torch_dtype),
                             mask.to(torch_dtype), pair_mask.to(torch_dtype))

    tol = 5e-3 if torch_dtype == torch.float32 else 8e-2
    r_s, r_z = _rmse_ratio(out_s, ref_s), _rmse_ratio(out_z, ref_z)
    assert r_s < tol, f"s rmse_ratio={r_s:.3e} exceeds {tol:.0e} ({sc.dtype})"
    assert r_z < tol, f"z rmse_ratio={r_z:.3e} exceeds {tol:.0e} ({sc.dtype})"
