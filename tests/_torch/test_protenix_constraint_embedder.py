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
"""OSS equivalence tests for the Protenix constraint embedder."""
import os

import pytest
import torch
import torch.nn as nn
from tensorrt_llm_lite._utils import str_dtype_to_torch

from tensorrt_bionemo._torch.modules.protenix import ProtenixConstraintEmbedder
from tensorrt_bionemo.models.protenix.config import ConstraintEmbedderConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_constraint_embedder_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixConstraintEmbedderFromOSS


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"],
                         ids=["fp32", "bf16"])
def test_protenix_constraint_embedder(dtype: str):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(dtype)
    c_z, N, B = 32, 8, 1

    # Randomize the zero-initialized OSS projections.
    ref = RefProtenixConstraintEmbedderFromOSS.build(c_constraint_z=c_z).to(
        device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)

    config = ConstraintEmbedderConfig(c_constraint_z=c_z,
                                      pocket_enable=True,
                                      contact_enable=True,
                                      contact_atom_enable=True,
                                      dtype=dtype)
    model = ProtenixConstraintEmbedder(config).to(device).eval()
    converted = convert_constraint_embedder_torch(config,
                                                  ref.state_dict(),
                                                  prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    feat = {
        "pocket": torch.randn(B, N, N, 1, device=device),
        "contact": torch.randn(B, N, N, 2, device=device),
        "contact_atom": torch.randn(B, N, N, 2, device=device),
    }

    with torch.inference_mode():
        exp = ref(feat)
        act = model({k: v.to(torch_dtype) for k, v in feat.items()})

    r = _rmse_ratio(act, exp)
    tol = 2e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({dtype})"


def test_protenix_constraint_embedder_disabled_returns_none():
    """The all-disabled default has no parameters or output."""
    config = ConstraintEmbedderConfig()  # all *_enable default False
    model = ProtenixConstraintEmbedder(config).eval()
    assert sum(p.numel() for p in model.parameters()) == 0
    assert model({}) is None
