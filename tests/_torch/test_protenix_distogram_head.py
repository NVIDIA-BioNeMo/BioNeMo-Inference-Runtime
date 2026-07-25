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
"""OSS equivalence tests for the Protenix distogram head."""
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.modules.protenix import ProtenixDistogramHead
from tensorrt_bionemo.models.protenix.config import DistogramHeadConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_distogram_head_torch
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixDistogramHeadFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 32
    c_z: int = 256
    no_bins: int = 64
    batch_size: int = 1
    dtype: str = "float32"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize(
    "sc", [Scenario(dtype="float32"),
           Scenario(dtype="bfloat16")],
    ids=["fp32", "bf16"])
def test_protenix_distogram_head(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    # Randomize the zero-initialized OSS projection.
    ref = RefProtenixDistogramHeadFromOSS.build(
        c_z=sc.c_z, no_bins=sc.no_bins).to(device=device,
                                           dtype=torch.float32).eval()
    with torch.no_grad():
        nn.init.normal_(ref.linear.weight, std=0.1)
        nn.init.normal_(ref.linear.bias, std=0.1)

    config = DistogramHeadConfig(c_z=sc.c_z,
                                 no_bins=sc.no_bins,
                                 dtype=sc.dtype)
    model = ProtenixDistogramHead(c_z=config.c_z,
                                  no_bins=config.no_bins,
                                  dtype=config.torch_dtype).to(device).eval()
    converted = convert_distogram_head_torch(config,
                                             ref.state_dict(),
                                             prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    z = torch.randn(sc.batch_size,
                    sc.n_token,
                    sc.n_token,
                    sc.c_z,
                    device=device)

    with torch.inference_mode():
        exp = ref(z.float())
        act = model(z.to(torch_dtype))

    assert act.shape == (sc.batch_size, sc.n_token, sc.n_token, sc.no_bins)
    # Symmetric logits are invariant under token-axis swap.
    torch.testing.assert_close(act, act.transpose(-2, -3))

    r = _rmse_ratio(act, exp)
    tol = 2e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
