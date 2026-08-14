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
"""OSS equivalence tests for Protenix relative-position encoding."""

import os
from dataclasses import dataclass

import pytest
import torch

from bionemo_ir._torch.layers.position_encoders import RelativePositionEncoder
from bionemo_ir.models.protenix.config import RelativePositionEncodingConfig
from bionemo_ir.models.protenix.convert import convert_relative_position_encoding_torch
from bionemo_ir.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import RefProtenixRelativePositionEncodingFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 96
    n_chains: int = 4
    tokens_per_residue: int = 2
    r_max: int = 32
    s_max: int = 2
    c_z: int = 256
    batch_size: int = 1
    dtype: str = "float32"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _make_features(sc: Scenario, device: torch.device) -> dict:
    """Build token features that exercise cross-chain and token-offset bins."""
    N = sc.n_token
    asym = torch.empty(N, dtype=torch.long)
    residue = torch.empty(N, dtype=torch.long)
    token = torch.empty(N, dtype=torch.long)
    for ci, idx in enumerate(torch.tensor_split(torch.arange(N), sc.n_chains)):
        asym[idx] = ci
        local = torch.arange(len(idx))
        residue[idx] = local // sc.tokens_per_residue
        token[idx] = local
    entity = asym // 2
    sym = asym % 2

    def _batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).expand(sc.batch_size, -1).contiguous().to(device)

    return {
        "asym_id": _batch(asym),
        "residue_index": _batch(residue),
        "entity_id": _batch(entity),
        "token_index": _batch(token),
        "sym_id": _batch(sym),
    }


@pytest.mark.parametrize("sc", [Scenario(dtype="float32"), Scenario(dtype="bfloat16")], ids=["fp32", "bf16"])
def test_protenix_relative_position_encoding(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    ref = (
        RefProtenixRelativePositionEncodingFromOSS.build(r_max=sc.r_max, s_max=sc.s_max, c_z=sc.c_z)
        .to(device=device, dtype=torch.float32)
        .eval()
    )

    config = RelativePositionEncodingConfig(r_max=sc.r_max, s_max=sc.s_max, c_z=sc.c_z, dtype=sc.dtype)
    model = (
        RelativePositionEncoder(
            token_z=config.c_z,
            r_max=config.r_max,
            s_max=config.s_max,
            fix_sym_check=config.fix_sym_check,
            cyclic_pos_enc=config.cyclic_pos_enc,
            dtype=config.torch_dtype,
        )
        .to(device)
        .eval()
    )
    converted = convert_relative_position_encoding_torch(config, ref.state_dict(), prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    feat = _make_features(sc, device)

    with torch.inference_mode():
        oss_relp = ref.generate_relp({k: v.clone() for k, v in feat.items()})["relp"]
        exp = ref(oss_relp)
        trt_relp = model.generate_relp(**feat)
        act = model(relp=trt_relp)

    # Bucketed one-hot features must match exactly.
    torch.testing.assert_close(trt_relp, oss_relp)

    r = _rmse_ratio(act, exp)
    tol = 2e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
