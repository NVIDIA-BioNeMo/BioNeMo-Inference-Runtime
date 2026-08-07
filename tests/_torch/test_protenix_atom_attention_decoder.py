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
"""OSS equivalence tests for the Protenix atom decoder (AF3 Algorithm 6)."""

import math
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.modules.protenix import ProtenixAtomAttentionDecoder
from tensorrt_bionemo.models.protenix.config import AtomAttentionDecoderConfig
from tensorrt_bionemo.models.protenix.convert import convert_atom_attention_decoder_torch
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import RefProtenixAtomAttentionDecoderFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 64
    n_token: int = 16
    c_token: int = 768
    c_atom: int = 128
    c_atompair: int = 16
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    batch_size: int = 1
    dtype: str = "float32"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32"),
        Scenario(dtype="bfloat16"),
    ],
    ids=["fp32", "bf16"],
)
def test_protenix_atom_attention_decoder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    B, Na, Nt = sc.batch_size, sc.n_atoms, sc.n_token

    # Randomize the OSS reference for non-trivial coverage.
    ref = (
        RefProtenixAtomAttentionDecoderFromOSS.build(
            n_blocks=sc.n_blocks,
            n_heads=sc.n_heads,
            c_token=sc.c_token,
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
            n_queries=sc.n_queries,
            n_keys=sc.n_keys,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)

    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        dtype=sc.dtype,
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    converted = convert_atom_attention_decoder_torch(config, ref.state_dict(), prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    K = math.ceil(Na / sc.n_queries)
    atom_to_token_idx = (torch.arange(Na, device=device) // (Na // Nt)).long().unsqueeze(0).expand(B, Na)
    a = torch.randn(B, Nt, sc.c_token, device=device)
    q_skip = torch.randn(B, Na, sc.c_atom, device=device)
    c_skip = torch.randn(B, Na, sc.c_atom, device=device)
    p_skip = torch.randn(B, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device)

    with torch.inference_mode():
        exp = ref(atom_to_token_idx, a, q_skip, c_skip, p_skip)
        act = model(
            atom_to_token_idx, a.to(torch_dtype), q_skip.to(torch_dtype), c_skip.to(torch_dtype), p_skip.to(torch_dtype)
        )

    r = _rmse_ratio(act, exp)
    tol = 3e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
