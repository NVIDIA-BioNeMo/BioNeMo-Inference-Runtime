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
"""OSS equivalence tests for the Protenix atom transformer (AF3 Algorithm 7)."""

import math
import os
from dataclasses import dataclass

import pytest
import torch

from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import ProtenixDiffusionTransformer
from tensorrt_bionemo.configs import DiffusionTransformerConfig
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.create_and_load_weights_from_protenixoss import convert_atom_transformer
from tests.common.test_utils.protenix.ref_layers_from_oss import RefProtenixAtomTransformerFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 256
    c_atom: int = 128
    c_atompair: int = 16
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    batch_size: int = 1
    dtype: str = "float32"
    precompute_bias: bool = True


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32"),
        Scenario(dtype="bfloat16"),
        Scenario(dtype="float32", n_atoms=200),
        Scenario(dtype="bfloat16", n_atoms=200),
        Scenario(dtype="float32", precompute_bias=False),
        Scenario(dtype="bfloat16", precompute_bias=False),
    ],
    ids=["fp32", "bf16", "fp32_ragged", "bf16_ragged", "fp32_no_precompute", "bf16_no_precompute"],
)
def test_protenix_atom_transformer(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    bs = sc.batch_size

    ref = (
        RefProtenixAtomTransformerFromOSS.build(
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
            n_blocks=sc.n_blocks,
            n_heads=sc.n_heads,
            n_queries=sc.n_queries,
            n_keys=sc.n_keys,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )

    # Defaults select the local atom variant.
    dit_config = DiffusionTransformerConfig(
        num_blocks=sc.n_blocks,
        num_heads=sc.n_heads,
        dim=sc.c_atom,
        dim_single_cond=sc.c_atom,
        dim_pairwise=sc.c_atompair,
        dtype=sc.dtype,
        precompute_bias=sc.precompute_bias,
    )
    model = ProtenixDiffusionTransformer(dit_config).to(device).eval()
    convert_atom_transformer(ref, model)

    n_blocks_win = math.ceil(sc.n_atoms / sc.n_queries)
    q = torch.randn(bs, sc.n_atoms, sc.c_atom, device=device)
    c = torch.randn(bs, sc.n_atoms, sc.c_atom, device=device)
    p_lm = torch.randn(bs, n_blocks_win, sc.n_queries, sc.n_keys, sc.c_atompair, device=device)
    atom_mask = torch.ones(bs, sc.n_atoms, device=device)

    attn_metadata = model.build_attn_metadata(n_blocks_win, sc.n_queries, sc.n_keys, device)

    with torch.inference_mode():
        ref_out = ref(q, c, p_lm)
        out = model(
            q.to(torch_dtype),
            c.to(torch_dtype),
            p_lm.to(torch_dtype),
            atom_mask,
            sc.n_queries,
            sc.n_keys,
            attn_metadata,
        )

    r = _rmse_ratio(out, ref_out)
    tol = 2e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
