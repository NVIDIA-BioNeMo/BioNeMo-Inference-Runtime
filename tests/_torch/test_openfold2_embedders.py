# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from dataclasses import dataclass

import pytest
import torch
from test_utils.openfold.create_and_load_weights import (
    create_input_embedder_weights,
    create_recycling_embedder_weights,
    load_input_embedder_weights_torch,
    load_recycling_embedder_weights_torch,
)
from test_utils.openfold.ref_layers import RefInputEmbedder, RefInputEmbedderMultimer, RefRecyclingEmbedder

from tensorrt_bionemo._torch.modules.openfold2.embedders import InputEmbedder, InputEmbedderMultimer, RecyclingEmbedder
from tensorrt_bionemo.configs import BaseConfig


@dataclass(kw_only=True, frozen=True)
class Scenario:
    N_res: int = 32
    N_clust: int = 64
    batch_size: int = 1
    dtype: str = "float32"


@pytest.mark.parametrize("sc", [Scenario(), Scenario(N_res=64, N_clust=32)])
def test_input_embedder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

    ref_mod = RefInputEmbedder.load_weights().cuda().eval()
    config = BaseConfig(
        tf_dim=ref_mod.tf_dim,
        msa_dim=ref_mod.msa_dim,
        c_z=ref_mod.c_z,
        c_m=ref_mod.c_m,
        relpos_k=ref_mod.relpos_k,
        dtype=sc.dtype,
    )
    mod = InputEmbedder(config).cuda().eval()

    dtype = config.torch_dtype

    ref_weights = create_input_embedder_weights(ref_mod)
    load_input_embedder_weights_torch(mod, ref_weights)

    token_feat = torch.randn(sc.batch_size, sc.N_res, config.tf_dim, device="cuda", dtype=dtype)
    residue_index = torch.randint(0, 128, (sc.batch_size, sc.N_res), device="cuda", dtype=torch.int32)
    msa_feat = torch.randn(sc.batch_size, sc.N_clust, sc.N_res, config.msa_dim, device="cuda", dtype=dtype)

    with torch.no_grad():
        ref_m, ref_z = ref_mod(token_feat, residue_index, msa_feat)
        m, z = mod(token_feat, residue_index, msa_feat)

    torch.testing.assert_close(ref_m, m, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ref_z, z, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sc", [Scenario(), Scenario(N_res=64, N_clust=32)])
def test_input_embedder_multimer(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    if os.environ.get("ALPHAFOLD2_MULTIMER_1_CKPT") is None:
        pytest.skip("ALPHAFOLD2_MULTIMER_1_CKPT is not set")

    ref_mod = RefInputEmbedderMultimer.load_weights().cuda().eval()
    config = BaseConfig(
        tf_dim=ref_mod.tf_dim,
        msa_dim=ref_mod.msa_dim,
        c_z=ref_mod.c_z,
        c_m=ref_mod.c_m,
        max_relative_idx=ref_mod.max_relative_idx,
        use_chain_relative=ref_mod.use_chain_relative,
        max_relative_chain=ref_mod.max_relative_chain,
        dtype=sc.dtype,
    )
    mod = InputEmbedderMultimer(config).cuda().eval()

    dtype = config.torch_dtype

    ref_weights = create_input_embedder_weights(ref_mod)
    load_input_embedder_weights_torch(mod, ref_weights)

    token_feat = torch.randn(sc.batch_size, sc.N_res, config.tf_dim, device="cuda", dtype=dtype)
    residue_index = torch.randint(0, 128, (sc.batch_size, sc.N_res), device="cuda", dtype=torch.int32)
    msa_feat = torch.randn(sc.batch_size, sc.N_clust, sc.N_res, config.msa_dim, device="cuda", dtype=dtype)
    asym_id = torch.randint(0, 42, (sc.batch_size, sc.N_res), device="cuda", dtype=torch.int32)
    entity_id = torch.randint(0, 42, (sc.batch_size, sc.N_res), device="cuda", dtype=torch.int32)
    sym_id = torch.randint(0, 42, (sc.batch_size, sc.N_res), device="cuda", dtype=torch.int32)
    with torch.no_grad():
        ref_m, ref_z = ref_mod(token_feat, residue_index, msa_feat, asym_id, entity_id, sym_id)
        m, z = mod(token_feat, residue_index, msa_feat, asym_id, entity_id, sym_id)

    torch.testing.assert_close(ref_m, m, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ref_z, z, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sc", [Scenario(), Scenario(N_res=64)])
def test_recycling_embedder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

    ref_mod = RefRecyclingEmbedder.load_weights().cuda().eval()
    config = BaseConfig(
        c_m=ref_mod.c_m, c_z=ref_mod.c_z, min_bin=ref_mod.min_bin, max_bin=ref_mod.max_bin, no_bins=ref_mod.no_bins
    )
    mod = RecyclingEmbedder(config).cuda().eval()

    dtype = config.torch_dtype

    ref_weights = create_recycling_embedder_weights(ref_mod)
    load_recycling_embedder_weights_torch(mod, ref_weights)

    m = torch.randn(sc.batch_size, sc.N_res, config.c_m, device="cuda", dtype=dtype)
    z = torch.randn(sc.batch_size, sc.N_res, sc.N_res, config.c_z, device="cuda", dtype=dtype)
    x = torch.randn(sc.batch_size, sc.N_res, 3, device="cuda", dtype=dtype)

    with torch.no_grad():
        ref_m, ref_z = ref_mod(m, z, x)
        m, z = mod(m, z, x)

    torch.testing.assert_close(ref_m, m, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ref_z, z, rtol=1e-4, atol=1e-4)
