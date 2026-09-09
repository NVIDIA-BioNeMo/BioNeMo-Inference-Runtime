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
from test_utils.boltz.create_and_load_weights import create_affinity_module_weights, load_affinity_module_weights_torch
from test_utils.boltz.ref_layers import RefAffinityModule

from bionemo_ir._torch.modules.boltz.affinity import AffinityModule
from bionemo_ir.models.boltz2.config import AffinityModuleConfig
from bionemo_ir.utils import str_dtype_to_torch


@dataclass(kw_only=True, frozen=True)
class Scenario:
    seq_len: int = 32
    dtype: str = "float32"
    num_dist_bins: int = 64
    token_z: int = 128
    token_s: int = 384


@pytest.mark.parametrize("sc", [Scenario(), Scenario(seq_len=256)])
def test_boltz_affinity_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    s = torch.randn(bs, sc.seq_len, sc.token_s, dtype=torch.float32).to(device)
    z = torch.randn(bs, sc.seq_len, sc.seq_len, sc.token_z, dtype=torch.float32).to(device)
    distogram = torch.randint(0, sc.num_dist_bins, (bs, sc.seq_len, sc.seq_len), dtype=torch.int32).to(device)
    cross_pair_mask_0 = torch.randint(0, 2, (bs, sc.seq_len, sc.seq_len), dtype=torch.float32).to(device)
    cross_pair_mask_1 = torch.randint(0, 2, (bs, sc.seq_len, sc.seq_len, 1), dtype=torch.float32).to(device)

    ref_module = RefAffinityModule.load_weights().to(device)
    ref_module.eval()

    weights_and_biases = create_affinity_module_weights(from_ref=ref_module)

    affinity_module = AffinityModule(
        AffinityModuleConfig(
            token_s=sc.token_s,
            token_z=sc.token_z,
            num_dist_bins=sc.num_dist_bins,
            pairformer_num_blocks=ref_module.pairformer_num_blocks,
            pairwise_head_width=ref_module.pairwise_head_width,
            pairwise_num_heads=4,
            dtype=sc.dtype,
            architecture="boltz2_affinity_module",
        )
    ).to(device)

    attn_metadatas = {}
    load_affinity_module_weights_torch(affinity_module, weights_and_biases)

    with torch.no_grad():
        ref_pred, ref_logits = ref_module(s, z, distogram, cross_pair_mask_0, cross_pair_mask_1)
        out_pred, out_logits, out_embedding = affinity_module(
            s,
            z,
            distogram,
            cross_pair_mask_0,
            cross_pair_mask_1,
            attn_metadatas=attn_metadatas,
        )

    torch.testing.assert_close(ref_pred, out_pred, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(ref_logits, out_logits, rtol=1e-5, atol=1e-3)
    assert out_embedding.shape[-1] == sc.token_s
