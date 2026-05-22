# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import pytest
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.openfold.create_and_load_weights import (
    create_extra_msa_block_weights, load_extra_msa_block_weights_torch)
from test_utils.openfold.ref_layers import RefExtraMSABlock

from tensorrt_bionemo._torch.attention_backend.utils import \
    precompute_pair_masks
from tensorrt_bionemo._torch.modules.openfold2.trunk import ExtraMSABlock
from tensorrt_bionemo.mapping import Mapping
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 64
    n_seq: int = 128
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"
    opm_chunk_size: Optional[int] = None
    opm_mask_chunk_size: Optional[int] = None


def _create_extra_msa_block(ref_module, sc, torch_dtype):
    """Helper to build an ExtraMSABlock from a reference module + scenario."""
    return ExtraMSABlock(
        local_layer_idx=0,
        c_m=ref_module.c_m,
        c_z=ref_module.c_z,
        c_hidden_msa_att=ref_module.c_hidden_msa_att,
        c_hidden_opm=ref_module.c_hidden_opm,
        c_hidden_mul=ref_module.c_hidden_mul,
        c_hidden_pair_att=ref_module.c_hidden_pair_att,
        no_heads_msa=ref_module.no_heads_msa,
        no_heads_pair=ref_module.no_heads_pair,
        transition_n=ref_module.transition_n,
        opm_first=ref_module.opm_first,
        triangle_attn_backend=sc.triangle_attn_backend,
        support_batch=True,
        dtype=torch_dtype,
        triangle_attn_node_chunk_size=0,
        eps=ref_module.eps,
        inf=ref_module.inf,
        opm_chunk_size=sc.opm_chunk_size,
        opm_mask_chunk_size=sc.opm_mask_chunk_size,
        mapping=Mapping(),
    )


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV"),
    Scenario(triangle_attn_backend="VANILLA",
             opm_chunk_size=16,
             opm_mask_chunk_size=16),
    Scenario(triangle_attn_backend="CUEQUIV",
             opm_chunk_size=16,
             opm_mask_chunk_size=16),
    Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
],
                         ids=[
                             "vanilla", "cueequiv", "vanilla_chunked",
                             "cueequiv_chunked", "cutedsl_bf16"
                         ])
def test_extra_msa_block(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefExtraMSABlock.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_extra_msa_block_weights(from_ref=ref_module)
    m = torch.randn(bs,
                    sc.n_seq,
                    sc.n_res,
                    ref_module.c_m,
                    dtype=torch.float32).cuda()
    z = torch.randn(bs,
                    sc.n_res,
                    sc.n_res,
                    ref_module.c_z,
                    dtype=torch.float32).cuda()
    # In production, both ``seq_mask`` and ``msa_row_mask`` are
    # left-aligned (``ones`` + right-only zero padding in the collator),
    # and ``msa_mask[b, s, n] = msa_row_mask[b, s] * seq_mask[b, n]``.
    seq_mask = make_left_aligned_mask(bs,
                                      sc.n_res,
                                      dtype=torch.float32,
                                      device="cuda",
                                      min_valid=sc.n_res // 2)
    msa_row_mask = make_left_aligned_mask(bs,
                                          sc.n_seq,
                                          dtype=torch.float32,
                                          device="cuda",
                                          min_valid=sc.n_seq // 2)
    msa_mask = msa_row_mask[..., None] * seq_mask[..., None, :]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

    module = _create_extra_msa_block(ref_module, sc, torch_dtype)

    load_extra_msa_block_weights_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()
    ref_module.eval()

    with torch.no_grad():
        ref_m_f32, ref_z_f32 = ref_module(m, z, msa_mask, pair_mask)

        m_t = m.to(torch_dtype)
        z_t = z.to(torch_dtype)
        msa_mask_t = msa_mask.to(torch_dtype)
        pair_mask_t = pair_mask.to(torch_dtype)

        output_m, output_z = module(m_t, z_t, msa_mask_t, pair_mask_t)

    if torch_dtype == torch.float32:
        torch.testing.assert_close(output_m, ref_m_f32, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(output_z, ref_z_f32, atol=1e-3, rtol=1e-4)
    else:
        ref_module_typed = ref_module.to(torch_dtype)
        with torch.no_grad():
            ref_m_typed, ref_z_typed = ref_module_typed(
                m_t, z_t, msa_mask_t, pair_mask_t)

        tol_mult = 10.0 if sc.triangle_attn_backend == "CuTeDSL" else 2.0
        for name, out, ref_typed, ref_f32 in [
            ("MSA", output_m, ref_m_typed, ref_m_f32),
            ("Pair", output_z, ref_z_typed, ref_z_f32),
        ]:
            diff_ours = torch.max(torch.abs(out.float() - ref_f32))
            diff_ref = torch.max(torch.abs(ref_typed.float() - ref_f32))
            assert diff_ours <= tol_mult * diff_ref + 1e-3, (
                f"{name}: ours_diff={diff_ours}, ref_diff={diff_ref}")


# ---------------------------------------------------------------------------
# Tests for precomputed pair masks on ExtraMSABlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV"),
    Scenario(triangle_attn_backend="VANILLA", dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV", dtype="bfloat16"),
    Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
])
def test_extra_msa_block_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    ref_module = RefExtraMSABlock.load_weights()
    ref_module = ref_module.to(device)
    weights_and_biases = create_extra_msa_block_weights(from_ref=ref_module)

    module = _create_extra_msa_block(ref_module, sc, torch_dtype)
    load_extra_msa_block_weights_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()

    m = torch.randn(bs,
                    sc.n_seq,
                    sc.n_res,
                    ref_module.c_m,
                    dtype=torch_dtype,
                    device=device)
    z = torch.randn(bs,
                    sc.n_res,
                    sc.n_res,
                    ref_module.c_z,
                    dtype=torch_dtype,
                    device=device)
    seq_mask = make_left_aligned_mask(bs,
                                      sc.n_res,
                                      dtype=torch.float32,
                                      device=device,
                                      min_valid=sc.n_res // 2)
    msa_row_mask = make_left_aligned_mask(bs,
                                          sc.n_seq,
                                          dtype=torch.float32,
                                          device=device,
                                          min_valid=sc.n_seq // 2)
    msa_mask = (msa_row_mask[..., None] *
                seq_mask[..., None, :]).to(torch_dtype)
    pair_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(torch_dtype)

    precomputed = precompute_pair_masks(sc.triangle_attn_backend,
                                        pair_mask,
                                        inf=ref_module.inf,
                                        dtype=torch_dtype)

    with torch.inference_mode():
        out_m, out_z = module(m, z, msa_mask, pair_mask)
        out_m_pre, out_z_pre = module(m,
                                      z,
                                      msa_mask,
                                      pair_mask,
                                      precomputed_masks=precomputed)

    torch.testing.assert_close(out_m_pre, out_m, atol=0, rtol=0)
    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)
