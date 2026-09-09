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

import os
from dataclasses import dataclass

import pytest
import torch

from bionemo_ir._torch.attention_backend.utils import precompute_pair_masks
from bionemo_ir._torch.modules.openfold3.trunk import MSAModuleBlock
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl
from tests.common.test_utils.openfold3.create_and_load_weights_from_of3oss import (
    create_msa_module_block_weights_from_of3oss_torch,
    load_msa_module_block_weights_from_of3oss_torch,
)
from tests.common.test_utils.openfold3.ref_layers_from_oss import RefMSAModuleBlockFromOF3OSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 63
    n_seq: int = 127
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


def _create_msa_module_block(ref_module, sc, torch_dtype, last_block=False):
    """Helper to build an MSAModuleBlock from a reference module + scenario."""
    c_m = ref_module.msa_att_row.c_in
    c_z = ref_module.outer_product_mean.c_z
    is_of3oss = isinstance(ref_module, RefMSAModuleBlockFromOF3OSS)
    return MSAModuleBlock(
        local_layer_idx=0,
        c_m=c_m,
        c_z=c_z,
        c_hidden_msa_att=ref_module.msa_att_row.c_hidden,
        c_hidden_opm=ref_module.outer_product_mean.c_hidden,
        c_hidden_mul=(ref_module.pair_stack.tri_mul_out.c_hidden if is_of3oss else ref_module.tri_mul_out.c_hidden),
        c_hidden_pair_att=(
            ref_module.pair_stack.tri_att_start.c_hidden if is_of3oss else ref_module.tri_attn_start.c_hidden
        ),
        no_heads_msa=ref_module.msa_att_row.no_heads,
        no_heads_pair=(
            ref_module.pair_stack.tri_att_start.no_heads if is_of3oss else ref_module.tri_attn_start.no_heads
        ),
        transition_n=ref_module.msa_transition.n,
        opm_first=False,
        triangle_attn_backend=sc.triangle_attn_backend,
        support_batch=True,
        dtype=torch_dtype,
        eps=1e-5,
        inf=ref_module.msa_att_row.inf,
        outer_product_mean_bias={"proj_a": False, "proj_b": False, "proj_o": True},
        tri_mul_out_bias={"p_in": False, "g_in": False, "p_out": False, "g_out": False},
        tri_mul_in_bias={"p_in": False, "g_in": False, "p_out": False, "g_out": False},
        tri_attn_start_bias={"q": False, "k": False, "v": False, "g": False, "z": False, "o": False},
        tri_attn_end_bias={"q": False, "k": False, "v": False, "g": False, "z": False, "o": False},
        tri_attn_transposed_bias=False,
        last_block=last_block,
    )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(triangle_attn_backend="VANILLA"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
    ],
    ids=["vanilla", "cueequiv", "cutedsl_bf16"],
)
def test_msa_module_block(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    c_m = 64
    c_z = 128

    m = torch.randn(bs, sc.n_seq, sc.n_res, c_m, dtype=torch.float32).cuda()
    z = torch.randn(bs, sc.n_res, sc.n_res, c_z, dtype=torch.float32).cuda()
    # Production-style left-aligned masks: ``seq_mask`` and
    # ``msa_row_mask`` are both ``ones() + right-only zero-pad`` from the
    # collator, and ``msa_mask[b, s, n] = msa_row_mask[b, s] * seq_mask[b, n]``.
    # Use a near-full mask (only a few right-padded positions): the OPM
    # / pair-stack downstream normalizes by ``sum(mask)``; very sparse
    # masks legitimately diverge between production and reference at
    # fp32 due to small denominators. The objective here is to exercise
    # the left-aligned code path, not to stress mask-weighted reductions.
    seq_mask = make_left_aligned_mask(bs, sc.n_res, dtype=torch.float32, device="cuda", min_valid=max(sc.n_res - 4, 1))
    msa_row_mask = make_left_aligned_mask(
        bs, sc.n_seq, dtype=torch.float32, device="cuda", min_valid=max(sc.n_seq - 4, 1)
    )
    msa_mask = msa_row_mask[..., None] * seq_mask[..., None, :]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

    ref_module = RefMSAModuleBlockFromOF3OSS.load_weights()
    ref_module = ref_module.to(device)

    assert c_m == ref_module.msa_att_row.c_in
    assert c_z == ref_module.outer_product_mean.c_z

    module = _create_msa_module_block(ref_module, sc, torch_dtype)

    weights_and_biases = create_msa_module_block_weights_from_of3oss_torch(from_ref=ref_module)
    load_msa_module_block_weights_from_of3oss_torch(module, weights_and_biases)
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

    # Mask outputs at fully-padded positions before comparing: PyTorch
    # softmax-of-all-(-inf) emits NaN at padded query rows and the
    # CuTeDSL left-mask kernel handles ``actual_s_kv = 0`` rows via an
    # early-exit work tile.  Both implementations agree on the *valid*
    # sub-block; only compare there.
    m_keep = (msa_row_mask[..., None] * seq_mask[..., None, :]).unsqueeze(-1).float()  # [B, S, N, 1]
    z_keep = pair_mask.unsqueeze(-1).float()  # [B, N, N, 1]

    def _masked(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0) * keep

    if torch_dtype == torch.float32:
        torch.testing.assert_close(_masked(output_m, m_keep), _masked(ref_m_f32, m_keep), atol=2e-1, rtol=1e-2)
        torch.testing.assert_close(_masked(output_z, z_keep), _masked(ref_z_f32, z_keep), atol=2e-1, rtol=1e-2)
    else:
        ref_module_typed = ref_module.to(torch_dtype)
        with torch.no_grad():
            ref_m_typed, ref_z_typed = ref_module_typed(m_t, z_t, msa_mask_t, pair_mask_t)

        for name, out, ref_typed, ref_f32, keep in [
            ("MSA", output_m, ref_m_typed, ref_m_f32, m_keep),
            ("Pair", output_z, ref_z_typed, ref_z_f32, z_keep),
        ]:
            diff_ours = torch.max(torch.abs(_masked(out, keep) - _masked(ref_f32, keep)))
            diff_ref = torch.max(torch.abs(_masked(ref_typed, keep) - _masked(ref_f32, keep)))
            assert diff_ours <= 2.0 * diff_ref + 1e-3, f"{name}: ours_diff={diff_ours}, ref_diff={diff_ref}"


# ---------------------------------------------------------------------------
# Tests for precomputed pair masks on MSAModuleBlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(triangle_attn_backend="VANILLA"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="VANILLA", dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV", dtype="bfloat16"),
        Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
    ],
)
def test_msa_module_block_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    ref_module = RefMSAModuleBlockFromOF3OSS.load_weights()
    ref_module = ref_module.to(device)

    module = _create_msa_module_block(ref_module, sc, torch_dtype)

    weights_and_biases = create_msa_module_block_weights_from_of3oss_torch(from_ref=ref_module)
    load_msa_module_block_weights_from_of3oss_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()

    c_m = ref_module.msa_att_row.c_in
    c_z = ref_module.outer_product_mean.c_z

    m = torch.randn(bs, sc.n_seq, sc.n_res, c_m, dtype=torch_dtype, device=device)
    z = torch.randn(bs, sc.n_res, sc.n_res, c_z, dtype=torch_dtype, device=device)
    msa_mask = torch.randint(0, 2, (bs, sc.n_seq, sc.n_res), dtype=torch.float32, device=device).to(torch_dtype)
    seq_mask = make_left_aligned_mask(bs, sc.n_res, dtype=torch.float32, device=device)
    pair_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(torch_dtype)

    precomputed = precompute_pair_masks(
        sc.triangle_attn_backend, pair_mask, inf=ref_module.msa_att_row.inf, dtype=torch_dtype
    )

    with torch.inference_mode():
        out_m, out_z = module(m, z, msa_mask, pair_mask)
        out_m_pre, out_z_pre = module(m, z, msa_mask, pair_mask, precomputed_masks=precomputed)

    torch.testing.assert_close(out_m_pre, out_m, atol=0, rtol=0)
    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)
