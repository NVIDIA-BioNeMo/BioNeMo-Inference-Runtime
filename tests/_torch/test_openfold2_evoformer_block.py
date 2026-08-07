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
from unittest.mock import MagicMock

import pytest
import torch
from test_utils.openfold.create_and_load_weights import (
    create_evoformer_block_weights,
    load_evoformer_block_weights_torch,
)
from test_utils.openfold.ref_layers import RefEvoformerBlock

from tensorrt_bionemo._torch.attention_backend.utils import PrecomputedPairMasks, precompute_pair_masks
from tensorrt_bionemo._torch.layers.transformers.evoformer import EvoformerBlock
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 64
    n_seq: int = 128
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


def _create_evoformer_block(ref_module, sc, torch_dtype, pair_mask_left_aligned=True):
    """Helper to build an EvoformerBlock from a reference module + scenario."""
    return EvoformerBlock(
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
        no_column_attention=ref_module.no_column_attention,
        opm_first=ref_module.opm_first,
        triangle_attn_backend=sc.triangle_attn_backend,
        support_batch=True,
        dtype=torch_dtype,
        pair_mask_left_aligned=pair_mask_left_aligned,
        triangle_attn_node_chunk_size=0,
        eps=ref_module.eps,
        inf=ref_module.inf,
    )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(triangle_attn_backend="VANILLA"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
    ],
)
def test_evoformer_block(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    ref_module = RefEvoformerBlock.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_evoformer_block_weights(from_ref=ref_module)
    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_module.c_m, dtype=torch.float32).cuda()
    z = torch.randn(bs, sc.n_res, sc.n_res, ref_module.c_z, dtype=torch.float32).cuda()
    # In production, both ``seq_mask`` and ``msa_row_mask`` are
    # left-aligned (``ones`` + right-only zero padding in the collator),
    # and ``msa_mask[b, s, n] = msa_row_mask[b, s] * seq_mask[b, n]``.
    # Build the same structure here so the CuTeDSL MSA-row attention's
    # left-mask kernel sees the production padding pattern.
    seq_mask = make_left_aligned_mask(bs, sc.n_res, dtype=torch.float32, device="cuda", min_valid=sc.n_res // 2)
    msa_row_mask = make_left_aligned_mask(bs, sc.n_seq, dtype=torch.float32, device="cuda", min_valid=sc.n_seq // 2)
    msa_mask = msa_row_mask[..., None] * seq_mask[..., None, :]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

    module = _create_evoformer_block(ref_module, sc, torch_dtype)

    load_evoformer_block_weights_torch(module, weights_and_biases)
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

    # Mask outputs at padded positions before comparing: PyTorch reference
    # produces NaN at fully-masked-key softmax rows (seq_mask padding) and
    # the CuTeDSL MSA-row attention emits early-exit (garbage) tiles for
    # MSA rows where ``msa_row_mask = 0``.  Both implementations agree on
    # the *valid* sub-block; only compare there.
    m_keep = (msa_row_mask[..., None] * seq_mask[..., None, :]).unsqueeze(-1).float()  # [B, S, N, 1]
    z_keep = pair_mask.unsqueeze(-1).float()  # [B, N, N, 1]

    def _masked(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0) * keep

    if torch_dtype == torch.float32:
        torch.testing.assert_close(_masked(output_m, m_keep), _masked(ref_m_f32, m_keep), atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(_masked(output_z, z_keep), _masked(ref_z_f32, z_keep), atol=1e-3, rtol=1e-4)
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
# Tests for precomputed pair masks on EvoformerBlock
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
def test_evoformer_block_precomputed_masks(sc: Scenario):
    """Outputs with precomputed masks must exactly match the original path."""
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    ref_module = RefEvoformerBlock.load_weights()
    ref_module = ref_module.to(device)
    weights_and_biases = create_evoformer_block_weights(from_ref=ref_module)

    module = _create_evoformer_block(ref_module, sc, torch_dtype)
    load_evoformer_block_weights_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()

    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_module.c_m, dtype=torch_dtype, device=device)
    z = torch.randn(bs, sc.n_res, sc.n_res, ref_module.c_z, dtype=torch_dtype, device=device)
    seq_mask = make_left_aligned_mask(bs, sc.n_res, dtype=torch.float32, device=device, min_valid=sc.n_res // 2)
    msa_row_mask = make_left_aligned_mask(bs, sc.n_seq, dtype=torch.float32, device=device, min_valid=sc.n_seq // 2)
    msa_mask = (msa_row_mask[..., None] * seq_mask[..., None, :]).to(torch_dtype)
    pair_mask = (seq_mask[..., None] * seq_mask[..., None, :]).to(torch_dtype)

    precomputed = precompute_pair_masks(sc.triangle_attn_backend, pair_mask, inf=ref_module.inf, dtype=torch_dtype)

    with torch.inference_mode():
        out_m, out_z = module(m, z, msa_mask, pair_mask)
        out_m_pre, out_z_pre = module(m, z, msa_mask, pair_mask, precomputed_masks=precomputed)

    torch.testing.assert_close(out_m_pre, out_m, atol=0, rtol=0)
    torch.testing.assert_close(out_z_pre, out_z, atol=0, rtol=0)


@pytest.mark.parametrize("pair_mask_left_aligned", [True, False])
def test_evoformer_pair_mask_contract_propagates(pair_mask_left_aligned):
    ref_module = RefEvoformerBlock.load_weights()
    module = _create_evoformer_block(
        ref_module,
        Scenario(),
        torch.float32,
        pair_mask_left_aligned=pair_mask_left_aligned,
    )

    assert module.pair_mask_left_aligned is pair_mask_left_aligned
    assert module.tri_mul_out.pair_mask_left_aligned is pair_mask_left_aligned
    assert module.tri_mul_in.pair_mask_left_aligned is pair_mask_left_aligned
    assert module.tri_attn_start.pair_mask_left_aligned is pair_mask_left_aligned
    assert module.tri_attn_end.pair_mask_left_aligned is pair_mask_left_aligned


# ---------------------------------------------------------------------------
# Tests for dual_gemm_x_x ``actual_seqlen`` wiring in ``EvoformerBlock.forward``
# ---------------------------------------------------------------------------


class _SpyModule(torch.nn.Module):
    """``nn.Module`` wrapper around a ``MagicMock`` so PyTorch accepts it as a
    submodule replacement.  Forwards ``call_args`` / ``reset_mock`` through
    to the inner mock for spy-style assertions."""

    def __init__(self, output_like: torch.Tensor):
        super().__init__()
        self._spy = MagicMock(return_value=torch.zeros_like(output_like))

    def forward(self, *args, **kwargs):
        return self._spy(*args, **kwargs)

    @property
    def call_args(self):
        return self._spy.call_args

    def reset_mock(self):
        self._spy.reset_mock()


def test_evoformer_block_actual_seqlen_wiring():
    """``EvoformerBlock.forward`` forwards ``mask_bias`` /
    ``mask_bias_transposed`` as ``actual_seqlen`` to ``tri_mul_out`` /
    ``tri_mul_in`` only when the precomputed masks are in the CuTeDSL
    ``int32`` per-row-count form.  For default-backend (float additive
    bias) precomputed masks, and when no precomputed masks are supplied,
    both nodes must receive ``actual_seqlen=None`` so the dual_gemm_x_x
    wrapper falls back to the in-call ``mask.sum(-1)`` reduction.
    """
    torch.manual_seed(0)
    sc = Scenario(triangle_attn_backend="VANILLA")
    device = torch.device("cuda")
    torch_dtype = torch.float32

    ref_module = RefEvoformerBlock.load_weights()
    ref_module = ref_module.to(device)
    module = _create_evoformer_block(ref_module, sc, torch_dtype).to(device)

    B = 1
    m = torch.randn(B, sc.n_seq, sc.n_res, ref_module.c_m, dtype=torch_dtype, device=device)
    z = torch.randn(B, sc.n_res, sc.n_res, ref_module.c_z, dtype=torch_dtype, device=device)
    msa_mask = torch.ones(B, sc.n_seq, sc.n_res, dtype=torch_dtype, device=device)
    pair_mask = torch.ones(B, sc.n_res, sc.n_res, dtype=torch_dtype, device=device)

    module.outer_product_mean = _SpyModule(z)
    module.msa_att_row = _SpyModule(m)
    if not module.no_column_attention:
        module.msa_att_col = _SpyModule(m)
    module.msa_transition = _SpyModule(m)
    module.tri_mul_out = _SpyModule(z)
    module.tri_mul_in = _SpyModule(z)
    module.tri_attn_start = _SpyModule(z)
    module.tri_attn_end = _SpyModule(z)
    module.pair_transition = _SpyModule(z)

    mb_int32 = torch.zeros(B, sc.n_res, dtype=torch.int32, device=device)
    mb_int32_t = torch.zeros(B, sc.n_res, dtype=torch.int32, device=device)
    pre_cutedsl = PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=mb_int32,
        mask_bias_transposed=mb_int32_t,
    )
    module(m, z, msa_mask, pair_mask, precomputed_masks=pre_cutedsl)
    assert module.tri_mul_out.call_args.kwargs["actual_seqlen"] is mb_int32
    assert module.tri_mul_in.call_args.kwargs["actual_seqlen"] is mb_int32_t

    module.pair_mask_left_aligned = False
    module.tri_mul_out.reset_mock()
    module.tri_mul_in.reset_mock()
    module.tri_attn_start.reset_mock()
    module.tri_attn_end.reset_mock()
    module(m, z, msa_mask, pair_mask, precomputed_masks=pre_cutedsl)
    assert module.tri_mul_out.call_args.kwargs["actual_seqlen"] is None
    assert module.tri_mul_in.call_args.kwargs["actual_seqlen"] is None
    assert module.tri_attn_start.call_args.kwargs["mask_bias"] is None
    assert module.tri_attn_end.call_args.kwargs["mask_bias"] is None

    module.pair_mask_left_aligned = True
    module.tri_mul_out.reset_mock()
    module.tri_mul_in.reset_mock()
    mb_float = torch.zeros(B, sc.n_res, 1, 1, sc.n_res, dtype=torch_dtype, device=device)
    mb_float_t = torch.zeros(B, sc.n_res, 1, 1, sc.n_res, dtype=torch_dtype, device=device)
    pre_default = PrecomputedPairMasks(
        pair_mask=pair_mask,
        mask_bias=mb_float,
        mask_bias_transposed=mb_float_t,
    )
    module(m, z, msa_mask, pair_mask, precomputed_masks=pre_default)
    assert module.tri_mul_out.call_args.kwargs["actual_seqlen"] is None
    assert module.tri_mul_in.call_args.kwargs["actual_seqlen"] is None

    module.tri_mul_out.reset_mock()
    module.tri_mul_in.reset_mock()
    module(m, z, msa_mask, pair_mask, precomputed_masks=None)
    assert module.tri_mul_out.call_args.kwargs["actual_seqlen"] is None
    assert module.tri_mul_in.call_args.kwargs["actual_seqlen"] is None
