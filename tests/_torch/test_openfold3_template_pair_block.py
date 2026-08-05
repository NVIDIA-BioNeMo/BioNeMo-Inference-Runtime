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
from test_utils.openfold3.ref_layers_from_oss import RefTemplatePairBlockFromOF3OSS

from tensorrt_bionemo._torch.modules.openfold2.template import TemplatePairBlock
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl
from tests.common.test_utils.openfold3.create_and_load_weights_from_of3oss import (
    create_template_pair_block_weights_from_of3oss_torch,
    load_template_pair_block_weights_from_of3oss_torch,
)


def _assert_pipeline_style_left_aligned_pair_mask(mask_bt_nn: torch.Tensor) -> None:
    """Assert ``mask`` matches the structure produced for OF3 trunk templates.

    In production, ``pair_mask = token_mask[..., None] * token_mask[..., None, :]``
    with a collator-left-aligned ``token_mask`` (valid tokens then padding),
    then broadcast over templates as ``pair_mask[..., None, :, :]``
    (see ``tensorrt_bionemo/models/openfold3/modeling.py`` and
    ``tensorrt_bionemo/_torch/modules/openfold3/embedders.py`` —
    ``TemplateEmbedderAllAtom.forward``).  Such a mask is **non-increasing**
    along each residue axis (leading ones, then zeros).
    """
    m = (mask_bt_nn > 0.5).float()
    assert bool((m[..., :-1] >= m[..., 1:]).all()), "expected left-aligned columns (111...000) per pair row"
    mt = m.transpose(-1, -2).contiguous()
    assert bool((mt[..., :-1] >= mt[..., 1:]).all()), "expected left-aligned rows (111...000) per pair column"


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 63
    n_templ: int = 31
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(triangle_attn_backend="VANILLA"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="VANILLA", dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV", dtype="bfloat16"),
        Scenario(triangle_attn_backend="CuTeDSL", dtype="bfloat16"),
    ],
    ids=[
        "vanilla",
        "cuequiv",
        "vanilla_bf16",
        "cuequiv_bf16",
        "cutedsl_bf16",
    ],
)
def test_template_pair_stack_block(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    test_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    c_in = 64

    # (1) Input:
    #   - call RNG early in code block since impl of modules may change
    #   - input t, mask have float32 values.
    t_float = torch.randn(bs, sc.n_templ, sc.n_res, sc.n_res, c_in, dtype=torch.float32).cuda() * 20.0
    # Same layout as OF3 trunk: ``pair_mask[b,i,j] = seq[b,i]*seq[b,j]`` with a
    # left-aligned per-batch ``seq``, replicated across templates (embedder
    # uses ``pair_mask[..., None, :, :]``).
    seq_mask = make_left_aligned_mask(bs, sc.n_res, dtype=torch.float32, device=device, min_valid=max(sc.n_res - 4, 1))
    pair_mask_bn = seq_mask[..., None] * seq_mask[..., None, :]
    mask_float = pair_mask_bn[:, None, :, :].expand(bs, sc.n_templ, sc.n_res, sc.n_res).contiguous()
    _assert_pipeline_style_left_aligned_pair_mask(mask_float)

    # (2) Reference module creation
    #   - ref module stays in float32
    ref_module = RefTemplatePairBlockFromOF3OSS.load_weights()
    ref_module = ref_module.to(device)
    assert ref_module.tri_mul_out.c_z == c_in

    tri_mul_keys = ["p_in", "g_in", "p_out", "g_out"]
    tri_attn_keys = ["q", "k", "v", "g", "z", "o"]

    tri_mul_out_bias = dict.fromkeys(tri_mul_keys, False)
    tri_mul_in_bias = dict.fromkeys(tri_mul_keys, False)
    tri_attn_start_bias = dict.fromkeys(tri_attn_keys, False)
    tri_attn_end_bias = dict.fromkeys(tri_attn_keys, False)

    # (3) Test module creation
    #   - module weights and buffers changed to test_dtype
    module = TemplatePairBlock(
        local_layer_idx=0,
        c_t=ref_module.tri_mul_out.c_z,
        c_hidden_tri_att=ref_module.tri_att_start.c_hidden,
        c_hidden_tri_mul=ref_module.tri_mul_out.c_hidden,
        no_heads=ref_module.tri_att_start.no_heads,
        pair_transition_n=ref_module.pair_transition.n,
        tri_mul_first=ref_module.tri_mul_first,
        transition_type="swiglu",
        triangle_attn_backend=sc.triangle_attn_backend,
        tri_mul_out_bias=tri_mul_out_bias,
        tri_mul_in_bias=tri_mul_in_bias,
        tri_attn_start_bias=tri_attn_start_bias,
        tri_attn_end_bias=tri_attn_end_bias,
        dtype=test_dtype,
    )

    weights_and_biases = create_template_pair_block_weights_from_of3oss_torch(from_ref=ref_module)
    load_template_pair_block_weights_from_of3oss_torch(module, weights_and_biases, test_dtype)
    module = module.to(device)
    module.to(test_dtype).eval()
    ref_module.eval()

    # (4) Forward
    with torch.no_grad():
        t_float = t_float.flatten(0, 1)  # fused batch_dim and template_dim
        mask_float = mask_float.flatten(0, 1)  # fused batch_dim and template_dim
        z_keep = mask_float.unsqueeze(-1).float()

        def _masked(x: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0) * z_keep

        t_test_dtype = t_float.to(test_dtype)
        mask_test_dtype = mask_float.to(test_dtype)

        output_test_dtype = module(t_test_dtype, mask_test_dtype)
        ref_output_float = ref_module(t_float, mask_float)

        if test_dtype == torch.float32:
            torch.testing.assert_close(_masked(output_test_dtype), _masked(ref_output_float), atol=1e-3, rtol=1e-4)
            print(f"after test for test_dtype={test_dtype}")

        else:
            with torch.amp.autocast(device_type="cuda", dtype=test_dtype):
                ref_output = ref_module(t_float.to(test_dtype), mask_float.to(test_dtype))

            diff0_max = torch.max(torch.abs(_masked(output_test_dtype) - _masked(ref_output_float)))
            diff0_mean = torch.mean(torch.abs(_masked(output_test_dtype) - _masked(ref_output_float)))

            diff1_max = torch.max(torch.abs(_masked(ref_output) - _masked(ref_output_float)))
            diff1_mean = torch.mean(torch.abs(_masked(ref_output) - _masked(ref_output_float)))

            if sc.triangle_attn_backend == "CuTeDSL":
                assert diff0_max <= 2.0 * diff1_max + 1e-3, (
                    f"CuTeDSL bf16 drift vs fp32 ref: diff_ours={diff0_max}, diff_autocast_ref={diff1_max}"
                )
                assert diff0_mean <= 2.0 * diff1_mean + 5e-2, (
                    f"CuTeDSL bf16 mean drift: diff_ours={diff0_mean}, diff_autocast_ref={diff1_mean}"
                )
            else:
                assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
                assert abs(diff0_mean - diff1_mean) <= 0.2
            print(f"after test for test_dtype={test_dtype}")
