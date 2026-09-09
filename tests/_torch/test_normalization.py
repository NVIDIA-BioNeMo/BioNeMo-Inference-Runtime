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
from test_utils.boltz.create_and_load_weights import create_adaln_weights, load_adaln_weights_torch
from test_utils.boltz.ref_layers import RefAdaLN

from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir.utils import str_dtype_to_torch


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    torch_dtype: str = "float32"
    seq_len: int = 128


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim=768, dim_single_cond=768),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
    ],
)
def test_adaln(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_adaln = RefAdaLN.load_weights()
    ref_adaln = ref_adaln.to(device)

    weights_and_biases = create_adaln_weights(from_ref=ref_adaln)

    adaln = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=dtype)
    load_adaln_weights_torch(adaln, weights_and_biases, dtype=dtype)
    adaln.to(device)

    a = torch.randn(bs, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_adaln(a, s)
        a = a.to(dtype)
        s = s.to(dtype)
        ref_adaln = ref_adaln.to(dtype)

        ref_output = ref_adaln(a, s)
        output = adaln.forward(a, s)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


def test_adaln_rms_norm_matches_torch_formula():
    torch.manual_seed(3)
    device = torch.device("cuda")
    dim, dim_cond, seq_len = 32, 16, 11
    adaln = AdaLN(dim, dim_cond, dtype=torch.float32, norm_type="rms_norm").to(device)
    assert callable(adaln._fused_op)
    a = torch.randn(2, seq_len, dim, device=device)
    s = torch.randn(2, seq_len, dim_cond, device=device)
    mask = torch.ones(2, seq_len, device=device, dtype=torch.bool)
    mask[1, 7:] = False

    with torch.inference_mode():
        out = adaln(a, s, mask=mask)
        scale_bias = adaln.fused_s_scale_s_bias(adaln.s_norm(s))
        scale, bias = scale_bias.split([dim, dim], dim=-1)
        expected = torch.sigmoid(scale) * adaln.a_norm(a) + bias
        expected = expected * mask.unsqueeze(-1)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(out[1, 7:]) == 0


def test_adaln_mask_broadcasts_across_samples():
    """A rank-2 token mask must apply to every sample of a rank-4 AdaLN input."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dim, dim_cond, batch, samples, tokens = 8, 8, 2, 3, 5
    adaln = AdaLN(dim, dim_cond, dtype=torch.float32).to(device)
    adaln._fused_op = None
    a = torch.randn(batch, samples, tokens, dim, device=device)
    s = torch.randn(batch, 1, tokens, dim_cond, device=device)
    mask = torch.ones(batch, tokens, device=device)
    mask[:, 3:] = 0

    with torch.inference_mode():
        out = adaln(a, s, mask=mask)
        unmasked = adaln(a, s)

    assert out.shape == a.shape
    assert torch.count_nonzero(out[:, :, 3:]) == 0
    torch.testing.assert_close(out[:, :, :3], unmasked[:, :, :3])

    x3 = torch.ones(batch, tokens, dim, device=device)
    masked3 = AdaLN._maybe_mask(x3, mask)
    assert masked3.shape == x3.shape
    assert torch.count_nonzero(masked3[:, 3:]) == 0
