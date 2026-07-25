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

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_single_conditioning_weights, load_single_conditioning_weights_torch)
from test_utils.boltz.ref_layers import RefSingleConditioning

from tensorrt_bionemo._torch.layers.conditioning import SingleConditioning
from tensorrt_bionemo.utils import str_dtype_to_torch


@dataclass(kw_only=True, frozen=True)
class Scenario:
    seq_len: int = 928
    dtype: str = "float32"
    batch_size: int = 1


@pytest.mark.parametrize(
    "sc", [Scenario(dtype="float32"),
           Scenario(dtype="float16", seq_len=1024)])
def test_single_conditioning(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = sc.batch_size
    device = torch.device('cuda')
    ref_module = RefSingleConditioning.load_weights().to(device)
    ref_module.eval()

    weights_and_biases = create_single_conditioning_weights(
        from_ref=ref_module)
    token_s = ref_module.token_s
    dim_fourier = ref_module.dim_fourier
    num_transitions = len(ref_module.transitions)
    times = torch.randn(bs, dtype=torch.float32).cuda()
    s_trunk = torch.randn(bs, sc.seq_len, token_s, dtype=torch.float32).cuda()
    s_inputs = torch.randn(bs, sc.seq_len, token_s, dtype=torch.float32).cuda()

    dtype = str_dtype_to_torch(sc.dtype)

    model = SingleConditioning(token_s=token_s,
                               dim_fourier=dim_fourier,
                               num_transitions=num_transitions).to(device)
    load_single_conditioning_weights_torch(model, weights_and_biases)
    model.eval()
    model = model.to(dtype)

    with torch.inference_mode():
        ref_output_float, _ = ref_module(times, s_trunk, s_inputs)
        times = times.to(dtype)
        s_trunk = s_trunk.to(dtype)
        s_inputs = s_inputs.to(dtype)

        output, _ = model(times, s_trunk,
                          s_inputs)  # squeeze out multiplicity dimension
        output = output.squeeze(1)

        ref_module = ref_module.to(dtype)
        ref_output, _ = ref_module(times, s_trunk, s_inputs)

        if dtype == torch.float32:
            torch.testing.assert_close(output,
                                       ref_output_float,
                                       atol=5e-2,
                                       rtol=1e-2)
        else:
            # This is right way to check float16 and bfloat16 accuracy
            diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
            diff0_mean = torch.mean(
                torch.abs(output.float() - ref_output_float))
            diff1_max = torch.max(
                torch.abs(ref_output.float() - ref_output_float))
            diff1_mean = torch.mean(
                torch.abs(ref_output.float() - ref_output_float))

            assert abs(diff0_max - diff1_max) / torch.min(
                diff0_max, diff1_max) <= 0.5
            assert abs(diff0_mean - diff1_mean) <= 0.2
