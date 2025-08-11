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
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_pair_weighted_averaging_weights,
    load_pair_weighted_averaging_weights_torch)
from test_utils.boltz.ref_layers import RefPairWeightedAveraging

from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
])
def test_pair_weighted_averaging(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_m = RefPairWeightedAveraging.load_weights()
    ref_m = ref_m.to(device)

    weights_and_biases = create_pair_weighted_averaging_weights(from_ref=ref_m)

    pair_weighted_averaging = PairWeightedAveraging(c_m=ref_m.c_m,
                                                    c_z=ref_m.c_z,
                                                    c_h=ref_m.c_h,
                                                    num_heads=ref_m.num_heads,
                                                    dtype=dtype)
    load_pair_weighted_averaging_weights_torch(pair_weighted_averaging,
                                               weights_and_biases,
                                               dtype=dtype)
    pair_weighted_averaging.to(device)

    m = torch.randn(bs, 32, 64, ref_m.c_m, dtype=torch.float32).cuda()
    z = torch.randn(bs, 64, 64, ref_m.c_z, dtype=torch.float32).cuda()
    mask = torch.randn(bs, 64, 64, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_m(m, z, mask)
        m = m.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_m = ref_m.to(dtype)

        ref_output = ref_m(m, z, mask)
        output = pair_weighted_averaging.forward(m, z, mask)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2
