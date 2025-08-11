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
import tensorrt_llm
import torch
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_outer_product_mean_weights, load_outer_product_mean_weights_trt)
from test_utils.boltz.ref_layers import \
    RefOuterProductMean as BoltzRefOuterProductMean
from test_utils.openfold.ref_layers import \
    RefOuterProductMean as OpenFoldRefOuterProductMean

from tensorrt_bionemo._trt.layers.outer_product_mean import OuterProductMean
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: str = "float32"
    bs: int = 1
    si: int = 64
    sj: int = 32
    mode: "boltz"  # "boltz" or "openfold"


@pytest.mark.parametrize("sc", [
    Scenario(mode="boltz"),
    Scenario(mode="openfold"),
],
                         ids=["boltz", "openfold"])
def test_outer_product_mean(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    if sc.mode == "boltz":
        ref_omp = BoltzRefOuterProductMean.load_weights()
    else:
        ref_omp = OpenFoldRefOuterProductMean.load_weights()

    m = torch.randn(sc.bs, sc.si, sc.sj, ref_omp.c_in,
                    dtype=torch.float32).cuda()
    mask = torch.randint(0, 2, (sc.bs, sc.si, sc.sj),
                         dtype=torch.float32).cuda()

    if sc.mode == "boltz":
        ref_omp = BoltzRefOuterProductMean.load_weights()
    else:
        ref_omp = OpenFoldRefOuterProductMean.load_weights()
    ref_omp = ref_omp.to(device)

    weights_and_biases = create_outer_product_mean_weights(from_ref=ref_omp)

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        trt_m = Tensor(name='input_m',
                       shape=m.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_mask = Tensor(name='input_mask',
                          shape=mask.shape,
                          dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

        if sc.mode == "boltz":
            bias_flags = {"proj_a": False, "proj_b": False, "proj_o": True}
            norm_before_output = True
        else:
            bias_flags = {"proj_a": True, "proj_b": True, "proj_o": True}
            norm_before_output = False

        omp = OuterProductMean(c_in=ref_omp.c_in,
                               c_hidden=ref_omp.c_hidden,
                               c_out=ref_omp.c_out,
                               bias_flags=bias_flags,
                               norm_before_output=norm_before_output,
                               dtype=sc.dtype)
        load_outer_product_mean_weights_trt(omp,
                                            weights_and_biases,
                                            mapping=Mapping())
        output = omp(trt_m, trt_mask)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))

    builder_config = builder.create_builder_config(name="outer_product_mean",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify results
    inputs = {'input_m': m.to(torch_dtype), 'input_mask': mask.to(torch_dtype)}
    outputs = {
        'output':
        torch.empty((sc.bs, sc.sj, sc.sj, ref_omp.c_out),
                    dtype=torch_dtype,
                    device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)

    ref_output = ref_omp(m.to(torch_dtype), mask.to(torch_dtype))
    torch.cuda.synchronize()

    if sc.dtype == "float32":
        torch.testing.assert_close(outputs['output'],
                                   ref_output,
                                   atol=1e-3,
                                   rtol=1e-3)
