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

# isort: off
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.functional import Tensor
from test_utils.boltz.create_and_load_weights import (
    create_affinity_module_weights, load_affinity_module_weights_trt)
from test_utils.boltz.ref_layers import RefAffinityModule
# isort: on

from tensorrt_bionemo._trt.layers.affinity import AffinityModule
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz2.configs import AffinityModuleConfig


@dataclass(kw_only=True, frozen=True)
class Scenario:
    seq_len: int = 32
    dtype: str = "float32"
    num_dist_bins: int = 64
    token_z: int = 128
    token_s: int = 384


@pytest.mark.parametrize("sc", [Scenario(), Scenario(seq_len=256)])
def test_affinity_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    trt_dtype = str_dtype_to_trt(sc.dtype)
    trt_int32_dtype = str_dtype_to_trt("int32")
    device = torch.device('cuda')

    s = torch.randn(bs, sc.seq_len, sc.token_s, dtype=torch.float32).to(device)
    z = torch.randn(bs, sc.seq_len, sc.seq_len, sc.token_z,
                    dtype=torch.float32).to(device)
    distogram = torch.randint(0,
                              sc.num_dist_bins, (bs, sc.seq_len, sc.seq_len),
                              dtype=torch.int32).to(device)
    cross_pair_mask_0 = torch.randint(0,
                                      2, (bs, sc.seq_len, sc.seq_len),
                                      dtype=torch.float32).to(device)
    cross_pair_mask_1 = torch.randint(0,
                                      2, (bs, sc.seq_len, sc.seq_len, 1),
                                      dtype=torch.float32).to(device)

    ref_module = RefAffinityModule.load_weights().to(device)
    ref_module.eval()

    weights_and_biases = create_affinity_module_weights(from_ref=ref_module)

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        input_s = Tensor(name='input_s', shape=s.shape, dtype=trt_dtype)
        input_z = Tensor(name='input_z', shape=z.shape, dtype=trt_dtype)
        input_distogram = Tensor(name='input_distogram',
                                 shape=distogram.shape,
                                 dtype=trt_int32_dtype)
        input_cross_pair_mask_0 = Tensor(name='input_cross_pair_mask_0',
                                         shape=cross_pair_mask_0.shape,
                                         dtype=trt_dtype)
        input_cross_pair_mask_1 = Tensor(name='input_cross_pair_mask_1',
                                         shape=cross_pair_mask_1.shape,
                                         dtype=trt_dtype)

        config = AffinityModuleConfig(
            token_s=sc.token_s,
            token_z=sc.token_z,
            num_dist_bins=sc.num_dist_bins,
            pairformer_num_blocks=ref_module.pairformer_num_blocks,
            pairwise_head_width=ref_module.pairwise_head_width,
            pairwise_num_heads=ref_module.pairwise_num_heads,
            architecture="affinity_module",
            dtype=sc.dtype)
        layer = AffinityModule(config)
        load_affinity_module_weights_trt(layer, weights_and_biases, Mapping())
        pred_value, logits_binary = layer(input_s, input_z, input_distogram,
                                          input_cross_pair_mask_0,
                                          input_cross_pair_mask_1)

        pred_value.mark_output("output_pred_value", trt_dtype)
        logits_binary.mark_output("output_logits_binary", trt_dtype)

    builder_config = builder.create_builder_config(name="affinity_module",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)

    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {
        'input_s': s,
        'input_z': z,
        'input_distogram': distogram,
        'input_cross_pair_mask_0': cross_pair_mask_0,
        'input_cross_pair_mask_1': cross_pair_mask_1,
    }
    outputs = {
        'output_pred_value':
        torch.empty(tuple(pred_value.shape), dtype=torch_dtype, device="cuda"),
        'output_logits_binary':
        torch.empty(tuple(logits_binary.shape),
                    dtype=torch_dtype,
                    device="cuda"),
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    with torch.inference_mode():
        ref_pred_value, ref_logits_binary = ref_module(s, z, distogram,
                                                       cross_pair_mask_0,
                                                       cross_pair_mask_1)
        torch.cuda.synchronize()

    trt_pred_value = outputs['output_pred_value']
    trt_logits_binary = outputs['output_logits_binary']

    torch.testing.assert_close(trt_pred_value,
                               ref_pred_value,
                               atol=1e-3,
                               rtol=1e-4)
    torch.testing.assert_close(trt_logits_binary,
                               ref_logits_binary,
                               atol=1e-3,
                               rtol=1e-4)
