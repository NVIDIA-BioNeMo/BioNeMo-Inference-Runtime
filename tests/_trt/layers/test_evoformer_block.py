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
import tensorrt_llm_lite
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm_lite.functional import Tensor
from test_utils.openfold.create_and_load_weights import (
    create_evoformer_block_weights, load_evoformer_block_weights_trt)
from test_utils.openfold.ref_layers import RefEvoformerBlock

from tensorrt_bionemo._trt.layers.attention import AttentionParams
from tensorrt_bionemo._trt.layers.transformers import EvoformerBlock
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 64
    n_seq: int = 128
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV")
])
def test_evoformer_block(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    trt_dtype = str_dtype_to_trt(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefEvoformerBlock.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_evoformer_block_weights(from_ref=ref_module)
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
    msa_mask = torch.randint(0,
                             2, (bs, sc.n_seq, sc.n_res),
                             dtype=torch.float32).cuda()
    pair_mask = torch.randint(0,
                              2, (bs, sc.n_res, sc.n_res),
                              dtype=torch.float32).cuda()

    # construct trt network
    builder = tensorrt_llm_lite.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm_lite.net_guard(net):
        input_m = Tensor(name='input_m', shape=m.shape, dtype=trt_dtype)
        input_z = Tensor(name='input_z', shape=z.shape, dtype=trt_dtype)
        input_msa_mask = Tensor(name='input_msa_mask',
                                shape=msa_mask.shape,
                                dtype=trt_dtype)
        input_pair_mask = Tensor(name='input_pair_mask',
                                 shape=pair_mask.shape,
                                 dtype=trt_dtype)

        layer = EvoformerBlock(
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
            dtype=sc.dtype,
            eps=1e-5,
            inf=1e9,
            chunk_size=0,
            mapping=Mapping())

        load_evoformer_block_weights_trt(layer, weights_and_biases)

        output_m, output_z = layer(input_m,
                                   input_z,
                                   input_msa_mask,
                                   input_pair_mask,
                                   attention_params=AttentionParams())
        output_m.mark_output("output_m", trt_dtype)
        output_z.mark_output("output_z", trt_dtype)

    builder_config = builder.create_builder_config(name="evoformer_block",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    session = tensorrt_llm_lite.runtime.Session.from_serialized_engine(
        engine_buffer)

    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {
        'input_m': m,
        'input_z': z,
        'input_msa_mask': msa_mask,
        'input_pair_mask': pair_mask
    }
    outputs = {
        'output_m': torch.empty(m.shape, dtype=torch_dtype, device="cuda"),
        'output_z': torch.empty(z.shape, dtype=torch_dtype, device="cuda"),
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    with torch.inference_mode():
        ref_output_m, ref_output_z = ref_module(m, z, msa_mask, pair_mask)
    torch.cuda.synchronize()

    trt_output_m = outputs['output_m']
    trt_output_z = outputs['output_z']
    torch.testing.assert_close(trt_output_m,
                               ref_output_m,
                               atol=1e-3,
                               rtol=1e-4)
    torch.testing.assert_close(trt_output_z,
                               ref_output_z,
                               atol=1e-3,
                               rtol=1e-4)
