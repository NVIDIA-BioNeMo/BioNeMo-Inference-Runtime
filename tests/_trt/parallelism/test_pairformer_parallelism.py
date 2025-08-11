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
import traceback
from dataclasses import dataclass
from itertools import product

import pytest
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.plugin.plugin import (CustomAllReduceHelper,
                                        init_all_reduce_helper)
from test_utils.boltz.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_ref_torch,
    load_pairformer_layer_weights_trt)
from test_utils.boltz.ref_layers import RefPairformerLayer

from tensorrt_bionemo._trt.layers.attention import AttentionParams
from tensorrt_bionemo._trt.layers.transformers import PairformerLayerV1
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    seq_len: int = 128
    token_s: int = 32
    token_z: int = 128
    num_heads: int = 16
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    vanilla_attn_precision: str = "float32"
    dtype: str = "float32"
    max_attention_pairwise_tp_size: bool = True
    max_transition_tp_size: bool = True
    max_tri_mul_tp_size: bool = True
    chunk_size: int = 0
    eps: float = 1e-5
    tp_size: int = 1
    dcp_size: int = 1
    bs: int = 1


class PairformerParallelism:

    def __init__(self,
                 world_size: int,
                 rank: int,
                 dcp_size: int,
                 tp_size: int,
                 bs: int,
                 seq_len: int,
                 token_s: int,
                 token_z: int,
                 num_heads: int,
                 pairwise_head_width: int,
                 pairwise_num_heads: int,
                 max_attention_pairwise_tp_size: bool = True,
                 max_transition_tp_size: bool = True,
                 max_tri_mul_tp_size: bool = True,
                 vanilla_attn_precision: str = "float32",
                 dtype: str = "float32",
                 weights_and_biases: dict = None):
        self.world_size = world_size
        self.rank = rank
        self.dcp_size = dcp_size
        self.tp_size = tp_size
        self.bs = bs
        self.seq_len = seq_len
        self.token_s = token_s
        self.token_z = token_z
        self.dtype = dtype
        self.num_heads = num_heads
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.vanilla_attn_precision = vanilla_attn_precision
        self.max_attention_pairwise_tp_size = max_attention_pairwise_tp_size
        self.max_transition_tp_size = max_transition_tp_size
        self.max_tri_mul_tp_size = max_tri_mul_tp_size
        self.mapping = Mapping(world_size=self.world_size,
                               rank=self.rank,
                               dcp_size=self.dcp_size,
                               tp_size=self.tp_size)
        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream()

        self.torch_dtype = str_dtype_to_torch(self.dtype)
        self.weights_and_biases = weights_and_biases

    def build(self):
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        s_shape = [self.bs, self.seq_len, self.token_s]
        z_shape = [self.bs, self.seq_len, self.seq_len, self.token_z]
        mask_shape = [self.bs, self.seq_len]
        pairmask_shape = [self.bs, self.seq_len, self.seq_len]

        builder = Builder()
        model_name = "pairformer"

        builder_config = builder.create_builder_config(
            name=model_name,
            precision=self.dtype,
            timing_cache='model.cache',
            tensor_parallel=self.mapping.tp_size,
            strongly_typed=True,
            data_parallel=self.mapping.dcp_size)
        # Disable TF32 for accuracy in testing.
        # builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
        network = builder.create_network()
        network.plugin_config.to_legacy_setting()
        network.plugin_config.set_nccl_plugin(self.dtype)
        init_all_reduce_helper()
        _, all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
            self.mapping,
            CustomAllReduceHelper.max_workspace_size_auto(self.mapping.tp_size))

        with tensorrt_llm.net_guard(network):
            trt_s = Tensor(name='s',
                           shape=s_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
            trt_z = Tensor(name='z',
                           shape=z_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
            trt_mask = Tensor(name='mask',
                              shape=mask_shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
            trt_pairmask = Tensor(name='pairmask',
                                  shape=pairmask_shape,
                                  dtype=tensorrt_llm.str_dtype_to_trt(
                                      self.dtype))
            pairformer_layer = PairformerLayerV1(
                local_layer_idx=0,
                token_s=self.token_s,
                token_z=self.token_z,
                num_heads=self.num_heads,
                pairwise_head_width=self.pairwise_head_width,
                pairwise_num_heads=self.pairwise_num_heads,
                dtype=self.dtype,
                mapping=self.mapping,
                max_attention_pairwise_tp_size=self.
                max_attention_pairwise_tp_size,
                max_transition_tp_size=self.max_transition_tp_size,
                max_tri_mul_tp_size=self.max_tri_mul_tp_size)
            load_pairformer_layer_weights_trt(
                pairformer_layer, self.weights_and_biases, self.mapping,
                self.num_heads, self.token_s, self.token_z,
                self.max_attention_pairwise_tp_size,
                self.max_transition_tp_size, self.max_tri_mul_tp_size)
            attention_params = AttentionParams()
            output_s, output_z = pairformer_layer(
                trt_s,
                trt_z,
                trt_mask,
                trt_pairmask,
                attention_params=attention_params)
            output_s.mark_output("output_s",
                                 tensorrt_llm.str_dtype_to_trt(self.dtype))
            output_z.mark_output("output_z",
                                 tensorrt_llm.str_dtype_to_trt(self.dtype))
        engine_buffer = builder.build_engine(network, builder_config)
        assert engine_buffer is not None
        return engine_buffer

    def run(self, engine_buffer: str, inputs: dict):
        # This import is needed to initialize plugin registry for each rank
        pass

        # Disable TF32 for accuracy in testing.
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        trt_inputs = {}
        for k, v in inputs.items():
            trt_inputs[k] = v.to(self.device, dtype=self.torch_dtype)
        outputs = {
            'output_s':
            torch.empty(trt_inputs['s'].shape,
                        dtype=self.torch_dtype,
                        device="cuda"),
            'output_z':
            torch.empty(trt_inputs['z'].shape,
                        dtype=self.torch_dtype,
                        device="cuda")
        }
        session.run(inputs=trt_inputs,
                    outputs=outputs,
                    stream=self.stream.cuda_stream)
        trt_output_s = outputs['output_s']
        trt_output_z = outputs['output_z']
        torch.cuda.synchronize()

        ref_pairformer_layer = RefPairformerLayer(self.token_s,
                                                  self.token_z,
                                                  self.num_heads,
                                                  self.pairwise_head_width,
                                                  self.pairwise_num_heads,
                                                  no_update_s=False,
                                                  no_update_z=False)
        load_pairformer_layer_weights_ref_torch(ref_pairformer_layer,
                                                self.weights_and_biases)
        ref_pairformer_layer.to("cuda", dtype=self.torch_dtype)
        with torch.inference_mode():
            ref_output_s, ref_output_z = ref_pairformer_layer(
                trt_inputs['s'], trt_inputs['z'], trt_inputs['mask'],
                trt_inputs['pairmask'])
            torch.cuda.synchronize()
        torch.testing.assert_close(trt_output_s,
                                   ref_output_s,
                                   atol=1e-3,
                                   rtol=1e-4)
        torch.testing.assert_close(trt_output_z,
                                   ref_output_z,
                                   atol=1e-3,
                                   rtol=1e-4)


def run_single_rank(scenario: Scenario, inputs: dict, weights_and_biases: dict):
    rank = tensorrt_llm.mpi_rank()
    module = PairformerParallelism(
        world_size=scenario.tp_size * scenario.dcp_size,
        rank=rank,
        dcp_size=scenario.dcp_size,
        tp_size=scenario.tp_size,
        bs=scenario.bs,
        seq_len=scenario.seq_len,
        token_s=scenario.token_s,
        token_z=scenario.token_z,
        num_heads=scenario.num_heads,
        pairwise_head_width=scenario.pairwise_head_width,
        pairwise_num_heads=scenario.pairwise_num_heads,
        vanilla_attn_precision=scenario.vanilla_attn_precision,
        dtype=scenario.dtype,
        max_attention_pairwise_tp_size=scenario.max_attention_pairwise_tp_size,
        max_transition_tp_size=scenario.max_transition_tp_size,
        weights_and_biases=weights_and_biases)
    try:
        engine_buffer = module.build()
        module.run(engine_buffer, inputs)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _generate_scenarios():
    ret = []
    ids = []
    max_world_size = torch.cuda.device_count()
    for max_attention_pairwise_tp_size, max_transition_tp_size in product(
        [True, False], repeat=2):
        for tp_size, dcp_size in product([1, 2, 4], repeat=2):
            if tp_size * dcp_size > max_world_size:
                continue
            if tp_size > 4:
                continue
            ret.append(
                Scenario(tp_size=tp_size,
                         dcp_size=dcp_size,
                         max_attention_pairwise_tp_size=
                         max_attention_pairwise_tp_size,
                         max_transition_tp_size=max_transition_tp_size))
            ids.append(
                f"tp_{tp_size}-dcp_{dcp_size}-{max_attention_pairwise_tp_size}-{max_transition_tp_size}"
            )
    return ret, ids


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_scenarios()[0],
                         ids=_generate_scenarios()[1])
def test_pairformer_parallelism(scenario: Scenario):
    torch.manual_seed(42)
    bs = scenario.bs
    s = torch.randn(bs, scenario.seq_len, scenario.token_s)
    z = torch.randn(bs, scenario.seq_len, scenario.seq_len, scenario.token_z)
    mask = torch.randn(bs, scenario.seq_len)
    pairmask = torch.randn(bs, scenario.seq_len, scenario.seq_len)
    inputs = {"s": s, "z": z, "mask": mask, "pairmask": pairmask}
    weights_and_biases = create_pairformer_layer_weights(
        scenario.token_s, scenario.token_z, scenario.num_heads,
        scenario.pairwise_head_width, scenario.pairwise_num_heads,
        torch.float32)
    world_size = scenario.tp_size * scenario.dcp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(scenario, inputs, weights_and_biases)] * world_size))
        for r in results:
            assert r is True
