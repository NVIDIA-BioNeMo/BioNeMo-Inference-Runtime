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
import tensorrt as trt
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.plugin.plugin import (CustomAllReduceHelper,
                                        init_all_reduce_helper)
from test_utils.create_and_load_weights import (
    create_triangle_multiplication_node_weights,
    load_triangle_multiplication_node_weights_ref_torch,
    load_triangle_multiplication_node_weights_trt)
from test_utils.ref_layers import RefTriangleMultiplicationNode

from tensorrt_bionemo._trt.layers.triangle_nodes import (
    TriangleMultiplicationNode, TriangleMultiplicationNodeType)
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 64
    dtype: str = "float32"
    multiplication_type: TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.OUTGOING
    tp_size: int = 1
    dcp_size: int = 1
    seq_len: int = 32
    bs: int = 1


class TriangleMulNodesParallelism:

    def __init__(self, world_size: int, rank: int, dcp_size: int, tp_size: int,
                 bs: int, seq_len: int, dim: int, dtype: str,
                 multiplication_type: TriangleMultiplicationNodeType,
                 weights_and_biases: dict):
        self.world_size = world_size
        self.rank = rank
        self.dcp_size = dcp_size
        self.tp_size = tp_size
        self.bs = bs
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype
        self.torch_dtype = str_dtype_to_torch(self.dtype)
        self.multiplication_type = multiplication_type
        self.weights_and_biases = weights_and_biases
        self.mapping = Mapping(world_size=self.world_size,
                               rank=self.rank,
                               dcp_size=self.dcp_size,
                               tp_size=self.tp_size)
        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream()

    def build(self):
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        hidden_states_shape = [self.bs, self.seq_len, self.seq_len, self.dim]
        mask_shape = [self.bs, self.seq_len, self.seq_len]
        builder = Builder()
        builder = Builder()
        model_name = "triangle_nodes"

        builder_config = builder.create_builder_config(
            name=model_name,
            precision=self.dtype,
            timing_cache='model.cache',
            tensor_parallel=self.mapping.tp_size,
            strongly_typed=True,
            data_parallel=self.mapping.dcp_size)
        # Disable TF32 for accuracy in testing.
        builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
        network = builder.create_network()
        network.plugin_config.to_legacy_setting()
        network.plugin_config.set_nccl_plugin(self.dtype)
        init_all_reduce_helper()
        _, all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
            self.mapping,
            CustomAllReduceHelper.max_workspace_size_auto(self.mapping.tp_size))
        with tensorrt_llm.net_guard(network):
            trt_hidden_states = Tensor(name='input_x',
                                       shape=hidden_states_shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           self.dtype))
            trt_mask = Tensor(name='mask',
                              shape=mask_shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
            tri_mul_node = TriangleMultiplicationNode(
                local_layer_idx=0,
                dim=self.dim,
                dtype=self.dtype,
                support_batch=True,
                multiplication_type=self.multiplication_type,
                mapping=self.mapping)
            load_triangle_multiplication_node_weights_trt(
                tri_mul_node, self.weights_and_biases, self.mapping.tp_size,
                self.mapping.tp_rank)

            output = tri_mul_node(trt_hidden_states, trt_mask)
            output.mark_output("output",
                               tensorrt_llm.str_dtype_to_trt(self.dtype))

        engine_buffer = builder.build_engine(network, builder_config)
        assert engine_buffer is not None
        return engine_buffer

    def run(self, engine_buffer, inputs):
        # This import is needed to initialize plugin registry for each rank

        # Disable TF32 for accuracy in testing.
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        trt_inputs = {}
        for k, v in inputs.items():
            trt_inputs[k] = v.to(self.device, dtype=self.torch_dtype)
        outputs = {
            'output':
            torch.empty(trt_inputs['input_x'].shape,
                        dtype=self.torch_dtype,
                        device="cuda")
        }
        session.run(inputs=trt_inputs,
                    outputs=outputs,
                    stream=self.stream.cuda_stream)
        trt_output = outputs['output']
        torch.cuda.synchronize()

        ref_node = RefTriangleMultiplicationNode(
            self.dim,
            outgoing=self.multiplication_type ==
            TriangleMultiplicationNodeType.OUTGOING)
        ref_node.to("cuda", dtype=self.torch_dtype)
        load_triangle_multiplication_node_weights_ref_torch(
            ref_node, self.weights_and_biases)
        with torch.inference_mode():
            ref_output = ref_node(trt_inputs['input_x'], trt_inputs['mask'])
            torch.cuda.synchronize()
            torch.testing.assert_close(trt_output,
                                       ref_output,
                                       atol=1e-3,
                                       rtol=1e-4)


def run_single_rank(scenario: Scenario, inputs: dict, weights_and_biases: dict):
    import tensorrt_bionemo  # init plugin registry for each rank
    rank = tensorrt_llm.mpi_rank()
    module = TriangleMulNodesParallelism(
        world_size=scenario.tp_size * scenario.dcp_size,
        rank=rank,
        dcp_size=scenario.dcp_size,
        tp_size=scenario.tp_size,
        bs=scenario.bs,
        seq_len=scenario.seq_len,
        dim=scenario.dim,
        dtype=scenario.dtype,
        multiplication_type=scenario.multiplication_type,
        weights_and_biases=weights_and_biases)
    try:
        engine_buffer = module.build()
        module.run(engine_buffer, inputs)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _generate_scenarios():
    max_world_size = torch.cuda.device_count()
    scenarios = []
    ids = []
    for multiplication_type in [
            TriangleMultiplicationNodeType.OUTGOING,
            TriangleMultiplicationNodeType.INCOMING
    ]:
        for tp_size, dcp_size in product([1, 2, 4, 8], repeat=2):
            if tp_size * dcp_size > max_world_size:
                continue
            scenarios.append(
                Scenario(tp_size=tp_size,
                         dcp_size=dcp_size,
                         multiplication_type=multiplication_type))
            ids.append(f"{multiplication_type.name}_{tp_size}_{dcp_size}")

    return scenarios, ids


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_scenarios()[0],
                         ids=_generate_scenarios()[1])
def test_triangle_mul_nodes_parallelism(scenario: Scenario):
    torch.manual_seed(42)
    x = torch.randn(scenario.bs, scenario.seq_len, scenario.seq_len,
                    scenario.dim)
    mask = torch.randn(scenario.bs, scenario.seq_len, scenario.seq_len)
    world_size = scenario.tp_size * scenario.dcp_size
    torch_dtype = str_dtype_to_torch(scenario.dtype)
    inputs = {'input_x': x, 'mask': mask}
    weights_and_biases = create_triangle_multiplication_node_weights(
        dim=scenario.dim, torch_dtype=torch_dtype)

    # Run inference for each rank
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(scenario, inputs, weights_and_biases)] * world_size))
        for r in results:
            assert r is True
