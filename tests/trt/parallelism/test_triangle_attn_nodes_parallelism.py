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
import tempfile
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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
    create_triangle_attention_node_weights,
    load_triangle_attention_node_weights_ref_torch,
    load_triangle_attention_node_weights_trt)
from test_utils.ref_layers import RefTriangleAttentionNode

from tensorrt_bionemo.layers.attention import AttentionParams
from tensorrt_bionemo.layers.triangle_nodes import (TriangleAttentionNode,
                                                    TriangleAttentionNodeType)
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    c_in: int = 32
    c_hidden: int = 8
    num_heads: int = 4
    chunk_size: int = 0
    dtype: str = "float32"
    node_type: str = TriangleAttentionNodeType.STARTING
    tp_size: int = 1
    dcp_size: int = 1
    seq_len: int = 128
    n_optimization_profiles: int = 0


class TriangleAttnNodesParallelism:

    def __init__(
            self,
            world_size: int,
            rank: int,
            dcp_size: int,
            tp_size: int,
            seq_len: int,
            c_in: int,
            c_hidden: int,
            num_attention_heads: int,
            plain_attn_precision: str = "float32",
            node_type: TriangleAttentionNodeType = TriangleAttentionNodeType.
        STARTING,
            dtype: str = "float32",
            n_optimization_profiles: int = 0,
            inputs: dict = None,
            weights_and_biases: dict = None,
            temp_dir: Path = None,
            building: bool = False):
        tensorrt_llm.logger.set_level('info')
        self.mapping = Mapping(world_size=world_size,
                               rank=rank,
                               dcp_size=dcp_size,
                               tp_size=tp_size,
                               pp_size=1)
        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream()

        self.seq_len = seq_len
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_attention_heads = num_attention_heads
        self.plain_attn_precision = plain_attn_precision
        self.node_type = node_type
        self.dtype = dtype

        self.torch_dtype = str_dtype_to_torch(self.dtype)
        self.weights_and_biases = weights_and_biases

        self.n_optimization_profiles = n_optimization_profiles
        assert self.n_optimization_profiles <= 1, "Only 1 profiles are supported"

        self.inputs = inputs
        self.temp_dir = temp_dir
        self.building = building

        if self.mapping.tp_size > 1 and not self.building:
            self.ipc_buffers, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))

    def run(self, engine_path: str):
        import os

        # This import is needed to initialize plugin registry for each rank
        import tensorrt_llm

        # Disable TF32 for accuracy in testing.
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        inputs = {}
        for k, v in self.inputs.items():
            inputs[k] = v.to(self.device, dtype=self.torch_dtype)
        outputs = {
            'output':
            torch.empty(self.inputs['input_s'].shape,
                        dtype=self.torch_dtype,
                        device="cuda")
        }
        session.run(inputs=inputs,
                    outputs=outputs,
                    stream=self.stream.cuda_stream)
        trt_output = outputs['output']
        torch.cuda.synchronize()

        starting = self.node_type == TriangleAttentionNodeType.STARTING
        ref_node = RefTriangleAttentionNode(self.c_in, self.c_hidden,
                                            self.num_attention_heads, starting)
        ref_node.to("cuda", dtype=self.torch_dtype)
        load_triangle_attention_node_weights_ref_torch(ref_node,
                                                       self.weights_and_biases)
        with torch.inference_mode():
            ref_output = ref_node(inputs['input_s'], inputs['mask'])
            torch.cuda.synchronize()
            torch.testing.assert_close(trt_output,
                                       ref_output,
                                       atol=1e-3,
                                       rtol=1e-4)
        return True

    def build(self):
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        assert self.c_in == self.c_hidden * self.num_attention_heads

        hidden_states_shape = [self.seq_len, self.seq_len, self.c_in]
        mask_shape = [self.seq_len, self.seq_len]
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
        with tensorrt_llm.net_guard(network):
            trt_hidden_states = Tensor(name='input_s',
                                       shape=hidden_states_shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           self.dtype))
            trt_mask = Tensor(name='mask',
                              shape=mask_shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))
            tri_attn_node = TriangleAttentionNode(
                c_in=self.c_in,
                c_hidden=self.c_hidden,
                num_heads=self.num_attention_heads,
                local_layer_idx=0,
                dtype=self.dtype,
                chunk_size=0,
                node_type=self.node_type,
                mapping=self.mapping)
            load_triangle_attention_node_weights_trt(tri_attn_node,
                                                     self.weights_and_biases,
                                                     self.mapping.tp_size,
                                                     self.mapping.tp_rank)

            attention_params = AttentionParams(
                plain_attn_precision=self.plain_attn_precision)
            output = tri_attn_node(trt_hidden_states, trt_mask,
                                   attention_params)
            output.mark_output("output",
                               tensorrt_llm.str_dtype_to_trt(self.dtype))
        if self.n_optimization_profiles > 1:
            profile = builder.trt_builder.create_optimization_profile()
            seq_len_min = 32
            seq_len_max = 512
            seq_len_opt = self.seq_len

            profile.set_shape(trt_hidden_states.name,
                              [seq_len_min, seq_len_min, self.c_in],
                              [seq_len_opt, seq_len_opt, self.c_in],
                              [seq_len_max, seq_len_max, self.c_in])
            profile.set_shape(trt_mask.name, [seq_len_min, seq_len_min],
                              [seq_len_opt, seq_len_opt],
                              [seq_len_max, seq_len_max])

            builder_config.add_optimization_profile(profile)
        engine_path = Path(
            self.temp_dir
        ) / f"rank_{self.mapping.rank}.{self.mapping.tp_rank}.{self.mapping.dcp_rank}.plan"
        engine_buffer = builder.build_engine(network, builder_config)
        assert engine_buffer is not None
        with open(engine_path, "wb") as f:
            f.write(engine_buffer)
        tensorrt_llm.logger.info(
            f"Build engine for rank {self.mapping.rank}, tp_rank {self.mapping.tp_rank}, dcp_rank {self.mapping.dcp_rank} at {engine_path}"
        )
        return str(engine_path)


def run_single_rank(scenario: Scenario, engine_paths: list[str], inputs: dict,
                    weights_and_biases: dict):
    rank = tensorrt_llm.mpi_rank()
    module = TriangleAttnNodesParallelism(
        world_size=scenario.tp_size * scenario.dcp_size,
        rank=rank,
        dcp_size=scenario.dcp_size,
        tp_size=scenario.tp_size,
        seq_len=scenario.seq_len,
        c_in=scenario.c_in,
        c_hidden=scenario.c_hidden,
        num_attention_heads=scenario.num_heads,
        dtype=scenario.dtype,
        n_optimization_profiles=scenario.n_optimization_profiles,
        inputs=inputs,
        weights_and_biases=weights_and_biases)

    try:
        module.run(engine_paths[module.mapping.rank])
    except Exception:
        traceback.print_exc()
        raise
    return True


def _generate_scenarios():
    max_world_size = torch.cuda.device_count()
    scenarios = []
    ids = []
    for node_type in [
            TriangleAttentionNodeType.STARTING, TriangleAttentionNodeType.ENDING
    ]:
        for n_optimization_profiles in [0, 1]:
            for tp_size, dcp_size in product([1, 2, 4], repeat=2):
                if tp_size * dcp_size > max_world_size:
                    continue
                if tp_size > 4:
                    continue
                scenarios.append(
                    Scenario(tp_size=tp_size,
                             dcp_size=dcp_size,
                             node_type=node_type,
                             n_optimization_profiles=n_optimization_profiles))
                ids.append(
                    f"{node_type.name}_{n_optimization_profiles}_{tp_size}_{dcp_size}"
                )
    return scenarios, ids


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_scenarios()[0],
                         ids=_generate_scenarios()[1])
def test_triangle_nodes_parallelism(scenario: Scenario):
    torch.manual_seed(42)
    x = torch.randn(scenario.seq_len, scenario.seq_len, scenario.c_in)
    mask = torch.randn(scenario.seq_len, scenario.seq_len)
    world_size = scenario.tp_size * scenario.dcp_size
    torch_dtype = str_dtype_to_torch(scenario.dtype)
    inputs = {'input_s': x, 'mask': mask}
    weights_and_biases = create_triangle_attention_node_weights(
        scenario.c_in, scenario.c_hidden, scenario.num_heads, torch_dtype)

    temp_name = next(tempfile._get_candidate_names())
    temp_dir = Path("/tmp") / temp_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Run build engine for each rank first
    engine_paths = []
    for rank in range(world_size):
        engine_path = TriangleAttnNodesParallelism(
            world_size=world_size,
            rank=rank,
            dcp_size=scenario.dcp_size,
            tp_size=scenario.tp_size,
            seq_len=scenario.seq_len,
            c_in=scenario.c_in,
            c_hidden=scenario.c_hidden,
            num_attention_heads=scenario.num_heads,
            dtype=scenario.dtype,
            n_optimization_profiles=scenario.n_optimization_profiles,
            inputs={
                'input_s': x,
                'mask': mask
            },
            weights_and_biases=weights_and_biases,
            temp_dir=temp_dir,
            building=True).build()
        engine_paths.append(engine_path)

    # Run inference for each rank
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(scenario, engine_paths, inputs, weights_and_biases)] *
                 world_size))
        for r in results:
            assert r is True
