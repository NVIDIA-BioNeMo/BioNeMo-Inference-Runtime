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
import traceback

import pytest
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.functional import Tensor, concat, expand_dims

from tensorrt_bionemo._trt.functional import send_recv
from tensorrt_bionemo.mapping import Mapping


def _build_network(mapping: Mapping, input_shape: tuple[int],
                   dtype: torch.dtype) -> str:
    import tensorrt as trt
    from tensorrt_llm.builder import Builder
    builder = Builder()
    trt_dtype = "float32"
    model_name = "test_send_recv"
    builder_config = builder.create_builder_config(
        name=model_name,
        precision=trt_dtype,
        tensor_parallel=mapping.tp_size,
        strongly_typed=True,
        data_parallel=mapping.dcp_size)
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    network.plugin_config.set_nccl_plugin(trt_dtype)

    with tensorrt_llm.net_guard(network):
        chunk = Tensor(name="input", shape=input_shape, dtype=trt_dtype)
        all_chunks = [
            None,
        ] * mapping.dcp_size
        add_one = chunk + 1
        add_one = expand_dims(add_one, 0)
        all_chunks[mapping.dcp_rank] = add_one
        chunk_recv = chunk
        for i in range(1, mapping.dcp_size):
            chunk_recv = send_recv(chunk_recv,
                                   mapping.prev_dcp_rank(),
                                   mapping.next_dcp_rank(),
                                   group=mapping.dcp_group,
                                   group_stride=mapping.tp_size)
            add_one = chunk_recv + 1
            add_one = expand_dims(add_one, 0)
            all_chunks[(mapping.dcp_rank - i) % mapping.dcp_size] = add_one
        output = concat(all_chunks)
        output.mark_output("output", trt_dtype)
    engine_buffer = builder.build_engine(network, builder_config)
    assert engine_buffer is not None
    return engine_buffer


def _run(engine_buffer: str, chunk: torch.Tensor, outputs: dict[str,
                                                                torch.Tensor]):
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    session.run(inputs={"input": chunk},
                outputs=outputs,
                stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()


def run_single_rank(x: torch.Tensor, world_size: int, dcp_size: int,
                    tp_size: int):
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      dcp_size=dcp_size,
                      tp_size=tp_size)
    torch.cuda.set_device(rank)
    x = x.cuda()
    chunk = torch.chunk(x, mapping.dcp_size, dim=0)[mapping.dcp_rank]
    result = torch.empty(mapping.dcp_size,
                         mapping.tp_size,
                         x.shape[1],
                         x.shape[2],
                         dtype=torch.float32,
                         device="cuda")
    try:
        engine_buffer = _build_network(mapping, chunk.shape, chunk.dtype)
        _run(engine_buffer, chunk, {"output": result})
        torch.testing.assert_close(result, (x + 1).view(mapping.dcp_size,
                                                        mapping.tp_size,
                                                        x.shape[1], x.shape[2]))
    except Exception:
        traceback.print_exc()
        raise

    return True


def _generate_test_cases() -> list[tuple[int, int]]:
    world_size = torch.cuda.device_count()
    ret = []
    ret.append((2, 1))
    if world_size >= 4:
        ret.append((4, 1))
        ret.append((2, 2))
    if world_size >= 8:
        ret.append((8, 1))
        ret.append((4, 2))
        ret.append((2, 4))
    return ret


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("pair", _generate_test_cases())
def test_send_recv(pair):
    dcp_size, tp_size = pair
    world_size = dcp_size * tp_size
    x = torch.randn(world_size, 64, 128, dtype=torch.float32)

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(x, world_size, dcp_size, tp_size)] * world_size))
        for r in results:
            assert r is True


if __name__ == "__main__":
    test_send_recv((2, 1))
