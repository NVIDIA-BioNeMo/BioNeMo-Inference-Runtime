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

from tensorrt_bionemo._torch.distributed import (AllGatherMode, ParallelConfig,
                                                 allgather)
from tensorrt_bionemo.mapping import Mapping


def run_single_rank(x, y, tp_size, dcp_size):
    try:
        rank = tensorrt_llm.mpi_rank()
        torch.cuda.set_device(rank)
        mapping = Mapping(world_size=tp_size * dcp_size,
                          tp_size=tp_size,
                          dcp_size=dcp_size,
                          rank=rank)
        x = x.cuda()
        y = y.cuda()
        dp_chunk = x.shape[0] // dcp_size
        tp_chunk = x.shape[1] // tp_size
        chunk_x = x[mapping.dcp_rank * dp_chunk:(mapping.dcp_rank + 1) *
                    dp_chunk,
                    mapping.tp_rank * tp_chunk:(mapping.tp_rank + 1) * tp_chunk,
                    ...]
        chunk_y = y[mapping.dcp_rank * dp_chunk:(mapping.dcp_rank + 1) *
                    dp_chunk,
                    mapping.tp_rank * tp_chunk:(mapping.tp_rank + 1) * tp_chunk,
                    ...]

        parallel_config = ParallelConfig(tensor_parallel_size=tp_size,
                                         tensor_parallel_rank=mapping.tp_rank,
                                         data_parallel_size=dcp_size,
                                         data_parallel_rank=mapping.dcp_rank,
                                         gpus_per_node=tp_size * dcp_size,
                                         gather_output=True)
        chunk = chunk_x + chunk_y
        out_tp = allgather(chunk,
                           parallel_config,
                           mode=AllGatherMode.TP,
                           gather_dim=1)
        out_dp = allgather(out_tp,
                           parallel_config,
                           mode=AllGatherMode.DP,
                           gather_dim=0)

        ref_output = x + y
        torch.cuda.synchronize()
        torch.testing.assert_close(out_dp, ref_output, atol=1e-4, rtol=1e-3)
        return True
    except Exception:
        traceback.print_exc()
        raise


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
def test_dp_tp_allgather():
    torch.manual_seed(42)
    tp_size = 2
    dcp_size = 2
    if torch.cuda.device_count() < tp_size * dcp_size:
        tp_size = 1
        dcp_size = 2
    x = torch.randn(32, 16, 16)
    y = torch.randn(32, 16, 16)
    world_size = tp_size * dcp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(run_single_rank,
                               *zip(*[(x, y, tp_size, dcp_size)] * world_size))
        for r in results:
            assert r is True
