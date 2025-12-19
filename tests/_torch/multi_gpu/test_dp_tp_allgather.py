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
import torch
import torch.distributed as dist
from mpi4py.futures import MPIPoolExecutor

from tensorrt_bionemo._torch.distributed import (
    init_distributed_environment, register_dcp_group_coordinator,
    register_tp_group_coordinator)
from tensorrt_bionemo.mapping import Mapping
from tests.common.test_utils.mpi import set_mpi_env


def run_single_rank(x, y, tp_size, dcp_size):
    try:
        mpi_rank, mpi_world_size = set_mpi_env()
        torch.cuda.set_device(mpi_rank)
        init_distributed_environment(device_id=torch.device(mpi_rank))
        rank = torch.distributed.get_rank()
        assert rank == mpi_rank, "rank mismatch"
        mapping = Mapping(world_size=tp_size * dcp_size,
                          tp_size=tp_size,
                          dcp_size=dcp_size,
                          rank=rank)
        tp_group_coordinator = register_tp_group_coordinator(mapping)
        dcp_group_coordinator = register_dcp_group_coordinator(mapping)

        x = x.cuda()
        y = y.cuda()
        dp_chunk = x.shape[0] // dcp_size
        tp_chunk = x.shape[1] // tp_size
        chunk_x = x[mapping.dcp_rank * dp_chunk:(mapping.dcp_rank + 1) *
                    dp_chunk, mapping.tp_rank *
                    tp_chunk:(mapping.tp_rank + 1) * tp_chunk, ...]
        chunk_y = y[mapping.dcp_rank * dp_chunk:(mapping.dcp_rank + 1) *
                    dp_chunk, mapping.tp_rank *
                    tp_chunk:(mapping.tp_rank + 1) * tp_chunk, ...]

        chunk = chunk_x + chunk_y
        out_tp = tp_group_coordinator().all_gather(chunk, dim=1)
        # dist.barrier(group=dcp_group_coordinator().device_group)
        dcp_group_coordinator().barrier()
        out_dp = dcp_group_coordinator().all_gather(out_tp, dim=0)

        ref_output = x + y
        torch.cuda.synchronize()
        torch.testing.assert_close(out_dp, ref_output, atol=1e-4, rtol=1e-3)
        return True
    except Exception:
        traceback.print_exc()
    finally:
        dist.destroy_process_group()


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
