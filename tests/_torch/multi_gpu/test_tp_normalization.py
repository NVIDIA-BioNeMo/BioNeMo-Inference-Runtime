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

import pytest
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from test_utils.boltz.create_and_load_weights import (create_adaln_weights,
                                                      load_adaln_weights_torch)

from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo.mapping import Mapping


def run_single_rank(tensor_parallel_size, single_rank_forward_func, a, s,
                    weights_and_biases):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(a, s, tensor_parallel_size, rank,
                                 weights_and_biases)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def adaln_forward(a, s, tensor_parallel_size, rank, weights_and_biases):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    a = a.cuda()
    s = s.cuda()
    mapping = Mapping(world_size=tensor_parallel_size,
                      tp_size=tensor_parallel_size,
                      rank=rank)
    dtype = torch.float32
    adaln = AdaLN(
        dim=a.shape[-1],
        dim_single_cond=s.shape[-1],
        dtype=dtype,
        skip_create_weights=False,
        mapping=mapping,
    )
    load_adaln_weights_torch(adaln, weights_and_biases, dtype=dtype)
    adaln.cuda()

    multi_dev_output = adaln(a, s)
    single_dev_adaln = AdaLN(
        dim=a.shape[-1],
        dim_single_cond=s.shape[-1],
        dtype=dtype,
        skip_create_weights=False,
        mapping=Mapping(),
    )
    load_adaln_weights_torch(single_dev_adaln, weights_and_biases, dtype=dtype)

    single_dev_adaln.cuda()

    single_dev_output = single_dev_adaln(a, s)
    torch.cuda.synchronize()
    assert multi_dev_output.shape == single_dev_output.shape
    torch.testing.assert_close(multi_dev_output,
                               single_dev_output,
                               atol=1e-4,
                               rtol=1e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("dim", [128, 256], ids=["128", "256"])
@pytest.mark.parametrize("tp_size", [2, 4], ids=["tp2", "tp4"])
def test_tp_adaln(dim, tp_size):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Needs {tp_size} GPUs to run this test")
    torch.manual_seed(42)
    tensor_parallel_size = tp_size
    seq_len = 128
    b = 1
    s = torch.randn(b, seq_len, dim, dtype=torch.float32)
    a = torch.randn(b, seq_len, dim, dtype=torch.float32)

    weights_and_biases = create_adaln_weights(dim=dim, dim_single_cond=dim)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, adaln_forward, a, s,
                    weights_and_biases)] * tp_size))
        for r in results:
            assert r is True
