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

import numpy as np
import pytest
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from test_utils.create_and_load_weights import (
    create_triangle_attention_weights, load_triangle_attention_weights_torch)

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.attention import TriangleAttention
from tensorrt_bionemo.mapping import Mapping


def run_single_rank(single_rank_forward_func, tensor_parallel_size, input,
                    biases, hidden_size, num_attention_heads,
                    weights_and_biases, dtype, backend):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, biases, hidden_size,
                                 num_attention_heads, tensor_parallel_size,
                                 rank, weights_and_biases, dtype, backend)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def triangle_attn_forward(x, biases, hidden_size, num_attention_heads,
                          tensor_parallel_size, tensor_parallel_rank,
                          weights_and_biases, dtype, backend):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    x = x.cuda()
    biases = [bias.cuda() for bias in biases]

    mapping = Mapping(world_size=tensor_parallel_size,
                      tp_size=tensor_parallel_size,
                      rank=tensor_parallel_rank)
    metadata_cls = get_attention_backend(backend).Metadata
    attn_metadata = metadata_cls(mapping=mapping)

    tri_attn = TriangleAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        layer_idx=0,
        dtype=dtype,
        attn_backend=backend,
        skip_create_weights=False,
        mapping=mapping,
    )
    load_triangle_attention_weights_torch(tri_attn,
                                          weights_and_biases,
                                          dtype=dtype)
    tri_attn.cuda()

    # tri_attn = torch.compile(tri_attn, fullgraph=True)
    multi_dev_output = tri_attn.forward(x, biases, attn_metadata)

    # create single mapping
    mapping = Mapping()

    attn_metadata = metadata_cls(mapping=mapping)
    if backend == "TRIFAST":
        attn_metadata.closest_n = 2**int(np.ceil(np.log2(x.size(1))))
    single_dev_tri_attn = TriangleAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        layer_idx=0,
        dtype=dtype,
        attn_backend=backend,
        skip_create_weights=False,
        mapping=mapping,
    )
    load_triangle_attention_weights_torch(single_dev_tri_attn,
                                          weights_and_biases,
                                          dtype=dtype)

    if tensor_parallel_rank == 0:
        single_dev_output = single_dev_tri_attn.forward(x, biases,
                                                        attn_metadata)
        torch.cuda.synchronize()
        assert multi_dev_output.shape == single_dev_output.shape
        if dtype == torch.float32:
            torch.testing.assert_close(multi_dev_output,
                                       single_dev_output,
                                       atol=1e-3,
                                       rtol=1e-4)
        else:
            torch.testing.assert_close(multi_dev_output,
                                       single_dev_output,
                                       atol=1e-2,
                                       rtol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("backend", ["VANILLA", "TRIFAST"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_attention_heads", [4, 3],
                         ids=["balanced", "unbalanced"])
def test_triangle_attn_forward(backend, dtype, num_attention_heads):
    torch.manual_seed(42)
    seq_len = 32
    hidden_size = 128
    tensor_parallel_size = 2
    bs = 1
    original_mask = torch.randint(0, 2, (bs, seq_len, 1, 1, seq_len))
    biases = [
        original_mask.to(dtype) * torch.finfo(dtype).min,
        torch.randn(bs, num_attention_heads, seq_len, seq_len, dtype=dtype)
    ]
    x = torch.randn(bs, seq_len, seq_len, hidden_size, dtype=dtype)

    weights_and_biases = create_triangle_attention_weights(
        c_q=hidden_size,
        c_k=hidden_size,
        c_v=hidden_size,
        torch_dtype=torch.float32)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(triangle_attn_forward, tensor_parallel_size, x, biases,
                    hidden_size, num_attention_heads, weights_and_biases, dtype,
                    backend)] * 2))
        if num_attention_heads % 2 != 0:
            with pytest.raises(AssertionError):
                for r in results:
                    assert r is True
        else:
            for r in results:
                assert r is True
