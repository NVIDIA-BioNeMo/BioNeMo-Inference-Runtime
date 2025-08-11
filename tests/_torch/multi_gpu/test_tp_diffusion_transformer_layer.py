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
from test_utils.boltz.create_and_load_weights import (
    create_diffusion_transformer_layer_weights,
    load_diffusion_transformer_layer_weights_torch)

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.layers.transformers import \
    DiffusionTransformerLayer
from tensorrt_bionemo.mapping import Mapping


def run_single_rank(tensor_parallel_size, single_rank_forward_func, a, s, z,
                    mask, weights_and_biases):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(a, s, z, mask, tensor_parallel_size, rank,
                                 weights_and_biases)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def diffusion_transformer_layer_forward(a, s, z, mask, tensor_parallel_size,
                                        rank, weights_and_biases):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    a = a.cuda()
    s = s.cuda()
    z = z.cuda()
    mask = mask.cuda()
    mapping = Mapping(world_size=tensor_parallel_size,
                      tp_size=tensor_parallel_size,
                      rank=rank)
    dtype = torch.float32
    attn_pairwise_metadata_cls = get_attention_backend(
        "VANILLA", AttentionType.PAIRWISE).Metadata
    dt_layer = DiffusionTransformerLayer(
        layer_idx=0,
        num_heads=16,
        dim=a.shape[-1],
        dim_single_cond=s.shape[-1],
        dim_pairwise=z.shape[-1],
        with_pair_bias_cache=True,
        dtype=dtype,
        skip_create_weights=False,
        mapping=mapping,
    )
    load_diffusion_transformer_layer_weights_torch(dt_layer,
                                                   weights_and_biases,
                                                   dtype=dtype)
    dt_layer.cuda()

    multi_dev_output = dt_layer(
        a, s, z, mask, attn_metadata=attn_pairwise_metadata_cls(bias_cache={}))
    single_dev_dt_layer = DiffusionTransformerLayer(
        layer_idx=0,
        num_heads=16,
        dim=a.shape[-1],
        dim_single_cond=s.shape[-1],
        dim_pairwise=z.shape[-1],
        dtype=dtype,
        skip_create_weights=False,
        with_pair_bias_cache=True,
        mapping=Mapping(),
    )
    load_diffusion_transformer_layer_weights_torch(single_dev_dt_layer,
                                                   weights_and_biases,
                                                   dtype=dtype)

    single_dev_dt_layer.cuda()

    single_dev_output = single_dev_dt_layer(
        a, s, z, mask, attn_metadata=attn_pairwise_metadata_cls(bias_cache={}))
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
def test_tp_diffusion_transformer_layer(dim, tp_size):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Needs {tp_size} GPUs to run this test")
    torch.manual_seed(42)
    tensor_parallel_size = tp_size
    seq_len = 128
    b = 1
    s = torch.randn(b, seq_len, dim, dtype=torch.float32)
    a = torch.randn(b, seq_len, dim, dtype=torch.float32)
    z = torch.randn(b, seq_len, seq_len, dim, dtype=torch.float32)
    mask = torch.randn(b, seq_len, dtype=torch.float32)
    weights_and_biases = create_diffusion_transformer_layer_weights(
        num_heads=16, dim=dim, dim_single_cond=dim, dim_pairwise=dim)

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, diffusion_transformer_layer_forward,
                    a, s, z, mask, weights_and_biases)] * tp_size))
        for r in results:
            assert r is True
