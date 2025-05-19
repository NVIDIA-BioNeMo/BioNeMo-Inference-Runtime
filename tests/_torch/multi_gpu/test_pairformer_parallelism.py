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

import numpy as np
import pytest
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_torch)

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import PairformerLayer
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class PairformerScenario:
    token_s: int = 384
    token_z: int = 128
    num_heads: int = 16
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    no_update_s: bool = False
    no_update_z: bool = False
    dtype: str = "float32"
    tp_size: int = 1
    dcp_size: int = 1
    seq_len: int = 64
    max_transition_tp_size: bool = True
    max_attention_pairwise_tp_size: bool = True
    tri_attention_backend: str = "VANILLA"


def _generate_scenarios() -> list[PairformerScenario]:
    ret = []
    ids = []
    total_devs = torch.cuda.device_count()

    for seq_len, tri_attn_backend, dtype in product([64, 128],
                                                    ["VANILLA", "TRIFAST"],
                                                    ["float32", "bfloat16"]):
        for tp_size, dcp_size in product([1, 2, 4, 8], repeat=2):
            if tp_size * dcp_size > total_devs:
                continue
            if tp_size > 4:  # pairwise_num_heads
                continue
            if tri_attn_backend == "TRIFAST":
                if seq_len // dcp_size <= 16:
                    continue
            for max_transition_tp_size, max_attention_pairwise_tp_size in product(
                [True, False], repeat=2):
                ret.append(
                    PairformerScenario(
                        tp_size=tp_size,
                        dcp_size=dcp_size,
                        seq_len=seq_len,
                        max_transition_tp_size=max_transition_tp_size,
                        max_attention_pairwise_tp_size=
                        max_attention_pairwise_tp_size,
                        tri_attention_backend=tri_attn_backend,
                        dtype=dtype))
                ids.append(
                    f"{tp_size}-{dcp_size}-{seq_len}-{max_transition_tp_size}-{max_attention_pairwise_tp_size}-{tri_attn_backend}-{dtype}"
                )

    return ret, ids


def _pairformer_forward(s, z, mask, pair_mask, weights_and_biases, scenario,
                        rank):
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    tp_size = scenario.tp_size
    dcp_size = scenario.dcp_size
    token_s = scenario.token_s
    token_z = scenario.token_z
    num_heads = scenario.num_heads
    pairwise_head_width = scenario.pairwise_head_width
    pairwise_num_heads = scenario.pairwise_num_heads

    dtype = str_dtype_to_torch(scenario.dtype)

    s = s.cuda().to(dtype)
    z = z.cuda().to(dtype)
    mask = mask.cuda().to(dtype)
    pair_mask = pair_mask.cuda().to(dtype)

    mapping = Mapping(world_size=tp_size * dcp_size,
                      tp_size=tp_size,
                      dcp_size=dcp_size,
                      rank=rank)

    triangle_metadata_cls = get_attention_backend(
        scenario.tri_attention_backend).Metadata
    pairwise_metadata_cls = get_attention_backend("VANILLA").Metadata
    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=mapping),
        "pairwise_attn": pairwise_metadata_cls(mapping=mapping),
    }
    if scenario.tri_attention_backend == "TRIFAST":
        attn_metadatas["triangle_attn"].closest_n = 2**int(
            np.ceil(np.log2(scenario.seq_len // scenario.dcp_size)))

    pairformer_layer = PairformerLayer(
        layer_idx=0,
        token_s=token_s,
        token_z=token_z,
        num_heads=num_heads,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        dtype=dtype,
        triangle_attn_backend=scenario.tri_attention_backend,
        pairwise_attn_backend="VANILLA",
        skip_create_weights=False,
        max_attention_pairwise_tp_size=scenario.max_attention_pairwise_tp_size,
        max_transition_tp_size=scenario.max_transition_tp_size,
        mapping=mapping)
    pairformer_layer.cuda()
    pairformer_layer.eval()

    load_pairformer_layer_weights_torch(pairformer_layer,
                                        weights_and_biases,
                                        dtype=dtype)

    with torch.inference_mode():
        output = pairformer_layer(s, z, mask, pair_mask, attn_metadatas)

    mapping = Mapping()
    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=mapping),
        "pairwise_attn": pairwise_metadata_cls(mapping=mapping),
    }
    if scenario.tri_attention_backend == "TRIFAST":
        attn_metadatas["triangle_attn"].closest_n = 2**int(
            np.ceil(np.log2(scenario.seq_len)))

    single_dev_pairformer_layer = PairformerLayer(
        layer_idx=0,
        token_s=token_s,
        token_z=token_z,
        num_heads=num_heads,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        dtype=dtype,
        triangle_attn_backend=scenario.tri_attention_backend,
        pairwise_attn_backend="VANILLA",
        skip_create_weights=False,
        max_attention_pairwise_tp_size=scenario.max_attention_pairwise_tp_size,
        max_transition_tp_size=scenario.max_transition_tp_size,
        mapping=mapping)
    load_pairformer_layer_weights_torch(single_dev_pairformer_layer,
                                        weights_and_biases,
                                        dtype=dtype)
    single_dev_pairformer_layer.cuda()
    single_dev_pairformer_layer.eval()

    with torch.inference_mode():
        single_dev_output = single_dev_pairformer_layer(s, z, mask, pair_mask,
                                                        attn_metadatas)

    if scenario.dtype == "float32":
        torch.testing.assert_close(output,
                                   single_dev_output,
                                   atol=1e-3,
                                   rtol=1e-4)
    else:
        torch.testing.assert_close(output,
                                   single_dev_output,
                                   atol=6e-2,
                                   rtol=8e-3)


def run_pairformer_single_rank(single_rank_forward_func, s, z, mask, pair_mask,
                               weights_and_biases, scenario):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(s, z, mask, pair_mask, weights_and_biases,
                                 scenario, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_scenarios()[0],
                         ids=_generate_scenarios()[1])
def test_pairformer_parallelism(scenario: PairformerScenario):
    torch.manual_seed(42)
    bs = 1
    s = torch.randn(bs, scenario.seq_len, scenario.token_s, dtype=torch.float32)
    z = torch.randn(bs,
                    scenario.seq_len,
                    scenario.seq_len,
                    scenario.token_z,
                    dtype=torch.float32)
    mask = torch.randint(0, 2, (bs, scenario.seq_len), dtype=torch.float32)
    pair_mask = torch.randint(0,
                              2, (bs, scenario.seq_len, scenario.seq_len),
                              dtype=torch.float32)

    weights_and_biases = create_pairformer_layer_weights(
        token_s=scenario.token_s,
        token_z=scenario.token_z,
        num_heads=scenario.num_heads,
        pairwise_head_width=scenario.pairwise_head_width,
        pairwise_num_heads=scenario.pairwise_num_heads,
        torch_dtype=torch.float32)

    world_size = scenario.tp_size * scenario.dcp_size
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_pairformer_single_rank,
            *zip(*[(_pairformer_forward, s, z, mask, pair_mask,
                    weights_and_biases, scenario)] * world_size))
        for r in results:
            assert r is True
