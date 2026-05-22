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
import torch
import torch.distributed as dist
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_torch)

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.distributed import (
    init_distributed_environment, register_dcp_group_coordinator,
    register_tp_group_coordinator)
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerLayerV1
from tensorrt_bionemo.mapping import Mapping
from tests._torch import make_left_aligned_mask
from tests.common.test_utils.mpi import set_mpi_env
from tests.common.test_utils.tensor import mismatch_percentage


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
    # FIXME: enable max_transition_tp_size and max_attention_pairwise_tp_size when we have a way to test them
    # max_transition_tp_size: bool = True
    # max_attention_pairwise_tp_size: bool = True
    tri_attention_backend: str = "VANILLA"


def _generate_scenarios() -> list[PairformerScenario]:
    ret = []
    ids = []
    total_devs = torch.cuda.device_count()

    for seq_len, tri_attn_backend, dtype in product([16, 32], ["VANILLA"],
                                                    ["float32", "bfloat16"]):
        for tp_size, dcp_size in product([1, 2, 4, 8], repeat=2):
            if tp_size * dcp_size == 1:
                continue
            if tp_size * dcp_size > total_devs:
                continue
            if tp_size > 4:  # pairwise_num_heads
                continue

            ret.append(
                PairformerScenario(tp_size=tp_size,
                                   dcp_size=dcp_size,
                                   seq_len=seq_len,
                                   tri_attention_backend=tri_attn_backend,
                                   dtype=dtype))
            ids.append(
                f"tp{tp_size}-dcp{dcp_size}-{seq_len}-{tri_attn_backend}-{dtype}"
            )

    return ret, ids


def _pairformer_forward(s, z, mask, pair_mask, weights_and_biases, scenario):
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

    mpi_rank, mpi_world_size = set_mpi_env()
    init_distributed_environment(device_id=torch.device(mpi_rank))
    rank = torch.distributed.get_rank()
    assert rank == mpi_rank, "MPI rank and torch.distributed rank do not match"
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=tp_size * dcp_size,
                      tp_size=tp_size,
                      dcp_size=dcp_size,
                      rank=rank)
    # register default group coordinators
    _ = register_tp_group_coordinator(mapping)
    _ = register_dcp_group_coordinator(mapping)

    triangle_metadata_cls = get_attention_backend(
        scenario.tri_attention_backend, AttentionType.TRIANGLE).Metadata
    pairwise_metadata_cls = get_attention_backend(
        "VANILLA", AttentionType.PAIRWISE).Metadata
    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=mapping),
        "pairwise_attn": pairwise_metadata_cls(mapping=mapping),
    }

    s = s.cuda().to(dtype)
    z = z.cuda().to(dtype)
    mask = mask.cuda().to(dtype)
    pair_mask = pair_mask.cuda().to(dtype)

    pairformer_layer = PairformerLayerV1(
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
        max_attention_pairwise_tp_size=False,
        max_transition_tp_size=False,
        mapping=mapping,
        attention_initial_norm=True
    )  # Pairformer v1 uses attention_initial_norm
    pairformer_layer.cuda()
    pairformer_layer.eval()

    load_pairformer_layer_weights_torch(pairformer_layer,
                                        weights_and_biases,
                                        dtype=dtype)

    with torch.inference_mode():
        output_s, output_z = pairformer_layer(s, z, mask, pair_mask,
                                              attn_metadatas)

    mapping = Mapping()
    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=mapping),
        "pairwise_attn": pairwise_metadata_cls(mapping=mapping),
    }

    single_dev_pairformer_layer = PairformerLayerV1(
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
        max_attention_pairwise_tp_size=False,
        max_transition_tp_size=False,
        mapping=mapping,
        attention_initial_norm=True
    )  # Pairformer v1 uses attention_initial_norm
    load_pairformer_layer_weights_torch(single_dev_pairformer_layer,
                                        weights_and_biases,
                                        dtype=dtype)
    single_dev_pairformer_layer.cuda()
    single_dev_pairformer_layer.eval()

    with torch.inference_mode():
        single_dev_output_s, single_dev_output_z = single_dev_pairformer_layer(
            s, z, mask, pair_mask, attn_metadatas)

    if scenario.dtype == "float32":
        torch.testing.assert_close(output_s,
                                   single_dev_output_s,
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(output_z,
                                   single_dev_output_z,
                                   atol=1e-3,
                                   rtol=1e-3)
    else:
        # This not correct way to test bfloat16, but it is a good enough test for now.
        p = mismatch_percentage(output_s,
                                single_dev_output_s,
                                atol=1e-1,
                                rtol=1e-1)
        assert p < 0.5, f"Mismatch percentage: {p}% is too high"
        p = mismatch_percentage(output_z,
                                single_dev_output_z,
                                atol=1e-1,
                                rtol=1e-1)
        assert p < 0.5, f"Mismatch percentage: {p}% is too high"


def run_pairformer_single_rank(single_rank_forward_func, s, z, mask, pair_mask,
                               weights_and_biases, scenario):
    try:
        single_rank_forward_func(s, z, mask, pair_mask, weights_and_biases,
                                 scenario)
        return True
    except Exception:
        traceback.print_exc()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("scenario",
                         _generate_scenarios()[0],
                         ids=_generate_scenarios()[1])
def test_pairformer_parallelism(scenario: PairformerScenario):
    torch.manual_seed(42)
    bs = 1
    s = torch.randn(bs,
                    scenario.seq_len,
                    scenario.token_s,
                    dtype=torch.float32)
    z = torch.randn(bs,
                    scenario.seq_len,
                    scenario.seq_len,
                    scenario.token_z,
                    dtype=torch.float32)
    mask = make_left_aligned_mask(bs,
                                  scenario.seq_len,
                                  dtype=torch.float32,
                                  device="cpu")
    pair_mask = mask[..., None] * mask[..., None, :]

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
