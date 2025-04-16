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
from dataclasses import dataclass

import pytest
import torch
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_torch)
from test_utils.ref_layers import RefPairformerLayer

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.modules.transformers import PairformerLayer
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 16
    chunk_size: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(backend="VANILLA"),
        Scenario(backend="VANILLA", seq_len=128),
        # Scenario(backend="VANILLA", torch_dtype="bfloat16"), # FIXME
    ])
def test_pairformer_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(sc.backend).Metadata

    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_layer = RefPairformerLayer.load_weights()
    ref_layer = ref_layer.to(device)

    weights_and_biases = create_pairformer_layer_weights(from_ref=ref_layer)

    layer = PairformerLayer(
        layer_idx=0,
        token_s=ref_layer.token_s,
        token_z=ref_layer.token_z,
        num_heads=ref_layer.num_heads,
        pairwise_head_width=ref_layer.pairwise_head_width,
        pairwise_num_heads=ref_layer.pairwise_num_heads,
        dtype=dtype,
        attn_backend=sc.backend,
        skip_create_weights=False,
    )
    layer.to(device)
    load_pairformer_layer_weights_torch(layer, weights_and_biases, dtype)

    s = torch.randn(sc.seq_len, ref_layer.token_s).to(device)
    z = torch.randn(sc.seq_len, sc.seq_len, ref_layer.token_z).to(device)
    mask = torch.randn(sc.seq_len).to(device)
    pair_mask = torch.randn(sc.seq_len, sc.seq_len).to(device)

    attn_metadata = metadata_cls(mapping=Mapping())

    with torch.inference_mode():
        ref_s_float, ref_z_float = ref_layer(s, z, mask, pair_mask)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        pair_mask = pair_mask.to(dtype)
        ref_layer = ref_layer.to(dtype)

        ref_s, ref_z = ref_layer(s, z, mask, pair_mask)
        output_s, output_z = layer(s,
                                   z,
                                   mask,
                                   pair_mask,
                                   attn_metadata=attn_metadata)

    assert ref_s.shape == output_s.shape
    assert ref_z.shape == output_z.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_s, output_s, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)
    else:
        pass
