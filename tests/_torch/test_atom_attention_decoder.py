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
from test_utils.boltz.create_and_load_weights import (
    create_atom_attention_decoder_weights,
    load_atom_attention_decoder_weights_torch)
from test_utils.boltz.ref_layers import RefAtomAttentionDecoder

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo._torch.layers.transformers.atom import \
    AtomAttentionDecoder
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    BoltzDiffusionTransformer
from tensorrt_bionemo.configs import DiffusionTransformerConfig


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 928
    dtype: str = "float32"
    atom_window_queries: int = 32
    atom_window_keys: int = 128
    multiplicity: int = 1
    depth: int = 3
    heads: int = 4
    dim: int = 128
    n_res: int = 117
    batch_size: int = 1
    token_s: int = 768 // 2


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32", multiplicity=1),
    Scenario(dtype="float16", multiplicity=3),
    Scenario(dtype="float16", multiplicity=5)
])
def test_atom_attention_encoder(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = sc.batch_size
    device = torch.device('cuda')
    ref_module = RefAtomAttentionDecoder.load_weights().to(device)
    ref_module.eval()
    dtype = str_dtype_to_torch(sc.dtype)

    K = sc.n_atoms // sc.atom_window_queries
    W = sc.atom_window_queries
    H = sc.atom_window_keys
    keys_indexing_matrix = create_indexing_matrix(K, W, H, device)
    to_keys = lambda x: query_to_keys(x, keys_indexing_matrix, W, H)

    a = torch.randn(bs * sc.multiplicity, sc.n_res,
                    2 * sc.token_s).to(device) / 1000.0
    q = torch.randn(bs * sc.multiplicity, sc.n_atoms,
                    sc.dim).to(device) / 1000.0
    c = torch.randn(bs, sc.n_atoms, sc.dim).to(device) / 1000.0
    atom_dec_bias = torch.randn(bs, sc.n_atoms, H,
                                sc.heads * sc.depth).to(device) / 1000.0
    atom_mask = torch.randint(0,
                              2, (bs, sc.n_atoms),
                              device=device,
                              dtype=torch.float32)

    atom_to_token = torch.randn(bs, sc.n_atoms, sc.n_res).to(device) / 1000.0
    attn_metadata = AttentionMetadata(
        query_to_keys=to_keys,
        bias_cache={},
    )

    dim_single_cond = ref_module.atom_decoder.diffusion_transformer.layers[
        0].adaln.dim_single_cond
    diffusion_transformer_config = DiffusionTransformerConfig(
        architecture="boltz_diffusion_transformer",
        version="v2",
        num_blocks=sc.depth,
        num_heads=sc.heads,
        dim=sc.dim,
        dim_single_cond=dim_single_cond,
        dtype=sc.dtype)
    token_s = sc.token_s
    atom_s = ref_module.a_to_q_trans.out_features
    model = AtomAttentionDecoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=sc.atom_window_queries,
        atoms_per_window_keys=sc.atom_window_keys,
        diffusion_transformer_config=diffusion_transformer_config,
        diffusion_transformer_cls=BoltzDiffusionTransformer,
        dtype=dtype).to(device)
    model.eval()

    weights_and_biases = create_atom_attention_decoder_weights(
        from_ref=ref_module)
    load_atom_attention_decoder_weights_torch(model, weights_and_biases)

    with torch.no_grad():
        ref_output_float = ref_module(atom_to_token, atom_mask, a, q, c,
                                      atom_dec_bias, sc.multiplicity,
                                      attn_metadata)

        atom_to_token = atom_to_token.to(dtype)
        atom_dec_bias = atom_dec_bias.to(dtype)
        a = a.to(dtype)
        q = q.to(dtype)
        c = c.to(dtype)

        if dtype != torch.float32:
            ref_module = ref_module.to(dtype)
            ref_output = ref_module(atom_to_token, atom_mask, a, q, c,
                                    atom_dec_bias, sc.multiplicity,
                                    attn_metadata)
            ref_output = ref_output.view(bs, sc.multiplicity, sc.n_atoms, -1)
            ref_output = ref_output.to(torch.float32)

        a = a.view(bs, sc.multiplicity, sc.n_res, -1)
        q = q.view(bs, sc.multiplicity, sc.n_atoms, -1)
        c = c.unsqueeze(1)
        output = model(atom_to_token=atom_to_token,
                       atom_pad_mask=atom_mask,
                       a=a,
                       q=q,
                       c=c,
                       bias=atom_dec_bias,
                       attn_metadata=attn_metadata)
        output = output.to(torch.float32)

    ref_output_float = ref_output_float.view(bs, sc.multiplicity, sc.n_atoms,
                                             -1)

    if dtype == torch.float32:
        torch.testing.assert_close(output,
                                   ref_output_float,
                                   atol=5e-2,
                                   rtol=1e-2)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output - ref_output_float))
        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2
