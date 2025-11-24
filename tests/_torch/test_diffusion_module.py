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
    create_diffusion_module_weights, load_diffusion_module_weights_torch)
from test_utils.boltz.ref_layers import RefDiffusionModule

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo._torch.modules.boltz.structure import DiffusionModule
from tensorrt_bionemo.configs import DiffusionTransformerConfig


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 928
    dtype: str = "bfloat16"
    batch_size: int = 1
    atom_window_queries: int = 32
    atom_window_keys: int = 128
    n_res: int = 117
    dim: int = 128
    heads: int = 4
    depth: int = 3
    token_s: int = 768 // 2
    multiplicity: int = 1


def init_config(sc: Scenario, ref_module: RefDiffusionModule):
    token_transformer_config = DiffusionTransformerConfig(
        architecture="boltz_token_transformer",
        version="v2",
        num_blocks=24,
        num_heads=16,
        dim=2 * ref_module.token_s,
        dim_single_cond=2 * ref_module.token_s,
        dtype=sc.dtype)

    atom_attention_encoder_diff_transformer_config = DiffusionTransformerConfig(
        architecture="boltz_atom_attention_encoder_diffusion_transformer",
        version="v2",
        num_blocks=sc.depth,
        num_heads=sc.heads,
        dim=ref_module.atom_s,
        dim_single_cond=ref_module.atom_s,
        dtype=sc.dtype)

    atom_attention_decoder_diff_transformer_config = DiffusionTransformerConfig(
        architecture="boltz_atom_attention_decoder_diffusion_transformer",
        version="v2",
        num_blocks=sc.depth,
        num_heads=sc.heads,
        dim=ref_module.atom_s,
        dim_single_cond=ref_module.atom_s,
        dtype=sc.dtype)

    diffusion_module_config = DiffusionTransformerConfig(
        architecture="boltz_diffusion_module",
        version="v2",
        token_s=ref_module.token_s,
        atom_s=ref_module.atom_s,
        atoms_per_window_queries=sc.atom_window_queries,
        atoms_per_window_keys=sc.atom_window_keys,
        dim_fourier=ref_module.dim_fourier,
        conditioning_transition_layers=ref_module.
        conditioning_transition_layers,
        atom_encoder=atom_attention_encoder_diff_transformer_config,
        atom_decoder=atom_attention_decoder_diff_transformer_config,
        token_transformer=token_transformer_config,
        dtype=sc.dtype,
    )

    return diffusion_module_config


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32", multiplicity=5),
    Scenario(dtype="bfloat16", multiplicity=1)
])
def test_diffusion_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = sc.batch_size
    device = torch.device('cuda')
    ref_module = RefDiffusionModule.load_weights().to(device)
    ref_module.eval()

    dtype = str_dtype_to_torch(sc.dtype)

    K = sc.n_atoms // sc.atom_window_queries
    W = sc.atom_window_queries
    H = sc.atom_window_keys
    keys_indexing_matrix = create_indexing_matrix(K, W, H, device)
    to_keys = lambda x: query_to_keys(x, keys_indexing_matrix, W, H)

    q = torch.rand(bs, sc.n_atoms, sc.dim, dtype=torch.float32).to(device)
    c = torch.rand(bs, sc.n_atoms, sc.dim, dtype=torch.float32).to(device)
    r_noisy = torch.rand(bs, sc.n_atoms, 3, dtype=torch.float32).to(device)

    atom_enc_bias = torch.rand(bs,
                               sc.n_atoms,
                               H,
                               sc.heads * sc.depth,
                               dtype=torch.float32).to(device)
    atom_dec_bias = torch.rand(bs,
                               sc.n_atoms,
                               H,
                               sc.heads * sc.depth,
                               dtype=torch.float32).to(device)
    atom_token_bias = torch.rand(bs,
                                 sc.n_res,
                                 sc.n_res,
                                 sc.token_s,
                                 dtype=torch.float32).to(device)
    atom_pad_mask = torch.randint(0,
                                  2, (bs, sc.n_atoms),
                                  device=device,
                                  dtype=torch.float32)
    token_pad_mask = torch.randint(0,
                                   2, (bs, sc.n_res),
                                   device=device,
                                   dtype=torch.float32)

    s_inputs = torch.rand(bs, sc.n_res, sc.token_s,
                          dtype=torch.float32).to(device)
    s_trunk = torch.rand(bs, sc.n_res, sc.token_s,
                         dtype=torch.float32).to(device)
    times = torch.tensor([1.4], dtype=torch.float32).cuda()

    atom_to_token = torch.rand(bs, sc.n_atoms, sc.n_res,
                               dtype=torch.float32).to(device) / 1000.0
    attn_metadata = AttentionMetadata(
        query_to_keys=to_keys,
        bias_cache={},
    )

    weights_and_biases = create_diffusion_module_weights(from_ref=ref_module)
    diffusion_module_config = init_config(sc, ref_module)
    model = DiffusionModule(config=diffusion_module_config).cuda()
    load_diffusion_module_weights_torch(model, weights_and_biases)
    model.eval()

    with torch.inference_mode():
        ref_output_float = ref_module(atom_to_token, atom_pad_mask,
                                      token_pad_mask, s_inputs, s_trunk,
                                      r_noisy, times, q, c, atom_enc_bias,
                                      atom_token_bias, atom_dec_bias,
                                      sc.multiplicity, attn_metadata)

        atom_to_token = atom_to_token.to(dtype)
        atom_pad_mask = atom_pad_mask.to(dtype)
        token_pad_mask = token_pad_mask.to(dtype)

        q = q.to(dtype)
        c = c.to(dtype)
        r_noisy = r_noisy.to(dtype)
        atom_enc_bias = atom_enc_bias.to(dtype)
        atom_token_bias = atom_token_bias.to(dtype)
        atom_dec_bias = atom_dec_bias.to(dtype)
        s_inputs = s_inputs.to(dtype)
        s_trunk = s_trunk.to(dtype)

        ref_module = ref_module.to(dtype)
        ref_output = ref_module(atom_to_token, atom_pad_mask,
                                token_pad_mask, s_inputs, s_trunk, r_noisy,
                                times.to(dtype), q, c, atom_enc_bias,
                                atom_token_bias, atom_dec_bias, sc.multiplicity,
                                attn_metadata)
        r_noisy = r_noisy.unsqueeze(1)
        r_noisy = r_noisy.repeat_interleave(sc.multiplicity, 1)
        diffusion_conditioning_kwargs = {
            "q": q,
            "c": c,
            "atom_enc_bias": atom_enc_bias,
            "token_trans_bias": atom_token_bias,
            "atom_dec_bias": atom_dec_bias,
        }

        output = model(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            token_pad_mask=token_pad_mask,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            times=times,
            diffusion_conditioning_kwargs=diffusion_conditioning_kwargs,
            attn_metadata=attn_metadata)

        output = output[0].flatten(0, 1)

        if dtype == torch.float32:
            torch.testing.assert_close(output,
                                       ref_output_float,
                                       atol=1e-3,
                                       rtol=1e-4)
        else:
            diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
            diff0_mean = torch.mean(torch.abs(output.float() -
                                              ref_output_float))
            diff1_max = torch.max(
                torch.abs(ref_output.float() - ref_output_float))
            diff1_mean = torch.mean(
                torch.abs(ref_output.float() - ref_output_float))
            assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                          diff1_max) <= 0.5
            assert abs(diff0_mean - diff1_mean) <= 0.2
