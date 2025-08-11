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

from test_utils.boltz.ref_attn import \
    RefPairwiseSelfAttention as BoltzRefPairwiseSelfAttention
from test_utils.boltz.ref_attn import \
    RefTriangleAttention as BoltzRefTriangleAttention

from tensorrt_bionemo.hubs.checkpoint import load_hf_weights


class RefPairwiseSelfAttention(BoltzRefPairwiseSelfAttention):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.msa_att_row.mha",
                     state_dict: dict = None,
                     num_heads: int = 8):
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.linear_q.weight", None),
            (f"{layer_path}.linear_k.weight", None),
            (f"{layer_path}.linear_v.weight", None),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
            (f"{layer_path}.linear_o.weight", f"{layer_path}.linear_o.bias"),
        ]
        c_q = c_k = c_v = state_dict[f"{layer_path}.linear_q.weight"].shape[1]

        m = cls(c_s=c_q,
                c_z=None,
                num_heads=num_heads,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                },
                compute_pair_bias=False,
                initial_norm=False,
                transform_mask=False)
        layers = [m.proj_q, m.proj_k, m.proj_v, m.proj_g, m.proj_o]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])

        return m


class RefTriangleAttention(BoltzRefTriangleAttention):

    @classmethod
    def load_weights(cls,
                     model: str = "openfold2_ptm_1",
                     layer_path: str = "evoformer.blocks.0.msa_att_row.mha",
                     state_dict: dict = None,
                     num_heads: int = 8):
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)
        weights_biases_path = [
            (f"{layer_path}.linear_q.weight", None),
            (f"{layer_path}.linear_k.weight", None),
            (f"{layer_path}.linear_v.weight", None),
            (f"{layer_path}.linear_g.weight", f"{layer_path}.linear_g.bias"),
            (f"{layer_path}.linear_o.weight", f"{layer_path}.linear_o.bias"),
        ]
        c_q = c_k = c_v = state_dict[f"{layer_path}.linear_q.weight"].shape[1]
        c_hidden = c_q // num_heads
        m = cls(c_q=c_q,
                c_k=c_k,
                c_v=c_v,
                c_hidden=c_hidden,
                no_heads=num_heads,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                })
        layers = [m.linear_q, m.linear_k, m.linear_v, m.linear_g, m.linear_o]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m
