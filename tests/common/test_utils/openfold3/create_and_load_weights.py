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

import torch
from test_utils.openfold3.ref_layers_from_oss import \
    RefSwiGLUTransitionFromOF3OSS


def create_pair_transition_weights(
        from_ref: RefSwiGLUTransitionFromOF3OSS = None):
    layer_norm_weight = from_ref.layer_norm.weight.data
    layer_norm_bias = from_ref.layer_norm.bias.data
    linear_a_weight = from_ref.swiglu.linear_a.weight.data
    linear_b_weight = from_ref.swiglu.linear_b.weight.data
    linear_out_weight = from_ref.linear_out.weight.data
    return layer_norm_weight, layer_norm_bias, linear_a_weight, linear_b_weight, linear_out_weight


def load_pair_transition_weights_torch(module,
                                       weights_and_biases,
                                       dtype=torch.float32):
    layer_norm_weight, layer_norm_bias, linear_a_weight, linear_b_weight, linear_out_weight = weights_and_biases
    module.norm.weight.data.copy_(layer_norm_weight.to(dtype).to("cuda"))
    module.norm.bias.data.copy_(layer_norm_bias.to(dtype).to("cuda"))
    module.fused_fc2_fc1.weight.data.copy_(
        torch.cat([
            linear_b_weight.to(dtype).to("cuda"),
            linear_a_weight.to(dtype).to("cuda")
        ],
                  dim=0))
    module.fc3.weight.data.copy_(linear_out_weight.to(dtype).to("cuda"))


def create_msa_pair_weighted_averaging_weights(from_ref=None):
    layer_norm_m_weight = from_ref.layer_norm_m.weight.data
    layer_norm_m_bias = from_ref.layer_norm_m.bias.data
    layer_norm_z_weight = from_ref.layer_norm_z.weight.data
    layer_norm_z_bias = from_ref.layer_norm_z.bias.data

    linear_z_weight = from_ref.linear_z.weight.data
    linear_v_weight = from_ref.linear_v.weight.data
    linear_o_weight = from_ref.linear_o.weight.data
    linear_g_weight = from_ref.linear_g.weight.data
    return (layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight,
            layer_norm_z_bias, linear_z_weight, linear_v_weight,
            linear_o_weight, linear_g_weight)


def load_msa_pair_weighted_averaging_weights_torch(module,
                                                   weights_and_biases,
                                                   dtype=torch.float32):
    """Load reference ``PairWeightedAveraging`` weights into the production
    module. The production module fuses ``linear_v`` and ``linear_g`` into
    a single ``fused_proj_m_g`` layer (output split: ``[v, g]`` along
    dim=-1, see ``PairWeightedAveraging.fused_proj_m_g``); we concatenate them
    along the
    output dim (dim=0 of the Linear weight, since ``Linear.weight`` has
    shape ``(out, in)``).
    """
    (layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight,
     layer_norm_z_bias, linear_z_weight, linear_v_weight, linear_o_weight,
     linear_g_weight) = weights_and_biases
    module.norm_m.weight.data.copy_(layer_norm_m_weight.to(dtype).to("cuda"))
    module.norm_m.bias.data.copy_(layer_norm_m_bias.to(dtype).to("cuda"))
    module.norm_z.weight.data.copy_(layer_norm_z_weight.to(dtype).to("cuda"))
    module.norm_z.bias.data.copy_(layer_norm_z_bias.to(dtype).to("cuda"))
    module.proj_z.weight.data.copy_(linear_z_weight.to(dtype).to("cuda"))
    module.fused_proj_m_g.weight.data.copy_(
        torch.cat([
            linear_v_weight.to(dtype).to("cuda"),
            linear_g_weight.to(dtype).to("cuda"),
        ],
                  dim=0))
    module.proj_o.weight.data.copy_(linear_o_weight.to(dtype).to("cuda"))
