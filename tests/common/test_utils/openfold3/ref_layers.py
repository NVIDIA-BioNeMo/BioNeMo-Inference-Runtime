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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_bionemo.hubs import load_weights

from test_utils.boltz.ref_attn import RefPairwiseSelfAttention as BoltzRefPairwiseSelfAttention
from test_utils.boltz.ref_layers import (
    RefDiffusionTransformerLayer as BoltzRefDiffusionTransformerLayer,
    RefConditionedTransitionBlock as BoltzRefConditionedTransitionBlock,
    RefAdaLN as BoltzRefAdaLN)

class Openfold3RefPairwiseSelfAttention(BoltzRefPairwiseSelfAttention):
    @classmethod
    def load_weights(
            cls,
            model: str = "openfold3",
            layer_path: str = "pairformer_module.layers.0.attention",
            key_layer: str = "attention_pair_bias",
            state_dict: Optional[dict] = None) -> 'Openfold3RefPairwiseSelfAttention':

        extract_state_dict = {}
        for key in state_dict.keys():
            if layer_path + '.{}'.format(key_layer) in key:
                if "mha" in key:
                    name = key.replace(key_layer + ".mha",
                                        "pair_bias_attn")
                    name = name.replace("layer_norm_s", "norm_s")
                    name = name.replace("linear_q", "proj_q")
                    name = name.replace("linear_k", "proj_k")
                    name = name.replace("linear_v", "proj_v")
                    name = name.replace("linear_g", "proj_g")
                    name = name.replace("linear_z", "proj_z.0")
                    name = name.replace("linear_o", "proj_o")
                    extract_state_dict[name] = state_dict[key]
                
                if "layer_norm_z" in key:
                    name = key.replace(key_layer + ".layer_norm_z",
                        "pair_bias_attn.proj_z.0")
                    extract_state_dict[name] = state_dict[key]
                
                if "linear_z" in key:
                    name = key.replace(key_layer + ".linear_z",
                        "pair_bias_attn.proj_z.1")
                    extract_state_dict[name] = state_dict[key]

        assert extract_state_dict != {}, "extract weights is empty"
        layer_path = layer_path + '.pair_bias_attn'
        
        weights_biases_path = [
            (f"{layer_path}.norm_s.weight", f"{layer_path}.norm_s.bias"),
            (f"{layer_path}.proj_q.weight", f"{layer_path}.proj_q.bias"),
            (f"{layer_path}.proj_k.weight", None),
            (f"{layer_path}.proj_v.weight", None),
            (f"{layer_path}.proj_g.weight", None),
            (f"{layer_path}.proj_z.0.weight", None),
            (f"{layer_path}.proj_z.1.weight", None),
            (f"{layer_path}.proj_o.weight", None),
        ]
        c_s = extract_state_dict[f"{layer_path}.proj_q.weight"].shape[0]
        
        c_z = extract_state_dict[f"{layer_path}.proj_z.1.weight"].shape[1]
        num_heads = extract_state_dict[f"{layer_path}.proj_z.1.weight"].shape[0]
        
        if f"{layer_path}.norm_s.weight" in extract_state_dict:
            attn = cls(c_s, c_z, num_heads, initial_norm=True)
            attn.proj_z[0] = nn.LayerNorm(c_z, bias=False)
            layers = [
                attn.norm_s, attn.proj_q, attn.proj_k, attn.proj_v, attn.proj_g,
                attn.proj_z[0], attn.proj_z[1], attn.proj_o
            ]
        else:
            attn = cls(c_s, c_z, num_heads, initial_norm=False)
            attn.proj_z[0] = nn.LayerNorm(c_z, bias=False)
            layers = [
                attn.proj_q, attn.proj_k, attn.proj_v, attn.proj_g,
                attn.proj_z[0], attn.proj_z[1], attn.proj_o
            ]
            weights_biases_path = weights_biases_path[1:]

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(extract_state_dict[bias_path])
            layer.weight.data.copy_(extract_state_dict[weights_path])
        return attn

class Openfold3RefAdaLN(BoltzRefAdaLN):
    @classmethod
    def load_weights(
            cls,
            model: str = "openfold3",
            layer_path: str = "sample_diffusion.diffusion_module.diffusion_transformer.blocks.0",
            key_layer: str = "layer_norm_a",
            state_dict: Optional[dict] = None) -> 'Openfold3RefAdaLN':
        
        extract_state_dict = {}
        for key in state_dict.keys():
            if layer_path + '.{}'.format(key_layer) in key:
                name = key.replace(key_layer,
                                    "adaln")
                name = name.replace("layer_norm_s", "s_norm")
                name = name.replace("linear_g", "s_scale")
                name = name.replace("linear_s", "s_bias")
                extract_state_dict[name] = state_dict[key]

        assert extract_state_dict != {}, "extract weights is empty"
        layer_path = layer_path + '.adaln'
        weights_biases_path = [
            # (f"{layer_path}.a_norm.weight", f"{layer_path}.a_norm.bias"),
            (f"{layer_path}.s_norm.weight", None),
            (f"{layer_path}.s_scale.weight", f"{layer_path}.s_scale.bias"),
            (f"{layer_path}.s_bias.weight", None),
        ]
        dim = extract_state_dict[weights_biases_path[1][0]].shape[0]
        dim_single_cond = extract_state_dict[weights_biases_path[1][0]].shape[1]
        m = cls(dim, dim_single_cond)
        layers = [
            # m.a_norm,
            m.s_norm,
            m.s_scale,
            m.s_bias,
        ]
        m.a_norm.weight.data.copy_(torch.ones(dim))

        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(extract_state_dict[bias_path])
            layer.weight.data.copy_(extract_state_dict[weights_path])
        return m

class Openfold3RefConditionedTransitionBlock(BoltzRefConditionedTransitionBlock):
    """
    Key difference from Boltz implementation:
    - Boltz uses: x = silu(fc1(x)) * fc2(x) * fc3(x) (three-way multiplication)
    - OpenFold3 uses: x = silu(fc1(x)) * fc2(x) (two-way multiplication)
    """
    
    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = self.swish_gate(a)
        a = self.output_projection(s) * self.b_to_a(b)
        return a
    
    @classmethod
    def load_weights(
            cls,
            model: str = "openfold3",
            layer_path: str = "sample_diffusion.diffusion_module.diffusion_transformer.blocks.0",
            key_layer: str = "conditioned_transition",
            state_dict: Optional[dict] = None
    ) -> 'Openfold3RefConditionedTransitionBlock':
        
        adaln = Openfold3RefAdaLN.load_weights(state_dict=state_dict,
                                      layer_path=layer_path,
                                      key_layer="conditioned_transition.layer_norm")
        
        m = cls(adaln.dim, adaln.dim_single_cond)
        
        delattr(m, "a_to_b")

        setattr(m, "adaln", adaln)
        extract_state_dict = {}
        for key in state_dict.keys():
            if layer_path + '.{}'.format(key_layer) in key:
                if "layer_norm" in key:
                    continue

                if "swiglu" in key:
                    name = key.replace(key_layer + ".swiglu.linear_a.weight",
                                        "swish_gate.0_a.weight")
                    name = name.replace(key_layer + ".swiglu.linear_b.weight",
                                        "swish_gate.0_b.weight")
                    extract_state_dict[name] = state_dict[key]
                    
                if "linear_out" in key or "linear_g" in key:
                    name = key.replace(key_layer + ".linear_out",
                                        "b_to_a")
                    name = name.replace(key_layer + ".linear_g",
                                        "output_projection.0")
                    extract_state_dict[name] = state_dict[key]
                    

        assert extract_state_dict != {}, "extract weights is empty"
        weights_biases_path = [
            ([f"{layer_path}.swish_gate.0_a.weight", f"{layer_path}.swish_gate.0_b.weight"], None),
            (f"{layer_path}.b_to_a.weight", None),
            (f"{layer_path}.output_projection.0.weight",
             f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.swish_gate[0],
            m.b_to_a,
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(extract_state_dict[bias_path])
            
            # Check if merge weights need to be done
            if isinstance(weights_path, str):
                layer.weight.data.copy_(extract_state_dict[weights_path])
            else:
                merge_weight = []
                for weight_path in weights_path:
                    merge_weight.append(extract_state_dict[weight_path])
                merge_weight = torch.cat(merge_weight, dim=0)
                layer.weight.data.copy_(merge_weight)
        return m

class Openfold3RefDiffusionTransformerLayer(BoltzRefDiffusionTransformerLayer):
    @classmethod
    def load_weights(cls,
                     model: str = "openfold3",
                     layer_path: str = "sample_diffusion.diffusion_module.diffusion_transformer.blocks.0",
        )-> 'Openfold3RefDiffusionTransformerLayer':

        state_dict = load_weights(name=model, hub="local")
        
        adaln = Openfold3RefAdaLN.load_weights(state_dict=state_dict,
                                      layer_path=layer_path,
                                      key_layer="attention_pair_bias.layer_norm_a")
        pair_bias_attn = Openfold3RefPairwiseSelfAttention.load_weights(state_dict=state_dict, 
                                                           layer_path=layer_path, 
                                                           key_layer="attention_pair_bias")
        
        transition = Openfold3RefConditionedTransitionBlock.load_weights(
            state_dict=state_dict, layer_path=layer_path,
            key_layer="conditioned_transition")

        m = cls(heads=pair_bias_attn.num_heads,
                dim=adaln.dim,
                dim_single_cond=adaln.dim_single_cond,
                dim_pairwise=pair_bias_attn.c_z)

        setattr(m, "adaln", adaln)
        setattr(m, "pair_bias_attn", pair_bias_attn)
        setattr(m, "transition", transition)

        extract_state_dict = {}
        key_layer = "attention_pair_bias"
        for key in state_dict.keys():
            if layer_path + '.{}'.format(key_layer) in key:
                if "linear_ada_out" in key:
                    name = key.replace(key_layer + ".linear_ada_out",
                                        "output_projection.0")
                    extract_state_dict[name] = state_dict[key]
        assert extract_state_dict != {}, "extract weights is empty"
        weights_biases_path = [
            (f"{layer_path}.output_projection.0.weight",
             f"{layer_path}.output_projection.0.bias"),
        ]
        layers = [
            m.output_projection[0],
        ]
        for (weights_path, bias_path), layer in zip(weights_biases_path,
                                                    layers):
            if bias_path is not None:
                layer.bias.data.copy_(extract_state_dict[bias_path])
            layer.weight.data.copy_(extract_state_dict[weights_path])
        return m
