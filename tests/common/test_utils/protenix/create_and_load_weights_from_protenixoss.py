# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert OSS Protenix atom-transformer / atom-encoder weights to BioIR.

The BioIR ``ProtenixDiffusionTransformer`` reuses fused primitives (``AdaLN``,
``ConditionedTransitionBlock``), so parameter layouts differ from OSS and a
name/shape mapping is required. This mirrors OF3's
``create_and_load_weights_from_of3oss.py``.
"""

from __future__ import annotations

import torch


def _convert_adaln(oss_adaln, trt_adaln) -> None:
    """OSS ``AdaptiveLayerNorm`` -> BioIR ``AdaLN``.

    ``sigmoid(s_scale) * LN(a) + s_bias`` with
    ``s_scale = linear_s(LN_s(s))``, ``s_bias = linear_nobias_s(LN_s(s))``.
    BioIR fuses ``[s_scale; s_bias]`` into one Linear with a zero bias slot
    for the s_bias half.
    """
    trt_adaln.s_norm.weight.data.copy_(oss_adaln.layernorm_s.weight.data)
    fused_w = torch.cat([oss_adaln.linear_s.weight.data, oss_adaln.linear_nobias_s.weight.data], dim=0)
    fused_b = torch.cat([oss_adaln.linear_s.bias.data, torch.zeros_like(oss_adaln.linear_s.bias.data)], dim=0)
    trt_adaln.fused_s_scale_s_bias.weight.data.copy_(fused_w)
    trt_adaln.fused_s_scale_s_bias.bias.data.copy_(fused_b)


def _convert_block(oss_block, trt_layer) -> None:
    """OSS ``DiffusionTransformerBlock`` -> shared ``DiffusionTransformerLayer``.

    The Protenix atom transformer now reuses the shared DiT layer, so the
    Protenix ``AttentionPairBias`` maps onto ``pair_bias_attn`` (separate
    q/kv AdaLNs, fused KV, ``proj_z`` = LayerNorm + Linear), the AdaLN-zero
    output gate (OSS ``linear_a_last``) maps onto the layer ``output_projection``,
    and the conditioned transition onto ``transition``.
    """
    oa, ta = oss_block.attention_pair_bias, trt_layer.pair_bias_attn
    _convert_adaln(oa.layernorm_a, ta.layer_norm_a_q)
    _convert_adaln(oa.layernorm_kv, ta.layer_norm_a_k)
    ta.proj_q.weight.data.copy_(oa.attention.linear_q.weight.data)
    ta.proj_q.bias.data.copy_(oa.attention.linear_q.bias.data)
    # Fused KV: [k; v].
    ta.proj_kv.weight.data.copy_(
        torch.cat([oa.attention.linear_k.weight.data, oa.attention.linear_v.weight.data], dim=0)
    )
    ta.proj_g.weight.data.copy_(oa.attention.linear_g.weight.data)
    ta.proj_o.weight.data.copy_(oa.attention.linear_o.weight.data)
    ta.proj_z[0].weight.data.copy_(oa.layernorm_z.weight.data)
    ta.proj_z[1].weight.data.copy_(oa.linear_nobias_z.weight.data)

    # AdaLN-zero output gate: OSS linear_a_last -> layer output_projection.
    trt_layer.output_projection.weight.data.copy_(oa.linear_a_last.weight.data)
    trt_layer.output_projection.bias.data.copy_(oa.linear_a_last.bias.data)

    oc, tc = oss_block.conditioned_transition_block, trt_layer.transition
    _convert_adaln(oc.adaln, tc.adaln)
    # FusedSwiGLU 2-way packs z[:d]=value, z[d:2d]=gate; OSS computes
    # silu(a1)*a2 -> gate=a1, value=a2 -> pack [a2, a1].
    tc.fused_swl_a_to_b.weight.data.copy_(
        torch.cat([oc.linear_nobias_a2.weight.data, oc.linear_nobias_a1.weight.data], dim=0)
    )
    tc.b_to_a.weight.data.copy_(oc.linear_nobias_b.weight.data)
    tc.output_projection.weight.data.copy_(oc.linear_s.weight.data)
    tc.output_projection.bias.data.copy_(oc.linear_s.bias.data)


def convert_atom_transformer(oss_atom_transformer, trt_atom_transformer) -> None:
    """Copy OSS ``AtomTransformer`` weights into ``ProtenixDiffusionTransformer``."""
    oss_blocks = oss_atom_transformer.diffusion_transformer.blocks
    trt_layers = trt_atom_transformer.layers
    assert len(oss_blocks) == len(trt_layers), f"block count mismatch: {len(oss_blocks)} vs {len(trt_layers)}"
    for ob, tl in zip(oss_blocks, trt_layers, strict=True):
        _convert_block(ob, tl)


def convert_atom_attention_encoder(oss, trt) -> None:
    """Copy OSS ``AtomAttentionEncoder(has_coords=False)`` weights into the
    BioIR ``ProtenixAtomAttentionEncoder``.
    """
    trt.linear_no_bias_ref.weight.data.copy_(
        torch.cat(
            [
                oss.linear_no_bias_ref_pos.weight.data,
                oss.linear_no_bias_ref_charge.weight.data,
                oss.linear_no_bias_f.weight.data,
            ],
            dim=-1,
        )
    )
    trt.linear_no_bias_pair.weight.data.copy_(
        torch.cat(
            [
                oss.linear_no_bias_d.weight.data,
                oss.linear_no_bias_invd.weight.data,
                oss.linear_no_bias_v.weight.data,
            ],
            dim=-1,
        )
    )
    for name in ("linear_no_bias_cl", "linear_no_bias_cm", "linear_no_bias_q"):
        getattr(trt, name).weight.data.copy_(getattr(oss, name).weight.data)
    for i in (1, 3, 5):
        trt.small_mlp[i].weight.data.copy_(oss.small_mlp[i].weight.data)
    convert_atom_transformer(oss.atom_transformer, trt.atom_transformer)
