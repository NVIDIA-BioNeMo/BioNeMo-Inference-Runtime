# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from bionemo_ir.configs import BaseConfig
from bionemo_ir.hubs import load_weights
from bionemo_ir.logger import logger
from bionemo_ir.models.boltz1.convert import (
    convert_hf_diffusion_transformer as boltz1_convert_hf_diffusion_transformer,
)
from bionemo_ir.models.boltz1.convert import (
    convert_hf_diffusion_transformer_torch as boltz1_convert_hf_diffusion_transformer_torch,
)
from bionemo_ir.models.boltz1.convert import convert_hf_pairformer_torch as boltz1_convert_hf_pairformer_torch
from bionemo_ir.models.boltz1.convert import (
    get_pairwise_attn_weights,
    get_transition_weights,
    get_tri_attn_node_weights,
    get_tri_mul_node_weights,
)
from bionemo_ir.utils import str_dtype_to_torch


def get_post_pre_norm_weights(
    state_dict: dict, prefix: str, bioir_prefix: str, post_layer_norm: bool = False, dtype: str = "float32"
):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    pre_norm_weight = state_dict[f"{prefix}.pre_norm_s.weight"]
    pre_norm_bias = state_dict[f"{prefix}.pre_norm_s.bias"]

    ret[f"{bioir_prefix}.pre_norm_s.weight"] = pre_norm_weight.to(torch_dtype)
    ret[f"{bioir_prefix}.pre_norm_s.bias"] = pre_norm_bias.to(torch_dtype)

    if post_layer_norm:
        post_norm_weight = state_dict[f"{prefix}.post_norm_s.weight"]
        post_norm_bias = state_dict[f"{prefix}.post_norm_s.bias"]
        ret[f"{bioir_prefix}.post_norm_s.weight"] = post_norm_weight.to(torch_dtype)
        ret[f"{bioir_prefix}.post_norm_s.bias"] = post_norm_bias.to(torch_dtype)
    return ret


def convert_hf_pairformer(
    config: BaseConfig, pairformer_type: str = "structure", local_checkpoint: str = None, model_name: str = "boltz-2"
):
    """
    Convert a pairformer model from a Hugging Face checkpoint into the
    weights of this project's Torch pairformer module.
    """
    prefix = "pairformer_module.layers"
    if pairformer_type == "confidence":
        prefix = "pairformer_stack.layers"
        prefix = f"confidence_module.{prefix}"
    bioir_prefix = "layers"
    weights = {}
    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)

    logger.info(
        f"Loading weights for {pairformer_type} pairformer, dtype: {config.dtype}, num_blocks: {config.num_blocks}"
    )

    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_bioir_prefix = f"{bioir_prefix}.{i}"
        weights.update(
            get_post_pre_norm_weights(
                state_dict, f"{layer_prefix}", f"{layer_bioir_prefix}", config.post_layer_norm, dtype="float32"
            )
        )
        weights.update(
            get_pairwise_attn_weights(
                state_dict,
                f"{layer_prefix}.attention",
                f"{layer_bioir_prefix}.attention",
                config.num_heads,
                attention_initial_norm=config.attention_initial_norm,
                dtype="float32",
            )
        )
        weights.update(
            get_tri_attn_node_weights(
                state_dict, f"{layer_prefix}.tri_att_start", f"{layer_bioir_prefix}.tri_attn_start", dtype=config.dtype
            )
        )
        weights.update(
            get_tri_attn_node_weights(
                state_dict, f"{layer_prefix}.tri_att_end", f"{layer_bioir_prefix}.tri_attn_end", dtype=config.dtype
            )
        )
        weights.update(
            get_tri_mul_node_weights(
                state_dict, f"{layer_prefix}.tri_mul_out", f"{layer_bioir_prefix}.tri_mul_out", dtype=config.dtype
            )
        )
        weights.update(
            get_tri_mul_node_weights(
                state_dict, f"{layer_prefix}.tri_mul_in", f"{layer_bioir_prefix}.tri_mul_in", dtype=config.dtype
            )
        )
        weights.update(
            get_transition_weights(
                state_dict,
                f"{layer_prefix}.transition_s",
                f"{layer_bioir_prefix}.transition_s",
                config.token_s * 4,
                dtype="float32",
            )
        )
        weights.update(
            get_transition_weights(
                state_dict,
                f"{layer_prefix}.transition_z",
                f"{layer_bioir_prefix}.transition_z",
                config.token_z * 4,
                dtype=config.dtype,
            )
        )
    return weights


def convert_hf_pairformer_torch(
    config: BaseConfig = None,
    local_checkpoint: str = None,
    model_name: str = "boltz-2",
    weights: dict = None,
    pairformer_type: str = "structure",
    **kwargs,
):
    """
    Load a pairformer model from a PyTorch checkpoint.

    Args:
        module: The module to load the weights into.
        checkpoint_dir: The directory to load the checkpoint from.
        weights: The weights to load into the module.
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "pairformer_module."
    if pairformer_type == "confidence":
        prefix = "pairformer_stack."
        prefix = f"confidence_module.{prefix}"

    bioir_state_dict = boltz1_convert_hf_pairformer_torch(
        config=config, local_checkpoint=None, weights=state_dict, pairformer_type=pairformer_type, prefix=prefix
    )

    for name in state_dict.keys():
        if name.startswith(prefix) and (".pre_norm_s" in name or ".post_norm_s" in name):
            name = name.replace(".weight", "")
            name = name.replace(".bias", "")
            k = name.replace(prefix, "")
            if k not in bioir_state_dict:
                bioir_state_dict[k] = [{"weight": state_dict[f"{name}.weight"]}]
                if f"{name}.bias" in state_dict:
                    bioir_state_dict[k][0]["bias"] = state_dict[f"{name}.bias"]
    return bioir_state_dict


def convert_hf_diffusion_transformer_torch(
    config: BaseConfig = None, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    bioir_state_dict = boltz1_convert_hf_diffusion_transformer_torch(
        config=config, local_checkpoint=None, weights=state_dict, **kwargs
    )
    return bioir_state_dict


def convert_hf_diffusion_transformer(*args, **kwargs):
    if kwargs.get("model_name") is None:
        kwargs["model_name"] = "boltz-2"
    return boltz1_convert_hf_diffusion_transformer(*args, **kwargs)


def get_pairwise_conditioner_weights(
    state_dict: dict, prefix: str, bioir_prefix: str, num_transitions: int, dtype: str = "float32"
):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    init_proj_norm_weight = state_dict[f"{prefix}.dim_pairwise_init_proj.0.weight"]
    init_proj_norm_bias = state_dict[f"{prefix}.dim_pairwise_init_proj.0.bias"]
    init_proj_linear_weight = state_dict[f"{prefix}.dim_pairwise_init_proj.1.weight"]

    ret.update(
        {
            f"{bioir_prefix}.init_proj_norm.weight": init_proj_norm_weight.to(torch_dtype),
            f"{bioir_prefix}.init_proj_norm.bias": init_proj_norm_bias.to(torch_dtype),
            f"{bioir_prefix}.init_proj_linear.weight": init_proj_linear_weight.to(torch_dtype),
        }
    )
    for i in range(num_transitions):
        transition_layer_prefix = f"{prefix}.transitions.{i}"
        transition_layer_bioir_prefix = f"{bioir_prefix}.transitions.{i}"
        ret.update(
            get_transition_weights(
                state_dict,
                transition_layer_prefix,
                transition_layer_bioir_prefix,
                True,
                init_proj_linear_weight.shape[0] * 4,
                dtype=dtype,
            )
        )
    return ret


def get_pairformer_no_seq_weights(
    state_dict: dict, prefix: str, bioir_prefix: str, token_z: int = 128, dtype: str = "float32"
):
    str_dtype_to_torch(dtype)
    weights = {}
    weights.update(
        get_tri_attn_node_weights(state_dict, f"{prefix}.tri_att_start", f"{bioir_prefix}.tri_attn_start", dtype=dtype)
    )
    weights.update(
        get_tri_attn_node_weights(state_dict, f"{prefix}.tri_att_end", f"{bioir_prefix}.tri_attn_end", dtype=dtype)
    )
    weights.update(
        get_tri_mul_node_weights(state_dict, f"{prefix}.tri_mul_out", f"{bioir_prefix}.tri_mul_out", dtype=dtype)
    )
    weights.update(
        get_tri_mul_node_weights(state_dict, f"{prefix}.tri_mul_in", f"{bioir_prefix}.tri_mul_in", dtype=dtype)
    )

    weights.update(
        get_transition_weights(
            state_dict, f"{prefix}.transition_z", f"{bioir_prefix}.transition_z", token_z * 4, dtype=dtype
        )
    )
    return weights


def convert_hf_affinity_module_torch(
    config: BaseConfig = None,
    local_checkpoint: str = None,
    model_name: str = "boltz-2-affinity",
    weights: dict = None,
    affinity_module_name: str = "affinity_module1",
    **kwargs,
):
    """
    Load a affinity module model from a PyTorch checkpoint.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = f"{affinity_module_name}."
    all_keys = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairformer_stack.layers")])
    keys_layer_0 = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairformer_stack.layers.0.")])
    pairformer_num_blocks = all_keys // keys_layer_0

    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    bioir_state_dict = {}
    bioir_state_dict["dist_bin_pairwise_embed"] = [{"weight": module_state_dict["dist_bin_pairwise_embed.weight"]}]
    bioir_state_dict["fused_s_to_z"] = [
        {"weight": module_state_dict["s_to_z_prod_in1.weight"]},
        {"weight": module_state_dict["s_to_z_prod_in2.weight"]},
    ]
    bioir_state_dict["z_norm"] = [
        {"weight": module_state_dict["z_norm.weight"], "bias": module_state_dict["z_norm.bias"]}
    ]
    bioir_state_dict["z_linear"] = [{"weight": module_state_dict["z_linear.weight"]}]

    # weight for pairwise conditioner
    all_keys = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairwise_conditioner.transitions")])
    keys_layer_0 = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairwise_conditioner.transitions.0.")])
    num_transitions = all_keys // keys_layer_0
    bioir_state_dict["pairwise_conditioner.init_proj_norm"] = [
        {
            "weight": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.0.weight"],
            "bias": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.0.bias"],
        }
    ]
    bioir_state_dict["pairwise_conditioner.init_proj_linear"] = [
        {"weight": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.1.weight"]}
    ]
    for i in range(num_transitions):
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.norm"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.norm.weight"],
                "bias": module_state_dict[f"pairwise_conditioner.transitions.{i}.norm.bias"],
            }
        ]
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.fused_fc2_fc1"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc2.weight"],
            },
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc1.weight"],
            },
        ]
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.fc3"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc3.weight"],
            }
        ]

    # weight for pairformer stack
    for i in range(pairformer_num_blocks):
        for name in ["tri_mul_out", "tri_mul_in"]:
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_in"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_in.weight"],
                    "bias": module_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_in.bias"],
                }
            ]

            w = module_state_dict[f"pairformer_stack.layers.{i}.{name}.p_in.weight"]
            p_in_0_weight, p_in_1_weight = w.chunk(2, dim=0)
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.p_in"] = [
                {
                    "weight": p_in_0_weight,
                },
                {
                    "weight": p_in_1_weight,
                },
            ]

            w = module_state_dict[f"pairformer_stack.layers.{i}.{name}.g_in.weight"]
            g_in_0_weight, g_in_1_weight = w.chunk(2, dim=0)
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.g_in"] = [
                {
                    "weight": g_in_0_weight,
                },
                {
                    "weight": g_in_1_weight,
                },
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_out"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_out.weight"],
                    "bias": module_state_dict[f"pairformer_stack.layers.{i}.{name}.norm_out.bias"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.p_out"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.{name}.p_out.weight"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.{name}.g_out"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.{name}.g_out.weight"],
                }
            ]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            bioir_state_dict[f"pairformer_stack.layers.{i}.tri_attn_{name}.layer_norm"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.layer_norm.weight"],
                    "bias": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.layer_norm.bias"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.tri_attn_{name}.linear"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.linear.weight"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.tri_attn_{name}.mha.qkv_proj"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.mha.linear_q.weight"],
                },
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.mha.linear_k.weight"],
                },
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.mha.linear_v.weight"],
                },
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.tri_attn_{name}.mha.o_proj"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.mha.linear_o.weight"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.tri_attn_{name}.mha.g_proj"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.tri_att_{name}.mha.linear_g.weight"],
                }
            ]
        # weight for transition_s and transition_z
        for name in ["z"]:
            bioir_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.norm"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.norm.weight"],
                    "bias": module_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.norm.bias"],
                }
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.fused_fc2_fc1"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.fc2.weight"],
                },
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.fc1.weight"],
                },
            ]
            bioir_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.fc3"] = [
                {
                    "weight": module_state_dict[f"pairformer_stack.layers.{i}.transition_{name}.fc3.weight"],
                }
            ]

    # weight for affinity heads
    bioir_state_dict["affinity_heads.affinity_out_mlp_linear_0"] = [
        {
            "weight": module_state_dict["affinity_heads.affinity_out_mlp.0.weight"],
            "bias": module_state_dict["affinity_heads.affinity_out_mlp.0.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.affinity_out_mlp_linear_1"] = [
        {
            "weight": module_state_dict["affinity_heads.affinity_out_mlp.2.weight"],
            "bias": module_state_dict["affinity_heads.affinity_out_mlp.2.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_value_0"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_value.0.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_value.0.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_value_1"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_value.2.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_value.2.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_value_2"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_value.4.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_value.4.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_score_0"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_score.0.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_score.0.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_score_1"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_score.2.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_score.2.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_pred_score_2"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_pred_score.4.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_pred_score.4.bias"],
        }
    ]
    bioir_state_dict["affinity_heads.to_affinity_logits_binary"] = [
        {
            "weight": module_state_dict["affinity_heads.to_affinity_logits_binary.weight"],
            "bias": module_state_dict["affinity_heads.to_affinity_logits_binary.bias"],
        }
    ]
    return bioir_state_dict


def convert_hf_msa_module_torch(
    config: BaseConfig = None, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    """
    Convert a msa module model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    prefix = "msa_module."
    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    bioir_state_dict = {}
    bioir_state_dict["s_proj"] = [
        {
            "weight": module_state_dict["s_proj.weight"],
        }
    ]
    bioir_state_dict["msa_proj"] = [
        {
            "weight": module_state_dict["msa_proj.weight"],
        }
    ]

    for i in range(config.msa_blocks):
        # Weight for msa transition
        bioir_state_dict[f"layers.{i}.msa_transition.norm"] = [
            {
                "weight": module_state_dict[f"layers.{i}.msa_transition.norm.weight"],
                "bias": module_state_dict[f"layers.{i}.msa_transition.norm.bias"],
            }
        ]
        bioir_state_dict[f"layers.{i}.msa_transition.fused_fc2_fc1"] = [
            {
                "weight": module_state_dict[f"layers.{i}.msa_transition.fc2.weight"],
            },
            {
                "weight": module_state_dict[f"layers.{i}.msa_transition.fc1.weight"],
            },
        ]
        bioir_state_dict[f"layers.{i}.msa_transition.fc3"] = [
            {
                "weight": module_state_dict[f"layers.{i}.msa_transition.fc3.weight"],
            }
        ]

        # Weight for pair_weighted_averaging
        bioir_state_dict[f"layers.{i}.pair_weighted_averaging.norm_m"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_m.weight"],
                "bias": module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_m.bias"],
            }
        ]
        bioir_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z.weight"],
                "bias": module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z.bias"],
            }
        ]
        bioir_state_dict[f"layers.{i}.pair_weighted_averaging.fused_proj_m_g"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.proj_m.weight"],
            },
            {
                "weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.proj_g.weight"],
            },
        ]
        bioir_state_dict[f"layers.{i}.pair_weighted_averaging.proj_z"] = [
            {"weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.proj_z.weight"]}
        ]
        bioir_state_dict[f"layers.{i}.pair_weighted_averaging.proj_o"] = [
            {"weight": module_state_dict[f"layers.{i}.pair_weighted_averaging.proj_o.weight"]}
        ]

        # Weight for pairformer_layer
        for name in ["tri_mul_out", "tri_mul_in"]:
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_in"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_in.weight"],
                    "bias": module_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_in.bias"],
                }
            ]

            w = module_state_dict[f"layers.{i}.pairformer_layer.{name}.p_in.weight"]
            p_in_0_weight, p_in_1_weight = w.chunk(2, dim=0)
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.p_in"] = [
                {
                    "weight": p_in_0_weight,
                },
                {
                    "weight": p_in_1_weight,
                },
            ]

            w = module_state_dict[f"layers.{i}.pairformer_layer.{name}.g_in.weight"]
            g_in_0_weight, g_in_1_weight = w.chunk(2, dim=0)
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.g_in"] = [
                {
                    "weight": g_in_0_weight,
                },
                {
                    "weight": g_in_1_weight,
                },
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_out"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_out.weight"],
                    "bias": module_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_out.bias"],
                }
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.p_out"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.{name}.p_out.weight"],
                }
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.{name}.g_out"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.{name}.g_out.weight"],
                }
            ]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            bioir_state_dict[f"layers.{i}.pairformer_layer.tri_attn_{name}.layer_norm"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.layer_norm.weight"],
                    "bias": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.layer_norm.bias"],
                }
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.tri_attn_{name}.linear"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.linear.weight"],
                }
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.qkv_proj"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.mha.linear_q.weight"],
                },
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.mha.linear_k.weight"],
                },
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.mha.linear_v.weight"],
                },
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.o_proj"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.mha.linear_o.weight"],
                }
            ]
            bioir_state_dict[f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.g_proj"] = [
                {
                    "weight": module_state_dict[f"layers.{i}.pairformer_layer.tri_att_{name}.mha.linear_g.weight"],
                }
            ]
        # weight for transition_z
        bioir_state_dict[f"layers.{i}.pairformer_layer.transition_z.norm"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pairformer_layer.transition_z.norm.weight"],
                "bias": module_state_dict[f"layers.{i}.pairformer_layer.transition_z.norm.bias"],
            }
        ]
        bioir_state_dict[f"layers.{i}.pairformer_layer.transition_z.fused_fc2_fc1"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pairformer_layer.transition_z.fc2.weight"],
            },
            {
                "weight": module_state_dict[f"layers.{i}.pairformer_layer.transition_z.fc1.weight"],
            },
        ]
        bioir_state_dict[f"layers.{i}.pairformer_layer.transition_z.fc3"] = [
            {
                "weight": module_state_dict[f"layers.{i}.pairformer_layer.transition_z.fc3.weight"],
            }
        ]

        # weight for outer_product_mean
        bioir_state_dict[f"layers.{i}.outer_product_mean.norm"] = [
            {
                "weight": module_state_dict[f"layers.{i}.outer_product_mean.norm.weight"],
                "bias": module_state_dict[f"layers.{i}.outer_product_mean.norm.bias"],
            }
        ]
        bioir_state_dict[f"layers.{i}.outer_product_mean.fused_proj_a_b"] = [
            {
                "weight": module_state_dict[f"layers.{i}.outer_product_mean.proj_a.weight"],
            },
            {
                "weight": module_state_dict[f"layers.{i}.outer_product_mean.proj_b.weight"],
            },
        ]
        bioir_state_dict[f"layers.{i}.outer_product_mean.proj_o"] = [
            {
                "weight": module_state_dict[f"layers.{i}.outer_product_mean.proj_o.weight"],
                "bias": module_state_dict[f"layers.{i}.outer_product_mean.proj_o.bias"],
            }
        ]
    return bioir_state_dict


def _convert_pairformer_no_seq_block_torch(
    module_state_dict: dict, block_prefix: str, bioir_prefix: str, out_dict: dict
) -> None:
    """Convert a single ``PairformerNoSeqLayer`` block of HF weights into the
    BioIR dict-of-lists layout consumed by
    :func:`recursive_calling_load_weights`.

    Args:
        module_state_dict: Flat HF state-dict already stripped of any outer
            prefix (e.g. ``"template_module."``).
        block_prefix: HF prefix for this block (e.g. ``"pairformer.layers.0"``).
        bioir_prefix: BioIR target prefix (typically the same as
            ``block_prefix``).
        out_dict: Output dict, mutated in place.
    """
    for name in ["tri_mul_out", "tri_mul_in"]:
        out_dict[f"{bioir_prefix}.{name}.norm_in"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.{name}.norm_in.weight"],
                "bias": module_state_dict[f"{block_prefix}.{name}.norm_in.bias"],
            }
        ]

        w = module_state_dict[f"{block_prefix}.{name}.p_in.weight"]
        p_in_0_weight, p_in_1_weight = w.chunk(2, dim=0)
        out_dict[f"{bioir_prefix}.{name}.p_in"] = [
            {"weight": p_in_0_weight},
            {"weight": p_in_1_weight},
        ]

        w = module_state_dict[f"{block_prefix}.{name}.g_in.weight"]
        g_in_0_weight, g_in_1_weight = w.chunk(2, dim=0)
        out_dict[f"{bioir_prefix}.{name}.g_in"] = [
            {"weight": g_in_0_weight},
            {"weight": g_in_1_weight},
        ]
        out_dict[f"{bioir_prefix}.{name}.norm_out"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.{name}.norm_out.weight"],
                "bias": module_state_dict[f"{block_prefix}.{name}.norm_out.bias"],
            }
        ]
        out_dict[f"{bioir_prefix}.{name}.p_out"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.{name}.p_out.weight"],
            }
        ]
        out_dict[f"{bioir_prefix}.{name}.g_out"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.{name}.g_out.weight"],
            }
        ]

    for name in ["start", "end"]:
        out_dict[f"{bioir_prefix}.tri_attn_{name}.layer_norm"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.tri_att_{name}.layer_norm.weight"],
                "bias": module_state_dict[f"{block_prefix}.tri_att_{name}.layer_norm.bias"],
            }
        ]
        out_dict[f"{bioir_prefix}.tri_attn_{name}.linear"] = [
            {
                "weight": module_state_dict[f"{block_prefix}.tri_att_{name}.linear.weight"],
            }
        ]
        out_dict[f"{bioir_prefix}.tri_attn_{name}.mha.qkv_proj"] = [
            {"weight": module_state_dict[f"{block_prefix}.tri_att_{name}.mha.linear_q.weight"]},
            {"weight": module_state_dict[f"{block_prefix}.tri_att_{name}.mha.linear_k.weight"]},
            {"weight": module_state_dict[f"{block_prefix}.tri_att_{name}.mha.linear_v.weight"]},
        ]
        out_dict[f"{bioir_prefix}.tri_attn_{name}.mha.o_proj"] = [
            {"weight": module_state_dict[f"{block_prefix}.tri_att_{name}.mha.linear_o.weight"]}
        ]
        out_dict[f"{bioir_prefix}.tri_attn_{name}.mha.g_proj"] = [
            {"weight": module_state_dict[f"{block_prefix}.tri_att_{name}.mha.linear_g.weight"]}
        ]

    out_dict[f"{bioir_prefix}.transition_z.norm"] = [
        {
            "weight": module_state_dict[f"{block_prefix}.transition_z.norm.weight"],
            "bias": module_state_dict[f"{block_prefix}.transition_z.norm.bias"],
        }
    ]
    out_dict[f"{bioir_prefix}.transition_z.fused_fc2_fc1"] = [
        {"weight": module_state_dict[f"{block_prefix}.transition_z.fc2.weight"]},
        {"weight": module_state_dict[f"{block_prefix}.transition_z.fc1.weight"]},
    ]
    out_dict[f"{bioir_prefix}.transition_z.fc3"] = [
        {"weight": module_state_dict[f"{block_prefix}.transition_z.fc3.weight"]}
    ]


def convert_hf_template_module_torch(
    config: BaseConfig = None, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    """Convert ``template_module.*`` weights from an HF Boltz-2 checkpoint
    into the BioIR ``TemplateV2Module`` dict-of-lists layout used by
    :func:`recursive_calling_load_weights`.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "template_module."
    module_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}

    bioir_state_dict = {}
    bioir_state_dict["z_norm"] = [
        {
            "weight": module_state_dict["z_norm.weight"],
            "bias": module_state_dict["z_norm.bias"],
        }
    ]
    bioir_state_dict["v_norm"] = [
        {
            "weight": module_state_dict["v_norm.weight"],
            "bias": module_state_dict["v_norm.bias"],
        }
    ]
    bioir_state_dict["z_proj"] = [{"weight": module_state_dict["z_proj.weight"]}]
    bioir_state_dict["a_proj"] = [{"weight": module_state_dict["a_proj.weight"]}]
    bioir_state_dict["u_proj"] = [{"weight": module_state_dict["u_proj.weight"]}]

    num_blocks = config.template_blocks
    for i in range(num_blocks):
        _convert_pairformer_no_seq_block_torch(
            module_state_dict,
            block_prefix=f"pairformer.layers.{i}",
            bioir_prefix=f"pairformer.layers.{i}",
            out_dict=bioir_state_dict,
        )
    return bioir_state_dict


def convert_hf_input_embedder_torch(
    config: BaseConfig, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    """
    Convert a Boltz2 input embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    weights = {}
    layer_path = "input_embedder"
    embedding_layer_path = f"{layer_path}.atom_encoder"
    transformer_layer_path = f"{layer_path}.atom_attention_encoder"

    atom_transformer_weights = convert_hf_diffusion_transformer_torch(
        config.diffusion_transformer,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{transformer_layer_path}.atom_encoder.diffusion_transformer.",
    )

    for k, v in atom_transformer_weights.items():
        weights[f"atom_attention_encoder.atom_encoder.diffusion_transformer.{k}"] = v

    weights["atom_attention_encoder.atom_to_token_trans.0"] = [
        {"weight": state_dict[f"{transformer_layer_path}.atom_to_token_trans.0.weight"], "bias": None}
    ]

    weights_biases_path = {
        "embed_atom_features": (
            f"{embedding_layer_path}.embed_atom_features.weight",
            f"{embedding_layer_path}.embed_atom_features.bias",
        ),  # bias is present for Boltz2
        "embed_atompair_ref_pos": (f"{embedding_layer_path}.embed_atompair_ref_pos.weight", None),
        "embed_atompair_ref_dist": (f"{embedding_layer_path}.embed_atompair_ref_dist.weight", None),
        "embed_atompair_mask": (f"{embedding_layer_path}.embed_atompair_mask.weight", None),
        "c_to_p_trans_k.1": (f"{embedding_layer_path}.c_to_p_trans_k.1.weight", None),
        "c_to_p_trans_q.1": (f"{embedding_layer_path}.c_to_p_trans_q.1.weight", None),
        "p_mlp.1": (f"{embedding_layer_path}.p_mlp.1.weight", None),
        "p_mlp.3": (f"{embedding_layer_path}.p_mlp.3.weight", None),
        "p_mlp.5": (f"{embedding_layer_path}.p_mlp.5.weight", None),
    }

    for name, (weights_path, bias_path) in weights_biases_path.items():
        weights[f"atom_embedding.{name}"] = [
            {"weight": state_dict[weights_path], "bias": state_dict[bias_path] if bias_path is not None else None}
        ]

    weights["atom_enc_proj_z.0"] = [
        {
            "weight": state_dict[f"{layer_path}.atom_enc_proj_z.0.weight"],
            "bias": state_dict[f"{layer_path}.atom_enc_proj_z.0.bias"],
        }
    ]
    weights["atom_enc_proj_z.1"] = [
        {
            "weight": state_dict[f"{layer_path}.atom_enc_proj_z.1.weight"],
            "bias": None,
        }
    ]

    weights["res_type_encoding"] = [
        {
            "weight": state_dict[f"{layer_path}.res_type_encoding.weight"],
            "bias": None,
        }
    ]
    weights["msa_profile_encoding"] = [
        {
            "weight": state_dict[f"{layer_path}.msa_profile_encoding.weight"],
            "bias": None,
        }
    ]

    if config.add_method_conditioning:
        weights["method_conditioning_init"] = [
            {
                "weight": state_dict[f"{layer_path}.method_conditioning_init.weight"],
                "bias": None,
            }
        ]
    if config.add_modified_flag:
        weights["modified_conditioning_init"] = [
            {
                "weight": state_dict[f"{layer_path}.modified_conditioning_init.weight"],
                "bias": None,
            }
        ]
    if config.add_cyclic_flag:
        weights["cyclic_conditioning_init"] = [
            {
                "weight": state_dict[f"{layer_path}.cyclic_conditioning_init.weight"],
                "bias": None,
            }
        ]
    if config.add_mol_type_feat:
        weights["mol_type_conditioning_init"] = [
            {
                "weight": state_dict[f"{layer_path}.mol_type_conditioning_init.weight"],
                "bias": None,
            }
        ]
    return weights


def convert_hf_structure_module_torch(
    config: BaseConfig, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    """
    Convert a structure module model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    weights = {}

    # Convert for score model
    score_model_config = config.score_model
    layer_path = "structure_module.score_model"

    # Load for single conditioner, s_to_a_linear and a_norm
    weights["score_model.single_conditioner.norm_single"] = [
        {
            "weight": state_dict[f"{layer_path}.single_conditioner.norm_single.weight"],
            "bias": state_dict[f"{layer_path}.single_conditioner.norm_single.bias"],
        }
    ]

    weights["score_model.single_conditioner.single_embed"] = [
        {
            "weight": state_dict[f"{layer_path}.single_conditioner.single_embed.weight"],
            "bias": state_dict.get(f"{layer_path}.single_conditioner.single_embed.bias", None),
        }
    ]
    weights["score_model.single_conditioner.fourier_embed.proj"] = [
        {
            "weight": state_dict[f"{layer_path}.single_conditioner.fourier_embed.proj.weight"],
            "bias": state_dict.get(f"{layer_path}.single_conditioner.fourier_embed.proj.bias", None),
        }
    ]
    weights["score_model.single_conditioner.norm_fourier"] = [
        {
            "weight": state_dict[f"{layer_path}.single_conditioner.norm_fourier.weight"],
            "bias": state_dict[f"{layer_path}.single_conditioner.norm_fourier.bias"],
        }
    ]
    weights["score_model.single_conditioner.fourier_to_single"] = [
        {
            "weight": state_dict[f"{layer_path}.single_conditioner.fourier_to_single.weight"],
            "bias": state_dict.get(f"{layer_path}.single_conditioner.fourier_to_single.bias", None),
        }
    ]
    for i in range(score_model_config.conditioning_transition_layers):
        weights[f"score_model.single_conditioner.transitions.{i}.norm"] = [
            {
                "weight": state_dict[f"{layer_path}.single_conditioner.transitions.{i}.norm.weight"],
                "bias": state_dict[f"{layer_path}.single_conditioner.transitions.{i}.norm.bias"],
            }
        ]
        weights[f"score_model.single_conditioner.transitions.{i}.fused_fc2_fc1"] = [
            {
                "weight": state_dict[f"{layer_path}.single_conditioner.transitions.{i}.fc2.weight"],
                "bias": state_dict.get(f"{layer_path}.single_conditioner.transitions.{i}.fc2.bias", None),
            },
            {
                "weight": state_dict[f"{layer_path}.single_conditioner.transitions.{i}.fc1.weight"],
                "bias": state_dict.get(f"{layer_path}.single_conditioner.transitions.{i}.fc1.bias", None),
            },
        ]
        weights[f"score_model.single_conditioner.transitions.{i}.fc3"] = [
            {
                "weight": state_dict[f"{layer_path}.single_conditioner.transitions.{i}.fc3.weight"],
            }
        ]
    weights["score_model.s_to_a_linear.0"] = [
        {
            "weight": state_dict[f"{layer_path}.s_to_a_linear.0.weight"],
            "bias": state_dict[f"{layer_path}.s_to_a_linear.0.bias"],
        }
    ]
    weights["score_model.s_to_a_linear.1"] = [
        {"weight": state_dict[f"{layer_path}.s_to_a_linear.1.weight"], "bias": None}
    ]
    weights["score_model.a_norm"] = [
        {"weight": state_dict[f"{layer_path}.a_norm.weight"], "bias": state_dict[f"{layer_path}.a_norm.bias"]}
    ]

    # Load for atom attention encoder
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.atom_encoder,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{layer_path}.atom_attention_encoder.atom_encoder.diffusion_transformer.",
    )
    for k, v in DiT_weights.items():
        weights[f"score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.{k}"] = v

    weights["score_model.atom_attention_encoder.atom_to_token_trans.0"] = [
        {"weight": state_dict[f"{layer_path}.atom_attention_encoder.atom_to_token_trans.0.weight"], "bias": None}
    ]
    weights["score_model.atom_attention_encoder.r_to_q_trans"] = [
        {"weight": state_dict[f"{layer_path}.atom_attention_encoder.r_to_q_trans.weight"], "bias": None}
    ]
    # Load for atom attention decoder
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.atom_decoder,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{layer_path}.atom_attention_decoder.atom_decoder.diffusion_transformer.",
    )
    for k, v in DiT_weights.items():
        weights[f"score_model.atom_attention_decoder.atom_decoder.diffusion_transformer.{k}"] = v

    weights["score_model.atom_attention_decoder.a_to_q_trans"] = [
        {"weight": state_dict[f"{layer_path}.atom_attention_decoder.a_to_q_trans.weight"], "bias": None}
    ]
    weights["score_model.atom_attention_decoder.atom_feat_to_atom_pos_update.0"] = [
        {
            "weight": state_dict[f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.weight"],
            "bias": state_dict[f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.bias"],
        }
    ]
    weights["score_model.atom_attention_decoder.atom_feat_to_atom_pos_update.1"] = [
        {
            "weight": state_dict[f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.1.weight"],
            "bias": None,
        }
    ]
    # Load for token transformer
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.token_transformer,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{layer_path}.token_transformer.",
    )

    for k, v in DiT_weights.items():
        weights[f"score_model.token_transformer.{k}"] = v

    return weights


def convert_hf_diffusion_conditioning_torch(
    config: BaseConfig, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    bioir_state_dict = {}
    prefix = "diffusion_conditioning."
    module_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v
    all_keys = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairwise_conditioner.transitions")])
    keys_layer_0 = len([k for k in state_dict.keys() if k.startswith(f"{prefix}pairwise_conditioner.transitions.0.")])

    num_transitions = all_keys // keys_layer_0
    bioir_state_dict["pairwise_conditioner.init_proj_norm"] = [
        {
            "weight": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.0.weight"],
            "bias": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.0.bias"],
        }
    ]
    bioir_state_dict["pairwise_conditioner.init_proj_linear"] = [
        {"weight": module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.1.weight"]}
    ]
    for i in range(num_transitions):
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.norm"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.norm.weight"],
                "bias": module_state_dict[f"pairwise_conditioner.transitions.{i}.norm.bias"],
            }
        ]
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.fused_fc2_fc1"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc2.weight"],
            },
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc1.weight"],
            },
        ]
        bioir_state_dict[f"pairwise_conditioner.transitions.{i}.fc3"] = [
            {
                "weight": module_state_dict[f"pairwise_conditioner.transitions.{i}.fc3.weight"],
            }
        ]

    layer_path = "atom_encoder"
    weights_biases_path = {
        "embed_atom_features": (f"{layer_path}.embed_atom_features.weight", f"{layer_path}.embed_atom_features.bias"),
        "embed_atompair_ref_pos": (f"{layer_path}.embed_atompair_ref_pos.weight", None),
        "embed_atompair_ref_dist": (f"{layer_path}.embed_atompair_ref_dist.weight", None),
        "embed_atompair_mask": (f"{layer_path}.embed_atompair_mask.weight", None),
        "c_to_p_trans_k.1": (f"{layer_path}.c_to_p_trans_k.1.weight", None),
        "c_to_p_trans_q.1": (f"{layer_path}.c_to_p_trans_q.1.weight", None),
        "p_mlp.1": (f"{layer_path}.p_mlp.1.weight", None),
        "p_mlp.3": (f"{layer_path}.p_mlp.3.weight", None),
        "p_mlp.5": (f"{layer_path}.p_mlp.5.weight", None),
        "s_to_c_trans.0": (f"{layer_path}.s_to_c_trans.0.weight", f"{layer_path}.s_to_c_trans.0.bias"),
        "z_to_p_trans.0": (f"{layer_path}.z_to_p_trans.0.weight", f"{layer_path}.z_to_p_trans.0.bias"),
        "s_to_c_trans.1": (f"{layer_path}.s_to_c_trans.1.weight", None),
        "z_to_p_trans.1": (f"{layer_path}.z_to_p_trans.1.weight", None),
    }
    for name, (weights_path, bias_path) in weights_biases_path.items():
        bioir_state_dict[f"atom_embedding.{name}"] = [
            {
                "weight": module_state_dict[weights_path],
                "bias": module_state_dict[bias_path] if bias_path is not None else None,
            }
        ]

    # Add weights for computing biases
    layer_path = "atom_enc_proj_z"
    for i in range(config.atom_encoder.num_blocks):
        bioir_state_dict[f"atom_enc_proj_z.{i}.0"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.0.weight"],
                "bias": module_state_dict[f"{layer_path}.{i}.0.bias"],
            }
        ]
        bioir_state_dict[f"atom_enc_proj_z.{i}.1"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.1.weight"],
                "bias": None,
            }
        ]

    layer_path = "atom_dec_proj_z"
    for i in range(config.atom_decoder.num_blocks):
        bioir_state_dict[f"atom_dec_proj_z.{i}.0"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.0.weight"],
                "bias": module_state_dict[f"{layer_path}.{i}.0.bias"],
            }
        ]
        bioir_state_dict[f"atom_dec_proj_z.{i}.1"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.1.weight"],
                "bias": None,
            }
        ]

    layer_path = "token_trans_proj_z"
    for i in range(config.token_transformer.num_blocks):
        bioir_state_dict[f"token_trans_proj_z.{i}.0"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.0.weight"],
                "bias": module_state_dict[f"{layer_path}.{i}.0.bias"],
            }
        ]
        bioir_state_dict[f"token_trans_proj_z.{i}.1"] = [
            {
                "weight": module_state_dict[f"{layer_path}.{i}.1.weight"],
                "bias": None,
            }
        ]
    return bioir_state_dict


def convert_hf_confidence_module_torch(
    config: BaseConfig, local_checkpoint: str = None, model_name: str = "boltz-2", weights: dict = None, **kwargs
):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "confidence_module."
    extracted_weights = {}

    dist_bin_pairwise_embed_weights = [{"weight": state_dict[f"{prefix}dist_bin_pairwise_embed.weight"]}]

    s_to_z_weights = [
        {"weight": state_dict[f"{prefix}s_to_z.weight"], "bias": state_dict.get(f"{prefix}s_to_z.bias", None)}
    ]
    s_to_z_transpose_weights = [
        {
            "weight": state_dict[f"{prefix}s_to_z_transpose.weight"],
            "bias": state_dict.get(f"{prefix}s_to_z_transpose.bias", None),
        }
    ]
    s_to_z_prod_in1_weights = [
        {
            "weight": state_dict[f"{prefix}s_to_z_prod_in1.weight"],
            "bias": state_dict.get(f"{prefix}s_to_z_prod_in1.bias", None),
        }
    ]
    s_to_z_prod_in2_weights = [
        {
            "weight": state_dict[f"{prefix}s_to_z_prod_in2.weight"],
            "bias": state_dict.get(f"{prefix}s_to_z_prod_in2.bias", None),
        }
    ]
    s_to_z_prod_out_weights = [
        {
            "weight": state_dict[f"{prefix}s_to_z_prod_out.weight"],
            "bias": state_dict.get(f"{prefix}s_to_z_prod_out.bias", None),
        }
    ]
    s_to_z_prod_out_weights = [
        {
            "weight": state_dict[f"{prefix}s_to_z_prod_out.weight"],
            "bias": state_dict.get(f"{prefix}s_to_z_prod_out.bias", None),
        }
    ]

    s_inputs_norm_weights = [
        {
            "weight": state_dict[f"{prefix}s_inputs_norm.weight"],
            "bias": state_dict.get(f"{prefix}s_inputs_norm.bias", None),
        }
    ]
    s_norm_weights = [
        {"weight": state_dict[f"{prefix}s_norm.weight"], "bias": state_dict.get(f"{prefix}s_norm.bias", None)}
    ]
    z_norm_weights = [
        {"weight": state_dict[f"{prefix}z_norm.weight"], "bias": state_dict.get(f"{prefix}z_norm.bias", None)}
    ]

    s_input_to_s_weights = [
        {
            "weight": state_dict[f"{prefix}s_input_to_s.weight"],
            "bias": state_dict.get(f"{prefix}s_input_to_s.bias", None),
        }
    ]

    token_bonds_weights = [
        {"weight": state_dict[f"{prefix}token_bonds.weight"], "bias": state_dict.get(f"{prefix}token_bonds.bias", None)}
    ]

    token_bonds_type_weights = [{"weight": state_dict[f"{prefix}token_bonds_type.weight"]}]

    relative_position_encoder_weights = [
        {
            "weight": state_dict[f"{prefix}rel_pos.linear_layer.weight"],
            "bias": state_dict.get(f"{prefix}rel_pos.linear_layer.bias", None),
        }
    ]
    contact_conditioning_weights = {
        "fourier_embedding.proj": [
            {
                "weight": state_dict[f"{prefix}contact_conditioning.fourier_embedding.proj.weight"],
                "bias": state_dict[f"{prefix}contact_conditioning.fourier_embedding.proj.bias"],
            }
        ],
        "encoder": [
            {
                "weight": state_dict[f"{prefix}contact_conditioning.encoder.weight"],
                "bias": state_dict[f"{prefix}contact_conditioning.encoder.bias"],
            }
        ],
    }

    pairformer_weights = convert_hf_pairformer_torch(
        config.pairformer, model_name=model_name, pairformer_type="confidence"
    )

    confidence_heads_weights = {}
    confidence_prefix = "confidence_module.confidence_heads."
    for key in state_dict.keys():
        if "confidence_heads" in key:
            confidence_heads_weights[key.replace(confidence_prefix, "").replace(".weight", "")] = [
                {"weight": state_dict[key], "bias": None}
            ]

    extracted_weights["dist_bin_pairwise_embed"] = dist_bin_pairwise_embed_weights
    extracted_weights["s_to_z"] = s_to_z_weights
    extracted_weights["s_to_z_transpose"] = s_to_z_transpose_weights
    extracted_weights["s_to_z_prod_in1"] = s_to_z_prod_in1_weights
    extracted_weights["s_to_z_prod_in2"] = s_to_z_prod_in2_weights
    extracted_weights["s_to_z_prod_out"] = s_to_z_prod_out_weights
    extracted_weights["s_inputs_norm"] = s_inputs_norm_weights
    extracted_weights["s_norm"] = s_norm_weights
    extracted_weights["z_norm"] = z_norm_weights
    extracted_weights["s_input_to_s"] = s_input_to_s_weights
    extracted_weights["rel_pos.linear"] = relative_position_encoder_weights
    extracted_weights["token_bonds"] = token_bonds_weights
    extracted_weights["token_bonds_type"] = token_bonds_type_weights
    # This because the contact conditioning has nn.Parameter,
    # so we need to convert it to a dict for encoding_unspecified and encoding_unselected
    extracted_weights["contact_conditioning"] = [
        {
            "encoding_unspecified": state_dict[f"{prefix}contact_conditioning.encoding_unspecified"],
            "encoding_unselected": state_dict[f"{prefix}contact_conditioning.encoding_unselected"],
        }
    ]
    for k, v in contact_conditioning_weights.items():
        extracted_weights[f"contact_conditioning.{k}"] = v
    for k, v in pairformer_weights.items():
        extracted_weights[f"pairformer_stack.{k}"] = v
    for k, v in confidence_heads_weights.items():
        extracted_weights[f"confidence_heads.{k}"] = v

    return extracted_weights
