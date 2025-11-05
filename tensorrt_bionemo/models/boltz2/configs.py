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

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from tensorrt_llm import str_dtype_to_trt
from transformers import PretrainedConfig

from tensorrt_bionemo.config import (BuildModuleConfig, DimSpec,
                                     PretrainedModuleConfig)
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.models.boltz1.configs import (
    DiffusionTransformerConfig, InputEmbedderConfig, MSAModuleConfig,
    PairformerConfig, _create_optimization_profiles)
from tensorrt_bionemo.models.boltz1.const import TOKENS


class AffinityModuleConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 token_s: int = 384,
                 token_z: int = 128,
                 num_dist_bins: int = 64,
                 max_dist: int = 22,
                 pairformer_num_blocks: int = 8,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 triangle_attn_backend: str = 'VANILLA',
                 max_batch_size: int = 1,
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 **kwargs):
        super().__init__(**kwargs)
        self.token_s = token_s
        self.token_z = token_z
        self.num_dist_bins = num_dist_bins
        self.max_dist = max_dist
        self.pairformer_num_blocks = pairformer_num_blocks
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.triangle_attn_backend = triangle_attn_backend
        self.eps = eps
        self.inf = inf
        self.max_batch_size = max_batch_size

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_dtypes(self) -> dict[str, str]:
        return {
            "s": str_dtype_to_trt(self.dtype),
            "z": str_dtype_to_trt(self.dtype),
            "distogram": str_dtype_to_trt("int32"),
            "cross_pair_mask_0": str_dtype_to_trt(self.dtype),
            "cross_pair_mask_1": str_dtype_to_trt(self.dtype),
        }

    def get_input_shapes(self):
        batch_size = DimSpec(name="batch_size", dynamic=True)
        seqlen = DimSpec(name="seqlen", dynamic=True)
        token_s = DimSpec(name="token_s", size=self.token_s)
        token_z = DimSpec(name="token_z", size=self.token_z)

        return OrderedDict([
            ("s", (batch_size, seqlen, token_s)),
            ("z", (batch_size, seqlen, seqlen, token_z)),
            ("distogram", (batch_size, seqlen, seqlen)),
            ("cross_pair_mask_0", (batch_size, seqlen, seqlen)),
            ("cross_pair_mask_1", (batch_size, seqlen, seqlen,
                                   DimSpec(size=1, name="const_1"))),
        ])

    def get_output_shapes(self):
        batch_size = DimSpec(name="batch_size", dynamic=True)
        return OrderedDict([
            ("pred_value", (batch_size, 1)),
            ("logits_binary", (batch_size, 1)),
        ])


@dataclass
class AffinityModuleBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 16

    @property
    def optimization_profiles(self) -> list[Any]:
        return _create_optimization_profiles(self)


class Boltz2Config(PretrainedConfig):
    model_type = "boltz2"

    def __init__(self,
                 token_s: int,
                 token_z: int,
                 atom_s: int,
                 atom_z: int,
                 fix_sym_check: bool = True,
                 cyclic_pos_enc: bool = True,
                 bond_type_feature: bool = False,
                 conditioning_cutoff_min: float = 4.0,
                 conditioning_cutoff_max: float = 20.0,
                 structure_pairformer_config: PairformerConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 token_transformer_config: DiffusionTransformerConfig = None,
                 msa_module_config: MSAModuleConfig = None,
                 affinity_module_configs: dict[str, AffinityModuleConfig] = {},
                 input_embedder_config: InputEmbedderConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.token_s = token_s
        self.token_z = token_z
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc
        self.bond_type_feature = bond_type_feature
        self.conditioning_cutoff_min = conditioning_cutoff_min
        self.conditioning_cutoff_max = conditioning_cutoff_max
        self.structure_pairformer_config = structure_pairformer_config
        self.confidence_pairformer_config = confidence_pairformer_config
        self.token_transformer_config = token_transformer_config
        self.msa_module_config = msa_module_config
        self.affinity_module_configs = affinity_module_configs
        self.input_embedder_config = input_embedder_config
        if len(affinity_module_configs) > 0:
            self.is_affinity_model = True
        else:
            self.is_affinity_model = False

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        is_affinity=False,
                        **kwargs):
        if not is_affinity:
            ckpt = load_weights(name="boltz-2",
                                return_raw=True,
                                cache_path=checkpoint_dir)
        else:
            ckpt = load_weights(name="boltz-2-affinity",
                                return_raw=True,
                                cache_path=checkpoint_dir)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        hparams = state_dict["hyper_parameters"]

        token_s = hparams["token_s"]
        token_z = hparams["token_z"]
        atom_s = hparams["atom_s"]
        atom_z = hparams["atom_z"]
        atom_feature_dim = hparams["atom_feature_dim"]
        atoms_per_window_queries = hparams["atoms_per_window_queries"]
        atoms_per_window_keys = hparams["atoms_per_window_keys"]
        use_no_atom_char = hparams["use_no_atom_char"]
        use_atom_backbone_feat = hparams["use_atom_backbone_feat"]
        use_residue_feats_atoms = hparams["use_residue_feats_atoms"]
        fix_sym_check = hparams["fix_sym_check"]
        cyclic_pos_enc = hparams["cyclic_pos_enc"]
        bond_type_feature = hparams["bond_type_feature"]
        conditioning_cutoff_min = hparams["conditioning_cutoff_min"]
        conditioning_cutoff_max = hparams["conditioning_cutoff_max"]

        atom_encoder_depth = hparams["embedder_args"]["atom_encoder_depth"]
        atom_encoder_heads = hparams["embedder_args"]["atom_encoder_heads"]
        add_mol_type_feat = hparams["embedder_args"]["add_mol_type_feat"]
        add_method_conditioning = hparams["embedder_args"][
            "add_method_conditioning"]
        add_modified_flag = hparams["embedder_args"]["add_modified_flag"]
        add_cyclic_flag = hparams["embedder_args"]["add_cyclic_flag"]

        msa_pairwise_head_width = hparams["msa_args"]["pairwise_head_width"]
        msa_pairwise_num_heads = hparams["msa_args"]["pairwise_num_heads"]

        structure_pairformer_config = PairformerConfig(
            architecture="structure_pairformer",
            token_s=token_s,
            token_z=token_z,
            pairwise_head_width=msa_pairwise_head_width,
            pairwise_num_heads=msa_pairwise_num_heads,
            num_blocks=hparams["pairformer_args"]["num_blocks"],
            num_heads=hparams["pairformer_args"]["num_heads"],
            version="v2",
            dtype="float32")

        confidence_pairformer_config = PairformerConfig(
            architecture="confidence_pairformer",
            token_s=token_s,
            token_z=token_z,
            pairwise_head_width=msa_pairwise_head_width,
            pairwise_num_heads=msa_pairwise_num_heads,
            num_blocks=hparams["confidence_model_args"]["pairformer_args"]
            ["num_blocks"],
            num_heads=hparams["confidence_model_args"]["pairformer_args"]
            ["num_heads"],
            version="v2",
            dtype="float32")

        token_transformer_config = DiffusionTransformerConfig(
            architecture="token_transformer",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["token_transformer_depth"],
            num_heads=hparams["score_model_args"]["token_transformer_heads"],
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            version="v2")

        msa_module_config = MSAModuleConfig(
            architecture="msa_module",
            dtype="float32",
            msa_s=hparams["msa_args"]["msa_s"],
            token_z=token_z,
            token_s=token_s,
            msa_blocks=hparams["msa_args"]["msa_blocks"],
            num_tokens=len(TOKENS),
            pairwise_head_width=msa_pairwise_head_width,
            pairwise_num_heads=msa_pairwise_num_heads,
            use_paired_feature=True,
            version="v2")

        input_embedder_config = InputEmbedderConfig(
            architecture="input_embedder",
            dtype="float32",
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            pairwise_attn_backend="VANILLA",
            backend="torch",
            add_method_conditioning=add_method_conditioning,
            add_modified_flag=add_modified_flag,
            add_cyclic_flag=add_cyclic_flag,
            add_mol_type_feat=add_mol_type_feat,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
            version=
            "v2"  # Need put the version here to select diffusion transformer version
        )

        affinity_module_configs = {}
        if is_affinity:
            keys = []
            for key in hparams.keys():
                if key.startswith("affinity_model_args"):
                    keys.append(key)
            for key in keys:
                config = AffinityModuleConfig(
                    architecture="affinity_module",
                    dtype="float32",
                    token_s=token_s,
                    token_z=token_z,
                    num_dist_bins=hparams[key]["num_dist_bins"],
                    max_dist=hparams[key]["max_dist"],
                    pairformer_num_blocks=hparams[key]["pairformer_args"]
                    ["num_blocks"],
                    pairwise_head_width=msa_pairwise_head_width,
                    pairwise_num_heads=msa_pairwise_num_heads,
                )
                key = key.replace("model_args", "module")
                affinity_module_configs[key] = config

        return cls(token_s=token_s,
                   token_z=token_z,
                   atom_s=atom_s,
                   atom_z=atom_z,
                   fix_sym_check=fix_sym_check,
                   cyclic_pos_enc=cyclic_pos_enc,
                   bond_type_feature=bond_type_feature,
                   conditioning_cutoff_min=conditioning_cutoff_min,
                   conditioning_cutoff_max=conditioning_cutoff_max,
                   structure_pairformer_config=structure_pairformer_config,
                   confidence_pairformer_config=confidence_pairformer_config,
                   token_transformer_config=token_transformer_config,
                   msa_module_config=msa_module_config,
                   affinity_module_configs=affinity_module_configs,
                   input_embedder_config=input_embedder_config,
                   **kwargs)
