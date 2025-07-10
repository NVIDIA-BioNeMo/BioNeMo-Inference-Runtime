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
from transformers import PretrainedConfig

from tensorrt_bionemo.hubs.checkpoint import load_hf_weights

from .modules import (AffinityModuleConfig, PairformerConfig,
                      TokenTransformerConfig)


class Boltz1Config(PretrainedConfig):
    model_type = "boltz1"

    def __init__(self,
                 structure_pairformer_config: PairformerConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 token_transformer_config: TokenTransformerConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.structure_pairformer_config = structure_pairformer_config
        self.confidence_pairformer_config = confidence_pairformer_config
        self.token_transformer_config = token_transformer_config

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        **kwargs):
        if checkpoint_dir is None:
            ckpt = load_hf_weights(name="boltz-1", return_raw=True)
            state_dict = torch.load(ckpt,
                                    map_location="cpu",
                                    weights_only=False)
        else:
            state_dict = torch.load(checkpoint_dir,
                                    map_location="cpu",
                                    weights_only=False)
        hparams = state_dict["hyper_parameters"]

        token_s = hparams["token_s"]
        token_z = hparams["token_z"]
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
            version="v1",
            dtype="float32")
        confidence_pairformer_config = PairformerConfig(
            architecture="confidence_pairformer",
            token_s=token_s,
            token_z=token_z,
            pairwise_head_width=msa_pairwise_head_width,
            pairwise_num_heads=msa_pairwise_num_heads,
            num_blocks=hparams["pairformer_args"]["num_blocks"],
            num_heads=hparams["pairformer_args"]["num_heads"],
            version="v1",
            dtype="float32")
        token_transformer_config = TokenTransformerConfig(
            architecture="token_transformer",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["token_transformer_depth"],
            num_heads=hparams["score_model_args"]["token_transformer_heads"],
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            version="v1")
        return cls(structure_pairformer_config=structure_pairformer_config,
                   confidence_pairformer_config=confidence_pairformer_config,
                   token_transformer_config=token_transformer_config,
                   **kwargs)


class Boltz2Config(PretrainedConfig):
    model_type = "boltz2"

    def __init__(self,
                 structure_pairformer_config: PairformerConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 token_transformer_config: TokenTransformerConfig = None,
                 affinity_module_configs: dict[str, AffinityModuleConfig] = {},
                 **kwargs):
        super().__init__(**kwargs)

        self.structure_pairformer_config = structure_pairformer_config
        self.confidence_pairformer_config = confidence_pairformer_config
        self.token_transformer_config = token_transformer_config
        self.affinity_module_configs = affinity_module_configs
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
        if checkpoint_dir is None:
            if not is_affinity:
                ckpt = load_hf_weights(name="boltz-2", return_raw=True)
            else:
                ckpt = load_hf_weights(name="boltz-2-affinity", return_raw=True)
            state_dict = torch.load(ckpt,
                                    map_location="cpu",
                                    weights_only=False)
        else:
            state_dict = torch.load(checkpoint_dir,
                                    map_location="cpu",
                                    weights_only=False)
        hparams = state_dict["hyper_parameters"]

        token_s = hparams["token_s"]
        token_z = hparams["token_z"]
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

        token_transformer_config = TokenTransformerConfig(
            architecture="token_transformer",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["token_transformer_depth"],
            num_heads=hparams["score_model_args"]["token_transformer_heads"],
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            version="v2")

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

        return cls(structure_pairformer_config=structure_pairformer_config,
                   confidence_pairformer_config=confidence_pairformer_config,
                   token_transformer_config=token_transformer_config,
                   affinity_module_configs=affinity_module_configs,
                   **kwargs)
