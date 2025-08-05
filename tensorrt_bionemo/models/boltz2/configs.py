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
from transformers import PretrainedConfig

from tensorrt_bionemo.config import BuildModuleConfig, PretrainedModuleConfig, DimSpec
from tensorrt_bionemo.hubs.checkpoint import load_hf_weights
from tensorrt_bionemo.models.boltz1.configs import (MSAModuleConfig,
                                                    PairformerConfig,
                                                    TokenTransformerConfig)
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
                 structure_pairformer_config: PairformerConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 token_transformer_config: TokenTransformerConfig = None,
                 msa_module_config: MSAModuleConfig = None,
                 affinity_module_configs: dict[str, AffinityModuleConfig] = {},
                 **kwargs):
        super().__init__(**kwargs)

        self.structure_pairformer_config = structure_pairformer_config
        self.confidence_pairformer_config = confidence_pairformer_config
        self.token_transformer_config = token_transformer_config
        self.msa_module_config = msa_module_config
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
                   msa_module_config=msa_module_config,
                   affinity_module_configs=affinity_module_configs,
                   **kwargs)
