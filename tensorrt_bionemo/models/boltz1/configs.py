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

from .const import TOKENS


class PairformerConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 token_s: int = 384,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 num_blocks: int = 48,
                 num_heads: int = 16,
                 max_batch_size: int = 1,
                 max_transition_tp_size: bool = True,
                 max_attention_pairwise_tp_size: bool = True,
                 max_tri_mul_tp_size: bool = True,
                 triangle_attn_node_chunk_size: int = 0,
                 no_update_s: bool = False,
                 no_update_z: bool = False,
                 backend: str = "torch",
                 triangle_attn_backend: str = 'VANILLA',
                 pairwise_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 s_path_dtype: str = None,
                 post_layer_norm: bool = False,
                 triangle_attn_cueq_fallback_threshold: int = 0,
                 version: str = "v1",
                 **kwargs):
        super().__init__(**kwargs)

        self.token_s = token_s
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_batch_size = max_batch_size
        self.max_transition_tp_size = max_transition_tp_size
        self.max_attention_pairwise_tp_size = max_attention_pairwise_tp_size
        self.max_tri_mul_tp_size = max_tri_mul_tp_size
        self.triangle_attn_node_chunk_size = triangle_attn_node_chunk_size
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.backend = backend
        self.triangle_attn_backend = triangle_attn_backend
        self.pairwise_attn_backend = pairwise_attn_backend
        self.disable_custom_all_reduce = max_transition_tp_size or max_attention_pairwise_tp_size
        self.support_batch = support_batch
        self.s_path_dtype = s_path_dtype
        self.post_layer_norm = post_layer_norm
        self.triangle_attn_cueq_fallback_threshold = triangle_attn_cueq_fallback_threshold
        self.version = version

    @property
    def attention_initial_norm(self):
        if self.version == "v1":
            return True
        else:
            return False

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("s", (batch_size, seqlen,
                       DimSpec(size=self.token_s, name="token_s"))),
                ("z", (batch_size, seqlen, seqlen,
                       DimSpec(size=self.token_z, name="token_z"))),
                ("mask", (batch_size, seqlen)),
                ("pair_mask", (batch_size, seqlen, seqlen)),
            ])
        return OrderedDict([
            ("s", (seqlen, DimSpec(size=self.token_s, name="token_s"))),
            ("z", (seqlen, seqlen, DimSpec(size=self.token_z, name="token_z"))),
            ("mask", (seqlen, )),
            ("pair_mask", (seqlen, seqlen)),
        ])

    def get_output_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("output_s", (batch_size, seqlen,
                              DimSpec(size=self.token_s, name="token_s"))),
                ("output_z", (batch_size, seqlen, seqlen,
                              DimSpec(size=self.token_z, name="token_z"))),
            ])
        return OrderedDict([
            ("output_s", (seqlen, DimSpec(size=self.token_s, name="token_s"))),
            ("output_z", (seqlen, seqlen,
                          DimSpec(size=self.token_z, name="token_z"))),
        ])

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


def _create_optimization_profiles(self: BuildModuleConfig) -> list[Any]:
    input_shapes = self.module_config.get_input_shapes()

    if self.force_num_profiles == 0:
        return []

    min_seqlen = self.min_seqlen - self.min_seqlen % self.align
    max_seqlen = (self.max_seqlen + self.align - 1) // self.align * self.align
    assert (max_seqlen - min_seqlen) % self.force_num_profiles == 0
    step = (max_seqlen - min_seqlen) // self.force_num_profiles

    min_max_seqlens = []

    for i in range(self.force_num_profiles):
        min_max_seqlens.append(
            (min_seqlen + i * step, min_seqlen + (i + 1) * step))

    profiles = []
    for rmin, rmax in min_max_seqlens:
        profile = {}
        for k, v in input_shapes.items():
            min_shape = []
            opt_shape = []
            max_shape = []

            for spec in v:
                if spec.name == "seqlen":
                    min_shape.append(rmin)
                    opt_shape.append(rmax)
                    max_shape.append(rmax)
                elif spec.name == "num_particles":
                    min_shape.append(1)
                    opt_shape.append(self.module_config.max_num_particles)
                    max_shape.append(self.module_config.max_num_particles)
                elif spec.name == "batch_size":
                    min_shape.append(1)
                    opt_shape.append(self.module_config.max_batch_size)
                    max_shape.append(self.module_config.max_batch_size)
                else:
                    min_shape.append(spec.size)
                    opt_shape.append(spec.size)
                    max_shape.append(spec.size)
            profile[k] = (min_shape, opt_shape, max_shape)
        profiles.append(profile)
    return profiles


@dataclass
class PairformerBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 16

    @property
    def optimization_profiles(self) -> list[Any]:
        return _create_optimization_profiles(self)


class TokenTransformerConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 num_blocks: int,
                 num_heads: int,
                 dim: int = 768,
                 dim_single_cond: int = 768,
                 dim_pairwise: int = 128,
                 expansion_factor: int = 2,
                 max_num_particles: int = 1,
                 max_diffusion_samples: int = 1,
                 version: str = "v1",
                 pairwise_attn_backend: str = 'VANILLA',
                 backend: str = "torch",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim = dim
        self.dim_single_cond = dim_single_cond if dim_single_cond is not None else dim
        self.dim_pairwise = dim_pairwise
        self.version = version
        self.backend = backend
        self.expansion_factor = expansion_factor
        self.pairwise_attn_backend = pairwise_attn_backend
        self.max_num_particles = max_num_particles
        self.max_diffusion_samples = max_diffusion_samples
        self.max_batch_size = self.max_diffusion_samples * self.max_num_particles

    @property
    def attention_initial_norm(self):
        return False

    @property
    def with_pair_bias_cache(self):
        return self.version == "v1"

    @property
    def post_layer_norm(self):
        return False

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        batch_size = DimSpec(name="batch_size", dynamic=True)
        dim = DimSpec(name="dim", size=self.dim)
        dim_single_cond = DimSpec(name="dim_single_cond",
                                  size=self.dim_single_cond)
        dim_pairwise = DimSpec(name="dim_pairwise", size=self.dim_pairwise)
        num_heads = DimSpec(name="num_heads", size=self.num_heads)
        num_blocks = DimSpec(name="num_blocks", size=self.num_blocks)
        heads_times_blocks = DimSpec(name="heads_times_blocks",
                                     size=self.num_heads * self.num_blocks)
        z_shape = (DimSpec(size=1, name="n_seqs"), num_heads, seqlen, seqlen,
                   num_blocks)
        if self.version == "v2":
            z_shape = (DimSpec(size=1, name="n_seqs"), seqlen, seqlen,
                       heads_times_blocks)

        return OrderedDict([
            ("a", (batch_size, seqlen, dim)),
            ("s", (batch_size, seqlen, dim_single_cond)),
            ("z", z_shape),
            ("mask", (batch_size, seqlen)),
        ])

    def get_output_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        batch_size = DimSpec(name="batch_size", dynamic=True)
        dim = DimSpec(name="dim", size=self.dim)
        return OrderedDict([("output_a", (batch_size, seqlen, dim))])


@dataclass
class TokenTransformerBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 16

    @property
    def optimization_profiles(self) -> list[Any]:
        return _create_optimization_profiles(self)


class MSAModuleConfig(PretrainedModuleConfig):

    def __init__(self,
                 msa_s: int,
                 token_z: int,
                 token_s: int,
                 msa_blocks: int,
                 num_tokens: int,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 use_paired_feature: bool = True,
                 version: str = "v1",
                 triangle_attn_backend: str = 'VANILLA',
                 backend: str = "torch",
                 **kwargs):
        super().__init__(**kwargs)
        self.msa_s = msa_s
        self.token_z = token_z
        self.token_s = token_s
        self.msa_blocks = msa_blocks
        self.num_tokens = num_tokens
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.use_paired_feature = use_paired_feature
        self.version = version
        self.triangle_attn_backend = triangle_attn_backend
        self.backend = backend

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        batch_size = DimSpec(name="batch_size", dynamic=True)
        n_msa = DimSpec(name="n_msa", dynamic=True)
        token_z = DimSpec(name="token_z", size=self.token_z)
        token_s = DimSpec(name="token_s", size=self.token_s)

        return OrderedDict([
            ("z", (batch_size, seqlen, seqlen, token_z)),
            ("emb", (batch_size, seqlen, token_s)),
            ("msa", (batch_size, n_msa, seqlen)),
            ("has_deletion", (batch_size, n_msa, seqlen)),
            ("deletion_value", (batch_size, n_msa, seqlen)),
            ("msa_paired", (batch_size, n_msa, seqlen)),
            ("msa_mask", (batch_size, n_msa, seqlen)),
            ("token_pad_mask", (batch_size, seqlen)),
        ])

    def get_output_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        batch_size = DimSpec(name="batch_size", dynamic=True)
        token_z = DimSpec(name="token_z", size=self.token_z)
        return OrderedDict([("output_z", (batch_size, seqlen, seqlen, token_z))
                            ])


class Boltz1Config(PretrainedConfig):
    model_type = "boltz1"

    def __init__(self,
                 structure_pairformer_config: PairformerConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 token_transformer_config: TokenTransformerConfig = None,
                 msa_module_config: MSAModuleConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.structure_pairformer_config = structure_pairformer_config
        self.confidence_pairformer_config = confidence_pairformer_config
        self.token_transformer_config = token_transformer_config
        self.msa_module_config = msa_module_config

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
            use_paired_feature=False,
            version="v1")
        return cls(structure_pairformer_config=structure_pairformer_config,
                   confidence_pairformer_config=confidence_pairformer_config,
                   token_transformer_config=token_transformer_config,
                   msa_module_config=msa_module_config,
                   **kwargs)
