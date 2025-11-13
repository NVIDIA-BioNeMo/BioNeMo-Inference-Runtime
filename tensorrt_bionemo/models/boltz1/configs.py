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
from typing import Any, Optional, Union

import torch
from transformers import PretrainedConfig

from tensorrt_bionemo.config import (BuildModuleConfig, DimSpec,
                                     PretrainedModuleConfig)
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping

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
                 trimul_high_precision: bool = True,
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
        self.trimul_high_precision = trimul_high_precision
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


def _create_optimization_profiles(
        self: BuildModuleConfig,
        seqlen_key_names: list[str] = ["seqlen"]) -> list[Any]:
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
                if spec.name in seqlen_key_names:
                    min_shape.append(rmin)
                    opt_shape.append(rmax)
                    max_shape.append(rmax)
                elif spec.name == "num_particles":
                    min_shape.append(1)
                    opt_shape.append(self.module_config.max_num_particles)
                    max_shape.append(self.module_config.max_num_particles)
                elif spec.name == "num_diffusion_samples":
                    min_shape.append(1)
                    opt_shape.append(self.module_config.max_diffusion_samples)
                    max_shape.append(self.module_config.max_diffusion_samples)
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


class DiffusionTransformerConfig(PretrainedModuleConfig):

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
                 opm_chunk_size: Optional[int] = 16,
                 opm_mask_chunk_size: Optional[int] = 256,
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
        self.opm_chunk_size = opm_chunk_size
        self.opm_mask_chunk_size = opm_mask_chunk_size
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


class AtomDiffusionConfig(PretrainedModuleConfig):

    def __init__(self,
                 num_sampling_steps: int = 50,
                 sigma_min: float = 0.0004,
                 sigma_max: float = 160.0,
                 sigma_data: float = 16.0,
                 rho: float = 7,
                 P_mean: float = -1.2,
                 P_std: float = 1.5,
                 gamma_0: float = 0.8,
                 gamma_min: float = 1.0,
                 noise_scale: float = 1.003,
                 coordinate_augmentation: bool = True,
                 alignment_reverse_diff: bool = False,
                 synchronize_sigmas: bool = False,
                 accumulate_token_repr: bool = True,
                 version: str = "v1",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = 1.5 if version == "v1" else 1.638
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.version = version

        # FIXME: hardcoded for now
        self.dim_fourier = 256
        self.token_s = 384
        self.accumulate_token_repr = accumulate_token_repr and version == "v1"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


class InputEmbedderConfig(PretrainedModuleConfig):

    def __init__(self,
                 atom_s: int,
                 atom_z: int,
                 token_s: int,
                 token_z: int,
                 atoms_per_window_queries: int,
                 atoms_per_window_keys: int,
                 atom_feature_dim: int,
                 atom_encoder_depth: int,
                 atom_encoder_heads: int,
                 version: str = "v1",
                 pairwise_attn_backend: str = 'VANILLA',
                 backend: str = "torch",
                 **kwargs):
        super().__init__(**kwargs)
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.atom_feature_dim = atom_feature_dim
        self.atom_encoder_depth = atom_encoder_depth
        self.atom_encoder_heads = atom_encoder_heads
        self.backend = backend

        # See the convert.py for why use the version "v2" here.
        self.diffusion_transformer_config = DiffusionTransformerConfig(
            architecture="diffusion_transformer",
            dtype=self.dtype,
            num_blocks=atom_encoder_depth,
            num_heads=atom_encoder_heads,
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            pairwise_attn_backend=pairwise_attn_backend,
            backend=backend,
            mapping=self.mapping,
            version="v2")

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def set_dtype(self, value: Union[str, torch.dtype]):
        super().set_dtype(value)
        if hasattr(self, "diffusion_transformer_config"):
            self.diffusion_transformer_config.set_dtype(value)


class TrunkConfig(PretrainedModuleConfig):

    def __init__(self,
                 pairformer_config: PairformerConfig = None,
                 msa_module_config: MSAModuleConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pairformer_config = pairformer_config
        self.msa_module_config = msa_module_config

    def set_dtype(self, value: Union[str, torch.dtype]):
        super().set_dtype(value)
        if hasattr(self, "pairformer_config"):
            self.pairformer_config.set_dtype(value)
        if hasattr(self, "msa_module_config"):
            self.msa_module_config.set_dtype(value)

    def set_mapping(self, value: Mapping):
        super().set_mapping(value)
        if hasattr(self, "pairformer_config"):
            self.pairformer_config.set_mapping(value)
        if hasattr(self, "msa_module_config"):
            self.msa_module_config.set_mapping(value)

    def set_triangle_attn_backend(self, value: str):
        if hasattr(self, "pairformer_config"):
            self.pairformer_config.triangle_attn_backend = value
        if hasattr(self, "msa_module_config"):
            self.msa_module_config.triangle_attn_backend = value


class ScoreModelConfig(PretrainedModuleConfig):

    def __init__(self,
                 atom_s: int,
                 atom_z: int,
                 token_s: int,
                 token_z: int,
                 dim_fourier: int,
                 atoms_per_window_queries: int,
                 atoms_per_window_keys: int,
                 conditioning_transition_layers: int,
                 version: str = "v1",
                 atom_encoder_config: DiffusionTransformerConfig = None,
                 token_transformer_config: DiffusionTransformerConfig = None,
                 atom_decoder_config: DiffusionTransformerConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.dim_fourier = dim_fourier
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.conditioning_transition_layers = conditioning_transition_layers
        self.atom_encoder_config = atom_encoder_config
        self.token_transformer_config = token_transformer_config
        self.atom_decoder_config = atom_decoder_config
        self.version = version

    def set_dtype(self, value: Union[str, torch.dtype]):
        super().set_dtype(value)
        if hasattr(self, "atom_encoder_config"):
            self.atom_encoder_config.set_dtype(value)
        if hasattr(self, "token_transformer_config"):
            self.token_transformer_config.set_dtype(value)
        if hasattr(self, "atom_decoder_config"):
            self.atom_decoder_config.set_dtype(value)

    def set_mapping(self, value: Mapping):
        if hasattr(self, "atom_encoder_config"):
            self.atom_encoder_config.set_mapping(value)
        if hasattr(self, "token_transformer_config"):
            self.token_transformer_config.set_mapping(value)
        if hasattr(self, "atom_decoder_config"):
            self.atom_decoder_config.set_mapping(value)


class StructureModuleConfig(PretrainedModuleConfig):

    def __init__(self,
                 score_model_config: ScoreModelConfig = None,
                 atom_diffusion_config: AtomDiffusionConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.score_model_config = score_model_config
        self.atom_diffusion_config = atom_diffusion_config

    def set_dtype(self, value: Union[str, torch.dtype]):
        super().set_dtype(value)
        if hasattr(self, "score_model_config"):
            self.score_model_config.set_dtype(value)
        if hasattr(self, "atom_diffusion_config"):
            self.atom_diffusion_config.set_dtype(value)

    def set_mapping(self, value: Mapping):
        if hasattr(self, "score_model_config"):
            self.score_model_config.set_mapping(value)
        if hasattr(self, "atom_diffusion_config"):
            self.atom_diffusion_config.set_mapping(value)


class Boltz1Config(PretrainedConfig):
    model_type = "boltz1"

    def __init__(self,
                 global_config: PretrainedModuleConfig = None,
                 input_embedder_config: InputEmbedderConfig = None,
                 trunk_config: TrunkConfig = None,
                 structure_module_config: StructureModuleConfig = None,
                 confidence_pairformer_config: PairformerConfig = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.global_config = global_config
        self.input_embedder_config = input_embedder_config
        self.trunk_config = trunk_config
        self.structure_module_config = structure_module_config

        self.confidence_pairformer_config = confidence_pairformer_config

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        **kwargs):
        ckpt = load_weights(name="boltz-1",
                            return_raw=True,
                            cache_path=checkpoint_dir)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        hparams = state_dict["hyper_parameters"]

        num_bins = hparams["num_bins"]
        token_s = hparams["token_s"]
        token_z = hparams["token_z"]
        atom_s = hparams["atom_s"]
        atom_z = hparams["atom_z"]
        atom_feature_dim = hparams["atom_feature_dim"]
        atoms_per_window_queries = hparams["atoms_per_window_queries"]
        atoms_per_window_keys = hparams["atoms_per_window_keys"]
        atom_encoder_depth = hparams["embedder_args"]["atom_encoder_depth"]
        atom_encoder_heads = hparams["embedder_args"]["atom_encoder_heads"]
        msa_pairwise_head_width = hparams["msa_args"]["pairwise_head_width"]
        msa_pairwise_num_heads = hparams["msa_args"]["pairwise_num_heads"]

        # Conduct input embedder configuration
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
            backend="torch")

        # Conduct recycling configuration
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
        trunk_config = TrunkConfig(
            architecture="recycling",
            dtype="float32",
            pairformer_config=structure_pairformer_config,
            msa_module_config=msa_module_config)

        # Conduct score model configuration
        diffusion_process_args = hparams["diffusion_process_args"]
        atom_diffusion_config = AtomDiffusionConfig(
            architecture="atom_diffusion",
            dtype="float32",
            sigma_min=diffusion_process_args["sigma_min"],
            sigma_max=diffusion_process_args["sigma_max"],
            sigma_data=diffusion_process_args["sigma_data"],
            rho=diffusion_process_args["rho"],
            P_mean=diffusion_process_args["P_mean"],
            P_std=diffusion_process_args["P_std"],
            gamma_0=diffusion_process_args["gamma_0"],
            gamma_min=diffusion_process_args["gamma_min"],
            noise_scale=diffusion_process_args["noise_scale"],
            coordinate_augmentation=diffusion_process_args[
                "coordinate_augmentation"],
            alignment_reverse_diff=diffusion_process_args[
                "alignment_reverse_diff"],
            synchronize_sigmas=diffusion_process_args["synchronize_sigmas"],
            version="v1")

        atom_encoder_config = DiffusionTransformerConfig(
            architecture="atom_encoder",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["atom_encoder_depth"],
            num_heads=hparams["score_model_args"]["atom_encoder_heads"],
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=None,
            version="v2")
        token_transformer_config = DiffusionTransformerConfig(
            architecture="token_transformer",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["token_transformer_depth"],
            num_heads=hparams["score_model_args"]["token_transformer_heads"],
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            version="v2")
        atom_decoder_config = DiffusionTransformerConfig(
            architecture="atom_decoder",
            dtype="float32",
            num_blocks=hparams["score_model_args"]["atom_decoder_depth"],
            num_heads=hparams["score_model_args"]["atom_decoder_heads"],
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=None,
            version="v2")
        score_model_config = ScoreModelConfig(
            architecture="score_model",
            dtype="float32",
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            dim_fourier=hparams["score_model_args"]["dim_fourier"],
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            conditioning_transition_layers=hparams["score_model_args"]
            ["conditioning_transition_layers"],
            version="v1",
            atom_encoder_config=atom_encoder_config,
            token_transformer_config=token_transformer_config,
            atom_decoder_config=atom_decoder_config)
        structure_module_config = StructureModuleConfig(
            architecture="structure_module",
            dtype="float32",
            score_model_config=score_model_config,
            atom_diffusion_config=atom_diffusion_config)

        # Conduct confidence module configuration
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

        global_config = PretrainedModuleConfig(architecture="global",
                                               dtype="float32",
                                               token_s=token_s,
                                               token_z=token_z,
                                               atom_s=atom_s,
                                               atom_z=atom_z,
                                               num_bins=num_bins)

        return cls(global_config=global_config,
                   trunk_config=trunk_config,
                   confidence_pairformer_config=confidence_pairformer_config,
                   token_transformer_config=token_transformer_config,
                   input_embedder_config=input_embedder_config,
                   structure_module_config=structure_module_config,
                   **kwargs)
