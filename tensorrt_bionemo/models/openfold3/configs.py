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

from transformers import PretrainedConfig

from tensorrt_bionemo.config import (BuildModuleConfig, DimSpec,
                                     PretrainedModuleConfig)
from tensorrt_bionemo.models.boltz1.configs import (
    PairformerConfig, _create_optimization_profiles)

PRETRAINED_OF3_CONFIG = {
    "model": {
        "pairformer": {
            "c_s": 384,
            "c_z": 128,
            "c_hidden_pair_bias": 24,  # c_s / no_heads_pair_bias
            "no_heads_pair_bias": 16,
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "no_heads_pair": 4,
            "no_blocks": 48,
            "transition_type": "swiglu",
            "transition_n": 4,
            "inf": 1e9,
            "eps": 1e-5,
        },
        "token_transformer": {
            "c_a": 768,
            "c_s": 384,
            "c_z": 128,
            "c_hidden": 48,  # c_token / no_heads
            "no_heads": 16,
            "no_blocks": 24,
            "n_transition": 2,
            "inf": 1e9,
            "eps": 1e-5,
        }
    }
}


class TokenTransformerConfig(PretrainedModuleConfig):
    """ TODO: currently, it doesn't support for max_diffusion_samples > 1 """

    def __init__(
            self,
            *,
            num_blocks: int,
            num_heads: int,
            dim: int = 768,  # c_a
            dim_single_cond: int = 384,  # c_s
            dim_pairwise: int = 128,  # c_z
            expansion_factor: int = 2,  # n_transition
            max_num_particles: int = 1,
            max_diffusion_samples: int = 1,
            version: str = "v1",
            pairwise_attn_backend: str = 'VANILLA',
            backend: str = "torch",
            **kwargs):
        super().__init__(**kwargs)
        # assert max_diffusion_samples == 1, f"max_diffusion_samples must be 1 at the current version, but got {max_diffusion_samples}"
        # assert max_num_particles == 1, f"max_num_particles must be 1, but got {max_num_particles}"
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

    @property
    def conditioned_transition_using_silu(self):
        return True

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        batch_size = DimSpec(name="bs", size=1)
        num_diffusion_samples = DimSpec(name="num_diffusion_samples",
                                        dynamic=True)
        dim = DimSpec(name="dim", size=self.dim)
        dim_single_cond = DimSpec(name="dim_single_cond",
                                  size=self.dim_single_cond)
        dim_pairwise = DimSpec(name="dim_pairwise", size=self.dim_pairwise)

        return OrderedDict([
            ("a", (num_diffusion_samples, seqlen, dim)),
            ("s", (batch_size, seqlen, dim_single_cond)),
            ("z", (batch_size, seqlen, seqlen, dim_pairwise)),
            ("mask", (batch_size, seqlen)),
        ])

    def get_output_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)
        # batch_size = DimSpec(name="batch_size", dynamic=True)
        num_diffusion_samples = DimSpec(name="num_diffusion_samples",
                                        dynamic=True)
        dim = DimSpec(name="dim", size=self.dim)
        return OrderedDict([("output_a", (num_diffusion_samples, seqlen, dim))])


@dataclass
class TokenTransformerBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 16

    @property
    def optimization_profiles(self) -> list[Any]:
        return _create_optimization_profiles(self)


class OpenFold3Config(PretrainedConfig):
    model_type = "openfold3"

    def __init__(self,
                 pairformer_config: PairformerConfig = None,
                 token_transformer_config: TokenTransformerConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pairformer_config = pairformer_config
        self.token_transformer_config = token_transformer_config

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        pretrained_config: dict = None,
                        **kwargs):
        if pretrained_config is None:
            pretrained_config = PRETRAINED_OF3_CONFIG

        _c = pretrained_config["model"]["pairformer"]
        token_s = _c["c_s"]
        token_z = _c["c_z"]
        pairwise_head_width = _c["c_hidden_pair_att"]
        pairwise_num_heads = _c["no_heads_pair"]

        pairformer_config = PairformerConfig(
            architecture="pairformer",
            token_s=token_s,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            num_blocks=_c["no_blocks"],
            num_heads=_c["no_heads_pair_bias"],
            trimul_high_precision=False,
            version="v1",
            dtype="float32")

        _c = pretrained_config["model"]["token_transformer"]
        token_transformer_config = TokenTransformerConfig(
            architecture="token_transformer",
            num_blocks=_c["no_blocks"],
            num_heads=_c["no_heads"],
            dim=_c["c_a"],
            dim_single_cond=_c["c_s"],
            dim_pairwise=_c["c_z"],
            expansion_factor=_c["n_transition"],
            max_num_particles=1,
            max_diffusion_samples=1,
            version="v1",
            pairwise_attn_backend="VANILLA",
            dtype="float32")

        return cls(pairformer_config=pairformer_config,
                   token_transformer_config=token_transformer_config,
                   **kwargs)
