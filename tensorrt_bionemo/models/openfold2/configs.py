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
import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from transformers import PretrainedConfig

from tensorrt_bionemo.config import (BuildModuleConfig, DimSpec,
                                     PretrainedModuleConfig)
from tensorrt_bionemo.models.boltz1.configs import _create_optimization_profiles


class EvoformerStackConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 c_m: int,
                 c_z: int,
                 c_s: int,
                 c_hidden_msa_att: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 transition_n: int,
                 no_blocks: int,
                 no_column_attention: bool = False,
                 opm_first: bool = False,
                 support_batch: bool = True,
                 chunk_size: int = 0,
                 max_batch_size: int = 1,
                 triangle_attn_backend: str = 'VANILLA',
                 n_seq: int = 516,
                 **kwargs):
        super().__init__(**kwargs)
        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.c_hidden_msa_att = c_hidden_msa_att
        self.c_hidden_opm = c_hidden_opm
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_msa = no_heads_msa
        self.no_heads_pair = no_heads_pair
        self.transition_n = transition_n
        self.no_blocks = no_blocks
        self.max_batch_size = max_batch_size
        self.triangle_attn_backend = triangle_attn_backend
        self.no_column_attention = no_column_attention
        self.opm_first = opm_first
        self.support_batch = support_batch
        self.n_seq = n_seq
        self.chunk_size = chunk_size

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        n_res = DimSpec(name="n_res", dynamic=True)
        n_seq = DimSpec(name="n_seq", size=self.n_seq)
        c_m = DimSpec(size=self.c_m, name="c_m")
        c_z = DimSpec(size=self.c_z, name="c_z")

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("m", (batch_size, n_seq, n_res, c_m)),
                ("z", (batch_size, n_res, n_res, c_z)),
                ("msa_mask", (batch_size, n_seq, n_res)),
                ("pair_mask", (batch_size, n_res, n_res)),
            ])
        return OrderedDict([
            ("m", (n_seq, n_res, c_m)),
            ("z", (n_res, n_res, c_z)),
            ("msa_mask", (n_seq, n_res)),
            ("pair_mask", (n_res, n_res)),
        ])

    def get_output_shapes(self):
        n_res = DimSpec(name="n_res", dynamic=True)
        n_seq = DimSpec(name="n_seq", size=self.n_seq)
        c_m = DimSpec(size=self.c_m, name="c_m")
        c_z = DimSpec(size=self.c_z, name="c_z")
        c_s = DimSpec(size=self.c_s, name="c_s")

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("output_m", (batch_size, n_seq, n_res, c_m)),
                ("output_z", (batch_size, n_res, n_res, c_z)),
                ("output_s", (batch_size, n_res, c_s)),
            ])
        return OrderedDict([
            ("output_m", (n_seq, n_res, c_m)),
            ("output_z", (n_res, n_res, c_z)),
            ("output_s", (n_res, c_m)),
        ])


@dataclass
class EvoformerStackBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 16

    @property
    def optimization_profiles(self) -> list[Any]:
        return _create_optimization_profiles(self, seqlen_key_names=["n_res"])


class ExtraMSAStackConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 c_m: int,
                 c_z: int,
                 c_hidden_msa_att: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 no_blocks: int,
                 transition_n: int,
                 opm_first: bool = False,
                 support_batch: bool = True,
                 chunk_size: int = 0,
                 opm_chunk_size: Optional[int] = None,
                 opm_mask_chunk_size: Optional[int] = None,
                 max_batch_size: int = 1,
                 max_msa_size: int = 5120,
                 max_seq_len: int = 2048,
                 padding_inputs: bool = True,
                 triangle_attn_backend: str = 'VANILLA',
                 **kwargs):
        super().__init__(**kwargs)
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden_msa_att = c_hidden_msa_att
        self.c_hidden_opm = c_hidden_opm
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_msa = no_heads_msa
        self.no_heads_pair = no_heads_pair
        self.no_blocks = no_blocks
        self.transition_n = transition_n
        self.opm_first = opm_first
        self.support_batch = support_batch
        self.chunk_size = chunk_size
        self.opm_chunk_size = opm_chunk_size
        self.opm_mask_chunk_size = opm_mask_chunk_size
        self.max_batch_size = max_batch_size
        self.triangle_attn_backend = triangle_attn_backend
        self.max_msa_size = max_msa_size
        self.max_seq_len = max_seq_len
        self.padding_inputs = padding_inputs

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        n_res = DimSpec(name="n_res", dynamic=True)
        n_extra = DimSpec(name="n_extra", dynamic=True)
        c_m = DimSpec(size=self.c_m, name="c_m")
        c_z = DimSpec(size=self.c_z, name="c_z")

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("m", (batch_size, n_extra, n_res, c_m)),
                ("z", (batch_size, n_res, n_res, c_z)),
                ("msa_mask", (batch_size, n_extra, n_res)),
                ("pair_mask", (batch_size, n_res, n_res)),
            ])
        return OrderedDict([
            ("m", (n_extra, n_res, c_m)),
            ("z", (n_res, n_res, c_z)),
            ("msa_mask", (n_extra, n_res)),
            ("pair_mask", (n_res, n_res)),
        ])

    def get_output_shapes(self):
        n_res = DimSpec(name="n_res", dynamic=True)
        c_z = DimSpec(size=self.c_z, name="c_z")

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("output_z", (batch_size, n_res, n_res, c_z)),
            ])
        return OrderedDict([
            ("output_z", (n_res, n_res, c_z)),
        ])


PRETRAINED_OF2_CONFIG = {
    "model": {
        "evoformer_stack": {
            "c_m": 256,
            "c_z": 128,
            "c_hidden_msa_att": 32,
            "c_hidden_opm": 32,
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "c_s": 384,
            "no_heads_msa": 8,
            "no_heads_pair": 4,
            "no_blocks": 48,
            "transition_n": 4,
            "no_column_attention": False,
            "opm_first": False,
            "inf": 1e9,
            "eps": 1e-5,
        },
        "extra_msa_stack": {
            "c_hidden_msa_att": 8,
            "c_hidden_mul": 128,
            "c_hidden_opm": 32,
            "c_hidden_pair_att": 32,
            "c_m": 64,
            "c_z": 128,
            "eps": 1e-5,
            "inf": 1e9,
            "no_blocks": 4,
            "no_heads_msa": 8,
            "no_heads_pair": 4,
            "opm_first": False,
            "transition_n": 4,
        }
    }
}

PRETRAINED_OF2_MULTIMER_CONFIG = copy.deepcopy(PRETRAINED_OF2_CONFIG)
PRETRAINED_OF2_MULTIMER_CONFIG["model"]["evoformer_stack"]["opm_first"] = True
PRETRAINED_OF2_MULTIMER_CONFIG["model"]["extra_msa_stack"]["opm_first"] = True


class OpenFold2Config(PretrainedConfig):
    model_type = "openfold2"

    def __init__(self,
                 evoformer_stack_config: EvoformerStackConfig = None,
                 extra_msa_stack_config: ExtraMSAStackConfig = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.evoformer_stack_config = evoformer_stack_config
        self.extra_msa_stack_config = extra_msa_stack_config

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        pretrained_config: dict = None,
                        is_multimer: bool = False,
                        **kwargs):
        if pretrained_config is None:
            if is_multimer:
                pretrained_config = PRETRAINED_OF2_MULTIMER_CONFIG
            else:
                pretrained_config = PRETRAINED_OF2_CONFIG

        _c = pretrained_config["model"]["evoformer_stack"]
        evoformer_stack_config = EvoformerStackConfig(
            architecture="evoformer",
            c_m=_c["c_m"],
            c_z=_c["c_z"],
            c_s=_c["c_s"],
            c_hidden_msa_att=_c["c_hidden_msa_att"],
            c_hidden_opm=_c["c_hidden_opm"],
            c_hidden_mul=_c["c_hidden_mul"],
            c_hidden_pair_att=_c["c_hidden_pair_att"],
            no_heads_msa=_c["no_heads_msa"],
            no_heads_pair=_c["no_heads_pair"],
            transition_n=_c["transition_n"],
            no_blocks=_c["no_blocks"],
            no_column_attention=_c["no_column_attention"],
            opm_first=_c["opm_first"],
            chunk_size=_c.get("chunk_size", 0),
            support_batch=_c.get("support_batch", True),
            max_batch_size=_c.get("max_batch_size", 1),
            triangle_attn_backend=_c.get("triangle_attn_backend", "VANILLA"),
            dtype=_c.get("dtype", "float32"),
            norm_eps=_c.get("eps", 1e-5),
            mask_inf=_c.get("inf", 1e9),
            n_seq=_c.get("n_seq", 516),
        )
        _c = pretrained_config["model"]["extra_msa_stack"]
        extra_msa_stack_config = ExtraMSAStackConfig(
            architecture="extra_msa",
            c_m=_c["c_m"],
            c_z=_c["c_z"],
            c_hidden_msa_att=_c["c_hidden_msa_att"],
            c_hidden_opm=_c["c_hidden_opm"],
            c_hidden_mul=_c["c_hidden_mul"],
            c_hidden_pair_att=_c["c_hidden_pair_att"],
            no_heads_msa=_c["no_heads_msa"],
            no_heads_pair=_c["no_heads_pair"],
            no_blocks=_c["no_blocks"],
            transition_n=_c["transition_n"],
            opm_first=_c["opm_first"],
            support_batch=_c.get("support_batch", True),
            max_batch_size=_c.get("max_batch_size", 1),
            triangle_attn_backend=_c.get("triangle_attn_backend", "VANILLA"),
            dtype=_c.get("dtype", "float32"),
            norm_eps=_c.get("eps", 1e-5),
            mask_inf=_c.get("inf", 1e9),
            max_msa_size=_c.get("max_msa_size", 5120),
        )
        return cls(evoformer_stack_config=evoformer_stack_config,
                   extra_msa_stack_config=extra_msa_stack_config,
                   **kwargs)
