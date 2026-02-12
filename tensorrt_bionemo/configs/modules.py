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
from typing import Any, Optional

from tensorrt_llm_lite._utils import str_dtype_to_trt

from .base import (BaseConfig, BuildConfig, DimSpec,
                   create_optimization_profiles)


class PairformerConfig(BaseConfig):
    token_s: Optional[int] = None
    token_z: int = None
    pairwise_head_width: int = None
    pairwise_num_heads: int = None
    num_blocks: int = None
    num_heads: int = None
    max_transition_tp_size: bool = True
    max_attention_pairwise_tp_size: bool = True
    max_tri_mul_tp_size: bool = True
    triangle_attn_node_chunk_size: int = 0
    no_update_s: bool = False
    no_update_z: bool = False
    s_path_dtype: Optional[str] = None
    post_layer_norm: Optional[bool] = False
    triangle_attn_cueq_fallback_threshold: int = 0
    trimul_high_precision: bool = False
    attention_initial_norm: Optional[bool] = True
    version: str = "v1"


class PairformerBuildConfig(BuildConfig):
    """ TensorRT building configurations for Pairformer """

    def get_input_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        seqlen = DimSpec(name="seqlen", dynamic=True)

        if mc.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("s", (batch_size, seqlen,
                       DimSpec(size=mc.token_s, name="token_s"))),
                ("z", (batch_size, seqlen, seqlen,
                       DimSpec(size=mc.token_z, name="token_z"))),
                ("mask", (batch_size, seqlen)),
                ("pair_mask", (batch_size, seqlen, seqlen)),
            ])
        return OrderedDict([
            ("s", (seqlen, DimSpec(size=mc.token_s, name="token_s"))),
            ("z", (seqlen, seqlen, DimSpec(size=mc.token_z, name="token_z"))),
            ("mask", (seqlen, )),
            ("pair_mask", (seqlen, seqlen)),
        ])

    def get_output_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        seqlen = DimSpec(name="seqlen", dynamic=True)
        return OrderedDict([
            ("output_s", (seqlen, DimSpec(size=mc.token_s, name="token_s"))),
            ("output_z", (seqlen, seqlen,
                          DimSpec(size=mc.token_z, name="token_z"))),
        ])

    def get_optimization_profiles(self) -> list[Any]:
        return create_optimization_profiles(self)


class DiffusionTransformerConfig(BaseConfig):
    num_blocks: int = None
    num_heads: int = None
    dim: int = None
    dim_single_cond: int = None
    dim_pairwise: Optional[int] = None
    expansion_factor: int = None
    multiplicity: Optional[int] = 1
    attention_initial_norm: Optional[bool] = None
    post_layer_norm: Optional[bool] = None
    conditioned_transition_using_silu: Optional[bool] = None
    bias_proj: Optional[bool] = None


class DiffusionTransformerBuildConfig(BuildConfig):
    """ TensorRT building configurations for DiffusionTransformer """

    def get_input_shapes(self) -> OrderedDict[str, DimSpec]:
        # TODO: Support batch dimension
        mc = self.module_config
        seqlen = DimSpec(name="seqlen", dynamic=True)
        multiplicity = DimSpec(name="multiplicity", dynamic=True)
        dim = DimSpec(name="dim", size=mc.dim)
        dim_single_cond = DimSpec(name="dim_single_cond",
                                  size=mc.dim_single_cond)
        dim_pairwise = DimSpec(name="dim_pairwise", size=mc.dim_pairwise)
        # num_heads = DimSpec(name="num_heads", size=mc.num_heads)
        # num_blocks = DimSpec(name="num_blocks", size=mc.num_blocks)
        heads_times_blocks = DimSpec(name="heads_times_blocks",
                                     size=mc.num_heads * mc.num_blocks)

        if mc.version == "v2":
            z_shape = (DimSpec(size=1, name="n_seqs"), seqlen, seqlen,
                       heads_times_blocks)
        elif mc.version == "v1":
            z_shape = (DimSpec(size=1,
                               name="n_seqs"), seqlen, seqlen, dim_pairwise)
        else:
            raise ValueError(f"Invalid version: {mc.version}")

        return OrderedDict([
            ("a", (multiplicity, seqlen, dim)),
            ("s", (multiplicity, seqlen, dim_single_cond)),
            ("z", z_shape),
            ("mask", (multiplicity, seqlen)),
        ])

    def get_output_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        seqlen = DimSpec(name="seqlen", dynamic=True)
        multiplicity = DimSpec(name="multiplicity", dynamic=True)
        dim = DimSpec(name="dim", size=mc.dim)
        return OrderedDict([("output_a", (multiplicity, seqlen, dim))])

    def get_optimization_profiles(self) -> list[Any]:
        return create_optimization_profiles(self)


class MSAModuleConfig(BaseConfig):
    msa_s: int = None
    token_z: int = None
    token_s: int = None
    msa_blocks: int = None
    num_tokens: int = None
    pairwise_head_width: int = None
    pairwise_num_heads: int = None
    use_paired_feature: bool = True
    opm_chunk_size: int = None
    opm_mask_chunk_size: int = None
    version: str = "v1"


class EvoformerStackConfig(BaseConfig):
    c_m: int = None
    c_z: int = None
    c_s: int = None
    c_hidden_msa_att: int = None
    c_hidden_opm: int = None
    c_hidden_mul: int = None
    c_hidden_pair_att: int = None
    no_heads_msa: int = None
    no_heads_pair: int = None
    transition_n: int = None
    no_blocks: int = None
    no_column_attention: bool = False
    opm_first: bool = False
    chunk_size: int = 0
    n_seq: int = 516
    trimul_high_precision: bool = False
    opm_chunk_size: Optional[int] = None
    opm_mask_chunk_size: Optional[int] = None


class EvoformerStackBuildConfig(BuildConfig):
    """ TensorRT building configurations for EvoformerStack """

    def get_input_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        n_res = DimSpec(name="n_res", dynamic=True)
        n_seq = DimSpec(name="n_seq", size=mc.n_seq)
        c_m = DimSpec(size=mc.c_m, name="c_m")
        c_z = DimSpec(size=mc.c_z, name="c_z")

        if mc.support_batch:
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

    def get_output_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        n_res = DimSpec(name="n_res", dynamic=True)
        n_seq = DimSpec(name="n_seq", size=mc.n_seq)
        c_m = DimSpec(size=mc.c_m, name="c_m")
        c_z = DimSpec(size=mc.c_z, name="c_z")
        c_s = DimSpec(size=mc.c_s, name="c_s")

        if mc.support_batch:
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

    def get_optimization_profiles(self) -> list[Any]:
        return create_optimization_profiles(self, seqlen_key_names=["n_res"])


class ExtraMSAStackConfig(BaseConfig):
    c_m: int = None
    c_z: int = None
    c_hidden_msa_att: int = None
    c_hidden_opm: int = None
    c_hidden_mul: int = None
    c_hidden_pair_att: int = None
    no_heads_msa: int = None
    no_heads_pair: int = None
    no_blocks: int = None
    transition_n: int = None
    opm_first: bool = False
    support_batch: bool = True
    chunk_size: int = 0
    opm_chunk_size: Optional[int] = None
    opm_mask_chunk_size: Optional[int] = None
    max_msa_size: int = 5120
    padding_inputs: bool = True
    trimul_high_precision: bool = False


# This is a module for the affinity module in Boltz-2, will be deprecated in the future
class AffinityModuleConfig(BaseConfig):
    token_s: int = None
    token_z: int = None
    num_dist_bins: int = None
    max_dist: int = None
    pairformer_num_blocks: int = None
    pairwise_head_width: int = None
    pairwise_num_heads: int = None


class AffinityModuleBuildConfig(BuildConfig):

    def get_input_dtypes(self) -> dict[str, str]:
        return {
            "s": str_dtype_to_trt(self.dtype),
            "z": str_dtype_to_trt(self.dtype),
            "distogram": str_dtype_to_trt("int32"),
            "cross_pair_mask_0": str_dtype_to_trt(self.dtype),
            "cross_pair_mask_1": str_dtype_to_trt(self.dtype),
        }

    def get_input_shapes(self) -> OrderedDict[str, DimSpec]:
        mc = self.module_config
        batch_size = DimSpec(name="batch_size", dynamic=True)
        seqlen = DimSpec(name="seqlen", dynamic=True)
        token_s = DimSpec(name="token_s", size=mc.token_s)
        token_z = DimSpec(name="token_z", size=mc.token_z)

        return OrderedDict([
            ("s", (batch_size, seqlen, token_s)),
            ("z", (batch_size, seqlen, seqlen, token_z)),
            ("distogram", (batch_size, seqlen, seqlen)),
            ("cross_pair_mask_0", (batch_size, seqlen, seqlen)),
            ("cross_pair_mask_1", (batch_size, seqlen, seqlen,
                                   DimSpec(size=1, name="const_1"))),
        ])

    def get_output_shapes(self) -> OrderedDict[str, DimSpec]:
        batch_size = DimSpec(name="batch_size", dynamic=True)
        return OrderedDict([
            ("pred_value", (batch_size, 1)),
            ("logits_binary", (batch_size, 1)),
        ])

    def get_optimization_profiles(self) -> list[Any]:
        return create_optimization_profiles(self)
