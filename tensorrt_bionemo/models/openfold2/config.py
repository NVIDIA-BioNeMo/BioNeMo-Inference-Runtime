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

from tensorrt_bionemo.configs import BaseConfig


class OpenFold2Config(BaseConfig):
    c_z: int = 128
    c_s: int = 384

    trunk: BaseConfig = BaseConfig(
        evoformer_stack=BaseConfig(
            c_m=256,
            c_z=c_z,
            c_s=c_s,
            c_hidden_msa_att=32,
            c_hidden_opm=32,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            no_heads_msa=8,
            no_heads_pair=4,
            transition_n=4,
            no_blocks=48,
            no_column_attention=False,
            opm_first=False,
            n_seq=516,
            trimul_high_precision=False,
        ),
        extra_msa_stack=BaseConfig(
            c_m=64,
            c_z=c_z,
            c_hidden_msa_att=8,
            c_hidden_opm=32,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            no_heads_msa=8,
            no_heads_pair=4,
            opm_first=False,
            transition_n=4,
            trimul_high_precision=False,
        ),
    )


class OpenFold2MultimerConfig(BaseConfig):
    c_z: int = 128
    c_s: int = 384

    trunk: BaseConfig = BaseConfig(
        evoformer_stack=BaseConfig(
            c_m=256,
            c_z=c_z,
            c_s=c_s,
            c_hidden_msa_att=32,
            c_hidden_opm=32,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            no_heads_msa=8,
            no_heads_pair=4,
            transition_n=4,
            no_blocks=48,
            no_column_attention=False,
            opm_first=True,
            n_seq=516,
            trimul_high_precision=False,
        ),
        extra_msa_stack=BaseConfig(
            c_m=64,
            c_z=c_z,
            c_hidden_msa_att=8,
            c_hidden_opm=32,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            no_heads_msa=8,
            no_heads_pair=4,
            opm_first=True,
            transition_n=4,
            trimul_high_precision=False,
        ),
    )
