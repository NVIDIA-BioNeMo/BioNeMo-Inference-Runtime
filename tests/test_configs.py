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

from tensorrt_bionemo.models.boltz1.configs import (PairformerBuildConfig,
                                                    PairformerConfig)


def test_pairformer_build_config_no_batch():
    config = PairformerConfig(architecture="pairformer",
                              dtype="float32",
                              support_batch=False)
    build_config = PairformerBuildConfig(
        force_num_profiles=4,
        max_seqlen=2040,
        min_seqlen=65,
        align=32,
        module_config=config,
    )

    opt_profiles = build_config.optimization_profiles

    assert len(opt_profiles) == build_config.force_num_profiles

    opt0 = opt_profiles[0]
    assert opt0["s"] == ([64, 384], [560, 384], [560, 384])
    assert opt0["z"] == ([64, 64, 128], [560, 560, 128], [560, 560, 128])
    assert opt0["mask"] == ([64], [560], [560])
    assert opt0["pair_mask"] == ([64, 64], [560, 560], [560, 560])

    opt1 = opt_profiles[1]
    assert opt1["s"] == ([560, 384], [1056, 384], [1056, 384])
    assert opt1["z"] == ([560, 560, 128], [1056, 1056, 128], [1056, 1056, 128])
    assert opt1["mask"] == ([560], [1056], [1056])
    assert opt1["pair_mask"] == ([560, 560], [1056, 1056], [1056, 1056])

    opt2 = opt_profiles[2]
    assert opt2["s"] == ([1056, 384], [1552, 384], [1552, 384])
    assert opt2["z"] == ([1056, 1056, 128], [1552, 1552,
                                             128], [1552, 1552, 128])
    assert opt2["mask"] == ([1056], [1552], [1552])
    assert opt2["pair_mask"] == ([1056, 1056], [1552, 1552], [1552, 1552])

    opt3 = opt_profiles[3]
    assert opt3["s"] == ([1552, 384], [2048, 384], [2048, 384])
    assert opt3["z"] == ([1552, 1552, 128], [2048, 2048,
                                             128], [2048, 2048, 128])
    assert opt3["mask"] == ([1552], [2048], [2048])
    assert opt3["pair_mask"] == ([1552, 1552], [2048, 2048], [2048, 2048])


def test_pairformer_build_config_with_batch():
    config = PairformerConfig(architecture="pairformer",
                              dtype="float32",
                              support_batch=True)
    build_config = PairformerBuildConfig(
        force_num_profiles=4,
        max_seqlen=2040,
        min_seqlen=65,
        align=32,
        module_config=config,
    )

    opt_profiles = build_config.optimization_profiles

    assert len(opt_profiles) == build_config.force_num_profiles

    opt0 = opt_profiles[0]
    assert opt0["s"] == ([1, 64, 384], [1, 560, 384], [1, 560, 384])
    assert opt0["z"] == ([1, 64, 64, 128], [1, 560, 560,
                                            128], [1, 560, 560, 128])
    assert opt0["mask"] == ([1, 64], [1, 560], [1, 560])
    assert opt0["pair_mask"] == ([1, 64, 64], [1, 560, 560], [1, 560, 560])

    opt1 = opt_profiles[1]
    assert opt1["s"] == ([1, 560, 384], [1, 1056, 384], [1, 1056, 384])
    assert opt1["z"] == ([1, 560, 560, 128], [1, 1056, 1056,
                                              128], [1, 1056, 1056, 128])
    assert opt1["mask"] == ([1, 560], [1, 1056], [1, 1056])
    assert opt1["pair_mask"] == ([1, 560, 560], [1, 1056,
                                                 1056], [1, 1056, 1056])

    opt2 = opt_profiles[2]
    assert opt2["s"] == ([1, 1056, 384], [1, 1552, 384], [1, 1552, 384])
    assert opt2["z"] == ([1, 1056, 1056, 128], [1, 1552, 1552,
                                                128], [1, 1552, 1552, 128])
    assert opt2["mask"] == ([1, 1056], [1, 1552], [1, 1552])
    assert opt2["pair_mask"] == ([1, 1056, 1056], [1, 1552,
                                                   1552], [1, 1552, 1552])

    opt3 = opt_profiles[3]
    assert opt3["s"] == ([1, 1552, 384], [1, 2048, 384], [1, 2048, 384])
    assert opt3["z"] == ([1, 1552, 1552, 128], [1, 2048, 2048,
                                                128], [1, 2048, 2048, 128])
    assert opt3["mask"] == ([1, 1552], [1, 2048], [1, 2048])
    assert opt3["pair_mask"] == ([1, 1552, 1552], [1, 2048,
                                                   2048], [1, 2048, 2048])
