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

from tensorrt_bionemo.mapping import Mapping


def test_mapping():
    m = Mapping(world_size=8, gpus_per_node=8, tp_size=8)
    assert len(m.tp_groups) == 1
    assert len(m.pp_groups) == 8
    assert m.tp_group == [0, 1, 2, 3, 4, 5, 6, 7]

    m = Mapping(world_size=8,
                rank=0,
                gpus_per_node=8,
                tp_size=2,
                dp_size=2,
                pp_size=2)
    assert len(m.tp_groups) == 4
    assert len(m.dp_groups) == 4
    assert len(m.pp_groups) == 4
    assert m.tp_group == [0, 1]
    assert m.dp_group == [0, 2]
    assert m.pp_group == [0, 4]
    assert m.is_first_pp_rank()
    assert m.prep_pp_rank() == 4
    assert m.next_pp_rank() == 4

    m = Mapping(world_size=8,
                rank=5,
                gpus_per_node=8,
                tp_size=2,
                dp_size=4,
                pp_size=1)
    assert len(m.tp_groups) == 4
    assert len(m.dp_groups) == 2
    assert len(m.pp_groups) == 8
    assert m.tp_group == [4, 5]
    assert m.dp_group == [1, 3, 5, 7]
    assert m.tp_rank == 1
    assert m.dp_rank == 2
    assert m.pp_rank == 0
