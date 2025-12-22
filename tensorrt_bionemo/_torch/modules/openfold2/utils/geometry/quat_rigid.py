# Copyright 2021 DeepMind Technologies Limited
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.modules.openfold2.utils.geometry.rigid_matrix_vector import \
    Rigid3Array
from tensorrt_bionemo._torch.modules.openfold2.utils.geometry.rotation_matrix import \
    Rot3Array
from tensorrt_bionemo._torch.modules.openfold2.utils.geometry.vector import \
    Vec3Array
from tensorrt_bionemo.mapping import Mapping


class QuatRigid(nn.Module):

    def __init__(self,
                 c_hidden: int,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):

        super(QuatRigid, self).__init__()

        self.linear = Linear(c_hidden,
                             6,
                             bias=True,
                             dtype=dtype,
                             mapping=mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=skip_create_weights)

    def forward(self, activations: torch.Tensor) -> Rigid3Array:
        rigid_flat = self.linear(activations)

        rigid_flat = torch.unbind(rigid_flat, dim=-1)

        qx, qy, qz = rigid_flat[:3]
        qw = torch.ones_like(qx)
        translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(
            qw,
            qx,
            qy,
            qz,
            normalize=True,
        )
        translation = Vec3Array(*translation)
        return Rigid3Array(rotation, translation)
