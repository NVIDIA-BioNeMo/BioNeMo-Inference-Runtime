# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from pdb import set_trace as bp
from tensorrt_bionemo._torch.layers.linear import Linear,TensorParallelMode, WeightMode, WeightsLoadingConfig

class SampleModule(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype
        self.linear1 = nn.Linear(in_features=10, out_features=10, dtype=dtype)
        self.linear2 = nn.Linear(in_features=20, out_features=10, dtype=dtype)
        self.linear3 = nn.Linear(in_features=30, out_features=10, dtype=dtype)
        self.linear4 = nn.Linear(in_features=40, out_features=10, dtype=dtype)

    def forward(self, x1: torch.Tensor,
                      x2: torch.Tensor,
                      x3: torch.Tensor,
                      x4: torch.Tensor) -> torch.Tensor:

        x = self.linear1(x1)
        x += self.linear2(x2)
        x += self.linear3(x3)
        x += self.linear4(x4)
        return x

class MergeModule(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear = Linear(in_features=10 + 20 + 30 + 40, 
                             out_features=10,
                             bias=True,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             dtype=dtype,
                             mapping=None,
                             gather_output=True,
                             skip_create_weights=False,
                             weights_loading_config=WeightsLoadingConfig(
                                weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM))
    
    def load_weights(self, state_dict):
        merge_weights = []
        for i in range(len(state_dict.keys())//2):
            merge_weights.append({
                'weight': state_dict[f'linear{i+1}.weight'],
                'bias': state_dict[f'linear{i+1}.bias'],
            })

        self.linear.load_weights(merge_weights)  
    
    def forward(self, x1: torch.Tensor,
                      x2: torch.Tensor,
                      x3: torch.Tensor,
                      x4: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.linear(x)
        return x

@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1024
    dtype: torch.dtype = torch.float32

@pytest.mark.parametrize("sc", [
    Scenario(dtype=torch.float32),
    Scenario(dtype=torch.bfloat16),
])
def test_fused_all_linear_last_dim_mode(sc: Scenario):
    
    model = SampleModule(dtype=sc.dtype).cuda().eval()
    merge_model = MergeModule(dtype=sc.dtype).cuda().eval()
    merge_model.load_weights(model.state_dict())

    x1 = torch.randn(sc.batch_size, 10).to(sc.dtype).cuda()
    x2 = torch.randn(sc.batch_size, 20).to(sc.dtype).cuda()
    x3 = torch.randn(sc.batch_size, 30).to(sc.dtype).cuda()
    x4 = torch.randn(sc.batch_size, 40).to(sc.dtype).cuda()
    
    with torch.no_grad():
        x = model(x1, x2, x3, x4)
        x_merged = merge_model(x1, x2, x3, x4)

    if sc.dtype == torch.float32:
        assert torch.allclose(x, x_merged, atol=1e-3, rtol=1e-3)
    elif sc.dtype == torch.bfloat16:
        assert torch.allclose(x, x_merged, atol=5e-2, rtol=5e-2)