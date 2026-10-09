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
"""The denoising-step graph's token cap follows the device memory."""

import pytest
import torch

from bionemo_ir.models.boltz2.config import diffusion_graph_config
from bionemo_ir.models.openfold3.config import _diffusion_graph_config as openfold3_diffusion_graph_config
from bionemo_ir.models.protenix.config import _diffusion_graph_config as protenix_diffusion_graph_config


@pytest.mark.parametrize(
    "make_config", [diffusion_graph_config, openfold3_diffusion_graph_config, protenix_diffusion_graph_config]
)
@pytest.mark.parametrize(("total_gib", "expected"), [(80, 2048), (64, 2048), (48, 1024)])
def test_cap_follows_device_memory(monkeypatch, make_config, total_gib, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    properties = type("Properties", (), {"total_memory": total_gib << 30})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    dims = make_config().input_routing_config.input_acceptance_dims
    assert [spec.dim_len_max for spec in dims if spec.name == "num_tokens"] == [expected]
