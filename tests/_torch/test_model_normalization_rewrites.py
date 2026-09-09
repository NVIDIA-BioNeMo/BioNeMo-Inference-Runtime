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

import pytest
import torch.nn as nn

from bionemo_ir._torch.layers.normalization import FusedLayerNorm, HighPrecisionLayerNorm
from bionemo_ir.models.boltz1 import Boltz1
from bionemo_ir.models.boltz2 import Boltz2, Boltz2Affinity
from bionemo_ir.models.openfold2 import OpenFold2
from bionemo_ir.models.openfold3 import OpenFold3
from bionemo_ir.models.protenix import Protenix


@pytest.mark.parametrize(
    ("model_cls", "keeps_high_precision"),
    [
        pytest.param(OpenFold2, False, id="openfold2"),
        pytest.param(OpenFold3, True, id="openfold3"),
        pytest.param(Boltz1, False, id="boltz1"),
        pytest.param(Boltz2, False, id="boltz2"),
        pytest.param(Boltz2Affinity, False, id="boltz2-affinity"),
        pytest.param(Protenix, False, id="protenix"),
    ],
)
def test_supported_models_replace_plain_layernorm(
    model_cls: type[nn.Module],
    keeps_high_precision: bool,
) -> None:
    model = model_cls(include_load_weights=False)

    assert any(isinstance(module, FusedLayerNorm) for module in model.modules())
    assert not any(type(module) is nn.LayerNorm for module in model.modules())
    assert any(isinstance(module, HighPrecisionLayerNorm) for module in model.modules()) is keeps_high_precision
