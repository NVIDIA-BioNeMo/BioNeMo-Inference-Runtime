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

from unittest.mock import Mock

import pytest
import torch

from bionemo_ir.models.boltz2 import convert
from bionemo_ir.models.boltz2.config import ConfidenceModuleConfig


@pytest.mark.parametrize("supplied_weights", [True, False])
def test_confidence_reuses_checkpoint(supplied_weights: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "dist_bin_pairwise_embed.weight",
        "s_to_z.weight",
        "s_to_z_transpose.weight",
        "s_to_z_prod_in1.weight",
        "s_to_z_prod_in2.weight",
        "s_to_z_prod_out.weight",
        "s_inputs_norm.weight",
        "s_norm.weight",
        "z_norm.weight",
        "s_input_to_s.weight",
        "token_bonds.weight",
        "token_bonds_type.weight",
        "rel_pos.linear_layer.weight",
        "contact_conditioning.fourier_embedding.proj.weight",
        "contact_conditioning.fourier_embedding.proj.bias",
        "contact_conditioning.encoder.weight",
        "contact_conditioning.encoder.bias",
        "contact_conditioning.encoding_unspecified",
        "contact_conditioning.encoding_unselected",
    ]
    weights = {f"confidence_module.{name}": torch.randn(1) for name in names}
    pairformer_weights = {"marker": [{"weight": torch.randn(1)}]}
    loader = Mock(return_value=weights)
    pairformer = Mock(return_value=pairformer_weights)
    monkeypatch.setattr(convert, "load_weights", loader)
    monkeypatch.setattr(convert, "boltz1_convert_hf_pairformer_torch", pairformer)
    config = ConfidenceModuleConfig()

    actual = convert.convert_hf_confidence_module_torch(
        config, local_checkpoint="custom.ckpt", weights=weights if supplied_weights else None
    )

    if supplied_weights:
        loader.assert_not_called()
    else:
        loader.assert_called_once_with(name="boltz-2", cache_path="custom.ckpt")
    pairformer.assert_called_once_with(
        config=config.pairformer,
        local_checkpoint=None,
        weights=weights,
        pairformer_type="confidence",
        prefix="confidence_module.pairformer_stack.",
    )
    assert pairformer.call_args.kwargs["weights"] is weights
    assert actual["pairformer_stack.marker"] is pairformer_weights["marker"]
    assert actual["dist_bin_pairwise_embed"][0]["weight"] is weights["confidence_module.dist_bin_pairwise_embed.weight"]
