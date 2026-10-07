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

from bionemo_ir.models.openfold3 import convert
from bionemo_ir.models.openfold3.config import DiffusionModuleConfig, InputEmbedderAllAtomConfig


@pytest.mark.parametrize("supplied_weights", [True, False])
@pytest.mark.parametrize("module", ["input_embedder", "diffusion_module"])
def test_conversion_reuses_checkpoint(module: str, supplied_weights: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    atom_features = [
        "linear_ref_pos",
        "linear_ref_charge",
        "linear_ref_mask",
        "linear_ref_element",
        "linear_ref_atom_chars",
        "linear_ref_offset",
        "linear_inv_sq_dists",
        "linear_valid_mask",
    ]
    names = [f"atom_attn_enc.ref_atom_feature_embedder.{name}" for name in atom_features]
    names.append("atom_attn_enc.atom_transformer.layer_norm_z")
    if module == "input_embedder":
        config = InputEmbedderAllAtomConfig()
        converter = convert.convert_hf_input_embedder_torch
        names.extend(
            f"atom_attn_enc.{name}"
            for name in ["linear_l", "linear_m", "pair_mlp.1", "pair_mlp.3", "pair_mlp.5", "linear_q.0"]
        )
        names.extend(["linear_s", "linear_relpos", "linear_token_bonds", "linear_z_i", "linear_z_j"])
        prefixes = ["atom_attn_enc.atom_transformer"]
    else:
        config = DiffusionModuleConfig()
        converter = convert.convert_hf_diffusion_module_torch
        names.append("atom_attn_dec.atom_transformer.layer_norm_z")
        prefixes = ["atom_attn_enc.atom_transformer", "diffusion_transformer", "atom_attn_dec.atom_transformer"]
    weights = {f"{module}.{name}.weight": torch.ones(1) for name in names}
    transformer_weights = {"marker": [{"weight": torch.ones(1)}]}
    loader = Mock(return_value=weights)
    transformer = Mock(return_value=transformer_weights)
    monkeypatch.setattr(convert, "load_weights", loader)
    monkeypatch.setattr(convert, "convert_hf_diffusion_transformer_torch", transformer)

    actual = converter(config, local_checkpoint="custom.ckpt", weights=weights if supplied_weights else None)

    if supplied_weights:
        loader.assert_not_called()
    else:
        loader.assert_called_once_with(name="openfold3", cache_path="custom.ckpt")
    assert transformer.call_count == len(prefixes)
    for call, prefix in zip(transformer.call_args_list, prefixes, strict=True):
        assert call.kwargs["weights"] is weights
        assert call.kwargs["prefix"] == f"{module}.{prefix}.blocks"
        assert actual[f"{prefix}.marker"] is transformer_weights["marker"]
    key = "atom_attn_enc.atom_transformer.layer_norm_z"
    assert actual[key][0]["weight"] is weights[f"{module}.{key}.weight"]
