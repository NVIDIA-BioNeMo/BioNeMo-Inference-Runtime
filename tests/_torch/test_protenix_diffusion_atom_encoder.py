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
"""OSS equivalence tests for the coordinate-conditioned atom encoder."""
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.layers.linear import WeightMode
from tensorrt_bionemo._torch.modules.protenix import \
    ProtenixAtomAttentionEncoder
from tensorrt_bionemo.configs import DiffusionTransformerConfig
from tensorrt_bionemo.models.protenix.config import \
    DiffusionAtomAttentionEncoderConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_diffusion_atom_encoder_torch
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import (
    RefProtenixAtomAttentionEncoderFromOSS, update_input_feature_dict)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atom: int = 40
    n_token: int = 10
    n_sample: int = 2
    c_token: int = 32
    c_atom: int = 128
    c_atompair: int = 16
    c_s: int = 32
    c_z: int = 16
    n_blocks: int = 2
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    dtype: str = "float32"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _atom_transformer_config(sc: "Scenario") -> DiffusionTransformerConfig:
    """The Protenix local/atom DiffusionTransformer variant flags."""
    return DiffusionTransformerConfig(num_blocks=sc.n_blocks,
                                      num_heads=sc.n_heads,
                                      dim=sc.c_atom,
                                      dim_single_cond=sc.c_atom,
                                      dim_pairwise=sc.c_atompair,
                                      bias_proj=True,
                                      pair_norm=True,
                                      initial_norm=False,
                                      attention_initial_norm=False,
                                      use_ada_layer_norm=True,
                                      use_separate_layer_norm=True,
                                      chain_kv_norm=True,
                                      attn_output_gate=True,
                                      conditioned_transition_using_silu=True,
                                      transition_expansion_factor=2,
                                      pairwise_attention_backend="SDPA")


def _make_features(device: torch.device, n_atom: int, n_token: int) -> dict:
    torch.manual_seed(11)
    atom_to_token_idx = (torch.arange(n_atom, device=device) //
                         (n_atom // n_token)).long()
    element_idx = torch.randint(0, 128, (n_atom, ), device=device)
    name_chars_idx = torch.randint(0, 64, (n_atom, 4), device=device)
    features = {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": torch.randn(n_atom, 3, device=device),
        "ref_charge": torch.zeros(n_atom, device=device),
        "ref_mask": torch.ones(n_atom, device=device),
        "ref_element": F.one_hot(element_idx, 128).float(),
        "ref_atom_name_chars": F.one_hot(name_chars_idx, 64).float(),
        "ref_space_uid": atom_to_token_idx.clone(),
    }
    return update_input_feature_dict(features)


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32"),
    Scenario(dtype="bfloat16"),
],
                         ids=["fp32", "bf16"])
def test_protenix_diffusion_atom_encoder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    S = sc.n_sample

    ref = RefProtenixAtomAttentionEncoderFromOSS.build(
        has_coords=True,
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_blocks=sc.n_blocks,
        n_heads=sc.n_heads,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys).to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)

    config = DiffusionAtomAttentionEncoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=_atom_transformer_config(sc),
        dtype=sc.dtype)
    model = ProtenixAtomAttentionEncoder(config).to(device).eval()
    converted = convert_diffusion_atom_encoder_torch(config,
                                                     ref.state_dict(),
                                                     prefix="")
    assert model.linear_no_bias_ref.weights_loading_config.weight_mode == \
        WeightMode.FUSED_ALL_LINEAR_LAST_DIM
    assert model.linear_no_bias_pair.weights_loading_config.weight_mode == \
        WeightMode.FUSED_ALL_LINEAR_LAST_DIM
    for target, sources in {
            "linear_no_bias_ref.weight": ("ref_pos", "ref_charge", "f"),
            "linear_no_bias_pair.weight": ("d", "invd", "v"),
    }.items():
        expected = torch.cat([
            ref.state_dict()[f"linear_no_bias_{source}.weight"]
            for source in sources
        ],
                             dim=-1).to(torch_dtype)
        torch.testing.assert_close(converted[target], expected)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    feats = _make_features(device, sc.n_atom, sc.n_token)
    r_l = torch.randn(S, sc.n_atom, 3, device=device)
    s = torch.randn(S, sc.n_token, sc.c_s, device=device)
    z = torch.randn(S, sc.n_token, sc.n_token, sc.c_z, device=device)

    def _oss_args():
        return dict(atom_to_token_idx=feats["atom_to_token_idx"],
                    ref_pos=feats["ref_pos"],
                    ref_charge=feats["ref_charge"],
                    ref_mask=feats["ref_mask"],
                    ref_atom_name_chars=feats["ref_atom_name_chars"],
                    ref_element=feats["ref_element"],
                    d_lm=feats["d_lm"],
                    v_lm=feats["v_lm"],
                    pad_info=feats["pad_info"])

    def _batched(t):
        return t.unsqueeze(0).to(torch_dtype) if t.is_floating_point() \
            else t.unsqueeze(0)

    with torch.inference_mode():
        exp = ref(**_oss_args(), r_l=r_l, s=s, z=z)  # unbatched over N_sample
        act = model(atom_to_token_idx=feats["atom_to_token_idx"].unsqueeze(0),
                    ref_pos=_batched(feats["ref_pos"]),
                    ref_charge=_batched(feats["ref_charge"]),
                    ref_mask=_batched(feats["ref_mask"]),
                    ref_atom_name_chars=_batched(feats["ref_atom_name_chars"]),
                    ref_element=_batched(feats["ref_element"]),
                    d_lm=_batched(feats["d_lm"]),
                    v_lm=_batched(feats["v_lm"]),
                    pad_info=feats["pad_info"],
                    r_l=r_l.unsqueeze(0).to(torch_dtype),
                    s=s.unsqueeze(0).to(torch_dtype),
                    z=z.unsqueeze(0).to(torch_dtype))
    act = tuple(t.squeeze(0) for t in act)

    tol = 3e-3 if torch_dtype == torch.float32 else 6e-2
    for name, a, e in zip(("a", "q_l", "c_l", "p_lm"), act, exp, strict=True):
        r = _rmse_ratio(a, e)
        assert r < tol, f"{name} rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
