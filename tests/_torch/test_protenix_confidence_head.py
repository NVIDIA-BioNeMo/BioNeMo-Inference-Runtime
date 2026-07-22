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
"""OSS equivalence tests for the Protenix confidence head (AF3 Algorithm 31).

Real-checkpoint tests skip when the checkpoint is unavailable.
"""
import functools
import os
from dataclasses import dataclass

import pytest
import torch

from tensorrt_bionemo._torch.modules.protenix import ProtenixConfidenceHead
from tensorrt_bionemo.configs import PairformerConfig
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.models.protenix.config import ConfidenceHeadConfig
from tensorrt_bionemo.models.protenix.convert import \
    convert_confidence_head_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import \
    RefProtenixConfidenceHeadFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 12
    atoms_per_token: int = 4
    n_sample: int = 2
    dtype: str = "float32"  # head precision (fp32, matching OSS)
    pairformer_dtype: str = "float32"  # inner pairformer precision


# Full protenix-v2 confidence dimensions.
_FULL = dict(n_blocks=4,
             c_s=384,
             c_z=256,
             c_s_inputs=449,
             max_atoms_per_token=24,
             hidden_scale_up=True,
             distance_bin_start=3.25,
             distance_bin_end=52.0,
             distance_bin_step=1.25,
             stop_gradient=True)


@functools.lru_cache(maxsize=1)
def _weights() -> dict:
    """Real protenix-v2 weights via the hub (env var or HF auto-download)."""
    return load_weights_from_hubs(SupMat.ProtenixV2, local_files_only=False)


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean(
        (a - b)**2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _config(sc: Scenario) -> ConfidenceHeadConfig:
    """Build the full-size confidence config."""
    pf = PairformerConfig(token_s=_FULL["c_s"],
                          token_z=_FULL["c_z"],
                          num_blocks=_FULL["n_blocks"],
                          num_heads=16,
                          pairwise_head_width=32,
                          pairwise_num_heads=_FULL["c_z"] // 32,
                          no_update_s=False,
                          attention_initial_norm=True,
                          version="v1",
                          dtype=sc.pairformer_dtype,
                          triangle_attention_backend="VANILLA",
                          pairwise_attention_backend="SDPA")
    return ConfidenceHeadConfig(dtype=sc.dtype, pairformer_config=pf)


def _features(device: torch.device, n_token: int,
              atoms_per_token: int) -> dict:
    """Build contiguous atoms with one representative per token."""
    n_atom = n_token * atoms_per_token
    idx = torch.arange(n_atom, device=device)
    atom_to_token_idx = (idx // atoms_per_token).long()
    atom_to_tokatom_idx = (idx % atoms_per_token).long()
    return {
        "distogram_rep_atom_mask": (atom_to_tokatom_idx == 0),
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
    }


@pytest.fixture(scope="module")
def real_case():
    """Build the full-size OSS reference case once."""
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.manual_seed(42)
    try:
        weights = _weights()
    except Exception as exc:  # offline / hub unreachable
        pytest.skip(f"protenix-v2 checkpoint unavailable: {exc}")
    device = torch.device("cuda")

    ref = RefProtenixConfidenceHeadFromOSS.build(**_FULL).to(
        device=device, dtype=torch.float32).eval()
    ref.load_state_dict(
        {
            k[len("confidence_head."):]: v
            for k, v in weights.items() if k.startswith("confidence_head.")
        },
        strict=True)

    sc = Scenario()
    n_token, n_atom = sc.n_token, sc.n_token * sc.atoms_per_token
    feats = _features(device, sc.n_token, sc.atoms_per_token)
    s_inputs = torch.randn(1, n_token, _FULL["c_s_inputs"], device=device)
    s_trunk = torch.randn(1, n_token, _FULL["c_s"], device=device)
    z_trunk = torch.randn(1, n_token, n_token, _FULL["c_z"], device=device)
    x_pred = torch.randn(1, sc.n_sample, n_atom, 3, device=device)
    pair_mask = torch.ones(1, n_token, n_token, device=device)
    with torch.inference_mode():
        plddt, pae, pde, resolved = ref(feats,
                                        s_inputs,
                                        s_trunk,
                                        z_trunk,
                                        pair_mask,
                                        x_pred,
                                        triangle_multiplicative="torch",
                                        triangle_attention="torch")
    return dict(weights=weights,
                feats=feats,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                x_pred=x_pred,
                pair_mask=pair_mask,
                device=device,
                exp=dict(plddt_logits=plddt,
                         pae_logits=pae,
                         pde_logits=pde,
                         resolved_logits=resolved))


@pytest.mark.parametrize("sc", [
    Scenario(dtype="float32", pairformer_dtype="float32"),
    Scenario(dtype="float32", pairformer_dtype="bfloat16"),
],
                         ids=["fp32", "bf16_pairformer"])
def test_protenix_confidence_head(sc: Scenario, real_case):
    device = real_case["device"]

    config = _config(sc)
    model = ProtenixConfidenceHead(config).to(device).eval()
    converted = convert_confidence_head_torch(config,
                                              real_case["weights"],
                                              prefix="confidence_head")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (list(missing)[:5],
                                            list(unexpected)[:5])

    with torch.inference_mode():
        act = model(real_case["feats"],
                    real_case["s_inputs"],
                    real_case["s_trunk"],
                    real_case["z_trunk"],
                    real_case["x_pred"],
                    pair_mask=real_case["pair_mask"])

    any_bf16 = "bfloat16" in (sc.dtype, sc.pairformer_dtype)
    tol = 1.5e-1 if any_bf16 else 5e-3
    for key, exp in real_case["exp"].items():
        assert torch.isfinite(act[key]).all(), f"{key} non-finite ({sc.dtype})"
        r = _rmse_ratio(act[key], exp)
        assert r < tol, f"{key} rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"
