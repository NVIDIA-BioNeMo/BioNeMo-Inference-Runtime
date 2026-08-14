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
from test_utils.boltz.create_and_load_weights import create_template_module_weights, load_template_module_weights_torch
from test_utils.boltz.ref_layers import RefTemplateV2Module

from bionemo_ir._torch.modules.boltz.template import TemplateV2Module
from bionemo_ir.models.boltz2.config import TemplateV2ModuleConfig
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_mask
from tests._torch import skip_if_cutedsl as _skip_if_cutedsl


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


def _make_template_feats(
    B: int, T: int, N: int, num_tokens: int, num_bins: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build a dummy template feature dict matching the layout consumed by
    :class:`TemplateV2Module`/``RefTemplateV2Module``.

    Coordinates and frames are float32; mask/visibility ids stay integer-typed
    so the module's own up/down casts mirror real inference.
    """
    template_restype = torch.nn.functional.one_hot(
        torch.randint(0, num_tokens, (B, T, N), device=device), num_classes=num_tokens
    ).float()
    template_frame_rot = torch.eye(3, device=device).expand(B, T, N, 3, 3).contiguous()
    template_frame_t = torch.randn(B, T, N, 3, device=device)
    template_mask_frame = torch.randint(0, 2, (B, T, N), dtype=torch.float32, device=device)
    template_cb = torch.randn(B, T, N, 3, device=device)
    template_ca = torch.randn(B, T, N, 3, device=device)
    template_mask_cb = torch.randint(0, 2, (B, T, N), dtype=torch.float32, device=device)
    # ``visibility_ids`` are integer chain ids (cdist mask). Keep at most a
    # couple of distinct values so the equality test produces a non-trivial
    # mask.
    visibility_ids = torch.randint(0, 3, (B, T, N), device=device)
    # ``template_mask`` is reduced over dim=2 in the module
    # (``feats["template_mask"].any(dim=2)``). Upstream Boltz constructs it
    # as ``(T, N)`` per-sample (in ``boltz/data/feature/featurizerv2.py``),
    # giving ``(B, T, N)`` after batching so the reduction yields a
    # per-template ``(B, T)`` mask.
    template_mask = torch.randint(0, 2, (B, T, N), dtype=torch.float32, device=device)
    return {
        "template_restype": template_restype,
        "template_frame_rot": template_frame_rot,
        "template_frame_t": template_frame_t,
        "template_mask_frame": template_mask_frame,
        "template_cb": template_cb,
        "template_ca": template_ca,
        "template_mask_cb": template_mask_cb,
        "visibility_ids": visibility_ids,
        "template_mask": template_mask,
    }


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        Scenario(torch_dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="CuTeDSL", torch_dtype="bfloat16"),
    ],
)
def test_template_v2_module(sc: Scenario):
    _skip_if_cutedsl(sc.triangle_attn_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    token_z = 64
    template_dim = 32
    template_blocks = 2
    num_tokens = 33
    num_bins = 38

    ref = RefTemplateV2Module(
        token_z=token_z,
        template_dim=template_dim,
        template_blocks=template_blocks,
        num_tokens=num_tokens,
        num_bins=num_bins,
    ).to(device)
    wb = create_template_module_weights(from_ref=ref)

    cfg = TemplateV2ModuleConfig(
        token_z=token_z,
        template_dim=template_dim,
        template_blocks=template_blocks,
        num_tokens=num_tokens,
        num_bins=num_bins,
    )
    cfg.set_dtype(sc.torch_dtype)
    cfg.set_triangle_attention_backend(sc.triangle_attn_backend)

    mod = TemplateV2Module(cfg)
    load_template_module_weights_torch(mod, wb, dtype=dtype)
    mod.to(device)

    B, T, N = 1, 3, 32
    z = torch.randn(B, N, N, token_z, dtype=torch.float32, device=device)
    seq_mask = make_left_aligned_mask(B, N, dtype=torch.float32, device=device, min_valid=N // 2)
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    feats = _make_template_feats(B, T, N, num_tokens, num_bins, device)

    with torch.inference_mode():
        ref_float = ref(z, feats, pair_mask)

        z_dt = z.to(dtype)
        pair_mask_dt = pair_mask.to(dtype)

        ref_cast = ref.to(dtype)
        ref_out = ref_cast(z_dt, feats, pair_mask_dt)

        out = mod(z_dt, feats, pair_mask_dt)

    assert ref_out.shape == out.shape

    # Padded query rows in the inner pairformer's triangle attention softmax
    # over fully-masked keys -> NaN in the PyTorch reference; the CuTeDSL
    # left-mask kernel sees ``actual_s_kv`` clamped to >= 1 and emits
    # arithmetic garbage there. Both implementations agree on the *valid*
    # sub-block (rows/cols where ``seq_mask == 1``), so only compare there.
    keep = pair_mask.float().unsqueeze(-1)  # [B, N, N, 1]

    def _masked(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0) * keep

    if dtype == torch.float32:
        torch.testing.assert_close(_masked(out), _masked(ref_out), atol=1e-3, rtol=1e-4)
    else:
        # Compare bf16 outputs against ref_float using the same statistical
        # pattern as ``test_boltz_msa_module.test_msa_layer``: the BioIR
        # path's deviation from the fp32 reference should be on the same
        # order as the ref-bf16 path's deviation from fp32.
        d_out = _masked(out) - _masked(ref_float)
        d_ref = _masked(ref_out) - _masked(ref_float)

        diff0_max = torch.max(torch.abs(d_out))
        diff0_mean = torch.mean(torch.abs(d_out))
        diff1_max = torch.max(torch.abs(d_ref))
        diff1_mean = torch.mean(torch.abs(d_ref))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2
