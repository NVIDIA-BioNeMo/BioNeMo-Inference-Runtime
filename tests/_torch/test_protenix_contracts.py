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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression guards for Protenix public APIs, converter keys, and config policy."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
import torch.nn as nn

import tensorrt_bionemo._torch.modules.protenix as protenix_modules
import tensorrt_bionemo.models.protenix as protenix_models
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.models.protenix.config import DiffusionModuleConfig, ProtenixConfig, TemplateEmbedderConfig
from tensorrt_bionemo.models.protenix.convert import (
    _convert_protenix_atom_dit_block,
    _convert_protenix_token_dit_block,
    convert_template_embedder_torch,
)
from tensorrt_bionemo.models.protenix.modeling import Protenix

# Cache-only features owned by destructive inference.
_DIFFUSION_CONSUMED_FEATURES = (
    "relp",
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_atom_name_chars",
    "ref_element",
    "d_lm",
    "v_lm",
    "pad_info",
)

_MODULES_PUBLIC = [
    "ProtenixAtomAttentionDecoder",
    "ProtenixAtomAttentionEncoder",
    "ProtenixConfidenceHead",
    "ProtenixConfidenceSummary",
    "ProtenixConstraintEmbedder",
    "ProtenixDiffusionConditioning",
    "ProtenixDiffusionModule",
    "ProtenixDistogramHead",
    "ProtenixSampleDiffusion",
    "ProtenixInputFeatureEmbedder",
    "ProtenixMSAModule",
    "ProtenixTemplateEmbedder",
    "ProtenixTrunk",
]

_MODELS_PUBLIC = [
    "Protenix",
    "ProtenixConfig",
    "InputFeatureEmbedderConfig",
    "PRETRAINED_CONFIG_REGISTRY",
]

# Attributes consumed by weight loading and optimization.
_LOAD_ATTRS = (
    "input_embedder",
    "relative_position_encoding",
    "constraint_embedder",
    "linear_no_bias_sinit",
    "linear_no_bias_zinit1",
    "linear_no_bias_zinit2",
    "linear_no_bias_token_bond",
    "trunk",
    "diffusion_sampler",
    "distogram_head",
    "confidence_head",
    "confidence_summary",
)


def test_protenix_modules_public_api():
    assert list(protenix_modules.__all__) == _MODULES_PUBLIC
    for name in _MODULES_PUBLIC:
        assert getattr(protenix_modules, name) is not None


def test_protenix_models_public_api():
    assert list(protenix_models.__all__) == _MODELS_PUBLIC
    for name in _MODELS_PUBLIC:
        assert getattr(protenix_models, name) is not None


def test_protenix_forward_signature_stable():
    sig = inspect.signature(Protenix.forward)
    assert list(sig.parameters) == [
        "self",
        "batch",
        "recycling_steps",
        "num_sampling_steps",
        "diffusion_samples",
        "compact_output",
        "return_full_data",
        "consume_input_features",
    ]
    defaults = {name: param.default for name, param in sig.parameters.items() if name != "self"}
    assert defaults["recycling_steps"] == 3
    assert defaults["num_sampling_steps"] == 200
    assert defaults["diffusion_samples"] == 1
    assert defaults["compact_output"] is False
    assert defaults["return_full_data"] is True
    assert defaults["consume_input_features"] is False


def test_protenix_load_weight_attribute_names():
    model = Protenix(ProtenixConfig(), include_load_weights=False)
    for name in _LOAD_ATTRS:
        assert hasattr(model, name), name
    assert hasattr(model.diffusion_sampler, "diffusion_module")
    assert hasattr(model.diffusion_sampler.diffusion_module, "diffusion_transformer")
    registry = model.get_optimized_modules({})
    assert "token_transformer" in registry.get_accelerated_modules()


def test_protenix_inference_precision_defaults():
    cfg = Protenix().get_pretrained_config(SupMat.ProtenixV2)
    assert cfg.trunk_config.pair_state_dtype == "float32"
    dm = cfg.diffusion_module_config
    assert dm.token_transformer_config.dtype == "bfloat16"
    assert dm.atom_encoder_config.dtype == "float32"
    assert dm.atom_decoder_config.dtype == "float32"
    assert dm.atom_encoder_config.pairwise_attention_backend == "SDPA"
    assert dm.atom_decoder_config.pairwise_attention_backend == "SDPA"
    assert dm.diffusion_conditioning_config.z_pair_dtype == "bfloat16"
    ch_pf = cfg.confidence_head_config.pairformer_config
    assert ch_pf.dtype == "bfloat16"
    assert cfg.trunk_config.msa_module_config.dtype == "bfloat16"
    assert cfg.trunk_config.pairformer_config.dtype == "bfloat16"


def _synthetic_dit_block_weights(c: int = 8, c_z: int = 4) -> dict[str, torch.Tensor]:
    """Minimal OSS-shaped DiT block weights for converter key/layout checks."""
    w: dict[str, torch.Tensor] = {}

    def ln(prefix: str) -> None:
        w[f"{prefix}.weight"] = torch.ones(c)

    def lin(name: str, out_f: int, in_f: int, bias: bool = False) -> None:
        w[f"{name}.weight"] = torch.randn(out_f, in_f)
        if bias:
            w[f"{name}.bias"] = torch.randn(out_f)

    # Query AdaLN is shared by both variants.
    ln("attention_pair_bias.layernorm_a.layernorm_s")
    lin("attention_pair_bias.layernorm_a.linear_s", c, c, bias=True)
    lin("attention_pair_bias.layernorm_a.linear_nobias_s", c, c)
    # Atom attention adds a key/value AdaLN.
    ln("attention_pair_bias.layernorm_kv.layernorm_s")
    lin("attention_pair_bias.layernorm_kv.linear_s", c, c, bias=True)
    lin("attention_pair_bias.layernorm_kv.linear_nobias_s", c, c)

    attn = "attention_pair_bias.attention"
    lin(f"{attn}.linear_q", c, c, bias=True)
    lin(f"{attn}.linear_k", c, c)
    lin(f"{attn}.linear_v", c, c)
    lin(f"{attn}.linear_g", c, c)
    lin(f"{attn}.linear_o", c, c)
    w["attention_pair_bias.layernorm_z.weight"] = torch.ones(c_z)
    lin("attention_pair_bias.linear_nobias_z", c // 4 if c >= 4 else 1, c_z)
    w["attention_pair_bias.linear_nobias_z.weight"] = torch.randn(2, c_z)
    lin("attention_pair_bias.linear_a_last", c, c, bias=True)

    ct = "conditioned_transition_block"
    ln(f"{ct}.adaln.layernorm_s")
    lin(f"{ct}.adaln.linear_s", c, c, bias=True)
    lin(f"{ct}.adaln.linear_nobias_s", c, c)
    lin(f"{ct}.linear_nobias_a1", c, c)
    lin(f"{ct}.linear_nobias_a2", c, c)
    lin(f"{ct}.linear_nobias_b", c, c)
    lin(f"{ct}.linear_s", c, c, bias=True)
    return w


@pytest.mark.parametrize(
    "converter,tgt,has_kv_adaln",
    [
        (_convert_protenix_atom_dit_block, "layers.0", True),
        (_convert_protenix_token_dit_block, "layers.0", False),
    ],
)
def test_protenix_dit_converter_fusion_layout(converter, tgt, has_kv_adaln):
    c = 8
    src = {f"blk.{k}": v for k, v in _synthetic_dit_block_weights(c).items()}
    out: dict[str, torch.Tensor] = {}
    converter(out, src, "blk", tgt, torch.float32)

    # Fused layouts preserve source order.
    kv = out[f"{tgt}.pair_bias_attn.proj_kv.weight"]
    assert kv.shape[0] == 2 * c
    torch.testing.assert_close(kv[:c], src["blk.attention_pair_bias.attention.linear_k.weight"])
    torch.testing.assert_close(kv[c:], src["blk.attention_pair_bias.attention.linear_v.weight"])

    sw = out[f"{tgt}.transition.fused_swl_a_to_b.weight"]
    torch.testing.assert_close(sw[:c], src["blk.conditioned_transition_block.linear_nobias_a2.weight"])
    torch.testing.assert_close(sw[c:], src["blk.conditioned_transition_block.linear_nobias_a1.weight"])

    if has_kv_adaln:
        assert f"{tgt}.pair_bias_attn.layer_norm_a_q.s_norm.weight" in out
        assert f"{tgt}.pair_bias_attn.layer_norm_a_k.s_norm.weight" in out
        assert f"{tgt}.adaln.s_norm.weight" not in out
    else:
        assert f"{tgt}.adaln.s_norm.weight" in out
        assert f"{tgt}.pair_bias_attn.layer_norm_a_q.s_norm.weight" not in out


def test_protenix_template_pair_path_keys():
    """Template pair blocks must emit the same keys as the shared pair-path path."""
    config = TemplateEmbedderConfig(n_blocks=1, c=8, c_z=8)
    w: dict[str, torch.Tensor] = {}
    for ln in ("layernorm_z", "layernorm_v"):
        w[f"{ln}.weight"] = torch.ones(8)
        w[f"{ln}.bias"] = torch.zeros(8)
    for lin, in_f, out_f in (("linear_no_bias_z", 8, 8), ("linear_no_bias_a", 108, 8), ("linear_no_bias_u", 8, 8)):
        w[f"{lin}.weight"] = torch.randn(out_f, in_f)

    blk = "pairformer_stack.blocks.0"
    for mul in ("tri_mul_out", "tri_mul_in"):
        s = f"{blk}.{mul}"
        w[f"{s}.layer_norm_in.weight"] = torch.ones(8)
        w[f"{s}.layer_norm_in.bias"] = torch.zeros(8)
        w[f"{s}.layer_norm_out.weight"] = torch.ones(8)
        w[f"{s}.layer_norm_out.bias"] = torch.zeros(8)
        w[f"{s}.linear_a_p.weight"] = torch.randn(8, 8)
        w[f"{s}.linear_b_p.weight"] = torch.randn(8, 8)
        w[f"{s}.linear_a_g.weight"] = torch.randn(8, 8)
        w[f"{s}.linear_b_g.weight"] = torch.randn(8, 8)
        w[f"{s}.linear_z.weight"] = torch.randn(8, 8)
        w[f"{s}.linear_g.weight"] = torch.randn(8, 8)
    for att in ("tri_att_start", "tri_att_end"):
        s = f"{blk}.{att}"
        w[f"{s}.layer_norm.weight"] = torch.ones(8)
        w[f"{s}.layer_norm.bias"] = torch.zeros(8)
        w[f"{s}.linear.weight"] = torch.randn(1, 8)  # head bias linear
        for p in ("q", "k", "v", "o", "g"):
            w[f"{s}.mha.linear_{p}.weight"] = torch.randn(8, 8)
    pt = f"{blk}.pair_transition"
    w[f"{pt}.layernorm1.weight"] = torch.ones(8)
    w[f"{pt}.layernorm1.bias"] = torch.zeros(8)
    w[f"{pt}.linear_no_bias_a.weight"] = torch.randn(16, 8)
    w[f"{pt}.linear_no_bias_b.weight"] = torch.randn(16, 8)
    w[f"{pt}.linear_no_bias.weight"] = torch.randn(8, 16)

    out = convert_template_embedder_torch(config, w, prefix="")
    layer = "pairformer_stack.layers.0"
    for key in (
        f"{layer}.tri_mul_out.p_in.weight",
        f"{layer}.tri_mul_in.p_in.weight",
        f"{layer}.tri_attn_start.mha.qkv_proj.weight",
        f"{layer}.tri_attn_end.mha.qkv_proj.weight",
        f"{layer}.transition_z.fused_fc2_fc1.weight",
        f"{layer}.transition_z.fc3.weight",
        "linear_no_bias_a.weight",
        "layernorm_z.weight",
    ):
        assert key in out, key


class _StubTrunk(nn.Module):
    def __init__(self, pair_state_dtype=torch.float32):
        super().__init__()
        self.pair_state_dtype = pair_state_dtype

    def forward(self, batch, s_inputs, s_init, z_init, num_cycles=1):
        return s_init, z_init


class _StubEmbedder(nn.Module):
    def forward(self, batch, attn_metadata=None):
        n = batch["restype"].shape[-2]
        return torch.zeros(1, n, 449)


class _StubRPE(nn.Module):
    def generate_relp(self, **kwargs):
        n = kwargs["asym_id"].shape[-1]
        return torch.zeros(1, n, n, 195)

    def forward(self, relp=None):
        return torch.zeros(*relp.shape[:-1], 256)


class _StubConstraint(nn.Module):
    def forward(self, _):
        return None


class _StubDistogram(nn.Module):
    def forward(self, z):
        return torch.zeros(*z.shape[:-1], 64)


class _StubSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs: dict[str, Any] = {}

    def sample_coords(self, batch, *args, **kwargs):
        self.last_kwargs = dict(kwargs)
        self.last_kwargs["batch_keys"] = set(batch.keys())
        n_atom = batch.get("atom_to_token_idx", torch.zeros(1, 4)).shape[-1]
        return torch.zeros(1, kwargs.get("N_sample", 1), n_atom, 3)


def _tiny_batch(n_token: int = 4, n_atom: int = 4) -> dict[str, torch.Tensor]:
    return {
        "asym_id": torch.zeros(1, n_token),
        "residue_index": torch.arange(n_token).unsqueeze(0),
        "entity_id": torch.zeros(1, n_token),
        "token_index": torch.arange(n_token).unsqueeze(0),
        "sym_id": torch.zeros(1, n_token),
        "restype": torch.zeros(1, n_token, 32),
        "profile": torch.zeros(1, n_token, 32),
        "deletion_mean": torch.zeros(1, n_token, 1),
        "token_bonds": torch.zeros(1, n_token, n_token),
        "ref_pos": torch.zeros(1, n_atom, 3),
        "atom_to_token_idx": torch.arange(n_atom).unsqueeze(0) % n_token,
    }


def test_protenix_forward_output_modes_and_feature_ownership(monkeypatch):
    # Keep this API-only test on CPU.
    model = Protenix(ProtenixConfig(), include_load_weights=False).cpu()
    model.input_embedder = _StubEmbedder()
    model.relative_position_encoding = _StubRPE()
    model.constraint_embedder = _StubConstraint()
    model.trunk = _StubTrunk()
    model.distogram_head = _StubDistogram()
    stub_sampler = _StubSampler()
    model.diffusion_sampler = stub_sampler
    model.generate_attn_metadata = lambda batch: None  # type: ignore[method-assign]

    batch = _tiny_batch()
    owned = set(batch.keys())
    full = model.forward(
        batch,
        recycling_steps=0,
        num_sampling_steps=1,
        diffusion_samples=1,
        compact_output=False,
        consume_input_features=False,
    )
    assert set(full) >= {"s_inputs", "s", "z", "coordinate", "distogram_logits"}
    assert set(batch.keys()) == owned
    assert stub_sampler.last_kwargs["drop_consumed_features"] is False

    batch2 = _tiny_batch()
    compact = model.forward(
        batch2,
        recycling_steps=0,
        num_sampling_steps=1,
        diffusion_samples=1,
        compact_output=True,
        consume_input_features=True,
    )
    assert set(compact) == {"coordinate"}
    assert batch2 == {}
    assert stub_sampler.last_kwargs["drop_consumed_features"] is True
    assert stub_sampler.last_kwargs["drop_consumed_relp"] is True


def test_diffusion_consumed_feature_constant_locked():
    """Lock the feature-ownership tuple used by cache/destructive drop."""
    from tensorrt_bionemo._torch.modules.protenix import diffusion as diff_mod
    from tensorrt_bionemo._torch.modules.protenix._common import DIFFUSION_CONSUMED_FEATURES

    assert tuple(DIFFUSION_CONSUMED_FEATURES) == _DIFFUSION_CONSUMED_FEATURES
    src = inspect.getsource(diff_mod.ProtenixSampleDiffusion.sample_coords)
    assert "DIFFUSION_CONSUMED_FEATURES" in src


def test_diffusion_module_config_defaults_unchanged():
    cfg = DiffusionModuleConfig()
    assert cfg.diffusion_conditioning_config.z_pair_dtype == "bfloat16"
    assert cfg.token_transformer_config is not None
    assert cfg.atom_encoder_config.has_coords is True
    assert cfg.atom_decoder_config is not None
