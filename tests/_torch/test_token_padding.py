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
"""Model token-padding specs pad on token axes, and confidence pairformers run on aligned tokens."""

import pytest
import torch

from bionemo_ir._torch.layers.token_padding import pad_feature_dict, pad_trunk_tokens
from bionemo_ir._torch.modules.boltz.trunk import Trunk
from bionemo_ir._torch.modules.openfold3.confidence import PairformerEmbedding
from bionemo_ir._torch.modules.protenix import ProtenixConfidenceHead
from bionemo_ir.configs import PairformerConfig
from bionemo_ir.models.boltz1.config import Boltz1Config
from bionemo_ir.models.boltz2.config import Boltz2Config, MSAModuleConfig, TemplateV2ModuleConfig, TrunkConfig
from bionemo_ir.models.openfold3.config import AuxiliaryHeadsConfig
from bionemo_ir.models.protenix.config import ConfidenceHeadConfig
from bionemo_ir.pipeline.models.boltz2.const import num_tokens

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the layers build their weights on CUDA")

C_S = 64
C_Z = 32
C_S_INPUTS = 48
ATOMS_PER_TOKEN = 4
TOKENS = 13
PADDED_TOKENS = 16
SAMPLES = 3


def _pairformer_config(dtype: str) -> PairformerConfig:
    return PairformerConfig(
        token_s=C_S,
        token_z=C_Z,
        num_blocks=2,
        num_heads=4,
        pairwise_head_width=16,
        pairwise_num_heads=2,
        attention_initial_norm=True,
        version="v1",
        dtype=dtype,
    )


def _initialize(module: torch.nn.Module) -> torch.nn.Module:
    """Fill the checkpoint-backed parameters with finite, deterministic values."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("weight"):
                parameter.normal_(mean=0.0, std=0.1)
            else:
                parameter.zero_()
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.LayerNorm):
                submodule.reset_parameters()
    return module


def _record_tokens(pairformer: torch.nn.Module, tokens_seen: list[int]):
    return pairformer.register_forward_pre_hook(lambda _module, args: tokens_seen.append(args[1].shape[-2]))


def _boltz_msa_features(one_hot_msa: bool) -> dict[str, torch.Tensor]:
    """The MSA features the Boltz trunk and Boltz-1 confidence MSA module read, ``[B, S, N]``."""
    rows = 5
    msa = torch.zeros(1, rows, TOKENS, dtype=torch.long)
    return {
        # Boltz-1 featurizes the MSA one-hot; Boltz-2's MSA module one-hots it itself.
        "msa": torch.nn.functional.one_hot(msa, num_classes=num_tokens) if one_hot_msa else msa,
        "has_deletion": torch.zeros(1, rows, TOKENS),
        "deletion_value": torch.zeros(1, rows, TOKENS),
        "msa_paired": torch.zeros(1, rows, TOKENS),
        "msa_mask": torch.zeros(1, rows, TOKENS),
    }


def _assert_padded_on_token_axes(original: dict, padded: dict) -> None:
    assert original.keys() == padded.keys()
    for name, tensor in original.items():
        expected = tuple(PADDED_TOKENS if extent == TOKENS else extent for extent in tensor.shape)
        assert tuple(padded[name].shape) == expected, name


@pytest.mark.parametrize("config_cls", [Boltz1Config, Boltz2Config], ids=["boltz1", "boltz2"])
def test_boltz_trunk_spec_pads_every_tensor_on_its_token_axes(config_cls) -> None:
    tensors = {
        "s_init": torch.zeros(1, TOKENS, C_S),
        "s_inputs": torch.zeros(1, TOKENS, C_S_INPUTS),
        "z_init": torch.zeros(1, TOKENS, TOKENS, C_Z),
        "token_pad_mask": torch.zeros(1, TOKENS),
        **_boltz_msa_features(one_hot_msa=config_cls is Boltz1Config),
    }

    padded, _, n_true = pad_trunk_tokens(tensors, TOKENS, config_cls().trunk.token_pad_spec)

    assert n_true == TOKENS
    _assert_padded_on_token_axes(tensors, padded)


def test_boltz1_confidence_spec_pads_the_msa_features_on_their_token_axes() -> None:
    features = _boltz_msa_features(one_hot_msa=True)
    spec = Boltz1Config().confidence_module.token_pad_spec.feature_dict

    padded = pad_feature_dict({**features, "absent_optional": None}, PADDED_TOKENS - TOKENS, spec)

    assert padded.pop("absent_optional") is None
    _assert_padded_on_token_axes(features, padded)


def _boltz_template_features(templates: int = 2) -> dict[str, torch.Tensor]:
    """``TemplateV2Module`` inputs, ``[B, T, N, ...]``."""
    shape = (1, templates, TOKENS)
    return {
        "template_restype": torch.nn.functional.one_hot(torch.randint(0, num_tokens, shape), num_tokens).float(),
        "template_frame_rot": torch.eye(3).expand(*shape, 3, 3).contiguous(),
        "template_frame_t": torch.randn(*shape, 3),
        "template_mask_frame": torch.ones(shape),
        "template_cb": torch.randn(*shape, 3),
        "template_ca": torch.randn(*shape, 3),
        "template_mask_cb": torch.ones(shape),
        "visibility_ids": torch.zeros(shape, dtype=torch.long),
        "template_mask": torch.ones(shape),
    }


def test_boltz2_trunk_spec_pads_the_template_features_on_their_token_axes() -> None:
    features = _boltz_template_features()
    spec = Boltz2Config().trunk.token_pad_spec.feature_dict

    padded = pad_feature_dict(features, PADDED_TOKENS - TOKENS, spec)

    _assert_padded_on_token_axes(features, padded)


@requires_cuda
def test_boltz2_trunk_pads_template_features_and_matches_the_unpadded_trunk() -> None:
    pairformer = PairformerConfig(
        token_s=C_S,
        token_z=C_Z,
        num_blocks=1,
        num_heads=4,
        pairwise_head_width=16,
        pairwise_num_heads=2,
        attention_initial_norm=False,
        version="v2",
    )
    # The probes record every trunk call, which a capture's warmup and verification would add to.
    config = TrunkConfig(
        graph_optimization_config=None,
        use_templates_v2=True,
        msa_module=MSAModuleConfig(
            msa_s=16,
            token_z=C_Z,
            token_s=C_S,
            msa_blocks=1,
            pairwise_head_width=16,
            pairwise_num_heads=2,
            num_tokens=num_tokens,
            use_paired_feature=True,
            version="v2",
        ),
        pairformer=pairformer,
        template_module=TemplateV2ModuleConfig(
            token_z=C_Z,
            template_dim=16,
            template_blocks=1,
            pairwise_head_width=16,
            pairwise_num_heads=1,
            pairformer=pairformer.copy_and_validate(token_z=16, pairwise_num_heads=1, no_update_s=True),
        ),
    )
    trunk = _initialize(Trunk(config).cuda().eval())
    assert trunk.enable_token_pad
    rows = 5
    inputs = {
        "s_init": torch.randn(1, TOKENS, C_S),
        "z_init": torch.randn(1, TOKENS, TOKENS, C_Z),
        "s_inputs": torch.randn(1, TOKENS, C_S),
        "msa": torch.randint(0, num_tokens, (1, rows, TOKENS)),
        "has_deletion": torch.zeros(1, rows, TOKENS),
        "deletion_value": torch.zeros(1, rows, TOKENS),
        "msa_paired": torch.zeros(1, rows, TOKENS),
        "msa_mask": torch.ones(1, rows, TOKENS),
        "token_pad_mask": torch.ones(1, TOKENS),
    }
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    templates = {name: tensor.cuda() for name, tensor in _boltz_template_features().items()}
    tokens_seen: list[int] = []
    hook = _record_tokens(trunk.pairformer_module, tokens_seen)

    try:
        with torch.inference_mode():
            with_templates = trunk(**inputs, recycling_steps=0, template_feats=templates)
            trunk(**inputs, recycling_steps=0)
            trunk.enable_token_pad = False
            unpadded = trunk(**inputs, recycling_steps=0, template_feats=templates)
    finally:
        hook.remove()

    assert tokens_seen == [PADDED_TOKENS, PADDED_TOKENS, TOKENS]
    for actual, expected in zip(with_templates, unpadded, strict=True):
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        # fp32 GEMMs over the padded shapes round differently (TF32 in the dev container).
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("pairformer_dtype", ["float32", "bfloat16"])
def test_protenix_confidence_pairformer_runs_on_padded_tokens(pairformer_dtype: str) -> None:
    config = ConfidenceHeadConfig(
        c_s=C_S,
        c_z=C_Z,
        c_s_inputs=C_S_INPUTS,
        max_atoms_per_token=ATOMS_PER_TOKEN,
        pairformer_config=_pairformer_config(pairformer_dtype),
    )
    head = _initialize(ProtenixConfidenceHead(config).cuda().eval())
    assert head.enable_token_pad

    atoms = torch.arange(TOKENS * ATOMS_PER_TOKEN, device="cuda")
    feats = {
        "distogram_rep_atom_mask": atoms % ATOMS_PER_TOKEN == 0,
        "atom_to_token_idx": atoms // ATOMS_PER_TOKEN,
        "atom_to_tokatom_idx": atoms % ATOMS_PER_TOKEN,
    }
    inputs = (
        torch.randn(1, TOKENS, C_S_INPUTS, device="cuda"),
        torch.randn(1, TOKENS, C_S, device="cuda"),
        torch.randn(1, TOKENS, TOKENS, C_Z, device="cuda"),
        10 * torch.randn(1, SAMPLES, TOKENS * ATOMS_PER_TOKEN, 3, device="cuda"),
    )
    tokens_seen: list[int] = []
    hook = _record_tokens(head.pairformer_stack, tokens_seen)

    try:
        with torch.inference_mode():
            padded = head(feats, *inputs)
            head.enable_token_pad = False
            unpadded = head(feats, *inputs)
    finally:
        hook.remove()

    assert tokens_seen == [PADDED_TOKENS] * SAMPLES + [TOKENS] * SAMPLES
    for key, expected in unpadded.items():
        assert padded[key].shape == expected.shape, key
        # fp32 GEMMs over the padded shapes round differently (TF32 in the dev container).
        torch.testing.assert_close(padded[key], expected, rtol=1e-3, atol=2e-3, msg=key)


@requires_cuda
@pytest.mark.parametrize("apply_per_sample", [True, False], ids=["per-sample", "batched"])
def test_openfold3_confidence_pairformer_runs_on_padded_tokens(apply_per_sample: bool) -> None:
    embedding = PairformerEmbedding(
        pairformer=_pairformer_config("float32"),
        c_s_input=C_S_INPUTS,
        c_z=C_Z,
        min_bin=3.25,
        max_bin=50.75,
        no_bin=39,
        inf=1e9,
        token_pad_spec=AuxiliaryHeadsConfig().token_pad_spec,
    ).cuda()
    embedding = _initialize(embedding.eval())
    spec = embedding.token_pad_spec
    assert spec is not None

    inputs = {
        "si_input": torch.randn(1, 1, TOKENS, C_S_INPUTS, device="cuda"),
        "si": torch.randn(1, 1, TOKENS, C_S, device="cuda"),
        "zij": torch.randn(1, 1, TOKENS, TOKENS, C_Z, device="cuda"),
        "x_pred": 10 * torch.randn(1, SAMPLES, TOKENS, 3, device="cuda"),
        "single_mask": torch.ones(1, SAMPLES, TOKENS, device="cuda"),
        "pair_mask": torch.ones(1, 1, TOKENS, TOKENS, device="cuda"),
    }
    tokens_seen: list[int] = []
    hook = _record_tokens(embedding.pairformer_stack, tokens_seen)

    try:
        with torch.inference_mode():
            padded = embedding(**inputs, apply_per_sample=apply_per_sample)
            embedding.token_pad_spec = None
            unpadded = embedding(**inputs, apply_per_sample=apply_per_sample)
    finally:
        hook.remove()

    calls = SAMPLES if apply_per_sample else 1
    assert tokens_seen == [PADDED_TOKENS] * calls + [TOKENS] * calls
    for actual, expected in zip(padded, unpadded, strict=True):
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
