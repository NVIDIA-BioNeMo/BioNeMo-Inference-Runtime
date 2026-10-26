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

import collections
import datetime
import inspect
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend.utils import precompute_pair_masks, precompute_single_masks
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import SUPPORTED_SMS as ADALN_SUPPORTED_SMS
from bionemo_ir._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from bionemo_ir._torch.layers import transition as transition_layer
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.normalization import FusedLayerNorm, FusedRMSNorm
from bionemo_ir._torch.layers.transformers import (
    blocks as transformer_blocks,
)
from bionemo_ir._torch.layers.transformers import (
    local_latents_decoder,
    local_latents_encoder,
    local_latents_transformer,
)
from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock
from bionemo_ir._torch.layers.transition import PairTransition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
    precompute_trimul_metadata,
)
from bionemo_ir._torch.modules.rfproteina import (
    LocalLatentsTransformer,
    RFProteinaLatentDecoder,
    RFProteinaLatentEncoder,
    RFProteinaPairUpdate,
    RFProteinaRepresentationBuilder,
    chirality,
    representation_builder,
)
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint
from bionemo_ir.hubs.local import LOCAL_CHECKPOINTS
from bionemo_ir.hubs.support_matrix import FoldingSupportMatrix
from bionemo_ir.hubs.support_matrix import GenerativeSupportMatrix as SupMat
from bionemo_ir.models.rfproteina import (
    RFProteina,
    RFProteinaConfig,
    RFProteinaConversionReport,
    RFProteinaDecoderConfig,
    RFProteinaEncoderConfig,
    RFProteinaRepresentationConfig,
    RFProteinaTrunkConfig,
    load_rfproteina_decoder_checkpoint,
    load_rfproteina_encoder_checkpoint,
    load_rfproteina_representation_builder_checkpoint,
    load_rfproteina_trunk_checkpoint,
    merge_rfproteina_lora,
    rfproteina_decoder_config_from_checkpoint,
    rfproteina_encoder_config_from_checkpoint,
    rfproteina_encoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_weights,
)
from bionemo_ir.models.rfproteina import modeling as rfproteina_modeling
from bionemo_ir.models.rfproteina.convert import (
    RELEASED_LORA_ALPHA,
    _check_architecture,
    _convert_attention,
    _convert_encoder_inputs,
    convert_rfproteina_representation_builder_torch,
    fold_atom_chemistry,
    load_rfproteina_checkpoint_state,
)
from tests._torch import SM_VERSION, init_module_weights, skip_if_cutedsl, skip_if_no_cutedsl


def test_rfproteina_runtime_contracts_have_one_canonical_definition() -> None:
    from bionemo_ir._torch.modules.rfproteina import (
        atomized,
        contracts,
        flow_matcher,
        insertion,
        nucleic_acid,
        partial_redesign,
        representation_builder,
        rollout,
        sampler,
        search,
    )
    from bionemo_ir.models.rfproteina import convert, modeling

    compatibility_exports = (
        (atomized, "RFProteinaAtomizedOutput"),
        (flow_matcher, "ModalityReference"),
        (flow_matcher, "RFProteinaCorruptionOutput"),
        (insertion, "RFProteinaInsertionOutput"),
        (nucleic_acid, "RFProteinaNucleicAcidOutput"),
        (partial_redesign, "RFProteinaCleanStateEncoding"),
        (partial_redesign, "RFProteinaPartialRedesignOutput"),
        (representation_builder, "RFProteinaRepresentations"),
        (sampler, "RFProteinaSamplingOutput"),
        (sampler, "RFProteinaPartialSamplingOutput"),
        (search, "RFProteinaSearchCandidate"),
        (search, "RFProteinaSearchOutput"),
        (modeling, "RFProteinaPredictionOutput"),
        (modeling, "RFProteinaSearchOutput"),
        (convert, "RFProteinaConversionReport"),
    )
    for module, name in compatibility_exports:
        canonical = getattr(contracts, name)
        assert getattr(module, name) is canonical
        assert canonical.__module__ == contracts.__name__
    for module in (flow_matcher, rollout, sampler):
        assert module.RFProteinaState is contracts.RFProteinaState
    assert rollout.RFProteinaPrediction is contracts.RFProteinaPrediction
    assert sampler.RFProteinaPredictFn is contracts.RFProteinaPredictFn


def _rfproteina_attention(
    token_dim: int, pair_dim: int, num_heads: int, backend: str, dtype: torch.dtype, use_qk_norm: bool = True
) -> AttentionPairBias:
    """Build the shared AttentionPairBias with RFProteína's variant flags."""
    return AttentionPairBias(
        layer_idx=0,
        c_s=token_dim,
        c_z=pair_dim,
        num_heads=num_heads,
        initial_norm=True,
        norm_type="rms_norm",
        bias_proj=True,
        pair_norm=True,
        pair_norm_type="rms_norm",
        use_qk_norm=use_qk_norm,
        kv_bias=True,
        out_bias=True,
        use_ada_layer_norm=False,
        inf=1e4,
        eps=1e-5,
        dtype=dtype,
        attn_backend=backend,
    )


def _small_config(**updates) -> RFProteinaTrunkConfig:
    values = {
        "token_dim": 32,
        "pair_dim": 16,
        "condition_dim": 8,
        "latent_dim": 4,
        "num_layers": 2,
        "num_heads": 4,
        "trimul_hidden_dim": 8,
        "pair_write": "sum",
        "recycle_enabled": False,
        "dtype": "float32",
        "pairwise_attention_backend": "SDPA",
    }
    values.update(updates)
    return RFProteinaTrunkConfig(**values)


def test_rfproteina_generative_support_and_checkpoint_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SupMat.is_supported(SupMat.RFProteina)
    assert not FoldingSupportMatrix.is_supported(SupMat.RFProteina)

    checkpoint = LOCAL_CHECKPOINTS[SupMat.RFProteina]
    assert checkpoint.env == [
        "RFPROTEINA_LDM_CKPT",
        "RFPROTEINA_AE_CKPT",
    ]

    monkeypatch.setenv(checkpoint.env[0], "/checkpoints/ldm.ckpt")
    monkeypatch.setenv(checkpoint.env[1], "/checkpoints/ae.ckpt")
    assert resolve_rfproteina_checkpoint() == Path("/checkpoints/ldm.ckpt")
    assert resolve_rfproteina_autoencoder_checkpoint() == Path("/checkpoints/ae.ckpt")


def test_rfproteina_pretrained_config_selects_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_dtypes = []

    def select_backend(dtype: torch.dtype) -> str:
        selected_dtypes.append(dtype)
        return "SDPA"

    monkeypatch.setattr(
        rfproteina_modeling,
        "auto_select_pairwise_attention_backend",
        select_backend,
    )
    config = RFProteina.get_pretrained_config(SupMat.RFProteina)

    assert config.dtype == "bfloat16"
    assert config.trunk.dtype == "bfloat16"
    assert config.decoder.dtype == "float32"
    assert config.decoder.transformer_dtype == "bfloat16"
    assert config.representation.dtype == "float32"
    assert config.representation.pair_adaln_dtype == "bfloat16"
    assert config.trunk.pairwise_attention_backend == "SDPA"
    assert config.decoder.pairwise_attention_backend == "SDPA"
    assert selected_dtypes == [torch.bfloat16]
    with pytest.raises(ValueError, match="pretrained config not found"):
        RFProteina.get_pretrained_config("unknown")


def test_rfproteina_lazy_encoder_inherits_autoencoder_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    encoder_report = RFProteinaConversionReport(("encoder",), (), ())

    def encoder_config_from_checkpoint(checkpoint_path, **overrides):
        captured["config"] = (checkpoint_path, overrides)
        return _small_encoder_config(**overrides)

    def load_encoder(module, checkpoint_path, strict):
        captured["load"] = (module, checkpoint_path, strict)
        return encoder_report

    monkeypatch.setattr(
        rfproteina_modeling, "rfproteina_encoder_config_from_checkpoint", encoder_config_from_checkpoint
    )
    monkeypatch.setattr(rfproteina_modeling, "load_rfproteina_encoder_checkpoint", load_encoder)
    decoder = RFProteinaDecoderConfig(
        token_dim=32,
        pair_dim=16,
        condition_dim=8,
        latent_dim=4,
        num_layers=2,
        num_heads=4,
        dtype="float32",
        transformer_dtype="bfloat16",
        pairwise_attention_backend="SDPA",
    )
    model = RFProteina(RFProteinaConfig(dtype="float32", trunk=_small_config(), decoder=decoder))

    report = model.load_encoder_weights("ae.ckpt")

    assert report is encoder_report
    assert captured["config"] == (
        "ae.ckpt",
        {
            "dtype": "float32",
            "transformer_dtype": "bfloat16",
            "pairwise_attention_backend": "SDPA",
        },
    )
    assert model.encoder is not None
    assert model.encoder.transformer_dtype == torch.bfloat16
    assert model.encoder.transformer_layers[0].mha.in_proj.weight.dtype == torch.bfloat16
    assert model.encoder.init_repr_factory.linear_out.weight.dtype == torch.float32
    assert model.encoder.latent_decoder_mean_n_log_scale[1].weight.dtype == torch.float32
    assert captured["load"] == (model.encoder, "ae.ckpt", True)
    assert any(isinstance(module, FusedLayerNorm) for module in model.encoder.modules())
    assert not any(type(module) is nn.LayerNorm for module in model.encoder.modules())
    assert not any(type(module) is nn.RMSNorm for module in model.encoder.modules())
    assert not model._ensure_encoder(torch.device("cpu")).training


def test_rfproteina_constructs_and_loads_both_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    trunk_report = RFProteinaConversionReport(("trunk",), (), ())
    representation_report = RFProteinaConversionReport(("representation",), (), ())
    decoder_report = RFProteinaConversionReport(("decoder",), (), ())
    loaded = {}

    def load_trunk(module, checkpoint_path, strict, lora_alpha):
        loaded["trunk"] = (module, checkpoint_path, strict)
        return trunk_report

    def load_decoder(module, checkpoint_path, strict):
        loaded["decoder"] = (module, checkpoint_path, strict)
        return decoder_report

    def load_representation(module, checkpoint_path, strict, lora_alpha):
        loaded["representation"] = (module, checkpoint_path, strict)
        return representation_report

    monkeypatch.setattr(rfproteina_modeling, "load_rfproteina_trunk_checkpoint", load_trunk)
    monkeypatch.setattr(rfproteina_modeling, "load_rfproteina_representation_builder_checkpoint", load_representation)
    monkeypatch.setattr(rfproteina_modeling, "load_rfproteina_decoder_checkpoint", load_decoder)
    config = RFProteinaConfig(
        dtype="float32",
        trunk=_small_config(),
        decoder=RFProteinaDecoderConfig(
            token_dim=32,
            pair_dim=16,
            condition_dim=8,
            latent_dim=4,
            num_layers=2,
            num_heads=4,
            dtype="float32",
            pairwise_attention_backend="SDPA",
        ),
    )
    model = RFProteina(
        config,
        checkpoint_path="ldm.ckpt",
        autoencoder_checkpoint_path="ae.ckpt",
        include_load_weights=True,
    )

    assert isinstance(model.decoder, RFProteinaLatentDecoder)
    assert model.conversion_report is trunk_report
    assert model.representation_conversion_report is representation_report
    assert model.decoder_conversion_report is decoder_report
    assert loaded["trunk"] == (model.velocity_model, resolve_rfproteina_checkpoint("ldm.ckpt"), True)
    assert loaded["representation"] == (
        model.representation_builder,
        resolve_rfproteina_checkpoint("ldm.ckpt"),
        True,
    )
    assert loaded["decoder"] == (
        model.decoder,
        resolve_rfproteina_autoencoder_checkpoint("ae.ckpt"),
        True,
    )


def _oss_trunk_shapes(
    token_dim: int,
    pair_dim: int,
    condition_dim: int,
    latent_dim: int,
    num_layers: int,
    num_heads: int,
    pair_update_every: int,
    trimul_hidden_dim: int,
    transition_expansion: int,
    condition_transition_expansion: int,
    pair_transition_expansion: int,
) -> dict[str, torch.Tensor]:
    """Minimal OSS-named weight mapping carrying the architecture's shapes."""
    weights = {
        "nn.transformer_layers.0.mhba.mha.to_bias.weight": torch.empty(num_heads, pair_dim),
        "nn.transformer_layers.0.mhba.mha.to_qkv.weight": torch.empty(3 * token_dim, token_dim),
        "nn.transformer_layers.0.mhba.mha.q_layer_norm.weight": torch.empty(token_dim),
        "nn.transformer_layers.0.mhba.adaln.to_gamma.0.weight": torch.empty(token_dim, condition_dim),
        "nn.transformer_layers.0.transition.transition.linear_out.weight": torch.empty(
            token_dim, transition_expansion * token_dim
        ),
        "nn.transition_c_1.linear_out.weight": torch.empty(
            condition_dim, condition_transition_expansion * condition_dim
        ),
        "nn.local_latents_linear.1.weight": torch.empty(latent_dim, token_dim),
        "nn.insertion_q_proj.weight": torch.empty(token_dim, token_dim),
    }
    for layer in range(num_layers):
        weights[f"nn.transformer_layers.{layer}.mhba.mha.to_g.weight"] = torch.empty(token_dim, token_dim)
    for layer in range(0, num_layers - 1, pair_update_every):
        weights[f"nn.pair_update_layers.{layer}.tri_mult_out.linear_a_p.weight"] = torch.empty(
            trimul_hidden_dim, pair_dim
        )
        weights[f"nn.pair_update_layers.{layer}.tri_mult_out.linear_a_p.bias"] = torch.empty(trimul_hidden_dim)
        weights[f"nn.pair_update_layers.{layer}.transition_out.linear_1.weight"] = torch.empty(
            pair_transition_expansion * pair_dim, pair_dim
        )
    return weights


def test_rfproteina_released_config() -> None:
    config = RFProteinaConfig()
    assert config.trunk.enable_cudnn_graph
    assert config.trunk.enable_token_pad
    assert config.trunk.token_dim == 768
    assert config.trunk.pair_dim == 384
    assert config.trunk.condition_dim == 256
    assert config.trunk.latent_dim == 12
    assert config.trunk.num_heads == 12
    assert config.trunk.trimul_hidden_dim == 256
    assert config.trunk.pair_update_every == 2
    assert config.trunk.pair_write == "hadamard"
    assert config.trunk.recycle_enabled
    assert config.representation.input_schema == "latest"
    assert (config.representation.single_input_dim, config.representation.pair_input_dim) == (744, 107)
    assert config.decoder.input_schema == "unified"
    assert (config.decoder.single_input_dim, config.decoder.pair_input_dim) == (80, 0)
    # The released policy: thinking_medium over the densified tail, as rfd4-generate runs it.
    assert config.sampling.num_steps == 60
    assert not config.sampling.self_condition
    assert len(config.sampling.recycle_depths()) == 60
    assert sum(config.sampling.recycle_depths()) == 185
    assert config.sampling.condition_decoder_on_sequence_prediction
    for modality in (config.sampling.backbone, config.sampling.local_latents):
        assert (modality.schedule.tail_steps, modality.schedule.tail_step_multiplier) == (10, 2)
    assert config.sampling.backbone.step.sampling_mode == "vf_tsr"


def test_sequence_probability_representation_matches_340k_widths() -> None:
    config = RFProteinaRepresentationConfig(input_schema="sequence_probability", latent_dim=16)
    assert config.single_input_dim == 881
    assert config.condition_input_dim == 864
    assert (config.pair_input_dim, config.pair_condition_input_dim) == (107, 512)
    assert len(representation_builder.SEQUENCE_PROBABILITY_TOKENS) == config.num_sequence_probability_channels
    assert config.supports_nucleic_acid_secondary_structure


def _sequence_probability_batch(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    tokens = 3
    probabilities = {
        token: torch.full((1, tokens), torch.nan, device=device)
        for token in representation_builder.SEQUENCE_PROBABILITY_TOKENS
    }
    probabilities["ALA"] = torch.zeros(1, tokens, device=device)
    probabilities["CYS"] = torch.full((1, tokens), 0.25, device=device)
    static = {
        "padding_mask": torch.ones(1, tokens, device=device, dtype=torch.bool),
        "token_type": torch.zeros(1, tokens, device=device, dtype=torch.long),
        "atom37_mask": torch.ones(1, tokens, 37, device=device, dtype=torch.bool),
        "condition_cath-label_1_residue": torch.zeros(1, tokens, 3, device=device, dtype=torch.long),
        "condition_CoordinateGroup_1_atom": torch.full((1, tokens, 37), -1, device=device, dtype=torch.long),
        "condition_rasa-binned_1_atom": torch.zeros(1, tokens, 37, device=device),
        **{f"condition_sequence-probability-{token}_1_residue": value for token, value in probabilities.items()},
    }
    static["atom37_mask"][0, 0, 0] = False
    static["condition_rasa-binned_1_atom"][0, 0, 0] = 3
    static["condition_rasa-binned_1_atom"][0, 0, 1] = 2
    state = {
        "bb_ca": torch.zeros(1, tokens, 3, device=device),
        "local_latents": torch.zeros(1, tokens, 4, device=device),
    }
    times = {"bb_ca": torch.tensor([0.4], device=device), "local_latents": torch.tensor([0.2], device=device)}
    return static, state, times


def test_sequence_probability_features_and_recycle_condition() -> None:
    device = torch.device("cuda")
    trunk = _small_config(latent_dim=4)
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability",
        latent_dim=4,
        dtype="float32",
        pair_adaln_dtype="float32",
    )
    builder = RFProteinaRepresentationBuilder(config, trunk).to(device).eval()
    assert "conditional_residue_embedding.weight" not in builder.state_dict()
    static, state, times = _sequence_probability_batch(device)
    packed = builder._sequence_probabilities(static, state)
    channels = config.num_sequence_probability_channels
    ala = representation_builder.SEQUENCE_PROBABILITY_TOKENS.index("ALA")
    cys = representation_builder.SEQUENCE_PROBABILITY_TOKENS.index("CYS")
    arg = representation_builder.SEQUENCE_PROBABILITY_TOKENS.index("ARG")
    torch.testing.assert_close(packed[..., ala], torch.zeros(1, 3, device=device))
    torch.testing.assert_close(packed[..., channels + ala], torch.ones(1, 3, device=device))
    torch.testing.assert_close(packed[..., cys], torch.full((1, 3), 0.25, device=device))
    torch.testing.assert_close(packed[..., arg], torch.zeros(1, 3, device=device))
    torch.testing.assert_close(packed[..., channels + arg], torch.zeros(1, 3, device=device))
    rasa = builder._rasa_binned(static, state)
    assert rasa[0, 0, 0].item() == 0
    assert rasa[0, 0, 1].item() == 2

    missing = dict(static)
    del missing["condition_sequence-probability-ALA_1_residue"]
    with pytest.raises(KeyError, match="sequence_probabilities requires"):
        builder._sequence_probabilities(missing, state)

    with torch.inference_mode():
        first = builder._condition(static, state, times, static["padding_mask"], recycle_index=1)
        stacked = builder.condition_stack(static, state, times, static["padding_mask"], recycle_depth=3)
    torch.testing.assert_close(stacked[0], first)
    assert not torch.allclose(stacked[0], stacked[1])
    assert not torch.allclose(stacked[1], stacked[2])
    with pytest.raises(ValueError, match="carries a recycle feature"):
        RFProteinaRepresentationBuilder(
            RFProteinaRepresentationConfig(latent_dim=4, pair_adaln_dtype="float32"), trunk
        ).condition_stack(static, state, times, static["padding_mask"], 2)


def test_coordinate_group_permutation_draws_from_the_sampling_stream() -> None:
    """Upstream takes the permutation and the noise from one generator, so this must too.

    A stream that carries only the noise draws stays aligned until something else
    consumes from the reference's, and the group permutation does exactly that
    whenever an input has a group to permute. Every noise draw after the first
    such permutation then differs, which is invisible in the shape and order of
    the draws and shows up only in a design.
    """
    from bionemo_ir._torch.modules.rfproteina.groups import CONDITION_COORDINATE_GROUP
    from bionemo_ir._torch.sampling.random import normal_draws, permutations, randn

    device = torch.device("cuda")
    builder = RFProteinaRepresentationBuilder(
        RFProteinaRepresentationConfig(input_schema="sequence_probability", latent_dim=4, pair_adaln_dtype="float32"),
        _small_config(latent_dim=4),
    ).to(device)
    static = {CONDITION_COORDINATE_GROUP: torch.tensor([[0, 0, 1, 1, 2]], device=device)}

    def consume(stream: torch.Generator) -> torch.Tensor:
        """The permutation, then a noise draw, from one generator."""
        with (
            normal_draws(
                lambda shape, *, device, dtype: torch.randn(shape, device=device, dtype=dtype, generator=stream)
            ),
            permutations(lambda n, *, device: torch.randperm(n, device=device, generator=stream)),
        ):
            builder._coordinate_group(static)
            return randn((4,), device=device, dtype=torch.float32)

    stream = torch.Generator(device=device)
    stream.manual_seed(7)
    through_seam = consume(stream)

    reference = torch.Generator(device=device)
    reference.manual_seed(7)
    torch.randperm(3, device=device, generator=reference)
    expected = torch.randn(4, device=device, generator=reference)
    torch.testing.assert_close(through_seam, expected)


@pytest.mark.parametrize("width", [2, 3])
def test_remaining_pass_conditioning_counts_down_and_saturates(width: int) -> None:
    """The unified checkpoints condition on passes still to come, not on the pass itself."""
    device = torch.device("cuda")
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability",
        latent_dim=4,
        dtype="float32",
        pair_adaln_dtype="float32",
        recycle_conditioning="recycle_remaining",
        recycle_remaining_dim=width,
    )
    assert config.condition_input_dim == config.condition_base_dim + width
    builder = RFProteinaRepresentationBuilder(config, _small_config(latent_dim=4)).to(device).eval()

    def track(index: int, depth: int) -> list[float]:
        return builder._recycle_feature("recycle_remaining", index, depth, 1, device)[0].tolist()

    # A single pass, and the first pass of any run, read the all-zero slot.
    assert track(1, 1) == [0.0] * width
    assert track(1, 4) == [0.0] * width
    # Later passes one-hot the remaining count, which saturates at the last slot.
    for index in range(2, 5):
        expected = [0.0] * width
        expected[min(4 - index, width - 1)] = 1.0
        assert track(index, 4) == expected
    with pytest.raises(ValueError, match="no remaining count"):
        track(3, 2)

    static, state, times = _sequence_probability_batch(device)
    with torch.inference_mode():
        first = builder._condition(static, state, times, static["padding_mask"], recycle_index=1)
        stacked = builder.condition_stack(static, state, times, static["padding_mask"], recycle_depth=3)
    torch.testing.assert_close(stacked[0], first)
    assert not torch.allclose(stacked[1], stacked[2])


def test_absolute_coordinates_use_finite_values_when_the_mask_is_absent() -> None:
    device = torch.device("cuda")
    trunk = _small_config(latent_dim=4)
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability",
        latent_dim=4,
        dtype="float32",
        pair_adaln_dtype="float32",
    )
    builder = RFProteinaRepresentationBuilder(config, trunk).to(device).eval()
    static, state, _times = _sequence_probability_batch(device)
    coordinates = torch.full((1, 3, 37, 3), torch.nan, device=device)
    coordinates[0, 1, 1] = torch.tensor([10.0, 20.0, 30.0], device=device)
    coordinates[0, 1, 2, 0] = 5.0
    static["condition_coordinate_1_atom"] = coordinates

    feature = builder._conditional_coordinates(
        static,
        state,
        representation_builder.CONDITION_COORDINATE,
        representation_builder.CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )

    mask = feature[0, :, 37 * 3 :]
    assert mask[0].sum().item() == 0
    assert mask[1, 1].item() == 1
    assert mask[1, 2].item() == 1
    assert mask[2].sum().item() == 0
    torch.testing.assert_close(feature[0, 1, 3:6], torch.tensor([1.0, 2.0, 3.0], device=device))
    torch.testing.assert_close(feature[0, 1, 6:9], torch.tensor([0.5, 0.0, 0.0], device=device))

    explicit = torch.zeros(1, 3, 37, dtype=torch.bool, device=device)
    explicit[0, 0, 0] = True
    static["mask_coordinate_1_atom"] = explicit
    static["condition_coordinate_1_atom"] = torch.zeros(1, 3, 37, 3, device=device)
    static["condition_coordinate_1_atom"][0, 0, 0] = torch.tensor([10.0, 0.0, 0.0], device=device)
    masked = builder._conditional_coordinates(
        static,
        state,
        representation_builder.CONDITION_COORDINATE,
        representation_builder.CONDITION_COORDINATE_MASK,
        finite_marks_conditioned=True,
    )
    assert masked[0, :, 37 * 3 :].sum().item() == 1
    torch.testing.assert_close(masked[0, 0, :3], torch.tensor([1.0, 0.0, 0.0], device=device))

    static["condition_InternalCoordinate_1_atom"] = torch.ones(1, 3, 37, 3, device=device)
    internal = builder._conditional_coordinates(
        static,
        state,
        representation_builder.CONDITION_INTERNAL_COORDINATE,
        representation_builder.CONDITION_INTERNAL_COORDINATE_MASK,
    )
    assert internal.abs().sum().item() == 0


def test_internal_coordinates_read_their_selection_from_the_coordinate_group() -> None:
    """Upstream never emits ``mask_InternalCoordinate_1_atom``.

    It carries the selection in ``condition_CoordinateGroup_1_atom`` instead: -1
    is unconditioned, 0 is absolute conditioning, and a positive group is
    internal-coordinate conditioned. Reading finiteness cannot substitute --
    the payload below is finite everywhere -- so without the group the whole
    condition silently becomes 148 zero channels.
    """
    device = torch.device("cuda")
    trunk = _small_config(latent_dim=4)
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability",
        latent_dim=4,
        dtype="float32",
        pair_adaln_dtype="float32",
    )
    builder = RFProteinaRepresentationBuilder(config, trunk).to(device).eval()
    static, state, _times = _sequence_probability_batch(device)
    static["condition_InternalCoordinate_1_atom"] = torch.ones(1, 3, 37, 3, device=device)

    # Token 1 is internal-coordinate conditioned, token 2 is absolutely
    # conditioned, token 0 keeps the fixture's unconditioned -1.
    groups = static["condition_CoordinateGroup_1_atom"]
    groups[0, 1, :] = 1
    groups[0, 2, :] = representation_builder.COORDINATE_GROUP_ABSOLUTE

    feature = builder._conditional_coordinates(
        static,
        state,
        representation_builder.CONDITION_INTERNAL_COORDINATE,
        representation_builder.CONDITION_INTERNAL_COORDINATE_MASK,
        group_marks_conditioned=True,
    )

    mask = feature[0, :, 37 * 3 :]
    assert mask[0].sum().item() == 0, "unconditioned tokens must stay unselected"
    assert mask[1].sum().item() == 37, "a positive coordinate group is internal conditioning"
    assert mask[2].sum().item() == 0, "absolute conditioning is not internal conditioning"
    # Angstroms become nanometres, and only the selected token carries values.
    torch.testing.assert_close(feature[0, 1, :3], torch.tensor([0.1, 0.1, 0.1], device=device))
    assert feature[0, 0, : 37 * 3].abs().sum().item() == 0
    assert feature[0, 2, : 37 * 3].abs().sum().item() == 0

    # An explicit mask still wins when a batch does carry one.
    explicit = torch.zeros(1, 3, 37, dtype=torch.bool, device=device)
    explicit[0, 0, 5] = True
    static["mask_InternalCoordinate_1_atom"] = explicit
    overridden = builder._conditional_coordinates(
        static,
        state,
        representation_builder.CONDITION_INTERNAL_COORDINATE,
        representation_builder.CONDITION_INTERNAL_COORDINATE_MASK,
        group_marks_conditioned=True,
    )
    assert overridden[0, :, 37 * 3 :].sum().item() == 1


def _chirality_scene(rotation: torch.Tensor, neighbours: int = 3) -> dict[str, torch.Tensor]:
    """One stereocentre at token 0 whose neighbours are tokens 1.., each carrying a vector in slot 0.

    The supplied vectors point along the axes. The noisy anchors put each neighbour where ``rotation``
    turns its axis, at a different length, so a correct fit has to recover ``rotation`` from directions.
    Slot 1 belongs to no frame and token 5 is padding.
    """
    tokens, slots = 6, 2
    axes = torch.eye(3)
    vectors = torch.zeros(1, tokens, slots * 3)
    vectors[0, 1 : 1 + neighbours, :3] = axes[:neighbours]
    vectors[0, :, 3:] = torch.tensor([0.3, -0.5, 0.8])
    frames = torch.full((1, tokens, slots, 5), -1, dtype=torch.long)
    frames[0, 0, 0, :2] = 0
    frames[0, 0, 0, 2 : 2 + neighbours] = torch.arange(1, 1 + neighbours)
    owners = torch.full((1, tokens, slots), -1, dtype=torch.long)
    owners[0, :4, 0] = 0  # the frame is anchor token * slots + slot = 0
    centre = torch.tensor([1.0, 2.0, 3.0])
    anchors = torch.zeros(1, tokens, 3)
    anchors[0, 0] = centre
    anchors[0, 1 : 1 + neighbours] = centre + 0.7 * (axes[:neighbours] @ rotation)
    return {
        "vectors": vectors,
        "anchors": anchors,
        "padding_mask": torch.tensor([[True] * 5 + [False]]),
        "frame_atoms": frames,
        "slot_owner": owners,
    }


def _about_z(angle: float) -> torch.Tensor:
    """A proper rotation about z, for row vectors."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def test_chirality_alignment_recovers_the_rotation_that_turns_the_neighbours() -> None:
    rotation = _about_z(0.9)
    scene = _chirality_scene(rotation)

    aligned = chirality.align_chirality(**scene)

    expected = scene["vectors"].clone()
    expected[0, 1:4, :3] = rotation  # the axes, turned
    expected[0, 5] = 0  # padding is cleared; slot 1 and the unowned tokens are left alone
    torch.testing.assert_close(aligned, expected, atol=1e-5, rtol=0)


def test_chirality_alignment_completes_a_two_neighbour_fit_to_a_proper_rotation() -> None:
    """Two directions leave the third axis open; the fit must still return a rotation, not a reflection."""
    rotation = _about_z(-1.3)
    scene = _chirality_scene(rotation, neighbours=2)
    scene["vectors"][0, 3, 0:3] = torch.tensor([0.0, 0.0, 1.0])  # token 3 is owned too, carrying the third axis

    aligned = chirality.align_chirality(**scene)

    torch.testing.assert_close(aligned[0, 1:4, :3], rotation, atol=1e-5, rtol=0)
    fitted = chirality.proper_rotation(torch.eye(3) @ rotation)
    assert torch.linalg.det(fitted).item() == pytest.approx(1.0)


def test_chirality_alignment_can_move_the_vectors_to_the_noisy_origin() -> None:
    scene = _chirality_scene(_about_z(0.4))

    moved = chirality.align_chirality(**scene, rotate=False, translate=True)

    expected = scene["vectors"].clone()
    expected[0, :4, :3] += scene["anchors"][0, 0]  # both origin atoms are token 0, so the midpoint is its anchor
    expected[0, 5] = 0
    torch.testing.assert_close(moved, expected)


def test_chirality_alignment_returns_the_vectors_untouched_when_both_options_are_off() -> None:
    scene = _chirality_scene(_about_z(0.4))
    assert chirality.align_chirality(**scene, rotate=False, translate=False) is scene["vectors"]


def test_chirality_alignment_ignores_a_hosts_autocast() -> None:
    """A rotation fitted from bf16 products is not a rotation, so the fit must stay in float32."""
    scene = _chirality_scene(_about_z(0.9))
    plain = chirality.align_chirality(**scene)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast = chirality.align_chirality(**scene)

    torch.testing.assert_close(autocast, plain, atol=0, rtol=0)


def test_chirality_alignment_clears_padding_when_nothing_has_a_frame() -> None:
    scene = _chirality_scene(_about_z(0.4))
    scene["frame_atoms"][...] = -1

    aligned = chirality.align_chirality(**scene)

    expected = scene["vectors"].clone()
    expected[0, 5] = 0
    torch.testing.assert_close(aligned, expected)


def test_chirality_alignment_rejects_inputs_it_cannot_fit() -> None:
    padded = _chirality_scene(_about_z(0.4))
    padded["padding_mask"][0, 2] = False  # a neighbour that is not a real token
    with pytest.raises(ValueError, match="padding tokens"):
        chirality.align_chirality(**padded)

    broken = _chirality_scene(_about_z(0.4))
    broken["anchors"][0, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        chirality.align_chirality(**broken)


def _numpy_chirality_reference(scene: dict[str, torch.Tensor], rotate: bool, translate: bool) -> torch.Tensor:
    """The same alignment one frame at a time, with numpy's SVD and no batching."""
    vectors, anchors = scene["vectors"].double().numpy(), scene["anchors"].double().numpy()
    valid, frames, owners = scene["padding_mask"].numpy(), scene["frame_atoms"].numpy(), scene["slot_owner"].numpy()
    batch, tokens, width = vectors.shape
    slots = width // 3
    out = vectors.copy().reshape(batch, tokens, slots, 3)
    for b in range(batch):
        for n in range(tokens):
            for s in range(slots):
                if frames[b, n, s, 0] < 0 or not valid[b, n]:
                    continue
                origin = anchors[b, frames[b, n, s, :2]].mean(0)
                members = frames[b, n, s, 2:][frames[b, n, s, 2:] >= 0]
                supplied = vectors[b, members, s * 3 : s * 3 + 3]
                noisy = anchors[b, members] - origin
                noisy = noisy / np.maximum(np.linalg.norm(noisy, axis=-1, keepdims=True), 1e-8)
                left, _, right_t = np.linalg.svd(supplied.T @ noisy)
                signs = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right_t))])
                rotation = left @ signs @ right_t
                owned = owners[b] == n * slots + s
                shifted = out[b][owned] @ rotation if rotate else out[b][owned]
                out[b][owned] = shifted + origin if translate else shifted
    out = out.reshape(batch, tokens, width) * valid[..., None]
    return torch.from_numpy(out).float()


def _random_chirality_scene(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Two requests, three frames each on different slots, with unit vectors and anchors drawn at random."""
    batch, tokens, slots = 2, 13, 3
    vectors = torch.nn.functional.normalize(torch.randn(batch, tokens, slots, 3, generator=generator), dim=-1)
    frames = torch.full((batch, tokens, slots, 6), -1, dtype=torch.long)
    owners = torch.full((batch, tokens, slots), -1, dtype=torch.long)
    for b in range(batch):
        for token, slot, members in ((0, 0, (3, 4, 5, 6)), (1, 1, (7, 8, 9)), (2, 2, (10, 11, 4))):
            frames[b, token, slot, :2] = torch.tensor([token, token + 1])
            frames[b, token, slot, 2 : 2 + len(members)] = torch.tensor(members)
            owners[b, list(members), slot] = token * slots + slot
    padding = torch.ones(batch, tokens, dtype=torch.bool)
    padding[:, -1] = False
    return {
        "vectors": vectors.reshape(batch, tokens, slots * 3),
        "anchors": torch.randn(batch, tokens, 3, generator=generator),
        "padding_mask": padding,
        "frame_atoms": frames,
        "slot_owner": owners,
    }


@pytest.mark.parametrize("rotate, translate", [(True, False), (True, True), (False, True)])
def test_chirality_alignment_matches_a_frame_by_frame_reference(rotate: bool, translate: bool) -> None:
    scene = _random_chirality_scene(torch.Generator().manual_seed(3))

    got = chirality.align_chirality(**scene, rotate=rotate, translate=translate)

    torch.testing.assert_close(got, _numpy_chirality_reference(scene, rotate, translate), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rotate, translate", [(True, False), (True, True), (False, True)])
def test_chirality_alignment_matches_the_upstream_function(rotate: bool, translate: bool) -> None:
    upstream = pytest.importorskip("rfproteina.nn.noise_alignment")
    scene = _random_chirality_scene(torch.Generator().manual_seed(4))
    batch = {
        "x_t": {"bb_ca": scene["anchors"]},
        "padding_mask": scene["padding_mask"],
        "chiral_frame_atoms": scene["frame_atoms"],
        "chiral_slot_owner": scene["slot_owner"],
    }

    want = upstream.align_chirality(scene["vectors"], batch, translate=translate, rotate=rotate)

    torch.testing.assert_close(chirality.align_chirality(**scene, rotate=rotate, translate=translate), want)


def _chirality_builder(**flags) -> RFProteinaRepresentationBuilder:
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability", latent_dim=4, dtype="float32", pair_adaln_dtype="float32", **flags
    )
    return RFProteinaRepresentationBuilder(config, _small_config(latent_dim=4)).to("cuda").eval()


def _chirality_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    static, state, _times = _sequence_probability_batch(torch.device("cuda"))
    slots = 6  # the width the checkpoint's chirality feature has: 18 = 3 * 6
    vectors = torch.zeros(1, 3, slots * 3, device="cuda")
    vectors[0, 1, :3] = torch.tensor([1.0, 0.0, 0.0])
    vectors[0, 2, :3] = torch.tensor([0.0, 1.0, 0.0])
    frames = torch.full((1, 3, slots, 8), -1, dtype=torch.long, device="cuda")
    frames[0, 0, 0, :4] = torch.tensor([0, 0, 1, 2])
    owners = torch.full((1, 3, slots), -1, dtype=torch.long, device="cuda")
    owners[0, :, 0] = 0
    static.update(chirality=vectors, chiral_frame_atoms=frames, chiral_slot_owner=owners)
    # The neighbours now lie where a quarter turn about z puts their axes.
    state["bb_ca"] = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], device="cuda")
    return static, state


def test_builder_reads_chirality_in_the_noisy_frame_only_for_a_checkpoint_trained_that_way() -> None:
    static, state = _chirality_batch()
    padding = static["padding_mask"]

    supplied = _chirality_builder()._chirality(static, state, padding)
    aligned = _chirality_builder(noise_align_chirality=True)._chirality(static, state, padding)

    torch.testing.assert_close(supplied, static["chirality"])
    assert not torch.allclose(aligned, supplied)
    torch.testing.assert_close(
        aligned,
        chirality.align_chirality(
            static["chirality"],
            state["bb_ca"],
            padding,
            static["chiral_frame_atoms"],
            static["chiral_slot_owner"],
        ),
    )
    # A quarter turn about z sends x to y and y to -x.
    torch.testing.assert_close(aligned[0, 1, :3], torch.tensor([0.0, 1.0, 0.0], device="cuda"), atol=1e-5, rtol=0)
    torch.testing.assert_close(aligned[0, 2, :3], torch.tensor([-1.0, 0.0, 0.0], device="cuda"), atol=1e-5, rtol=0)


def test_builder_names_the_frame_tensors_noise_aligned_chirality_needs() -> None:
    static, state = _chirality_batch()
    del static["chiral_slot_owner"]
    with pytest.raises(KeyError, match="chiral_slot_owner"):
        _chirality_builder(noise_align_chirality=True)._chirality(static, state, static["padding_mask"])
    # The frame tensors matter only when the checkpoint asks for the alignment.
    _chirality_builder()._chirality(static, state, static["padding_mask"])


def test_noise_aligned_chirality_is_a_checkpoint_setting_the_weights_cannot_carry() -> None:
    weights = _representation_factory_weights("sequence_probability", 16)
    plain = rfproteina_representation_config_from_weights(weights)
    assert not plain.noise_align_chirality and not plain.noise_align_chirality_translate

    named = rfproteina_representation_config_from_weights(
        weights, noise_align_chirality=True, noise_align_chirality_translate=True
    )
    assert named.noise_align_chirality and named.noise_align_chirality_translate

    parameters = inspect.signature(RFProteina.from_reference_state_dicts).parameters
    assert {"noise_align_chirality", "noise_align_chirality_translate"} <= parameters.keys()


def _representation_factory_weights(schema: str, latent_dim: int) -> dict[str, torch.Tensor]:
    config = RFProteinaRepresentationConfig(input_schema=schema, latent_dim=latent_dim)
    weights = {
        "nn.init_repr_factory.linear_out.weight": torch.empty(8, config.single_input_dim),
        "nn.cond_factory.linear_out.weight": torch.empty(8, config.condition_input_dim),
        "nn.pair_repr_builder.init_repr_factory.linear_out.weight": torch.empty(8, config.pair_input_dim),
        "nn.pair_repr_builder.cond_factory.linear_out.weight": torch.empty(8, config.pair_condition_input_dim),
        "nn.local_latents_linear.1.weight": torch.empty(latent_dim, 8),
    }
    if schema == "sequence_probability":
        weights["nn.init_repr_factory.feat_creators.16.embedding.weight"] = torch.empty(11, 16)
    else:
        weights["nn.pair_repr_builder.init_repr_factory.ln_out.bias"] = torch.empty(8)
        weights["nn.pair_repr_builder.cond_factory.ln_out.bias"] = torch.empty(8)
        if schema == "latest":
            weights["nn.init_repr_factory.feat_creators.3.embedding.weight"] = torch.empty(33, 33)
        else:
            weights["nn.init_repr_factory.feat_creators.5.embedding.weight"] = torch.empty(33, 33)
    return weights


def test_representation_schema_is_read_from_checkpoint_shapes() -> None:
    sequence_probability = rfproteina_representation_config_from_weights(
        _representation_factory_weights("sequence_probability", 16), dtype="float32", pair_adaln_dtype="bfloat16"
    )
    assert sequence_probability.input_schema == "sequence_probability"
    assert sequence_probability.factory_norm == "rms_norm"
    assert sequence_probability.latent_dim == 16
    latest = rfproteina_representation_config_from_weights(_representation_factory_weights("latest", 12))
    assert latest.input_schema == "latest"
    assert latest.factory_norm == "layer_norm"
    legacy = rfproteina_representation_config_from_weights(
        _representation_factory_weights("legacy_self_conditioned", 8)
    )
    assert legacy.input_schema == "legacy_self_conditioned"
    assert sequence_probability.recycle_feature == "recycle_index"
    assert sequence_probability.num_token_kinds == 5
    kinds = _representation_factory_weights("sequence_probability", 16)
    kinds["nn.init_repr_factory.feat_creators.2.embedding.weight"] = torch.empty(7, 32)
    assert rfproteina_representation_config_from_weights(kinds).num_token_kinds == 7
    remaining = _representation_factory_weights("sequence_probability", 16)
    remaining["nn.cond_factory.linear_out.weight"] = torch.empty(8, 610)
    narrowed = rfproteina_representation_config_from_weights(remaining)
    assert narrowed.recycle_feature == "recycle_remaining"
    assert narrowed.recycle_remaining_dim == 2
    none = _representation_factory_weights("sequence_probability", 16)
    none["nn.cond_factory.linear_out.weight"] = torch.empty(8, 608)
    assert rfproteina_representation_config_from_weights(none).recycle_feature is None
    mismatched = _representation_factory_weights("sequence_probability", 16)
    mismatched["nn.cond_factory.linear_out.weight"] = torch.empty(8, 999)
    with pytest.raises(ValueError, match="recycle track"):
        rfproteina_representation_config_from_weights(mismatched)


def test_sequence_probability_conversion_loads_moved_embeddings() -> None:
    trunk = _small_config(latent_dim=4, token_dim=32, pair_dim=16, condition_dim=8)
    config = RFProteinaRepresentationConfig(
        input_schema="sequence_probability",
        latent_dim=4,
        dtype="float32",
        pair_adaln_dtype="float32",
        factory_norm="rms_norm",
    )
    module = RFProteinaRepresentationBuilder(config, trunk)
    charge = torch.randn(11, 16)
    element = torch.randn(128, 16)
    weights = {
        "nn.init_repr_factory.feat_creators.2.embedding.weight": torch.randn(5, 32),
        "nn.init_repr_factory.feat_creators.7.ss_embedding.weight": torch.randn(4, 16),
        "nn.init_repr_factory.feat_creators.16.embedding.weight": charge,
        "nn.init_repr_factory.feat_creators.17.embedding.weight": element,
        "nn.init_repr_factory.linear_out.weight": torch.randn(trunk.token_dim, config.single_input_dim),
        "nn.cond_factory.feat_creators.2.embedding_C.weight": torch.randn(6, 32),
        "nn.cond_factory.feat_creators.2.embedding_A.weight": torch.randn(44, 32),
        "nn.cond_factory.feat_creators.2.embedding_T.weight": torch.randn(1473, 32),
        "nn.cond_factory.linear_out.weight": torch.randn(trunk.condition_dim, config.condition_input_dim),
        "nn.pair_repr_builder.init_repr_factory.linear_out.weight": torch.randn(trunk.pair_dim, config.pair_input_dim),
        "nn.pair_repr_builder.init_repr_factory.ln_out.weight": torch.randn(trunk.pair_dim),
        "nn.pair_repr_builder.cond_factory.linear_out.weight": torch.randn(
            trunk.condition_dim, config.pair_condition_input_dim
        ),
        "nn.pair_repr_builder.cond_factory.ln_out.weight": torch.randn(trunk.condition_dim),
        "nn.pair_repr_builder.adaln.norm_cond.weight": torch.randn(trunk.condition_dim),
        "nn.pair_repr_builder.adaln.to_gamma.0.weight": torch.randn(trunk.pair_dim, trunk.condition_dim),
        "nn.pair_repr_builder.adaln.to_gamma.0.bias": torch.randn(trunk.pair_dim),
        "nn.pair_repr_builder.adaln.to_beta.weight": torch.randn(trunk.pair_dim, trunk.condition_dim),
    }
    converted = convert_rfproteina_representation_builder_torch(config, weights)
    module.load_state_dict(converted, strict=True)
    torch.testing.assert_close(module.charge_embedding.weight, charge)
    torch.testing.assert_close(module.element_embedding.weight, element)


def test_rfproteina_trunk_reads_one_condition_per_recycle_pass() -> None:
    config = _small_config(recycle_enabled=True, pair_write="hadamard")
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)

    class _PassThrough(nn.Module):
        def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            del mask
            return hidden

    # The condition transitions would hide whether each pass received its own row.
    module.transition_c_1 = _PassThrough()
    module.transition_c_2 = _PassThrough()
    mask = torch.ones(1, 6, device="cuda", dtype=torch.bool)
    single = torch.randn(1, 6, config.token_dim, device="cuda")
    pair = torch.randn(1, 6, 6, config.pair_dim, device="cuda")
    stacked = torch.randn(3, 1, 6, config.condition_dim, device="cuda")
    seen: list[torch.Tensor] = []

    def record(_module, args):
        seen.append(args[2].detach().clone())

    handle = module.transformer_layers[0].register_forward_pre_hook(record)
    with torch.inference_mode():
        module(single, pair, stacked, mask, recycle_depth=3)
    handle.remove()
    assert len(seen) == 3
    assert not torch.allclose(seen[0], seen[1])
    assert not torch.allclose(seen[1], seen[2])

    seen.clear()
    shared = torch.randn(1, 6, config.condition_dim, device="cuda")
    handle = module.transformer_layers[0].register_forward_pre_hook(record)
    with torch.inference_mode():
        module(single, pair, shared, mask, recycle_depth=3)
    handle.remove()
    assert len(seen) == 3
    torch.testing.assert_close(seen[0], seen[1])
    torch.testing.assert_close(seen[1], seen[2])
    with pytest.raises(ValueError, match="recycle passes"):
        module(single, pair, stacked[:2], mask, recycle_depth=3)


def test_rfproteina_cudnn_graph_flag_propagates_to_optimized_modules() -> None:
    enabled_config = _small_config(dtype="bfloat16")
    enabled_trunk = LocalLatentsTransformer(enabled_config)
    enabled_update = next(module for module in enabled_trunk.pair_update_layers if module is not None)

    assert enabled_update.enable_cudnn_graph
    assert isinstance(enabled_trunk.transformer_layers[0], ConditionedTransformerBlock)
    assert isinstance(enabled_update, PairUpdateBlock)
    assert enabled_update.transition_out.enable_cudnn_graph

    disabled_config = _small_config(dtype="bfloat16", enable_cudnn_graph=False)
    disabled_trunk = LocalLatentsTransformer(disabled_config)
    disabled_update = next(module for module in disabled_trunk.pair_update_layers if module is not None)

    assert not disabled_update.enable_cudnn_graph
    assert not disabled_update.transition_out.enable_cudnn_graph


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rfproteina_fused_masked_residual_matches_vanilla(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    torch.manual_seed(17)
    value = torch.randn(batch_size, 17, 32, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(value)
    valid_tokens = 14 - torch.arange(batch_size, device="cuda")
    mask = torch.arange(17, device="cuda").unsqueeze(0) < valid_tokens.unsqueeze(1)

    calls = 0
    fused_masked_residual = transformer_blocks.fused_masked_residual

    def tracked_fused(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fused_masked_residual(*args, **kwargs)

    monkeypatch.setattr(transformer_blocks, "fused_masked_residual", tracked_fused)
    expected = (value + update) * mask.unsqueeze(-1)
    destination = value.clone()
    actual = transformer_blocks._apply_masked_residual(destination, update, mask)

    assert calls == 1
    assert actual.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[~mask]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rfproteina_pair_adaln_conditions_each_design_once(dtype: torch.dtype) -> None:
    """Per-design gamma and beta through the fused AdaLN match the upstream per-pair formula."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    pair_dim, condition_dim, tokens = 384, 256, 17
    module = representation_builder._PairAdaptiveLayerNorm(pair_dim, condition_dim, eps=1e-5, dtype=dtype).to(device)
    with torch.no_grad():
        module.norm_cond.weight.normal_(1.0, 0.1)
        for linear in (module.to_gamma[0], module.to_beta):
            linear.weight.normal_(0.0, condition_dim**-0.5)
        module.to_gamma[0].bias.normal_(0.0, 0.5)
    if SM_VERSION in ADALN_SUPPORTED_SMS:
        assert type(module._adaln).__name__ == "AdaLNLayerNormSigmoidCuTe"
    pair = torch.randn(2, tokens, tokens, pair_dim, device=device, dtype=dtype)
    condition = torch.randn(2, condition_dim, device=device, dtype=dtype)
    token_mask = torch.arange(tokens, device=device) < torch.tensor([[tokens], [tokens - 3]], device=device)
    pair_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)

    with torch.inference_mode():
        actual = module(pair, condition, pair_mask)

    # The upstream formula, with each design's condition repeated for every pair, in FP32.
    normed_condition = F.rms_norm(condition.float(), (condition_dim,), module.norm_cond.weight.float(), 1e-5)
    gamma = torch.sigmoid(
        F.linear(normed_condition, module.to_gamma[0].weight.float(), module.to_gamma[0].bias.float())
    )
    beta = F.linear(normed_condition, module.to_beta.weight.float())
    expected = F.rms_norm(pair.float(), (pair_dim,), eps=1e-5) * gamma[:, None, None] + beta[:, None, None]
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual.float(), expected * pair_mask.unsqueeze(-1), atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(actual[~pair_mask]) == 0


@pytest.mark.parametrize("with_mask", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("residual", [False, True], ids=["no_residual", "residual"])
def test_rfproteina_cudnn_pair_transition_matches_vanilla(
    monkeypatch: pytest.MonkeyPatch,
    with_mask: bool,
    residual: bool,
) -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    pair_dim, expansion, tokens = 32, 2, 17
    vanilla = PairTransition(
        c_z=pair_dim,
        n=expansion,
        dtype=torch.bfloat16,
        enable_cudnn_graph=False,
    ).to(device)
    fused = PairTransition(
        c_z=pair_dim,
        n=expansion,
        dtype=torch.bfloat16,
        enable_cudnn_graph=True,
    ).to(device)
    init_module_weights(vanilla)
    fused.load_state_dict(vanilla.state_dict())
    pair = torch.randn(1, tokens, tokens, pair_dim, device=device, dtype=torch.bfloat16)
    token_mask = torch.arange(tokens, device=device) < tokens - 3
    pair_mask = (token_mask.unsqueeze(0) & token_mask.unsqueeze(1)).unsqueeze(0)

    calls = {
        "linear_relu": 0,
        "linear_mask": 0,
        "linear_residual": 0,
        "linear_mask_residual": 0,
    }
    cudnn_linear_relu = transition_layer.cudnn_linear_relu
    cudnn_linear_mask = transition_layer.cudnn_linear_mask
    cudnn_linear_residual = transition_layer.cudnn_linear_residual
    cudnn_linear_mask_residual = transition_layer.cudnn_linear_mask_residual

    def tracked_linear_relu(*args, **kwargs):
        calls["linear_relu"] += 1
        return cudnn_linear_relu(*args, **kwargs)

    def tracked_linear_mask(*args, **kwargs):
        calls["linear_mask"] += 1
        return cudnn_linear_mask(*args, **kwargs)

    def tracked_linear_residual(*args, **kwargs):
        calls["linear_residual"] += 1
        return cudnn_linear_residual(*args, **kwargs)

    def tracked_linear_mask_residual(*args, **kwargs):
        calls["linear_mask_residual"] += 1
        return cudnn_linear_mask_residual(*args, **kwargs)

    monkeypatch.setattr(transition_layer, "cudnn_linear_relu", tracked_linear_relu)
    monkeypatch.setattr(transition_layer, "cudnn_linear_mask", tracked_linear_mask)
    monkeypatch.setattr(transition_layer, "cudnn_linear_residual", tracked_linear_residual)
    monkeypatch.setattr(transition_layer, "cudnn_linear_mask_residual", tracked_linear_mask_residual)
    mask = pair_mask if with_mask else None
    with torch.inference_mode():
        expected = vanilla(pair, mask, residual=residual)
        actual = fused(pair, mask, residual=residual)

    assert calls == {
        "linear_relu": 1,
        "linear_mask": int(with_mask and not residual),
        "linear_residual": int(residual and not with_mask),
        "linear_mask_residual": int(residual and with_mask),
    }
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if with_mask and not residual:
        assert torch.count_nonzero(actual[:, -3:]) == 0
        assert torch.count_nonzero(actual[:, :, -3:]) == 0


def test_pair_transition_fused_residual_matches_pair_update_chain() -> None:
    """The fused transition tail matches the former masked residual chain."""
    torch.manual_seed(31)
    pair_dim, tokens = 32, 17
    transition = PairTransition(c_z=pair_dim, n=2, dtype=torch.bfloat16).cuda()
    init_module_weights(transition)
    pair = torch.randn(2, tokens, tokens, pair_dim, device="cuda", dtype=torch.bfloat16)
    valid_tokens = tokens - 3 - torch.arange(2, device="cuda")
    token_mask = torch.arange(tokens, device="cuda").unsqueeze(0) < valid_tokens.unsqueeze(1)
    pair_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
    mask_value = pair_mask.unsqueeze(-1).to(pair.dtype)
    pair.mul_(mask_value)

    with torch.inference_mode():
        unmasked_update = transition(pair, mask=None)
        expected = transformer_blocks.fused_masked_residual(pair.clone(), unmasked_update, mask_value)
        actual = transition(pair, pair_mask, residual=True)

    assert expected is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_pair_transition_prepares_cudnn_graphs_at_construction_and_cuda_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"linear_relu": 0, "linear_mask": 0}

    def prepare_linear_relu(*args: object, **kwargs: object) -> bool:
        calls["linear_relu"] += 1
        return True

    def prepare_linear_mask(*args: object, **kwargs: object) -> bool:
        calls["linear_mask"] += 1
        return True

    monkeypatch.setattr(transition_layer, "prepare_cudnn_linear_relu", prepare_linear_relu)
    monkeypatch.setattr(transition_layer, "prepare_cudnn_linear_mask", prepare_linear_mask)
    # The default static path holds no plan, so nothing is prepared for it.
    PairTransition(c_z=32, n=2, dtype=torch.bfloat16, enable_cudnn_graph=True)
    assert calls == {"linear_relu": 0, "linear_mask": 0}

    module = PairTransition(
        c_z=32,
        n=2,
        dtype=torch.bfloat16,
        enable_cudnn_graph=True,
        cudnn_dynamic_shapes=True,
    )
    assert calls == {"linear_relu": 1, "linear_mask": 1}

    module.cpu()
    assert calls == {"linear_relu": 1, "linear_mask": 1}
    module.cuda()
    assert calls == {"linear_relu": 2, "linear_mask": 2}


@pytest.mark.parametrize(
    "pair_dim,trimul_hidden_dim",
    [
        # The 2026-07-03 and 2026-07-22 releases: both keep a 196-wide trimul
        # while the newer one widens the pair track. The last pair is a
        # synthetic combination, there to show the two widths are derived
        # independently rather than tied to each other.
        (256, 196),
        (384, 196),
        (384, 256),
    ],
)
def test_rfproteina_trunk_config_derived_from_weights(pair_dim: int, trimul_hidden_dim: int) -> None:
    expected = {
        "token_dim": 768,
        "pair_dim": pair_dim,
        "condition_dim": 256,
        "latent_dim": 8,
        "num_layers": 14,
        "num_heads": 12,
        "pair_update_every": 2,
        "trimul_hidden_dim": trimul_hidden_dim,
        "transition_expansion": 4,
        "condition_transition_expansion": 2,
        "pair_transition_expansion": 2,
    }

    config = rfproteina_trunk_config_from_weights(_oss_trunk_shapes(**expected), dtype="float32")

    for field, value in expected.items():
        assert getattr(config, field) == value, field
    assert config.use_qk_norm
    assert config.use_insertion_head
    assert config.trimul_bias
    assert config.dtype == "float32"


def _with_lora_on_qkv(weights: dict[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    """The released PEFT layout for one layer: ``base_layer`` plus ``lora_A``/``lora_B``, no plain ``weight``."""
    adapted = dict(weights)
    name = "nn.transformer_layers.0.mhba.mha.to_qkv"
    base = adapted.pop(f"{name}.weight")
    adapted[f"{name}.base_layer.weight"] = torch.randn_like(base)
    adapted[f"{name}.lora_A.default.weight"] = torch.randn(rank, base.shape[1])
    adapted[f"{name}.lora_B.default.weight"] = torch.randn(base.shape[0], rank)
    return adapted


def test_rfproteina_unmerged_lora_is_rejected_not_dropped_or_a_missing_key() -> None:
    shapes = _oss_trunk_shapes(768, 384, 256, 32, 14, 12, 2, 256, 4, 2, 2)
    adapted = _with_lora_on_qkv(shapes, rank=4)

    with pytest.raises(ValueError, match=r"unmerged PEFT LoRA.*merge_rfproteina_lora"):
        rfproteina_trunk_config_from_weights(adapted)


def test_merge_rfproteina_lora_folds_the_adapter_into_the_base_weight() -> None:
    shapes = _oss_trunk_shapes(768, 384, 256, 32, 14, 12, 2, 256, 4, 2, 2)
    adapted = _with_lora_on_qkv(shapes, rank=4)
    name = "nn.transformer_layers.0.mhba.mha.to_qkv"

    merged = merge_rfproteina_lora(adapted, lora_alpha=8.0)

    expected = adapted[f"{name}.base_layer.weight"] + 2.0 * (
        adapted[f"{name}.lora_B.default.weight"] @ adapted[f"{name}.lora_A.default.weight"]
    )
    torch.testing.assert_close(merged[f"{name}.weight"], expected)
    assert not [key for key in merged if "lora_" in key or "base_layer" in key]
    config = rfproteina_trunk_config_from_weights(merged)
    assert (config.token_dim, config.latent_dim) == (768, 32)


def test_rfproteina_trunk_config_mismatch_names_the_wrong_fields() -> None:
    # Legacy tensor headers must identify every architecture mismatch against
    # the September z12 defaults.
    weights = _oss_trunk_shapes(
        token_dim=768,
        pair_dim=256,
        condition_dim=256,
        latent_dim=8,
        num_layers=14,
        num_heads=12,
        pair_update_every=2,
        trimul_hidden_dim=196,
        transition_expansion=4,
        condition_transition_expansion=2,
        pair_transition_expansion=2,
    )
    module = LocalLatentsTransformer(RFProteinaConfig().trunk)

    with pytest.raises(ValueError, match="does not match the checkpoint") as e:
        _check_architecture(module.config, weights, "nn")

    assert "pair_dim (config 384, checkpoint 256)" in str(e.value)
    assert "latent_dim (config 12, checkpoint 8)" in str(e.value)
    assert "trimul_hidden_dim (config 256, checkpoint 196)" in str(e.value)
    assert "pair_write (config hadamard, checkpoint sum)" in str(e.value)
    assert "recycle_enabled (config True, checkpoint False)" in str(e.value)


def test_rfproteina_pair_bias_attention_matches_sdpa_reference() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    token_dim, pair_dim, num_heads = 32, 16, 4
    head_dim = token_dim // num_heads
    module = _rfproteina_attention(token_dim, pair_dim, num_heads, "SDPA", torch.float32).to(device)
    # TRT-BNM Linear weights are uninitialized until a checkpoint load; fill
    # them so the module runs on finite weights regardless of test order.
    init_module_weights(module)
    b, n = 2, 11
    node = torch.randn(b, n, token_dim, device=device)
    pair = torch.randn(b, n, n, pair_dim, device=device)
    mask = torch.tensor([[1] * n, [1] * 7 + [0] * 4], device=device, dtype=torch.bool)

    s = module.norm_s(node)
    q, g, k, v = F.linear(s, module.in_proj.weight, module.in_proj.bias).split([token_dim] * 4, dim=-1)
    q = module.q_norm(q).view(b, n, num_heads, head_dim).transpose(1, 2)
    k = module.k_norm(k).view(b, n, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(b, n, num_heads, head_dim).transpose(1, 2)
    pair_bias = module.proj_z(pair).movedim(-1, 1)
    mask_bias = (1.0 - mask.float())[:, None, None, :] * -1e4
    reference = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask_bias + pair_bias,
    )
    reference = reference.transpose(1, 2).reshape_as(s)
    reference = module.proj_o(torch.sigmoid(g) * reference)

    with torch.inference_mode():
        actual = module(node, pair, mask)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


def test_rfproteina_attention_conversion_packs_oss_projections() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    token_dim, pair_dim, num_heads = 32, 16, 4
    shapes = {
        "node_norm.weight": (token_dim,),
        "q_layer_norm.weight": (token_dim,),
        "k_layer_norm.weight": (token_dim,),
        "to_qkv.weight": (3 * token_dim, token_dim),
        "to_qkv.bias": (3 * token_dim,),
        "to_g.weight": (token_dim, token_dim),
        "to_g.bias": (token_dim,),
        "to_out_node.weight": (token_dim, token_dim),
        "to_out_node.bias": (token_dim,),
        "pair_norm.weight": (pair_dim,),
        "to_bias.weight": (num_heads, pair_dim),
    }
    weights = {f"mhba.mha.{name}": torch.randn(shape, device=device) for name, shape in shapes.items()}
    converted = {}
    _convert_attention(converted, weights, "mhba.mha", "mha", torch.float32, use_qk_norm=True)
    module = _rfproteina_attention(token_dim, pair_dim, num_heads, "SDPA", torch.float32).to(device)
    module.load_state_dict({key.removeprefix("mha."): value for key, value in converted.items()}, strict=True)

    s = torch.randn(3, token_dim, device=device)
    q, g, k, v = F.linear(s, module.in_proj.weight, module.in_proj.bias).split([token_dim] * 4, dim=-1)
    expected_q, expected_k, expected_v = F.linear(
        s, weights["mhba.mha.to_qkv.weight"], weights["mhba.mha.to_qkv.bias"]
    ).chunk(3, dim=-1)
    expected_g = F.linear(s, weights["mhba.mha.to_g.weight"], weights["mhba.mha.to_g.bias"])
    for actual, expected in ((q, expected_q), (g, expected_g), (k, expected_k), (v, expected_v)):
        torch.testing.assert_close(actual, expected)


def test_rfproteina_cutedsl_attention_matches_sdpa() -> None:
    skip_if_cutedsl("CuTeDSL", "pairwise_attention")
    torch.manual_seed(11)
    device = torch.device("cuda")
    token_dim, pair_dim, num_heads = 64, 16, 1
    reference = _rfproteina_attention(token_dim, pair_dim, num_heads, "SDPA", torch.bfloat16).to(device)
    init_module_weights(reference)
    actual_module = _rfproteina_attention(token_dim, pair_dim, num_heads, "CuTeDSL", torch.bfloat16).to(device)
    actual_module.load_state_dict(reference.state_dict())
    node = torch.randn(2, 17, token_dim, device=device, dtype=torch.bfloat16)
    pair = torch.randn(2, 17, 17, pair_dim, device=device, dtype=torch.bfloat16)
    mask = torch.tensor([[1] * 17, [1] * 13 + [0] * 4], device=device, dtype=torch.bool)

    with torch.inference_mode():
        expected = reference(node, pair, mask)
        actual = actual_module(node, pair, mask)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def _reference_trimul(node, pair, pair_mask):
    normalized = F.layer_norm(pair, (pair.shape[-1],), node.norm_in.weight, node.norm_in.bias, node.eps)
    projected = F.linear(normalized, node.p_in.weight, node.p_in.bias)
    gates = F.linear(normalized, node.g_in.weight, node.g_in.bias)
    a_p, b_p = projected.chunk(2, dim=-1)
    a_g, b_g = gates.chunk(2, dim=-1)
    mask = pair_mask.unsqueeze(-1).to(pair.dtype)
    a = a_p * torch.sigmoid(a_g) * mask
    b = b_p * torch.sigmoid(b_g) * mask
    if node.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
        update = torch.einsum("bikd,bjkd->bijd", a, b)
    else:
        update = torch.einsum("bkid,bkjd->bijd", a, b)
    update = F.layer_norm(update, (update.shape[-1],), node.norm_out.weight, node.norm_out.bias, node.eps)
    update = F.linear(update, node.p_out.weight, node.p_out.bias)
    gate = torch.sigmoid(F.linear(normalized, node.g_out.weight, node.g_out.bias))
    return update * gate


def _reference_pair_update(module, single, pair, mask):
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pair_mask_value = pair_mask.unsqueeze(-1).to(pair.dtype)
    single = single * mask.unsqueeze(-1)
    normalized = F.layer_norm(
        single, (single.shape[-1],), module.layer_norm_in.weight, module.layer_norm_in.bias, module.layer_norm_in.eps
    )
    row, column = F.linear(normalized, module.linear_x.weight).chunk(2, dim=-1)
    if module.pair_write == "sum":
        pair = pair + row.unsqueeze(1) + column.unsqueeze(2)
    else:
        pair = pair + row.unsqueeze(1) * column.unsqueeze(2)
    pair = pair * pair_mask_value
    pair = (pair + _reference_trimul(module.tri_mult_out, pair, pair_mask)) * pair_mask_value
    pair = (pair + _reference_trimul(module.tri_mult_in, pair, pair_mask)) * pair_mask_value
    transition = F.layer_norm(
        pair,
        (pair.shape[-1],),
        module.transition_out.layer_norm.weight,
        module.transition_out.layer_norm.bias,
        module.transition_out.layer_norm.eps,
    )
    transition = F.relu(
        F.linear(transition, module.transition_out.linear_1.weight, module.transition_out.linear_1.bias)
    )
    transition = F.linear(transition, module.transition_out.linear_2.weight, module.transition_out.linear_2.bias)
    return (pair + transition * pair_mask_value) * pair_mask_value


@pytest.mark.parametrize("pair_write", ["sum", "hadamard"])
def test_rfproteina_pair_update_supports_nonstandard_hidden_dim(pair_write: str) -> None:
    torch.manual_seed(13)
    module = RFProteinaPairUpdate(
        layer_idx=0,
        token_dim=32,
        pair_dim=16,
        trimul_hidden_dim=8,
        pair_transition_expansion=2,
        pair_mask_left_aligned=True,
        trimul_high_precision=False,
        eps=1e-5,
        dtype=torch.float32,
        pair_write=pair_write,
    ).cuda()
    init_module_weights(module)
    single = torch.randn(2, 12, 32, device="cuda")
    pair = torch.randn(2, 12, 12, 16, device="cuda")
    pair_before = pair.clone()
    mask = torch.tensor([[1] * 12, [1] * 9 + [0] * 3], device="cuda", dtype=torch.bool)
    pair_masks = precompute_pair_masks(
        "CuTeDSL",
        mask.unsqueeze(-1) & mask.unsqueeze(-2),
        dtype=pair.dtype,
    )
    actual_seqlens = {}

    def capture_actual_seqlen(name):

        def hook(_module, _args, kwargs):
            metadata = kwargs["trimul_metadata"]
            actual_seqlens[name] = metadata.outgoing_actual_seqlen if name == "out" else metadata.incoming_actual_seqlen

        return hook

    handles = [
        module.tri_mult_out.register_forward_pre_hook(capture_actual_seqlen("out"), with_kwargs=True),
        module.tri_mult_in.register_forward_pre_hook(capture_actual_seqlen("in"), with_kwargs=True),
    ]
    trimul_metadata = precompute_trimul_metadata(
        pair,
        pair_masks.mask_bias,
        pair_masks.mask_bias_transposed,
    )

    with torch.inference_mode():
        reference = _reference_pair_update(module, single, pair, mask)
        output = module(single, pair, mask, trimul_metadata, precomputed_masks=pair_masks)
    for handle in handles:
        handle.remove()
    assert output.shape == pair.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(pair, pair_before)
    assert actual_seqlens["out"] is pair_masks.mask_bias
    assert actual_seqlens["in"] is pair_masks.mask_bias_transposed
    assert module.transition_out.auto_chunk_policy is not None
    assert torch.count_nonzero(output[1, 9:]) == 0
    assert torch.count_nonzero(output[1, :, 9:]) == 0


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("pair_write", ["sum", "hadamard"])
def test_rfproteina_cudnn_pair_update_matches_vanilla(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    pair_write: str,
) -> None:
    """Both write modes fuse injection and triangle residuals.

    PairTransition owns its final residual. Triangle updates use the CuTe
    epilogue when covered and the Triton residual otherwise.
    """
    torch.manual_seed(29)
    device = torch.device("cuda")
    kwargs = {
        "layer_idx": 0,
        "token_dim": 32,
        "pair_dim": 16,
        "trimul_hidden_dim": 8,
        "pair_transition_expansion": 2,
        "pair_mask_left_aligned": True,
        "trimul_high_precision": False,
        "eps": 1e-5,
        "dtype": torch.bfloat16,
        "pair_write": pair_write,
    }
    vanilla = RFProteinaPairUpdate(**kwargs, enable_cudnn_graph=False).to(device)
    fused = RFProteinaPairUpdate(**kwargs, enable_cudnn_graph=True).to(device)
    init_module_weights(vanilla)
    fused.load_state_dict(vanilla.state_dict())
    tokens = 17
    single = torch.randn(batch_size, tokens, 32, device=device, dtype=torch.bfloat16)
    pair = torch.randn(batch_size, tokens, tokens, 16, device=device, dtype=torch.bfloat16)
    valid_tokens = tokens - 3 - torch.arange(batch_size, device=device)
    token_mask = torch.arange(tokens, device=device).unsqueeze(0) < valid_tokens.unsqueeze(1)
    pair_masks = precompute_pair_masks(
        "CuTeDSL",
        token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2),
        dtype=pair.dtype,
    )

    calls = {"injection": 0, "residual": 0}
    fused_masked_residual = transformer_blocks.fused_masked_residual
    triton_injection = fused.pair_injection._fused_kernel

    def tracked_injection(*args, **kwargs):
        calls["injection"] += 1
        return triton_injection(*args, **kwargs)

    def tracked_residual(*args, **kwargs):
        calls["residual"] += 1
        return fused_masked_residual(*args, **kwargs)

    monkeypatch.setattr(fused.pair_injection, "_fused_kernel", tracked_injection)
    monkeypatch.setattr(transformer_blocks, "fused_masked_residual", tracked_residual)
    trimul_metadata = precompute_trimul_metadata(
        pair,
        pair_masks.mask_bias,
        pair_masks.mask_bias_transposed,
    )
    with torch.inference_mode():
        expected = vanilla(single, pair.clone(), token_mask, trimul_metadata, precomputed_masks=pair_masks)
        actual = fused(single, pair.clone(), token_mask, trimul_metadata, precomputed_masks=pair_masks)

    # Two triangle residuals per arm. The patch counts both arms.
    assert calls == {"injection": 1, "residual": 4}
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[:, -3:]) == 0
    assert torch.count_nonzero(actual[:, :, -3:]) == 0


@pytest.mark.parametrize(
    ("pair_dim", "trimul_hidden_dim", "fused_residual"),
    [
        (256, 196, False),
        (384, 196, False),
        (384, 256, True),
    ],
    ids=["public", "2026-07", "z12"],
)
def test_rfproteina_pair_update_dispatches_residual_only_x0_x1(
    pair_dim: int,
    trimul_hidden_dim: int,
    fused_residual: bool,
) -> None:
    """The trunk uses native widths and reserves CuTe x0_x1 for fused residuals."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 90):
        pytest.skip(f"RFProteína dual-GEMM tuning requires SM80 or SM90, got SM{SM_VERSION}")

    module = RFProteinaPairUpdate(
        layer_idx=0,
        token_dim=32,
        pair_dim=pair_dim,
        trimul_hidden_dim=trimul_hidden_dim,
        pair_transition_expansion=2,
        pair_mask_left_aligned=True,
        trimul_high_precision=False,
        eps=1e-5,
        dtype=torch.bfloat16,
    )
    for node in (module.tri_mult_out, module.tri_mult_in):
        assert node._dual_gemm_x_x_op_transpose.__name__ == "_invoke_cute_dual_gemm_x_x"
        assert node._dual_gemm_x0_x1_op.__name__ == "_invoke_vanilla_dual_gemm_x0_x1"
        aligned_pair = torch.empty(1, 256, 256, pair_dim, device="meta")
        if fused_residual:
            assert node._dual_gemm_x0_x1_residual_op.__name__ == "_invoke_cute_dual_gemm_x0_x1"
            assert node.can_fuse_residual(aligned_pair)
            assert not node.can_fuse_residual(aligned_pair[:, :255, :255])
        else:
            assert node._dual_gemm_x0_x1_residual_op is None
            assert not node.can_fuse_residual(aligned_pair)


def test_rfproteina_local_latents_transformer_shapes_and_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(17)
    config = _small_config()
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)
    metadata_calls = 0
    original_precompute = precompute_trimul_metadata

    def track_precompute(self, *args, **kwargs):
        nonlocal metadata_calls
        metadata_calls += 1
        return original_precompute(self, *args, **kwargs)

    monkeypatch.setattr(local_latents_transformer, "precompute_trimul_metadata", track_precompute)
    batch, length = 2, 10
    mask = torch.tensor([[1] * length, [1] * 6 + [0] * 4], device="cuda", dtype=torch.bool)
    single = torch.randn(batch, length, config.token_dim, device="cuda")
    pair = torch.randn(batch, length, length, config.pair_dim, device="cuda")
    condition = torch.randn(batch, length, config.condition_dim, device="cuda")
    attention_mask_biases = []

    def capture_mask_bias(_module, _args, kwargs):
        attention_mask_biases.append(kwargs["mask_bias"])

    handles = [
        layer.mha.register_forward_pre_hook(capture_mask_bias, with_kwargs=True) for layer in module.transformer_layers
    ]
    with torch.inference_mode():
        output = module(single, pair, condition, mask)
    for handle in handles:
        handle.remove()
    assert output["bb_ca"]["v"].shape == (batch, length, 3)
    assert output["local_latents"]["v"].shape == (batch, length, config.latent_dim)
    assert output["insertion"]["logits"].shape == (batch, length, length)
    assert torch.count_nonzero(output["bb_ca"]["v"][1, 6:]) == 0
    assert torch.count_nonzero(output["local_latents"]["v"][1, 6:]) == 0
    assert len(attention_mask_biases) == config.num_layers
    assert metadata_calls == 1
    assert all(mask_bias is attention_mask_biases[0] for mask_bias in attention_mask_biases[1:])
    # The layer loop runs on the tokens padded to 16, all of them masked.
    expected_mask_bias = (1 - F.pad(mask, (0, 6)).float()) * -config.mask_inf
    torch.testing.assert_close(attention_mask_biases[0], expected_mask_bias[:, None, None, :])


def test_rfproteina_local_latents_transformer_runs_requested_recycle_depth() -> None:
    config = _small_config(recycle_enabled=True, pair_write="hadamard")
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)
    mask = torch.ones(1, 6, device="cuda", dtype=torch.bool)
    inputs = (
        torch.randn(1, 6, config.token_dim, device="cuda"),
        torch.randn(1, 6, 6, config.pair_dim, device="cuda"),
        torch.randn(1, 6, config.condition_dim, device="cuda"),
        mask,
    )
    passes = 0

    def count_passes(_module, _args):
        nonlocal passes
        passes += 1

    handle = module.transformer_layers[0].register_forward_pre_hook(count_passes)
    with torch.inference_mode():
        output = module(*inputs, recycle_depth=3)
    handle.remove()

    assert passes == 3
    assert torch.isfinite(output["bb_ca"]["v"]).all()

    non_recycling = LocalLatentsTransformer(_small_config()).cuda().eval()
    with pytest.raises(ValueError, match="requires a recycling checkpoint"):
        non_recycling(*inputs, recycle_depth=2)


def test_rfproteina_trunk_reads_a_shared_condition_as_one_per_pass() -> None:
    """A shared condition and a per-pass axis carrying that same condition agree exactly.

    The caller supplies whichever form the checkpoint asks for, and the step takes the
    pass's condition either way, so the two must not drift.
    """
    torch.manual_seed(31)
    config = _small_config(recycle_enabled=True, num_layers=3, dtype="bfloat16")
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)
    mask = torch.tensor([[1] * 10, [1] * 7 + [0] * 3], device="cuda", dtype=torch.bool)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    single = torch.randn(2, 10, config.token_dim, **bf16)
    pair = torch.randn(2, 10, 10, config.pair_dim, **bf16)
    condition = torch.randn(2, 10, config.condition_dim, **bf16)
    with torch.inference_mode():
        shared = module(single, pair, condition, mask, recycle_depth=3)
        per_pass = module(single, pair, condition.expand(3, *condition.shape), mask, recycle_depth=3)

    for head, fields in per_pass.items():
        for field, expected in fields.items():
            torch.testing.assert_close(shared[head][field], expected, atol=0, rtol=0)


def test_rfproteina_trunk_recycles_every_pass_including_the_zero_state() -> None:
    """Every pass recycles, so one captured step serves them all whatever the depth.

    The first pass folds in the zero state, whose bias-free norm and projection are
    exactly zero. Adding that to the initial state returns it bit for bit, so giving
    the first pass the same form as the rest costs the result nothing.
    """
    torch.manual_seed(37)
    config = _small_config(recycle_enabled=True, dtype="bfloat16")
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)
    calls = []
    handles = [
        getattr(module, name).register_forward_hook(lambda _module, _args, _output, name=name: calls.append(name))
        for name in ("s_cycle_proj", "z_cycle_proj")
    ]
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    mask = torch.ones(1, 6, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        module(
            torch.randn(1, 6, config.token_dim, **bf16),
            torch.randn(1, 6, 6, config.pair_dim, **bf16),
            torch.randn(1, 6, config.condition_dim, **bf16),
            mask,
            recycle_depth=3,
        )
        for handle in handles:
            handle.remove()
        zero_single = module.s_cycle_proj(module.s_cycle_norm(torch.zeros(1, 6, config.token_dim, **bf16)))
        zero_pair = module.z_cycle_proj(module.z_cycle_norm(torch.zeros(1, 6, 6, config.pair_dim, **bf16)))

    assert calls == ["s_cycle_proj", "z_cycle_proj"] * 3, "every pass recycles, the first included"
    assert torch.count_nonzero(zero_single) == 0
    assert torch.count_nonzero(zero_pair) == 0
    # The identity the uniform first pass rests on: adding that zero changes nothing.
    initial = torch.randn(1, 6, config.token_dim, **bf16)
    assert torch.equal(initial + zero_single, initial)


def test_cutedsl_single_masks_count_valid_keys_once() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    masks = precompute_single_masks("CuTeDSL", mask, activation_dtype=torch.bfloat16)
    assert masks.mask_bias.dtype == torch.int32
    assert masks.mask_bias.tolist() == [3, 2]
    assert torch.equal(masks.excluded, ~mask.unsqueeze(-1))
    assert torch.equal(masks.value_mask, mask.unsqueeze(-1).to(torch.bfloat16))


def test_rfproteina_pair_bias_groups_follow_pair_updates() -> None:
    updates = nn.ModuleList([nn.Identity() if i % 2 == 0 else None for i in range(13)])
    groups = local_latents_transformer._shared_pair_bias_groups(updates, 14, max_layers=5)
    assert groups == [(0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13,)]
    # Without updates, the 64-head cap splits the run.
    assert local_latents_transformer._shared_pair_bias_groups(nn.ModuleList([None] * 4), 5, max_layers=2) == [
        (0, 1),
        (2, 3),
        (4,),
    ]


@pytest.mark.parametrize("backend", ["CuTeDSL", "SDPA"])
def test_rfproteina_trunk_shares_pair_bias_projection_bit_exactly(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Layers between pair updates project their biases in one pass, matching per-layer projections exactly."""
    if backend == "CuTeDSL":
        skip_if_cutedsl("CuTeDSL", "pairwise_attention")
    torch.manual_seed(23)
    config = _small_config(
        token_dim=128,
        pair_dim=64,
        num_layers=5,
        pair_update_every=2,
        recycle_enabled=True,
        dtype="bfloat16",
        pairwise_attention_backend=backend,
    )
    module = LocalLatentsTransformer(config).cuda().eval()
    init_module_weights(module)
    if 2 not in module._shared_pair_bias:
        pytest.skip("the shared projection needs the streaming pair-bias kernel")
    batch, length = 2, 24
    mask = torch.tensor([[1] * length, [1] * 19 + [0] * 5], device="cuda", dtype=torch.bool)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    inputs = (
        torch.randn(batch, length, config.token_dim, **bf16),
        torch.randn(batch, length, length, config.pair_dim, **bf16),
        torch.randn(batch, length, config.condition_dim, **bf16),
        mask,
    )
    projected_heads = []
    forward = LNProjMoveaxisPad.forward

    def record(self, z, *args, **kwargs):
        out = forward(self, z, *args, **kwargs)
        projected_heads.append(out.shape[-3])
        return out

    monkeypatch.setattr(LNProjMoveaxisPad, "forward", record)
    with torch.inference_mode():
        shared = module(*inputs, recycle_depth=2)
        shared_heads, projected_heads[:] = list(projected_heads), []
        projections, module._shared_pair_bias = module._shared_pair_bias, {}
        per_layer = module(*inputs, recycle_depth=2)

    heads = config.num_heads
    # Each pass: layer 0 alone, then layers (1, 2) and (3, 4) in one projection each.
    assert shared_heads == [heads, 2 * heads, 2 * heads] * 2
    assert projected_heads == [heads] * config.num_layers * 2
    assert all(layer.mha.bias_proj for layer in module.transformer_layers)
    for head, fields in per_layer.items():
        for field, expected in fields.items():
            torch.testing.assert_close(shared[head][field], expected, atol=0, rtol=0)
    assert torch.count_nonzero(shared["bb_ca"]["v"][1, 19:]) == 0
    module._shared_pair_bias = projections


@pytest.mark.parametrize(
    ("pair_dim", "trimul_hidden_dim"),
    [(64, 64), (384, 256)],
    ids=["kf-chain", "z12"],
)
def test_rfproteina_local_latents_transformer_token_padding_fuses_trimul_residual(
    monkeypatch: pytest.MonkeyPatch,
    pair_dim: int,
    trimul_hidden_dim: int,
) -> None:
    """Padding 13 tokens to 16 lets every trimul fuse its residual without changing the heads."""
    skip_if_no_cutedsl()
    if SM_VERSION not in (80, 90):
        pytest.skip(f"RFProteína dual-GEMM tuning requires SM80 or SM90, got SM{SM_VERSION}")

    torch.manual_seed(17)
    dims = {
        "pair_dim": pair_dim,
        "trimul_hidden_dim": trimul_hidden_dim,
        "pair_write": "hadamard",
        "recycle_enabled": True,
        "dtype": "bfloat16",
    }
    padded = LocalLatentsTransformer(_small_config(**dims)).cuda().eval()
    init_module_weights(padded)
    unpadded = LocalLatentsTransformer(_small_config(**dims, enable_token_pad=False)).cuda().eval()
    unpadded.load_state_dict(padded.state_dict())
    padded_nodes = [module for module in padded.modules() if isinstance(module, TriangleMultiplicationNode)]
    assert padded_nodes and all(node.pad_tokens for node in padded_nodes)
    assert not any(module.pad_tokens for module in unpadded.modules() if isinstance(module, TriangleMultiplicationNode))

    calls = {"unfused": 0, "kf": 0}
    masked_residual = PairUpdateBlock._masked_residual
    kf_forward = TriangleMultiplicationNode._kf_forward_if_supported

    def count_unfused(self, *args):
        calls["unfused"] += 1
        return masked_residual(self, *args)

    def count_kf(self, *args):
        output = kf_forward(self, *args)
        calls["kf"] += output is not None
        return output

    monkeypatch.setattr(PairUpdateBlock, "_masked_residual", count_unfused)
    monkeypatch.setattr(TriangleMultiplicationNode, "_kf_forward_if_supported", count_kf)
    pairs = {"padded": [], "unpadded": []}
    for name, module in (("padded", padded), ("unpadded", unpadded)):
        module.pair_update_layers[0].register_forward_hook(
            lambda _module, _args, output, name=name: pairs[name].append(output.clone())
        )
    length = 13
    mask = torch.arange(length, device="cuda") < torch.tensor([[length], [9]], device="cuda")
    inputs = (
        torch.randn(2, length, padded.token_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(2, length, length, pair_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(2, length, padded.config.condition_dim, device="cuda", dtype=torch.bfloat16),
        mask,
    )
    # Two trimuls in the one pair update, over two recycling passes.
    residuals = 4
    with torch.inference_mode():
        expected = unpadded(*inputs, recycle_depth=2)
        assert calls == {"unfused": residuals, "kf": 0}
        calls["unfused"] = 0
        actual = padded(*inputs, recycle_depth=2)
    assert calls == {"unfused": 0, "kf": residuals if padded_nodes[0]._kf_ops is not None else 0}

    assert len(pairs["padded"]) == len(pairs["unpadded"]) == 2
    for padded_pair, expected_pair in zip(pairs["padded"], pairs["unpadded"], strict=True):
        assert padded_pair.shape[1:3] == (16, 16)
        assert torch.count_nonzero(padded_pair[:, length:]) == torch.count_nonzero(padded_pair[:, :, length:]) == 0
        # About one BF16 ulp at the pair's largest magnitudes.
        torch.testing.assert_close(padded_pair[:, :length, :length], expected_pair, atol=6e-2, rtol=1.6e-2)
    for head, key in (("bb_ca", "v"), ("local_latents", "v"), ("insertion", "logits")):
        torch.testing.assert_close(actual[head][key], expected[head][key], atol=2e-3, rtol=1.6e-2)


def test_rfproteina_load_weights_folds_released_lora_by_path() -> None:
    flow = resolve_rfproteina_checkpoint()
    autoencoder = resolve_rfproteina_autoencoder_checkpoint()
    if flow is None or not flow.is_file() or autoencoder is None or not autoencoder.is_file():
        pytest.skip("RFPROTEINA_LDM_CKPT and RFPROTEINA_AE_CKPT are not set to the released checkpoints")

    # The in-memory entry point sizes the model from the weights; the path loader must then reach the same tensors.
    flow_state = load_rfproteina_checkpoint_state(flow)
    model = RFProteina.from_reference_state_dicts(flow_state, load_rfproteina_checkpoint_state(autoencoder))
    expected = {key: value.clone() for key, value in model.velocity_model.state_dict().items()}
    with torch.no_grad():
        for value in model.velocity_model.parameters():
            value.zero_()

    report = model.load_weights(flow)

    assert not report.missing_target_keys
    for key, value in model.velocity_model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0, msg=key)
    with pytest.raises(ValueError, match="unmerged PEFT LoRA"):
        model.load_weights(flow, lora_alpha=None)


def test_rfproteina_released_checkpoint_conversion() -> None:
    checkpoint = resolve_rfproteina_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("RFPROTEINA_LDM_CKPT is not set to the released checkpoint")

    state = merge_rfproteina_lora(load_rfproteina_checkpoint_state(checkpoint), lora_alpha=RELEASED_LORA_ALPHA)
    trunk_config = rfproteina_trunk_config_from_weights(state)
    module = LocalLatentsTransformer(trunk_config)
    report = load_rfproteina_trunk_checkpoint(module, state)
    assert not report.missing_target_keys
    assert len(report.loaded_keys) == len(module.state_dict())
    assert any(key.startswith("init_repr_factory") for key in report.ignored_source_keys)

    # Read the feature schema off the checkpoint, so a release that moves it still loads.
    representation_config = rfproteina_representation_config_from_weights(
        state,
        latent_dim=trunk_config.latent_dim,
        dtype="float32",
        pair_adaln_dtype=trunk_config.dtype,
    )
    representation_builder = RFProteinaRepresentationBuilder(representation_config, trunk_config)
    representation_report = load_rfproteina_representation_builder_checkpoint(representation_builder, state)
    assert not representation_report.missing_target_keys
    assert len(representation_report.loaded_keys) == len(representation_builder.state_dict())

    module = module.cuda().eval()
    config = module.config
    with torch.inference_mode():
        output = module(
            torch.randn(1, 8, config.token_dim, device="cuda", dtype=config.torch_dtype),
            torch.randn(1, 8, 8, config.pair_dim, device="cuda", dtype=config.torch_dtype),
            torch.randn(1, 8, config.condition_dim, device="cuda", dtype=config.torch_dtype),
            torch.ones(1, 8, device="cuda", dtype=torch.bool),
        )
    assert torch.isfinite(output["bb_ca"]["v"]).all()
    assert torch.isfinite(output["local_latents"]["v"]).all()


def test_rfproteina_decoder_shapes_and_ca_anchoring() -> None:
    config = RFProteinaDecoderConfig(
        dtype="float32", num_layers=2, token_dim=96, pair_dim=32, condition_dim=16, num_heads=6
    )
    module = RFProteinaLatentDecoder(config).cuda().eval()
    init_module_weights(module)

    backbone_ca = torch.randn(2, 12, 3, device="cuda")
    latents = torch.randn(2, 12, config.latent_dim, device="cuda")
    mask = torch.ones(2, 12, dtype=torch.bool, device="cuda")
    mask[1, 7:] = False
    attention_mask_biases = []

    def capture_mask_bias(_module, _args, kwargs):
        attention_mask_biases.append(kwargs["mask_bias"])

    handles = [
        layer.mha.register_forward_pre_hook(capture_mask_bias, with_kwargs=True) for layer in module.transformer_layers
    ]

    with torch.inference_mode():
        output = module(backbone_ca, latents, mask)
    for handle in handles:
        handle.remove()

    assert output["coords"].shape == (2, 12, 37, 3)
    assert output["sequence_logits"].shape == (2, 12, config.num_residue_types)
    assert output["atom_mask"].shape == (2, 12, 37)
    assert torch.isfinite(output["coords"]).all()
    assert len(attention_mask_biases) == config.num_layers
    assert all(mask_bias is attention_mask_biases[0] for mask_bias in attention_mask_biases[1:])

    # The generated CA is authoritative and the padded tail stays empty.
    torch.testing.assert_close(output["coords"][..., 1, :], backbone_ca)
    torch.testing.assert_close(
        output["atom_mask"],
        local_latents_decoder.get_residue_atom_mask(
            output["residue_type"],
            config.num_residue_types,
            mask,
        ),
    )
    assert torch.count_nonzero(output["sequence_logits"][1, 7:]) == 0


def test_rfproteina_atom37_masks_match_unified_contract() -> None:
    unified = local_latents_decoder.get_atom37_mask_table(n_residue_types=33)
    legacy = local_latents_decoder.get_atom37_mask_table(n_residue_types=20)

    assert unified.shape == (33, 37)
    torch.testing.assert_close(legacy, unified[1:21])
    assert torch.equal(
        torch.where(unified[19])[0],
        torch.tensor([0, 1, 2, 3, 4, 5, 12, 13, 20, 21, 31, 32]),
    )
    assert torch.all(unified[22:26, 36])
    assert torch.all(unified[27:31, 36])
    assert not unified[26, 36]
    assert not unified[31, 36]
    assert torch.equal(torch.where(unified[32])[0], torch.tensor([1]))


def test_rfproteina_atom37_table_takes_a_vocabulary_from_the_schemas() -> None:
    """A wider vocabulary is a schema the caller resolves, not something the layer loads."""
    from bionemo_ir.data.schemas import residue_vocabulary

    vocabulary = residue_vocabulary(147)
    assert vocabulary is not None and vocabulary.num_residue_types == 147

    table = local_latents_decoder.get_atom37_mask_table(n_residue_types=147, occupancy=vocabulary.atom37_occupancy())
    assert table.shape == (147, 37)
    # The unified vocabulary extends the released one: the shared classes keep
    # their meaning, and the amino acids gain the OXT slot the wider table adds.
    unified = local_latents_decoder.get_atom37_mask_table(n_residue_types=33)
    torch.testing.assert_close(table[:33, :36], unified[:, :36])
    assert torch.equal(torch.where(table[32])[0], torch.tensor([1]))

    with pytest.raises(ValueError, match="bionemo_ir.data.schemas.residue_vocabulary"):
        local_latents_decoder.get_atom37_mask_table(n_residue_types=147)
    with pytest.raises(ValueError, match="describes 147 residue types, expected 33"):
        local_latents_decoder.get_atom37_mask_table(n_residue_types=33, occupancy=vocabulary.atom37_occupancy())


def test_rfproteina_unified_decoder_restricts_residue_classes_by_token_type() -> None:
    config = RFProteinaDecoderConfig(
        dtype="float32", num_layers=1, token_dim=32, pair_dim=16, condition_dim=8, num_heads=4, num_residue_types=33
    )
    module = RFProteinaLatentDecoder(config).cuda().eval()
    init_module_weights(module)
    backbone_ca = torch.randn(1, 5, 3, device="cuda")
    latents = torch.randn(1, 5, config.latent_dim, device="cuda")
    mask = torch.ones(1, 5, dtype=torch.bool, device="cuda")
    token_type = torch.tensor([[0, 1, 2, 3, 4]], device="cuda")

    with torch.inference_mode():
        output = module(backbone_ca, latents, mask, token_type=token_type)

    allowed = local_latents_decoder.get_allowed_residue_type_mask(device=torch.device("cuda"), n_residue_types=33)
    observed = output["sequence_logits"] != torch.finfo(torch.float32).min
    torch.testing.assert_close(observed, allowed[token_type])
    torch.testing.assert_close(
        output["atom_mask"],
        local_latents_decoder.get_residue_atom_mask(output["residue_type"], 33, mask),
    )
    assert output["residue_type"][0, 3] == 32
    assert torch.all(
        (output["residue_type"][0, :3] >= torch.tensor([1, 22, 27], device="cuda"))
        & (output["residue_type"][0, :3] < torch.tensor([22, 27, 32], device="cuda"))
    )

    invalid_token_type = token_type.clone()
    invalid_token_type[0, 3] = 5
    with pytest.raises(ValueError, match="invalid values \\[5\\]"):
        module(backbone_ca, latents, mask, token_type=invalid_token_type)


class _FixedHead(nn.Module):
    """An output head that returns chosen values, whatever the features are."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        return self.value.to(single.dtype)


#: Unified residue classes and atom37 slots the native-rule tests refer to.
_ALA, _GLY, _TYR = 1, 8, 19
_N, _CA, _C, _CB, _O = 0, 1, 2, 3, 4


def _native_rule_decoder(with_presence_head: bool) -> RFProteinaLatentDecoder:
    """A 33-class decoder whose heads say TYR, ALA, ALA, padding, and which atoms of ALA exist."""
    config = RFProteinaDecoderConfig(
        dtype="float32",
        num_layers=1,
        token_dim=32,
        pair_dim=16,
        condition_dim=8,
        num_heads=4,
        num_residue_types=33,
        use_atom_presence=with_presence_head,
    )
    module = RFProteinaLatentDecoder(config).cuda().eval()
    init_module_weights(module)
    logits = torch.full((1, 4, 33), -5.0, device="cuda")
    logits[0, 0, _TYR] = logits[0, 1, _ALA] = logits[0, 2, _ALA] = 5.0
    module.logit_linear = _FixedHead(logits)
    if with_presence_head:
        presence = torch.full((1, 4, 37), -1.0, device="cuda")
        presence[0, 1, [_N, _CA, _C, _CB]] = 1.0
        module.atom_presence_linear = _FixedHead(presence)
    return module


def _native_rule_inputs() -> dict[str, torch.Tensor]:
    """Token 0 is conditioned on GLY against a TYR prediction, token 1 is free, token 2 agrees on ALA."""
    required = torch.zeros(1, 4, 37, dtype=torch.bool, device="cuda")
    required[0, 0, [_N, _CA, _C, _CB, _O]] = True  # CB is not an atom of GLY.
    required[0, 2, _O] = True
    return {
        "backbone_ca": torch.randn(1, 4, 3, device="cuda"),
        "local_latents": torch.randn(1, 4, 12, device="cuda"),
        "mask": torch.tensor([[True, True, True, False]], device="cuda"),
        "token_type": torch.zeros(1, 4, dtype=torch.long, device="cuda"),
        "conditional_residue_type": torch.tensor([[_GLY, 0, _ALA, 0]], device="cuda"),
        "conditional_residue_mask": torch.tensor([[True, False, True, False]], device="cuda"),
        "required_atom_mask": required,
    }


def _atoms(*slots: int) -> torch.Tensor:
    atoms = torch.zeros(37, dtype=torch.bool, device="cuda")
    atoms[list(slots)] = True
    return atoms


def test_rfproteina_presence_decoder_follows_the_native_output_rules() -> None:
    unified = local_latents_decoder.get_atom37_mask_table(torch.device("cuda"), 33)
    assert torch.equal(unified[_GLY], _atoms(_N, _CA, _C, _O))
    assert torch.equal(unified[_ALA], _atoms(_N, _CA, _C, _CB, _O))
    module = _native_rule_decoder(with_presence_head=True)
    inputs = _native_rule_inputs()

    with torch.inference_mode():
        output = module(**inputs)
        without_required = module(**{**inputs, "required_atom_mask": None})
        free = module(
            **{**inputs, "conditional_residue_type": None, "conditional_residue_mask": None, "required_atom_mask": None}
        )

    # A conditioned token keeps its residue and a free one keeps the decoder's.
    assert output["residue_type"].tolist() == [[_GLY, _ALA, _ALA, 0]]
    assert free["residue_type"].tolist() == [[_TYR, _ALA, _ALA, 0]]
    # Conditioned GLY: its legal atoms, which the head denies, are all required but CB, which GLY lacks.
    assert torch.equal(output["atom_mask"][0, 0], _atoms(_N, _CA, _C, _O))
    # Free ALA: the head's atoms, so no O although ALA could carry one.
    assert torch.equal(output["atom_mask"][0, 1], _atoms(_N, _CA, _C, _CB))
    # Conditioned ALA: the reference atom and the required O.
    assert torch.equal(output["atom_mask"][0, 2], _atoms(_CA, _O))
    assert not output["atom_mask"][0, 3].any()
    # Without the requirement only the reference atom survives the head, and the predicted
    # coordinates of a required atom are the ones the decoder gave all along.
    assert torch.equal(without_required["atom_mask"][0, 0], _atoms(_CA))
    assert torch.equal(without_required["atom_mask"][0, 2], _atoms(_CA))
    torch.testing.assert_close(output["coords"], without_required["coords"], atol=0, rtol=0)
    torch.testing.assert_close(output["sequence_logits"], without_required["sequence_logits"], atol=0, rtol=0)


def test_rfproteina_decoder_without_presence_head_keeps_what_it_predicts() -> None:
    module = _native_rule_decoder(with_presence_head=False)
    inputs = _native_rule_inputs()

    with torch.inference_mode():
        output = module(**inputs)

    # The first-release 33-class autoencoder has no presence head, and the native decode it
    # matches has neither rule: the argmax and its full atom table stand.
    assert output["residue_type"].tolist() == [[_TYR, _ALA, _ALA, 0]]
    torch.testing.assert_close(
        output["atom_mask"],
        local_latents_decoder.get_residue_atom_mask(output["residue_type"], 33, inputs["mask"]),
    )


def test_rfproteina_decoder_rejects_a_misshapen_required_atom_mask() -> None:
    module = _native_rule_decoder(with_presence_head=True)
    inputs = _native_rule_inputs()
    inputs["required_atom_mask"] = inputs["required_atom_mask"][..., :36]

    with pytest.raises(ValueError, match="required_atom_mask must have shape"):
        module(**inputs)


def test_rfproteina_decoder_pair_features_are_head_count_safe() -> None:
    """RFProteína's 12 heads are not a power of two.

    ``moveaxis_pad`` rounds its head block up to the next power of two, so a
    missing head mask writes past the pair-bias buffer. That only faults at
    larger N, hence the two sizes.
    """
    config = RFProteinaDecoderConfig(
        input_schema="legacy",
        latent_dim=8,
        num_residue_types=20,
        sequence_separation_bins=127,
        use_pair_bias=True,
        precompute_bias=True,
        dtype="float32",
        num_layers=1,
        num_heads=12,
    )
    module = RFProteinaLatentDecoder(config).cuda().eval()
    init_module_weights(module)

    for num_tokens in (16, 128):
        mask = torch.ones(1, num_tokens, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            output = module(
                torch.randn(1, num_tokens, 3, device="cuda"),
                torch.randn(1, num_tokens, config.latent_dim, device="cuda"),
                mask,
            )
        torch.cuda.synchronize()
        assert torch.isfinite(output["coords"]).all()


def test_rfproteina_decoder_precomputed_bias_matches_per_layer() -> None:
    """The mega-GEMM bias path must agree with projecting inside each layer.

    Valid only because the decoder never updates its pair representation, so
    every layer projects the same tensor.

    Asserted on the biases rather than end to end: a randomly initialized
    decoder saturates its attention logits, so it amplifies the residual
    float32 difference by seven orders of magnitude and says nothing useful.
    End-to-end agreement is covered by the checkpoint-backed comparison
    against the OSS reference.
    """
    config = RFProteinaDecoderConfig(
        input_schema="legacy",
        latent_dim=8,
        num_residue_types=20,
        sequence_separation_bins=127,
        use_pair_bias=True,
        precompute_bias=True,
        dtype="float32",
        num_layers=3,
        token_dim=96,
        pair_dim=32,
        condition_dim=16,
        num_heads=6,
    )
    decoder = RFProteinaLatentDecoder(config).cuda().eval()
    init_module_weights(decoder)

    backbone_ca = torch.randn(1, 24, 3, device="cuda")
    latents = torch.randn(1, 24, 8, device="cuda")
    mask = torch.ones(1, 24, dtype=torch.bool, device="cuda")

    with torch.inference_mode():
        _, pair = decoder.build_representations(backbone_ca, latents, mask)
        precomputed = decoder._precompute_all_biases(pair)
        for layer, got in zip(decoder.transformer_layers, precomputed, strict=True):
            norm, proj = layer.mha.proj_z[0], layer.mha.proj_z[1]
            want = proj(norm(pair)).movedim(-1, -3)
            torch.testing.assert_close(got, want, atol=1e-6 * want.abs().max(), rtol=0)

    assert len(precomputed) == len(decoder.transformer_layers)


@pytest.mark.parametrize("fused", [True, False])
def test_rfproteina_decoder_pair_gather_matches_one_hot(monkeypatch: pytest.MonkeyPatch, fused: bool) -> None:
    """Gathering the pair projection must equal projecting the one-hots.

    The decoder's two pair features are one-hot, so it looks their projection
    up in a table instead of building ``[B, N, N, bins]`` one-hots, their
    concatenation and a masked copy. Bias-free projection of a one-hot is a
    weight column, so this is exact, not approximate -- hence ``torch.equal``.

    ``fused=False`` forces the two-gather fallback taken when the combined
    table would be too large.
    """
    if not fused:
        monkeypatch.setattr(local_latents_decoder, "_MAX_FUSED_TABLE_BYTES", 0)

    config = RFProteinaDecoderConfig(
        input_schema="legacy",
        latent_dim=8,
        num_residue_types=20,
        sequence_separation_bins=127,
        use_pair_bias=True,
        precompute_bias=True,
        dtype="float32",
        num_layers=2,
        token_dim=96,
        pair_dim=64,
        condition_dim=16,
        num_heads=8,
    )
    decoder = RFProteinaLatentDecoder(config).cuda().eval()
    torch.nn.init.normal_(decoder.pair_rep_factory.linear_out.weight, std=0.3)

    num_tokens = 40
    backbone_ca = torch.randn(1, num_tokens, 3, device="cuda")
    mask = torch.ones(1, num_tokens, dtype=torch.bool, device="cuda")
    mask[0, -7:] = False
    token_index = torch.arange(1, num_tokens + 1, device="cuda").unsqueeze(0)
    separation_bins = config.sequence_separation_bins
    distance_bins = config.pair_distance_bins

    with torch.inference_mode():
        features = torch.cat(
            [
                F.one_hot(
                    local_latents_decoder.sequence_separation_bin_indices(token_index, separation_bins), separation_bins
                ),
                F.one_hot(
                    local_latents_decoder.pair_distance_bin_indices(
                        backbone_ca, config.pair_distance_min, config.pair_distance_max, distance_bins
                    ),
                    distance_bins,
                ),
            ],
            dim=-1,
        ).float()
        valid = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).float()
        want = decoder.pair_rep_factory.linear_out(features * valid) * valid
        got = decoder._build_pair_representation(backbone_ca, mask, token_index)

    assert torch.equal(got, want)
    assert not got[0, -7:].any(), "masked rows must project to zero"


def _small_encoder_config(**updates) -> RFProteinaEncoderConfig:
    values = {
        "token_dim": 32,
        "pair_dim": 16,
        "condition_dim": 8,
        "latent_dim": 4,
        "num_layers": 2,
        "num_heads": 4,
        "input_schema": "legacy",
        "num_residue_types": 20,
        "sequence_separation_bins": 127,
        "use_pair_bias": True,
        "dtype": "float32",
        "transformer_dtype": "float32",
        "pairwise_attention_backend": "SDPA",
        "precompute_bias": True,
    }
    values.update(updates)
    return RFProteinaEncoderConfig(**values)


def test_rfproteina_rewrites_fused_norms_in_velocity_decoder_and_encoder() -> None:
    decoder = RFProteinaDecoderConfig(
        token_dim=32,
        pair_dim=16,
        condition_dim=8,
        latent_dim=4,
        num_layers=2,
        num_heads=4,
        dtype="float32",
        transformer_dtype="float32",
        pairwise_attention_backend="SDPA",
    )
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_config(),
            decoder=decoder,
            encoder=_small_encoder_config(),
        )
    )

    for module in (model.velocity_model, model.decoder, model.encoder):
        assert module is not None
        assert any(isinstance(child, FusedLayerNorm) for child in module.modules())
        assert not any(type(child) is nn.LayerNorm for child in module.modules())
        assert not any(type(child) is nn.RMSNorm for child in module.modules())
    assert any(isinstance(child, FusedRMSNorm) for child in model.velocity_model.modules())
    assert any(isinstance(child, FusedRMSNorm) for child in model.representation_builder.modules())


def test_rfproteina_unified_encoder_omits_pair_bias_by_default() -> None:
    config = RFProteinaEncoderConfig()
    assert not config.use_pair_bias
    assert not config.precompute_bias


def test_rfproteina_encoder_pair_projection_matches_dense_features() -> None:
    config = _small_encoder_config(num_layers=1)
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    init_module_weights(encoder)
    residue_type = torch.tensor([[0, 1, 7, 18, 3, 4]], device="cuda")
    mask = torch.tensor([[True, True, True, True, True, False]], device="cuda")
    atom_mask = local_latents_decoder.get_residue_atom_mask(residue_type, 20, mask)
    coordinates = torch.randn(1, 6, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    token_index = torch.arange(1, 7, device="cuda").unsqueeze(0)

    with torch.inference_mode():
        _, pair, encoding_mask = encoder.build_representations(
            coordinates,
            atom_mask,
            residue_type,
            mask,
            token_index=token_index,
        )
        separation, distances, orientations = local_latents_encoder._pair_feature_indices(
            coordinates,
            atom_mask,
            token_index,
            config.sequence_separation_bins,
            config.distance_bins,
            config.orientation_bins,
        )
        pair_valid = encoding_mask.unsqueeze(-1) & encoding_mask.unsqueeze(-2)
        ca_valid = atom_mask[..., 1] & encoding_mask
        coordinate_valid = ca_valid.unsqueeze(-1) & ca_valid.unsqueeze(-2)
        orientation_valid = coordinate_valid & atom_mask[..., 3].unsqueeze(-1)
        features = torch.cat(
            [
                F.one_hot(separation, config.sequence_separation_bins) * pair_valid.unsqueeze(-1),
                *[F.one_hot(value, config.distance_bins) * coordinate_valid.unsqueeze(-1) for value in distances],
                *[
                    F.one_hot(value, config.orientation_bins) * orientation_valid.unsqueeze(-1)
                    for value in orientations
                ],
            ],
            dim=-1,
        ).float()
        expected = encoder.pair_rep_factory.linear_out(features)
        expected *= pair_valid.unsqueeze(-1)

    torch.testing.assert_close(pair, expected, atol=2e-7, rtol=2e-6)
    assert not pair[:, -1].any()
    # Gly has no CB, and the released orientation feature uses its historical
    # directional source-CB mask.
    assert not features[0, 2, :, -5 * config.orientation_bins :].any()


def test_rfproteina_encoder_masks_unsupported_rows_and_owns_sampling_rng() -> None:
    config = _small_encoder_config()
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    init_module_weights(encoder)
    residue_type = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device="cuda")
    padding_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]],
        device="cuda",
    )
    can_be_encoded = torch.tensor(
        [[True, True, False, True, False], [False, True, True, False, True]],
        device="cuda",
    )
    atom_mask = local_latents_decoder.get_residue_atom_mask(residue_type, 20, padding_mask)
    coordinates = torch.randn(2, 5, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)

    outputs = []
    with torch.inference_mode():
        for seed in (17, 17, 18):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            outputs.append(
                encoder(
                    coordinates,
                    atom_mask,
                    residue_type,
                    padding_mask,
                    can_be_encoded=can_be_encoded,
                    generator=generator,
                )
            )
        deterministic = encoder(
            coordinates, atom_mask, residue_type, padding_mask, can_be_encoded=can_be_encoded, sample=False
        )

    torch.testing.assert_close(outputs[0]["z_latent"], outputs[1]["z_latent"])
    assert not torch.allclose(outputs[0]["z_latent"][:, :2], outputs[2]["z_latent"][:, :2])
    torch.testing.assert_close(deterministic["z_latent"], deterministic["mean"])
    assert not outputs[0]["encoding_mask"][0, 2]
    assert not outputs[0]["z_latent"][0, [2, 4]].any()


def test_rfproteina_encoder_left_aligns_interior_mask_for_cutedsl() -> None:
    skip_if_cutedsl("CuTeDSL", "pairwise_attention")
    values = {
        "token_dim": 256,
        "pair_dim": 16,
        "condition_dim": 8,
        "latent_dim": 4,
        "num_layers": 1,
        "num_heads": 4,
        "input_schema": "legacy",
        "num_residue_types": 20,
        "sequence_separation_bins": 127,
        "use_pair_bias": True,
        "dtype": "float32",
        "transformer_dtype": "bfloat16",
        "precompute_bias": True,
    }
    reference = (
        RFProteinaLatentEncoder(RFProteinaEncoderConfig(**values, pairwise_attention_backend="SDPA")).cuda().eval()
    )
    init_module_weights(reference)
    actual = (
        RFProteinaLatentEncoder(RFProteinaEncoderConfig(**values, pairwise_attention_backend="CuTeDSL")).cuda().eval()
    )
    actual.load_state_dict(reference.state_dict())

    residue_type = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device="cuda")
    padding_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]],
        device="cuda",
    )
    can_be_encoded = torch.tensor(
        [[True, True, False, True, False], [False, True, True, False, True]],
        device="cuda",
    )
    atom_mask = local_latents_decoder.get_residue_atom_mask(residue_type, 20, padding_mask)
    torch.manual_seed(7)
    coordinates = torch.randn(2, 5, 37, 3, device="cuda") * 0.2 * atom_mask.unsqueeze(-1)

    with torch.inference_mode():
        expected = reference(
            coordinates, atom_mask, residue_type, padding_mask, can_be_encoded=can_be_encoded, sample=False
        )
        got = actual(coordinates, atom_mask, residue_type, padding_mask, can_be_encoded=can_be_encoded, sample=False)

    torch.testing.assert_close(got["encoding_mask"], can_be_encoded)
    for key in ("mean", "log_scale", "z_latent"):
        torch.testing.assert_close(got[key][can_be_encoded], expected[key][can_be_encoded], atol=1e-3, rtol=2e-3)
        assert not got[key][~can_be_encoded].any()


def _chemistry_encoder_config(**updates) -> RFProteinaEncoderConfig:
    """A small unified encoder that reads per-atom chemistry, as the 147-class autoencoder's does."""
    values = {
        "input_schema": "unified",
        "use_pair_bias": False,
        "precompute_bias": False,
        "num_residue_types": 33,
        "num_token_kinds": 7,
        "atom_chemistry": True,
        "num_elements": 9,
        "element_embedding_dim": 4,
        "num_atom_charges": 11,
        "atom_charge_embedding_dim": 6,
    }
    values.update(updates)
    return _small_encoder_config(**values)


def _released_chemistry_tensors(config: RFProteinaEncoderConfig, seed: int) -> dict[str, torch.Tensor]:
    """The released encoder's tensors that the chemistry fold reads, in the released layout."""
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return 0.2 * torch.randn(*shape, generator=generator)

    return {
        "projection": draw(config.token_dim, config.single_input_dim),
        "element": draw(config.num_elements, config.element_embedding_dim),
        "charge": draw(config.num_atom_charges, config.atom_charge_embedding_dim),
        "token_type": draw(config.num_token_kinds, config.token_type_embedding_dim),
    }


def _released_layout(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The released checkpoint's names for the tensors in ``source``."""
    prefix = "encoder.init_repr_factory"
    return {
        f"{prefix}.linear_out.weight": source["projection"],
        f"{prefix}.feat_creators.0.embedding.weight": source["token_type"],
        f"{prefix}.feat_creators.5.element_embedding.weight": source["element"],
        f"{prefix}.feat_creators.5.charge_embedding.weight": source["charge"],
    }


def _convert_inputs(config: RFProteinaEncoderConfig, source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    _convert_encoder_inputs(converted, _released_layout(source), "encoder", config, config.torch_dtype)
    return converted


def _load_folded(encoder: RFProteinaLatentEncoder, source: dict[str, torch.Tensor]) -> None:
    """Load the released tensors through the converter, as a checkpoint load does."""
    result = encoder.load_state_dict(_convert_inputs(encoder.config, source), strict=False)
    assert not result.unexpected_keys


def _unfused_single(
    config: RFProteinaEncoderConfig, source: dict[str, torch.Tensor], inputs: dict, encoding_mask: torch.Tensor
) -> torch.Tensor:
    """The released single projection as it was built before the chemistry was folded.

    Embeddings are concatenated per slot into a wide feature vector, and one matmul projects all of it.
    """
    coordinates, atom_mask = inputs["coordinates"], inputs["atom_mask"]
    occupied = atom_mask.bool()
    coordinate_mask = atom_mask.to(coordinates.dtype)
    masked = coordinates * coordinate_mask.unsqueeze(-1)
    relative = (masked - masked[:, :, 1].unsqueeze(-2)) * coordinate_mask.unsqueeze(-1)
    elements = inputs["atom37_element"].long().masked_fill(~occupied, 0)
    charges = (inputs["atom37_charge"].long() + config.atom_charge_offset).clamp(0, config.num_atom_charges - 1)
    charges = charges.masked_fill(~occupied, 0)
    chemistry = torch.cat((F.embedding(elements, source["element"]), F.embedding(charges, source["charge"])), dim=-1)
    features = torch.cat(
        (
            F.embedding(inputs["token_type"], source["token_type"]),
            torch.cat((masked.flatten(-2), coordinate_mask), dim=-1),
            torch.cat((relative.flatten(-2), coordinate_mask), dim=-1),
            inputs["chain_iid"].unsqueeze(-1),
            (inputs["incident_bond_type_counts"] * occupied.unsqueeze(-1)).flatten(-2),
            (chemistry * occupied.unsqueeze(-1)).flatten(-2),
        ),
        dim=-1,
    )
    valid = encoding_mask.unsqueeze(-1).to(features.dtype)
    return ((features * valid) @ source["projection"].t()) * valid


def _chemistry_inputs(config: RFProteinaEncoderConfig, seed: int, device: str = "cuda") -> dict:
    generator = torch.Generator().manual_seed(seed)
    batch, tokens = 2, 7
    atom_mask = torch.rand(batch, tokens, 37, generator=generator) < 0.6
    atom_mask[..., 1] = True
    # Elements and charges span their tables, including the ends, and charges overshoot the clamp.
    element = torch.randint(0, config.num_elements, (batch, tokens, 37), generator=generator)
    element[0, 0, :3] = torch.tensor([0, config.num_elements - 1, 3])
    inputs = {
        "coordinates": torch.randn(batch, tokens, 37, 3, generator=generator) * atom_mask.unsqueeze(-1),
        "atom_mask": atom_mask,
        "residue_type": torch.randint(0, 20, (batch, tokens), generator=generator),
        "token_type": torch.randint(0, config.num_token_kinds, (batch, tokens), generator=generator),
        "chain_iid": torch.randint(0, 3, (batch, tokens), generator=generator).float(),
        "atom37_element": element,
        "atom37_charge": torch.randint(-6, 9, (batch, tokens, 37), generator=generator),
        "incident_bond_type_counts": torch.randint(
            0, 4, (batch, tokens, 37, config.incident_bond_type_bins), generator=generator
        ).float(),
    }
    return {name: value.to(device) for name, value in inputs.items()}


def _encode_single(encoder: RFProteinaLatentEncoder, inputs: dict, mask: torch.Tensor, can_be_encoded: torch.Tensor):
    with torch.inference_mode():
        return encoder.build_representations(
            inputs["coordinates"],
            inputs["atom_mask"],
            inputs["residue_type"],
            mask,
            can_be_encoded=can_be_encoded,
            token_type=inputs["token_type"],
            chain_iid=inputs["chain_iid"],
            atom37_element=inputs["atom37_element"],
            atom37_charge=inputs["atom37_charge"],
            incident_bond_type_counts=inputs["incident_bond_type_counts"],
        )


def _encoding_masks() -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor([[True] * 6 + [False], [True] * 7], device="cuda")
    can_be_encoded = torch.tensor(
        [[True, True, False, True, True, True, True], [True] * 4 + [False] + [True] * 2], device="cuda"
    )
    return mask, can_be_encoded


@pytest.mark.parametrize("dtype, tolerance", [("float32", 2e-5), ("bfloat16", 5e-2)])
def test_encoder_folded_chemistry_matches_the_unfused_projection(dtype: str, tolerance: float) -> None:
    config = _chemistry_encoder_config(dtype=dtype)
    source = _released_chemistry_tensors(config, seed=1)
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    _load_folded(encoder, source)
    inputs = _chemistry_inputs(config, seed=2)
    mask, can_be_encoded = _encoding_masks()

    single, _pair, encoding_mask = _encode_single(encoder, inputs, mask, can_be_encoded)

    assert not encoding_mask[0, 2] and not encoding_mask[1, 4] and not encoding_mask[0, 6], "the fixture masks tokens"
    reference = _unfused_single(config, {k: v.cuda() for k, v in source.items()}, inputs, encoding_mask)
    scale = reference.abs().max().item()
    torch.testing.assert_close(single.float(), reference, atol=tolerance * scale, rtol=tolerance)
    assert not single[~encoding_mask].any(), "tokens outside the encoding mask stay zero"


def test_encoder_chemistry_lookup_is_one_row_per_occupied_slot() -> None:
    """With every other feature zero, an atom's contribution is exactly its element and charge rows."""
    config = _chemistry_encoder_config()
    source = _released_chemistry_tensors(config, seed=3)
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    _load_folded(encoder, source)
    atom_mask = torch.zeros(1, 2, 37, dtype=torch.bool, device="cuda")
    atom_mask[0, 0, 5] = atom_mask[0, 0, 20] = True
    element = torch.zeros(1, 2, 37, dtype=torch.long, device="cuda")
    element[0, 0, 5], element[0, 0, 20] = 3, 7
    charge = torch.zeros(1, 2, 37, dtype=torch.long, device="cuda")
    charge[0, 0, 5] = 2  # formal charge +2 is class 6
    token_valid = torch.ones(1, 2, dtype=torch.bool, device="cuda")

    with torch.inference_mode():
        _counts, chemistry = encoder._atom_chemistry_features(
            atom_mask, element, charge, None, (1, 2), "cuda", token_valid
        )

    columns = source["projection"][:, encoder.config.dense_input_dim :].cuda().reshape(config.token_dim, 37, -1)
    split = config.element_embedding_dim

    def through(slot: int, element_id: int, charge_class: int) -> torch.Tensor:
        return (
            columns[:, slot, :split] @ source["element"][element_id].cuda()
            + columns[:, slot, split:] @ source["charge"][charge_class].cuda()
        )

    charge_zero = config.atom_charge_offset  # formal charge 0
    expected = through(5, 3, charge_zero + 2) + through(20, 7, charge_zero)
    torch.testing.assert_close(chemistry[0, 0], expected, atol=1e-5, rtol=1e-5)
    assert not chemistry[0, 1].any(), "a token with no occupied slot has nothing to look up"


def test_chemistry_fold_is_each_slots_columns_times_each_embedding_row() -> None:
    config = _chemistry_encoder_config()
    source = _released_chemistry_tensors(config, seed=4)
    dense = config.dense_input_dim
    element, charge = fold_atom_chemistry(source["projection"][:, dense:], source["element"], source["charge"])

    assert element.dtype == charge.dtype == torch.float32
    assert element.shape == (37 * config.num_elements, config.token_dim)
    assert charge.shape == (37 * config.num_atom_charges, config.token_dim)
    columns = source["projection"][:, dense:].reshape(config.token_dim, 37, -1)
    split = config.element_embedding_dim
    for slot in (0, 1, 17, 36):
        for k in range(config.num_elements):
            torch.testing.assert_close(
                element[slot * config.num_elements + k], columns[:, slot, :split] @ source["element"][k]
            )
        for k in range(config.num_atom_charges):
            torch.testing.assert_close(
                charge[slot * config.num_atom_charges + k], columns[:, slot, split:] @ source["charge"][k]
            )

    with pytest.raises(ValueError, match="columns wide"):
        fold_atom_chemistry(source["projection"][:, dense + 1 :], source["element"], source["charge"])


def test_encoder_conversion_replaces_the_chemistry_columns_with_their_tables() -> None:
    config = _chemistry_encoder_config()
    source = _released_chemistry_tensors(config, seed=7)
    consumed = set()

    class Tracking(dict):
        def __getitem__(self, key):
            consumed.add(key)
            return super().__getitem__(key)

    converted: dict[str, torch.Tensor] = {}
    _convert_encoder_inputs(converted, Tracking(_released_layout(source)), "encoder", config, torch.float32)

    assert consumed == set(_released_layout(source)), "every released tensor the fold replaces is read"
    assert set(converted) == {
        "init_repr_factory.linear_out.weight",
        "token_type_embedding.weight",
        "element_embedding.weight",
        "atom_charge_embedding.weight",
    }
    dense = config.dense_input_dim
    torch.testing.assert_close(converted["init_repr_factory.linear_out.weight"], source["projection"][:, :dense])
    torch.testing.assert_close(converted["token_type_embedding.weight"], source["token_type"])
    element, charge = fold_atom_chemistry(source["projection"][:, dense:], source["element"], source["charge"])
    torch.testing.assert_close(converted["element_embedding.weight"], element)
    torch.testing.assert_close(converted["atom_charge_embedding.weight"], charge)
    # The converted layout is the module's: loading it needs no reshaping.
    state = RFProteinaLatentEncoder(config).state_dict()
    assert {name: tuple(value.shape) for name, value in converted.items()} == {
        name: tuple(state[name].shape) for name in converted
    }

    # Without chemistry the projection is copied whole and no table appears.
    plain = _small_encoder_config(
        input_schema="unified", use_pair_bias=False, precompute_bias=False, num_residue_types=33
    )
    plain_source = {**source, "projection": torch.randn(plain.token_dim, plain.single_input_dim)}
    plain_converted = _convert_inputs(plain, plain_source)
    assert set(plain_converted) == {"init_repr_factory.linear_out.weight", "token_type_embedding.weight"}
    torch.testing.assert_close(plain_converted["init_repr_factory.linear_out.weight"], plain_source["projection"])


def test_chemistry_encoder_projects_only_its_dense_columns() -> None:
    config = _chemistry_encoder_config()
    assert config.chemistry_input_dim == 37 * (4 + 6)
    assert config.dense_input_dim + config.chemistry_input_dim == config.single_input_dim

    state = RFProteinaLatentEncoder(config).state_dict()
    assert state["init_repr_factory.linear_out.weight"].shape == (config.token_dim, config.dense_input_dim)
    assert state["element_embedding.weight"].shape == (37 * config.num_elements, config.token_dim)
    assert state["atom_charge_embedding.weight"].shape == (37 * config.num_atom_charges, config.token_dim)

    plain = _small_encoder_config(
        input_schema="unified", use_pair_bias=False, precompute_bias=False, num_residue_types=33
    )
    assert plain.chemistry_input_dim == 0 and plain.dense_input_dim == plain.single_input_dim
    assert not {"element_embedding.weight", "atom_charge_embedding.weight"} & set(
        RFProteinaLatentEncoder(plain).state_dict()
    )


@pytest.mark.parametrize("chemistry", [True, False], ids=["with_chemistry", "without_chemistry"])
def test_unified_encoder_encodes_with_or_without_chemistry(chemistry: bool) -> None:
    config = _chemistry_encoder_config(atom_chemistry=chemistry)
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    init_module_weights(encoder)
    inputs = _chemistry_inputs(config, seed=8)
    mask, can_be_encoded = _encoding_masks()
    extra = {"atom37_element": inputs["atom37_element"], "atom37_charge": inputs["atom37_charge"]} if chemistry else {}

    with torch.inference_mode():
        output = encoder(
            inputs["coordinates"],
            inputs["atom_mask"],
            inputs["residue_type"],
            mask,
            can_be_encoded=can_be_encoded,
            token_type=inputs["token_type"],
            chain_iid=inputs["chain_iid"],
            sample=False,
            **extra,
        )

    assert torch.isfinite(output["mean"]).all()
    assert not output["z_latent"][~output["encoding_mask"]].any()
    assert output["z_latent"][output["encoding_mask"]].abs().sum() > 0


def test_encoder_rejects_an_element_that_would_read_the_next_slots_table() -> None:
    config = _chemistry_encoder_config()
    encoder = RFProteinaLatentEncoder(config).cuda().eval()
    inputs = _chemistry_inputs(config, seed=5)
    mask = torch.ones(2, 7, dtype=torch.bool, device="cuda")
    inputs["atom37_element"][0, 1, 1] = config.num_elements  # slot 1 is occupied in the fixture
    with pytest.raises(ValueError, match="atom37_element"):
        _encode_single(encoder, inputs, mask, mask)

    # The same value on an unoccupied slot is never read.
    inputs["atom37_element"][0, 1, 1] = 0
    free = (~inputs["atom_mask"]).nonzero()[0]
    inputs["atom37_element"][free[0], free[1], free[2]] = 10**6
    _encode_single(encoder, inputs, mask, mask)


def test_rfproteina_encoder_conversion_folds_the_released_chemistry_exactly() -> None:
    checkpoint = resolve_rfproteina_autoencoder_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("RFPROTEINA_AE_CKPT is not set to the released autoencoder checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    config = rfproteina_encoder_config_from_weights(state)
    if not config.atom_chemistry:
        pytest.skip("the released autoencoder does not read per-atom chemistry")
    encoder = RFProteinaLatentEncoder(config)
    report = load_rfproteina_encoder_checkpoint(encoder, state)
    assert not report.missing_target_keys
    assert not report.ignored_source_keys, "the fold reads every released tensor it replaces"

    prefix = "encoder.init_repr_factory"
    source = {
        "projection": state[f"{prefix}.linear_out.weight"].float(),
        "element": state[f"{prefix}.feat_creators.5.element_embedding.weight"].float(),
        "charge": state[f"{prefix}.feat_creators.5.charge_embedding.weight"].float(),
        "token_type": state[f"{prefix}.feat_creators.0.embedding.weight"].float(),
    }
    dense = config.dense_input_dim
    converted = encoder.state_dict()
    assert len(report.loaded_keys) == len(converted)
    # Every converted tensor against its source: the dense columns are copied, the lookups are products.
    torch.testing.assert_close(converted["init_repr_factory.linear_out.weight"], source["projection"][:, :dense])
    torch.testing.assert_close(converted["token_type_embedding.weight"], source["token_type"])
    columns = source["projection"][:, dense:].reshape(config.token_dim, 37, -1)
    split = config.element_embedding_dim
    for slot in (0, 1, 36):
        for k in (0, 6, config.num_elements - 1):
            torch.testing.assert_close(
                converted["element_embedding.weight"][slot * config.num_elements + k],
                columns[:, slot, :split] @ source["element"][k],
                atol=1e-5,
                rtol=1e-4,
            )
        for k in (0, 4, config.num_atom_charges - 1):
            torch.testing.assert_close(
                converted["atom_charge_embedding.weight"][slot * config.num_atom_charges + k],
                columns[:, slot, split:] @ source["charge"][k],
                atol=1e-5,
                rtol=1e-4,
            )

    encoder = encoder.cuda().eval()
    inputs = _chemistry_inputs(config, seed=6)
    inputs["token_type"] = inputs["token_type"] % 3  # kinds the released table has, whatever the fixture drew
    mask = torch.ones(2, 7, dtype=torch.bool, device="cuda")
    single, _pair, encoding_mask = _encode_single(encoder, inputs, mask, mask)
    reference = _unfused_single(config, {k: v.cuda() for k, v in source.items()}, inputs, encoding_mask)
    torch.testing.assert_close(single, reference, atol=2e-5 * reference.abs().max().item(), rtol=2e-5)


@pytest.mark.parametrize("transformer_dtype", ["float32", "bfloat16"])
def test_rfproteina_autoencoder_encoder_conversion(transformer_dtype: str) -> None:
    checkpoint = resolve_rfproteina_autoencoder_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("RFPROTEINA_AE_CKPT is not set to the released autoencoder checkpoint")

    config = rfproteina_encoder_config_from_checkpoint(checkpoint, transformer_dtype=transformer_dtype)
    assert (config.token_dim, config.pair_dim, config.num_layers, config.single_input_dim, config.pair_input_dim) == (
        768,
        256,
        12,
        1587,
        0,
    )
    assert config.input_schema == "unified"
    assert not config.use_pair_bias
    module = RFProteinaLatentEncoder(config)
    report = load_rfproteina_encoder_checkpoint(module, checkpoint)
    assert not report.missing_target_keys
    assert not report.ignored_source_keys
    assert len(report.loaded_keys) == len(module.state_dict())

    module = module.cuda().eval()
    assert module.transformer_layers[0].mha.in_proj.weight.dtype == getattr(torch, transformer_dtype)
    residue_type = torch.tensor([[1, 2, 8, 19]], device="cuda")
    mask = torch.ones_like(residue_type, dtype=torch.bool)
    atom_mask = local_latents_decoder.get_residue_atom_mask(residue_type, 33, mask)
    coordinates = torch.randn(1, 4, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    with torch.inference_mode():
        output = module(
            coordinates,
            atom_mask,
            residue_type,
            mask,
            token_type=torch.zeros_like(residue_type),
            chain_iid=torch.zeros_like(residue_type),
            sample=False,
        )
    assert torch.isfinite(output["mean"]).all()
    torch.testing.assert_close(output["z_latent"], output["mean"])


def test_rfproteina_encoder_matches_oss_release(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = resolve_rfproteina_autoencoder_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("RFPROTEINA_AE_CKPT is not set to the released autoencoder checkpoint")
    monkeypatch.setenv("CLUSTER", "local")
    try:
        import rfproteina  # noqa: F401
        from rfproteina.partial_autoencoder.encoder import EncoderTransformer
    except ImportError:
        pytest.skip("the optional OSS RFProteína validation package is absent")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    oss = EncoderTransformer(**checkpoint_data["hyper_parameters"]["cfg_ae"]["nn_ae"])
    oss.load_state_dict(
        {
            key.removeprefix("encoder."): value
            for key, value in checkpoint_data["state_dict"].items()
            if key.startswith("encoder.")
        },
        strict=True,
    )
    oss = oss.cuda().eval()

    config = rfproteina_encoder_config_from_checkpoint(checkpoint, precompute_bias=False)
    native = RFProteinaLatentEncoder(config)
    load_rfproteina_encoder_checkpoint(native, checkpoint)
    native = native.cuda().eval()

    residue_type = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 19]], device="cuda")
    mask = torch.ones_like(residue_type, dtype=torch.bool)
    atom_mask = local_latents_decoder.get_residue_atom_mask(residue_type, 33, mask)
    torch.manual_seed(31)
    coordinates_nm = torch.randn(1, 8, 37, 3, device="cuda") * 0.2 * atom_mask.unsqueeze(-1)
    token_index = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]], device="cuda")
    # The released network reads per-atom chemistry; draw it the way the OSS feature factory expects it.
    chemistry = torch.Generator().manual_seed(32)
    element = torch.randint(1, config.num_elements, (1, 8, 37), generator=chemistry).cuda()
    charge = torch.randint(-1, 2, (1, 8, 37), generator=chemistry).cuda()
    bond_counts = torch.randint(0, 4, (1, 8, 37, config.incident_bond_type_bins), generator=chemistry).float().cuda()
    batch = {
        "padding_mask": mask,
        "has_known_index": mask,
        "can_be_encoded": mask,
        "token_type": torch.zeros_like(residue_type),
        "chain_iid": torch.zeros_like(residue_type),
        "residue_type": residue_type,
        "coords": coordinates_nm * 10.0,
        "coords_nm": coordinates_nm,
        "coord_mask": atom_mask,
        "atom37_mask": atom_mask,
        "within_index_group_res_idx": token_index,
        "condition_index-group_1_token": torch.zeros_like(token_index),
        "is_in_context_conditioning": torch.zeros_like(mask),
        "supports_local_latent": mask,
        "ground_truth_element": element,
        "charge": charge,
        "incident_bond_type_counts": bond_counts,
    }
    chemistry_inputs = {
        "atom37_element": element,
        "atom37_charge": charge,
        "incident_bond_type_counts": bond_counts,
    }
    with torch.inference_mode():
        oss_single = oss.init_repr_factory(batch)
        native_single, native_pair, _ = native.build_representations(
            coordinates_nm,
            atom_mask,
            residue_type,
            mask,
            token_type=batch["token_type"],
            token_index=token_index,
            chain_iid=batch["chain_iid"],
            **chemistry_inputs,
        )
        torch.manual_seed(77)
        oss_output = oss(batch)
        torch.manual_seed(77)
        native_output = native(
            coordinates_nm, atom_mask, residue_type, mask, token_index=token_index, **chemistry_inputs
        )

    # The chemistry lookups are folded into products, so the single track differs by float rounding only.
    torch.testing.assert_close(native_single, oss_single, atol=2e-5 * oss_single.abs().max().item(), rtol=2e-5)
    assert native_pair is None
    for key in ("mean", "log_scale", "z_latent"):
        torch.testing.assert_close(native_output[key], oss_output[key], atol=6e-4, rtol=3e-4)


@pytest.mark.parametrize("use_pretrained_config", [False, True])
def test_rfproteina_autoencoder_decoder_conversion(use_pretrained_config: bool) -> None:
    checkpoint = resolve_rfproteina_autoencoder_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("RFPROTEINA_AE_CKPT is not set to the released autoencoder checkpoint")

    if use_pretrained_config:
        pretrained = RFProteina.get_pretrained_config().decoder
        dtype = pretrained.dtype
        transformer_dtype = pretrained.transformer_dtype
        pair_backend = pretrained.pairwise_attention_backend
    else:
        dtype = "float32"
        transformer_dtype = "float32"
        pair_backend = "SDPA"
    config = rfproteina_decoder_config_from_checkpoint(
        checkpoint,
        dtype=dtype,
        transformer_dtype=transformer_dtype,
        pairwise_attention_backend=pair_backend,
    )
    assert (config.token_dim, config.pair_dim, config.num_layers) == (768, 256, 12)
    assert (config.input_schema, config.latent_dim, config.single_input_dim, config.pair_input_dim) == (
        "unified",
        32,
        214,
        0,
    )
    assert not config.use_pair_bias

    module = RFProteinaLatentDecoder(config)
    report = load_rfproteina_decoder_checkpoint(module, checkpoint)
    assert not report.missing_target_keys
    assert not report.ignored_source_keys
    assert len(report.loaded_keys) == len(module.state_dict())

    module = module.cuda().eval()
    backbone_ca = torch.randn(1, 16, 3, device="cuda")
    with torch.inference_mode():
        output = module(
            backbone_ca,
            torch.randn(1, 16, config.latent_dim, device="cuda"),
            torch.ones(1, 16, dtype=torch.bool, device="cuda"),
        )
    assert torch.isfinite(output["coords"]).all()
    torch.testing.assert_close(output["coords"][..., 1, :], backbone_ca.to(output["coords"].dtype))
    if use_pretrained_config:
        assert module.transformer_layers[0].mha.in_proj.weight.dtype == (torch.bfloat16)
        assert module.init_repr_factory.linear_out.weight.dtype == torch.float32
        assert module.logit_linear[1].weight.dtype == torch.float32
        assert output["sequence_logits"].dtype == torch.float32


def test_encoder_encoding_mask_leaves_the_caller_masks_untouched() -> None:
    """The encoder narrows its own encoding mask; the caller's tensors are inputs, not scratch.

    A bool mask is its own ``.to(torch.bool)``, so narrowing one in place writes
    through to whatever the caller passed.
    """
    encoder = RFProteinaLatentEncoder(_small_encoder_config())
    mask = torch.ones(1, 4, dtype=torch.bool)
    can_be_encoded = torch.tensor([[True, False, True, True]])
    token_type = torch.tensor([[0, 0, 1, 0]])

    encoding_mask = encoder._encoding_mask(mask, can_be_encoded, token_type)

    assert mask.all(), "the encoder narrowed the caller's mask in place"
    assert can_be_encoded.tolist() == [[True, False, True, True]]
    assert encoding_mask.tolist() == [[True, False, False, True]]


class _NeverConstructed:
    """Stands for the training metadata a released checkpoint carries.

    Reading weights must not resolve this class, so neither hook may fire.
    """

    constructed = False

    def __init__(self) -> None:
        type(self).constructed = True

    def __setstate__(self, state: object) -> None:
        type(self).constructed = True


def _write_lightning_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Write a checkpoint shaped like the released ones, metadata included."""
    state = {"nn.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "nn.bias": torch.zeros(2)}
    torch.save(
        {
            "state_dict": state,
            "epoch": 169,
            "global_step": 340000,
            # A global weights-only rejects, and a container it cannot SETITEM
            # even once allowlisted -- the two ways the released files fail.
            "hyper_parameters": {"created": datetime.datetime(2026, 9, 4), "cfg": _NeverConstructed()},
            "loops": collections.defaultdict(dict, {"fit_loop": {"epoch": 169}}),
        },
        path,
    )
    return state


def test_released_checkpoint_metadata_does_not_block_weight_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chk-ema.ckpt"
    expected = _write_lightning_checkpoint(checkpoint)

    # What the loader used to do, and why the released files were rejected.
    with pytest.raises(pickle.UnpicklingError):
        torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)

    state = load_rfproteina_checkpoint_state(checkpoint)

    assert set(state) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(state[key], value)


def test_checkpoint_metadata_globals_are_never_resolved(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chk-ema.ckpt"
    _write_lightning_checkpoint(checkpoint)
    _NeverConstructed.constructed = False

    load_rfproteina_checkpoint_state(checkpoint)

    assert not _NeverConstructed.constructed


def test_checkpoint_without_tensors_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "empty-ema.ckpt"
    torch.save({"state_dict": {"epoch": 1}}, checkpoint)

    with pytest.raises(TypeError, match="no tensors"):
        load_rfproteina_checkpoint_state(checkpoint)


def test_in_memory_state_still_passes_through_untouched() -> None:
    state = {"nn.weight": torch.ones(2)}

    assert load_rfproteina_checkpoint_state({"state_dict": state}) is state
    assert load_rfproteina_checkpoint_state(state) is state
