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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.layers.linear import WeightMode
from bionemo_ir._torch.layers.sequence_local_atom import build_local_attn_metadata
from bionemo_ir._torch.modules.protenix import ProtenixAtomAttentionEncoder
from bionemo_ir._torch.modules.protenix._common import atom_encoder_kwargs
from bionemo_ir._torch.modules.protenix.atom_attention import atom_windows
from bionemo_ir.configs import DiffusionTransformerConfig
from bionemo_ir.models.protenix.config import DiffusionAtomAttentionEncoderConfig
from bionemo_ir.models.protenix.convert import convert_diffusion_atom_encoder_torch
from bionemo_ir.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import (
    RefProtenixAtomAttentionEncoderFromOSS,
    update_input_feature_dict,
)


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
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _atom_transformer_config(sc: "Scenario") -> DiffusionTransformerConfig:
    """The Protenix local/atom DiffusionTransformer variant flags."""
    return DiffusionTransformerConfig(
        num_blocks=sc.n_blocks,
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
        pairwise_attention_backend="SDPA",
    )


def _make_features(device: torch.device, n_atom: int, n_token: int) -> dict:
    torch.manual_seed(11)
    atom_to_token_idx = (torch.arange(n_atom, device=device) // (n_atom // n_token)).long()
    element_idx = torch.randint(0, 128, (n_atom,), device=device)
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


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32"),
        Scenario(dtype="bfloat16"),
    ],
    ids=["fp32", "bf16"],
)
def test_protenix_diffusion_atom_encoder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    S = sc.n_sample

    ref = (
        RefProtenixAtomAttentionEncoderFromOSS.build(
            has_coords=True,
            c_token=sc.c_token,
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
            c_s=sc.c_s,
            c_z=sc.c_z,
            n_blocks=sc.n_blocks,
            n_heads=sc.n_heads,
            n_queries=sc.n_queries,
            n_keys=sc.n_keys,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
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
        dtype=sc.dtype,
    )
    model = ProtenixAtomAttentionEncoder(config).to(device).eval()
    converted = convert_diffusion_atom_encoder_torch(config, ref.state_dict(), prefix="")
    assert model.linear_no_bias_ref.weights_loading_config.weight_mode == WeightMode.FUSED_ALL_LINEAR_LAST_DIM
    assert model.linear_no_bias_pair.weights_loading_config.weight_mode == WeightMode.FUSED_ALL_LINEAR_LAST_DIM
    for target, sources in {
        "linear_no_bias_ref.weight": ("ref_pos", "ref_charge", "f"),
        "linear_no_bias_pair.weight": ("d", "invd", "v"),
    }.items():
        expected = torch.cat([ref.state_dict()[f"linear_no_bias_{source}.weight"] for source in sources], dim=-1).to(
            torch_dtype
        )
        torch.testing.assert_close(converted[target], expected)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    feats = _make_features(device, sc.n_atom, sc.n_token)
    r_l = torch.randn(S, sc.n_atom, 3, device=device)
    s = torch.randn(S, sc.n_token, sc.c_s, device=device)
    z = torch.randn(S, sc.n_token, sc.n_token, sc.c_z, device=device)

    def _oss_args():
        return {
            "atom_to_token_idx": feats["atom_to_token_idx"],
            "ref_pos": feats["ref_pos"],
            "ref_charge": feats["ref_charge"],
            "ref_mask": feats["ref_mask"],
            "ref_atom_name_chars": feats["ref_atom_name_chars"],
            "ref_element": feats["ref_element"],
            "d_lm": feats["d_lm"],
            "v_lm": feats["v_lm"],
            "pad_info": feats["pad_info"],
        }

    def _batched(t):
        return t.unsqueeze(0).to(torch_dtype) if t.is_floating_point() else t.unsqueeze(0)

    with torch.inference_mode():
        exp = ref(**_oss_args(), r_l=r_l, s=s, z=z)  # unbatched over N_sample
        act = model(
            atom_to_token_idx=feats["atom_to_token_idx"].unsqueeze(0),
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
            z=z.unsqueeze(0).to(torch_dtype),
        )
    act = tuple(t.squeeze(0) for t in act)

    tol = 3e-3 if torch_dtype == torch.float32 else 6e-2
    for name, a, e in zip(("a", "q_l", "c_l", "p_lm"), act, exp, strict=True):
        r = _rmse_ratio(a, e)
        assert r < tol, f"{name} rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch_size,n_sample", [(1, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize("precompute", [False, True])
def test_cached_atom_biases(dtype: str, batch_size: int, n_sample: int, precompute: bool) -> None:
    torch.manual_seed(42)
    sc = Scenario(dtype=dtype)
    tc = _atom_transformer_config(sc)
    tc.precompute_bias = precompute
    config = DiffusionAtomAttentionEncoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=tc,
        dtype=dtype,
    )
    model = ProtenixAtomAttentionEncoder(config).cuda().eval()
    # Both paths on the attention backend, so that only the bias preparation differs; the windowed kernel has
    # its own parity test.
    model.atom_transformer.disable_window_attention()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    feats = _make_features(torch.device("cuda"), sc.n_atom, sc.n_token)
    args = atom_encoder_kwargs(feats)
    args = {
        key: value.unsqueeze(0).expand(batch_size, *value.shape).contiguous()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in args.items()
    }
    s = torch.randn(batch_size, sc.n_token, sc.c_s, device="cuda")
    z = torch.randn(batch_size, sc.n_token, sc.n_token, sc.c_z, device="cuda")
    with torch.inference_mode():
        cl, plm, metadata = model.prepare_coords_cache(**args, s=s, z=z)
        biases = model.prepare_pair_biases(plm, sc.n_atom, metadata)
        assert all(bias.shape[0] == batch_size for bias in biases)
        for _ in range(2):
            coords = torch.randn(batch_size, n_sample, sc.n_atom, 3, device="cuda")
            expected = model(
                **args,
                s=s.unsqueeze(1).expand(-1, n_sample, -1, -1),
                z=z.unsqueeze(1).expand(-1, n_sample, -1, -1, -1),
                r_l=coords,
            )
            with (
                patch.object(
                    model.atom_transformer, "prepare_pair_biases", side_effect=AssertionError("Bias recomputed")
                ),
                patch.object(
                    model.atom_transformer, "_precompute_all_biases", side_effect=AssertionError("Bias recomputed")
                ),
            ):
                actual = model.run_coords_cached(
                    args["atom_to_token_idx"], cl, plm, coords, sc.n_token, metadata, prepared_pair_biases=biases
                )
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    result,
                    reference,
                    atol=2e-3 if dtype == "bfloat16" else 1e-5,
                    rtol=2e-2 if dtype == "bfloat16" else 1e-4,
                )


@pytest.mark.parametrize("length", [1, 31, 32, 33, 64])
def test_atom_windows_preserve_valid_pair_offsets(length):
    positions = torch.arange(length * 3).reshape(1, length, 3).float()
    groups = torch.arange(length).reshape(1, length) // 4
    window = atom_windows(positions, groups, build_local_attn_metadata(32, 128), 32, 128)
    reference = update_input_feature_dict({"ref_pos": positions[0], "ref_space_uid": groups[0]})
    for name in ("d_lm", "v_lm"):
        assert torch.equal(window[name][0], reference[name])
    for name, value in reference["pad_info"].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(window["pad_info"][name], value)
        else:
            assert window["pad_info"][name] == value
    assert window["d_lm"].shape == (1, (length + 31) // 32, 32, 128, 3)
    for block in range((length + 31) // 32):
        for query in range(min(32, length - block * 32)):
            index = block * 32 + query
            key_slot = index - block * 32 + 48
            assert window["pad_info"]["mask_trunked"][block, query, key_slot]
            assert torch.equal(window["d_lm"][0, block, query, key_slot], torch.zeros(3))
            assert window["v_lm"][0, block, query, key_slot, 0]
    assert int(window["pad_info"]["mask_trunked"].sum()) >= length


@pytest.mark.parametrize("n_sample", [1, 3])
def test_windowed_prepared_encoder_matches_per_step_path(n_sample: int) -> None:
    """With prepared biases, with or without the prepared conditioning, the encoder's attention runs on the windowed
    kernel, reading q, k and v in place and the samples' shared bias; it matches the per-step path (gathered key
    windows, per-sample biases)."""
    from bionemo_ir._torch.attention_backend.window_attention import window_attention_supported
    from bionemo_ir._torch.layers.attention import AttentionPairBias

    if not torch.cuda.is_available() or not window_attention_supported(torch.bfloat16, 32, 32, 128):
        pytest.skip("no windowed attention kernel for this device")
    torch.manual_seed(42)
    sc = Scenario(dtype="bfloat16", n_atom=200, n_token=50)
    config = DiffusionAtomAttentionEncoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=_atom_transformer_config(sc),
        dtype="bfloat16",
    )
    model = ProtenixAtomAttentionEncoder(config).cuda().eval()
    assert model.atom_transformer.window_attention_enabled
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    feats = _make_features(torch.device("cuda"), sc.n_atom, sc.n_token)
    args = {
        key: value.unsqueeze(0).contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in atom_encoder_kwargs(feats).items()
    }
    s = torch.randn(1, sc.n_token, sc.c_s, device="cuda")
    z = torch.randn(1, sc.n_token, sc.n_token, sc.c_z, device="cuda")
    windowed = []
    run_windowed = AttentionPairBias._windowed_attention

    def counted(self, *call_args):
        windowed.append(call_args[0].shape)
        return run_windowed(self, *call_args)

    with torch.inference_mode():
        cl, plm, metadata = model.prepare_coords_cache(**args, s=s, z=z)
        biases = model.prepare_pair_biases(plm, sc.n_atom, metadata)
        conditioning = model.prepare_conditioning(cl, plm.shape[1], metadata)
        coords = torch.randn(1, n_sample, sc.n_atom, 3, device="cuda")
        run = (args["atom_to_token_idx"], cl, plm, coords, sc.n_token, metadata)
        expected = model.run_coords_cached(*run)
        assert not windowed
        with patch.object(AttentionPairBias, "_windowed_attention", counted):
            actual = [
                model.run_coords_cached(*run, prepared_pair_biases=biases, **cached)
                for cached in ({}, {"prepared_conditioning": conditioning})
            ]
    assert len(windowed) == 2 * len(model.atom_transformer.layers)
    for outputs in actual:
        for name, result, reference in zip(("a", "q_l", "c_l", "p_lm"), outputs, expected, strict=True):
            assert _rmse_ratio(result, reference) < 2e-2, name
            torch.testing.assert_close(result.float(), reference.float(), atol=5e-2, rtol=5e-2, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cached_encoder_hands_the_samples_shared_biases_on() -> None:
    """The cached encoder hands its transformer one bias per batch item, which the item's samples share; when the
    transformer declines the prepared conditioning under CUDA autocast, it also gets ``c_l``. (A stand-in transformer
    records its inputs.)"""
    torch.manual_seed(42)
    sc = Scenario(n_sample=3)
    config = DiffusionAtomAttentionEncoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=_atom_transformer_config(sc),
        dtype=sc.dtype,
    )
    model = ProtenixAtomAttentionEncoder(config).cuda().eval()
    feats = _make_features(torch.device("cuda"), sc.n_atom, sc.n_token)
    args = {
        key: value.unsqueeze(0).contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in atom_encoder_kwargs(feats).items()
    }
    s = torch.randn(1, sc.n_token, sc.c_s, device="cuda")
    z = torch.randn(1, sc.n_token, sc.n_token, sc.c_z, device="cuda")
    seen = []

    class Recorder(torch.nn.Module):
        dtype = torch.float32

        def forward(self, q, c, *args, prepared_pair_biases=None, prepared_conditioning=None, **kwargs):
            sizes = {bias.shape[0] for bias in prepared_pair_biases}
            seen.append((sizes, c is not None, prepared_conditioning is not None))
            return q

    with torch.inference_mode():
        cl, plm, metadata = model.prepare_coords_cache(**args, s=s, z=z)
        biases = model.prepare_pair_biases(plm, sc.n_atom, metadata)
        conditioning = model.prepare_conditioning(cl, plm.shape[1], metadata)
        assert conditioning is not None, "the prepared conditioning did not engage"
        model.atom_transformer = Recorder()
        coords = torch.randn(1, sc.n_sample, sc.n_atom, 3, device="cuda")
        run = (args["atom_to_token_idx"], cl, plm, coords, sc.n_token, metadata)
        cached = {"prepared_pair_biases": biases, "prepared_conditioning": conditioning}
        model.run_coords_cached(*run, **cached)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.run_coords_cached(*run, **cached)
    assert seen == [({1}, False, True), ({1}, True, False)]


def test_window_route_follows_the_device(monkeypatch) -> None:
    """The window route is chosen again whenever the module moves: off the GPU, or on a device without the kernel,
    the layers lose it and a cached step runs on the attention backend; back on a serving GPU they get it."""
    from bionemo_ir._torch.attention_backend.window_attention import window_attention_supported
    from bionemo_ir._torch.layers import attention

    if not torch.cuda.is_available() or not window_attention_supported(torch.bfloat16, 32, 32, 128):
        pytest.skip("no windowed attention kernel for this device")
    torch.manual_seed(42)
    sc = Scenario(dtype="bfloat16", n_atom=200, n_token=50)
    config = DiffusionAtomAttentionEncoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        c_s=sc.c_s,
        c_z=sc.c_z,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=_atom_transformer_config(sc),
        dtype="bfloat16",
    )
    model = ProtenixAtomAttentionEncoder(config).cuda().eval()
    transformer = model.atom_transformer
    assert transformer.window_attention_enabled
    model.cpu()
    assert not transformer.window_attention_enabled
    model.cuda()
    assert transformer.window_attention_enabled
    monkeypatch.setattr(attention, "auto_select_window_attention_backend", lambda *args, **kwargs: None)
    model.cuda()
    assert all(layer.pair_bias_attn.window_attention is None for layer in transformer.layers)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    feats = _make_features(torch.device("cuda"), sc.n_atom, sc.n_token)
    args = {
        key: value.unsqueeze(0).contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in atom_encoder_kwargs(feats).items()
    }
    s = torch.randn(1, sc.n_token, sc.c_s, device="cuda")
    z = torch.randn(1, sc.n_token, sc.n_token, sc.c_z, device="cuda")
    with torch.inference_mode():
        cl, plm, metadata = model.prepare_coords_cache(**args, s=s, z=z)
        biases = model.prepare_pair_biases(plm, sc.n_atom, metadata)
        conditioning = model.prepare_conditioning(cl, plm.shape[1], metadata)
        coords = torch.randn(1, 3, sc.n_atom, 3, device="cuda")
        run = (args["atom_to_token_idx"], cl, plm, coords, sc.n_token, metadata)
        expected = model.run_coords_cached(*run, prepared_pair_biases=biases)
        actual = model.run_coords_cached(*run, prepared_pair_biases=biases, prepared_conditioning=conditioning)
    for name, result, reference in zip(("a", "q_l", "c_l", "p_lm"), actual, expected, strict=True):
        torch.testing.assert_close(result.float(), reference.float(), atol=5e-2, rtol=5e-2, msg=name)
