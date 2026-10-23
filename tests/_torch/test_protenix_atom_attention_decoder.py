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
"""OSS equivalence tests for the Protenix atom decoder (AF3 Algorithm 6)."""

import math
import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.modules.protenix import ProtenixAtomAttentionDecoder
from bionemo_ir.models.protenix.config import AtomAttentionDecoderConfig
from bionemo_ir.models.protenix.convert import convert_atom_attention_decoder_torch
from bionemo_ir.utils import str_dtype_to_torch
from tests.common.test_utils.protenix.ref_layers_from_oss import RefProtenixAtomAttentionDecoderFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 64
    n_token: int = 16
    c_token: int = 768
    c_atom: int = 128
    c_atompair: int = 16
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    batch_size: int = 1
    dtype: str = "float32"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32"),
        Scenario(dtype="bfloat16"),
    ],
    ids=["fp32", "bf16"],
)
def test_protenix_atom_attention_decoder(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)
    B, Na, Nt = sc.batch_size, sc.n_atoms, sc.n_token

    # Randomize the OSS reference for non-trivial coverage.
    ref = (
        RefProtenixAtomAttentionDecoderFromOSS.build(
            n_blocks=sc.n_blocks,
            n_heads=sc.n_heads,
            c_token=sc.c_token,
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
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

    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        dtype=sc.dtype,
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    converted = convert_atom_attention_decoder_torch(config, ref.state_dict(), prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    K = math.ceil(Na / sc.n_queries)
    atom_to_token_idx = (torch.arange(Na, device=device) // (Na // Nt)).long().unsqueeze(0).expand(B, Na)
    a = torch.randn(B, Nt, sc.c_token, device=device)
    q_skip = torch.randn(B, Na, sc.c_atom, device=device)
    c_skip = torch.randn(B, Na, sc.c_atom, device=device)
    p_skip = torch.randn(B, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device)

    with torch.inference_mode():
        exp = ref(atom_to_token_idx, a, q_skip, c_skip, p_skip)
        act = model(
            atom_to_token_idx, a.to(torch_dtype), q_skip.to(torch_dtype), c_skip.to(torch_dtype), p_skip.to(torch_dtype)
        )

    r = _rmse_ratio(act, exp)
    tol = 3e-3 if torch_dtype == torch.float32 else 5e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch_size,n_sample", [(1, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize("precompute", [False, True])
def test_cached_decoder_biases(dtype: str, batch_size: int, n_sample: int, precompute: bool) -> None:
    """Biases prepared once from ``p_lm`` replace every folded sample's per-step projection."""
    torch.manual_seed(42)
    sc = Scenario(dtype=dtype)
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(dtype)
    transformer = AtomAttentionDecoderConfig().atom_transformer_config.model_copy(
        update={"precompute_bias": precompute}
    )
    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        atom_transformer_config=transformer,
        dtype=dtype,
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    # Both paths on the attention backend, so that only the bias preparation differs; the windowed kernel has
    # its own parity test.
    model.atom_transformer.disable_window_attention()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.1)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()

    rows, K = batch_size * n_sample, math.ceil(sc.n_atoms / sc.n_queries)
    a2t = (torch.arange(sc.n_atoms, device=device) // (sc.n_atoms // sc.n_token)).expand(rows, -1)
    a = torch.randn(rows, sc.n_token, sc.c_token, device=device, dtype=torch_dtype)
    q_skip = torch.randn(rows, sc.n_atoms, sc.c_atom, device=device, dtype=torch_dtype)
    c_skip = torch.randn_like(q_skip)
    p_lm = torch.randn(batch_size, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device, dtype=torch_dtype)
    p_skip = p_lm.unsqueeze(1).expand(batch_size, n_sample, *p_lm.shape[1:]).reshape(rows, *p_lm.shape[1:])
    reject = AssertionError("Bias recomputed")
    with torch.inference_mode():
        metadata = model.atom_transformer.build_attn_metadata(K, sc.n_queries, sc.n_keys, device)
        biases = model.prepare_pair_biases(p_lm, sc.n_atoms, metadata)
        assert all(bias.shape[0] == batch_size for bias in biases)
        expected = model(a2t, a, q_skip, c_skip, p_skip)
        with (
            patch.object(model.atom_transformer, "_precompute_all_biases", side_effect=reject),
            patch.object(AttentionPairBias, "project_pair_bias", side_effect=reject),
        ):
            actual = model(a2t, a, q_skip, c_skip, p_skip, prepared_pair_biases=biases)
        unbiased = model(a2t, a, q_skip, c_skip, p_skip, prepared_pair_biases=[torch.zeros_like(b) for b in biases])

    exact = torch_dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5 if exact else 2e-3, rtol=1e-4 if exact else 2e-2)
    assert not torch.allclose(unbiased, expected, atol=1e-3), "pair biases do not reach the decoder output"


def test_cached_decoder_hands_c_skip_on_when_its_transformer_declines_the_cache() -> None:
    """With the cache the decoder drops c_skip; under CUDA autocast its transformer declines the prepared conditioning,
    so the decoder hands c_skip on. (A stand-in transformer records its inputs: the atom kernels run without autocast.)"""
    torch.manual_seed(0)
    sc = Scenario(dtype="float32")
    device = torch.device("cuda")
    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token, c_atom=sc.c_atom, c_atompair=sc.c_atompair, n_queries=sc.n_queries, n_keys=sc.n_keys
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    K = math.ceil(sc.n_atoms / sc.n_queries)
    a2t = (torch.arange(sc.n_atoms, device=device) // (sc.n_atoms // sc.n_token)).expand(1, -1)
    a = torch.randn(1, sc.n_token, sc.c_token, device=device)
    q_skip = torch.randn(1, sc.n_atoms, sc.c_atom, device=device)
    c_skip = torch.randn_like(q_skip)
    p_lm = torch.randn(1, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device)
    seen = []

    class Recorder(torch.nn.Module):
        dtype = torch.float32

        def forward(self, q, c, *args, prepared_conditioning=None, **kwargs):
            seen.append((c is not None, prepared_conditioning is not None))
            return q

    with torch.inference_mode():
        metadata = model.atom_transformer.build_attn_metadata(K, sc.n_queries, sc.n_keys, device)
        biases = model.prepare_pair_biases(p_lm, sc.n_atoms, metadata)
        conditioning = model.prepare_conditioning(c_skip, K, metadata)
        assert conditioning is not None, "the prepared conditioning did not engage"
        model.atom_transformer = Recorder()
        cached = {"prepared_pair_biases": biases, "prepared_conditioning": conditioning}
        model(a2t, a, q_skip, c_skip, p_lm, metadata, **cached)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(a2t, a, q_skip, c_skip, p_lm, metadata, **cached)
    assert seen == [(False, True), (True, False)]


@pytest.mark.parametrize("n_sample", [1, 3])
def test_windowed_prepared_decoder_matches_per_step_path(n_sample: int) -> None:
    """With prepared biases, with or without the prepared conditioning, the decoder's attention runs on the windowed
    kernel, reading q, k and v in place and the samples' shared bias; it matches the per-step path (gathered key
    windows, per-sample biases)."""
    from bionemo_ir._torch.attention_backend.window_attention import window_attention_supported

    if not torch.cuda.is_available() or not window_attention_supported(torch.bfloat16, 32, 32, 128):
        pytest.skip("no windowed attention kernel for this device")
    torch.manual_seed(42)
    sc = Scenario(dtype="bfloat16", n_atoms=200, n_token=50)
    device = torch.device("cuda")
    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token,
        c_atom=sc.c_atom,
        c_atompair=sc.c_atompair,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        dtype="bfloat16",
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    assert model.atom_transformer.window_attention_enabled
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.1)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()

    K = math.ceil(sc.n_atoms / sc.n_queries)
    a2t = (torch.arange(sc.n_atoms, device=device) // (sc.n_atoms // sc.n_token)).expand(n_sample, -1)
    a = torch.randn(n_sample, sc.n_token, sc.c_token, device=device, dtype=torch.bfloat16)
    q_skip = torch.randn(n_sample, sc.n_atoms, sc.c_atom, device=device, dtype=torch.bfloat16)
    c_shared = torch.randn(1, sc.n_atoms, sc.c_atom, device=device, dtype=torch.bfloat16)
    p_lm = torch.randn(1, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device, dtype=torch.bfloat16)
    c_skip = c_shared.expand(n_sample, -1, -1).contiguous()
    p_skip = p_lm.expand(n_sample, *p_lm.shape[1:]).contiguous()
    windowed = []
    run_windowed = AttentionPairBias._windowed_attention

    def counted(self, *args):
        windowed.append(args[0].shape)
        return run_windowed(self, *args)

    with torch.inference_mode():
        metadata = model.atom_transformer.build_attn_metadata(K, sc.n_queries, sc.n_keys, device)
        biases = model.prepare_pair_biases(p_lm, sc.n_atoms, metadata)
        conditioning = model.prepare_conditioning(c_shared, K, metadata)
        expected = model(a2t, a, q_skip, c_skip, p_skip)
        assert not windowed
        with patch.object(AttentionPairBias, "_windowed_attention", counted):
            actual = [
                model(a2t, a, q_skip, c_skip, p_skip, prepared_pair_biases=biases, **cached)
                for cached in ({}, {"prepared_conditioning": conditioning})
            ]
    assert len(windowed) == 2 * len(model.atom_transformer.layers)
    for result in actual:
        assert _rmse_ratio(result, expected) < 2e-2
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cached_decoder_hands_the_samples_shared_biases_on() -> None:
    """The cached decoder hands its transformer one bias per batch item, which the item's samples share, also when
    the transformer declines the prepared conditioning under CUDA autocast. (A stand-in transformer records its
    inputs.)"""
    torch.manual_seed(0)
    sc = Scenario(dtype="float32")
    device = torch.device("cuda")
    config = AtomAttentionDecoderConfig(
        c_token=sc.c_token, c_atom=sc.c_atom, c_atompair=sc.c_atompair, n_queries=sc.n_queries, n_keys=sc.n_keys
    )
    model = ProtenixAtomAttentionDecoder(config).to(device).eval()
    n_sample, K = 3, math.ceil(sc.n_atoms / sc.n_queries)
    a2t = (torch.arange(sc.n_atoms, device=device) // (sc.n_atoms // sc.n_token)).expand(n_sample, -1)
    a = torch.randn(n_sample, sc.n_token, sc.c_token, device=device)
    q_skip = torch.randn(n_sample, sc.n_atoms, sc.c_atom, device=device)
    c_shared = torch.randn(1, sc.n_atoms, sc.c_atom, device=device)
    p_lm = torch.randn(1, K, sc.n_queries, sc.n_keys, sc.c_atompair, device=device)
    seen = []

    class Recorder(torch.nn.Module):
        dtype = torch.float32

        def forward(self, q, c, *args, prepared_pair_biases=None, prepared_conditioning=None, **kwargs):
            seen.append(({bias.shape[0] for bias in prepared_pair_biases}, prepared_conditioning is not None))
            return q

    with torch.inference_mode():
        metadata = model.atom_transformer.build_attn_metadata(K, sc.n_queries, sc.n_keys, device)
        biases = model.prepare_pair_biases(p_lm, sc.n_atoms, metadata)
        conditioning = model.prepare_conditioning(c_shared, K, metadata)
        assert conditioning is not None, "the prepared conditioning did not engage"
        model.atom_transformer = Recorder()
        c_skip = c_shared.expand(n_sample, -1, -1)
        cached = {"prepared_pair_biases": biases, "prepared_conditioning": conditioning}
        model(a2t, a, q_skip, c_skip, p_lm, metadata, **cached)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(a2t, a, q_skip, c_skip, p_lm, metadata, **cached)
    assert seen == [({1}, True), ({1}, False)]
