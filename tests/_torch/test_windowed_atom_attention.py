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
"""The Boltz, OpenFold3 and Protenix atom transformers on the windowed attention kernel."""

import importlib
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from bionemo_ir._torch.attention_backend.window_attention import window_attention_supported
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.transformers import atom as boltz_atom
from bionemo_ir._torch.layers.transformers.atom import AtomTransformer
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    BoltzDiffusionTransformer,
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from bionemo_ir._torch.modules.openfold3 import sequence_local_atom_attention as openfold3_atom
from bionemo_ir._torch.modules.protenix import ProtenixAtomAttentionDecoder, ProtenixAtomAttentionEncoder
from bionemo_ir.configs import DiffusionTransformerConfig
from bionemo_ir.models.openfold3.config import DiffusionModuleConfig
from bionemo_ir.models.protenix.config import AtomAttentionDecoderConfig, DiffusionAtomAttentionEncoderConfig

QUERIES, KEYS, DIM, PAIR, HEADS, LAYERS = 32, 128, 128, 16, 4, 2
KINDS = ("boltz", "openfold3", "protenix")
# Each model's atom-transformer variant.
_FLAGS = {
    "boltz": {"bias_proj": False, "conditioned_transition_using_silu": False, "version": "v2"},
    "openfold3": {
        "bias_proj": True,
        "initial_norm": False,
        "use_ada_layer_norm": True,
        "use_separate_layer_norm": True,
        "conditioned_transition_using_silu": True,
        "shared_pair_norm": True,
        "version": "v1",
    },
    "protenix": {
        "bias_proj": True,
        "initial_norm": False,
        "use_ada_layer_norm": True,
        "use_separate_layer_norm": True,
        "chain_kv_norm": True,
        "conditioned_transition_using_silu": True,
        "version": "v1",
    },
}


@pytest.fixture(scope="module", autouse=True)
def _windowed_kernel() -> None:
    if not torch.cuda.is_available() or not window_attention_supported(torch.bfloat16, DIM // HEADS, QUERIES, KEYS):
        pytest.skip("no windowed attention kernel on this device")


def _build(kind: str, dtype: str) -> torch.nn.Module:
    """A model's atom transformer, its layer stack routed as its encoders and decoders route it."""
    config = DiffusionTransformerConfig(
        num_blocks=LAYERS,
        num_heads=HEADS,
        dim=DIM,
        dim_single_cond=DIM,
        dim_pairwise=PAIR,
        expansion_factor=2,
        pair_norm=True,
        post_layer_norm=False,
        attention_initial_norm=False,
        pairwise_attention_backend="SDPA",
        dtype=dtype,
        **_FLAGS[kind],
    )
    if kind == "boltz":
        return AtomTransformer(QUERIES, KEYS, config, BoltzDiffusionTransformer).cuda().eval()
    model = (OpenFold3DiffusionTransformer if kind == "openfold3" else ProtenixDiffusionTransformer)(config)
    model = model.cuda().eval()
    model.enable_window_attention(QUERIES, KEYS)
    return model


def _layers(model: torch.nn.Module):
    return model.diffusion_transformer if isinstance(model, AtomTransformer) else model


def _randomize(model: torch.nn.Module) -> None:
    """Random weights, with nonzero biases, so that in_proj's k/v bias would show if windowed keys took it."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.1)
            elif name.endswith("weight"):
                param.normal_(1.0, 0.1)
            else:
                param.normal_(std=0.05)


def _case(kind: str, batch: int, samples: int, windows: int) -> tuple[Callable, torch.Tensor]:
    """Inputs with padded atoms; return a run of a model on them and the output rows of the real atoms.

    The run takes the model, the dtype to run in, and whether the pair biases are prepared (with the key mask)
    or the call passes the mask.
    """
    atoms = windows * QUERIES
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(windows, QUERIES, KEYS, torch.device("cuda"))
    mask = torch.ones(batch, atoms, device="cuda")
    mask[:, atoms - 11 :] = 0
    if kind == "boltz":
        q = torch.randn(batch, samples, atoms, DIM, device="cuda")
        c = torch.randn(batch, atoms, DIM, device="cuda")
        bias = torch.randn(batch, atoms, KEYS, LAYERS * HEADS, device="cuda")

        def run(model: torch.nn.Module, dtype: torch.dtype, prepared: bool = True) -> torch.Tensor:
            args = (q.to(dtype), c.to(dtype), bias.to(dtype), mask.to(dtype), metadata)
            if not prepared:
                return model(*args)
            return model(*args, prepared_pair_biases=model.prepare_pair_biases(*args[2:]))

        return run, mask.bool()[:, None].expand(batch, samples, atoms)
    if kind == "openfold3":
        a = torch.randn(batch, samples, windows, QUERIES, DIM, device="cuda")
        s = torch.randn(batch, 1, windows, QUERIES, DIM, device="cuda")
        z = torch.randn(batch, 1, windows, QUERIES, KEYS, PAIR, device="cuda")
        blocks = mask.view(batch, windows, QUERIES)

        def run(model: torch.nn.Module, dtype: torch.dtype, prepared: bool = True) -> torch.Tensor:
            args = {"a": a.to(dtype), "s": s.to(dtype), "z": z.to(dtype), "attn_metadata": metadata}
            if not prepared:
                return model(**args, mask=blocks.to(dtype))
            return model(**args, prepared_pair_biases=model.prepare_pair_biases(args["z"], blocks.to(dtype), metadata))

        return run, blocks.bool()[:, None].expand(batch, samples, windows, QUERIES)
    # A batch item's samples are consecutive rows, which share the item's prepared bias.
    a = torch.randn(batch * samples, atoms, DIM, device="cuda")
    s = torch.randn(batch, atoms, DIM, device="cuda").repeat_interleave(samples, 0)
    z = torch.randn(batch, windows, QUERIES, KEYS, PAIR, device="cuda")

    def run(model: torch.nn.Module, dtype: torch.dtype, prepared: bool = True) -> torch.Tensor:
        args = (a.to(dtype), s.to(dtype))
        if not prepared:
            rows = mask.repeat_interleave(samples, 0).to(dtype)
            return model(*args, z.repeat_interleave(samples, 0).to(dtype), rows, QUERIES, KEYS, metadata)
        biases = model.prepare_pair_biases(z.unsqueeze(1).to(dtype), mask.to(dtype), metadata)
        return model(*args, z.to(dtype), None, QUERIES, KEYS, metadata, prepared_pair_biases=biases)

    return run, mask.bool().repeat_interleave(samples, 0)


def _counting_windowed_calls() -> tuple[list, object]:
    calls = []
    windowed_attention = AttentionPairBias._windowed_attention

    def counted(self, *args):
        calls.append(self)
        return windowed_attention(self, *args)

    return calls, patch.object(AttentionPairBias, "_windowed_attention", counted)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch,samples,windows", [(1, 1, 4), (2, 3, 7)])
def test_windowed_kernel_is_as_accurate_as_the_backend(kind: str, batch: int, samples: int, windows: int) -> None:
    """On prepared biases every layer runs the windowed kernel, which lands as close to an fp32 run of the same
    weights as the attention backend does in bf16; the two bf16 runs round in a different order, so they differ
    by about an ulp of the output."""
    torch.manual_seed(0)
    model = _build(kind, "bfloat16")
    _randomize(model)
    reference = _build(kind, "float32")
    reference.load_state_dict({name: value.float() for name, value in model.state_dict().items()})
    assert _layers(model).window_attention_enabled
    assert not _layers(reference).window_attention_enabled
    run, rows = _case(kind, batch, samples, windows)
    calls, counting = _counting_windowed_calls()
    with torch.inference_mode():
        expected = run(reference, torch.float32)
        with counting:
            actual = run(model, torch.bfloat16).float()
            assert len(calls) == LAYERS
            _layers(model).disable_window_attention()
            backend = run(model, torch.bfloat16).float()
    assert len(calls) == LAYERS

    def errors(result: torch.Tensor) -> tuple[float, float]:
        difference = (result - expected)[rows]
        return (difference.norm() / expected[rows].norm()).item(), difference.abs().max().item()

    (relative, largest), (backend_relative, backend_largest) = errors(actual), errors(backend)
    assert backend_relative < 2e-2
    assert relative <= 1.1 * backend_relative, (relative, backend_relative)
    assert largest <= 2 * backend_largest, (largest, backend_largest)


@pytest.mark.parametrize("kind", KINDS)
def test_calls_that_pass_the_mask_keep_the_backend(kind: str) -> None:
    """Pair biases that do not carry the key mask come with the mask, which only the attention backend applies."""
    torch.manual_seed(0)
    model = _build(kind, "bfloat16")
    _randomize(model)
    run, _ = _case(kind, 2, 3, 5)
    calls, counting = _counting_windowed_calls()
    with torch.inference_mode(), counting:
        run(model, torch.bfloat16, prepared=False)
    assert _layers(model).window_attention_enabled
    assert not calls


@pytest.mark.parametrize("kind", KINDS)
def test_windowed_kernel_replays_in_a_cuda_graph(kind: str) -> None:
    torch.manual_seed(0)
    model = _build(kind, "bfloat16")
    _randomize(model)
    run, _ = _case(kind, 2, 3, 5)
    calls, counting = _counting_windowed_calls()
    with torch.inference_mode():
        eager = run(model, torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(model, torch.bfloat16)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with counting, torch.cuda.graph(graph):
            captured = run(model, torch.bfloat16)
        captured.zero_()
        graph.replay()
    assert len(calls) == LAYERS
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def _diffusion_atom_transformers(name: str, dtype: str) -> list[torch.nn.Module]:
    """The atom-transformer layer stacks of a model's diffusion atom encoder and decoder, on the GPU."""
    torch_dtype = getattr(torch, dtype)
    if name in ("boltz1", "boltz2"):
        config = importlib.import_module(f"bionemo_ir.models.{name}.config").ScoreModelConfig()
        windows = (config.atoms_per_window_queries, config.atoms_per_window_keys)
        encoder = boltz_atom.AtomAttentionEncoder(
            config.atom_s,
            config.token_s,
            *windows,
            config.atom_encoder.model_copy(update={"dtype": dtype}),
            BoltzDiffusionTransformer,
            dtype=torch_dtype,
        )
        decoder = boltz_atom.AtomAttentionDecoder(
            config.token_s,
            config.atom_s,
            *windows,
            config.atom_decoder.model_copy(update={"dtype": dtype}),
            BoltzDiffusionTransformer,
            dtype=torch_dtype,
        )
        stacks = [encoder.atom_encoder.diffusion_transformer, decoder.atom_decoder.diffusion_transformer]
    elif name == "openfold3":
        config = DiffusionModuleConfig()
        encoder = openfold3_atom.AtomAttentionEncoder(
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            n_query=config.n_query,
            n_key=config.n_key,
            c_s=config.c_s,
            c_z=config.c_z,
            atom_transformer_config=config.atom_transformer_encoder_config.model_copy(update={"dtype": dtype}),
            add_noisy_pos=config.add_noisy_pos,
            dtype=torch_dtype,
        )
        decoder = openfold3_atom.AtomAttentionDecoder(
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            c_hidden=config.c_hidden,
            n_query=config.n_query,
            n_key=config.n_key,
            atom_attn_decoder_config=config.atom_transformer_decoder_config.model_copy(update={"dtype": dtype}),
            dtype=torch_dtype,
        )
        stacks = [encoder.atom_transformer, decoder.atom_transformer]
    else:
        encoder = ProtenixAtomAttentionEncoder(DiffusionAtomAttentionEncoderConfig(dtype=dtype))
        decoder = ProtenixAtomAttentionDecoder(AtomAttentionDecoderConfig(dtype=dtype))
        stacks = [encoder.atom_transformer, decoder.atom_transformer]
    encoder.cuda()
    decoder.cuda()
    return stacks


@pytest.mark.parametrize("name", ["boltz1", "boltz2", "openfold3", "protenix"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_diffusion_atom_attention_takes_the_windowed_kernel_in_bf16(name: str, dtype: str) -> None:
    """The diffusion atom encoder and decoder serve prepared biases with the windowed kernel in bf16; in fp32,
    which the kernel does not take, the attention backend serves them."""
    for stack in _diffusion_atom_transformers(name, dtype):
        assert stack.window_attention_enabled is (dtype == "bfloat16")


def test_window_route_follows_the_dtype() -> None:
    """The route is chosen again whenever the module moves, here to a dtype without the kernel and back."""
    model = _build("boltz", "bfloat16")
    layers = _layers(model)
    assert layers.window_attention_enabled
    model.float()
    assert all(layer.pair_bias_attn.window_attention is None for layer in layers.layers)
    model.bfloat16()
    assert layers.window_attention_enabled
    layers.disable_window_attention()
    model.bfloat16()
    assert not layers.window_attention_enabled
