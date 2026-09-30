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
"""Prepared atom pair biases that carry the key mask match the per-call mask."""

import pytest
import torch

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.transformers.atom import AtomTransformer
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    BoltzDiffusionTransformer,
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from bionemo_ir.configs import DiffusionTransformerConfig

WINDOWS, QUERIES, KEYS, DIM, PAIR, HEADS = 3, 32, 128, 128, 16, 4
# Atoms past this are padding, so the key mask drops more than the out-of-range slots.
ATOMS = WINDOWS * QUERIES - 11


def _config(dtype: str, bias_proj: bool) -> DiffusionTransformerConfig:
    return DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=HEADS,
        dim=DIM,
        dim_single_cond=DIM,
        dim_pairwise=PAIR,
        bias_proj=bias_proj,
        pair_norm=True,
        dtype=dtype,
        pairwise_attention_backend="SDPA",
        version="v1",
        mask_inf=1e9,
        post_layer_norm=False,
        attention_initial_norm=False,
    )


def _init(model: torch.nn.Module) -> torch.nn.Module:
    model = model.cuda().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    return model


def _case(model_type: type, dtype: str, batch: int, samples: int):
    """Return the model, forward kwargs, the pair it prepares from, and the mask forward takes."""
    torch.manual_seed(7)
    td = getattr(torch, dtype)
    atoms = WINDOWS * QUERIES
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(WINDOWS, QUERIES, KEYS, torch.device("cuda"))
    mask = torch.ones(batch, atoms, device="cuda", dtype=td)
    mask[:, ATOMS:] = 0
    if model_type is ProtenixDiffusionTransformer:
        model = _init(ProtenixDiffusionTransformer(_config(dtype, bias_proj=True)))
        a = torch.randn(batch, atoms, DIM, device="cuda", dtype=td)
        z = torch.randn(batch, WINDOWS, QUERIES, KEYS, PAIR, device="cuda", dtype=td)
        kwargs = {"a": a, "s": torch.randn_like(a), "z": z, "n_queries": QUERIES, "n_keys": KEYS}
        return model, kwargs, metadata, z.unsqueeze(1), mask
    if model_type is OpenFold3DiffusionTransformer:
        model = _init(OpenFold3DiffusionTransformer(_config(dtype, bias_proj=True)))
        a = torch.randn(batch, samples, WINDOWS, QUERIES, DIM, device="cuda", dtype=td)
        s = torch.randn(batch, 1, WINDOWS, QUERIES, DIM, device="cuda", dtype=td)
        z = torch.randn(batch, 1, WINDOWS, QUERIES, KEYS, PAIR, device="cuda", dtype=td)
        return model, {"a": a, "s": s, "z": z}, metadata, z, mask.view(batch, WINDOWS, QUERIES)
    model = _init(
        AtomTransformer(
            attn_window_queries=QUERIES,
            attn_window_keys=KEYS,
            diffusion_transformer_config=_config(dtype, bias_proj=False),
            diffusion_transformer_cls=BoltzDiffusionTransformer,
        )
    )
    q = torch.randn(batch, samples, atoms, DIM, device="cuda", dtype=td)
    bias = torch.randn(batch, atoms, KEYS, 2 * HEADS, device="cuda", dtype=td)
    kwargs = {"q": q, "c": torch.randn(batch, atoms, DIM, device="cuda", dtype=td), "bias": bias}
    return model, kwargs, metadata, bias, mask


def _prepare(model: torch.nn.Module, pair: torch.Tensor, mask: torch.Tensor | None, metadata) -> list[torch.Tensor]:
    if isinstance(model, AtomTransformer):
        return model.prepare_pair_biases(pair, mask, metadata)
    return model.prepare_pair_biases(pair, mask, metadata) if mask is not None else model.prepare_pair_biases(pair)


@pytest.mark.parametrize(
    "model_type,samples",
    [
        (OpenFold3DiffusionTransformer, 1),
        (OpenFold3DiffusionTransformer, 2),
        (ProtenixDiffusionTransformer, 1),
        (AtomTransformer, 1),
        (AtomTransformer, 2),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch", [1, 2])
def test_key_mask_in_prepared_biases(model_type: type, samples: int, dtype: str, batch: int) -> None:
    model, kwargs, metadata, pair, mask = _case(model_type, dtype, batch, samples)
    boltz = model_type is AtomTransformer
    with torch.inference_mode():
        if boltz:
            # AtomTransformer takes its mask and drops it itself once biases are prepared.
            expected = model(**kwargs, mask=mask, attn_metadata=metadata).clone()
            biases = _prepare(model, pair, mask, metadata)
            call = {**kwargs, "mask": mask, "attn_metadata": metadata, "prepared_pair_biases": biases}
        else:
            expected = model(
                **kwargs, mask=mask, attn_metadata=metadata, prepared_pair_biases=_prepare(model, pair, None, metadata)
            )
            expected = expected.clone()
            biases = _prepare(model, pair, mask, metadata)
            call = {**kwargs, "mask": None, "attn_metadata": metadata, "prepared_pair_biases": biases}
        torch.testing.assert_close(model(**call), expected, rtol=0, atol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                model(**call)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = model(**call)
        next(iter(kwargs.values())).add_(0.1)
        graph.replay()
        torch.testing.assert_close(captured, model(**call), rtol=0, atol=0)


def _attention(backend: str) -> AttentionPairBias:
    return AttentionPairBias(layer_idx=0, c_s=DIM, c_z=PAIR, num_heads=HEADS, attn_backend=backend).cuda()


def test_add_key_mask_needs_sequence_local_additive_attention() -> None:
    pair = torch.zeros(1, 1, WINDOWS, HEADS, QUERIES, KEYS, device="cuda")
    mask = torch.ones(1, WINDOWS, QUERIES, device="cuda")
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(WINDOWS, QUERIES, KEYS, torch.device("cuda"))
    with pytest.raises(ValueError, match="sequence-local"):
        _attention("SDPA").add_key_mask(pair, mask, AttentionMetadata())
    with pytest.raises(ValueError, match="sequence-local"):
        _attention("CuTeDSL").add_key_mask(pair, mask, metadata)


def test_attention_without_a_mask_needs_a_pair_bias() -> None:
    s = torch.zeros(1, 1, WINDOWS, QUERIES, DIM, device="cuda")
    with pytest.raises(ValueError, match="pair bias"):
        _attention("SDPA")._prep_mask_bias(s, None, None, None)
