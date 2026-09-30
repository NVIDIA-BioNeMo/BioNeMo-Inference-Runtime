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
"""Static atom bias reuse across model families."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from bionemo_ir._torch.modules.openfold3.sequence_local_atom_attention import AtomAttentionEncoder
from bionemo_ir.configs import DiffusionTransformerConfig


@pytest.mark.parametrize(
    "model_type,shared_norm",
    [
        (OpenFold3DiffusionTransformer, False),
        (OpenFold3DiffusionTransformer, True),
        (ProtenixDiffusionTransformer, False),
    ],
)
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prepared_atom_biases(
    model_type: type[OpenFold3DiffusionTransformer] | type[ProtenixDiffusionTransformer],
    shared_norm: bool,
    precompute: bool,
    dtype: str,
    batch_size: int,
) -> None:
    torch.manual_seed(42)
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=4,
        dim=128,
        dim_single_cond=128,
        dim_pairwise=16,
        bias_proj=True,
        pair_norm=True,
        shared_pair_norm=shared_norm,
        precompute_bias=precompute,
        dtype=dtype,
        pairwise_attention_backend="SDPA",
        version="v1",
    )
    model = model_type(config).cuda().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    td = config.torch_dtype
    b, windows, queries, keys = batch_size, 2, 32, 128
    a = torch.randn(b, windows * queries, 128, device="cuda", dtype=td)
    s = torch.randn_like(a)
    z = torch.randn(b, 1, windows, queries, keys, 16, device="cuda", dtype=td)
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(windows, queries, keys, a.device)
    if model_type is ProtenixDiffusionTransformer:
        kwargs = {
            "a": a,
            "s": s,
            "z": z.squeeze(1),
            "mask": a.new_ones(b, windows * queries),
            "n_queries": queries,
            "n_keys": keys,
            "attn_metadata": metadata,
        }
    else:
        kwargs = {
            "a": a.reshape(b, 1, windows, queries, 128),
            "s": s.reshape(b, 1, windows, queries, 128),
            "z": z,
            "mask": a.new_ones(b, 1, windows, queries),
            "attn_metadata": metadata,
        }
    with torch.inference_mode():
        biases = model.prepare_pair_biases(z)
        saved = [bias.clone() for bias in biases]
        for step in range(2):
            kwargs["a"] = kwargs["a"] + 0.1 * step
            expected = model(**kwargs).clone()
            with patch.object(model, "_precompute_all_biases", side_effect=AssertionError("Bias recomputed")):
                actual = model(**kwargs, prepared_pair_biases=biases)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for bias, original in zip(biases, saved, strict=True):
            torch.testing.assert_close(bias, original, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(**kwargs, prepared_pair_biases=biases)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = model(**kwargs, prepared_pair_biases=biases)
        for _ in range(2):
            kwargs["a"].add_(0.1)
            expected = model(**kwargs, prepared_pair_biases=biases).clone()
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="count"):
            model(**kwargs, prepared_pair_biases=biases[:1])
        assert all(layer.pair_bias_attn.bias_proj for layer in model.layers)


@pytest.mark.parametrize("prepared", ["none", "single", "pair"])
def test_incomplete_atom_cache(prepared: str) -> None:
    encoder = AtomAttentionEncoder.__new__(AtomAttentionEncoder)
    nn.Module.__init__(encoder)
    encoder.n_query = 32
    encoder.add_noisy_pos = True
    encoder.linear_q = nn.Identity()
    encoder.atom_transformer = nn.Identity()
    encoder.dtype = encoder.atom_transformer.dtype = torch.float32
    atoms = 32
    q = torch.randn(1, atoms, 8)
    c = torch.randn_like(q)
    pair = torch.randn(1, 1, 32, 128, 4)
    mask = torch.ones(1, atoms)
    batch = {"token_mask": mask, "atom_to_token_index": torch.arange(atoms).unsqueeze(0)}
    kwargs = {"batch": batch, "atom_mask": mask, "attn_metadata": None, "rl": torch.randn(1, atoms, 3)}
    with (
        patch.object(encoder, "get_atom_reps", return_value=(q, c, pair)),
        patch.object(encoder.atom_transformer, "forward", side_effect=lambda **kw: kw["a"]) as transformer,
    ):
        expected = encoder(**kwargs)
        actual = encoder(
            **kwargs,
            prepared_cl=c if prepared == "single" else None,
            prepared_plm=pair if prepared == "pair" else None,
            prepared_pair_biases=[torch.full((1,), float("nan"))],
        )
        assert transformer.call_args.kwargs["prepared_pair_biases"] is None
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
