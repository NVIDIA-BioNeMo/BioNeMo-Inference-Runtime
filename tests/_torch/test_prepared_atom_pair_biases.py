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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    LocalLayerConditioning,
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
    _conditioned_local_layer,
    _project_local_conditioning,
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


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prepared_openfold3_atom_conditioning(windowed: bool, dtype: str, batch_size: int) -> None:
    """Conditioning projected once matches each layer's own projections of the same ``s`` bit for bit."""
    torch.manual_seed(0)
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=4,
        dim=128,
        dim_single_cond=128,
        dim_pairwise=16,
        bias_proj=True,
        shared_pair_norm=True,
        initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        conditioned_transition_using_silu=True,
        dtype=dtype,
        pairwise_attention_backend="SDPA",
        version="v1",
    )
    model = OpenFold3DiffusionTransformer(config).cuda().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.1)
    windows, queries, keys = 3, 32, 128
    if windowed:
        model.enable_window_attention(queries, keys)
        if model.layers[0].pair_bias_attn.window_attention is None:
            pytest.skip("no windowed attention backend for this dtype and device")
    td = config.torch_dtype
    samples = 3
    a = torch.randn(batch_size, samples, windows, queries, 128, device="cuda", dtype=td)
    s = torch.randn(batch_size, 1, windows, queries, 128, device="cuda", dtype=td)
    z = torch.randn(batch_size, 1, windows, queries, keys, 16, device="cuda", dtype=td)
    mask = torch.ones(batch_size, 1, windows, queries, device="cuda", dtype=td)
    mask[..., -1, 7:] = 0
    metadata = ProtenixDiffusionTransformer.build_attn_metadata(windows, queries, keys, a.device)
    with torch.inference_mode():
        biases = model.prepare_pair_biases(z, mask, metadata)
        fused_conditioning = model.prepare_local_conditioning(s, metadata)
        assert fused_conditioning is not None
        conditioning = [
            tuple(LocalLayerConditioning(*entry)._replace(transition_cond=None)) for entry in fused_conditioning
        ]
        kwargs = {"a": a, "s": s, "z": z, "mask": None, "attn_metadata": metadata, "prepared_pair_biases": biases}
        expected = model(**kwargs)
        actual = model(**kwargs, prepared_conditioning=conditioning)
        assert torch.equal(actual, expected)
        fused = model(**kwargs, prepared_conditioning=fused_conditioning)
        repeated = model(**kwargs, prepared_conditioning=model.prepare_local_conditioning(s, metadata))
        assert torch.equal(fused, repeated)
        rel = ((fused - expected).float().abs().mean() / expected.float().abs().mean()).item()
        assert rel < 0.02, f"fused atom transitions differ by mean-relative {rel:.2%} (> 2%)"
        with pytest.raises(ValueError, match="carry the mask"):
            model(**{**kwargs, "mask": mask}, prepared_conditioning=conditioning)


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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_local_conditioning_reuses_transition_projection(batch_size: int) -> None:
    """Preparing a fused table reuses the projected AdaLN halves; denoising steps project nothing again."""
    s = torch.randn(batch_size, 1, 2, 3, 4)
    a = torch.randn(batch_size, 2, 2, 3, 4)
    projected = torch.randn(batch_size, 1, 2, 3, 8)
    table = torch.randn(6, 12)
    transition_adaln = SimpleNamespace(
        dim=4, project_condition=Mock(return_value=projected), normalize=Mock(return_value=a)
    )
    transition = SimpleNamespace(
        adaln=transition_adaln,
        fused_condition=Mock(return_value=table),
        run_fused=Mock(return_value=a),
        _swiglu_mlp=Mock(return_value=a),
        _gated_sigmoid_op=Mock(return_value=a),
        output_projection=SimpleNamespace(weight=torch.zeros(4, 4), bias=torch.zeros(4)),
    )
    attention = Mock(return_value=a)
    attention.window_attention = None
    attention.layer_norm_a_q = SimpleNamespace(dim=4, project_condition=Mock(return_value=projected))
    attention.layer_norm_a_k = SimpleNamespace(dim=4, project_condition=Mock(return_value=projected))
    attention.output_projection = Mock(return_value=s)
    layer = SimpleNamespace(pair_bias_attn=attention, transition=transition)
    metadata = SimpleNamespace(query_to_keys=Mock(return_value=s))

    (conditioning,) = _project_local_conditioning([layer], s, metadata)
    prepared = LocalLayerConditioning(*conditioning)
    if batch_size == 1:
        transition.fused_condition.assert_called_once()
        given_s, given_scale, given_shift = transition.fused_condition.call_args.args
        assert given_s is s
        assert given_scale is prepared.transition_scale
        assert given_shift is prepared.transition_shift
        assert prepared.transition_cond is table
    else:
        transition.fused_condition.assert_not_called()
        assert prepared.transition_cond is None

    for _ in range(2):
        _conditioned_local_layer(layer, a, torch.zeros(1), metadata, None, conditioning)
    transition_adaln.project_condition.assert_called_once_with(s)
    attention.layer_norm_a_q.project_condition.assert_called_once_with(s)
    attention.layer_norm_a_k.project_condition.assert_called_once_with(s)
    attention.output_projection.assert_called_once_with(s)
    assert transition.run_fused.call_count == (2 if batch_size == 1 else 0)
    assert transition.fused_condition.call_count == (1 if batch_size == 1 else 0)
    assert transition_adaln.normalize.call_count == (0 if batch_size == 1 else 2)
