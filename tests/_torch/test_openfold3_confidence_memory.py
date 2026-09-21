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

import weakref

import pytest
import torch
from torch import nn

from bionemo_ir._torch.layers.normalization import replace_with_fused_layernorm
from bionemo_ir._torch.modules.openfold3.confidence import (
    AuxiliaryHeadsAllAtom,
    DistogramHead,
    PairformerEmbedding,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
    _finalize_aux_outputs_,
    _project_pair_logits,
    _symmetrize_pair_logits_,
)
from bionemo_ir._torch.utils import (
    CHUNK_REGISTRY,
    CONFIDENCE_PAIR_EMBEDDING,
    CONFIDENCE_PAIR_PROJECTION,
    CONFIDENCE_TRIANGLE_ATTENTION,
    ChunkPolicy,
)
from bionemo_ir.configs import PairformerConfig
from bionemo_ir.models.openfold3.config import AuxiliaryHeadsConfig


class _FakePairformer(nn.Module):
    def forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del single_mask, pair_mask
        return si + 1, zij + 2


class _FakePairformerEmbedding(PairformerEmbedding):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.pairformer_stack = _FakePairformer()

    def embed_zij(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        del si_input
        sample_value = x_pred[..., 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return zij + sample_value


@pytest.mark.parametrize(("input_samples", "batch_shape"), [(1, ()), (3, (2, 3))])
def test_per_sample_pairformer_preserves_batch_and_sample_axes(
    input_samples: int, batch_shape: tuple[int, ...]
) -> None:
    samples, tokens = 3, 5
    module = _FakePairformerEmbedding()
    si_input = torch.randn(*batch_shape, input_samples, tokens, 7)
    si = torch.randn(*batch_shape, input_samples, tokens, 6)
    zij = torch.randn(*batch_shape, input_samples, tokens, tokens, 8)
    x_pred = torch.randn(*batch_shape, samples, tokens, 3)
    single_mask = torch.ones(*batch_shape, samples, tokens)
    pair_mask = torch.ones(*batch_shape, input_samples, tokens, tokens)

    actual_si, actual_zij = module.per_sample_pairformer_emb(si_input, si, zij, x_pred, single_mask, pair_mask)

    expected_si = (si + 1).expand(*batch_shape, samples, tokens, 6)
    expected_zij = zij.expand(*batch_shape, samples, tokens, tokens, 8) + x_pred[..., 0, 0, None, None, None]
    expected_zij = expected_zij + 2
    assert torch.equal(actual_si, expected_si)
    assert torch.equal(actual_zij, expected_zij)


def test_per_sample_pairformer_iterator_releases_previous_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_size, samples, tokens = 1, 2, 5
    module = _FakePairformerEmbedding()
    si_input = torch.randn(batch_size, 1, tokens, 7)
    si = torch.randn(batch_size, 1, tokens, 6)
    zij = torch.randn(batch_size, 1, tokens, tokens, 8)
    x_pred = torch.randn(batch_size, samples, tokens, 3)
    single_mask = torch.ones(batch_size, samples, tokens)
    pair_mask = torch.ones(batch_size, 1, tokens, tokens)
    previous_refs: tuple[weakref.ReferenceType[torch.Tensor], weakref.ReferenceType[torch.Tensor]] | None = None
    embed_calls = 0
    original_embed_zij = module.embed_zij

    def check_previous_sample_released(**kwargs) -> torch.Tensor:
        nonlocal embed_calls
        if embed_calls:
            assert previous_refs is not None
            assert all(reference() is None for reference in previous_refs)
        embed_calls += 1
        return original_embed_zij(**kwargs)

    monkeypatch.setattr(module, "embed_zij", check_previous_sample_released)
    iterator = module.iter_per_sample_pairformer_emb(si_input, si, zij, x_pred, single_mask, pair_mask)
    _, si_sample, zij_sample = next(iterator)
    previous_refs = (weakref.ref(si_sample), weakref.ref(zij_sample))
    del si_sample, zij_sample

    next(iterator)

    assert embed_calls == 2


def test_per_sample_pairformer_preallocates_gpu_output_without_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(43)
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size, samples, tokens = 2, 3, 5
    module = _FakePairformerEmbedding().to(device)
    si_input = torch.randn(batch_size, 1, tokens, 7, device=device)
    si = torch.randn(batch_size, 1, tokens, 6, device=device)
    zij = torch.randn(batch_size, 1, tokens, tokens, 8, device=device)
    x_pred = torch.randn(batch_size, samples, tokens, 3, device=device)
    single_mask = torch.ones(batch_size, samples, tokens, device=device)
    pair_mask = torch.ones(batch_size, 1, tokens, tokens, device=device)
    originals = tuple(value.clone() for value in (si_input, si, zij, x_pred, single_mask, pair_mask))

    expected_si = (si + 1).expand(batch_size, samples, tokens, 6)
    expected_zij = zij.expand(batch_size, samples, tokens, tokens, 8) + x_pred[..., 0, 0, None, None, None]
    expected_zij = expected_zij + 2

    def reject_cat(*_args, **_kwargs):
        raise AssertionError("GPU inference output must not concatenate per-sample tensors")

    monkeypatch.setattr(torch, "cat", reject_cat)
    with torch.inference_mode():
        actual_si, actual_zij = module.per_sample_pairformer_emb(
            si_input,
            si,
            zij,
            x_pred,
            single_mask,
            pair_mask,
        )

    assert actual_si.is_contiguous()
    assert actual_zij.is_contiguous()
    assert actual_si.data_ptr() != si.data_ptr()
    assert actual_zij.data_ptr() != zij.data_ptr()
    assert torch.equal(actual_si, expected_si)
    assert torch.equal(actual_zij, expected_zij)
    for value, original in zip((si_input, si, zij, x_pred, single_mask, pair_mask), originals, strict=True):
        assert torch.equal(value, original)


def test_auxiliary_heads_cpu_offload_matches_real_pairformer(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(31)
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size, samples, tokens, channels = 1, 3, 8, 32
    config = AuxiliaryHeadsConfig(c_s_input=16, c_z=channels, max_atoms_per_token=2)
    config.pairformer = PairformerConfig(
        token_s=channels,
        token_z=channels,
        num_blocks=1,
        num_heads=4,
        pairwise_num_heads=4,
        pairwise_head_width=8,
        trimul_high_precision=False,
        triangle_attention_backend="VANILLA",
        pairwise_attention_backend="SDPA",
        s_path_dtype="bfloat16",
        version="v1",
    )
    for head_config in (config.pae, config.pde, config.distogram):
        head_config.c_z = channels
    for head_config in (config.lddt, config.experimentally_resolved):
        head_config.c_s = channels
        head_config.max_atoms_per_token = 2
    config.set_dtype("bfloat16")
    module = AuxiliaryHeadsAllAtom(config).to(device).eval()
    replace_with_fused_layernorm(module)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    token_mask = torch.ones(batch_size, 1, tokens, device=device)
    batch = {
        "token_mask": token_mask,
        "atom_mask": token_mask.clone(),
        "num_atoms_per_token": torch.ones_like(token_mask, dtype=torch.int64),
        "asym_id": torch.ones_like(token_mask, dtype=torch.int64),
        "start_atom_index": torch.arange(tokens, device=device).expand(batch_size, 1, tokens),
        "restype": torch.nn.functional.one_hot(
            torch.zeros(batch_size, 1, tokens, device=device, dtype=torch.int64), num_classes=32
        ).float(),
        "is_protein": torch.ones_like(token_mask),
        "is_atomized": torch.ones_like(token_mask),
        "is_dna": torch.zeros_like(token_mask),
        "is_rna": torch.zeros_like(token_mask),
    }
    si_input = torch.randn(batch_size, 1, tokens, config.c_s_input, device=device)
    output = {
        "si_trunk": torch.randn(batch_size, 1, tokens, channels, device=device),
        "zij_trunk": torch.randn(batch_size, 1, tokens, tokens, channels, device=device),
        "atom_positions_predicted": torch.randn(batch_size, samples, tokens, 3, device=device) * 10,
    }
    original_output = {name: value.clone() for name, value in output.items()}

    with torch.inference_mode():
        module.apply_per_sample = False
        module.offload_pairformer_outputs = False
        expected = module(batch, si_input, output)
        module.apply_per_sample = True

        original_stream = module._stream_pair_heads
        original_distogram = module.distogram.forward
        original_pairformer = module.pairformer_embedding.forward
        streamed_pair_output_alias: dict[str, torch.Tensor] | None = None
        streaming_events: list[str] = []

        def record_stream(**kwargs):
            nonlocal streamed_pair_output_alias
            streaming_events.append("pair_heads")
            result = original_stream(**kwargs)
            streamed_pair_output_alias = result[1]
            return result

        def record_distogram(z):
            streaming_events.append("distogram")
            return original_distogram(z)

        monkeypatch.setattr(module, "_stream_pair_heads", record_stream)
        monkeypatch.setattr(module.distogram, "forward", record_distogram)
        pae_policy = module.pae.projection_chunk_policy
        pde_policy = module.pde.projection_chunk_policy
        module.pae.projection_chunk_policy = pae_policy.replace(min_size=1)
        module.pde.projection_chunk_policy = pde_policy.replace(min_size=1)
        streamed = module(batch, si_input, output)
        streamed_events = streaming_events.copy()
        module.pae.projection_chunk_policy = pae_policy
        module.pde.projection_chunk_policy = pde_policy

        module.offload_pairformer_outputs = True
        offloaded_events: list[str] = []
        offloaded_pair_outputs: dict[str, torch.Tensor] = {}

        def record_empty_cache() -> None:
            offloaded_events.append("empty_cache")

        def record_cpu_stream(**kwargs):
            offloaded_events.append("pair_heads")
            result = original_stream(**kwargs)
            offloaded_pair_outputs.update(result[1])
            return result

        def reject_pair_archive(*_args, **_kwargs):
            raise AssertionError("streamed CPU offload must not build a full BF16 pair archive")

        def record_offloaded_distogram(z):
            offloaded_events.append("distogram")
            return original_distogram(z)

        monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
        monkeypatch.setattr(module, "_stream_pair_heads", record_cpu_stream)
        monkeypatch.setattr(module.pairformer_embedding, "forward", reject_pair_archive)
        monkeypatch.setattr(module.distogram, "forward", record_offloaded_distogram)
        module.pde.projection_chunk_policy = pde_policy.replace(min_size=1)
        assert module._should_stream_pair_heads_to_cpu(output["zij_trunk"])
        assert module._should_stream_pair_heads_to_cpu(output["zij_trunk"].cpu())
        offloaded = module(batch, si_input, output)
        module.pde.projection_chunk_policy = pde_policy

    assert streamed_events == ["pair_heads", "distogram"]
    assert offloaded_events == ["empty_cache", "pair_heads", "distogram"]
    assert streamed_pair_output_alias == {}
    assert set(offloaded_pair_outputs) == {"pae_logits", "pde_logits"}
    for value in offloaded_pair_outputs.values():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
    assert expected.keys() == streamed.keys() == offloaded.keys()
    for name, reference in expected.items():
        assert reference.device == device
        assert streamed[name].device == device
        expected_device = torch.device("cpu") if name in ("pae_logits", "pde_logits") else device
        assert offloaded[name].device == expected_device
        assert streamed[name].dtype == offloaded[name].dtype == reference.dtype == torch.float32
        assert torch.isfinite(streamed[name]).all()
        assert torch.isfinite(offloaded[name]).all()
        torch.testing.assert_close(streamed[name], reference, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(offloaded[name].cpu(), streamed[name].cpu(), atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(streamed["distogram_logits"], expected["distogram_logits"], atol=0, rtol=0)
    for name, original in original_output.items():
        assert torch.equal(output[name], original)

    capture_pairformer_flags: list[bool] = []

    def record_capture_pairformer(*args, **kwargs):
        capture_pairformer_flags.append(kwargs["offload_to_cpu"])
        return original_pairformer(*args, **kwargs)

    monkeypatch.setattr(module.pairformer_embedding, "forward", record_capture_pairformer)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.inference_mode():
        captured = module(batch, si_input, output)

    assert capture_pairformer_flags == [False]
    assert captured.keys() == expected.keys()
    for name, reference in expected.items():
        assert captured[name].device == device
        torch.testing.assert_close(captured[name], reference, atol=3e-3, rtol=3e-3)


def test_confidence_pairformer_uses_scoped_triangle_attention_chunks() -> None:
    pairformer_config = PairformerConfig(
        token_s=8,
        token_z=8,
        pairwise_head_width=2,
        pairwise_num_heads=4,
        num_blocks=2,
        num_heads=2,
        dtype="bfloat16",
        skip_create_weights=True,
    )
    module = PairformerEmbedding(
        pairformer=pairformer_config,
        c_s_input=7,
        c_z=8,
        min_bin=3.25,
        max_bin=50.75,
        no_bin=3,
        inf=1e9,
        dtype=torch.bfloat16,
        skip_create_weights=True,
    )

    policy = CHUNK_REGISTRY[CONFIDENCE_TRIANGLE_ATTENTION]
    assert policy.enabled is True
    assert policy.chunk_size == 512
    for layer in module.pairformer_stack.layers:
        assert layer.tri_attn_start.chunk_policy is policy
        assert layer.tri_attn_end.chunk_policy is policy


def _make_pair_embedding(dtype: torch.dtype) -> PairformerEmbedding:
    pairformer_config = PairformerConfig(
        token_s=8,
        token_z=8,
        pairwise_head_width=2,
        pairwise_num_heads=4,
        num_blocks=1,
        num_heads=2,
        dtype="bfloat16",
        skip_create_weights=True,
    )
    return PairformerEmbedding(
        pairformer=pairformer_config,
        c_s_input=6,
        c_z=8,
        min_bin=3.25,
        max_bin=20.75,
        no_bin=5,
        inf=1e9,
        dtype=dtype,
    )


def test_confidence_pair_embedding_chunks_rows_with_random_weights() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size, samples, tokens = 2, 3, 8
    dtype = torch.bfloat16
    module = _make_pair_embedding(dtype).to(device).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    si_input = torch.randn(batch_size, 1, tokens, 6, dtype=dtype, device=device)
    zij = torch.randn(batch_size, 1, tokens, tokens, 8, dtype=dtype, device=device)
    x_pred = torch.randn(batch_size, samples, tokens, 3, dtype=dtype, device=device) * 5
    originals = tuple(value.clone() for value in (si_input, zij, x_pred))
    policy = CHUNK_REGISTRY[CONFIDENCE_PAIR_EMBEDDING]

    module.pair_embedding_chunk_policy = policy.replace(enabled=False)
    with torch.inference_mode():
        expected = module.embed_zij(si_input, zij, x_pred)

    seen_rows: list[int] = []

    def record_rows(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_rows.append(inputs[0].shape[-3])

    handle = module.linear_distance.register_forward_pre_hook(record_rows)
    module.pair_embedding_chunk_policy = policy.replace(chunk_size=3, min_size=1)
    try:
        with torch.inference_mode():
            actual = module.embed_zij(si_input, zij, x_pred)
    finally:
        handle.remove()

    assert seen_rows == [3, 3, 2]
    assert actual.shape == expected.shape == (batch_size, samples, tokens, tokens, 8)
    assert actual.dtype == expected.dtype == dtype
    assert actual.data_ptr() != zij.data_ptr()
    assert torch.count_nonzero(expected).item() > expected.numel() // 2
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    for value, original in zip((si_input, zij, x_pred), originals, strict=True):
        assert torch.equal(value, original)


def test_per_sample_pairformer_cpu_staging_preserves_output() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size, samples, tokens = 1, 3, 5
    module = _FakePairformerEmbedding().to(device)
    si_input = torch.randn(batch_size, tokens, 7, device=device)
    si = torch.randn(batch_size, tokens, 6, device=device)
    zij = torch.randn(batch_size, tokens, tokens, 8, device=device)
    x_pred = torch.randn(batch_size, samples, tokens, 3, device=device)
    single_mask = torch.ones(batch_size, samples, tokens, device=device)
    pair_mask = torch.ones(batch_size, tokens, tokens, device=device)

    expected = module.per_sample_pairformer_emb(si_input, si, zij, x_pred, single_mask, pair_mask)
    actual = module.per_sample_pairformer_emb(
        si_input,
        si,
        zij,
        x_pred,
        single_mask,
        pair_mask,
        offload_to_cpu=True,
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.device == device
        assert actual_tensor.dtype == expected_tensor.dtype
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


@pytest.mark.parametrize("head_type", [PredictedAlignedErrorHead, PredictedDistanceErrorHead])
def test_pair_error_head_projection_chunks_token_rows(head_type: type[nn.Module]) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 16
    head = head_type(c_z=8, c_out=6, dtype=torch.bfloat16).to(device)
    replace_with_fused_layernorm(head)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    zij = torch.randn(1, 3, tokens, tokens, 8, dtype=torch.bfloat16, device=device)
    policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)
    assert policy is not None
    assert policy.dim == 2

    head.projection_chunk_policy = policy.replace(enabled=False)
    with torch.inference_mode():
        expected = head(zij)

    seen_rows: list[int] = []

    def record_rows(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_rows.append(inputs[0].shape[-3])

    handle = head.layer_norm.register_forward_pre_hook(record_rows)
    head.projection_chunk_policy = policy.replace(chunk_size=7, min_size=1)
    try:
        with torch.inference_mode():
            actual = head(zij)
    finally:
        handle.remove()

    assert seen_rows == [7, 7, 2]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


class _ProjectionLifetimeProbe(nn.Module):
    def __init__(self, c_out: int) -> None:
        super().__init__()
        self.c_out = c_out
        self.output_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        assert all(reference() is None for reference in self.output_refs)
        output = value[..., : self.c_out].clone()
        self.output_refs.append(weakref.ref(output))
        return output


def test_pair_logit_projection_releases_copied_rows_before_next_call() -> None:
    zij = torch.randn(1, 8, 8, 4)
    linear = _ProjectionLifetimeProbe(c_out=2)
    with torch.inference_mode():
        actual = _project_pair_logits(
            zij,
            nn.Identity(),
            linear,
            c_out=2,
            policy=ChunkPolicy(chunk_size=3, min_size=1),
        )

    assert len(linear.output_refs) == 3
    assert torch.equal(actual, zij[..., :2])


def test_distogram_projection_releases_copied_rows_before_next_call() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    head = DistogramHead(c_z=4, c_out=2, dtype=torch.bfloat16).to(device)
    linear = _ProjectionLifetimeProbe(c_out=2).to(device)
    head.linear = linear
    head.projection_chunk_policy = ChunkPolicy(chunk_size=3, min_size=1)
    z = torch.randn(1, 8, 8, 4, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        actual = head(z)

    assert len(linear.output_refs) == 3
    expected = z[..., :2] + z[..., :2].transpose(-2, -3)
    assert torch.equal(actual, expected)


def test_pde_head_uses_dense_projection_during_cuda_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(47)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 16
    head = PredictedDistanceErrorHead(c_z=8, c_out=6, dtype=torch.bfloat16).to(device)
    replace_with_fused_layernorm(head)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    zij = torch.randn(1, 3, tokens, tokens, 8, dtype=torch.bfloat16, device=device)
    policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)
    assert policy is not None

    head.projection_chunk_policy = policy.replace(enabled=False)
    with torch.inference_mode():
        expected = head(zij)

    seen_rows: list[int] = []

    def record_rows(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_rows.append(inputs[0].shape[-3])

    handle = head.layer_norm.register_forward_pre_hook(record_rows)
    head.projection_chunk_policy = policy.replace(chunk_size=7, min_size=1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    try:
        with torch.inference_mode():
            actual = head(zij)
    finally:
        handle.remove()

    assert seen_rows == [tokens]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_distogram_head_chunks_projection_and_symmetry_with_random_weights() -> None:
    torch.manual_seed(59)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 16
    head = DistogramHead(c_z=8, c_out=6, dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    z = torch.randn(2, tokens, tokens, 8, dtype=torch.bfloat16, device=device)
    original = z.clone()
    checkpoint_keys = set(head.state_dict())
    policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)
    assert policy is not None

    head.projection_chunk_policy = policy.replace(enabled=False)
    with torch.inference_mode():
        expected = head(z)

    seen_rows: list[int] = []

    def record_rows(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_rows.append(inputs[0].shape[-3])

    handle = head.linear.register_forward_pre_hook(record_rows)
    head.projection_chunk_policy = policy.replace(chunk_size=7, min_size=1)
    try:
        with torch.inference_mode():
            actual = head(z)
    finally:
        handle.remove()

    assert seen_rows == [7, 7, 2]
    assert set(head.state_dict()) == checkpoint_keys
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, actual.transpose(-2, -3))
    assert torch.count_nonzero(expected).item() > expected.numel() // 2
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    assert torch.equal(z, original)


def test_pair_logits_block_symmetry_is_exact_and_in_place() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    logits = torch.randn(2, 3, 16, 16, 6, dtype=torch.bfloat16, device=device)
    expected = logits + logits.transpose(-2, -3)
    actual = logits.clone()

    result = _symmetrize_pair_logits_(actual, block_size=7)

    assert result is actual
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("mode", "expected_cache_calls"),
    [("inference", 2), ("capture", 0)],
)
def test_finalize_aux_outputs_cache_reclaim_guards(
    mode: str,
    expected_cache_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(53)
    device = torch.device("cuda", torch.cuda.current_device())
    aux_out = {
        "distogram_logits": torch.randn(1, 4, 4, 3, dtype=torch.bfloat16, device=device),
        "plddt_logits": torch.randn(1, 7, 5, dtype=torch.bfloat16, device=device),
        "pae_logits": torch.randn(1, 3, 4, 4, 6, dtype=torch.bfloat16, device=device),
        "pde_logits": torch.randn(1, 3, 4, 4, 6, dtype=torch.bfloat16, device=device),
    }
    expected = {name: value.detach().float() for name, value in aux_out.items()}
    empty_cache_calls = 0

    def record_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    if mode == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def finalize() -> dict[str, torch.Tensor]:
        return _finalize_aux_outputs_(
            aux_out,
            out_device=device,
            out_dtype=torch.float32,
            keep_pair_logits_on_cpu=False,
            reclaim_cuda_cache=True,
        )

    with torch.inference_mode():
        actual = finalize()

    assert actual is aux_out
    assert empty_cache_calls == expected_cache_calls
    for name in expected:
        assert actual[name].device == device
        assert actual[name].dtype == torch.float32
        assert torch.equal(actual[name], expected[name])
