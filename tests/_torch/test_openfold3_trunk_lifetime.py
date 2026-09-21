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
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.layers.transformers.pairformer import PairformerLayerV1
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir.models.openfold3.modeling import OpenFold3


class _ScaledPairUpdate(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return value * self.scale


def _make_pair_residual_probe() -> PairformerLayerV1:
    layer = PairformerLayerV1.__new__(PairformerLayerV1)
    torch.nn.Module.__init__(layer)
    layer.dtype = torch.float32
    layer.pair_mask_left_aligned = False
    layer.tri_mul_out = _ScaledPairUpdate(1)
    layer.tri_mul_in = _ScaledPairUpdate(2)
    layer.tri_attn_start = _ScaledPairUpdate(3)
    layer.tri_attn_end = _ScaledPairUpdate(4)
    layer.transition_z = _ScaledPairUpdate(5)
    return layer


@pytest.mark.parametrize(
    "mode",
    [
        "default",
        "inference",
        pytest.param("capture", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
    ],
)
def test_pairformer_reuses_owned_pair_storage_only_when_safe(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda" if mode == "capture" else "cpu")
    z = torch.randn(2, 3, 3, 4, device=device)
    original = z.clone()
    pair_mask = torch.ones(2, 3, 3, device=device)
    layer = _make_pair_residual_probe().to(device)
    if mode == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.inference_mode():
        expected = layer._transform_z(original.clone(), pair_mask)

    def transform() -> torch.Tensor:
        return layer._transform_z(z, pair_mask, inplace_safe=mode != "default")

    with torch.inference_mode():
        actual = transform()

    assert torch.equal(actual, expected)
    if mode == "inference":
        assert actual.data_ptr() == z.data_ptr()
    else:
        assert actual.data_ptr() != z.data_ptr()
        assert torch.equal(z, original)


class _TrunkLifetimeProbe:
    forward = OpenFold3.forward

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.diffusion_module = SimpleNamespace(dtype=torch.float32)
        self.trunk_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.expected: tuple[torch.Tensor, ...] = ()

    def generate_attn_metadata(self, batch: dict[str, torch.Tensor]) -> None:
        return None

    def feature_extraction(
        self, batch: dict[str, torch.Tensor], num_cycles: int, attn_metadata: None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = tuple(
            torch.randn(shape, dtype=torch.bfloat16, device=self.device)
            for shape in ((1, 3, 4), (1, 3, 4), (1, 3, 3, 4))
        )
        self.trunk_refs = [weakref.ref(value) for value in values]
        self.expected = tuple(value.float().unsqueeze(1) for value in values)
        return values

    def prediction(
        self, si_input: torch.Tensor, si_trunk: torch.Tensor, zij_trunk: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        # Unsqueezed views keep their original tensor alive: this fails when
        # forward retains its bf16 locals while passing temporary fp32 casts.
        assert all(reference() is None for reference in self.trunk_refs)
        for actual, expected in zip((si_input, si_trunk, zij_trunk), self.expected, strict=True):
            assert actual.dtype == torch.float32
            assert torch.equal(actual, expected)
        return {"zij_trunk": zij_trunk}


def test_prediction_releases_superseded_trunk_precision() -> None:
    probe = _TrunkLifetimeProbe()
    output = probe.forward({"token_mask": torch.ones(1, 3)})
    assert torch.equal(output["zij_trunk"], probe.expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prediction_reclaims_dead_cuda_cache_before_inference(monkeypatch) -> None:
    events: list[str] = []
    probe = _TrunkLifetimeProbe("cuda")
    original_prediction = probe.prediction

    def record_prediction(**kwargs):
        events.append("prediction")
        return original_prediction(**kwargs)

    monkeypatch.setattr(probe, "prediction", record_prediction)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with torch.inference_mode():
        output = probe.forward({"token_mask": torch.ones(1, 3, device="cuda")})

    assert events == ["empty_cache", "prediction"]
    assert torch.equal(output["zij_trunk"], probe.expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prediction_keeps_cache_during_cuda_graph_capture(monkeypatch) -> None:
    probe = _TrunkLifetimeProbe("cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def reject_empty_cache() -> None:
        raise AssertionError("empty_cache is not capture safe")

    monkeypatch.setattr(torch.cuda, "empty_cache", reject_empty_cache)
    with torch.inference_mode():
        output = probe.forward({"token_mask": torch.ones(1, 3, device="cuda")})

    assert torch.equal(output["zij_trunk"], probe.expected[2])


class _PairInputProbe:
    def __init__(self, values: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, min_size: int) -> None:
        self.values = values
        self.pair_build_chunk_policy = ChunkPolicy(chunk_size=2, min_size=min_size)

    def __call__(self, **_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.values


class _RecycleUpdateProbe:
    _update_recycle_pair = OpenFold3._update_recycle_pair

    def __init__(self, device: torch.device, dtype: torch.dtype, *, min_size: int) -> None:
        channels = 8
        self.recycle_pair_update_chunk_policy = ChunkPolicy(chunk_size=3, min_size=min_size)
        self.layer_norm_z = torch.nn.LayerNorm(channels, device=device, dtype=dtype)
        self.linear_z = torch.nn.Linear(channels, channels, bias=False, device=device, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recycle_pair_update_chunks_owned_rows_match_dense(dtype: torch.dtype) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(42)
    probe = _RecycleUpdateProbe(device, dtype, min_size=1)
    z_init = torch.randn(1, 8, 8, 8, device=device, dtype=dtype)
    z = torch.randn_like(z_init)
    original_z_init = z_init.clone()

    with torch.inference_mode():
        expected = z_init + probe.linear_z(probe.layer_norm_z(z))

    normalized_rows: list[int] = []
    handle = probe.layer_norm_z.register_forward_pre_hook(
        lambda _module, inputs: normalized_rows.append(inputs[0].shape[-3])
    )
    original_ptr = z.data_ptr()
    try:
        with torch.inference_mode():
            actual = probe._update_recycle_pair(z_init, z)
    finally:
        handle.remove()

    assert normalized_rows == [3, 3, 2]
    assert actual.data_ptr() == original_ptr
    if dtype == torch.bfloat16:
        assert torch.equal(actual, expected)
    else:
        # Hopper can select different FP32 GEMM tilings for the dense and
        # row-chunked linear calls. Production BF16 remains bitwise exact.
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(z_init, original_z_init)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_recycle_pair_update_releases_copied_rows_before_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    probe = _RecycleUpdateProbe(device, torch.float32, min_size=1)
    z_init = torch.randn(1, 8, 8, 8, device=device)
    z = torch.randn_like(z_init)
    update_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def assert_previous_update_released(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        assert all(reference() is None for reference in update_refs)

    original_copy = torch.Tensor.copy_

    def record_copy(destination: torch.Tensor, source: torch.Tensor, *args, **kwargs):
        update_refs.append(weakref.ref(source))
        return original_copy(destination, source, *args, **kwargs)

    handle = probe.layer_norm_z.register_forward_pre_hook(assert_previous_update_released)
    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    try:
        with torch.inference_mode():
            probe._update_recycle_pair(z_init, z)
    finally:
        handle.remove()

    assert len(update_refs) == 3


@pytest.mark.parametrize(
    ("mode", "min_size"),
    [
        pytest.param("short", 16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
        pytest.param("capture", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
        ("cpu", 1),
    ],
)
def test_recycle_pair_update_keeps_dense_fallbacks(
    mode: str,
    min_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu" if mode == "cpu" else "cuda", torch.cuda.current_device() if mode != "cpu" else 0)
    probe = _RecycleUpdateProbe(device, torch.float32, min_size=min_size)
    z_init = torch.randn(1, 8, 8, 8, device=device)
    z = torch.randn_like(z_init)
    original = z.clone()
    if mode == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with torch.inference_mode():
        actual = probe._update_recycle_pair(z_init, z)

    assert actual.data_ptr() != z.data_ptr()
    assert torch.equal(z, original)


class _FeatureExtractionCacheProbe:
    feature_extraction = OpenFold3.feature_extraction
    _update_recycle_pair = OpenFold3._update_recycle_pair

    def __init__(self, device: torch.device, *, min_size: int = 1) -> None:
        tokens, channels = 4, 2
        self.config = SimpleNamespace(trunk=SimpleNamespace(pairformer=SimpleNamespace(torch_dtype=torch.float32)))
        input_values = (
            torch.randn(1, tokens, channels, device=device),
            torch.randn(1, tokens, channels, device=device),
            torch.randn(1, tokens, tokens, channels, device=device),
        )
        self.input_embedder = _PairInputProbe(input_values, min_size=min_size)
        self.layer_norm_z = lambda value: value
        self.linear_z = lambda value: value
        self.template_embedder = lambda **kwargs: torch.zeros_like(kwargs["z"])
        self.msa_module_embedder = lambda **_kwargs: (
            torch.zeros(1, 1, tokens, channels, device=device),
            torch.ones(1, 1, tokens, device=device),
        )
        self.msa_module = lambda _m, z, **_kwargs: z
        self.layer_norm_s = lambda value: value
        self.linear_s = lambda value: value
        self.pairformer_stack = lambda *, s, z, **_kwargs: (s, z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mode", "min_size", "expected_calls"),
    [
        ("inference", 1, 4),
        ("short", 8, 0),
        ("capture", 1, 0),
    ],
)
def test_row_built_input_reclaims_cache_between_recycles(
    mode: str,
    min_size: int,
    expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    probe = _FeatureExtractionCacheProbe(device, min_size=min_size)
    cache_calls = 0

    def record_empty_cache() -> None:
        nonlocal cache_calls
        cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode == "capture")
    with torch.inference_mode():
        probe.feature_extraction(
            batch={"token_mask": torch.ones(1, 4, device=device)},
            num_cycles=4,
        )

    assert cache_calls == expected_calls


class _PairCacheConditioning:
    def __init__(self, events: list[str], *, min_size: int) -> None:
        self.events = events
        self.pair_projection_chunk_policy = ChunkPolicy(chunk_size=2, min_size=min_size, dim=2, min_rank=5)

    def prepare_pair(self, *, batch: dict[str, torch.Tensor], zij_trunk: torch.Tensor) -> torch.Tensor:
        assert zij_trunk.device.type == "cpu"
        self.events.append("prepare_pair")
        return zij_trunk.to(device=batch["token_mask"].device)


class _PairCacheSampler:
    use_conditioning = True

    def __init__(self, events: list[str], expected_pair: torch.Tensor, *, expect_prepared: bool) -> None:
        self.events = events
        self.expected_pair = expected_pair
        self.expect_prepared = expect_prepared

    def __call__(
        self,
        *,
        batch: dict[str, torch.Tensor],
        zij_trunk: torch.Tensor,
        prepared_zij: torch.Tensor | None,
        **_kwargs,
    ) -> torch.Tensor:
        self.events.append("sample")
        if self.expect_prepared:
            assert prepared_zij is not None
            assert prepared_zij.data_ptr() == zij_trunk.data_ptr()
        else:
            assert prepared_zij is None
        assert zij_trunk.is_cuda
        assert torch.equal(zij_trunk, self.expected_pair)
        return torch.zeros((*batch["atom_mask"].shape, 3), device=zij_trunk.device)


class _PairCacheAuxHeads:
    def __init__(self, events: list[str], expected_pair: torch.Tensor) -> None:
        self.events = events
        self.expected_pair = expected_pair

    def __call__(self, *, output: dict[str, torch.Tensor], **_kwargs) -> dict[str, torch.Tensor]:
        self.events.append("aux_heads")
        assert output["zij_trunk"].is_cuda
        assert torch.equal(output["zij_trunk"], self.expected_pair)
        return {}


class _PairCachePredictionProbe:
    prediction = OpenFold3.prediction

    def __init__(self, events: list[str], expected_pair: torch.Tensor, *, min_size: int, expect_prepared: bool) -> None:
        conditioning = _PairCacheConditioning(events, min_size=min_size)
        self.diffusion_module = SimpleNamespace(diffusion_conditioning=conditioning)
        self.diffusion_sampler = _PairCacheSampler(events, expected_pair, expect_prepared=expect_prepared)
        self.aux_heads = _PairCacheAuxHeads(events, expected_pair)
        self.no_rollout_steps = 2
        self.no_rollout_samples = 1
        self.noise_schedule = SimpleNamespace(sigma_data=16.0, s_max=160.0, s_min=4e-4, p=7.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mode", "min_size", "expect_prepared", "expected_events"),
    [
        (
            "inference",
            1,
            True,
            ["empty_cache", "prepare_pair", "empty_cache", "sample", "empty_cache", "aux_heads"],
        ),
        ("short", 8, False, ["sample", "aux_heads"]),
        ("capture", 1, False, ["sample", "aux_heads"]),
    ],
)
def test_prediction_stages_trunk_pair_around_cached_rollout(
    mode: str,
    min_size: int,
    expect_prepared: bool,
    expected_events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    events: list[str] = []
    tokens, channels = 4, 2
    expected_pair = torch.randn(1, 1, tokens, tokens, channels, device=device)
    probe = _PairCachePredictionProbe(
        events,
        expected_pair,
        min_size=min_size,
        expect_prepared=expect_prepared,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode == "capture")
    batch = {
        "token_mask": torch.ones(1, 1, tokens, device=device),
        "atom_mask": torch.ones(1, 1, tokens, device=device),
    }
    with torch.inference_mode():
        output = probe.prediction(
            batch=batch,
            si_input=torch.randn(1, 1, tokens, channels, device=device),
            si_trunk=torch.randn(1, 1, tokens, channels, device=device),
            zij_trunk=expected_pair.clone(),
        )

    assert events == expected_events
    assert torch.equal(output["zij_trunk"], expected_pair)
