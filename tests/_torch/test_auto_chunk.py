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

import types
from unittest import mock

import pytest
import torch

# isort: off
from tensorrt_bionemo._torch.auto_chunk import (
    _AUTOCHUNK_MIN_FLOOR, AUTOCHUNK_MIN_AUTO, CHUNK_REGISTRY,
    DEFAULT_AUTOCHUNK_MIN_REF, DEFAULT_MSA_AUTOCHUNK_MIN,
    DEFAULT_MSA_CHUNK_ROWS, DEFAULT_PAIR_CHUNK_ROWS, DIFFUSION_PAIR_TRANSITION,
    MSA_TRANSITION, OUTER_PRODUCT_MEAN, PAIR_TRANSITION,
    PAIR_WEIGHTED_AVERAGING, TRIANGLE_ATTENTION, ChunkPolicy, ChunkRegistry,
    chunk_apply, default_autochunk_min, iter_chunks)
# isort: on


def _force(chunk_size, dim=0, min_rank=0):
    """A policy that always chunks small CPU tensors (explicit ``min_size=0``)."""
    return ChunkPolicy(chunk_size=chunk_size,
                       min_size=0,
                       dim=dim,
                       min_rank=min_rank)


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,chunk,expected",
    [
        (10, 4, [(0, 4), (4, 4), (8, 2)]),  # short final tail
        (8, 4, [(0, 4), (4, 4)]),  # exact multiple
        (3, 10, [(0, 3)]),  # chunk larger than total
        (0, 4, []),  # empty
        (1, 1, [(0, 1)]),
        (5, 1, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
    ])
def test_iter_chunks(total, chunk, expected):
    assert list(iter_chunks(total, chunk)) == expected


def test_iter_chunks_is_a_contiguous_cover():
    total, chunk = 23, 5
    spans = list(iter_chunks(total, chunk))
    assert sum(length for _, length in spans) == total
    cursor = 0
    for start, length in spans:
        assert start == cursor  # no gaps / overlaps
        assert 0 < length <= chunk
        cursor += length
    assert cursor == total


# ---------------------------------------------------------------------------
# ChunkPolicy
# ---------------------------------------------------------------------------


def test_policy_resolved_min_size_explicit():
    assert ChunkPolicy(min_size=1234).resolved_min_size() == 1234
    assert ChunkPolicy(min_size=0).resolved_min_size() == 0


def test_policy_resolved_min_size_auto_delegates_to_default():
    # AUTOCHUNK_MIN_AUTO defers to default_autochunk_min() (both read the same cached value).
    assert ChunkPolicy(min_size=AUTOCHUNK_MIN_AUTO).resolved_min_size(
    ) == default_autochunk_min()


def test_policy_should_chunk_size():
    p = ChunkPolicy(chunk_size=4, min_size=10)
    assert p.should_chunk_size(11)
    assert not p.should_chunk_size(10)  # strictly greater only
    assert not p.should_chunk_size(9)
    assert not p.replace(enabled=False).should_chunk_size(1000)
    assert not p.replace(chunk_size=0).should_chunk_size(1000)


def test_policy_should_chunk_tensor_gates():
    p = ChunkPolicy(chunk_size=2, min_size=3, dim=1, min_rank=4)
    big = torch.zeros(1, 5, 5, 2)  # rank 4, dim=1 extent 5 > 3
    assert p.should_chunk(big)
    assert not p.should_chunk(torch.zeros(1, 3, 5, 2))  # extent 3, not > 3
    assert not p.should_chunk(torch.zeros(1, 5, 2))  # rank 3 < min_rank 4
    assert not p.replace(enabled=False).should_chunk(big)
    assert not p.replace(chunk_size=0).should_chunk(big)
    assert not p.should_chunk(None)  # non-tensor
    assert not p.should_chunk([1, 2, 3])


def test_policy_min_rank_zero_disables_rank_gate():
    p = ChunkPolicy(chunk_size=2, min_size=0, dim=0, min_rank=0)
    assert p.should_chunk(torch.zeros(5))  # rank-1 tensor still chunks


def test_policy_replace_is_pure():
    p = ChunkPolicy(chunk_size=4, min_size=10, dim=1)
    p2 = p.replace(chunk_size=8)
    assert (p2.chunk_size, p2.min_size, p2.dim) == (8, 10, 1)
    assert p.chunk_size == 4  # frozen original unchanged


# ---------------------------------------------------------------------------
# ChunkRegistry: register / get / set / enable / disable
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    reg = ChunkRegistry()
    pol = ChunkPolicy(chunk_size=7)
    assert reg.register("a", pol) is pol
    assert reg.get("a") is pol
    assert "a" in reg
    assert reg["a"] is pol
    assert reg.names() == ["a"]
    assert reg.get("missing") is None
    sentinel = ChunkPolicy(chunk_size=1)
    assert reg.get("missing", sentinel) is sentinel
    assert "missing" not in reg


def test_registry_register_overwrite_flag():
    reg = ChunkRegistry()
    p1, p2 = ChunkPolicy(chunk_size=1), ChunkPolicy(chunk_size=2)
    reg.register("a", p1)
    reg.register("a", p2)  # default: overwrite
    assert reg.get("a") is p2
    reg.register("a", p1, overwrite=False)  # keep existing
    assert reg.get("a") is p2


def test_registry_set_overrides_selected_fields():
    reg = ChunkRegistry()
    reg.register("a", ChunkPolicy(chunk_size=4, min_size=10, dim=1,
                                  min_rank=4))
    new = reg.set("a", min_size=99)
    assert new.min_size == 99  # overridden
    assert (new.chunk_size, new.dim, new.min_rank) == (4, 1, 4)  # preserved
    assert reg.get("a") is new  # stored in place


def test_registry_set_creates_from_defaults_when_absent():
    reg = ChunkRegistry()
    new = reg.set("brand_new", chunk_size=13)
    assert new.chunk_size == 13
    # remaining fields fall back to ChunkPolicy() defaults.
    assert new.min_size == AUTOCHUNK_MIN_AUTO
    assert new.dim == 1 and new.min_rank == 4 and new.enabled is True
    assert reg.get("brand_new") is new


def test_registry_enable_disable():
    reg = ChunkRegistry()
    reg.register("a", ChunkPolicy(chunk_size=9,
                                  min_size=5,
                                  dim=2,
                                  enabled=True))
    disabled = reg.disable("a")
    assert disabled.enabled is False
    assert reg.get("a").enabled is False
    # disable/enable must preserve the other fields.
    assert (disabled.chunk_size, disabled.min_size, disabled.dim) == (9, 5, 2)
    enabled = reg.enable("a")
    assert enabled.enabled is True
    assert reg.get("a").enabled is True


def test_global_registry_builtin_defaults():
    # Read-only assertions on the shared registry (do NOT mutate it here).
    for name in (PAIR_TRANSITION, DIFFUSION_PAIR_TRANSITION, MSA_TRANSITION,
                 PAIR_WEIGHTED_AVERAGING, OUTER_PRODUCT_MEAN,
                 TRIANGLE_ATTENTION):
        assert name in CHUNK_REGISTRY
    assert CHUNK_REGISTRY.get(
        PAIR_TRANSITION).chunk_size == DEFAULT_PAIR_CHUNK_ROWS
    assert CHUNK_REGISTRY.get(
        DIFFUSION_PAIR_TRANSITION).chunk_size == DEFAULT_PAIR_CHUNK_ROWS
    assert CHUNK_REGISTRY.get(
        MSA_TRANSITION).chunk_size == DEFAULT_MSA_CHUNK_ROWS
    assert CHUNK_REGISTRY.get(
        MSA_TRANSITION).min_size == DEFAULT_MSA_AUTOCHUNK_MIN
    assert CHUNK_REGISTRY.get(OUTER_PRODUCT_MEAN).chunk_size == 128
    # Triangle attention is off by default (flash kernels already bound memory).
    assert CHUNK_REGISTRY.get(TRIANGLE_ATTENTION).enabled is False


# ---------------------------------------------------------------------------
# default_autochunk_min: memory scaling (GPU query mocked, cache guarded)
# ---------------------------------------------------------------------------


@pytest.fixture
def autochunk_cache_guard():
    """Clear the lru_cache around the test so mocked device memory can't leak in/out."""
    default_autochunk_min.cache_clear()
    yield
    default_autochunk_min.cache_clear()


def _autochunk_min_for_gpu(total_gb):
    """Resolve default_autochunk_min() as if the current GPU had ``total_gb`` of memory."""
    default_autochunk_min.cache_clear()
    props = types.SimpleNamespace(total_memory=int(total_gb * (1024**3)))
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.current_device", return_value=0), \
         mock.patch("torch.cuda.get_device_properties", return_value=props):
        return default_autochunk_min()


@pytest.mark.parametrize(
    "total_gb,expected",
    [
        (80.0, 2560),  # reference GPU -> the anchor
        (320.0, 5120),  # 4x memory -> sqrt(4)=2x threshold
        (20.0, 1280),  # 1/4 memory -> 1/2 threshold
        (100.0, 2816),  # sqrt(1.25)*2560 = 2862.5 -> round to 128-multiple
        (1.0, _AUTOCHUNK_MIN_FLOOR),  # tiny GPU -> clamped at the floor
    ])
def test_default_autochunk_min_scaling(total_gb, expected,
                                       autochunk_cache_guard):
    assert _autochunk_min_for_gpu(total_gb) == expected


@pytest.mark.parametrize("total_gb", [8.0, 16.0, 24.0, 48.0, 141.0, 640.0])
def test_default_autochunk_min_is_floored_128_multiple(total_gb,
                                                       autochunk_cache_guard):
    val = _autochunk_min_for_gpu(total_gb)
    assert val >= _AUTOCHUNK_MIN_FLOOR
    assert val % 128 == 0


def test_default_autochunk_min_no_cuda_returns_reference(
        autochunk_cache_guard):
    default_autochunk_min.cache_clear()
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert default_autochunk_min() == DEFAULT_AUTOCHUNK_MIN_REF


def test_default_autochunk_min_falls_back_on_error(autochunk_cache_guard):
    default_autochunk_min.cache_clear()
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.current_device", return_value=0), \
         mock.patch("torch.cuda.get_device_properties",
                    side_effect=RuntimeError("no device")):
        assert default_autochunk_min() == DEFAULT_AUTOCHUNK_MIN_REF


# ---------------------------------------------------------------------------
# chunk_apply
# ---------------------------------------------------------------------------


def test_chunk_apply_single_tensor_matches_dense():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    fn = lambda t: t * 2 + 1
    assert torch.equal(chunk_apply(fn, x, policy=_force(3)), fn(x))


def test_chunk_apply_rank4_dim1_matches_dense():
    # Real pair-activation shape [B, N, N, C]; default policy dim=1, min_rank=4.
    x = torch.randn(1, 10, 6, 4)
    fn = lambda t: t.sin() * 3 - 0.5
    out = chunk_apply(fn, x, policy=ChunkPolicy(chunk_size=4, min_size=0))
    assert torch.equal(out, fn(x))  # position-wise -> bit-identical


def test_chunk_apply_slices_into_expected_chunks():
    x = torch.zeros(10, 2)
    seen = []
    chunk_apply(lambda t: (seen.append(t.shape[0]) or t), x, policy=_force(3))
    assert seen == [3, 3, 3, 1]


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        dict(enabled=False),  # master switch off
        dict(min_size=100),  # below threshold
        dict(chunk_size=0),  # chunking disabled
    ])
def test_chunk_apply_dense_fast_path_single_call(policy_kwargs):
    x = torch.zeros(10, 2)
    calls = []
    policy = _force(3).replace(**policy_kwargs)
    out = chunk_apply(lambda t: (calls.append(1) or t * 2), x, policy=policy)
    assert len(calls) == 1  # one dense call, not a loop
    assert torch.equal(out, x * 2)


def test_chunk_apply_tuple_output():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out = chunk_apply(lambda t: (t * 2, t + 1), x, policy=_force(3))
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], x * 2)
    assert torch.equal(out[1], x + 1)


def test_chunk_apply_list_output():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out = chunk_apply(lambda t: [t * 2, t - 1], x, policy=_force(3))
    assert isinstance(out, list) and len(out) == 2
    assert torch.equal(out[0], x * 2)
    assert torch.equal(out[1], x - 1)


def test_chunk_apply_misaligned_tensor_is_passed_through_whole():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)  # dim-0 extent 10
    bias = torch.tensor([100.0, 200.0])  # extent 2 on dim 0 -> not aligned
    seen = []

    def fn(t, b):
        seen.append(tuple(b.shape))
        return t + b  # broadcasts [chunk, 2] + [2]

    out = chunk_apply(fn, x, bias, policy=_force(3))
    assert torch.equal(out, x + bias)
    # the misaligned bias is forwarded unsliced to every chunk call.
    assert seen == [(2, ), (2, ), (2, ), (2, )]


def test_chunk_apply_none_is_passed_through():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    seen = []

    def fn(t, extra):
        seen.append(extra)
        return t

    out = chunk_apply(fn, x, None, policy=_force(3))
    assert torch.equal(out, x)
    assert seen == [None, None, None, None]


def test_chunk_apply_kwargs_are_passed_through():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out = chunk_apply(lambda t, *, scale: t * scale,
                      x,
                      policy=_force(3),
                      scale=5)
    assert torch.equal(out, x * 5)


def test_chunk_apply_cat_dim_override():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    # slice rows on dim 0, but each fn output moves them to dim 1 -> concat there.
    out = chunk_apply(lambda t: t.t(), x, policy=_force(3, dim=0), cat_dim=1)
    assert torch.equal(out, x.t())


def test_chunk_apply_cat_dim_override_tuple():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out = chunk_apply(lambda t: (t.t(), (t * 2).t()),
                      x,
                      policy=_force(3, dim=0),
                      cat_dim=1)
    assert torch.equal(out[0], x.t())
    assert torch.equal(out[1], (x * 2).t())


def test_chunk_apply_default_policy_below_threshold_is_dense():
    # policy=None -> registry PAIR_TRANSITION; its (memory-scaled) threshold isn't tripped by a
    # tiny CPU tensor, so it takes the single dense call.
    x = torch.randn(1, 8, 8, 4)
    calls = []
    out = chunk_apply(lambda t: (calls.append(1) or t + 1), x)
    assert torch.equal(out, x + 1)
    assert len(calls) == 1


def test_chunk_apply_no_chunked_tensors_is_dense():
    assert chunk_apply(lambda: 42) == 42
