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

from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_bionemo._torch.custom_ops import dispatcher
from tensorrt_bionemo._torch.custom_ops.base import (IGNORE_BENCHMARK_SCORE,
                                                     MAX_BENCHMARK_SCORE,
                                                     MID_BENCHMARK_SCORE)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the dispatcher LRU cache before every test."""
    dispatcher._IMPL_CACHE.clear()
    yield
    dispatcher._IMPL_CACHE.clear()


def _make_fake_impl(name="FakeImpl", score=MAX_BENCHMARK_SCORE):
    impl = MagicMock()
    impl.__name__ = name
    impl.is_supported.return_value = True
    impl.get_benchmark_score.return_value = score
    impl.apply.return_value = torch.zeros(1)
    return impl


def _make_cpu_tensor(*shape, dtype=torch.float16):
    return torch.randn(*shape, dtype=dtype)


class TestCacheHitAndMiss:

    def test_second_call_same_args_is_cache_hit(self):
        fake = _make_fake_impl()
        x = _make_cpu_tensor(4, 128)
        w1 = _make_cpu_tensor(256, 128)
        w2 = _make_cpu_tensor(256, 128)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            r1 = dispatcher.get_custom_ops_impl(
                "fused_sigmoid_gated_dual_gemm", x, w1, w2)
            r2 = dispatcher.get_custom_ops_impl(
                "fused_sigmoid_gated_dual_gemm", x, w1, w2)

        assert r1 is not None and r2 is not None
        assert mock_find.call_count == 1

    def test_different_shapes_cause_cache_miss(self):
        fake = _make_fake_impl()
        w1 = _make_cpu_tensor(256, 128)
        w2 = _make_cpu_tensor(256, 128)
        x_a = _make_cpu_tensor(4, 128)
        x_b = _make_cpu_tensor(8, 128)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            dispatcher.get_custom_ops_impl("fused_sigmoid_gated_dual_gemm",
                                           x_a, w1, w2)
            dispatcher.get_custom_ops_impl("fused_sigmoid_gated_dual_gemm",
                                           x_b, w1, w2)

        assert mock_find.call_count == 2

    def test_different_dtypes_cause_cache_miss(self):
        fake = _make_fake_impl()
        w1_f16 = _make_cpu_tensor(256, 128, dtype=torch.float16)
        w2_f16 = _make_cpu_tensor(256, 128, dtype=torch.float16)
        w1_bf16 = _make_cpu_tensor(256, 128, dtype=torch.bfloat16)
        w2_bf16 = _make_cpu_tensor(256, 128, dtype=torch.bfloat16)
        x_f16 = _make_cpu_tensor(4, 128, dtype=torch.float16)
        x_bf16 = _make_cpu_tensor(4, 128, dtype=torch.bfloat16)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            dispatcher.get_custom_ops_impl("fused_sigmoid_gated_dual_gemm",
                                           x_f16, w1_f16, w2_f16)
            dispatcher.get_custom_ops_impl("fused_sigmoid_gated_dual_gemm",
                                           x_bf16, w1_bf16, w2_bf16)

        assert mock_find.call_count == 2

    def test_different_ops_names_cause_cache_miss(self):
        fake = _make_fake_impl()
        x = _make_cpu_tensor(4, 128)
        w1 = _make_cpu_tensor(256, 128)
        w2 = _make_cpu_tensor(256, 128)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            dispatcher.get_custom_ops_impl("fused_sigmoid_gated_dual_gemm", x,
                                           w1, w2)
            dispatcher.get_custom_ops_impl(
                "fused_sigmoid_gated_dual_gemm_dual_x", x, w1, w2)

        assert mock_find.call_count == 2


class TestCacheNoneResult:

    def test_none_result_is_cached(self):
        x = _make_cpu_tensor(4, 128)
        w1 = _make_cpu_tensor(256, 128)
        w2 = _make_cpu_tensor(256, 128)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=None) as mock_find:
            r1 = dispatcher.get_custom_ops_impl(
                "fused_sigmoid_gated_dual_gemm", x, w1, w2)
            r2 = dispatcher.get_custom_ops_impl(
                "fused_sigmoid_gated_dual_gemm", x, w1, w2)

        assert r1 is None and r2 is None
        assert mock_find.call_count == 1


class TestLRUEviction:

    def test_eviction_when_exceeding_max_size(self):
        fake = _make_fake_impl()
        original_max = dispatcher._CACHE_MAX_SIZE

        try:
            dispatcher._CACHE_MAX_SIZE = 3

            with patch.object(dispatcher, "_find_best_impl",
                              return_value=fake) as mock_find:
                for i in range(4):
                    x = _make_cpu_tensor(i + 1, 128)
                    w = _make_cpu_tensor(256, 128)
                    dispatcher.get_custom_ops_impl("op", x, w)

                assert mock_find.call_count == 4
                assert len(dispatcher._IMPL_CACHE) == 3

                # Re-call with the first shape → should be evicted, causing a miss
                x0 = _make_cpu_tensor(1, 128)
                w = _make_cpu_tensor(256, 128)
                dispatcher.get_custom_ops_impl("op", x0, w)
                assert mock_find.call_count == 5
        finally:
            dispatcher._CACHE_MAX_SIZE = original_max

    def test_lru_touch_prevents_eviction(self):
        fake = _make_fake_impl()
        original_max = dispatcher._CACHE_MAX_SIZE

        try:
            dispatcher._CACHE_MAX_SIZE = 3

            with patch.object(dispatcher, "_find_best_impl",
                              return_value=fake) as mock_find:
                ws = [_make_cpu_tensor(256, 128)] * 2
                xs = [_make_cpu_tensor(i + 1, 128) for i in range(3)]
                for x in xs:
                    dispatcher.get_custom_ops_impl("op", x, *ws)
                assert mock_find.call_count == 3

                # Touch the oldest entry (shape 1,128) so it becomes most-recent
                dispatcher.get_custom_ops_impl("op", xs[0], *ws)
                assert mock_find.call_count == 3  # still a hit

                # Now insert a new entry → should evict xs[1] (the actual LRU), not xs[0]
                x_new = _make_cpu_tensor(99, 128)
                dispatcher.get_custom_ops_impl("op", x_new, *ws)
                assert mock_find.call_count == 4
                assert len(dispatcher._IMPL_CACHE) == 3

                # xs[0] should still be cached (was touched), xs[1] should be evicted
                dispatcher.get_custom_ops_impl("op", xs[0], *ws)
                assert mock_find.call_count == 4  # hit

                dispatcher.get_custom_ops_impl("op", xs[1], *ws)
                assert mock_find.call_count == 5  # miss — was evicted
        finally:
            dispatcher._CACHE_MAX_SIZE = original_max


class TestKwargsHandling:

    def test_kwargs_produce_same_cache_key_regardless_of_order(self):
        fake = _make_fake_impl()
        x = _make_cpu_tensor(4, 128)
        b1 = _make_cpu_tensor(256)
        b2 = _make_cpu_tensor(256)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            dispatcher.get_custom_ops_impl("op", x, b1=b1, b2=b2)
            dispatcher.get_custom_ops_impl("op", x, b2=b2, b1=b1)

        assert mock_find.call_count == 1

    def test_none_kwarg_vs_absent_kwarg_are_different(self):
        fake = _make_fake_impl()
        x = _make_cpu_tensor(4, 128)

        with patch.object(dispatcher, "_find_best_impl",
                          return_value=fake) as mock_find:
            dispatcher.get_custom_ops_impl("op", x, mask=None)
            dispatcher.get_custom_ops_impl("op", x)

        assert mock_find.call_count == 2


class TestBestImplSelection:
    """Test _find_best_impl without mocking it — mock the impls instead."""

    def test_picks_highest_benchmark_score(self):
        slow = _make_fake_impl("slow", score=MID_BENCHMARK_SCORE)
        fast = _make_fake_impl("fast", score=MAX_BENCHMARK_SCORE)
        x = _make_cpu_tensor(4, 128)

        with patch.dict(dispatcher._OPS_IMPLS_MAP, {"op": [slow, fast]}):
            result = dispatcher._find_best_impl("op", x)

        assert result is fast

    def test_skips_unsupported(self):
        unsupported = _make_fake_impl("unsupported", score=MAX_BENCHMARK_SCORE)
        unsupported.is_supported.return_value = False
        supported = _make_fake_impl("supported", score=MID_BENCHMARK_SCORE)
        x = _make_cpu_tensor(4, 128)

        with patch.dict(dispatcher._OPS_IMPLS_MAP,
                        {"op": [unsupported, supported]}):
            result = dispatcher._find_best_impl("op", x)

        assert result is supported

    def test_skips_ignore_score(self):
        ignored = _make_fake_impl("ignored", score=IGNORE_BENCHMARK_SCORE)
        ok = _make_fake_impl("ok", score=MID_BENCHMARK_SCORE)
        x = _make_cpu_tensor(4, 128)

        with patch.dict(dispatcher._OPS_IMPLS_MAP, {"op": [ignored, ok]}):
            result = dispatcher._find_best_impl("op", x)

        assert result is ok

    def test_returns_none_when_all_unsupported(self):
        impl = _make_fake_impl()
        impl.is_supported.return_value = False
        x = _make_cpu_tensor(4, 128)

        with patch.dict(dispatcher._OPS_IMPLS_MAP, {"op": [impl]}):
            result = dispatcher._find_best_impl("op", x)

        assert result is None

    def test_returns_none_for_unknown_op(self):
        result = dispatcher._find_best_impl("nonexistent_op",
                                            _make_cpu_tensor(4, 128))
        assert result is None
