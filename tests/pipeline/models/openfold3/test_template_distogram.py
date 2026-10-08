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
"""Template distogram matches the dense reference for every input dtype."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3.common import create_template_distogram


def _reference(coords, pb_mask, pair_mask, min_bin, max_bin, n_bins, inf_value) -> torch.Tensor:
    coords = np.asarray(coords)
    distogram = np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1, keepdims=True)
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)], axis=-1)
    binned = torch.tensor(((distogram > lower) * (distogram < upper)).astype(distogram.dtype), dtype=torch.float32)
    pair = (pb_mask[..., None] * pb_mask[..., None, :])[..., None]
    return binned * pair * pair_mask


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16, np.int64])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.float64])
def test_matches_dense_reference(dtype: type, mask_dtype: torch.dtype) -> None:
    rng = np.random.default_rng(0)
    coords = (rng.normal(size=(2, 40, 3)) * rng.choice([0.1, 1.0, 10.0], size=(2, 40, 3))).astype(dtype)
    pb_mask = torch.from_numpy(rng.random((2, 40)) > 0.2).to(mask_dtype)
    chains = rng.integers(0, 3, 40)
    pair_mask = torch.from_numpy(chains[:, None] == chains[None, :])[None, ..., None].to(mask_dtype)
    args = (coords, pb_mask, pair_mask, 3.25, 50.75, 39, 1e8)
    got, expected = create_template_distogram(*args), _reference(*args)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert torch.equal(got, expected)


@pytest.mark.parametrize("edges", [(0.0, 3.0, 4), (3.0, 0.0, 4), (2.0, 2.0, 4), (1.0, 1.0, 1), (0.0, 3.0, 0)])
@pytest.mark.parametrize("mask_value", [0.0, 1.0, 0.5, -1.0, float("nan"), float("inf")])
def test_boundaries_and_mask_fallbacks(edges: tuple[float, float, int], mask_value: float) -> None:
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]])
    pb = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    pair = torch.full((1, 4, 4, 1), mask_value)
    args = (coords, pb, pair, *edges, 1e8)
    actual, expected = create_template_distogram(*args), _reference(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "mask_kind", ["binary", "binary-dense", "fractional", "nan", "negative-zero", "float64", "grad"]
)
def test_row_shards_match_reference(dtype: type, mask_kind: str) -> None:
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(2, 513, 3)).astype(dtype)
    coords[0, 5] = np.nan
    pb = torch.from_numpy((rng.random((2, 513)) > 0.2).astype(np.float32))
    chains = torch.arange(513) // 64
    pair = (chains[:, None] == chains[None, :])[None, ..., None].float()
    if mask_kind == "binary-dense":
        pair.fill_(1)
    if mask_kind == "fractional":
        pb[0, 1] = 0.3
    elif mask_kind == "nan":
        pair[0, 1, 2, 0] = float("nan")
    elif mask_kind == "negative-zero":
        pair[0, 1, 2, 0] = -0.0
    elif mask_kind == "float64":
        pb = pb.double()
    elif mask_kind == "grad":
        pb.requires_grad_()
    args = (coords, pb, pair, 0.1, 4.0, 3, 1e8)
    threads = torch.get_num_threads()
    before = pb.detach().clone(), pair.clone()
    result, expected = create_template_distogram(*args), _reference(*args)
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(result), torch.signbit(expected))
    torch.testing.assert_close(pb, before[0], equal_nan=True)
    torch.testing.assert_close(pair, before[1], equal_nan=True)
    assert torch.get_num_threads() == threads
    assert result.requires_grad == expected.requires_grad


def test_concurrent_row_shards() -> None:
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(3)
    coords = rng.normal(size=(1, 513, 3))
    pb = torch.ones((1, 513))
    pair = torch.ones((1, 513, 513, 1))
    args = (coords, pb, pair, 0.1, 4.0, 3, 1e8)
    expected = _reference(*args)
    threads = torch.get_num_threads()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: create_template_distogram(*args), range(4)))
    for result in results:
        assert torch.equal(result, expected)
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("slots", [[], [1, 3]])
def test_shards_preserve_template_slots(slots: list[int]) -> None:
    coords = np.random.default_rng(7).normal(size=(4, 513, 3))
    pb = torch.zeros((4, 513))
    pb[slots] = 1
    coords[[i for i in range(4) if i not in slots]] = np.nan
    chains = torch.arange(513) // 64
    pair = (chains[:, None] == chains[None, :])[None, ..., None].float()
    args = (coords, pb, pair, 0.1, 4.0, 3, 1e8)
    result, expected = create_template_distogram(*args), _reference(*args)
    assert result.shape == (4, 513, 513, 3)
    assert torch.equal(result, expected)
    assert torch.equal(torch.signbit(result), torch.signbit(expected))


def test_pool_fork_lifecycle() -> None:
    import os
    import subprocess
    import sys
    import textwrap

    if not hasattr(os, "register_at_fork"):
        pytest.skip("Fork hooks are unavailable")
    script = textwrap.dedent("""\
        import os
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event, Timer
        from bionemo_ir.pipeline.models.openfold3 import common

        lock = common._DISTOGRAM_LOCK
        common._run_distogram_batch(lambda _: None, [(0, 1)])
        pool = common._DISTOGRAM_POOL[1]
        started, release, completed = Event(), Event(), Event()

        def batch():
            def block(_):
                started.set()
                assert release.wait(5)
                completed.set()
            common._run_distogram_batch(block, [(0, 1)])

        def child_check():
            assert completed.is_set()
            assert common._DISTOGRAM_LOCK is not lock
            assert common._DISTOGRAM_POOL is None
            values = []
            common._run_distogram_batch(lambda _: values.append(os.getpid()), [(0, 1)])
            assert values == [os.getpid()]
            child_pool = common._DISTOGRAM_POOL[1]
            assert child_pool is not pool
            child_pool.shutdown()

        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(batch)
            assert started.wait(5)
            timer = Timer(0.2, release.set)
            timer.start()
            child = os.fork()
            if child == 0:
                try:
                    child_check()
                    second_lock = common._DISTOGRAM_LOCK
                    grandchild = os.fork()
                    if grandchild == 0:
                        assert common._DISTOGRAM_LOCK is not second_lock
                        child_check()
                        os._exit(0)
                    _, status = os.waitpid(grandchild, 0)
                    assert os.waitstatus_to_exitcode(status) == 0
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            future.result(timeout=5)
            timer.join()
        assert common._DISTOGRAM_LOCK is lock and not lock.locked()
        values = []
        common._run_distogram_batch(lambda _: values.append(os.getpid()), [(0, 1)])
        assert values == [os.getpid()]
        assert common._DISTOGRAM_POOL[1] is pool
        pool.shutdown()
        """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
