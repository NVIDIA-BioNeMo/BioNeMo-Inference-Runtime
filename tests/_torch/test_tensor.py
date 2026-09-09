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

import pytest
import torch

from bionemo_ir._torch.utils.tensor import batched_gather, tensor_tree_map, tree_map

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_batched_gather_indexes_on_the_same_device(device: str) -> None:
    data = torch.tensor([[10, 20, 30], [40, 50, 60]], device=device)
    inds = torch.tensor([[2, 0], [1, 1]], device=device)
    out = batched_gather(data, inds, dim=1, no_batch_dims=1)
    expected = torch.tensor([[30, 10], [50, 50]], device=device)
    torch.testing.assert_close(out, expected)


def test_tree_map_maps_supported_nodes() -> None:
    tree = {"t": torch.ones(2), "inner": [torch.zeros(1)]}
    out = tree_map(lambda t: t + 1, tree, torch.Tensor)
    assert torch.equal(out["t"], torch.full((2,), 2.0))
    assert torch.equal(out["inner"][0], torch.ones(1))


def test_tree_map_leaves_non_leaf_values_unchanged() -> None:
    tree = {"t": torch.ones(2), "n": 3, "s": "keep", "x": 1.5}
    out = tree_map(lambda t: t * 2, tree, torch.Tensor)
    assert torch.equal(out["t"], torch.full((2,), 2.0))
    assert out["n"] == 3
    assert out["s"] == "keep"
    assert out["x"] == 1.5


def test_tensor_tree_map_preserves_non_tensor_leaves() -> None:
    tree = {"t": torch.zeros(1), "meta": None, "scale": 0.25}
    out = tensor_tree_map(lambda t: t + 1, tree)
    assert torch.equal(out["t"], torch.ones(1))
    assert out["meta"] is None
    assert out["scale"] == 0.25
