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
"""Fixtures shared by the CUDA-graph runtime tests."""

import pytest

import bionemo_ir._torch.graph_optimization.cuda_graph.runtime as trk
from bionemo_ir._torch.graph_optimization.tensor_copy_utils import _tensor_leaves


@pytest.fixture
def copied(monkeypatch) -> list[int]:
    """Record the ids of the tensors each replay copies into static buffers."""
    ids: list[int] = []
    original = trk._copy_tensors_into

    def recording(dest, src):
        ids.extend(id(leaf) for leaf in _tensor_leaves(src))
        return original(dest, src)

    monkeypatch.setattr(trk, "_copy_tensors_into", recording)
    return ids
