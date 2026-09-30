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
"""Triton ``query_to_keys`` window gather against ``query_to_keys_optimized``."""

import pytest
import torch

from bionemo_ir._torch.layers.sequence_local_atom import (
    build_local_attn_metadata,
    create_gather_indices,
    query_to_keys_optimized,
)
from bionemo_ir.dsl_kernels.triton.query_to_keys import query_to_keys_triton

W, H = 32, 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("num_blocks", [1, 2, 5, 40])
@pytest.mark.parametrize("shape", ["bnd", "bkwd", "bmkwd", "wide"])
def test_query_to_keys_triton_bit_exact(num_blocks: int, shape: str) -> None:
    """The Triton gather reproduces ``query_to_keys_optimized`` bit for bit."""
    gather_indices, _ = create_gather_indices(num_blocks, W, H, torch.device("cuda"))
    dims = {
        "bnd": (2, num_blocks * W, 7),
        "bkwd": (2, num_blocks, W, 128),
        "bmkwd": (1, 5, num_blocks, W, 3),
        "wide": (1, num_blocks, W, 200),
    }[shape]
    x = torch.randn(dims, device="cuda")
    ref = query_to_keys_optimized(x, gather_indices, W=W, H=H)
    assert torch.equal(query_to_keys_triton(x, W=W, H=H), ref)
    ids = torch.randint(0, 50, dims, device="cuda")
    assert torch.equal(query_to_keys_triton(ids, W=W, H=H), query_to_keys_optimized(ids, gather_indices, W=W, H=H))


def test_query_to_keys_triton_rejects_partial_blocks() -> None:
    """A 3-d input must hold whole query blocks, as in the reference."""
    with pytest.raises(ValueError, match="multiple"):
        query_to_keys_triton(torch.randn(1, 3 * W + 5, 4, device="cuda"), W=W, H=H)


def test_local_metadata_serves_cpu_inputs() -> None:
    """The metadata gather falls back to the torch gather for CPU tensors."""
    x = torch.randn(1, 3, W, 5)
    gather_indices, _ = create_gather_indices(3, W, H, x.device)
    got = build_local_attn_metadata(W, H).query_to_keys(x)
    assert torch.equal(got, query_to_keys_optimized(x, gather_indices, W=W, H=H))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_query_to_keys_triton_cached_launch(dtype: torch.dtype) -> None:
    """Cached dtypes, the JIT fallback dtype and a misaligned source stay bit-exact."""
    gather_indices, _ = create_gather_indices(4, W, H, torch.device("cuda"))
    x = torch.randn(1, 2, 4, W, 3, device="cuda").to(dtype)
    ref = query_to_keys_optimized(x, gather_indices, W=W, H=H)
    assert torch.equal(query_to_keys_triton(x, W=W, H=H), ref)
    storage = torch.randn(x.numel() + 1, device="cuda").to(dtype)
    misaligned = storage[1:].view(x.shape)
    assert misaligned.data_ptr() % 16
    ref = query_to_keys_optimized(misaligned, gather_indices, W=W, H=H)
    assert torch.equal(query_to_keys_triton(misaligned, W=W, H=H), ref)


def test_query_to_keys_triton_cuda_graph() -> None:
    """The cached launch captures into and replays from a CUDA graph."""
    x = torch.randn(1, 5, 6, W, 128, device="cuda", dtype=torch.bfloat16)
    gather_indices, _ = create_gather_indices(6, W, H, x.device)
    query_to_keys_triton(x, W=W, H=H)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        query_to_keys_triton(x, W=W, H=H)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = query_to_keys_triton(x, W=W, H=H)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, query_to_keys_optimized(x, gather_indices, W=W, H=H))
