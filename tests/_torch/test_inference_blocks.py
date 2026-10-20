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

"""Caller-supplied mask and permutation contracts for reusable inference blocks."""

import pytest
import torch

from bionemo_ir._torch.attention_backend.utils import precompute_pair_masks, precompute_single_masks
from bionemo_ir._torch.layers.transformers.blocks import (
    ConditionedTransformerBlock,
    PairUpdateBlock,
    precompute_trimul_metadata,
    with_cuda_graph,
)
from tests._torch import SM_VERSION, init_module_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _skip_unsupported_cutedsl(backend: str) -> None:
    if backend == "CuTeDSL" and SM_VERSION not in (80, 86, 89, 90):
        pytest.skip(f"pairwise CuTeDSL requires SM80/86/89/90, got SM{SM_VERSION}")


@pytest.mark.parametrize("kind", ["SDPA", "CuTeDSL", "pair", "pair_without_triangles"])
def test_mask_and_permutation_contract(kind: str) -> None:
    torch.manual_seed(17)
    _skip_unsupported_cutedsl(kind)
    if kind.startswith("pair"):
        block = PairUpdateBlock(
            0,
            128,
            16,
            16,
            2,
            False,
            False,
            1e-5,
            torch.bfloat16,
            use_tri_mult=kind == "pair",
        )
    else:
        block = ConditionedTransformerBlock(0, 128, 16, 8, 2, 2, kind, 1e6, True, 1e-5, torch.bfloat16)
    block = block.cuda().eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.1, 0.1)
    x = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(2, 7, 7, 16, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(2, 7, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], device="cuda", dtype=torch.bool)
    order = torch.tensor([6, 2, 5, 0, 1, 4, 3], device="cuda")
    trimul_metadata = precompute_trimul_metadata(pair, None, None)
    inputs = (x, pair, mask, trimul_metadata) if kind.startswith("pair") else (x, pair, cond, mask)
    permuted = (x[:, order], pair[:, order][:, :, order], mask[:, order])
    if kind.startswith("pair"):
        permuted = (*permuted, trimul_metadata)
    else:
        permuted = (*permuted[:2], cond[:, order], permuted[-1])
    originals = [value.clone() for value in inputs if isinstance(value, torch.Tensor)]
    with torch.no_grad():
        expected = block(*inputs)
        actual = block(*permuted)
    expected = expected[:, order]
    if kind.startswith("pair"):
        expected = expected[:, :, order]
        valid = mask[:, order, None] & mask[:, None, order]
    else:
        valid = mask[:, order]
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~valid]) == 0
    input_tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
    for value, before in zip(input_tensors, originals, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="use native blocks for gradients"):
        block(*inputs)


def test_pair_block_zeroes_nonfinite_masked_inputs() -> None:
    torch.manual_seed(18)
    block = PairUpdateBlock(
        0,
        128,
        16,
        16,
        2,
        False,
        False,
        1e-5,
        torch.bfloat16,
        use_tri_mult=False,
    ).cuda()
    init_module_weights(block)
    pair = torch.randn(1, 7, 7, 16, device="cuda", dtype=torch.bfloat16)
    token_mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0]], device="cuda", dtype=torch.bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    pair[~pair_mask] = float("nan")
    with torch.no_grad():
        output = block(
            torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16),
            pair,
            token_mask,
            precompute_trimul_metadata(pair, None, None),
        )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~pair_mask]) == 0


def test_pair_block_reuses_precomputed_masks() -> None:
    """Masks a trunk computes once give the pair block's per-call results bit for bit."""
    torch.manual_seed(29)
    block = PairUpdateBlock(0, 128, 16, 16, 2, False, False, 1e-5, torch.bfloat16, use_tri_mult=False).cuda()
    init_module_weights(block)
    token_mask = torch.arange(7, device="cuda") < torch.tensor([[7], [4]], device="cuda")
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    single = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(2, 7, 7, 16, device="cuda", dtype=torch.bfloat16).masked_fill(~pair_mask[..., None], 0)
    metadata = precompute_trimul_metadata(pair, None, None)
    pair_masks = precompute_pair_masks("CuTeDSL", pair_mask, inf=1e6, activation_dtype=torch.bfloat16)
    single_masks = precompute_single_masks("SDPA", token_mask, inf=1e6, activation_dtype=torch.bfloat16)
    with torch.no_grad():
        expected = block(single, pair, token_mask, metadata)
        actual = block(
            single, pair, token_mask, metadata, precomputed_masks=pair_masks, precomputed_single_masks=single_masks
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["SDPA", "CuTeDSL"])
def test_conditioned_block_reuses_precomputed_masks_and_condition(backend: str) -> None:
    """Masks and condition projections a trunk computes once give the block's per-layer results bit for bit."""
    torch.manual_seed(23)
    _skip_unsupported_cutedsl(backend)
    block = ConditionedTransformerBlock(
        0, 128, 16, 8, 2, 2, backend, 1e6, True, 1e-5, torch.bfloat16, sanitize_masked_inputs=False
    ).cuda()
    init_module_weights(block)
    mask = torch.arange(12, device="cuda") < torch.tensor([[12], [9]], device="cuda")
    pair_mask = mask[:, :, None] & mask[:, None, :]
    x = torch.randn(2, 12, 128, device="cuda", dtype=torch.bfloat16).masked_fill(~mask[..., None], 0)
    pair = torch.randn(2, 12, 12, 16, device="cuda", dtype=torch.bfloat16).masked_fill(~pair_mask[..., None], 0)
    condition = torch.randn(2, 12, 8, device="cuda", dtype=torch.bfloat16).masked_fill(~mask[..., None], 0)
    hoisted = precompute_single_masks(backend, mask, inf=1e6, activation_dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = block(x, pair, condition, mask)
        actual = block(
            x,
            pair,
            condition,
            mask,
            precomputed_single_masks=hoisted,
            projected_condition=block.project_condition(condition, mask),
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["SDPA", "CuTeDSL"])
def test_conditioned_block_zeroes_nonfinite_masked_inputs(backend: str) -> None:
    torch.manual_seed(19)
    _skip_unsupported_cutedsl(backend)
    block = ConditionedTransformerBlock(0, 128, 16, 8, 2, 2, backend, 1e6, True, 1e-5, torch.bfloat16).cuda()
    init_module_weights(block)
    x = torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(1, 7, 7, 16, device="cuda", dtype=torch.bfloat16)
    condition = torch.randn(1, 7, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0]], device="cuda", dtype=torch.bool)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    x[~mask] = float("nan")
    condition[~mask] = float("inf")
    pair[~pair_mask] = float("nan")
    with torch.no_grad():
        output = block(x, pair, condition, mask)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~mask]) == 0


def test_cutedsl_attention_accepts_arbitrary_float16_masks() -> None:
    torch.manual_seed(20)
    _skip_unsupported_cutedsl("CuTeDSL")
    block = ConditionedTransformerBlock(0, 128, 16, 8, 2, 2, "CuTeDSL", 1e6, True, 1e-5, torch.float16).cuda()
    init_module_weights(block)
    x = torch.randn(1, 7, 128, device="cuda", dtype=torch.float16)
    pair = torch.randn(1, 7, 7, 16, device="cuda", dtype=torch.float16)
    cond = torch.randn(1, 7, 8, device="cuda", dtype=torch.float16)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0]], device="cuda", dtype=torch.bool)
    with torch.no_grad():
        output = block(x, pair, cond, mask)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~mask]) == 0


@pytest.mark.parametrize("backend", ["SDPA", "CuTeDSL"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_graph_replay_refreshes_masks(backend: str, batch_size: int) -> None:
    _skip_unsupported_cutedsl(backend)
    block = ConditionedTransformerBlock(0, 128, 16, 8, 2, 2, backend, 1e6, True, 1e-5, torch.bfloat16).cuda()
    init_module_weights(block)
    # The tracker is its own module and defaults to training mode, where the
    # runtime hands every call straight to the inner block (graphs are unsafe
    # under autograd). Without this the capture assertions below can only ever
    # see an untouched state table.
    graph = with_cuda_graph(block).eval()
    inputs = (
        torch.randn(batch_size, 7, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch_size, 7, 7, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch_size, 7, 8, device="cuda", dtype=torch.bfloat16),
        torch.ones(batch_size, 7, device="cuda", dtype=torch.bool),
    )
    with torch.no_grad():
        for _ in range(8):
            graph(*inputs)
        assert not graph.fallback_to_eager_by_key
        assert any(state.graph is not None for state in graph.graph_state_by_key.values())
        previous = graph(*inputs)
        saved = previous.clone()
        inputs[-1][:, 1::2] = False
        inputs[-1][-1] = False
        inputs[0].add_(1)
        actual = graph(*inputs)
        expected = block(*inputs)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(previous, saved, atol=0, rtol=0)
    assert torch.count_nonzero(actual[~inputs[-1]]) == 0
    graph.reset()
