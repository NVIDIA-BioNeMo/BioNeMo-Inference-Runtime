# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_triangle_attention_node_weights,
    create_triangle_multiplication_node_weights,
    load_triangle_attention_node_weights_torch,
    load_triangle_multiplication_node_weights_torch,
)
from test_utils.boltz.ref_layers import RefTriangleAttentionNode, RefTriangleMultiplicationNode

from bionemo_ir._torch.attention_backend import AttentionType, get_attention_backend
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleAttentionNode,
    TriangleAttentionNodeType,
    TriangleMultiplicationMetadata,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
    precompute_trimul_metadata,
    set_trimul_token_padding,
)
from bionemo_ir._torch.utils import ChunkPolicy, recursive_calling_load_weights
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import SM_VERSION, make_left_aligned_pair_mask, skip_if_no_cutedsl


@dataclass(kw_only=True, frozen=True)
class AttnNodeScenario:
    backend: str
    seq_len: int = 32
    c_in: int = 128
    c_hidden: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    chunk_size: int = 0
    torch_dtype: str = "float32"
    starting: bool = True


@dataclass(kw_only=True, frozen=True)
class MulNodeScenario:
    seq_len: int = 16
    mul_type: TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.OUTGOING
    torch_dtype: str = "float32"
    high_precision: bool = True


@pytest.mark.parametrize(
    "s",
    [
        AttnNodeScenario(backend="VANILLA"),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16"),
        AttnNodeScenario(backend="SDPA"),
        AttnNodeScenario(backend="SDPA", torch_dtype="bfloat16"),
        AttnNodeScenario(backend="VANILLA", chunk_size=16),
        AttnNodeScenario(backend="VANILLA", chunk_size=8),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=16),
        AttnNodeScenario(backend="VANILLA", torch_dtype="bfloat16", chunk_size=8),
    ],
)
def test_triangle_attention_node(s: AttnNodeScenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend, AttentionType.TRIANGLE).Metadata
    bs = 1

    dtype = str_dtype_to_torch(s.torch_dtype)
    device = torch.device("cuda")

    ref_node = RefTriangleAttentionNode.load_weights(no_heads=s.num_attention_heads, starting=s.starting)
    ref_node.to(device)
    ref_node.eval()

    weights_and_biases = create_triangle_attention_node_weights(from_ref=ref_node)
    chunk_policy = ChunkPolicy(chunk_size=s.chunk_size, min_size=1) if s.chunk_size else ChunkPolicy(enabled=False)
    node = TriangleAttentionNode(
        c_in=s.c_in,
        c_hidden=s.c_hidden,
        num_heads=ref_node.num_heads,
        node_type=TriangleAttentionNodeType.STARTING if s.starting else TriangleAttentionNodeType.ENDING,
        dtype=dtype,
        attn_backend=s.backend,
        skip_create_weights=False,
        chunk_policy=chunk_policy,
    )
    node.to(device)
    load_triangle_attention_node_weights_torch(node, weights_and_biases, dtype)
    attn_metadata = metadata_cls()
    x = torch.randn(bs, s.seq_len, s.seq_len, s.c_in, dtype=torch.float32).cuda()
    mask = make_left_aligned_pair_mask(bs, s.seq_len, dtype=torch.float32, device="cuda")

    seen_qkv_rows: list[int] = []

    def record_qkv_rows(_module, inputs) -> None:
        seen_qkv_rows.append(inputs[0].shape[1])

    handle = node.mha.in_proj.register_forward_pre_hook(record_qkv_rows)
    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_output = ref_node(x, mask)
        try:
            output = node(x, mask, attn_metadata=attn_metadata)
        finally:
            handle.remove()

    expected_qkv_rows = (
        [min(s.chunk_size, s.seq_len - start) for start in range(0, s.seq_len, s.chunk_size)]
        if s.chunk_size
        else [s.seq_len]
    )
    assert seen_qkv_rows == expected_qkv_rows
    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    elif dtype == torch.bfloat16:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert diff0_max <= diff1_max * 1.5 + 1e-3
        assert diff0_mean <= diff1_mean + 0.05


@pytest.mark.parametrize(
    "s",
    [
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.INCOMING, torch_dtype="bfloat16"),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16"),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", high_precision=False),
        MulNodeScenario(mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", seq_len=512),
        MulNodeScenario(
            mul_type=TriangleMultiplicationNodeType.OUTGOING, torch_dtype="bfloat16", seq_len=512, high_precision=False
        ),
    ],
)
def test_triangle_multiplication_node(s: MulNodeScenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(s.torch_dtype)
    ref_node = RefTriangleMultiplicationNode.load_weights(
        outgoing=s.mul_type == TriangleMultiplicationNodeType.OUTGOING
    )
    ref_node.to(device)
    ref_node.eval()
    bs = 1

    weights_and_biases = create_triangle_multiplication_node_weights(from_ref=ref_node)
    node = TriangleMultiplicationNode(
        dim=ref_node.dim,
        multiplication_type=s.mul_type,
        dtype=dtype,
        skip_create_weights=False,
        high_precision=s.high_precision,
    )
    node.to(device)
    load_triangle_multiplication_node_weights_torch(node, weights_and_biases, dtype)

    x = torch.randn(bs, s.seq_len, s.seq_len, ref_node.dim, device="cuda")
    mask = make_left_aligned_pair_mask(1, s.seq_len, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        ref_output_float = ref_node(x, mask)
        x = x.to(dtype)
        mask = mask.to(dtype)
        ref_node = ref_node.to(dtype)
        ref_output = ref_node(x, mask)
        output = node(x, mask, precompute_trimul_metadata(x, None, None))

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-4)
    elif dtype == torch.bfloat16:
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.05


@pytest.mark.parametrize(
    ("hidden_dim", "dtype", "high_precision"),
    [
        pytest.param(196, torch.bfloat16, False, id="unaligned-bf16"),
        pytest.param(128, torch.bfloat16, False, id="aligned-bf16"),
        pytest.param(196, torch.float32, True, id="unaligned-fp32"),
    ],
)
def test_trimul_output_gate_keeps_native_widths(
    hidden_dim: int,
    dtype: torch.dtype,
    high_precision: bool,
) -> None:
    node = TriangleMultiplicationNode(
        dim=128,
        hidden_dim=hidden_dim,
        multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
        dtype=dtype,
        high_precision=high_precision,
        bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True},
    ).cuda()
    assert not hasattr(node, "_k_pad_cache")
    assert not hasattr(node, "_k_align_or_off")

    x0 = torch.randn(2, 3, 128, device="cuda", dtype=dtype)
    x1 = torch.randn(2, 3, hidden_dim, device="cuda", dtype=dtype)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def operation(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return torch.empty(2, 3, 128, device="cuda", dtype=dtype)

    node._dual_gemm_x0_x1_op = operation
    output = node._output_gate(x0, x1)

    assert output.shape == (2, 3, 128)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is x0
    assert args[1] is x1
    assert args[2] is node.g_out.weight
    assert args[3] is node.p_out.weight
    assert node.p_out.weight.shape == (128, hidden_dim)
    assert kwargs == {"actual_seqlen": None, "residual": None}


@pytest.mark.parametrize(
    "multiplication_type",
    [TriangleMultiplicationNodeType.OUTGOING, TriangleMultiplicationNodeType.INCOMING],
)
def test_trimul_fused_residual_matches_masked_update(multiplication_type) -> None:
    """Fuse the output gate and masked residual for aligned pair storage."""
    skip_if_no_cutedsl("dual_gemm_x0_x1")
    if SM_VERSION not in (86, 90, 100, 103):
        pytest.skip(f"this fused triangle residual integration case requires SM86/90/100/103 (current SM{SM_VERSION})")

    torch.manual_seed(73)
    node = TriangleMultiplicationNode(
        dim=384,
        hidden_dim=256,
        multiplication_type=multiplication_type,
        dtype=torch.bfloat16,
        high_precision=False,
        pair_mask_left_aligned=True,
    ).cuda()
    with torch.no_grad():
        for parameter in node.parameters():
            parameter.normal_(std=0.03)

    tokens, valid = 256, 249
    pair = torch.randn(1, tokens, tokens, 384, device="cuda", dtype=torch.bfloat16) * 0.03
    token_mask = torch.arange(tokens, device="cuda") < valid
    pair_mask = (token_mask[:, None] & token_mask[None, :]).unsqueeze(0)
    actual_out = pair_mask.sum(dim=-1, dtype=torch.int32)
    actual_in = pair_mask.sum(dim=-2, dtype=torch.int32)
    pair_mask_value = pair_mask.unsqueeze(-1).to(pair.dtype)
    assert node.can_fuse_residual(pair)
    trimul_metadata = precompute_trimul_metadata(pair, actual_out, actual_in)

    with torch.inference_mode():
        update = node._forward_impl(pair, pair_mask, trimul_metadata)
        expected = pair.clone().add_(update).mul_(pair_mask_value)
        actual_output = node._forward_impl(
            pair,
            pair_mask,
            trimul_metadata,
            residual=True,
        )

    torch.testing.assert_close(actual_output, expected, atol=1e-2, rtol=1e-2)
    assert torch.count_nonzero(actual_output[:, valid:]) == 0
    assert torch.count_nonzero(actual_output[:, :, valid:]) == 0

    unaligned = torch.empty(1, tokens + 1, tokens + 1, 384, device="meta")
    assert not node.can_fuse_residual(unaligned)


def test_trimul_metadata_precomputes_both_padded_row_lengths() -> None:
    """Token padding metadata is reusable across outgoing and incoming nodes."""
    skip_if_no_cutedsl()
    if SM_VERSION != 90:
        pytest.skip(f"token-aligned TriMul requires SM90 (current SM{SM_VERSION})")

    pair = torch.empty(2, 257, 259, 128, dtype=torch.bfloat16, device="cuda")
    outgoing = torch.randint(0, 260, (2, 257), dtype=torch.int32, device="cuda")
    incoming = torch.randint(0, 258, (2, 259), dtype=torch.int32, device="cuda")

    metadata = precompute_trimul_metadata(pair, outgoing, incoming)

    assert metadata.token_pad_multiple == 8
    assert metadata.outgoing_actual_seqlen is outgoing
    assert metadata.incoming_actual_seqlen is incoming
    assert metadata.padded_outgoing_actual_seqlen.shape == (2, 264)
    assert metadata.padded_incoming_actual_seqlen.shape == (2, 264)
    torch.testing.assert_close(metadata.padded_outgoing_actual_seqlen[:, :257], outgoing)
    torch.testing.assert_close(metadata.padded_incoming_actual_seqlen[:, :259], incoming)
    assert torch.count_nonzero(metadata.padded_outgoing_actual_seqlen[:, 257:]) == 0
    assert torch.count_nonzero(metadata.padded_incoming_actual_seqlen[:, 259:]) == 0


_KF_NODE = {
    "dim": 128,
    "dtype": torch.bfloat16,
    "high_precision": False,
    "bias_flags": {"p_in": True, "g_in": True, "p_out": True, "g_out": True},
    "pad_tokens": True,
}
_DIRECTIONS = [TriangleMultiplicationNodeType.OUTGOING, TriangleMultiplicationNodeType.INCOMING]


def _kf_trimul_node(**overrides: object) -> TriangleMultiplicationNode:
    """A CUDA node on the SM90 TriMul KF chain with realistic weights, or skip without the chain."""
    if SM_VERSION != 90:
        pytest.skip(f"the TriMul KF chain requires SM90 (current SM{SM_VERSION})")
    node = TriangleMultiplicationNode(**(_KF_NODE | overrides)).cuda()
    if node._kf_ops is None:
        pytest.skip("this build ships no TriMul KF kernels for the node")
    with torch.no_grad():
        for name, parameter in node.named_parameters():
            if name.startswith("norm") and name.endswith("weight"):
                parameter.normal_(1.0, 0.2)
            elif name.endswith("bias"):
                parameter.normal_(0.0, 0.2)
            else:
                parameter.normal_(0.0, parameter.shape[-1] ** -0.5)
    return node


def _kf_inputs(tokens: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, TriangleMultiplicationMetadata]:
    """A bf16 pair, a left-aligned pair mask with two different lengths, and their row lengths."""
    x = torch.randn(2, tokens, tokens, dim, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([tokens - 7, tokens - 40], device="cuda")
    token_mask = torch.arange(tokens, device="cuda") < lengths[:, None]
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    metadata = precompute_trimul_metadata(x, pair_mask.sum(-1, dtype=torch.int32), pair_mask.sum(-2, dtype=torch.int32))
    return x, pair_mask, metadata


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    return (difference / torch.linalg.vector_norm(expected.float())).item()


def test_trimul_kf_chain_needs_token_padding_and_a_supported_node() -> None:
    if SM_VERSION != 90:
        pytest.skip(f"the TriMul KF chain requires SM90 (current SM{SM_VERSION})")
    supported = _KF_NODE | {"skip_create_weights": True}
    nodes = torch.nn.ModuleList(
        TriangleMultiplicationNode(**(supported | {"multiplication_type": direction, "pad_tokens": False}))
        for direction in _DIRECTIONS
    )
    assert all(node._kf_ops is None for node in nodes)
    set_trimul_token_padding(nodes, True)
    if any(node._kf_ops is None for node in nodes):
        pytest.skip("this build ships no TriMul KF kernels for the node")
    assert all(node.pad_tokens for node in nodes)
    set_trimul_token_padding(nodes, False)
    assert all(not node.pad_tokens and node._kf_ops is None for node in nodes)

    for override in (
        {"hidden_dim": 64},
        {"dim": 384},
        {"dtype": torch.float32},
        {"high_precision": True},
        {"mean_normalization": True},
        {"pair_mask_left_aligned": False},
    ):
        assert TriangleMultiplicationNode(**(supported | override))._kf_ops is None, override


@pytest.mark.parametrize("residual", [False, True], ids=["update", "residual"])
@pytest.mark.parametrize("multiplication_type", _DIRECTIONS, ids=lambda direction: direction.name.lower())
@pytest.mark.parametrize(
    ("dim", "tokens", "bias"),
    [
        pytest.param(128, 128, True, id="C128-N128-streaming"),
        pytest.param(128, 256, False, id="C128-N256-pingpong-nobias"),
        pytest.param(256, 256, True, id="C256-N256-row-stats"),
        pytest.param(256, 256, False, id="C256-N256-row-stats-nobias"),
    ],
)
def test_trimul_kf_chain_matches_dual_gemm_path(
    dim: int,
    tokens: int,
    bias: bool,
    multiplication_type: TriangleMultiplicationNodeType,
    residual: bool,
) -> None:
    torch.manual_seed(31)
    node = _kf_trimul_node(
        dim=dim,
        multiplication_type=multiplication_type,
        bias_flags=dict.fromkeys(("p_in", "g_in", "p_out", "g_out"), bias),
    )
    x, pair_mask, metadata = _kf_inputs(tokens, dim)

    with torch.inference_mode():
        actual = node._kf_forward_if_supported(x, pair_mask, metadata, residual)
        assert actual is not None
        torch.testing.assert_close(node(x, pair_mask, metadata, residual=residual), actual, atol=0, rtol=0)
        expected = node._forward_impl(x, pair_mask, metadata, residual=residual)

    assert actual.shape == x.shape and actual.dtype == torch.bfloat16
    assert _relative_l2(actual, expected) < 1e-2
    if residual:
        assert torch.count_nonzero(actual[~pair_mask]) == 0


@pytest.mark.parametrize("multiplication_type", _DIRECTIONS, ids=lambda direction: direction.name.lower())
def test_trimul_kf_chain_counts_mask_rows_without_metadata_lengths(
    multiplication_type: TriangleMultiplicationNodeType,
) -> None:
    torch.manual_seed(37)
    node = _kf_trimul_node(multiplication_type=multiplication_type)
    x, pair_mask, metadata = _kf_inputs(256, 128)

    with torch.inference_mode():
        expected = node._kf_forward_if_supported(x, pair_mask, metadata, True)
        actual = node._kf_forward_if_supported(x, pair_mask, TriangleMultiplicationMetadata(), True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_trimul_kf_chain_leaves_unaligned_tokens_to_the_dual_gemm_path() -> None:
    node = _kf_trimul_node()
    x, pair_mask, metadata = _kf_inputs(250, 128)

    with torch.inference_mode():
        assert node._kf_forward_if_supported(x, pair_mask, metadata, True) is None
        actual = node(x, pair_mask, metadata, residual=True)
        expected = node._forward_impl(x, pair_mask, metadata, residual=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_trimul_kf_chain_refolds_after_load_weights() -> None:
    torch.manual_seed(41)
    node = _kf_trimul_node()
    x, pair_mask, metadata = _kf_inputs(128, 128)
    with torch.inference_mode():
        before = node(x, pair_mask, metadata)
    assert node._kf_folds is not None

    norm_in = {"weight": 2 * node.norm_in.weight.detach(), "bias": node.norm_in.bias.detach() + 0.5}
    recursive_calling_load_weights(node, {"norm_in": [norm_in]}, filter_func=lambda name, _: name != "norm_in")
    assert node._kf_folds is None

    with torch.inference_mode():
        actual = node(x, pair_mask, metadata)
        expected = node._forward_impl(x, pair_mask, metadata)
    assert _relative_l2(actual, expected) < 1e-2
    assert _relative_l2(before, expected) > 0.1
