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
from bionemo_ir._torch.layers import triangle_nodes as triangle_nodes_module
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleAttentionNode,
    TriangleAttentionNodeType,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
)
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import make_left_aligned_pair_mask


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

    handle = node.mha.qkv_proj.register_forward_pre_hook(record_qkv_rows)
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
        output = node(x, mask)

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
    ("hidden_dim", "expected_k1"),
    # 196 is an unaligned trimul hidden width and is not a multiple of the 8
    # elements a 128-bit copy moves; 128 is already aligned.
    [(196, 200), (128, 128)],
    ids=["unaligned", "aligned"],
)
def test_trimul_zero_extends_unaligned_hidden_width(hidden_dim: int, expected_k1: int) -> None:
    node = TriangleMultiplicationNode(
        dim=128,
        hidden_dim=hidden_dim,
        multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
        dtype=torch.bfloat16,
        high_precision=False,
        bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True},
    ).cuda()
    assert node._x1_k == expected_k1
    assert node._x0_k == 128

    # Linear allocates with torch.empty, so seed real values before
    # comparing: uninitialised memory can hold NaN, which never equals
    # itself.
    with torch.no_grad():
        node.p_out.weight.normal_()

    padded = node._k_padded_weight("p_out", node.p_out.weight, node._x1_k)
    assert padded.shape == (128, expected_k1)
    assert torch.equal(padded[:, :hidden_dim], node.p_out.weight)
    assert torch.all(padded[:, hidden_dim:] == 0)

    # Cached between calls...
    assert node._k_padded_weight("p_out", node.p_out.weight, node._x1_k).data_ptr() == padded.data_ptr()
    # ...but never stale once the source weight is rewritten.
    with torch.no_grad():
        node.p_out.weight.normal_()
    refreshed = node._k_padded_weight("p_out", node.p_out.weight, node._x1_k)
    assert torch.equal(refreshed[:, :hidden_dim], node.p_out.weight)


def test_trimul_keeps_raw_width_in_fp32() -> None:
    """fp32 falls back to the vanilla gate, where padding buys nothing."""
    node = TriangleMultiplicationNode(
        dim=128,
        hidden_dim=196,
        multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
        dtype=torch.float32,
        high_precision=True,
    )
    assert node._x1_k == 196
    assert node._k_align_or_off == -1


def test_trimul_skip_create_weights_skips_cueq_bias_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def get_cueq_trimul_api() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(triangle_nodes_module, "_get_cueq_trimul_api", get_cueq_trimul_api)
    node = TriangleMultiplicationNode(
        dim=384,
        hidden_dim=256,
        dtype=torch.bfloat16,
        high_precision=False,
        bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True},
        skip_create_weights=True,
    )

    assert calls == 0
    assert node._cueq_trimul_api is None
    for linear in (node.p_in, node.g_in, node.p_out, node.g_out):
        assert not linear._weights_created


def _cueq_384x256_trimul_node() -> TriangleMultiplicationNode:
    return TriangleMultiplicationNode(
        dim=384,
        hidden_dim=256,
        multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
        dtype=torch.bfloat16,
        high_precision=False,
        bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True},
    )


def test_cueq_trimul_api_requires_operation_and_support_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def operation(*args: object, **kwargs: object) -> None:
        return None

    def is_supported(*args: object, **kwargs: object) -> bool:
        return True

    class OperationOnly:
        triangle_multiplicative_update = staticmethod(operation)

    class CompleteApi:
        triangle_multiplicative_update = staticmethod(operation)
        triangle_multiplicative_update_is_supported = staticmethod(is_supported)

    triangle_nodes_module._get_cueq_trimul_api.cache_clear()
    try:
        monkeypatch.setattr(triangle_nodes_module, "import_module", lambda _name: OperationOnly)
        assert triangle_nodes_module._get_cueq_trimul_api() is None

        triangle_nodes_module._get_cueq_trimul_api.cache_clear()
        monkeypatch.setattr(triangle_nodes_module, "import_module", lambda _name: CompleteApi)
        assert triangle_nodes_module._get_cueq_trimul_api() == (operation, is_supported)
    finally:
        triangle_nodes_module._get_cueq_trimul_api.cache_clear()


def test_cueq_384x256_trimul_constructor_requires_internal_api_and_sm90(monkeypatch: pytest.MonkeyPatch) -> None:
    def operation(*args: object, **kwargs: object) -> None:
        return None

    def is_supported(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(triangle_nodes_module, "_get_cueq_trimul_api", lambda: (operation, is_supported))
    monkeypatch.setattr(triangle_nodes_module, "get_sm_version", lambda: 89)
    assert _cueq_384x256_trimul_node()._cueq_trimul_api is None

    monkeypatch.setattr(triangle_nodes_module, "get_sm_version", lambda: 90)
    assert _cueq_384x256_trimul_node()._cueq_trimul_api == (operation, is_supported)

    calls = 0

    def missing_api() -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(triangle_nodes_module, "_get_cueq_trimul_api", missing_api)
    assert _cueq_384x256_trimul_node()._cueq_trimul_api is None
    assert calls == 1

    other_shape = TriangleMultiplicationNode(
        dim=128,
        hidden_dim=128,
        dtype=torch.bfloat16,
        high_precision=False,
    )
    assert other_shape._cueq_trimul_api is None
    assert calls == 1


def test_cueq_384x256_trimul_runtime_gate_and_bool_mask_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    support_calls: list[tuple[torch.Tensor, str, torch.Tensor, int]] = []
    operation_calls: list[tuple[torch.Tensor, dict[str, object]]] = []

    def is_supported(
        x: torch.Tensor,
        *,
        direction: str,
        mask: torch.Tensor,
        c_hidden: int,
    ) -> bool:
        support_calls.append((x, direction, mask, c_hidden))
        return True

    def operation(x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        operation_calls.append((x, kwargs))
        return x

    monkeypatch.setattr(triangle_nodes_module, "_get_cueq_trimul_api", lambda: (operation, is_supported))
    monkeypatch.setattr(triangle_nodes_module, "get_sm_version", lambda: 90)
    node = _cueq_384x256_trimul_node().cuda().eval()

    short_x = torch.empty(1, 256, 256, 384, device="cuda", dtype=torch.bfloat16)
    short_mask = torch.ones(1, 256, 256, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        assert node._cueq_forward_if_supported(short_x, short_mask) is None
    assert support_calls == []
    assert operation_calls == []

    x = torch.empty(1, 257, 257, 384, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 257, 257, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        output = node(x, mask)
    assert output is x

    assert len(support_calls) == 1
    support_x, direction, support_mask, c_hidden = support_calls[0]
    assert support_x is x
    assert direction == "outgoing"
    assert support_mask.dtype == torch.bfloat16
    assert support_mask.is_contiguous()
    assert c_hidden == 256

    assert len(operation_calls) == 1
    operation_x, kwargs = operation_calls[0]
    assert operation_x is x
    assert kwargs["mask"] is support_mask
    assert kwargs["p_in_weight"] is node.p_in.weight
    assert kwargs["g_in_weight"] is node.g_in.weight
    assert kwargs["p_out_weight"] is node.p_out.weight
    assert kwargs["g_out_weight"] is node.g_out.weight


def test_internal_cueq_384x256_trimul_matches_bioir() -> None:
    if triangle_nodes_module._get_cueq_trimul_api() is None:
        pytest.skip("internal cuEquivariance TriMul API is not installed")
    if triangle_nodes_module.get_sm_version() != 90:
        pytest.skip("internal cuEquivariance 384x256 TriMul requires SM90")

    torch.manual_seed(20260908)
    node = _cueq_384x256_trimul_node().cuda().eval()
    with torch.no_grad():
        for name, parameter in node.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            elif name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.normal_(mean=0, std=0.02)

    x = torch.randn(1, 384, 384, 384, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 384, 384, device="cuda", dtype=torch.bool)
    cueq_api = node._cueq_trimul_api
    assert cueq_api is not None
    with torch.inference_mode():
        node._cueq_trimul_api = None
        reference = node(x, mask)
        node._cueq_trimul_api = cueq_api
        actual = node(x, mask)

    relative_l2 = torch.linalg.vector_norm(actual.float() - reference.float()) / torch.linalg.vector_norm(
        reference.float()
    )
    assert relative_l2 < 5e-3
