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
"""Triangle-attention residual paths: the fused SM90 and SM80 epilogues, with and without the output bias and
the output gate, and the unfused fallback."""

import pytest
import torch

from bionemo_ir._torch.custom_ops.attn_epilogue import AttnEpilogue, get_attn_epilogue_op
from bionemo_ir._torch.custom_ops.attn_epilogue import cutedsl as epilogue_cutedsl
from bionemo_ir._torch.layers.triangle_nodes import TriangleAttentionNode, TriangleAttentionNodeType
from bionemo_ir._torch.utils import ChunkPolicy
from tests._torch import (
    cutedsl_test_modes,
    make_left_aligned_pair_mask,
    require_cubin_family,
    run_cutedsl_test_mode,
    skip_if_epilogue_tile_exceeds_smem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

NODE_TYPES = [TriangleAttentionNodeType.STARTING, TriangleAttentionNodeType.ENDING]
MODES = cutedsl_test_modes("bionemo_ir._torch.custom_ops.attn_epilogue._source")


def _require_mode(mode: str) -> None:
    if mode == "cubin":
        require_cubin_family("attn_epilogue")


def _node(
    node_type: TriangleAttentionNodeType,
    *,
    head_dim: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    attn_backend: str = "CuTeDSL",
    chunk_size: int = 0,
    bias: bool = False,
) -> TriangleAttentionNode:
    torch.manual_seed(0)
    policy = ChunkPolicy(chunk_size=chunk_size, min_size=1) if chunk_size else ChunkPolicy(enabled=False)
    node = TriangleAttentionNode(
        c_in=128,
        c_hidden=head_dim,
        num_heads=4,
        node_type=node_type,
        dtype=dtype,
        attn_backend=attn_backend,
        chunk_policy=policy,
        mha_bias_flags={"q": False, "k": False, "v": False, "g": bias, "o": bias},
    ).cuda()
    with torch.no_grad():
        for name, parameter in node.named_parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, parameter.shape[1] ** -0.5)
            else:
                parameter.normal_(1 if name.endswith("weight") else 0, 0.1)
    return node


def _fused_node(node_type: TriangleAttentionNodeType, **kwargs) -> TriangleAttentionNode:
    node = _node(node_type, **kwargs)
    if node.mha._epilogue is None:
        pytest.skip("the fused epilogue needs SM90 and the CuTeDSL sources")
    return node


def _count_o_proj_calls(node: TriangleAttentionNode) -> list[int]:
    calls: list[int] = []
    node.mha.o_proj.register_forward_hook(lambda *_: calls.append(1))
    return calls


def _inputs(batch: int, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq_len, seq_len, 128, device="cuda", dtype=dtype)
    return x, make_left_aligned_pair_mask(batch, seq_len, dtype=dtype, device="cuda")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("node_type", NODE_TYPES, ids=lambda t: t.name.lower())
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize(
    ("head_dim", "seq_len", "chunk_size"),
    [(32, 64, 0), (32, 100, 0), (32, 100, 24), (64, 100, 0), (64, 100, 24), (128, 100, 0), (128, 100, 24)],
)
def test_fused_epilogue_matches_separate_residual_add(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    node_type: TriangleAttentionNodeType,
    inplace: bool,
    head_dim: int,
    seq_len: int,
    chunk_size: int,
) -> None:
    _require_mode(mode)
    node = _fused_node(node_type, head_dim=head_dim, chunk_size=chunk_size)
    calls = _count_o_proj_calls(node)
    x, mask = _inputs(1, seq_len, torch.bfloat16)
    with torch.inference_mode():
        expected = x + node(x, mask)
        calls.clear()
        actual = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            epilogue_cutedsl.AttnEpilogueCuTe,
            epilogue_cutedsl,
            lambda: node(x.clone(), mask, residual=True, inplace_residual=inplace),
        )
        residual = x.clone()
        stored = node(residual, mask, residual=True, inplace_residual=inplace)

    assert not calls
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)
    assert torch.equal(stored, actual)
    if inplace:
        assert stored.data_ptr() == residual.data_ptr()
    else:
        assert torch.equal(residual, x)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("node_type", NODE_TYPES, ids=lambda t: t.name.lower())
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize("head_dim", [32, 128])
def test_fused_epilogue_adds_the_output_bias(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    node_type: TriangleAttentionNodeType,
    inplace: bool,
    head_dim: int,
) -> None:
    _require_mode(mode)
    node = _fused_node(node_type, head_dim=head_dim, bias=True)
    assert node.mha.o_proj.bias is not None
    calls = _count_o_proj_calls(node)
    x, mask = _inputs(1, 100, torch.bfloat16)
    with torch.inference_mode():
        expected = x + node(x, mask)
        calls.clear()
        actual = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            epilogue_cutedsl.AttnEpilogueCuTe,
            epilogue_cutedsl,
            lambda: node(x.clone(), mask, residual=True, inplace_residual=inplace),
        )

    assert not calls
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("transposed", [False, True], ids=["starting", "ending"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_sm80_kernel_matches_separate_residual_add(head_dim: int, transposed: bool, inplace: bool, bias: bool) -> None:
    # The Ampere kernel runs on any SM80+ device; select its tuning explicitly.
    if "source" not in MODES:
        pytest.skip("SM80 CUBINs do not run on SM90; only the source path can target this device")
    width = 4 * head_dim
    backend = epilogue_cutedsl.AttnEpilogueCuTe(4, head_dim, 128, has_bias=bias)
    backend._sm_version = 80
    skip_if_epilogue_tile_exceeds_smem(backend)
    op = AttnEpilogue(backend, 4, head_dim, 128, has_bias=bias)
    torch.manual_seed(head_dim)
    rows = columns = 70
    mha_o = torch.randn(rows * columns, 4, head_dim, device="cuda", dtype=torch.bfloat16).view(
        rows, columns, 4, head_dim
    )
    gate = torch.randn(1, rows, columns, 4 * width + 32, device="cuda", dtype=torch.bfloat16)[..., :width]
    weight = (torch.randn(128, width, device="cuda") * width**-0.5).to(torch.bfloat16)
    o_bias = (torch.randn(128, device="cuda") * 0.1).to(torch.bfloat16) if bias else None
    base = torch.randn(1, columns, rows, 128, device="cuda", dtype=torch.bfloat16)
    residual = base.transpose(1, 2) if transposed else base.clone()
    gated = mha_o.reshape(1, rows, columns, width) * gate.sigmoid()
    expected = residual + torch.nn.functional.linear(gated, weight, o_bias)

    with torch.inference_mode():
        target = residual.clone() if inplace else residual
        actual = op(mha_o, gate, weight, target, target if inplace else None, bias=o_bias)

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)
    if inplace:
        assert actual is target


def test_sm80_kernel_rejects_a_copy_pass_taller_than_its_tile() -> None:
    source = pytest.importorskip("bionemo_ir._torch.custom_ops.attn_epilogue._source")
    # 256 threads copying 16 bytes each cover 64 rows of a 32-wide K slice per pass: twice a 32-row tile.
    tile = {"tile_m": 32, "tile_k": 32, "num_stages": 3, "atom_layout_mnk": (2, 2, 2)}
    with pytest.raises(ValueError, match="cannot tile"):
        source.make_kernel("sm80", tile, 4, 32)
    source.make_kernel("sm80", {**tile, "tile_k": 64}, 4, 32)


# ``(batch, rows, gate_rows)``: every row reads gate row ``row // (rows / gate_rows)``.
GATE_ROWS = [(1, 5, 1), (2, 4, 2), (2, 3, 3)]
GATE_ROW_IDS = ["broadcast", "partial_broadcast", "rowwise"]


def _output_gate_operands(head_dim: int, batch: int, rows: int, gate_rows: int, bias: bool) -> tuple:
    torch.manual_seed(head_dim + rows)
    width, columns = 4 * head_dim, 77
    mha_o = torch.randn(batch * rows, columns, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    # The gate is a column slice of a fused projection, as AttentionPairBias leaves it.
    gate = torch.randn(batch, rows, columns, 4 * width, device="cuda", dtype=torch.bfloat16)[..., width : 2 * width]
    weight = (torch.randn(128, width, device="cuda") * width**-0.5).to(torch.bfloat16)
    o_bias = (torch.randn(128, device="cuda") * 0.1).to(torch.bfloat16) if bias else None
    residual = torch.randn(batch, rows, columns, 128, device="cuda", dtype=torch.bfloat16)
    logits = (2 * torch.randn(batch, gate_rows, columns, 128, device="cuda")).to(torch.bfloat16)
    return mha_o, gate, weight, o_bias, residual, logits


def _output_gate_reference(mha_o, gate, weight, o_bias, residual, logits) -> torch.Tensor:
    update = torch.nn.functional.linear(mha_o.reshape(gate.shape) * gate.sigmoid(), weight, o_bias)
    output_gate = logits.sigmoid().repeat_interleave(residual.shape[1] // logits.shape[1], dim=1)
    # The gated update joins the residual add, which rounds once.
    return (residual.float() + output_gate.float() * update.float()).to(residual.dtype)


def _run_output_gate(op: AttnEpilogue, operands: tuple, inplace: bool) -> torch.Tensor | None:
    mha_o, gate, weight, o_bias, residual, logits = operands
    target = residual.clone() if inplace else residual
    return op(mha_o, gate, weight, target, target if inplace else None, bias=o_bias, output_gate=logits)


def _assert_output_gate_matches(actual: torch.Tensor | None, operands: tuple) -> None:
    assert actual is not None
    expected = _output_gate_reference(*operands)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)
    # Only the GEMM's summation order may move a rounding.
    assert (actual == expected).float().mean() > 0.999


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize(("batch", "rows", "gate_rows"), GATE_ROWS, ids=GATE_ROW_IDS)
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
def test_output_gate_folds_into_the_residual_add(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    bias: bool,
    batch: int,
    rows: int,
    gate_rows: int,
    inplace: bool,
) -> None:
    _require_mode(mode)
    op = get_attn_epilogue_op(torch.bfloat16, 4, head_dim, 128, has_bias=bias, has_output_gate=True)
    if op is None:
        pytest.skip("the fused epilogue needs SM80+ and the CuTeDSL sources or CUBINs")
    operands = _output_gate_operands(head_dim, batch, rows, gate_rows, bias)
    with torch.inference_mode():
        actual = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            epilogue_cutedsl.AttnEpilogueCuTe,
            epilogue_cutedsl,
            lambda: _run_output_gate(op, operands, inplace),
        )
    _assert_output_gate_matches(actual, operands)


@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize(("batch", "rows", "gate_rows"), GATE_ROWS, ids=GATE_ROW_IDS)
def test_sm80_output_gate_folds_into_the_residual_add(
    head_dim: int, bias: bool, batch: int, rows: int, gate_rows: int
) -> None:
    if "source" not in MODES:
        pytest.skip("SM80 CUBINs do not run on SM90; only the source path can target this device")
    backend = epilogue_cutedsl.AttnEpilogueCuTe(4, head_dim, 128, has_bias=bias, has_output_gate=True)
    backend._sm_version = 80
    skip_if_epilogue_tile_exceeds_smem(backend)
    op = AttnEpilogue(backend, 4, head_dim, 128, has_bias=bias, has_output_gate=True)
    operands = _output_gate_operands(head_dim, batch, rows, gate_rows, bias)
    with torch.inference_mode():
        actual = _run_output_gate(op, operands, inplace=False)
    _assert_output_gate_matches(actual, operands)


def test_output_gate_declines_what_the_kernel_cannot_take() -> None:
    op = get_attn_epilogue_op(torch.bfloat16, 4, 32, 128, has_output_gate=True)
    if op is None:
        pytest.skip("the fused epilogue needs SM80+ and the CuTeDSL sources or CUBINs")
    mha_o, gate, weight, _, residual, logits = _output_gate_operands(32, 1, 3, 3, bias=False)
    with torch.inference_mode():
        # Gate rows that do not divide the residual rows, and a missing gate.
        assert op(mha_o, gate, weight, residual, output_gate=logits[:, :2]) is None
        assert op(mha_o, gate, weight, residual) is None
    # The gate's residual-sized tile leaves no room beside a 512-wide resident weight.
    assert get_attn_epilogue_op(torch.bfloat16, 4, 128, 128, has_output_gate=True) is None


@pytest.mark.parametrize(
    ("node_type", "fused"),
    [(TriangleAttentionNodeType.STARTING, True), (TriangleAttentionNodeType.ENDING, False)],
    ids=["starting", "ending"],
)
def test_fused_epilogue_folds_batch_only_where_strides_allow(node_type: TriangleAttentionNodeType, fused: bool) -> None:
    # The ending node's transposed pair keeps ``B`` apart from ``I``, so a
    # batch of two cannot fold into the kernel's ``B*I`` mode.
    node = _fused_node(node_type)
    calls = _count_o_proj_calls(node)
    x, mask = _inputs(2, 40, torch.bfloat16)
    with torch.inference_mode():
        expected = x + node(x, mask)
        calls.clear()
        actual = node(x, mask, residual=True)

    assert bool(calls) is not fused
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)


@pytest.mark.parametrize("node_type", NODE_TYPES, ids=lambda t: t.name.lower())
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize("chunk_size", [0, 16], ids=["dense", "chunked"])
def test_unfused_residual_matches_separate_add(
    node_type: TriangleAttentionNodeType, inplace: bool, chunk_size: int
) -> None:
    node = _node(node_type, dtype=torch.float32, attn_backend="VANILLA", chunk_size=chunk_size)
    assert node.mha._epilogue is None
    x, mask = _inputs(1, 40, torch.float32)
    residual = x.clone()
    with torch.inference_mode():
        expected = x + node(x, mask)
        actual = node(residual, mask, residual=True, inplace_residual=inplace)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if inplace:
        assert actual.data_ptr() == residual.data_ptr()
    else:
        assert torch.equal(residual, x)
