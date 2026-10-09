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
from bionemo_ir._torch.layers.attention import MSAAttention
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
    channels: int = 128,
    num_heads: int = 4,
) -> TriangleAttentionNode:
    torch.manual_seed(0)
    policy = ChunkPolicy(chunk_size=chunk_size, min_size=1) if chunk_size else ChunkPolicy(enabled=False)
    node = TriangleAttentionNode(
        c_in=channels,
        c_hidden=head_dim,
        num_heads=num_heads,
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


def _inputs(batch: int, seq_len: int, dtype: torch.dtype, channels: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq_len, seq_len, channels, device="cuda", dtype=dtype)
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
@pytest.mark.parametrize(("seq_len", "chunk_size"), [(40, 0), (100, 0), (100, 24), (200, 0)])
def test_streamed_epilogue_matches_separate_residual_add(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    node_type: TriangleAttentionNodeType,
    inplace: bool,
    seq_len: int,
    chunk_size: int,
) -> None:
    """Protenix's 256-channel pair, H=8 and D=32, takes the streamed kernel, or the tiled SM80 one for few rows."""
    _require_mode(mode)
    node = _fused_node(node_type, channels=256, num_heads=8, chunk_size=chunk_size)
    calls = _count_o_proj_calls(node)
    x, mask = _inputs(1, seq_len, torch.bfloat16, channels=256)
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


@pytest.mark.parametrize("column", [False, True], ids=["row", "column"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize(
    ("mode", "channels", "fused"),
    [pytest.param(mode, 256, True, id=f"c256-{mode}") for mode in MODES]
    + [pytest.param("fallback", 64, False, id="c64-fallback")],
)
def test_msa_attention_residual_matches_separate_add(
    mode: str, monkeypatch: pytest.MonkeyPatch, column: bool, inplace: bool, channels: int, fused: bool
) -> None:
    """OpenFold2's MSA row and column attention add their residual in the epilogue when it serves the shape."""
    _require_mode(mode)
    torch.manual_seed(0)
    layer = MSAAttention(
        local_layer_idx=0,
        c_in=channels,
        num_heads=8,
        c_z=128,
        triangle_attn_backend="CuTeDSL",
        need_project_z=not column,
        transpose_input=column,
        dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, parameter.shape[1] ** -0.5)
            else:
                parameter.normal_(1 if name.endswith("weight") else 0, 0.1)
    if fused and layer.mha._epilogue is None:
        pytest.skip("the fused epilogue needs SM90 and the CuTeDSL sources")
    assert (layer.mha._epilogue is not None) == fused
    calls = _count_o_proj_calls(layer)
    m = torch.randn(1, 24, 100, channels, device="cuda", dtype=torch.bfloat16)
    z = None if column else torch.randn(1, 100, 100, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 24, 100, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = m + layer(m, z, mask)
        calls.clear()
        residual = m.clone()
        if fused:
            actual = run_cutedsl_test_mode(
                mode,
                monkeypatch,
                epilogue_cutedsl.AttnEpilogueCuTe,
                epilogue_cutedsl,
                lambda: layer(residual.copy_(m), z, mask, residual=True, inplace_residual=inplace),
            )
        else:
            actual = layer(residual, z, mask, residual=True, inplace_residual=inplace)

    assert bool(calls) != fused
    assert actual.shape == m.shape
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)
    if inplace and fused:
        assert actual.data_ptr() == residual.data_ptr()
    elif not inplace:
        assert torch.equal(residual, m)


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


# A token transformer's epilogue, 12 heads of 64 projecting to 768 channels: a width in NO_RESIDUAL_WIDTHS.
NO_RESIDUAL_SHAPE = (12, 64, 768)


def _no_residual_operands(batch: int, rows: int, columns: int, bias: bool) -> tuple:
    heads, head_dim, channels = NO_RESIDUAL_SHAPE
    width = heads * head_dim
    torch.manual_seed(batch * rows + columns)
    mha_o = torch.randn(batch * rows, columns, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # The gate is a column slice of a fused projection, as the attention layers leave it.
    gate = torch.randn(batch, rows, columns, 2 * width, device="cuda", dtype=torch.bfloat16)[..., width:]
    weight = (torch.randn(channels, width, device="cuda") * width**-0.5).to(torch.bfloat16)
    o_bias = (torch.randn(channels, device="cuda") * 0.1).to(torch.bfloat16) if bias else None
    return mha_o, gate, weight, o_bias


def _no_residual_reference(mha_o, gate, weight, o_bias) -> torch.Tensor:
    return torch.nn.functional.linear(mha_o.reshape(gate.shape) * gate.sigmoid(), weight, o_bias)


def _nan_destination(gate: torch.Tensor) -> torch.Tensor:
    """A destination that poisons any read of it, so an accumulating kernel fails rather than passes."""
    return gate.new_full((*gate.shape[:-1], NO_RESIDUAL_SHAPE[2]), float("nan"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize(("batch", "rows", "columns"), [(2, 1, 136), (1, 3, 77)], ids=["tokens", "pairs"])
def test_residual_free_epilogue_writes_the_bare_projection(
    mode: str, monkeypatch: pytest.MonkeyPatch, bias: bool, batch: int, rows: int, columns: int
) -> None:
    _require_mode(mode)
    op = get_attn_epilogue_op(torch.bfloat16, *NO_RESIDUAL_SHAPE, has_bias=bias, has_residual=False)
    if op is None:
        pytest.skip("the fused epilogue needs SM80+ and the CuTeDSL sources or CUBINs")
    operands = _no_residual_operands(batch, rows, columns, bias)
    mha_o, gate, weight, o_bias = operands
    destination = _nan_destination(gate)
    with torch.inference_mode():
        allocated = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            epilogue_cutedsl.AttnEpilogueCuTe,
            epilogue_cutedsl,
            lambda: op(mha_o, gate, weight, None, bias=o_bias),
        )
        written = op(mha_o, gate, weight, None, destination, bias=o_bias)

    expected = _no_residual_reference(*operands)
    torch.testing.assert_close(allocated, expected, atol=1e-2, rtol=1.6e-2)
    assert written is destination
    torch.testing.assert_close(written, expected, atol=1e-2, rtol=1.6e-2)


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_sm80_residual_free_epilogue_writes_the_bare_projection(bias: bool) -> None:
    if "source" not in MODES:
        pytest.skip("SM80 CUBINs do not run on SM90; only the source path can target this device")
    heads, head_dim, channels = NO_RESIDUAL_SHAPE
    backend = epilogue_cutedsl.AttnEpilogueCuTe(heads, head_dim, channels, has_bias=bias, has_residual=False)
    backend._sm_version = 80
    skip_if_epilogue_tile_exceeds_smem(backend)
    op = AttnEpilogue(backend, heads, head_dim, channels, has_bias=bias, has_residual=False)
    mha_o, gate, weight, o_bias = operands = _no_residual_operands(2, 1, 136, bias)
    destination = _nan_destination(gate)
    with torch.inference_mode():
        actual = op(mha_o, gate, weight, None, destination, bias=o_bias)

    assert actual is destination
    torch.testing.assert_close(actual, _no_residual_reference(*operands), atol=1e-2, rtol=1.6e-2)


def test_residual_free_epilogue_declines_what_it_cannot_take() -> None:
    # The output gate scales the update inside the residual add, so it needs one.
    assert get_attn_epilogue_op(torch.bfloat16, *NO_RESIDUAL_SHAPE, has_output_gate=True, has_residual=False) is None
    bare = get_attn_epilogue_op(torch.bfloat16, *NO_RESIDUAL_SHAPE, has_residual=False)
    accumulating = get_attn_epilogue_op(torch.bfloat16, *NO_RESIDUAL_SHAPE)
    if bare is None or accumulating is None:
        pytest.skip("the fused epilogue needs SM80+ and the CuTeDSL sources or CUBINs")
    mha_o, gate, weight, _ = _no_residual_operands(1, 1, 40, bias=False)
    residual = gate.new_zeros((*gate.shape[:-1], NO_RESIDUAL_SHAPE[2]))
    with torch.inference_mode():
        # Each op serves only the residual state it was built for.
        assert bare(mha_o, gate, weight, residual) is None
        assert accumulating(mha_o, gate, weight, None) is None


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
