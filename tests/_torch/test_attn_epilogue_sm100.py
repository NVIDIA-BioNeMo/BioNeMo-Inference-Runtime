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
"""The SM100 and SM103 attention epilogues: the CuTeDSL triangle and token kernels against the unfused layer."""

import itertools
import sys

import pytest
import torch

from bionemo_ir._torch.custom_ops.attn_epilogue import AttnEpilogue, get_attn_epilogue_op
from bionemo_ir._torch.custom_ops.attn_epilogue import cutedsl as epilogue_cutedsl
from bionemo_ir._torch.custom_ops.attn_epilogue import ops as epilogue_ops
from bionemo_ir._torch.custom_ops.attn_epilogue._config import (
    SM100_OUTPUT_GATE_WIDTHS,
    SM100_TOKEN_TUNING,
    TOKEN_WIDTH,
    nearest_anchor,
    tuned_configs,
)
from bionemo_ir._torch.custom_ops.attn_epilogue._cubin import (
    AttnEpilogueCubinExecutable,
    AttnEpilogueTokenCubinExecutable,
)
from bionemo_ir._torch.custom_ops.attn_epilogue.ops import BLACKWELL_SMS, SLAB, AttnEpilogueToken
from bionemo_ir._torch.layers.attention import AttentionPairBias, MSAAttention
from bionemo_ir._torch.layers.triangle_nodes import TriangleAttentionNode, TriangleAttentionNodeType
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryError, populate_compiled_cache_from_library
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    make_left_aligned_pair_mask,
    require_cubin_family,
    source_module_available,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and SM_VERSION not in BLACKWELL_SMS, reason="the kernels are SM100/103"
    ),
]
_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.attn_epilogue._source"
MODES = cutedsl_test_modes(_SOURCE_MODULE)
_COMPILER_MODULES = ("cutlass", "tvm_ffi")
# The token kernel's view of the attention output: H*D / SLAB heads of SLAB channels.
_TOKEN_VIEW = (TOKEN_WIDTH // SLAB, SLAB)
# (C = H*D, has_bias, has_output_gate) of every triangle kernel: the bias at both widths, the gate where it fits.
TRIANGLE_VARIANTS = [
    (channels, has_bias, has_output_gate)
    for channels in (128, 256)
    for has_bias, has_output_gate in itertools.product(
        (False, True), (False, True) if channels in SM100_OUTPUT_GATE_WIDTHS else (False,)
    )
]


def _sm100_cubins_shipped() -> bool:
    """Whether the kernel library carries every image this GPU's tunings name, each at its anchor: the triangle
    kernels, biased and gated ones included, the token kernels with and without the bias, and the SM80 kernel at
    each small-call anchor of a config file."""
    sm = SM_VERSION
    token_tunings = tuned_configs(sm, *_TOKEN_VIEW, TOKEN_WIDTH)
    probes = [
        (
            lambda library, launcher, variant=variant, rows=tuning.rows: AttnEpilogueCubinExecutable(
                library, launcher, sm, variant[0] // SLAB, SLAB, *variant, rows
            ),
            tuning,
            False,
        )
        for variant in TRIANGLE_VARIANTS
        for tuning in tuned_configs(sm, variant[0] // SLAB, SLAB, variant[0])
    ]
    probes += [
        (
            lambda library, launcher, has_bias=has_bias, rows=tuning.rows: AttnEpilogueCubinExecutable(
                library, launcher, sm, *_TOKEN_VIEW, TOKEN_WIDTH, has_bias, True, rows
            ),
            tuning,
            False,
        )
        for has_bias in (False, True)
        for tuning in token_tunings[:-1]
    ]
    probes += [
        (
            lambda library, launcher, pairs=pairs, has_bias=has_bias: AttnEpilogueTokenCubinExecutable(
                library, launcher, sm, *_TOKEN_VIEW, pairs, has_bias, rows=token_tunings[-1].rows
            ),
            token_tunings[-1],
            pairs,
        )
        for pairs in (False, True)
        for has_bias in (False, True)
    ]
    try:
        for load, tuning, pairs in probes:
            spec = populate_compiled_cache_from_library({}, (), "attn_epilogue", load)._config.spec
            # A library built before a tuning's anchor existed serves the nearest image instead.
            if (spec.bucket, spec.kernel_sm, spec.pairs) != (
                tuning.rows,
                80 if tuning.kernel_abi == "sm80" else 100,
                pairs,
            ):
                return False
    except CuTeDSLKernelLibraryError:
        return False
    return True


def _require_sm100_cubins() -> None:
    """Skip unless the kernel library carries every attention epilogue CUBIN this GPU's tunings name."""
    require_cubin_family("attn_epilogue")
    if not _sm100_cubins_shipped():
        pytest.skip("this _cutedsl_kernels extension carries no SM100 attention epilogue CUBINs yet")


def _force_cubins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the kernels from their CUBINs, with the CuTeDSL compiler out of reach as in a release install."""
    _require_sm100_cubins()
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    for name in {*(name for name in sys.modules if name.partition(".")[0] in _COMPILER_MODULES), *_COMPILER_MODULES}:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture(params=MODES)
def kernel_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Serve the CuTeDSL kernels from their source or from their CUBINs.

    Each test starts with empty executable caches: an unforced call returns whatever a cached key holds, so a CUBIN
    an earlier ``cubin`` test on the same worker loaded would stand in for the source kernel.
    """
    monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
    for backend in (epilogue_cutedsl.AttnEpilogueCuTe, epilogue_cutedsl.AttnEpilogueTokenCuTe):
        monkeypatch.setattr(backend, "_compiled_cache", {})
    if request.param == "cubin":
        _force_cubins(monkeypatch)
    elif not source_module_available(_SOURCE_MODULE):
        pytest.fail("source test mode requires the CuTeDSL kernel sources")
    return request.param


requires_kernel = pytest.mark.usefixtures("kernel_mode")
requires_source = pytest.mark.skipif(
    not source_module_available(_SOURCE_MODULE), reason="the CuTeDSL kernel sources are not installed"
)

BF16 = {"device": "cuda", "dtype": torch.bfloat16}


def _assert_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1.6e-2)
    # The kernels keep the unfused rounding points, so only the GEMM's summation order may move a rounding; a
    # rounding point of their own would move far more.
    assert (actual == expected).float().mean() > 0.99


def _replayed(op: AttnEpilogue, *operands: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
    """Capture one call into a fresh destination in a CUDA graph, replay it and return the destination."""
    output = torch.empty_like(operands[3])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*operands, output, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    return output


class _NoKernel:
    """A backend whose kernel must not run."""

    def executable(self, device_index: int):
        raise AssertionError("the CuTeDSL kernel ran")


# ---- The triangle attention's epilogue (H*D = C of 128 or 256, residual, optional bias and output gate) ----

# (heads, head_dim, channels): OpenFold3's triangle attention, Protenix-v2's, and another split of Protenix's.
SHAPES = [(4, 32, 128), (8, 32, 256), (4, 64, 256)]
# (B*I, J): a ragged J below one 128-row tile, an odd tile count that leaves a pair's second CTA idle, a
# ragged J over several tiles, and more tiles than CTA pairs, so each pair loops.
SIZES = [(1, 77), (1, 300), (3, 130), (37, 1000)]
NODE_TYPES = [TriangleAttentionNodeType.STARTING, TriangleAttentionNodeType.ENDING]


def _operands(shape: tuple[int, int, int], rows: int, columns: int, transposed: bool, batch: int = 1):
    heads, head_dim, channels = shape
    width = heads * head_dim
    torch.manual_seed(batch * rows * columns + width)
    mha_o = torch.randn(batch * rows, columns, heads, head_dim, **BF16)
    # The gate is a column slice of the fused [q; k; v; g] projection.
    gate = torch.randn(batch, rows, columns, 4 * width, **BF16)[..., 3 * width :]
    weight = (torch.randn(channels, width, device="cuda") * width**-0.5).to(torch.bfloat16)
    # The ending node's residual is the transposed pair.
    if transposed:
        residual = torch.randn(batch, columns, rows, channels, **BF16).transpose(1, 2)
    else:
        residual = torch.randn(batch, rows, columns, channels, **BF16)
    return mha_o, gate, weight, residual


def _unfused(
    mha_o: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return residual + torch.nn.functional.linear(mha_o.reshape(gate.shape) * gate.sigmoid(), weight, bias)


def test_blackwell_takes_the_cute_epilogue_for_triangle_widths(kernel_mode: str) -> None:
    for (heads, head_dim, channels), has_bias in itertools.product(SHAPES, (False, True)):
        gates = (False, True) if channels in SM100_OUTPUT_GATE_WIDTHS else (False,)
        for has_output_gate in gates:
            op = get_attn_epilogue_op(
                torch.bfloat16, heads, head_dim, channels, has_bias=has_bias, has_output_gate=has_output_gate
            )
            assert type(op) is AttnEpilogue
            assert (op._has_bias, op._has_output_gate) == (has_bias, has_output_gate)
            # One backend per anchor: the SM80 kernel's small-call anchors where a config file names them, then
            # the native kernel's.
            tunings = tuned_configs(SM_VERSION, channels // SLAB, SLAB, channels)
            assert op._anchors == [tuning.rows for tuning in tunings]
            assert tunings[-1].kernel_abi == "sm100"
            for backend in op._backends.values():
                executable = backend.executable(torch.cuda.current_device())
                assert isinstance(executable, AttnEpilogueCubinExecutable) == (kernel_mode == "cubin")
        # No residual keeps the unfused path.
        assert get_attn_epilogue_op(torch.bfloat16, heads, head_dim, channels, has_residual=False) is None
    # So does an output gate at C = 256, whose logit slots do not fit beside that width's Wo.
    assert get_attn_epilogue_op(torch.bfloat16, 8, 32, 256, has_output_gate=True) is None
    # So do untuned widths and other dtypes.
    assert get_attn_epilogue_op(torch.bfloat16, 4, 16, 64) is None
    assert get_attn_epilogue_op(torch.bfloat16, 4, 128, 128) is None
    assert get_attn_epilogue_op(torch.float32, 4, 32, 128) is None


@requires_kernel
@pytest.mark.parametrize(("channels", "has_bias", "has_output_gate"), TRIANGLE_VARIANTS, ids=str)
def test_each_anchor_serves_the_calls_nearest_it(channels: int, has_bias: bool, has_output_gate: bool) -> None:
    """A call at each anchor runs that anchor's kernel, the SM80 kernel at a config file's small anchors and the
    native kernel above them, and matches the unfused layer."""
    shape = (channels // SLAB, SLAB, channels)
    op = get_attn_epilogue_op(torch.bfloat16, *shape, has_bias=has_bias, has_output_gate=has_output_gate)
    backends = dict(op._backends)
    for tuning in tuned_configs(SM_VERSION, *shape):
        # A square call at the anchor: the triangle attention's pair rows.
        size = max(2, round(tuning.rows**0.5))
        assert nearest_anchor(op._anchors, size * size) == tuning.rows
        op._backends = {
            anchor: backend if anchor == tuning.rows else _NoKernel() for anchor, backend in backends.items()
        }
        mha_o, gate, weight, residual = _operands(shape, size, size, transposed=False)
        bias = _bias(channels) if has_bias else None
        logits = _logits(1, size, size) if has_output_gate else None
        with torch.inference_mode():
            if has_output_gate:
                expected = _gated_unfused(mha_o, gate, weight, residual, logits, bias)
            else:
                expected = _unfused(mha_o, gate, weight, residual, bias)
            actual = op(mha_o, gate, weight, residual, bias=bias, output_gate=logits)
        assert actual is not None, tuning
        _assert_matches(actual, expected)


@requires_kernel
@pytest.mark.parametrize("shape", SHAPES, ids=["c128", "c256", "c256_h4"])
@pytest.mark.parametrize(("rows", "columns"), SIZES, ids=lambda size: str(size))
@pytest.mark.parametrize("transposed", [False, True], ids=["starting", "ending"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
def test_matches_the_unfused_layer(
    shape: tuple[int, int, int], rows: int, columns: int, transposed: bool, inplace: bool
) -> None:
    op = get_attn_epilogue_op(torch.bfloat16, *shape)
    mha_o, gate, weight, residual = _operands(shape, rows, columns, transposed)
    with torch.inference_mode():
        expected = _unfused(mha_o, gate, weight, residual)
        target = residual.clone() if inplace else residual
        actual = op(mha_o, gate, weight, target, target if inplace else None)

    assert actual is not None
    _assert_matches(actual, expected)
    if inplace:
        assert actual is target


@requires_kernel
@pytest.mark.parametrize("shape", SHAPES[:2], ids=["c128", "c256"])
def test_batched_starting_node_matches_the_unfused_layer(shape: tuple[int, int, int]) -> None:
    """B = 2 folds into the pair rows: B*I rows of the kernel's views."""
    op = get_attn_epilogue_op(torch.bfloat16, *shape)
    mha_o, gate, weight, residual = _operands(shape, 3, 300, False, batch=2)
    with torch.inference_mode():
        expected = _unfused(mha_o, gate, weight, residual)
        actual = op(mha_o, gate, weight, residual)

    assert actual is not None
    _assert_matches(actual, expected)


# (heads, head_dim, channels) of the biased output projections: OpenFold2's triangle and MSA attentions.
BIAS_SHAPES = [(4, 32, 128), (8, 32, 256)]
# The atom transformers' attention, whose update the adaLN-Zero output gate scales.
GATE_SHAPE = (4, 32, 128)
# (batch, rows, gate_rows, columns): pair row i of the folded B*I rows reads output-gate row i // (I / I_y). One
# gate row for every row below one tile, a partial broadcast with B = 2 folded, one gate row per pair row over an
# odd tile count, and a partial broadcast over more tiles than CTA pairs, so each pair loops.
GATE_CASES = [(1, 5, 1, 77), (2, 4, 2, 300), (1, 3, 3, 300), (1, 40, 8, 1000)]
GATE_IDS = ["broadcast", "partial", "rowwise", "partial_looping"]


def _bias(channels: int) -> torch.Tensor:
    # As large as the update, so adding it before or after the update's rounding gives different outputs.
    return torch.randn(channels, **BF16)


def _logits(batch: int, gate_rows: int, columns: int) -> torch.Tensor:
    return (2 * torch.randn(batch, gate_rows, columns, GATE_SHAPE[2], device="cuda")).to(torch.bfloat16)


def _gated_unfused(
    mha_o: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The SM90 kernel's output gate: the gated update joins the residual add, which rounds once."""
    update = torch.nn.functional.linear(mha_o.reshape(gate.shape) * gate.sigmoid(), weight, bias)
    output_gate = logits.sigmoid().repeat_interleave(residual.shape[1] // logits.shape[1], dim=1)
    return (residual.float() + output_gate.float() * update.float()).to(torch.bfloat16)


@requires_kernel
@pytest.mark.parametrize("shape", BIAS_SHAPES, ids=["c128", "c256"])
@pytest.mark.parametrize(("rows", "columns"), SIZES, ids=lambda size: str(size))
@pytest.mark.parametrize("transposed", [False, True], ids=["starting", "ending"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
def test_bias_matches_the_unfused_layer(
    shape: tuple[int, int, int], rows: int, columns: int, transposed: bool, inplace: bool
) -> None:
    """The bias joins the fp32 accumulator before the update rounds to bf16, as the unfused projection adds it."""
    op = get_attn_epilogue_op(torch.bfloat16, *shape, has_bias=True)
    mha_o, gate, weight, residual = _operands(shape, rows, columns, transposed)
    bias = _bias(shape[2])
    with torch.inference_mode():
        expected = _unfused(mha_o, gate, weight, residual, bias)
        target = residual.clone() if inplace else residual
        actual = op(mha_o, gate, weight, target, target if inplace else None, bias=bias)

    assert actual is not None
    _assert_matches(actual, expected)
    if inplace:
        assert actual is target


@requires_kernel
@pytest.mark.parametrize(("batch", "rows", "gate_rows", "columns"), GATE_CASES, ids=GATE_IDS)
@pytest.mark.parametrize("has_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
def test_output_gate_matches_the_unfused_layer(
    batch: int, rows: int, gate_rows: int, columns: int, has_bias: bool, inplace: bool
) -> None:
    op = get_attn_epilogue_op(torch.bfloat16, *GATE_SHAPE, has_bias=has_bias, has_output_gate=True)
    mha_o, gate, weight, residual = _operands(GATE_SHAPE, rows, columns, False, batch=batch)
    logits = _logits(batch, gate_rows, columns)
    bias = _bias(GATE_SHAPE[2]) if has_bias else None
    with torch.inference_mode():
        expected = _gated_unfused(mha_o, gate, weight, residual, logits, bias)
        target = residual.clone() if inplace else residual
        actual = op(mha_o, gate, weight, target, target if inplace else None, bias=bias, output_gate=logits)

    assert actual is not None
    _assert_matches(actual, expected)
    if inplace:
        assert actual is target


def _assert_gated_add_rounds_in_fp32_first(op, mha_o, gate, weight, residual, logits) -> None:
    """``update * gate + z`` rounds to fp32 and then to bf16, not once to bf16.

    With a residual far below the gated update, the fp32 sum drops it, and a product that sits exactly on a bf16
    tie rounds to even as it does without a residual; a single rounding would let the residual break the tie.
    Random operands of similar magnitudes never meet this case.
    """
    tiny = torch.where(torch.rand_like(residual, dtype=torch.float32) < 0.5, -1e-30, 1e-30).to(torch.bfloat16)
    with torch.inference_mode():
        bare = op(mha_o, gate, weight, torch.zeros_like(residual), output_gate=logits)
        actual = op(mha_o, gate, weight, tiny, output_gate=logits)

    assert bare is not None and actual is not None
    # Where the gated update is exactly zero, the sum is the residual itself.
    assert torch.equal(actual, torch.where(bare == 0, tiny, bare))


@requires_kernel
def test_output_gate_rounds_the_gated_add_in_fp32_first() -> None:
    """The triangle kernel's gated add rounds to fp32 first; see :func:`_assert_gated_add_rounds_in_fp32_first`."""
    op = get_attn_epilogue_op(torch.bfloat16, *GATE_SHAPE, has_output_gate=True)
    _assert_gated_add_rounds_in_fp32_first(op, *_operands(GATE_SHAPE, 4, 300, False), _logits(1, 1, 300))


@requires_kernel
def test_flagged_epilogue_declines_calls_it_was_not_built_for() -> None:
    """A biased or gated op takes exactly its bias or output gate, a 16-byte aligned bias, and gate rows dividing
    the pair rows."""
    biased = get_attn_epilogue_op(torch.bfloat16, *GATE_SHAPE, has_bias=True)
    gated = get_attn_epilogue_op(torch.bfloat16, *GATE_SHAPE, has_output_gate=True)
    biased._backends = gated._backends = dict.fromkeys(biased._backends, _NoKernel())
    mha_o, gate, weight, residual = _operands(GATE_SHAPE, 3, 77, False)
    bias = _bias(GATE_SHAPE[2])
    logits = _logits(1, 3, 77)
    misaligned = torch.randn(GATE_SHAPE[2] + 8, **BF16)[1 : 1 + GATE_SHAPE[2]]
    with torch.inference_mode():
        assert biased(mha_o, gate, weight, residual) is None
        assert biased(mha_o, gate, weight, residual, bias=misaligned) is None
        assert gated(mha_o, gate, weight, residual) is None
        assert gated(mha_o, gate, weight, residual, bias=bias, output_gate=logits) is None
        assert gated(mha_o, gate, weight, residual, output_gate=logits[:, :2]) is None
        # The operands each op was built for reach its kernel.
        for op, flags in ((biased, {"bias": bias}), (gated, {"output_gate": logits})):
            with pytest.raises(AssertionError, match="the CuTeDSL kernel ran"):
                op(mha_o, gate, weight, residual, **flags)


def _node(
    node_type: TriangleAttentionNodeType,
    heads: int,
    head_dim: int,
    channels: int,
    chunk_size: int,
    bias: bool = False,
):
    torch.manual_seed(0)
    node = TriangleAttentionNode(
        c_in=channels,
        c_hidden=head_dim,
        num_heads=heads,
        node_type=node_type,
        dtype=torch.bfloat16,
        attn_backend="SDPA",
        chunk_policy=ChunkPolicy(chunk_size=chunk_size, min_size=1) if chunk_size else ChunkPolicy(enabled=False),
        mha_bias_flags={"q": False, "k": False, "v": False, "g": bias, "o": bias},
    ).cuda()
    _init(node)
    assert type(node.mha._epilogue) is AttnEpilogue
    return node


@requires_kernel
@pytest.mark.parametrize(
    ("heads", "head_dim", "channels", "bias"),
    [(*SHAPES[0], False), (*SHAPES[1], False), (*BIAS_SHAPES[0], True)],
    ids=["c128", "c256", "c128_bias"],
)
@pytest.mark.parametrize("node_type", NODE_TYPES, ids=lambda t: t.name.lower())
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize(("seq_len", "chunk_size"), [(64, 0), (100, 0), (100, 24), (256, 0)])
def test_triangle_node_matches_separate_residual_add(
    heads: int,
    head_dim: int,
    channels: int,
    bias: bool,
    node_type: TriangleAttentionNodeType,
    inplace: bool,
    seq_len: int,
    chunk_size: int,
) -> None:
    """Row counts off the 128-row tiles, row chunks and the ending node's transposed pair, through the layer.

    OpenFold2's triangle attention biases its gate and output projections; the epilogue adds the output bias.
    """
    node = _node(node_type, heads, head_dim, channels, chunk_size, bias)
    assert node.mha._epilogue._has_bias is bias
    calls: list[int] = []
    node.mha.o_proj.register_forward_hook(lambda *_: calls.append(1))
    x = torch.randn(1, seq_len, seq_len, channels, **BF16)
    mask = make_left_aligned_pair_mask(1, seq_len, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        expected = x + node(x, mask)
        calls.clear()
        residual = x.clone()
        actual = node(residual, mask, residual=True, inplace_residual=inplace)

    assert not calls
    _assert_matches(actual, expected)
    if inplace:
        assert actual.data_ptr() == residual.data_ptr()
    else:
        assert torch.equal(residual, x)


def _init(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, parameter.shape[1] ** -0.5)
            else:
                parameter.normal_(1 if name.endswith("weight") else 0, 0.1)


@requires_kernel
@pytest.mark.parametrize("column", [False, True], ids=["row", "column"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
def test_msa_attention_adds_its_bias_and_residual_in_the_epilogue(column: bool, inplace: bool) -> None:
    """OpenFold2's MSA row and column attention: 8 heads of 32 projecting to 256 channels through a biased output
    projection. The column attention's residual is the transposed MSA."""
    torch.manual_seed(0)
    layer = MSAAttention(
        local_layer_idx=0,
        c_in=256,
        num_heads=8,
        c_z=128,
        triangle_attn_backend="SDPA",
        need_project_z=not column,
        transpose_input=column,
        dtype=torch.bfloat16,
    ).cuda()
    _init(layer)
    assert type(layer.mha._epilogue) is AttnEpilogue and layer.mha._epilogue._has_bias
    calls: list[int] = []
    layer.mha.o_proj.register_forward_hook(lambda *_: calls.append(1))
    m = torch.randn(1, 24, 100, 256, **BF16)
    z = None if column else torch.randn(1, 100, 100, 128, **BF16)
    mask = torch.ones(1, 24, 100, **BF16)
    with torch.inference_mode():
        expected = m + layer(m, z, mask)
        calls.clear()
        residual = m.clone()
        actual = layer(residual, z, mask, residual=True, inplace_residual=inplace)

    assert not calls
    _assert_matches(actual, expected)
    if inplace:
        assert actual.data_ptr() == residual.data_ptr()
    else:
        assert torch.equal(residual, m)


@requires_kernel
@pytest.mark.parametrize(("batch", "samples"), [(1, 1), (1, 5), (2, 3)], ids=["one_sample", "samples", "batched"])
def test_atom_attention_gates_its_update_in_the_epilogue(
    monkeypatch: pytest.MonkeyPatch, batch: int, samples: int
) -> None:
    """The atom transformers' pair-bias attention (4 heads of 32, C = 128) runs its adaLN-Zero output gate in the
    epilogue, one gate row serving every sample of its token."""
    torch.manual_seed(0)
    attention = AttentionPairBias(
        layer_idx=0,
        c_s=GATE_SHAPE[2],
        c_z=16,
        num_heads=GATE_SHAPE[0],
        initial_norm=False,
        bias_proj=True,
        output_gate_dim=GATE_SHAPE[2],
        dtype=torch.bfloat16,
        attn_backend="SDPA",
    ).cuda()
    _init(attention)
    epilogue = attention._epilogue
    assert type(epilogue) is AttnEpilogue and epilogue._has_output_gate and not epilogue._has_bias
    tokens = 96
    s = torch.randn(batch, samples, tokens, GATE_SHAPE[2], **BF16)
    # The conditioning holds one row per token, broadcast over the samples.
    condition = torch.randn(batch, 1, tokens, GATE_SHAPE[2], **BF16)
    z = torch.randn(batch, tokens, tokens, 16, **BF16)
    mask = torch.ones(batch, tokens, **BF16)
    residual = torch.randn_like(s)
    served: list[bool] = []
    call = AttnEpilogue.__call__

    def counting(self: AttnEpilogue, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor | None:
        output = call(self, *args, **kwargs)
        served.append(output is not None)
        return output

    monkeypatch.setattr(AttnEpilogue, "__call__", counting)
    with torch.inference_mode():
        actual = attention(s, z, mask, single_embedding=condition, residual=residual)
        attention._epilogue = None
        expected = attention(s, z, mask, single_embedding=condition, residual=residual)

    assert served == [True]
    # The unfused layer rounds the gated update before the residual add; the epilogue rounds their sum once.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@requires_source
@pytest.mark.parametrize(
    ("channels", "has_bias", "has_output_gate"),
    [variant for variant in TRIANGLE_VARIANTS if any(variant[1:])],
    ids=lambda value: str(value),
)
def test_forcing_cubins_takes_the_biased_and_gated_cubins(
    monkeypatch: pytest.MonkeyPatch, channels: int, has_bias: bool, has_output_gate: bool
) -> None:
    """Each anchor's biased or gated triangle CUBIN, the SM80 kernel's and the native one's, reproduces its source
    kernel bit for bit, also from a CUDA graph."""
    shape = (channels // 32, 32, channels)
    op = get_attn_epilogue_op(torch.bfloat16, *shape, has_bias=has_bias, has_output_gate=has_output_gate)
    calls = []
    for anchor in op._anchors:
        # A square call at the anchor, so that anchor's kernel serves it.
        size = max(2, round(anchor**0.5))
        flags = {}
        if has_bias:
            flags["bias"] = _bias(channels)
        if has_output_gate:
            flags["output_gate"] = _logits(1, size, size)
        calls.append((anchor, _operands(shape, size, size, False), flags))
    with torch.inference_mode():
        # Forcing the CUBINs cannot be undone, so every source result comes first.
        expected = [op(*operands, **flags) for _, operands, flags in calls]
        _force_cubins(monkeypatch)
        for (anchor, operands, flags), want in zip(calls, expected, strict=True):
            assert nearest_anchor(op._anchors, operands[1].shape[1] * operands[1].shape[2]) == anchor
            assert torch.equal(op(*operands, **flags), want)
            assert isinstance(op._backends[anchor].executable(torch.cuda.current_device()), AttnEpilogueCubinExecutable)
            assert torch.equal(_replayed(op, *operands, **flags), want)


@requires_source
def test_forcing_cubins_after_construction_takes_the_cubin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A layer builds its op once, so the switch is read per call, also inside a CUDA graph capture; the CUBIN
    reproduces the source kernel bit for bit."""
    shape = SHAPES[1]
    op = get_attn_epilogue_op(torch.bfloat16, *shape)
    # A call at each anchor: the SM80 kernel's small-call ones where a config file names them, then the native one.
    calls = [
        (anchor, _operands(shape, size, size, False)) for anchor in op._anchors for size in [max(2, round(anchor**0.5))]
    ]
    with torch.inference_mode():
        expected = [op(*operands) for _, operands in calls]
        _force_cubins(monkeypatch)
        for (anchor, operands), want in zip(calls, expected, strict=True):
            assert torch.equal(op(*operands), want)
            assert isinstance(op._backends[anchor].executable(torch.cuda.current_device()), AttnEpilogueCubinExecutable)
            assert torch.equal(_replayed(op, *operands), want)


@requires_kernel
def test_cute_epilogue_replays_from_a_cuda_graph() -> None:
    """Building the op compiles or loads its kernel, so capture launches without compiling."""
    shape = SHAPES[1]
    op = get_attn_epilogue_op(torch.bfloat16, *shape)
    operands = _operands(shape, 2, 200, False)
    with torch.inference_mode():
        expected = op(*operands)
        assert torch.equal(_replayed(op, *operands), expected)


@requires_kernel
def test_epilogue_declines_operands_it_cannot_take_without_a_copy() -> None:
    shape = SHAPES[0]
    op = get_attn_epilogue_op(torch.bfloat16, *shape)
    mha_o, gate, weight, residual = _operands(shape, 8, 40, False)
    channels = shape[2]
    with torch.inference_mode():
        # A heads-outer attention output, and a residual whose rows are not 16-byte aligned.
        heads_outer = mha_o.transpose(1, 2).contiguous().transpose(1, 2)
        assert op(heads_outer, gate, weight, residual) is None
        misaligned = torch.randn(1, 8, 40, channels + 4, **BF16)[..., 4:]
        assert op(mha_o, gate, weight, misaligned) is None
        # A contiguous weight whose base is not 16-byte aligned.
        flat = torch.randn(weight.numel() + 8, **BF16)
        assert op(mha_o, gate, flat[1 : 1 + weight.numel()].view(weight.shape), residual) is None
        assert op(mha_o, gate, weight, residual) is not None


# ---- The token transformers' epilogue (H*D = C = 768, residual and output gate, optional bias) ----

# (heads, head_dim, channels) of Protenix's and OpenFold3's diffusion token transformer.
TOKEN_SHAPE = (16, 48, 768)
# RFProteína's trunk, whose output projection has a bias.
RFPROTEINA_SHAPE = (12, 64, 768)
SAMPLES = 5


def _token_operands(
    tokens: int,
    gate_rows: int,
    *,
    batch: int = 1,
    samples: int = SAMPLES,
    padded: int = 0,
    shape: tuple[int, int, int] = TOKEN_SHAPE,
) -> tuple[torch.Tensor, ...]:
    """The layer's operands; ``padded`` extra tokens per sample make the gate's samples stride unevenly, so its
    rows no longer read as one flat view."""
    heads, head_dim, width = shape
    torch.manual_seed(batch * tokens * gate_rows + padded)
    mha_o = torch.randn(batch * samples, tokens, heads, head_dim, **BF16)
    # The gate is a column slice of the fused [q; g; k; v] projection.
    gate = torch.randn(batch, samples, tokens + padded, 4 * width, **BF16)[:, :, :tokens, width : 2 * width]
    weight = (torch.randn(width, width, device="cuda") * width**-0.5).to(torch.bfloat16)
    residual = torch.randn(batch, samples, tokens, width, **BF16)
    # Output-gate logits, broadcast over the samples (one row) or one per sample.
    logits = (2 * torch.randn(batch, gate_rows, tokens, width, device="cuda") - 1).to(torch.bfloat16)
    return mha_o, gate, weight, residual, logits


def _token_op(
    monkeypatch: pytest.MonkeyPatch,
    variant: str | None = None,
    clusters: int | None = None,
    *,
    has_bias: bool = False,
    shape: tuple[int, int, int] = TOKEN_SHAPE,
):
    """The token op; ``variant`` serves every call with that kernel, small calls included, and ``clusters`` caps its
    persistent grid (a small cap makes each cluster loop over several tiles)."""
    if clusters is not None:
        monkeypatch.setattr(epilogue_ops, "max_token_clusters", lambda heads, head_dim, device, force_cubin: clusters)
    if variant is not None:
        monkeypatch.setattr(epilogue_ops, "_token_variant", lambda rows, clusters: variant)
        # Keep only the token kernel's own tuning, at its anchor, so no call takes the SM80 kernel's small anchors.
        tunings = epilogue_ops.tuned_configs
        monkeypatch.setattr(epilogue_ops, "tuned_configs", lambda *shape: tunings(*shape)[-1:])
    op = get_attn_epilogue_op(torch.bfloat16, *shape, has_bias=has_bias, has_output_gate=True)
    assert isinstance(op, AttnEpilogueToken)
    return op


def test_blackwell_takes_the_cute_epilogue_for_the_token_width(kernel_mode: str) -> None:
    """Both head splits of the width take the token kernel, with or without the bias (RFProteína's trunk has one)."""
    for shape in (TOKEN_SHAPE, RFPROTEINA_SHAPE):
        for has_bias in (False, True):
            op = get_attn_epilogue_op(torch.bfloat16, *shape, has_bias=has_bias, has_output_gate=True)
            assert isinstance(op, AttnEpilogueToken)
            assert op._has_bias is has_bias
            assert set(op._variants) == set(SM100_TOKEN_TUNING[SM_VERSION])
            for backend in op._variants.values():
                executable = backend.executable(torch.cuda.current_device())
                assert isinstance(executable, AttnEpilogueTokenCubinExecutable) == (kernel_mode == "cubin")
    # No output gate or no residual keeps the unfused path, bias or not.
    for flags in ({}, {"has_bias": True}, {"has_output_gate": False, "has_residual": False}):
        assert get_attn_epilogue_op(torch.bfloat16, *TOKEN_SHAPE, **flags) is None


@requires_kernel
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_small_token_calls_take_the_tuned_sm80_kernel(has_bias: bool) -> None:
    """Calls nearer a config file's small anchors than the token kernel's run that anchor's SM80 kernel; a call at
    the token kernel's anchor runs the token kernel. Both match the unfused layer."""
    tunings = tuned_configs(SM_VERSION, *_TOKEN_VIEW, TOKEN_WIDTH)
    op = get_attn_epilogue_op(torch.bfloat16, *TOKEN_SHAPE, has_bias=has_bias, has_output_gate=True)
    assert isinstance(op, AttnEpilogueToken)
    small = [tuning.rows for tuning in tunings[:-1]]
    assert op._anchors == small
    # Calls nearer an SM80 anchor than the token kernel's: up to their midpoint, a tie going to SM80.
    assert op._small_rows == ((small[-1] + tunings[-1].rows) // 2 if small else -1)
    small, variants = dict(op._backends), dict(op._variants)
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    for tuning in tunings:
        token_kernel = tuning is tunings[-1]
        op._backends = {anchor: backend if anchor == tuning.rows else _NoKernel() for anchor, backend in small.items()}
        op._variants = variants if token_kernel else dict.fromkeys(variants, _NoKernel())
        mha_o, gate, weight, residual, logits = _token_operands(max(1, tuning.rows // SAMPLES), SAMPLES)
        with torch.inference_mode():
            expected = _gated_unfused(mha_o, gate, weight, residual, logits, bias)
            actual = op(mha_o, gate, weight, residual, bias=bias, output_gate=logits)
        assert actual is not None, tuning
        _assert_matches(actual, expected)


@pytest.mark.parametrize(
    ("tokens", "variant"),
    [(1, "triples"), (250, "triples"), (563, "triples"), (564, "pairs"), (1127, "pairs"), (1153, "pairs")],
)
def test_token_variant_by_call_size(tokens: int, variant: str) -> None:
    """One wave of 22 six-CTA clusters (B200) runs triples over up to 22 row tiles; larger calls run pairs."""
    assert epilogue_ops._token_variant(SAMPLES * tokens, 22) == variant


@requires_kernel
def test_no_cluster_fits_leaves_the_layer_unfused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Where no six-CTA cluster fits (a small partition of the GPU), the token layer keeps the unfused path."""
    monkeypatch.setattr(epilogue_ops, "max_token_clusters", lambda heads, head_dim, device, force_cubin: 0)
    assert get_attn_epilogue_op(torch.bfloat16, *TOKEN_SHAPE, has_output_gate=True) is None


@requires_kernel
def test_cluster_count_is_queried_once_per_device() -> None:
    """The occupancy query loads a probe or the token kernel's CUBIN, so each device pays for it once."""
    device = torch.cuda.current_device()
    force_cubin = CuteKernelCache.force_cubin()
    query = epilogue_cutedsl.max_token_clusters
    query.cache_clear()
    first = query(*_TOKEN_VIEW, device, force_cubin)
    assert first >= 1
    assert query(*_TOKEN_VIEW, device, force_cubin) == first
    assert query.cache_info().misses == 1


@requires_source
def test_the_token_cubin_counts_the_clusters_the_cutedsl_runtime_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the CuTeDSL runtime the token CUBIN sizes the persistent grid: to the same clusters."""
    device = torch.cuda.current_device()
    query = epilogue_cutedsl.max_token_clusters
    query.cache_clear()
    expected = query(*_TOKEN_VIEW, device, False)
    _force_cubins(monkeypatch)
    assert query(*_TOKEN_VIEW, device, True) == expected


@requires_source
def test_a_forced_run_counts_the_clusters_with_the_token_cubin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forced CUBINs size the grid with the token CUBIN even where the CuTeDSL runtime is importable, and each
    mode keeps its own count."""
    _require_sm100_cubins()
    device = torch.cuda.current_device()
    query = epilogue_cutedsl.max_token_clusters
    query.cache_clear()
    loads = []
    load_cubin = epilogue_cutedsl.AttnEpilogueTokenCuTe._load_cubin_executable

    def counting_load_cubin(self, *args):
        loads.append(self)
        return load_cubin(self, *args)

    monkeypatch.setattr(epilogue_cutedsl.AttnEpilogueTokenCuTe, "_load_cubin_executable", counting_load_cubin)
    expected = query(*_TOKEN_VIEW, device, False)
    assert not loads
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    assert query(*_TOKEN_VIEW, device, True) == expected
    assert loads
    assert query.cache_info().currsize == 2


@requires_kernel
@pytest.mark.parametrize("variant", ["triples", "pairs"])
@pytest.mark.parametrize(
    ("tokens", "clusters"),
    # Less than a tile, a ragged tile, an odd tile count (a pair's second CTA duplicates the last tile), several
    # tiles, and a grid of two clusters that loops over many tiles.
    [(2, None), (77, None), (130, None), (373, None), (1000, 2)],
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("gate_rows", [1, SAMPLES], ids=["broadcast_gate", "gate_per_sample"])
@pytest.mark.parametrize("inplace", [False, True], ids=["out_of_place", "inplace"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_token_epilogue_matches_the_unfused_layer(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    tokens: int,
    clusters: int | None,
    gate_rows: int,
    inplace: bool,
    has_bias: bool,
) -> None:
    op = _token_op(monkeypatch, variant, clusters, has_bias=has_bias)
    mha_o, gate, weight, residual, logits = _token_operands(tokens, gate_rows)
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    with torch.inference_mode():
        expected = _gated_unfused(mha_o, gate, weight, residual, logits, bias)
        target = residual.clone() if inplace else residual
        actual = op(mha_o, gate, weight, target, target if inplace else None, bias=bias, output_gate=logits)

    assert actual is not None
    _assert_matches(actual, expected)
    if inplace:
        assert actual is target


@requires_kernel
@pytest.mark.parametrize(
    ("batch", "samples", "gate_rows"), [(1, 5, 1), (2, 4, 2), (2, 3, 3)], ids=["broadcast", "partial", "rowwise"]
)
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_output_gate_rows_serve_their_share_of_pair_rows(
    batch: int, samples: int, gate_rows: int, has_bias: bool
) -> None:
    """Pair row ``i`` reads output-gate row ``i // (I / I_y)``, also with B = 2 folded into the rows."""
    op = get_attn_epilogue_op(torch.bfloat16, *TOKEN_SHAPE, has_bias=has_bias, has_output_gate=True)
    mha_o, gate, weight, residual, logits = _token_operands(150, gate_rows, batch=batch, samples=samples)
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    with torch.inference_mode():
        expected = _gated_unfused(mha_o, gate, weight, residual, logits, bias)
        actual = op(mha_o, gate, weight, residual, bias=bias, output_gate=logits)

    assert actual is not None
    _assert_matches(actual, expected)


@requires_kernel
@pytest.mark.parametrize(("tokens", "gate_rows"), [(77, 1), (373, SAMPLES), (1000, 1)], ids=lambda value: str(value))
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_token_variants_agree_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, tokens: int, gate_rows: int, has_bias: bool
) -> None:
    """Triples and pairs keep the same rounding points and sum K in the same order."""
    operands = _token_operands(tokens, gate_rows)
    mha_o, gate, weight, residual, logits = operands
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    outputs = []
    for variant in ("triples", "pairs"):
        op = _token_op(monkeypatch, variant, has_bias=has_bias)
        with torch.inference_mode():
            outputs.append(op(mha_o, gate, weight, residual, bias=bias, output_gate=logits))

    assert all(output is not None for output in outputs)
    assert torch.equal(*outputs)


@requires_kernel
@pytest.mark.parametrize("variant", ["triples", "pairs"])
@pytest.mark.parametrize("shape", [TOKEN_SHAPE, RFPROTEINA_SHAPE], ids=["h16_d48", "h12_d64"])
def test_token_bias_joins_the_accumulator_before_it_rounds(
    monkeypatch: pytest.MonkeyPatch, variant: str, shape: tuple[int, int, int]
) -> None:
    """``bf16(acc + b)`` with the bias on its own channel, the unfused layer's rounding, bit for bit.

    Each row's attention output is zero but for one channel and both gates saturate to exactly 1, so every
    accumulator is one exact product and the expected output needs no GEMM. Rounding the accumulator before adding
    the bias, or adding the bias to another channel, changes many outputs; with random operands such differences
    hide among the GEMM's own summation-order roundings.
    """
    op = _token_op(monkeypatch, variant, has_bias=True, shape=shape)
    mha_o, gate, weight, residual, logits = _token_operands(373, 1, shape=shape)
    bias = _bias(TOKEN_WIDTH)
    rows, width = mha_o.shape[0] * mha_o.shape[1], shape[0] * shape[1]
    channel = torch.arange(rows, device="cuda") % width
    values = torch.randn(rows, **BF16)
    one_hot = torch.zeros(rows, width, **BF16)
    one_hot[torch.arange(rows, device="cuda"), channel] = values
    # sigmoid(20) rounds to 1 in bf16, with the kernels' approximations too.
    gate.fill_(20)
    logits.fill_(20)
    update = (values.float()[:, None] * weight.float()[:, channel].T + bias.float()).to(torch.bfloat16)
    expected = (residual.float() + update.float().view(residual.shape)).to(torch.bfloat16)
    with torch.inference_mode():
        actual = op(one_hot.view(mha_o.shape), gate, weight, residual, bias=bias, output_gate=logits)

    assert actual is not None
    assert torch.equal(actual, expected)


@requires_kernel
@pytest.mark.parametrize("variant", ["triples", "pairs"])
def test_token_epilogue_rounds_the_gated_add_in_fp32_first(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    """The token kernel's gated add rounds to fp32 first; see :func:`_assert_gated_add_rounds_in_fp32_first`."""
    _assert_gated_add_rounds_in_fp32_first(_token_op(monkeypatch, variant), *_token_operands(373, 1))


@requires_kernel
def test_token_epilogue_declines_samples_that_do_not_step_evenly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate whose samples are not evenly strided has no flat row view, so the op leaves the call unfused."""
    op = _token_op(monkeypatch, "pairs")
    op._variants = dict.fromkeys(op._variants, _NoKernel())
    mha_o, gate, weight, residual, logits = _token_operands(373, 1, padded=3)
    with torch.inference_mode():
        assert op(mha_o, gate, weight, residual, output_gate=logits) is None


@requires_source
@pytest.mark.parametrize("variant", ["triples", "pairs"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_forcing_cubins_after_construction_takes_the_token_cubin(
    monkeypatch: pytest.MonkeyPatch, variant: str, has_bias: bool
) -> None:
    """A layer builds its op once, so the switch is read per call, also inside a CUDA graph capture; the CUBIN
    reproduces the source kernel bit for bit."""
    op = _token_op(monkeypatch, variant, has_bias=has_bias)
    *operands, logits = _token_operands(373, 1)
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    with torch.inference_mode():
        expected = op(*operands, bias=bias, output_gate=logits)
        _force_cubins(monkeypatch)
        assert torch.equal(op(*operands, bias=bias, output_gate=logits), expected)
        executable = op._variants[variant].executable(torch.cuda.current_device())
        assert isinstance(executable, AttnEpilogueTokenCubinExecutable)
        assert torch.equal(_replayed(op, *operands, bias=bias, output_gate=logits), expected)


@requires_kernel
@pytest.mark.parametrize("variant", ["triples", "pairs"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_token_epilogue_replays_from_a_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, variant: str, has_bias: bool
) -> None:
    """Building the op compiles or loads its kernels, so capture launches without compiling."""
    op = _token_op(monkeypatch, variant, has_bias=has_bias)
    *operands, logits = _token_operands(959, 1)
    bias = _bias(TOKEN_WIDTH) if has_bias else None
    with torch.inference_mode():
        expected = op(*operands, bias=bias, output_gate=logits)
        assert torch.equal(_replayed(op, *operands, bias=bias, output_gate=logits), expected)
