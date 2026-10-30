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
    create_conditioned_transition_block_weights,
    load_conditioned_transition_block_weights_torch,
)
from test_utils.boltz.ref_layers import RefConditionedTransitionBlock

from bionemo_ir._torch.custom_ops.dual_gemm_x_x import cutedsl as dual_gemm_x_x_cutedsl
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import ops as dual_gemm_x_x_ops
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock, MSATransition, PairTransition, Transition
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryExecutable
from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import SM_VERSION, init_module_weights, skip_if_not_sm90, skip_if_not_sm100_family


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim_single: int = 768
    dim_single_cond: int = 768
    expansion_factor: int = 2
    torch_dtype: str = "float32"
    seq_len: int = 128


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim_single=768, dim_single_cond=768),
        Scenario(dim_single=768, dim_single_cond=768, torch_dtype="bfloat16"),
        Scenario(dim_single=768, dim_single_cond=768, seq_len=256),
        Scenario(dim_single=768, dim_single_cond=768, torch_dtype="bfloat16", seq_len=256),
    ],
)
def test_conditioned_transition_block(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_cond_trans = RefConditionedTransitionBlock.load_weights()
    ref_cond_trans = ref_cond_trans.to(device)

    weights_and_biases = create_conditioned_transition_block_weights(from_ref=ref_cond_trans)

    cond_trans = ConditionedTransitionBlock(
        dim_single=sc.dim_single, dim_single_cond=sc.dim_single_cond, expansion_factor=sc.expansion_factor, dtype=dtype
    )
    load_conditioned_transition_block_weights_torch(cond_trans, weights_and_biases, dtype=dtype)
    cond_trans.to(device)

    a = torch.randn(bs, sc.seq_len, sc.dim_single, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_cond_trans(a, s)
        a = a.to(dtype)
        s = s.to(dtype)
        ref_cond_trans = ref_cond_trans.to(dtype)

        ref_output = ref_cond_trans(a, s)
        output = cond_trans.forward(a, s)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 1.0
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("mult", [3, 5], ids=["S3", "S5"])
def test_conditioned_transition_block_broadcast(mult: int):
    """Cond input `s` is shared (size-1 multiplicity dim) while `a` varies.

    Exercises the broadcast path of the fused gated-sigmoid output gate:
    ``a`` is ``[B, S, I, d]`` and ``s`` is ``[B, 1, I, d_cond]``.
    """
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs, seq_len, dim = 2, 96, 768
    dtype = torch.bfloat16
    device = torch.device("cuda")

    ref_cond_trans = RefConditionedTransitionBlock.load_weights().to(device)
    weights_and_biases = create_conditioned_transition_block_weights(from_ref=ref_cond_trans)

    cond_trans = ConditionedTransitionBlock(dim_single=dim, dim_single_cond=dim, expansion_factor=2, dtype=dtype)
    load_conditioned_transition_block_weights_torch(cond_trans, weights_and_biases, dtype=dtype)
    cond_trans.to(device)

    a = torch.randn(bs, mult, seq_len, dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, 1, seq_len, dim, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_cond_trans(a, s)
        a = a.to(dtype)
        s = s.to(dtype)
        ref_cond_trans = ref_cond_trans.to(dtype)

        ref_output = ref_cond_trans(a, s)
        output = cond_trans.forward(a, s)

    assert output.shape == a.shape
    diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
    diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
    diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
    diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))
    assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 1.0
    assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("torch_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("chunk_rows", [32, 24], ids=["even", "partial"])
def test_transition_auto_chunk(torch_dtype: str, chunk_rows: int):
    """``Transition.auto_chunk_policy`` row-chunks the pair FFN identically to the dense path.

    The FFN is position-wise, so slicing dim=1 and concatenating is numerically identical (same
    instance dense vs chunked, no golden weights needed). Also checks the rank gate: a rank-3
    single-rep activation is left on the dense path (``min_rank=4``).
    """
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(torch_dtype)
    dim, hidden, n = 128, 512, 64

    policy = ChunkPolicy(chunk_size=chunk_rows, min_size=1, dim=1, min_rank=4)
    transition = Transition(dim, hidden, dtype=dtype, auto_chunk_policy=policy).to(device)
    transition.eval()
    # The chunked path itself: the LayerNorm-fused kernel would take the whole map in one call.
    transition._w_norm_op = None
    # Constructed weights are zero-initialized (production loads them); give them real values so
    # the dense-vs-chunked comparison is meaningful rather than 0 == 0.
    with torch.no_grad():
        for p in transition.parameters():
            p.normal_(mean=0.0, std=0.1)

    tol = {"atol": 1e-4, "rtol": 1e-4} if dtype == torch.float32 else {"atol": 3e-3, "rtol": 3e-3}

    # Rank-4 pair activation [B, N, N, dim]: policy trips -> chunked path must match dense.
    z = torch.randn(1, n, n, dim, dtype=dtype, device=device)
    with torch.inference_mode():
        dense = transition._forward_impl(z)
        chunked = transition(z)
    assert chunked.shape == dense.shape
    torch.testing.assert_close(chunked, dense, **tol)

    # Rank-3 single-rep activation [B, N, dim]: rank gate keeps it on the dense path.
    s = torch.randn(1, n, dim, dtype=dtype, device=device)
    assert not policy.should_chunk(s)
    with torch.inference_mode():
        torch.testing.assert_close(transition(s), transition._forward_impl(s), **tol)


@pytest.mark.parametrize(
    ("transition_type", "kwargs"),
    [
        (PairTransition, {"c_z": 32, "n": 4}),
        (MSATransition, {"c_m": 32, "n": 4}),
    ],
    ids=["pair", "msa"],
)
def test_relu_transition_skip_create_weights_defers_linears(
    transition_type: type[PairTransition] | type[MSATransition],
    kwargs: dict[str, int],
) -> None:
    transition = transition_type(
        **kwargs,
        skip_create_weights=True,
        enable_cudnn_graph=True,
        cudnn_dynamic_shapes=True,
    )

    for linear in (transition.linear_1, transition.linear_2):
        assert not linear._weights_created
        assert dict(linear.named_parameters()) == {}
    assert transition._cudnn_graph_plans == {}

    transition.to(dtype=torch.bfloat16)
    assert transition._cudnn_graph_plans == {}


@pytest.mark.parametrize("chunk_rows", [8, 7], ids=["even", "partial"])
@pytest.mark.parametrize("with_mask", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("residual", [False, True], ids=["no_residual", "residual"])
def test_pair_transition_auto_chunk(chunk_rows: int, with_mask: bool, residual: bool):
    """PairTransition chunks pair rows without changing its ReLU FFN."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    dim, expansion, n = 32, 4, 24
    policy = ChunkPolicy(chunk_size=chunk_rows, min_size=1, dim=1, min_rank=4)
    transition = PairTransition(c_z=dim, n=expansion, dtype=torch.float32, auto_chunk_policy=policy).to(device).eval()
    with torch.no_grad():
        for parameter in transition.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    z = torch.randn(1, n, n, dim, device=device)
    mask = torch.ones(1, n, n, dtype=torch.bool, device=device)
    mask[:, -3:, :] = False
    mask_or_none = mask if with_mask else None
    with torch.inference_mode():
        dense = transition._forward_impl(z, mask_or_none, residual=residual)
        chunked = transition(z, mask_or_none, residual=residual)

    torch.testing.assert_close(chunked, dense, atol=1e-5, rtol=1e-5)
    if with_mask and not residual:
        assert torch.count_nonzero(chunked[:, -3:]) == 0


@pytest.mark.parametrize("cudnn_dynamic_shapes", [False, True], ids=["static", "dynamic"])
def test_msa_transition_cudnn_graph_matches_vanilla(cudnn_dynamic_shapes: bool) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    channels, expansion = 256, 4
    vanilla = MSATransition(
        c_m=channels,
        n=expansion,
        dtype=torch.bfloat16,
        enable_cudnn_graph=False,
    ).to(device)
    fused = MSATransition(
        c_m=channels,
        n=expansion,
        dtype=torch.bfloat16,
        enable_cudnn_graph=True,
        cudnn_dynamic_shapes=cudnn_dynamic_shapes,
    ).to(device)
    # cuDNN against the plain linears, so neither takes the LayerNorm-fused kernel.
    vanilla._w_norm_op = fused._w_norm_op = None
    # Only the opt-in holds plans; the default path builds them per row count.
    assert set(fused._cudnn_graph_plans) == ({"linear_relu", "linear_mask"} if cudnn_dynamic_shapes else set())
    with torch.no_grad():
        for parameter in vanilla.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    fused.load_state_dict(vanilla.state_dict())

    m = torch.randn(1, 3, 17, channels, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 3, 17, device=device, dtype=torch.bool)
    mask[:, 1, -3:] = False
    with torch.inference_mode():
        expected = vanilla(m, mask)
        actual = fused(m, mask)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[:, 1, -3:]) == 0


def test_transition_normalize_false_skips_layernorm():
    torch.manual_seed(5)
    device = torch.device("cuda")
    dim, hidden = 32, 64
    module = Transition(dim, hidden, dtype=torch.float32, normalize=False).to(device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    assert module.norm is None
    assert "norm.weight" not in module.state_dict()

    x = torch.randn(2, 9, dim, device=device)
    mask = torch.ones(2, 9, device=device, dtype=torch.bool)
    mask[1, 6:] = False
    with torch.inference_mode():
        out = module(x, mask)
        expected = module.fc3(module._swiglu(module.fused_fc2_fc1(x)))
        expected = expected * mask.unsqueeze(-1)
    torch.testing.assert_close(out, expected)
    assert torch.count_nonzero(out[1, 6:]) == 0


class _CountingKernel:
    """Delegate to a kernel op and count the calls that reach it."""

    def __init__(self, op) -> None:
        self.op = op
        self.calls = 0

    def accepts(self, *args):
        return self.op.accepts(*args)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.op(*args, **kwargs)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("residual,inplace", [(False, False), (True, False), (True, True)])
def test_transition_fuses_each_chunk(batch: int, normalize: bool, masked: bool, residual: bool, inplace: bool) -> None:
    """A partial final chunk fuses its own residual, including non-contiguous batch slices.

    At C = 128, H = 256, the one SwiGLU shape SM90 still ships the transition MLP for, and only without the residual:
    SM90 runs residual calls as the LayerNorm-fused kernel, so those cases skip there.
    """
    skip_if_not_sm90()
    torch.manual_seed(29)
    layer = init_module_weights(
        Transition(
            128,
            256,
            normalize=normalize,
            dtype=torch.bfloat16,
            auto_chunk_policy=ChunkPolicy(chunk_size=5, min_size=1),
        ).cuda()
    )
    # The per-chunk transition MLP is the path for maps the LayerNorm-fused kernel declines.
    layer._w_norm_op = None
    if layer._fused_mlp_ops[(masked, residual)] is None:
        pytest.skip("this GPU ships no SwiGLU transition MLP for C=128, H=256 with this mask and residual")
    counter = _CountingKernel(layer._fused_mlp_ops[(masked, residual)])
    layer._fused_mlp_ops[(masked, residual)] = counter
    x = torch.randn(batch, 13, 17, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(batch, 13, 17, device="cuda") > 0.3).to(x.dtype) if masked else None
    original = x.clone()
    with torch.inference_mode():
        update = layer._forward_impl(x, mask)
        expected = x + update if residual else update
        output = layer(x, mask, residual=residual, inplace=inplace)
    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)
    assert counter.calls == 3
    assert (output.data_ptr() == x.data_ptr()) == inplace
    if not inplace:
        assert torch.equal(x, original)
    if masked and residual:
        assert torch.equal(output[mask == 0], original[mask == 0])


@pytest.mark.parametrize("layout", ["transposed", "misaligned"])
def test_transition_chunk_layout_and_graph_replay(layout: str) -> None:
    """Chunk fusion preserves aliasing and fallback for unsupported residual alignment.

    SM90 runs residual SwiGLU calls as the LayerNorm-fused kernel and ships no transition MLP for them, so it skips
    there.
    """
    skip_if_not_sm90()
    layer = init_module_weights(
        Transition(64, 128, dtype=torch.bfloat16, auto_chunk_policy=ChunkPolicy(chunk_size=5, min_size=1)).cuda()
    )
    if layer._fused_mlp_ops[(False, True)] is None:
        pytest.skip("this GPU ships no SwiGLU transition MLP for C=64, H=128 with the residual")
    counter = _CountingKernel(layer._fused_mlp_ops[(False, True)])
    layer._fused_mlp_ops[(False, True)] = counter
    storage = torch.randn(2, 17, 13, 64 if layout == "transposed" else 65, device="cuda", dtype=torch.bfloat16)
    x = storage.transpose(1, 2) if layout == "transposed" else storage[0:1, :, :, 1:]
    with torch.inference_mode():
        original = x.clone()
        expected = x + layer._forward_impl(x)
        output = layer(x, residual=True, inplace=True)
        assert output is x
        torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)
        if layout == "misaligned":
            assert counter.calls == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer(x, residual=True, inplace=True)
        x.copy_(-original)
        expected = x + layer._forward_impl(x)
        graph.replay()
        torch.testing.assert_close(x, expected, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("residual", [False, True])
def test_pair_transition_fuses_each_chunk(residual: bool) -> None:
    """PairTransition tries the ReLU kernel per slice instead of bypassing it under chunking.

    At C = 256, H = 512, the one ReLU shape SM90 ships the transition MLP for.
    """
    skip_if_not_sm90()
    layer = init_module_weights(
        PairTransition(256, 2, dtype=torch.bfloat16, auto_chunk_policy=ChunkPolicy(chunk_size=5, min_size=1)).cuda()
    )
    # The per-chunk transition MLP is the path for maps the LayerNorm-fused kernel declines.
    layer._w_norm_op = None
    counter = _CountingKernel(layer._fused_mlp_ops[residual])
    layer._fused_mlp_ops[residual] = counter
    x = torch.randn(2, 13, 17, 256, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(2, 13, 17, device="cuda") > 0.3).to(x.dtype)
    with torch.inference_mode():
        expected = layer._forward_impl(x, mask, residual=residual)
        output = layer(x, mask, residual=residual)
    assert counter.calls == 3
    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)
    if residual:
        assert torch.equal(output[mask == 0], x[mask == 0])


@pytest.mark.parametrize(
    ("dim_single", "using_silu", "kernel_attr"),
    [(128, True, "_fused_mlp_op"), (768, True, "_dual_gemm_silu_op"), (128, False, "_fused_mlp_op")],
    ids=["atom-128-fused-mlp", "token-768-dual-gemm", "atom-128-3way-fused-mlp"],
)
@pytest.mark.parametrize(
    "leading_shape",
    [(2, 64), (2, 2048), (2, 1, 2, 32), (2, 1, 2, 1024)],
    ids=["3d-small", "3d-large", "protenix-5d-small", "protenix-5d-large"],
)
def test_conditioned_transition_kernel_path_matches_split_projection(
    dim_single: int, using_silu: bool, kernel_attr: str, leading_shape: tuple[int, ...]
) -> None:
    """A SwiGLU block selects its shape's kernel path, matching the split projection.

    A 3-way block has no dual-GEMM path, so its unfused projection is the one it falls back to.
    """
    torch.manual_seed(7)
    module = (
        ConditionedTransitionBlock(
            dim_single=dim_single,
            dim_single_cond=128,
            expansion_factor=2,
            using_silu=using_silu,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    if getattr(module, kernel_attr) is None:
        pytest.skip(f"{kernel_attr} does not ship for width {dim_single} on this GPU")
    assert (module._dual_gemm_silu_op is not None) == using_silu
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)

    a = torch.randn(*leading_shape, dim_single, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(*leading_shape, 128, device="cuda", dtype=torch.bfloat16)
    kernel = _CountingKernel(getattr(module, kernel_attr))
    setattr(module, kernel_attr, kernel)
    with torch.inference_mode():
        fused = module(a, s)
        setattr(module, kernel_attr, None)
        split = module(a, s)

    rows = a.numel() // dim_single
    below_threshold = kernel_attr == "_dual_gemm_silu_op" and rows < module._dual_gemm_min_rows
    assert kernel.calls == (0 if below_threshold else 1)
    torch.testing.assert_close(fused, split, atol=2e-2, rtol=2e-2)


def _token_swiglu_block() -> ConditionedTransitionBlock:
    """A diffusion-token SwiGLU block (768 wide, 1536 hidden) with model-scale weights, or skip."""
    torch.manual_seed(11)
    module = (
        ConditionedTransitionBlock(
            dim_single=768, dim_single_cond=128, expansion_factor=2, using_silu=True, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    if module._dual_gemm_silu_op is None:
        pytest.skip("the 768/1536 SiLU dual GEMM does not ship on this GPU")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    return module


def _split_projection(module: ConditionedTransitionBlock, a: torch.Tensor) -> torch.Tensor:
    """The block's MLP on its unfused path: one projection, the SwiGLU pass, then ``b_to_a``."""
    return module.b_to_a(module._swiglu(module.fused_swl_a_to_b(a)))


@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["below", "at", "above"])
def test_conditioned_transition_dual_gemm_starts_at_its_min_rows(offset: int) -> None:
    """On SM100 the dual GEMM runs from its shipped row count; fewer rows take the unfused path, bit for bit."""
    skip_if_not_sm100_family()
    module = _token_swiglu_block()
    assert module._dual_gemm_min_rows == 2305
    kernel = _CountingKernel(module._dual_gemm_silu_op)
    module._dual_gemm_silu_op = kernel
    a = torch.randn(module._dual_gemm_min_rows + offset, 768, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        output = module._swiglu_mlp(a)
        split = _split_projection(module, a)

    if offset < 0:
        assert kernel.calls == 0
        torch.testing.assert_close(output, split, atol=0, rtol=0)
    else:
        assert kernel.calls == 1
        torch.testing.assert_close(output, split, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("leading_shape", "dual_gemm"),
    [((5, 42), False), ((1, 5, 42), False), ((1, 5, 373), False), ((5, 959), True), ((1, 5, 1734), True)],
    ids=["protenix-42", "of3-42", "of3-373", "protenix-959", "of3-1734"],
)
def test_conditioned_transition_token_mlp_path_on_blackwell(leading_shape: tuple[int, ...], dual_gemm: bool) -> None:
    """On SM100 the token MLP of a small complex runs unfused and a large one runs the dual GEMM.

    Protenix calls it on ``[5, N, 768]`` and OpenFold3 on ``[1, 5, N, 768]``; both have ``5 N`` rows.
    """
    skip_if_not_sm100_family()
    module = _token_swiglu_block()
    assert module._dual_gemm_min_rows > 0
    kernel = _CountingKernel(module._dual_gemm_silu_op)
    module._dual_gemm_silu_op = kernel
    a = torch.randn(*leading_shape, 768, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        output = module._swiglu_mlp(a)
        split = _split_projection(module, a)

    assert kernel.calls == int(dual_gemm)
    torch.testing.assert_close(output, split, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", ["forced", "source_free"])
def test_conditioned_transition_dual_gemm_runs_cubins_chosen_at_call_time(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block built with sources still runs its dual GEMM from the packaged CUBINs when they are all a call has.

    ``forced`` sets ``CUTEDSL_FORCE_CUBIN`` after construction; ``source_free`` makes the source adapter
    unimportable, as a public build ships it.
    """
    try:
        launcher = library_runtime._load_kernel_library().dual_gemm_x_x
    except library_runtime.CuTeDSLKernelLibraryUnavailable as error:
        pytest.skip(f"no _cutedsl_kernels extension: {error}")
    module = _token_swiglu_block()
    try:
        # 96 tokens select the S=128 bucket of every tuned 768/1536 bundle; no mask, no bias, the silu gate.
        launcher.make_kernel_config(SM_VERSION, 768, 1536, 128, launcher.DType.BFLOAT16, False, False, False, True)
    except ValueError:
        pytest.skip(f"the CUBIN corpus has no SM{SM_VERSION} 768/1536 silu dual GEMM")
    module._dual_gemm_min_rows = 0
    a = torch.randn(1, 5, 96, 768, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        split = _split_projection(module, a)

    monkeypatch.setattr(dual_gemm_x_x_cutedsl.DualGemmXxCuTe, "_compiled_cache", {})
    monkeypatch.setattr(dual_gemm_x_x_ops, "_DUAL_GEMM_XX_CUTE", None)
    if mode == "forced":
        monkeypatch.setenv(dual_gemm_x_x_cutedsl.FORCE_CUBIN_ENV, "1")
    else:
        monkeypatch.delenv(dual_gemm_x_x_cutedsl.FORCE_CUBIN_ENV, raising=False)

        def source_unavailable(*_args, **_kwargs):
            raise ModuleNotFoundError("CuTeDSL source removed, as in a public build")

        monkeypatch.setattr(dual_gemm_x_x_cutedsl, "load_source_module", source_unavailable)
    with torch.inference_mode():
        output = module._swiglu_mlp(a)

    assert isinstance(dual_gemm_x_x_ops._DUAL_GEMM_XX_CUTE._last_exe, CuTeDSLKernelLibraryExecutable)
    torch.testing.assert_close(output, split, atol=2e-2, rtol=2e-2)
