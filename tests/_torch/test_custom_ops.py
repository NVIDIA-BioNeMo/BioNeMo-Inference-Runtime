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

import pytest
import torch
import torch.nn as nn
import triton

from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import get_dual_gemm_x0_x1_op
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import get_dual_gemm_x_x_op
from bionemo_ir._torch.custom_ops.fused_masked_residual import fused_masked_residual
from bionemo_ir._torch.custom_ops.fused_pair_injection import PairInjection
from bionemo_ir.dsl_kernels.triton.fused_pair_injection import (
    _BLOCK_D_CANDIDATES,
    _pick_block_d,
    fused_pair_injection,
)

from . import make_left_aligned_pair_mask


def create_linear_layers(K: int = 128, N: int = 256, dtype=torch.bfloat16, has_bias=False):
    linear0 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear0.weight)
    if has_bias:
        torch.nn.init.uniform_(linear0.bias)

    linear1 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear1.weight)
    if has_bias:
        torch.nn.init.uniform_(linear1.bias)

    return linear0.to(dtype), linear1.to(dtype)


def test_fused_sigmoid_gated_dual_gemm():
    torch.manual_seed(42)
    seq_len = 256
    dtype = torch.bfloat16
    K = 128
    N = 256
    linear0, linear1 = create_linear_layers(K=K, N=N, dtype=dtype, has_bias=False)
    x = torch.randn(1, seq_len, seq_len, K, device="cuda").contiguous().to(dtype)
    # The default ``get_dual_gemm_x_x_op`` dispatch (pair_mask_left_aligned=
    # True) routes to the CuTe LM kernel, which masks via a per-row prefix
    # count -- a random binary mask violates that contract, so use a
    # left-aligned pair mask matching the kernel's assumption.
    mask = make_left_aligned_pair_mask(1, seq_len, dtype=dtype, device="cuda")
    ref_x = linear0(x).sigmoid() * linear1(x)
    ref_x = ref_x * mask.unsqueeze(-1)

    op = get_dual_gemm_x_x_op(dtype, transpose_out=False, N=N, K=K)
    out = op(x, linear0.weight, linear1.weight, linear0.bias, linear1.bias, mask)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)

    x = torch.randn(1, seq_len, seq_len, K, device="cuda").contiguous().to(dtype)
    linear0, linear1 = create_linear_layers(K=K, N=N, dtype=dtype, has_bias=True)

    ref_x = linear0(x).sigmoid() * linear1(x)
    ref_x = ref_x * mask.unsqueeze(-1)

    op = get_dual_gemm_x_x_op(dtype, transpose_out=False, N=N, K=K)
    out = op(x, linear0.weight, linear1.weight, linear0.bias, linear1.bias, mask)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)


def test_fused_sigmoid_gated_dual_gemm_dual_x():
    torch.manual_seed(42)
    seq_len = 256
    dtype = torch.bfloat16
    K = 128
    N = 256
    linear0, linear1 = create_linear_layers(K=K, N=N, dtype=dtype, has_bias=False)
    x0 = torch.randn(1, seq_len, seq_len, K, device="cuda").contiguous().to(dtype)
    x1 = torch.randn(1, seq_len, seq_len, K, device="cuda").contiguous().to(dtype)
    ref_x = linear0(x0).sigmoid() * linear1(x1)

    op = get_dual_gemm_x0_x1_op(dtype, transpose_out=False, N=N, K=K)
    out = op(x0, x1, linear0.weight, linear1.weight, linear0.bias, linear1.bias, None)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)

    linear0, linear1 = create_linear_layers(K=K, N=N, dtype=dtype, has_bias=True)
    ref_x = linear0(x0).sigmoid() * linear1(x1)
    op = get_dual_gemm_x0_x1_op(dtype, transpose_out=False, N=N, K=K)
    out = op(x0, x1, linear0.weight, linear1.weight, linear0.bias, linear1.bias, None)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Fused sequence-to-pair injection
# ---------------------------------------------------------------------------

#: Token counts and pair widths that are not multiples of the kernel's tiles, so
#: the partial tiles along both axes carry masked lanes. ``1`` degenerates both
#: axes, and ``385`` and ``130`` are a residue past a tile boundary.
_UNALIGNED_TOKENS = (1, 7, 17, 31, 33, 63)
_UNALIGNED_PAIR_DIMS = (1, 48, 100, 130, 196, 385)


def _injection_operands(
    batch_size: int,
    tokens: int,
    pair_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build injection operands with the layout ``RFProteinaPairUpdate`` passes.

    ``row`` and ``column`` are halves of one chunked projection, so their token
    stride is ``2 * pair_dim`` rather than ``pair_dim``. Building them as fresh
    contiguous tensors would test a layout the module never produces.
    """
    pair = torch.randn(batch_size, tokens, tokens, pair_dim, device="cuda", dtype=dtype)
    projection = torch.randn(batch_size, tokens, 2 * pair_dim, device="cuda", dtype=dtype)
    row, column = projection.chunk(2, dim=-1)
    mask = make_left_aligned_pair_mask(batch_size, tokens, dtype=dtype).unsqueeze(-1)
    return pair, row, column, mask


def _eager_injection(
    pair: torch.Tensor,
    row: torch.Tensor,
    column: torch.Tensor,
    mask: torch.Tensor,
    pair_write: str,
) -> torch.Tensor:
    """The PyTorch chain the kernel replaces, kept op for op."""
    if pair_write == "sum":
        injected = pair + row.unsqueeze(1)
        injected.add_(column.unsqueeze(2))
    else:
        injected = pair + row.unsqueeze(1) * column.unsqueeze(2)
    return injected.mul_(mask)


@pytest.mark.parametrize("pair_write", ["sum", "hadamard"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fused_pair_injection_matches_eager_chain_unaligned(
    pair_write: str,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    """The kernel is bit-identical to the eager chain off the tile boundaries.

    Bit-identical rather than close is the contract: the kernel substitutes for
    the eager chain inside a released rollout, so any rounding difference would
    move the structures it generates. That is also why the hadamard product is
    rounded before the add instead of being contracted into an FMA, which would
    be the more accurate answer but not the one upstream computes.
    """
    torch.manual_seed(31)
    for tokens in _UNALIGNED_TOKENS:
        for pair_dim in _UNALIGNED_PAIR_DIMS:
            operands = _injection_operands(batch_size, tokens, pair_dim, dtype)
            expected = _eager_injection(*operands, pair_write)
            actual = fused_pair_injection(*operands, hadamard=pair_write == "hadamard")
            mismatched = int((actual != expected).sum().item())
            assert mismatched == 0, (
                f"{mismatched} elements differ at B={batch_size}, N={tokens}, C={pair_dim}, "
                f"pair_write={pair_write}, dtype={dtype}"
            )


@pytest.mark.parametrize("pair_write", ["sum", "hadamard"])
def test_pair_injection_custom_op_matches_its_eager_fallback(pair_write: str) -> None:
    """The custom op's fused and fallback paths agree bit for bit.

    Dropping ``_fused_kernel`` is how the fallback is reached on a CUDA device,
    and comparing against it proves the fused path is the one being exercised
    rather than the test passing on a silent fallback.
    """
    torch.manual_seed(37)
    pair_dim, tokens, batch_size = 130, 17, 2
    operands = _injection_operands(batch_size, tokens, pair_dim, torch.bfloat16)

    fused = PairInjection(pair_dim=pair_dim, pair_write=pair_write).cuda()
    assert fused._fused_kernel is not None, "the kernel should compile on a CUDA device"
    eager = PairInjection(pair_dim=pair_dim, pair_write=pair_write).cuda()
    eager._fused_kernel = None

    with torch.inference_mode():
        actual = fused(*operands)
        expected = eager(*operands)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Injection must hand back owned storage; the caller accumulates into it.
    assert actual.data_ptr() != operands[0].data_ptr()


@pytest.mark.parametrize("pair_dim", [*_UNALIGNED_PAIR_DIMS, 128, 256, 384, 512])
def test_pair_injection_tile_heuristic_minimizes_idle_lanes(pair_dim: int) -> None:
    """The picked feature tile wastes no more lanes than any candidate would."""
    picked = _pick_block_d(pair_dim)
    assert picked == triton.next_power_of_2(picked), "the tile must be a power of two"

    def idle_lanes(block_d: int) -> int:
        return triton.cdiv(pair_dim, block_d) * block_d - pair_dim

    candidates = [block_d for block_d in _BLOCK_D_CANDIDATES if block_d <= pair_dim]
    if candidates:
        assert idle_lanes(picked) == min(idle_lanes(block_d) for block_d in candidates)
    else:
        # Narrower than every candidate, so one rounded-up tile covers the width.
        assert triton.cdiv(pair_dim, picked) == 1


def _residual_operands(
    batch_size: int, tokens: int, pair_dim: int, dtype: torch.dtype, *, token_pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a residual's operands, optionally with the update behind a pad.

    ``token_pad`` rounds the update's token axes up and returns a view into
    that larger buffer, which is exactly what a token-padded trimul hands back.
    The surrounding buffer is filled with values that must not be read.
    """
    pair = torch.randn(batch_size, tokens, tokens, pair_dim, device="cuda", dtype=dtype)
    padded_tokens = -(-tokens // token_pad) * token_pad if token_pad > 1 else tokens
    buffer = torch.randn(batch_size, padded_tokens, padded_tokens, pair_dim, device="cuda", dtype=dtype)
    update = buffer[:, :tokens, :tokens]
    mask = (torch.rand(batch_size, tokens, tokens, 1, device="cuda") > 0.2).to(dtype)
    return pair, update, mask


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("token_pad", [1, 8])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_fused_masked_residual_matches_eager_chain(dtype: torch.dtype, token_pad: int, batch_size: int) -> None:
    """The kernel is bit-identical to ``pair.add_(update).mul_(mask)``.

    ``token_pad=8`` makes the update a strided view into a padded buffer, the
    layout this kernel exists to take without a repack. The result must not
    depend on that layout, and must not pick up the surrounding pad.
    """
    torch.manual_seed(41)
    for tokens in _UNALIGNED_TOKENS:
        for pair_dim in _UNALIGNED_PAIR_DIMS:
            pair, update, mask = _residual_operands(batch_size, tokens, pair_dim, dtype, token_pad=token_pad)
            expected = pair.clone().add_(update).mul_(mask)
            actual = fused_masked_residual(pair.clone(), update, mask)
            assert actual is not None, "the kernel should cover these operands"
            mismatched = int((actual != expected).sum().item())
            assert mismatched == 0, (
                f"{mismatched} elements differ at B={batch_size}, N={tokens}, C={pair_dim}, "
                f"token_pad={token_pad}, dtype={dtype}"
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("token_pad", [1, 8])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_fused_masked_residual_matches_single_residual(dtype: torch.dtype, token_pad: int, batch_size: int) -> None:
    """The same kernel covers single-representation residuals."""
    torch.manual_seed(42)
    tokens, channels = 17, 130
    value = torch.randn(batch_size, tokens, channels, device="cuda", dtype=dtype)
    padded_tokens = -(-tokens // token_pad) * token_pad if token_pad > 1 else tokens
    buffer = torch.randn(batch_size, padded_tokens, channels, device="cuda", dtype=dtype)
    update = buffer[:, :tokens]
    mask = (torch.rand(batch_size, tokens, 1, device="cuda") > 0.2).to(dtype)

    expected = value.clone().add_(update).mul_(mask)
    actual = fused_masked_residual(value.clone(), update, mask)

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_masked_residual_accumulates_in_place() -> None:
    """The residual writes through to the pair storage it was handed.

    Its caller chains several of these into one buffer, so returning a fresh
    tensor would silently double the pair representation's peak memory.
    """
    torch.manual_seed(43)
    pair, update, mask = _residual_operands(1, 17, 130, torch.bfloat16, token_pad=8)
    actual = fused_masked_residual(pair, update, mask)
    assert actual is not None
    assert actual.data_ptr() == pair.data_ptr()


def test_fused_masked_residual_declines_operands_it_cannot_address() -> None:
    """An unsupported layout returns ``None`` so the caller can fall back."""
    pair, update, mask = _residual_operands(1, 17, 130, torch.bfloat16, token_pad=8)
    # Channels strided is the one axis the kernel requires packed.
    assert fused_masked_residual(pair, update[..., ::2], mask[..., :1]) is None
    assert fused_masked_residual(pair, update, mask.to(torch.float32)) is None
    assert fused_masked_residual(pair.cpu(), update.cpu(), mask.cpu()) is None
