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

import threading

import pytest
import torch
import torch.nn as nn
import triton

import bionemo_ir.dsl_kernels.triton.fused_pair_injection as pair_injection_triton
from bionemo_ir._torch.custom_ops.dual_gemm_x0_x1 import get_dual_gemm_x0_x1_op
from bionemo_ir._torch.custom_ops.dual_gemm_x_x import get_dual_gemm_x_x_op
from bionemo_ir._torch.custom_ops.fused_masked_residual import fused_masked_residual
from bionemo_ir._torch.custom_ops.fused_pair_injection import PairInjection
from bionemo_ir.dsl_kernels.triton.fused_pair_injection import (
    _BLOCK_D_CANDIDATES,
    _BLOCK_J_DEFAULT,
    FusedPairInjection,
    _pick_block_d,
    _pick_launch_config,
    _resolve_device_launch_config,
    _resolve_launch_config,
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
_UNALIGNED_PAIR_DIMS = (1, 48, 100, 130, 196, 384, 385)


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
    cache_key = fused._fused_kernel._last_launch.cache_key
    assert cache_key is not None
    major, minor = torch.cuda.get_device_capability(operands[0].device)
    assert cache_key[:2] == (operands[0].device.index, major * 10 + minor)


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


@pytest.mark.parametrize("sm_version", [100, 103])
def test_pair_injection_uses_exact_sm_tuning(sm_version: int) -> None:
    """Released Blackwell architectures use their measured C=384 constants."""
    config = _pick_launch_config(pair_dim=384, sm_version=sm_version)
    assert config == (32, 128, 4, 3)


def test_pair_injection_preserves_sm90_defaults() -> None:
    """Untuned architectures retain the H100 constants and Triton launch defaults."""
    config = _pick_launch_config(pair_dim=384, sm_version=90)
    assert config.block_j == _BLOCK_J_DEFAULT
    assert config.block_d == _pick_block_d(384)
    assert config.num_warps is None
    assert config.num_stages is None


@pytest.mark.parametrize("sm_version", [80, 89, 90, 107, 120, 121, 999])
@pytest.mark.parametrize("pair_dim", [128, 196, 384, 512])
def test_pair_injection_untuned_sms_preserve_defaults(sm_version: int, pair_dim: int) -> None:
    """Every architecture outside the exact table retains the prior defaults."""
    config = _pick_launch_config(pair_dim=pair_dim, sm_version=sm_version)
    assert config.block_j == _BLOCK_J_DEFAULT
    assert config.block_d == _pick_block_d(pair_dim)
    assert config.num_warps is None
    assert config.num_stages is None


@pytest.mark.parametrize("sm_version", [100, 103])
@pytest.mark.parametrize("pair_dim", [128, 196, 512])
def test_pair_injection_untuned_widths_preserve_defaults(sm_version: int, pair_dim: int) -> None:
    """Blackwell tuning applies only to the measured C=384 width."""
    config = _pick_launch_config(pair_dim=pair_dim, sm_version=sm_version)
    assert config.block_j == _BLOCK_J_DEFAULT
    assert config.block_d == _pick_block_d(pair_dim)
    assert config.num_warps is None
    assert config.num_stages is None


@pytest.mark.parametrize(
    ("block_j", "block_d", "expected"),
    [
        (8, None, (8, 128, None, None)),
        (None, 64, (16, 64, None, None)),
        (8, 64, (8, 64, None, None)),
    ],
)
def test_pair_injection_explicit_overrides_bypass_tuning(
    block_j: int | None,
    block_d: int | None,
    expected: tuple[int, int, None, None],
) -> None:
    """Diagnostic tile overrides never inherit tuned launch options."""
    assert _resolve_launch_config(384, block_j, block_d, sm_version=100) == expected


def test_pair_injection_uses_executing_device_in_heterogeneous_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Functional dispatch must not reuse logical GPU 0's architecture."""
    sms = {0: 100, 1: 90}
    monkeypatch.setattr(
        pair_injection_triton,
        "get_sm_version",
        lambda device=None: sms[torch.device(device).index if not isinstance(device, int) else device],
    )
    assert _resolve_device_launch_config(384, None, None, torch.device("cuda:0")) == (32, 128, 4, 3)
    assert _resolve_device_launch_config(384, None, None, torch.device("cuda:1")) == (16, 128, None, None)


class _FakeCudaDeviceContext:
    """Model CUDA current-device restoration without requiring multiple GPUs."""

    def __init__(self, device: torch.device | int, state: dict[str, int], events: list[tuple[str, int]]) -> None:
        self.device = device if isinstance(device, int) else device.index
        assert self.device is not None
        self.state = state
        self.events = events
        self.previous = state["current"]

    def __enter__(self) -> None:
        self.events.append(("enter", self.device))
        self.state["current"] = self.device

    def __exit__(self, *_exc: object) -> None:
        self.events.append(("exit", self.device))
        self.state["current"] = self.previous


class _FakeCudaTensor:
    """Tensor contract used by pair-injection validation and launch code."""

    def __init__(self, shape: tuple[int, ...], device: int = 1) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.device = torch.device("cuda", device)
        self.dtype = torch.bfloat16
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        self._strides = tuple(reversed(strides))

    def is_contiguous(self) -> bool:
        return True

    def get_device(self) -> int:
        return self.device.index

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self) -> int:
        return id(self)


@pytest.mark.parametrize(
    ("sm_version", "pair_dim", "block_j", "block_d", "expected"),
    [
        (100, 384, None, None, (32, 128, 4, 3)),
        (103, 384, None, None, (32, 128, 4, 3)),
        (90, 384, None, None, (16, 128, None, None)),
        (100, 196, None, None, (16, 128, None, None)),
        (100, 384, 8, 64, (8, 64, None, None)),
    ],
)
def test_pair_injection_functional_launch_uses_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    pair_dim: int,
    block_j: int | None,
    block_d: int | None,
    expected: tuple[int, int, int | None, int | None],
) -> None:
    """The functional launch consumes the exact table, fallback, or override."""
    state = {"current": 0}
    events: list[tuple[str, int]] = []
    launched = {}

    class FakeKernel:
        def __getitem__(self, grid: tuple[int, int, int]):
            def launch(*_args: object, **kwargs: int | bool) -> None:
                launched.update(current_device=state["current"], grid=grid, kwargs=kwargs)

            return launch

    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: _FakeCudaDeviceContext(device, state, events),
    )
    monkeypatch.setattr(pair_injection_triton, "get_sm_version", lambda _device: sm_version)
    monkeypatch.setattr(pair_injection_triton, "_fused_pair_injection_kernel", FakeKernel())

    pair = _FakeCudaTensor((1, 17, 17, pair_dim))
    row = _FakeCudaTensor((1, 17, pair_dim))
    column = _FakeCudaTensor((1, 17, pair_dim))
    mask = _FakeCudaTensor((1, 17, 17, 1))
    output = _FakeCudaTensor(pair.shape)
    fused_pair_injection(
        pair,
        row,
        column,
        mask,
        hadamard=True,
        output=output,
        block_j=block_j,
        block_d=block_d,
    )

    expected_j, expected_d, expected_warps, expected_stages = expected
    kwargs = launched["kwargs"]
    assert launched["current_device"] == pair.device.index
    assert kwargs["BLOCK_J"] == expected_j
    assert kwargs["BLOCK_D"] == expected_d
    assert kwargs.get("num_warps") == expected_warps
    assert kwargs.get("num_stages") == expected_stages
    assert state["current"] == 0
    assert events == [("enter", 1), ("exit", 1)]


@pytest.mark.parametrize(
    ("sm_version", "block_j", "block_d", "expected"),
    [
        (100, None, None, (32, 128, 4, 3)),
        (103, None, None, (32, 128, 4, 3)),
        (90, None, None, (16, 128, None, None)),
        (100, 8, 64, (8, 64, None, None)),
    ],
)
def test_pair_injection_class_compile_uses_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    block_j: int | None,
    block_d: int | None,
    expected: tuple[int, int, int | None, int | None],
) -> None:
    """The cached API passes the selected constants to its compile boundary."""
    state = {"current": 1}
    events: list[tuple[str, int]] = []
    compiled = {}

    def compile_for_dtypes(_self: FusedPairInjection, _kernel: object, **kwargs: object) -> dict:
        compiled.update(kwargs)
        return {torch.bfloat16: object()}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: _FakeCudaDeviceContext(device, state, events),
    )
    monkeypatch.setattr(pair_injection_triton, "get_sm_version", lambda _device: sm_version)
    monkeypatch.setattr(FusedPairInjection, "_global_cache", {})
    monkeypatch.setattr(FusedPairInjection, "compile_for_dtypes", compile_for_dtypes)

    kernel = FusedPairInjection(384, hadamard=True, block_j=block_j, block_d=block_d)
    expected_j, expected_d, expected_warps, expected_stages = expected
    assert kernel._last_launch.config == expected
    assert compiled["BLOCK_J"] == expected_j
    assert compiled["BLOCK_D"] == expected_d
    assert compiled.get("num_warps") == expected_warps
    assert compiled.get("num_stages") == expected_stages


@pytest.mark.parametrize("use_driver", [False, True])
def test_pair_injection_class_launch_uses_operand_device(
    monkeypatch: pytest.MonkeyPatch,
    use_driver: bool,
) -> None:
    """Both cached launch paths use the operand device and restore the caller."""
    state = {"current": 0}
    events: list[tuple[str, int]] = []
    launches = []

    class FakeDriver:
        def launch_with(self, _values: tuple[int, ...], *_grid: int) -> None:
            launches.append(("driver", state["current"]))

    class FakeCachedKernel:
        driver = FakeDriver() if use_driver else None

        def launch(self, _grid: tuple[int, int, int], *_args: object) -> None:
            launches.append(("fallback", state["current"]))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: _FakeCudaDeviceContext(device, state, events),
    )
    monkeypatch.setattr(pair_injection_triton, "get_sm_version", lambda _device: 100)
    monkeypatch.setattr(
        FusedPairInjection,
        "_global_cache",
        {(1, 100, 384, True, torch.bfloat16, 32, 128, 4, 3): {torch.bfloat16: FakeCachedKernel()}},
    )

    kernel = FusedPairInjection(384, hadamard=True)
    pair = _FakeCudaTensor((1, 17, 17, 384))
    row = _FakeCudaTensor((1, 17, 384))
    column = _FakeCudaTensor((1, 17, 384))
    mask = _FakeCudaTensor((1, 17, 17, 1))
    output = _FakeCudaTensor(pair.shape)
    kernel(pair, row, column, mask, output=output)

    assert launches == [(("driver" if use_driver else "fallback"), 1)]
    assert kernel._last_launch.config == (32, 128, 4, 3)
    assert state["current"] == 0
    assert events == [("enter", 1), ("exit", 1)]


def test_pair_injection_shared_instance_keeps_device_kernel_through_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent calls keep the kernel selected for each operand device."""
    thread_state = threading.local()
    thread_state.current = 0
    device_zero_ready = threading.Event()
    device_one_done = threading.Event()
    launches: list[tuple[int, int]] = []

    class ThreadCudaDeviceContext:
        def __init__(self, device: torch.device | int) -> None:
            self.device = device if isinstance(device, int) else device.index
            assert self.device is not None
            self.previous = getattr(thread_state, "current", 0)

        def __enter__(self) -> None:
            thread_state.current = self.device

        def __exit__(self, *_exc: object) -> None:
            thread_state.current = self.previous

    class FakeDriver:
        def __init__(self, compiled_device: int) -> None:
            self.compiled_device = compiled_device

        def launch_with(self, _values: tuple[int, ...], *_grid: int) -> None:
            launches.append((getattr(thread_state, "current", 0), self.compiled_device))

    class FakeCachedKernel:
        def __init__(self, compiled_device: int) -> None:
            self.driver = FakeDriver(compiled_device)

    def empty_like(tensor: _FakeCudaTensor) -> _FakeCudaTensor:
        assert tensor.device.index is not None
        if tensor.device.index == 0:
            device_zero_ready.set()
            assert device_one_done.wait(timeout=10), "device 1 did not finish before device 0 launched"
        return _FakeCudaTensor(tensor.shape, device=tensor.device.index)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: getattr(thread_state, "current", 0))
    monkeypatch.setattr(torch.cuda, "device", ThreadCudaDeviceContext)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    monkeypatch.setattr(
        pair_injection_triton,
        "get_sm_version",
        lambda device: {0: 100, 1: 103}[device if isinstance(device, int) else device.index],
    )

    config = (32, 128, 4, 3)
    monkeypatch.setattr(
        FusedPairInjection,
        "_global_cache",
        {
            (device, sm, 384, True, torch.bfloat16, *config): {torch.bfloat16: FakeCachedKernel(device)}
            for device, sm in ((0, 100), (1, 103))
        },
    )
    kernel = FusedPairInjection(384, hadamard=True)

    def call(device: int) -> None:
        pair = _FakeCudaTensor((1, 17, 17, 384), device=device)
        row = _FakeCudaTensor((1, 17, 384), device=device)
        column = _FakeCudaTensor((1, 17, 384), device=device)
        mask = _FakeCudaTensor((1, 17, 17, 1), device=device)
        kernel(pair, row, column, mask)
        if device == 1:
            device_one_done.set()

    thread_zero = threading.Thread(target=call, args=(0,))
    thread_one = threading.Thread(target=call, args=(1,))
    thread_zero.start()
    assert device_zero_ready.wait(timeout=10)
    thread_one.start()
    thread_zero.join(timeout=15)
    thread_one.join(timeout=15)

    assert not thread_zero.is_alive()
    assert not thread_one.is_alive()
    assert sorted(launches) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("hadamard", [False, True])
@pytest.mark.parametrize("extra_dtype_first", [False, True])
def test_pair_injection_instances_keep_requested_dtype_after_cache_warmup(
    monkeypatch: pytest.MonkeyPatch,
    hadamard: bool,
    extra_dtype_first: bool,
) -> None:
    """An instance requesting an extra dtype works regardless of cache order."""
    monkeypatch.setattr(FusedPairInjection, "_global_cache", {})
    dtypes = [torch.bfloat16, torch.float64]
    if extra_dtype_first:
        dtypes.reverse()

    for dtype in [*dtypes, *dtypes]:
        kernel = FusedPairInjection(384, hadamard=hadamard, dtype=dtype)
        pair = torch.full((1, 17, 17, 384), 2, device="cuda", dtype=dtype)
        row = torch.full((1, 17, 384), 3, device="cuda", dtype=dtype)
        column = torch.full_like(row, 4)
        mask = torch.ones((1, 17, 17, 1), device="cuda", dtype=dtype)
        mask[:, -1] = 0
        expected = torch.full_like(pair, 14 if hadamard else 9) * mask

        actual = kernel(pair, row, column, mask)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("pair_write", ["sum", "hadamard"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [17, 33])
def test_tuned_pair_injection_matches_old_config_and_eager_bytes(
    pair_write: str,
    dtype: torch.dtype,
    tokens: int,
) -> None:
    """The C=384 table entry executes without changing any output byte."""
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in (100, 103):
        pytest.skip("the tuned launch requires SM100 or SM103")

    torch.manual_seed(43)
    operands = _injection_operands(1, tokens, 384, dtype)
    expected = _eager_injection(*operands, pair_write)
    hadamard = pair_write == "hadamard"
    tuned = fused_pair_injection(*operands, hadamard=hadamard)
    old = fused_pair_injection(*operands, hadamard=hadamard, block_j=16, block_d=128)
    cached_kernel = FusedPairInjection(384, hadamard=hadamard, dtype=dtype)
    cached = cached_kernel(*operands)

    assert cached_kernel._last_launch.config == (32, 128, 4, 3)
    assert torch.equal(tuned.view(torch.uint8), old.view(torch.uint8))
    assert torch.equal(tuned.view(torch.uint8), cached.view(torch.uint8))
    valid = operands[-1].bool().expand_as(expected)
    expected_valid = expected[valid].contiguous().view(torch.uint8)
    for actual in (tuned, old, cached):
        assert torch.equal(actual, expected)
        assert torch.equal(actual[valid].contiguous().view(torch.uint8), expected_valid)
        assert torch.count_nonzero(actual[~valid]) == 0


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
