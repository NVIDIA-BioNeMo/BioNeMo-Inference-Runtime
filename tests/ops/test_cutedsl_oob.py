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
"""OOB tests for CuTeDSL dual GEMM and attention.

Each case puts an operand against a guard page. Launch in a subprocess (this
module: ``python test_cutedsl_oob.py <case> [--pin-sm <sm>]``) because the
fault is sticky.

Each case runs against the sources once per SM, and once against this GPU's
packaged CUBIN.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests._torch import SM_VERSION, skip_if_no_cutedsl, source_module_available
from tests.common.test_utils.cuda_guard_page import GuardedArena, vmm_supported

OK_SENTINEL = "OOB_PROBE_OK"
# Keep "read out of bounds" distinct from "never reached the launch".
EXIT_OOB = 1
EXIT_UNSUPPORTED = 2
EXIT_ERROR = 3

_ILLEGAL_ACCESS_MARKERS = ("cudaerrorillegaladdress", "illegal memory access", "cuda_error_illegal_address")

# Every SM these kernel families ship configs for.
SM_VERSIONS = (80, 86, 89, 90)
FORCE_CUBIN_ENV = "CUTEDSL_FORCE_CUBIN"


def _check(out: torch.Tensor) -> None:
    """Smoke-check the result; the numerical suites assert correctness."""
    finite = bool(torch.isfinite(out).all())
    print(f"output {tuple(out.shape)} finite={finite}")
    if not finite:
        raise AssertionError("kernel produced non-finite output")


@dataclass(frozen=True)
class _DualGemm:
    """A shape whose M tail pushes ``b_idx`` past ``actual_seqlen[kernel_B]``.

    ``kernel_B`` must be a multiple of 4 to sit flush against the guard page, and
    ``M = kernel_B * I_dim`` must not be a multiple of the MMA M tile or there
    are no tail rows.
    """

    K: int
    N: int
    kernel_B: int
    I_dim: int
    transpose_out: bool = False
    has_bias: bool = False

    def run(self) -> None:
        from bionemo_ir._torch.custom_ops.dual_gemm_x_x.cutedsl import DualGemmXxCuTe

        dtype = torch.bfloat16
        m = self.kernel_B * self.I_dim
        print(
            f"K={self.K} N={self.N} kernel_B={self.kernel_B} I_dim={self.I_dim} M={m} "
            f"(M%64={m % 64}, M%128={m % 128}) transpose_out={self.transpose_out} bias={self.has_bias}"
        )

        arena = GuardedArena()
        actual_seqlen = arena.tensor((self.kernel_B,), torch.int32)
        print(f"actual_seqlen: {arena.describe(actual_seqlen)}")
        lengths = torch.arange(1, self.kernel_B + 1, dtype=torch.int32) % (self.I_dim + 1)
        actual_seqlen.copy_(lengths.clamp_(min=1).cuda())

        x = torch.randn(1, self.kernel_B, self.I_dim, self.K, device="cuda").to(dtype)
        w0 = torch.randn(self.N, self.K, device="cuda").to(dtype)
        w1 = torch.randn(self.N, self.K, device="cuda").to(dtype)
        bias = torch.randn(self.N, device="cuda").to(dtype) if self.has_bias else None

        op = DualGemmXxCuTe()
        out = op(x, w0, w1, bias, bias, transpose_out=self.transpose_out, actual_seqlen=actual_seqlen)
        _check(out)


@dataclass(frozen=True)
class _PairwiseAttention:
    """A bias width that is 8-aligned but not tile-aligned.

    8-aligned keeps the host wrapper's padding a no-op, so the guarded tensor
    reaches the kernel; not tile-aligned leaves the gap the bug read into.
    ``B*H*Sq*Skv`` must be a multiple of 8 to sit flush against the guard page.
    """

    batch: int
    heads: int
    head_dim: int
    seqlen_q: int
    seqlen_kv: int

    def run(self) -> None:
        from bionemo_ir._torch.attention_backend.pairwise_attention.cutedsl import (
            PairwiseAttentionCuTeLeftMask,
            PairwiseAttentionCuTeLeftMaskMetadata,
        )

        dtype = torch.bfloat16
        shape = (self.batch, self.heads, self.seqlen_q, self.seqlen_kv)
        print(
            f"B={self.batch} H={self.heads} D={self.head_dim} Sq={self.seqlen_q} Skv={self.seqlen_kv} "
            f"(Skv%64={self.seqlen_kv % 64}, Skv%128={self.seqlen_kv % 128})"
        )

        arena = GuardedArena()
        bias = arena.tensor(shape, dtype)
        print(f"bias: {arena.describe(bias)}")
        bias.copy_(torch.randn(shape, device="cuda").to(dtype))

        q = torch.randn(self.batch, self.seqlen_q, self.heads, self.head_dim, device="cuda").to(dtype)
        kv_shape = (self.batch, self.seqlen_kv, self.heads, self.head_dim)
        k, v = (torch.randn(kv_shape, device="cuda").to(dtype) for _ in range(2))
        # Full length, so the mainloop's top block reaches the bias tail.
        actual_s_kv = torch.full((self.batch,), self.seqlen_kv, dtype=torch.int32, device="cuda")

        op = PairwiseAttentionCuTeLeftMask(0, self.heads, self.head_dim, num_kv_heads=self.heads)
        metadata = PairwiseAttentionCuTeLeftMaskMetadata()
        metadata.kv_packed = False
        _check(op.forward(q, k, v, biases=[actual_s_kv, bias], metadata=metadata))


@dataclass(frozen=True)
class _TriangleAttention:
    """The triangle-attention form of :class:`_PairwiseAttention`."""

    batch: int
    i_dim: int
    heads: int
    head_dim: int
    seqlen: int

    def run(self) -> None:
        from bionemo_ir._torch.attention_backend.triangle_attention.cutedsl import (
            TriangleAttentionCuTeLeftMask,
            TriangleAttentionCuTeLeftMaskMetadata,
        )

        dtype = torch.bfloat16
        shape = (self.batch, self.heads, self.seqlen, self.seqlen)
        print(
            f"B={self.batch} I={self.i_dim} H={self.heads} D={self.head_dim} J={self.seqlen} "
            f"(J%64={self.seqlen % 64}, J%128={self.seqlen % 128})"
        )

        arena = GuardedArena()
        bias = arena.tensor(shape, dtype)
        print(f"bias: {arena.describe(bias)}")
        bias.copy_(torch.randn(shape, device="cuda").to(dtype))

        qkv_shape = (self.batch, self.i_dim, self.seqlen, self.heads, self.head_dim)
        q, k, v = (torch.randn(qkv_shape, device="cuda").to(dtype) for _ in range(3))
        actual_s_kv = torch.full((self.batch, self.i_dim), self.seqlen, dtype=torch.int32, device="cuda")

        op = TriangleAttentionCuTeLeftMask(0, self.heads, self.head_dim, num_kv_heads=self.heads)
        metadata = TriangleAttentionCuTeLeftMaskMetadata()
        metadata.qkv_packed = False
        _check(op.forward(q, k, v, biases=[actual_s_kv, bias], metadata=metadata))


CASES: dict[str, _DualGemm | _PairwiseAttention | _TriangleAttention] = {
    "dual_gemm_k128": _DualGemm(K=128, N=128, kernel_B=100, I_dim=95),
    # Tiny M, whose tail rows run far past kernel_B.
    "dual_gemm_k128_min_tail": _DualGemm(K=128, N=128, kernel_B=4, I_dim=3),
    # The transposed epilogue stores through a different path.
    "dual_gemm_k128_transposed": _DualGemm(K=128, N=128, kernel_B=8, I_dim=127, transpose_out=True, has_bias=True),
    "dual_gemm_n256": _DualGemm(K=128, N=256, kernel_B=12, I_dim=85),
    # Protenix trimul shape.
    "dual_gemm_universal": _DualGemm(K=256, N=512, kernel_B=100, I_dim=95),
    # I_dim remainders vs the 64 / 128 MMA-M tile.
    "dual_gemm_k128_idim_1": _DualGemm(K=128, N=128, kernel_B=4, I_dim=1),
    "dual_gemm_k128_idim_17": _DualGemm(K=128, N=128, kernel_B=8, I_dim=17),
    "dual_gemm_k128_idim_65": _DualGemm(K=128, N=128, kernel_B=4, I_dim=65),
    "dual_gemm_universal_idim_33": _DualGemm(K=256, N=512, kernel_B=8, I_dim=33),
    # Each head dim is its own kernel instantiation.
    "pairwise_attention_d32": _PairwiseAttention(batch=2, heads=4, head_dim=32, seqlen_q=104, seqlen_kv=104),
    "pairwise_attention_d48": _PairwiseAttention(batch=1, heads=4, head_dim=48, seqlen_q=136, seqlen_kv=72),
    "pairwise_attention_d64": _PairwiseAttention(batch=2, heads=2, head_dim=64, seqlen_q=8, seqlen_kv=8),
    # seqlen_kv stays 8-aligned so host padding is a no-op; seqlen_q need not be.
    # Remainders vs the 64 / 128 attention tiles, including rectangular Sq != Skv.
    "pairwise_attention_d32_q17_kv24": _PairwiseAttention(batch=2, heads=4, head_dim=32, seqlen_q=17, seqlen_kv=24),
    "pairwise_attention_d48_q9_kv56": _PairwiseAttention(batch=1, heads=4, head_dim=48, seqlen_q=9, seqlen_kv=56),
    "pairwise_attention_d64_q25_kv40": _PairwiseAttention(batch=2, heads=2, head_dim=64, seqlen_q=25, seqlen_kv=40),
    "pairwise_attention_d32_q65_kv88": _PairwiseAttention(batch=1, heads=2, head_dim=32, seqlen_q=65, seqlen_kv=88),
    "pairwise_attention_d48_q15_kv120": _PairwiseAttention(batch=1, heads=2, head_dim=48, seqlen_q=15, seqlen_kv=120),
    "triangle_attention_d32": _TriangleAttention(batch=1, i_dim=4, heads=4, head_dim=32, seqlen=104),
    "triangle_attention_d64": _TriangleAttention(batch=1, i_dim=2, heads=4, head_dim=64, seqlen=200),
    "triangle_attention_d128": _TriangleAttention(batch=2, i_dim=2, heads=2, head_dim=128, seqlen=8),
    "triangle_attention_d32_i1_s24": _TriangleAttention(batch=1, i_dim=1, heads=4, head_dim=32, seqlen=24),
    "triangle_attention_d64_i3_s40": _TriangleAttention(batch=1, i_dim=3, heads=4, head_dim=64, seqlen=40),
    "triangle_attention_d128_i5_s56": _TriangleAttention(batch=1, i_dim=5, heads=2, head_dim=128, seqlen=56),
    "triangle_attention_d32_i7_s88": _TriangleAttention(batch=1, i_dim=7, heads=2, head_dim=32, seqlen=88),
}

_SOURCE_MODULES = {
    _DualGemm: "bionemo_ir._torch.custom_ops.dual_gemm_x_x._source",
    _PairwiseAttention: "bionemo_ir._torch.attention_backend.pairwise_attention._source",
    _TriangleAttention: "bionemo_ir._torch.attention_backend.triangle_attention._source",
}


def _pin_sm(sm: int) -> None:
    """Make every op select the kernel it would choose on *sm*.

    Each op reads ``torch.cuda.get_device_capability()`` in its constructor.
    Only selection moves -- the kernel still compiles for the real device -- so
    an Ampere kernel can be pinned on Hopper but not the reverse.
    """
    capability = (sm // 10, sm % 10)
    torch.cuda.get_device_capability = lambda *args, **kwargs: capability
    print(f"pinned capability to {capability}")


def _run_case_and_exit() -> None:
    """Child entry point: run one case, then leave without destructors."""
    parser = argparse.ArgumentParser(description="Run one guard-page OOB case.")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--pin-sm", type=int, default=None, help="select this SM's kernels (source mode only)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.init()
    if not vmm_supported():
        print("device lacks virtual memory management")
        sys.stdout.flush()
        os._exit(EXIT_UNSUPPORTED)
    if args.pin_sm is not None:
        _pin_sm(args.pin_sm)

    try:
        CASES[args.case].run()
    except Exception as exc:  # noqa: BLE001 - the fault is the result
        described = f"{type(exc).__name__}: {exc}"
        if any(marker in described.lower() for marker in _ILLEGAL_ACCESS_MARKERS):
            print(f"OOB_PROBE_FAULT: {described}")
            code = EXIT_OOB
        else:
            print(f"OOB_PROBE_ERROR: {described}")
            traceback.print_exc()
            code = EXIT_ERROR
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    print(OK_SENTINEL)
    sys.stdout.flush()
    # The arena outlives torch's teardown; skip destructors entirely.
    os._exit(0)


# Sources can be pinned to any SM up to this GPU's; a CUBIN pack only ever
# loads its own architecture, so that run carries no SM axis.
_IMPLEMENTATIONS = [pytest.param("source", sm, id=f"source-sm{sm}") for sm in SM_VERSIONS]
_IMPLEMENTATIONS.append(pytest.param("cubin", None, id="cubin"))


@pytest.mark.parametrize(("mode", "sm"), _IMPLEMENTATIONS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_cutedsl_kernel_stays_in_bounds(case: str, mode: str, sm: int | None):
    """The kernel must not read past ``actual_seqlen`` or the bias."""
    skip_if_no_cutedsl()

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    # Blame the launch that faulted, not a later sync.
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    # Surfaces PyTorch's own device asserts; the guard page catches the kernels.
    env["TORCH_USE_CUDA_DSA"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH", "")]))

    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), case]
    if mode == "cubin":
        pytest.importorskip("bionemo_ir.libs._cutedsl_kernels")
        env[FORCE_CUBIN_ENV] = "1"
    else:
        if not source_module_available(_SOURCE_MODULES[type(CASES[case])]):
            pytest.skip("build carries no kernel sources")
        if sm > SM_VERSION:
            pytest.skip(f"SM{sm} kernels need SM{sm} hardware (this GPU is SM{SM_VERSION})")
        env.pop(FORCE_CUBIN_ENV, None)
        cmd += ["--pin-sm", str(sm)]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,  # a cold CuTeDSL compile dominates; the launch is tiny
        check=False,
    )
    target = "the packaged CUBIN" if mode == "cubin" else f"the SM{sm} source"
    detail = f"case={case} via {target}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"

    if result.returncode == EXIT_UNSUPPORTED:
        pytest.skip(f"guard page unsupported on this device\n{detail}")
    if result.returncode == EXIT_ERROR:
        pytest.fail(f"probe never reached the launch, so nothing was verified\n{detail}")
    if result.returncode == EXIT_OOB:
        pytest.fail(f"kernel crossed the guard page: it read outside its operand\n{detail}")

    assert result.returncode == 0, f"probe exited {result.returncode}\n{detail}"
    assert OK_SENTINEL in result.stdout, f"probe did not report success\n{detail}"


if __name__ == "__main__":
    _run_case_and_exit()
