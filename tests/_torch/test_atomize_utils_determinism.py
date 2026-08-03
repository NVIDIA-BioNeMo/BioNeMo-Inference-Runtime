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
"""Unit-level determinism guard for ``aggregate_atom_feat_to_tokens``.

The atom->token aggregation in OpenFold3's ``AtomAttentionEncoder`` scatters
atom features into their token buckets with CUDA ``scatter_add_``. That op
accumulates colliding indices with ``atomicAdd`` in an unspecified order, and
because floating-point addition is not associative the result is **not**
reproducible run-to-run -- *regardless of dtype* (fp32 only shrinks the
per-call noise, it does not remove it; ``test_raw_cuda_scatter_add_is_*``
below demonstrates this directly).

The encoder runs on every one of the ~200 diffusion rollout steps, and the
rollout is an iterated map, so this tiny per-call noise compounds into visibly
divergent predicted structures (this is the root cause behind the OpenFold3
eager non-determinism caught by
``model_forwards/test_model_forward_eager_determinism.py``).

The fix wraps the scatter in a deterministic-algorithms context. These tests
pin that fix at the unit level -- fast and checkpoint-free -- asserting:

  * ``aggregate_atom_feat_to_tokens`` is **bit-identical** across repeated
    calls on the same input (both bf16 and fp32 inputs);
  * the deterministic result still matches a reference segment-mean / -sum
    (the fix changes *reduction order*, not the math);
  * the raw, unwrapped CUDA ``scatter_add_`` on the same high-collision input
    really is non-deterministic (so the fix is load-bearing, not a no-op).
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_bionemo._torch.modules.openfold3.utils.atomize_utils import (
    _deterministic_algorithms, aggregate_atom_feat_to_tokens)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="scatter_add_ non-determinism is CUDA-specific")

# High atom:token ratio => many colliding scatter indices, which is what makes
# the atomic-add ordering (and thus the non-determinism) actually bite.
N_BATCH = 2
N_ATOM = 2048
N_TOKEN = 96
C_FEAT = 128
EPS = 1e-9
_REPEATS = 25


def _make_inputs(dtype: torch.dtype, seed: int = 0):
    """Build a realistic atom->token aggregation problem on CUDA.

    Returns the kwargs for ``aggregate_atom_feat_to_tokens`` plus the raw
    pieces a reference implementation needs.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    atom_to_token_index = torch.randint(
        0, N_TOKEN, (N_BATCH, N_ATOM), device=dev, generator=g)
    # Mostly-present atoms with a realistic fraction masked out.
    atom_mask = (torch.rand(N_BATCH, N_ATOM, device=dev, generator=g)
                 > 0.1).to(dtype)
    token_mask = torch.ones(N_BATCH, N_TOKEN, device=dev, dtype=dtype)
    atom_feat = torch.randn(
        N_BATCH, N_ATOM, C_FEAT, device=dev, dtype=dtype, generator=g)
    return {
        "token_mask": token_mask,
        "atom_to_token_index": atom_to_token_index,
        "atom_mask": atom_mask,
        "atom_feat": atom_feat,
        "atom_dim": -2,
    }


def _reference(inputs: dict, aggregate_fn: str) -> torch.Tensor:
    """Deterministic segment sum/mean via a one-hot matmul (cuBLAS GEMM is
    reproducible run-to-run), mirroring the function's masking + eps."""
    idx = inputs["atom_to_token_index"]
    atom_mask = inputs["atom_mask"].float()
    feat = inputs["atom_feat"].float() * atom_mask.unsqueeze(-1)
    # Masked atoms route to a dropped bucket (index == N_TOKEN), exactly like
    # the function's ``torch.where(atom_mask, idx, n_token)``.
    idx = torch.where(inputs["atom_mask"].bool(), idx, N_TOKEN)
    onehot = F.one_hot(idx, N_TOKEN + 1).float()  # [B, N_atom, n_token+1]
    summed = torch.einsum("bna,bnc->bac", onehot, feat)[:, :N_TOKEN, :]
    if aggregate_fn == "sum":
        return summed
    count = torch.einsum("bna,bn->ba", onehot, atom_mask)[:, :N_TOKEN]
    return summed / (count.unsqueeze(-1) + EPS)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("aggregate_fn", ["mean", "sum"])
def test_aggregate_atom_feat_is_bit_identical_run_to_run(dtype, aggregate_fn):
    """The fix's core guarantee: repeated calls on identical inputs return
    bit-identical results."""
    inputs = _make_inputs(dtype)
    ref = aggregate_atom_feat_to_tokens(**inputs, aggregate_fn=aggregate_fn)
    torch.cuda.synchronize()
    for i in range(_REPEATS):
        out = aggregate_atom_feat_to_tokens(**inputs, aggregate_fn=aggregate_fn)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (
            f"run {i} diverged (dtype={dtype}, agg={aggregate_fn}): "
            f"max|Δ|={(out.float() - ref.float()).abs().max().item():.3e} -- "
            "atom->token scatter lost determinism")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("aggregate_fn", ["mean", "sum"])
def test_aggregate_atom_feat_matches_reference(dtype, aggregate_fn):
    """The deterministic scatter changes reduction order, not the math:
    output still matches a one-hot-matmul reference."""
    inputs = _make_inputs(dtype)
    out = aggregate_atom_feat_to_tokens(
        **inputs, aggregate_fn=aggregate_fn).float()
    ref = _reference(inputs, aggregate_fn)
    # fp32 accumulation internally; bf16 only at the cast back to input dtype.
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (3e-2, 3e-2)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_cuda_scatter_add_is_nondeterministic(dtype):
    """Demonstrate the bug the fix addresses: the *unwrapped* CUDA
    ``scatter_add_`` on this high-collision input is not reproducible.

    This is what makes the fix load-bearing. If a future GPU/PyTorch makes the
    raw op deterministic the contrast vanishes harmlessly -- so we skip (never
    fail red) rather than assert, while the determinism guarantee of the fix is
    asserted unconditionally in
    ``test_aggregate_atom_feat_is_bit_identical_run_to_run``.
    """
    inputs = _make_inputs(dtype)
    idx = inputs["atom_to_token_index"].unsqueeze(-1).expand(-1, -1, C_FEAT)
    src = inputs["atom_feat"]

    def raw_scatter():
        out = torch.zeros(N_BATCH, N_TOKEN, C_FEAT, device="cuda", dtype=dtype)
        out.scatter_add_(dim=1, index=idx, src=src)
        torch.cuda.synchronize()
        return out

    base = raw_scatter()
    diverged = any(not torch.equal(raw_scatter(), base) for _ in range(_REPEATS))
    if not diverged:
        pytest.skip("raw CUDA scatter_add_ happened to be deterministic on "
                    "this device/build; fix-side determinism still asserted "
                    "elsewhere")

    # And confirm the fix makes that very same scatter reproducible.
    with _deterministic_algorithms():
        det_base = torch.zeros(
            N_BATCH, N_TOKEN, C_FEAT, device="cuda", dtype=dtype).scatter_add_(
                1, idx, src)
        torch.cuda.synchronize()
        for _ in range(_REPEATS):
            det = torch.zeros(
                N_BATCH, N_TOKEN, C_FEAT, device="cuda",
                dtype=dtype).scatter_add_(1, idx, src)
            torch.cuda.synchronize()
            assert torch.equal(det, det_base)


def test_deterministic_context_restores_global_flag():
    """The scoped helper must not leak PyTorch's global determinism setting."""
    before = torch.are_deterministic_algorithms_enabled()
    with _deterministic_algorithms():
        assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.are_deterministic_algorithms_enabled() == before
