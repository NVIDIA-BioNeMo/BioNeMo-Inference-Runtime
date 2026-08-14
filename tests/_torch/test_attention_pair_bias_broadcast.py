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
"""Tests for AttentionPairBias._prep_mask_bias broadcast behaviour.

The SDPA path must handle inputs where ``s`` has extra leading dimensions
that ``z`` does not — e.g. the atom transformer path where ``s`` is
``[B, 1, K, N_q, C]`` but ``z`` is ``[B, K, N_q, N_k, C_z]``.

Regression: prior to the fix, ``unsqueeze(-4)`` inserted the broadcast-1
at position 2 instead of position 1, causing a shape mismatch inside
``F.scaled_dot_product_attention``.
"""

from dataclasses import dataclass

import pytest
import torch

from bionemo_ir._torch.layers.attention import AttentionPairBias
from tests._torch import skip_if_cutedsl

SEED = 42


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _bounded_init(model: torch.nn.Module) -> None:
    """Initialise all parameters with bounded values.

    Xavier-uniform for 2-D+ tensors (projections), uniform(-0.1, 0.1) for
    1-D tensors (LayerNorm weight/bias, linear biases).  This avoids the
    degenerate zero-weight regime from the model's default init while
    keeping magnitudes safe for bf16.
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p, -0.1, 0.1)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    """Parameterised test shape descriptor."""

    name: str
    backend: str
    # s shape (token / atom embedding fed to the attention layer)
    s_shape: tuple[int, ...]
    # z shape (pair representation / bias)
    z_shape: tuple[int, ...]
    # mask shape
    mask_shape: tuple[int, ...]
    c_s: int = 128
    c_z: int = 16
    num_heads: int = 4
    dtype: torch.dtype = torch.float32
    # When set, provide a single_embedding of the same shape as s
    use_single_embedding: bool = False


# ── Standard token-transformer path (multiplicity S at dim 1) ────────────────
# Use odd N_tok (33) to stress-test Triton kernel padding logic.
_TOKEN_XFORMER = Scenario(
    name="token_xformer_S5",
    backend="SDPA",
    s_shape=(1, 5, 33, 128),  # [B, S, N_tok, C_s]
    z_shape=(1, 33, 33, 16),  # [B, N_tok, N_tok, C_z]
    mask_shape=(1, 33),  # [B, N_tok]
)

_TOKEN_XFORMER_S1 = Scenario(
    name="token_xformer_S1",
    backend="SDPA",
    s_shape=(1, 1, 33, 128),  # [B, 1, N_tok, C_s]
    z_shape=(1, 33, 33, 16),  # [B, N_tok, N_tok, C_z]
    mask_shape=(1, 33),
)

# ── Atom-transformer path (unsqueeze(1) gives [B, 1, K, N_q, C]) ────────────
_ATOM_XFORMER = Scenario(
    name="atom_xformer_K8",
    backend="SDPA",
    s_shape=(1, 1, 8, 32, 128),  # [B, 1, K, N_q, C_s]
    z_shape=(1, 8, 32, 32, 16),  # [B, K, N_q, N_k, C_z]
    mask_shape=(1, 8, 32),  # [B, K, N_q]
)

_ATOM_XFORMER_K25 = Scenario(
    name="atom_xformer_K25",
    backend="SDPA",
    s_shape=(1, 1, 25, 32, 128),  # [B, 1, K=25, N_q, C_s]
    z_shape=(1, 25, 32, 32, 16),  # [B, K=25, N_q, N_k, C_z]
    mask_shape=(1, 25, 32),  # [B, K, N_q]
)

# ── Atom-transformer inside diffusion (already has sample dim) ────────────────
_ATOM_XFORMER_DIFFUSION = Scenario(
    name="atom_xformer_diffusion_S5",
    backend="SDPA",
    s_shape=(1, 5, 8, 32, 128),  # [B, S, K, N_q, C_s]
    z_shape=(1, 8, 32, 32, 16),  # [B, K, N_q, N_k, C_z]
    mask_shape=(1, 8, 32),  # [B, K, N_q]
)

# ── No extra dims (standard 3-D s) ───────────────────────────────────────────
_SIMPLE = Scenario(
    name="simple_no_extra_dims",
    backend="SDPA",
    s_shape=(1, 31, 128),  # [B, N, C_s]
    z_shape=(1, 31, 31, 16),  # [B, N, N, C_z]
    mask_shape=(1, 31),  # [B, N]
)

# ── VANILLA backend (uses same broadcast logic) ─────────────────────────────
_ATOM_VANILLA = Scenario(
    name="atom_xformer_vanilla",
    backend="VANILLA",
    s_shape=(1, 1, 8, 32, 128),
    z_shape=(1, 8, 32, 32, 16),
    mask_shape=(1, 8, 32),
)

# ── CuTeDSL backend (token transformer, bf16 required) ──────────────────────
_TOKEN_CUTEDSL_S1 = Scenario(
    name="token_xformer_cutedsl_S1",
    backend="CuTeDSL",
    s_shape=(1, 1, 33, 128),
    z_shape=(1, 33, 33, 16),
    mask_shape=(1, 33),
    dtype=torch.bfloat16,
)

_TOKEN_CUTEDSL_S5 = Scenario(
    name="token_xformer_cutedsl_S5",
    backend="CuTeDSL",
    s_shape=(1, 5, 33, 128),
    z_shape=(1, 33, 33, 16),
    mask_shape=(1, 33),
    dtype=torch.bfloat16,
)

_SIMPLE_CUTEDSL = Scenario(
    name="simple_cutedsl",
    backend="CuTeDSL",
    s_shape=(1, 31, 128),
    z_shape=(1, 31, 31, 16),
    mask_shape=(1, 31),
    dtype=torch.bfloat16,
)


@pytest.mark.parametrize(
    "sc",
    [
        _SIMPLE,
        _TOKEN_XFORMER_S1,
        _TOKEN_XFORMER,
        _ATOM_XFORMER,
        _ATOM_XFORMER_K25,
        _ATOM_XFORMER_DIFFUSION,
        _ATOM_VANILLA,
        _SIMPLE_CUTEDSL,
        _TOKEN_CUTEDSL_S1,
        _TOKEN_CUTEDSL_S5,
    ],
    ids=lambda sc: sc.name,
)
def test_prep_mask_bias_shapes(sc: Scenario):
    """pair_bias and mask_bias must be broadcastable to q after _prep_qkv."""
    skip_if_cutedsl(sc.backend)
    device = torch.device("cuda")
    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=sc.dtype,
        bias_proj=True,
        initial_norm=False,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    biases = attn._prep_mask_bias(s, z, mask, mask_bias=None)
    mask_bias, pair_bias = biases

    if sc.backend == "CuTeDSL":
        # CuTeDSL handles broadcasting internally; _prep_mask_bias does NOT
        # insert extra dimensions.  mask_bias is just mask.float().
        assert mask_bias.ndim == len(sc.mask_shape)
    else:
        # After _prep_qkv, q has shape [*batch, H, S_Q, D] — one extra dim
        # relative to s.  pair_bias must have the same ndim as q.
        expected_ndim = s.ndim + 1
        assert pair_bias.ndim == expected_ndim, f"pair_bias.ndim={pair_bias.ndim} != s.ndim+1={expected_ndim}"
        assert mask_bias.ndim == pair_bias.ndim, f"mask_bias.ndim={mask_bias.ndim} != pair_bias.ndim={pair_bias.ndim}"

        # Verify the two biases can actually be added (no broadcast error)
        _ = mask_bias + pair_bias


@pytest.mark.parametrize(
    "sc",
    [
        _SIMPLE,
        _TOKEN_XFORMER_S1,
        _TOKEN_XFORMER,
        _ATOM_XFORMER,
        _ATOM_XFORMER_DIFFUSION,
        _ATOM_VANILLA,
    ],
    ids=lambda sc: sc.name,
)
def test_prep_mask_bias_with_precomputed(sc: Scenario):
    """When mask_bias is already precomputed, the same broadcast must hold."""
    device = torch.device("cuda")
    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=sc.dtype,
        bias_proj=True,
        initial_norm=False,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    # Precomputed mask_bias: [*, 1, 1, S_KV]
    precomputed = (1 - mask.float()) * -1e9
    precomputed = precomputed[..., None, None, :]

    biases = attn._prep_mask_bias(s, z, mask, mask_bias=precomputed)
    mask_bias, pair_bias = biases

    expected_ndim = s.ndim + 1
    assert pair_bias.ndim == expected_ndim
    assert mask_bias.ndim == pair_bias.ndim
    _ = mask_bias + pair_bias


@pytest.mark.parametrize(
    "sc",
    [
        _ATOM_XFORMER,
        _TOKEN_XFORMER,
        _SIMPLE,
    ],
    ids=lambda sc: sc.name,
)
def test_full_forward_sdpa(sc: Scenario):
    """Full AttentionPairBias.forward must succeed (no broadcast crash) for SDPA."""
    device = torch.device("cuda")
    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=sc.dtype,
        bias_proj=True,
        initial_norm=True,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    with torch.no_grad():
        out = attn(s, z, mask)

    assert out.shape == s.shape, f"output shape {out.shape} != input shape {s.shape}"
    assert torch.isfinite(out).all(), "output contains non-finite values"


@pytest.mark.parametrize(
    "sc",
    [
        _SIMPLE_CUTEDSL,
        _TOKEN_CUTEDSL_S1,
        _TOKEN_CUTEDSL_S5,
    ],
    ids=lambda sc: sc.name,
)
def test_full_forward_cutedsl(sc: Scenario):
    """Full AttentionPairBias.forward must succeed with CuTeDSL backend."""
    skip_if_cutedsl(sc.backend)
    device = torch.device("cuda")

    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=torch.float32,
        bias_proj=True,
        initial_norm=True,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(dtype=sc.dtype, device=device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    with torch.no_grad():
        out = attn(s, z, mask)

    assert out.shape == s.shape, f"output shape {out.shape} != input shape {s.shape}"
    assert torch.isfinite(out).all(), "output contains non-finite values"


@pytest.mark.parametrize(
    "sc",
    [
        _SIMPLE_CUTEDSL,
        _TOKEN_CUTEDSL_S1,
        _TOKEN_CUTEDSL_S5,
    ],
    ids=lambda sc: sc.name,
)
def test_cutedsl_uses_fused_triton_kernel(sc: Scenario):
    """Verify the CuTeDSL path dispatches through the fused Triton
    ``LNProjMoveaxisPad._fused_kernel`` rather than the vanilla fallback.
    """
    skip_if_cutedsl(sc.backend)
    device = torch.device("cuda")

    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=torch.float32,
        bias_proj=True,
        initial_norm=True,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(dtype=sc.dtype, device=device)

    ln_proj = attn._ln_proj_moveaxis_pad
    assert ln_proj is not None, "bias_proj=True should create _ln_proj_moveaxis_pad"
    assert ln_proj._fused_kernel is not None, (
        "fused Triton kernel was not instantiated — CuTeDSL path would silently fall back to vanilla PyTorch"
    )

    original_kernel = ln_proj._fused_kernel
    call_count = 0

    class _CountingProxy:
        """Transparent proxy that counts calls then delegates."""

        def __call__(self_, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_kernel(*args, **kwargs)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    ln_proj._fused_kernel = _CountingProxy()
    try:
        with torch.no_grad():
            out = attn(s, z, mask)
    finally:
        ln_proj._fused_kernel = original_kernel

    assert call_count > 0, "LNProjMoveaxisPad._fused_kernel was never called — the fused Triton path was not exercised"
    assert out.shape == s.shape


@pytest.mark.parametrize(
    "sc",
    [
        _ATOM_XFORMER,
        _ATOM_XFORMER_DIFFUSION,
    ],
    ids=lambda sc: sc.name,
)
def test_atom_xformer_bias_alignment(sc: Scenario):
    """Specifically verify that the unsqueeze(1) dim in pair_bias aligns
    with the unsqueeze(1) dim in s — the root cause of the original bug.

    After _prep_mask_bias, pair_bias dim-1 must be size 1 (the broadcast
    dim matching s's extra multiplicity/sample dimension).
    """
    device = torch.device("cuda")
    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=sc.dtype,
        bias_proj=True,
        initial_norm=False,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    biases = attn._prep_mask_bias(s, z, mask, mask_bias=None)
    mask_bias, pair_bias = biases

    # s has 1 at dim-1; pair_bias must also have 1 at dim-1
    assert pair_bias.shape[1] == 1, (
        f"pair_bias dim-1 should be 1 (broadcast), got {pair_bias.shape[1]} (shape={tuple(pair_bias.shape)})"
    )
    assert mask_bias.shape[1] == 1, (
        f"mask_bias dim-1 should be 1 (broadcast), got {mask_bias.shape[1]} (shape={tuple(mask_bias.shape)})"
    )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(
            name="no_bias_proj_sdpa",
            backend="SDPA",
            s_shape=(1, 1, 8, 32, 128),
            z_shape=(1, 8, 4, 32, 32),  # pre-projected: [B, K, H, N_q, N_k]
            mask_shape=(1, 8, 32),
        ),
    ],
    ids=lambda sc: sc.name,
)
def test_prep_mask_bias_no_bias_proj(sc: Scenario):
    """When bias_proj=False, z is passed through as-is; broadcast still works."""
    device = torch.device("cuda")
    attn = AttentionPairBias(
        layer_idx=0,
        c_s=sc.c_s,
        c_z=sc.c_z,
        num_heads=sc.num_heads,
        dtype=sc.dtype,
        bias_proj=False,
        initial_norm=False,
        attn_backend=sc.backend,
    )
    _bounded_init(attn)
    attn = attn.to(device)

    s = torch.randn(*sc.s_shape, device=device, dtype=sc.dtype)
    z = torch.randn(*sc.z_shape, device=device, dtype=sc.dtype)
    mask = torch.ones(*sc.mask_shape, device=device, dtype=sc.dtype)

    biases = attn._prep_mask_bias(s, z, mask, mask_bias=None)
    mask_bias, pair_bias = biases

    expected_ndim = s.ndim + 1
    assert pair_bias.ndim == expected_ndim
    assert mask_bias.ndim == pair_bias.ndim
    _ = mask_bias + pair_bias
