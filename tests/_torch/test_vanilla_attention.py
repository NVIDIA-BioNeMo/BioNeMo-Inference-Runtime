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
import importlib
import os

import pytest
import torch
from test_utils.boltz.ref_attn import plain_mha

from tensorrt_bionemo._torch import _cutedsl_kernel_library as library_runtime
from tensorrt_bionemo._torch._kernel_config_loader import get_config_file_name, load_kernel_configs
from tensorrt_bionemo._torch.attention_backend import AttentionType, get_attention_backend
from tensorrt_bionemo._torch.attention_backend.interface import AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.pairwise_attention.cutedsl import (
    PairwiseAttentionCuTeLeftMask,
    PairwiseAttentionCuTeLeftMaskMetadata,
)
from tensorrt_bionemo._torch.attention_backend.pairwise_attention.vanilla import VanillaPairwiseAttention
from tensorrt_bionemo._torch.attention_backend.triangle_attention import _config as triangle_config
from tensorrt_bionemo._torch.attention_backend.triangle_attention import cutedsl as triangle_cutedsl
from tensorrt_bionemo._torch.attention_backend.triangle_attention.cutedsl import (
    TriangleAttentionCuTeLeftMask,
    TriangleAttentionCuTeLeftMaskMetadata,
)
from tensorrt_bionemo._torch.attention_backend.triangle_attention.sdpa import SDPATriangleAttention
from tensorrt_bionemo._torch.attention_backend.triangle_attention.vanilla import VanillaTriangleAttention
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    make_left_aligned_mask,
    skip_if_no_cutedsl,
    skip_if_not_sm90,
)


def _pw_meta(kv_packed: bool) -> PairwiseAttentionCuTeLeftMaskMetadata:
    m = PairwiseAttentionCuTeLeftMaskMetadata()
    m.kv_packed = kv_packed
    return m


def _tri_meta(qkv_packed: bool) -> TriangleAttentionCuTeLeftMaskMetadata:
    m = TriangleAttentionCuTeLeftMaskMetadata()
    m.qkv_packed = qkv_packed
    return m


_TRIANGLE_CUTEDSL_MODE_CACHES: dict[str, dict] = {
    "source": {},
    "cubin": {},
}
_TRIANGLE_CUTEDSL_SOURCE_MODULE = "tensorrt_bionemo.dsl_kernels.cute.sm80_triangle_attn_left_mask"


def _configure_triangle_cutedsl_mode(mode: str, monkeypatch) -> None:
    """Force one implementation path without allowing silent fallback."""
    mode_cache = _TRIANGLE_CUTEDSL_MODE_CACHES[mode]
    monkeypatch.setattr(TriangleAttentionCuTeLeftMask, "_compiled_cache", mode_cache)

    if mode == "cubin":
        try:
            importlib.import_module("tensorrt_bionemo.libs._cutedsl_kernels")
        except ImportError:
            pytest.fail("CUBIN test mode requires the _cutedsl_kernels extension")

        def source_unavailable(_implementation):
            raise ModuleNotFoundError("CuTeDSL source disabled by CUBIN test mode")

        monkeypatch.setattr(triangle_config, "resolve_implementation", source_unavailable)
        monkeypatch.setattr(library_runtime, "_kernel_library", None)
        return

    def reject_cubin_fallback(*_args, **_kwargs):
        raise AssertionError("source test mode unexpectedly fell back to the CUBIN library")

    monkeypatch.setattr(triangle_cutedsl, "populate_compiled_cache_from_library", reject_cubin_fallback)


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_triangle(seq_len, dtype):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    bs = 1

    q = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)
    k = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)
    v = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)

    vanilla_attn = VanillaTriangleAttention(layer_idx, num_heads, head_dim, num_kv_heads=num_heads)

    biases = [
        torch.randn(bs, seq_len, 1, 1, seq_len).cuda().to(dtype),
        torch.randn(bs, 1, num_heads, seq_len, seq_len).cuda().to(dtype),
    ]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(q, k, v, biases=[biases[0], biases[1].squeeze(1)], metadata=metadata)
    assert vanilla_out.shape == (bs, seq_len, seq_len, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape
    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out, plain_out, atol=5e-2, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_attention_for_triangle(dtype):
    torch.manual_seed(42)
    batch_size, outer_size, seq_len = 2, 3, 17
    num_heads, head_dim = 4, 16
    device = torch.device("cuda")

    q = torch.randn(batch_size, outer_size, seq_len, num_heads * head_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    biases = [
        torch.randn(batch_size, outer_size, 1, 1, seq_len, dtype=dtype, device=device),
        torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=dtype, device=device),
    ]

    vanilla = VanillaTriangleAttention(0, num_heads, head_dim, num_kv_heads=num_heads)
    sdpa = SDPATriangleAttention(0, num_heads, head_dim, num_kv_heads=num_heads)

    with torch.inference_mode():
        expected = vanilla.forward(q, k, v, biases=biases)
        actual = sdpa.forward(q, k, v, biases=biases)

    assert get_attention_backend("SDPA", AttentionType.TRIANGLE) is SDPATriangleAttention
    torch.testing.assert_close(
        actual, expected, atol=5e-2 if dtype == torch.bfloat16 else 1e-4, rtol=1e-2 if dtype == torch.bfloat16 else 1e-4
    )


@pytest.mark.parametrize("batch_size", [16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_pairwise(batch_size, dtype):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    q_size = 32
    kv_size = 128

    q = torch.randn(batch_size, q_size, num_heads * head_dim).to(dtype)
    k = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)
    v = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)

    vanilla_attn = VanillaPairwiseAttention(layer_idx, num_heads, head_dim, num_kv_heads=num_heads)

    biases = [torch.randn(batch_size, 1, 1, kv_size), torch.randn(batch_size, num_heads, q_size, kv_size)]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(q, k, v, biases=biases, metadata=metadata)
    assert vanilla_out.shape == (batch_size, q_size, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape

    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out, plain_out, atol=5e-2, rtol=1e-4)


# ---------------------------------------------------------------------------
# CuTeDSL left-mask pairwise attention vs Vanilla — self/cross, with mult
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,q_size,kv_size,num_heads,head_dim,mult",
    [
        (1, 32, 32, 4, 32, 1),
        (1, 32, 32, 4, 32, 5),
        (1, 32, 128, 4, 32, 1),
        (1, 32, 128, 4, 32, 5),
        (1, 117, 117, 16, 48, 1),
        (1, 117, 117, 16, 48, 5),
        (29, 32, 128, 4, 32, 1),
        (29, 32, 128, 4, 32, 5),
    ],
    ids=[
        "self-B1-S32-H4D32-m1",
        "self-B1-S32-H4D32-m5",
        "cross-B1-Q32K128-H4D32-m1",
        "cross-B1-Q32K128-H4D32-m5",
        "self-B1-S117-H16D48-m1",
        "self-B1-S117-H16D48-m5",
        "cross-B29-Q32K128-H4D32-m1",
        "cross-B29-Q32K128-H4D32-m5",
    ],
)
@pytest.mark.parametrize("kv_packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("mask_form", ["binary", "actual_s_kv"])
def test_pairwise_left_mask_vs_vanilla(batch_size, q_size, kv_size, num_heads, head_dim, mult, kv_packed, mask_form):
    """Compare CuTeDSL left-mask pairwise attention against vanilla reference.

    ``mask_form`` covers both accepted ``biases[0]`` forms: ``binary`` is a
    left-aligned 0/1 mask ``[B, Sk]`` reduced internally, ``actual_s_kv`` is the
    pre-reduced int32 count of leading 1s ``[B]``.
    """
    skip_if_no_cutedsl()
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = torch.bfloat16
    device = torch.device("cuda")

    B = batch_size
    B_flat = B * mult
    Sq, Sk, H, D = q_size, kv_size, num_heads, head_dim

    q = torch.randn(B_flat, Sq, H * D, dtype=dtype, device=device)

    if kv_packed:
        kv = torch.randn(B_flat, Sk, 2, H * D, dtype=dtype, device=device)
        k = kv[..., 0, :]
        v = kv[..., 1, :]
    else:
        k = torch.randn(B_flat, Sk, H * D, dtype=dtype, device=device)
        v = torch.randn(B_flat, Sk, H * D, dtype=dtype, device=device)

    binary_mask = make_left_aligned_mask(B, Sk, dtype=torch.float32, device=device)
    if mask_form == "binary":
        cute_mask_input = binary_mask
    else:
        cute_mask_input = binary_mask.sum(dim=-1).to(torch.int32)
    pair_bias = torch.randn(B, H, Sq, Sk, dtype=dtype, device=device)

    additive_mask = (1.0 - binary_mask) * -1e9
    if mult > 1:
        additive_mask_expanded = (
            additive_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, mult, 1, 1, Sk).reshape(B_flat, 1, 1, Sk)
        )
        pair_bias_expanded = pair_bias.unsqueeze(1).expand(B, mult, H, Sq, Sk).reshape(B_flat, H, Sq, Sk)
    else:
        additive_mask_expanded = additive_mask[:, None, None, :]
        pair_bias_expanded = pair_bias

    vanilla_attn = VanillaPairwiseAttention(0, H, D, num_kv_heads=H)
    vanilla_out = vanilla_attn.forward(
        q,
        k.contiguous(),
        v.contiguous(),
        biases=[additive_mask_expanded, pair_bias_expanded],
        metadata=AttentionMetadata(),
    )

    cute_attn = PairwiseAttentionCuTeLeftMask(0, H, D, num_kv_heads=H)
    cute_out = cute_attn.forward(q, k, v, biases=[cute_mask_input, pair_bias], metadata=_pw_meta(kv_packed))

    assert cute_out.shape == vanilla_out.shape, f"Shape mismatch: cute={cute_out.shape}, vanilla={vanilla_out.shape}"
    diff_max = torch.max(torch.abs(cute_out.float() - vanilla_out.float()))
    diff_mean = torch.mean(torch.abs(cute_out.float() - vanilla_out.float()))
    assert diff_max < 1e-1, f"max diff {diff_max:.4f} >= 1e-1"
    assert diff_mean < 1e-2, f"mean diff {diff_mean:.4f} >= 1e-2"


# ---------------------------------------------------------------------------
# CuTeDSL left-mask triangle attention vs Vanilla — J padding, s_kv shapes
# ---------------------------------------------------------------------------


# bf16 aligns to 8 elements, so J off that multiple exercises J_padded != J.
@pytest.mark.parametrize(
    "bs,I,J,num_heads,head_dim",
    [
        (1, 32, 13, 4, 32),
        (1, 32, 29, 4, 32),
        (1, 13, 13, 4, 32),
        (1, 29, 29, 4, 32),
        (1, 16, 33, 4, 32),
        (1, 16, 65, 4, 32),
        (1, 32, 32, 4, 32),
        (1, 64, 64, 4, 32),
        (1, 17, 117, 4, 32),
    ],
    ids=[
        "B1-I32-J13-H4D32",
        "B1-I32-J29-H4D32",
        "B1-I13-J13-H4D32",
        "B1-I29-J29-H4D32",
        "B1-I16-J33-H4D32",
        "B1-I16-J65-H4D32",
        "B1-I32-J32-H4D32-aligned",
        "B1-I64-J64-H4D32-aligned",
        "B1-I17-J117-H4D32",
    ],
)
@pytest.mark.parametrize("qkv_packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("s_kv_shape", ["BI", "flat", "B_broadcast"])
@pytest.mark.parametrize(
    "cutedsl_mode", cutedsl_test_modes(_TRIANGLE_CUTEDSL_SOURCE_MODULE), ids=lambda mode: f"impl-{mode}"
)
def test_triangle_left_mask_vs_vanilla(
    bs, I, J, num_heads, head_dim, qkv_packed, s_kv_shape, cutedsl_mode, monkeypatch
):
    """Compare CuTeDSL left-mask triangle attention against vanilla reference.

    Non-multiple-of-8 J values exercise the backend's pad/unpad paths, and
    ``qkv_packed`` slices q/k/v from one fused buffer instead of passing
    contiguous tensors. ``s_kv_shape`` covers the three forms
    :func:`_to_actual_s_kv_int32` accepts: per-row ``[B, I]``, flattened
    ``[B*I]``, and one count per batch ``[B]`` broadcast across I (the OpenFold2
    ``pair_mask = seq_mask^T seq_mask`` shape).

    Public builds test CUBINs only; private CI sets
    ``TRTBNM_TEST_CUTEDSL_MODES=source,cubin`` for both.
    """
    skip_if_no_cutedsl()
    _configure_triangle_cutedsl_mode(cutedsl_mode, monkeypatch)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = torch.bfloat16
    device = torch.device("cuda")

    H, D = num_heads, head_dim

    if qkv_packed:
        qkv = torch.randn(bs, I, J, 3, H * D, dtype=dtype, device=device)
        q = qkv[..., 0, :]
        k = qkv[..., 1, :]
        v = qkv[..., 2, :]
    else:
        q = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)
        k = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)
        v = torch.randn(bs, I, J, H * D, dtype=dtype, device=device)

    # ``B_broadcast`` needs one count shared by every row of a batch to honor
    # its contract; the other forms give each (b, i) row its own.
    if s_kv_shape == "B_broadcast":
        n_valid_b = torch.randint(low=1, high=J + 1, size=(bs,), device=device, dtype=torch.int64)
        n_valid = n_valid_b.unsqueeze(1).expand(bs, I).contiguous()
    else:
        n_valid = None
    binary_mask = make_left_aligned_mask(bs, I, J, dtype=torch.float32, device=device, n_valid=n_valid)
    s_kv_BI = binary_mask.sum(dim=-1).to(torch.int32)  # [B, I]
    if s_kv_shape == "BI":
        actual_s_kv = s_kv_BI
    elif s_kv_shape == "flat":
        actual_s_kv = s_kv_BI.reshape(bs * I).contiguous()
    else:
        actual_s_kv = s_kv_BI[:, 0].contiguous()  # [B]

    pair_bias = torch.randn(bs, H, J, J, dtype=dtype, device=device)

    additive_mask = (1.0 - binary_mask) * -1e9
    additive_mask_vanilla = additive_mask.unsqueeze(-2).unsqueeze(-2)

    vanilla_attn = VanillaTriangleAttention(0, H, D, num_kv_heads=H)
    vanilla_out = vanilla_attn.forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        biases=[additive_mask_vanilla, pair_bias],
        metadata=AttentionMetadata(),
    )

    cute_attn = TriangleAttentionCuTeLeftMask(0, H, D, num_kv_heads=H)
    cute_out = cute_attn.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=_tri_meta(qkv_packed))

    is_cubin_executable = isinstance(
        cute_attn._last_executable,
        library_runtime.CuTeDSLKernelLibraryExecutable,
    )
    assert is_cubin_executable == (cutedsl_mode == "cubin")

    assert cute_out.shape == vanilla_out.shape, f"Shape mismatch: cute={cute_out.shape}, vanilla={vanilla_out.shape}"
    diff_max = torch.max(torch.abs(cute_out.float() - vanilla_out.float()))
    diff_mean = torch.mean(torch.abs(cute_out.float() - vanilla_out.float()))
    assert diff_max < 1e-1, f"max diff {diff_max:.4f} >= 1e-1"
    assert diff_mean < 1e-2, f"mean diff {diff_mean:.4f} >= 1e-2"


# ---------------------------------------------------------------------------
# CuTeDSL left-mask triangle attention — native Hopper (SM90) kernel path
# ---------------------------------------------------------------------------


def _binds_hopper_kernel(head_dim: int, bucket: int) -> bool:
    """Whether the tuned tile at ``bucket`` binds the native Hopper kernel.

    Same ``mma_tiler_mn`` discriminator as ``_config.get_kernel_config``, read
    from the JSON so it also works in CUBIN mode, where resolving the kernel
    class is deliberately broken.
    """
    bundle = load_kernel_configs(
        triangle_config._TRI_CONFIGS_DIR,
        get_config_file_name(SM_VERSION, D=head_dim),
    )
    return "mma_tiler_mn" in bundle.configs[f"S={bucket}"]


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "I,J",
    [(16, 128), (16, 127), (320, 128)],
    ids=["I16-J128-anchor0", "I16-J127-padded", "I320-J128-anchor384"],
)
@pytest.mark.parametrize("qkv_packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize(
    "cutedsl_mode", cutedsl_test_modes(_TRIANGLE_CUTEDSL_SOURCE_MODULE), ids=lambda mode: f"impl-{mode}"
)
def test_triangle_left_mask_native_sm90(head_dim, I, J, qkv_packed, cutedsl_mode, monkeypatch):
    """Compare the native Hopper triangle-attention kernel against vanilla.

    D=32 binds the Ampere kernel class even on SM90, so
    ``test_triangle_left_mask_vs_vanilla`` never reaches the Hopper launcher's
    TMA and cluster path. D=64/128/256 do, and the test asserts that binding so
    a retune cannot silently drop it. ``J=127`` pads to 128, ``qkv_packed``
    changes descriptor strides, and the two ``I`` values select both anchors.
    """
    skip_if_not_sm90()
    _configure_triangle_cutedsl_mode(cutedsl_mode, monkeypatch)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = torch.bfloat16
    device = torch.device("cuda")

    bs, H, D = 1, 4, head_dim

    if qkv_packed:
        qkv = torch.randn(bs, I, J, 3, H * D, dtype=dtype, device=device)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
    else:
        q, k, v = (torch.randn(bs, I, J, H * D, dtype=dtype, device=device) for _ in range(3))

    binary_mask = make_left_aligned_mask(bs, I, J, dtype=torch.float32, device=device)
    actual_s_kv = binary_mask.sum(dim=-1).to(torch.int32)
    pair_bias = torch.randn(bs, H, J, J, dtype=dtype, device=device)
    additive_mask = ((1.0 - binary_mask) * -1e9).unsqueeze(-2).unsqueeze(-2)

    vanilla_out = VanillaTriangleAttention(0, H, D, num_kv_heads=H).forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        biases=[additive_mask, pair_bias],
        metadata=AttentionMetadata(),
    )

    cute_attn = TriangleAttentionCuTeLeftMask(0, H, D, num_kv_heads=H)
    cute_out = cute_attn.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=_tri_meta(qkv_packed))

    variant = cute_attn._last_variant
    assert _binds_hopper_kernel(variant.head_dim, variant.bucket), (
        f"D={head_dim} S={variant.bucket} no longer binds the Hopper kernel on "
        "SM90, so this test no longer covers the native launcher"
    )

    is_cubin_executable = isinstance(
        cute_attn._last_executable,
        library_runtime.CuTeDSLKernelLibraryExecutable,
    )
    assert is_cubin_executable == (cutedsl_mode == "cubin")
    if is_cubin_executable:
        # The registry, not just the tuned config, must pick the Hopper spec;
        # that is what routes the launch through launch_sm90().
        spec_name = type(cute_attn._last_executable._config.spec).__name__
        assert spec_name == "KernelSpecSM90", f"D={head_dim} launched via {spec_name}, not the native SM90 launcher"

    assert cute_out.shape == vanilla_out.shape, f"Shape mismatch: cute={cute_out.shape}, vanilla={vanilla_out.shape}"
    diff_max = torch.max(torch.abs(cute_out.float() - vanilla_out.float()))
    diff_mean = torch.mean(torch.abs(cute_out.float() - vanilla_out.float()))
    assert diff_max < 1e-1, f"max diff {diff_max:.4f} >= 1e-1"
    assert diff_mean < 1e-2, f"mean diff {diff_mean:.4f} >= 1e-2"
