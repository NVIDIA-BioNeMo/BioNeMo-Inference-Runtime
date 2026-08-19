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
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_pair_weighted_averaging_weights,
    load_pair_weighted_averaging_weights_torch,
)
from test_utils.boltz.ref_layers import RefPairWeightedAveraging

from bionemo_ir._torch import _cutedsl_kernel_library as library_runtime
from bionemo_ir._torch.auto_chunk import ChunkPolicy
from bionemo_ir._torch.custom_ops import pair_weighted_averaging as pwa_ops
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import (
    PairWeightedAveragingCuTe,
    _select_pwa_config_bucket,
    get_pair_weighted_averaging_op,
    is_profitable_pwa_shape,
    select_pwa_config,
)
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import cutedsl as pwa_cutedsl
from bionemo_ir._torch.custom_ops.pair_weighted_averaging._config import (
    _select_pwa_config_selection_bucket,
)
from bionemo_ir._torch.layers.pair_averaging import PairWeightedAveraging
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import SM_VERSION, cutedsl_test_modes, skip_if_no_cutedsl

_CUTEDSL_SM = (80, 90, 100, 103)
_PWA_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.pair_weighted_averaging._source"
_PWA_TEST_MODES = cutedsl_test_modes(_PWA_SOURCE_MODULE)

# (c_h, c_m) tuples with tuned configs and CUBINs, and the models behind them.
_PWA_DIMS = [(32, 64), (8, 64), (8, 128)]
_PWA_DIM_IDS = ["boltz-d32cm64", "of3-d8cm64", "protenix-d8cm128"]
_PWA_MODE_CACHES: dict[str, dict] = {
    "source": {},
    "cubin": {},
}


def _configure_pwa_mode(mode: str, monkeypatch) -> None:
    """Force one PWA implementation without allowing silent fallback."""
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(PairWeightedAveragingCuTe, "_compiled_cache", _PWA_MODE_CACHES[mode])

    if mode == "cubin":
        try:
            importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
        except ImportError:
            pytest.fail("CUBIN test mode requires the _cutedsl_kernels extension")

        try:
            source_module = importlib.import_module("bionemo_ir._torch.custom_ops.pair_weighted_averaging._source")
        except ImportError:
            source_module = None
        if source_module is not None:

            def source_unavailable(*_args, **_kwargs):
                raise ModuleNotFoundError("CuTeDSL source disabled by CUBIN test mode")

            monkeypatch.setattr(source_module, "resolve_pwa_source", source_unavailable)
        monkeypatch.setattr(library_runtime, "_kernel_library", None)
        return

    def reject_cubin_fallback(*_args, **_kwargs):
        raise AssertionError("source test mode unexpectedly fell back to the CUBIN library")

    monkeypatch.setattr(pwa_cutedsl, "populate_compiled_cache_from_library", reject_cubin_fallback)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    # Sequence(S)-rows per chunk via a registry-style ChunkPolicy (None -> dense path). Row-chunking
    # over S is numerically identical, so the chunked output must still match the golden ref.
    chunk: int | None = None
    n_seq: int = 32
    n_res: int = 64


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        # fp16 / bf16 single-GPU on a registered (H, c_h, c_m) tuple routes
        # through the SM80 fused PWA CuTe custom op; fp32 stays on the eager path.
        Scenario(torch_dtype="bfloat16"),
        Scenario(torch_dtype="float16"),
        # Larger N exercises the kernel's hierarchical N-then-S config selection.
        Scenario(torch_dtype="bfloat16", n_res=128),
        # fp32 cannot use the fused kernel, so these exercise the auto-chunk fallback.
        Scenario(chunk=1),
        Scenario(chunk=3, n_seq=8),
    ],
    ids=["float32", "bfloat16", "float16", "bfloat16_n128", "chunk1_fp32", "chunk3_partial_fp32"],
)
def test_pair_weighted_averaging(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_m = RefPairWeightedAveraging.load_weights()
    ref_m = ref_m.to(device)

    weights_and_biases = create_pair_weighted_averaging_weights(from_ref=ref_m)

    # A low min_size makes the policy trip at the test's token count, so `chunk` scenarios exercise
    # the head-chunked accumulate path; `chunk=None` leaves the registry default (dense here).
    chunk_policy = ChunkPolicy(chunk_size=sc.chunk, min_size=1) if sc.chunk is not None else None
    pair_weighted_averaging = PairWeightedAveraging(
        c_m=ref_m.c_m, c_z=ref_m.c_z, c_h=ref_m.c_h, num_heads=ref_m.num_heads, dtype=dtype, chunk_policy=chunk_policy
    )
    load_pair_weighted_averaging_weights_torch(pair_weighted_averaging, weights_and_biases, dtype=dtype)
    pair_weighted_averaging.to(device)

    # On a CuTeDSL-capable GPU the half-precision path (boltz-2 PWA is H=8,
    # c_h=32, c_m=64) must resolve to the fused custom op -- guards against a
    # silent fall-back to eager.
    if dtype in (torch.float16, torch.bfloat16) and SM_VERSION in _CUTEDSL_SM:
        assert pair_weighted_averaging._pwa_op_eligible
        assert isinstance(get_pair_weighted_averaging_op(dtype), PairWeightedAveragingCuTe)

    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_m.c_m, dtype=torch.float32).cuda()
    z = torch.randn(bs, sc.n_res, sc.n_res, ref_m.c_z, dtype=torch.float32).cuda()
    # 0/1 pair mask: the layer applies ``(1 - mask) * -inf`` with inf=1e9, so a
    # random *normal* mask overflows fp16 (-> +/-inf -> NaN softmax); a 0/1 mask
    # keeps the masked bias at 0 / -1e9 and is the realistic input anyway.
    mask = torch.randint(0, 2, (bs, sc.n_res, sc.n_res), dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_m(m, z, mask)
        m = m.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_m = ref_m.to(dtype)

        ref_output = ref_m(m, z, mask)
        output = pair_weighted_averaging.forward(m, z, mask)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("s_rows", [1, 2, 3, 5], ids=["s1", "s2", "s3_partial", "s5_all"])
def test_pair_weighted_averaging_chunk_matches_dense(s_rows: int):
    """Sequence(S)-row chunking (registry policy) matches the dense path.

    Same instance dense vs chunked (no golden weights needed): each S-slice is independent (the
    attention mixes only the token dims), so the concatenated result is numerically identical.
    """
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")

    pwa = PairWeightedAveraging(c_m=64, c_z=32, c_h=16, num_heads=8, dtype=torch.float32).to(device)
    pwa.eval()
    # Constructed weights are zero-initialized (production loads them); give them real values so
    # the dense-vs-chunked comparison is meaningful rather than 0 == 0.
    with torch.no_grad():
        for p in pwa.parameters():
            p.normal_(mean=0.0, std=0.1)
    m = torch.randn(1, 5, 48, 64, device=device)  # S=5
    z = torch.randn(1, 48, 48, 32, device=device)
    mask = torch.randint(0, 2, (1, 48, 48), dtype=torch.float32, device=device)

    with torch.inference_mode():
        # Registry default (memory-scaled min_size) does not trip at S=5 -> dense reference.
        dense = pwa(m, z, mask)

        pwa.chunk_policy = ChunkPolicy(chunk_size=s_rows, min_size=1)  # chunk over S
        chunked = pwa(m, z, mask)

    torch.testing.assert_close(chunked, dense, atol=1e-4, rtol=1e-4)


# Self-contained fused-kernel/eager parity across dtypes and tail shapes.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("c_h,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
@pytest.mark.parametrize(
    "n_seq,n_res",
    [(16, 128), (8, 100), (4, 256), (8, 130), (200, 128), (48, 33), (17, 127), (31, 129)],
    ids=["s16n128", "s8n100", "s4n256", "s8n130", "s200n128", "s48n33", "s17n127", "s31n129"],
)
@pytest.mark.parametrize("cutedsl_mode", _PWA_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_pwa_cute_matches_eager(dtype, c_h, c_m, n_seq, n_res, cutedsl_mode, monkeypatch):
    skip_if_no_cutedsl("pair_weighted_averaging")
    torch.manual_seed(0)
    _configure_pwa_mode(cutedsl_mode, monkeypatch)
    c_z, num_heads = 128, 8

    layer = PairWeightedAveraging(c_m=c_m, c_z=c_z, c_h=c_h, num_heads=num_heads, dtype=dtype).cuda()
    with torch.no_grad():
        for lin in (layer.fused_proj_m_g, layer.proj_z, layer.proj_o):
            lin.weight.normal_(0, 1.0 / lin.weight.shape[1] ** 0.5)

    # A registered (H, c_h, c_m) tuple resolves to the CuTe backend on this GPU.
    assert layer._pwa_op_eligible
    assert isinstance(get_pair_weighted_averaging_op(dtype, D=c_h, c_m=c_m, H=num_heads), PairWeightedAveragingCuTe)

    m = torch.randn(1, n_seq, n_res, c_m, dtype=dtype, device="cuda") * 0.5
    z = torch.randn(1, n_res, n_res, c_z, dtype=dtype, device="cuda") * 0.5
    mask = (torch.rand(1, n_res, n_res, device="cuda") < 0.9).to(dtype)

    with torch.inference_mode():
        out_kernel = layer(m, z, mask)
        layer._pwa_op_eligible = False  # force the original eager path
        out_eager = layer(m, z, mask)

    assert out_kernel.shape == out_eager.shape == (1, n_seq, n_res, c_m)
    diff = (out_kernel.float() - out_eager.float()).abs()
    rel_l2 = (diff.norm() / out_eager.float().norm().clamp_min(1e-6)).item()
    assert rel_l2 < 2e-2, f"kernel vs eager rel_l2={rel_l2:.3e} (dtype={dtype}, S={n_seq}, N={n_res})"


def test_pair_weighted_averaging_op_selector():
    """The selector gates the CuTe op on dtype, hardware, and registered dims."""
    op_fp32 = get_pair_weighted_averaging_op(torch.float32)
    # Unregistered tuples: an untuned head dim, an untuned channel width, and a
    # registered head dim paired with the wrong channel width.
    op_d16 = get_pair_weighted_averaging_op(torch.bfloat16, D=16)
    op_cm256 = get_pair_weighted_averaging_op(torch.bfloat16, D=8, c_m=256)
    op_d32cm128 = get_pair_weighted_averaging_op(torch.bfloat16, D=32, c_m=128)
    if SM_VERSION in _CUTEDSL_SM:
        for D, c_m in _PWA_DIMS:
            op = get_pair_weighted_averaging_op(torch.bfloat16, D=D, c_m=c_m)
            assert isinstance(op, PairWeightedAveragingCuTe), f"D={D}, c_m={c_m}"
    for op in (op_fp32, op_d16, op_cm256, op_d32cm128):
        assert not isinstance(op, PairWeightedAveragingCuTe)


@pytest.mark.parametrize(
    "D,c_m,tokens,expected",
    [
        (32, 64, 1536, True),
        (32, 64, 1537, False),
        (8, 64, 2048, True),
        (8, 64, 2049, False),
        (8, 128, 2049, False),
    ],
    ids=["d32-at-limit", "d32-over-limit", "d8-at-limit", "d8-over-limit", "d8cm128-over-limit"],
)
def test_pair_weighted_averaging_token_limit(D, c_m, tokens, expected):
    """Past its measured crossover the D=8 kernel must defer to chunked eager."""
    H = 8
    w = torch.zeros(1, H, tokens, tokens, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(1, 2, tokens, H * D, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(c_m, H * D, dtype=torch.bfloat16, device="cuda")
    # is_supported() answers for the GPU it runs on, and PWA ships CUBINs for
    # _CUTEDSL_SM alone, so elsewhere it declines whatever the shape. The
    # crossover rule below is dimensions only and holds everywhere.
    fused_here = expected and SM_VERSION in _CUTEDSL_SM
    assert PairWeightedAveragingCuTe().is_supported(w, v, weight) is fused_here
    assert is_profitable_pwa_shape(H, D, c_m, tokens) is expected


def test_pair_weighted_averaging_declines_before_building_inputs(monkeypatch):
    """An unprofitable shape must reach the eager fallback, not the fused op."""
    layer = PairWeightedAveraging(c_m=64, c_z=128, c_h=8, num_heads=8, dtype=torch.bfloat16).cuda()
    assert layer._pwa_op_eligible
    monkeypatch.setattr(
        "bionemo_ir._torch.layers.pair_averaging.is_profitable_pwa_shape",
        lambda *_args, **_kwargs: False,
    )

    def fail(*args, **kwargs):
        raise AssertionError("the fused path must not be entered for an unprofitable shape")

    monkeypatch.setattr(layer, "_forward_fused", fail)
    tokens = 32
    m = torch.zeros(1, 4, tokens, 64, dtype=torch.bfloat16, device="cuda")
    z = torch.zeros(1, tokens, tokens, 128, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(1, tokens, tokens, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        assert layer(m, z, mask).shape == (1, 4, tokens, 64)


@pytest.mark.parametrize("sm", [100, 103])
def test_pair_weighted_averaging_supports_blackwell(monkeypatch, sm):
    sentinel = object()
    monkeypatch.setattr(pwa_ops, "get_sm_version", lambda: sm)
    monkeypatch.setattr(pwa_ops, "_pwa_cute_instance", sentinel)
    assert pwa_ops.get_pair_weighted_averaging_op(torch.bfloat16) is sentinel


def test_pair_weighted_averaging_config_selects_n_bucket_before_s():
    """Changing S must not make a fixed N jump to another tuned N bucket."""
    # For N=1024, S=1600 is closer to the N=1024/S=2048 variant than S=1024.
    # A sqrt(N*S) selector would instead pick the N=1536/S=1024 anchor.
    config = select_pwa_config(sm_version=80, I=1024, J=1024, S=1600, dtype_str="bf16")

    assert config.KO_TILE == 32
    assert config.num_stages_w == 4

    selected, variants = _select_pwa_config_bucket(sm_version=80, I=1024, J=1024, S=1600, dtype_str="bf16")
    assert selected.config_key() == config.config_key()
    # N=1024 has three S anchors but two unique kernel configs; duplicates do
    # not need a second compilation.
    assert len({variant.config_key() for variant in variants}) == 2


@pytest.mark.parametrize(
    "D,c_m,I,J,S",
    [
        (32, 64, 256, 1024, 1600),
        (8, 64, 256, 1024, 512),
        (8, 128, 384, 1536, 512),
    ],
    ids=["boltz-rectangular", "of3-rectangular", "protenix-rectangular"],
)
def test_pwa_cubin_and_python_select_same_rectangular_bucket(D, c_m, I, J, S):
    skip_if_no_cutedsl("pair_weighted_averaging")
    library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    launcher = library.pair_weighted_averaging
    selected, _ = _select_pwa_config_selection_bucket(
        SM_VERSION,
        I,
        J,
        S,
        "bf16",
        D=D,
        c_m=c_m,
    )
    config = launcher.make_kernel_config(
        SM_VERSION,
        I,
        J,
        S,
        D,
        c_m,
        launcher.DType.BFLOAT16,
    )

    assert config.n_anchor == selected.n_anchor
    assert config.s_anchor == selected.s_anchor
    assert config.tile_i == selected.params.TILE_I
    assert config.tile_s == selected.params.TILE_S
    assert config.tile_j == selected.params.TILE_J
    assert config.num_threads == selected.params.num_threads


@pytest.mark.parametrize("cutedsl_mode", _PWA_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_pair_weighted_averaging_reuses_dynamic_shape_executable(
    cutedsl_mode,
    monkeypatch,
):
    """One always-predicated executable serves aligned and ragged extents."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    torch.manual_seed(0)
    _configure_pwa_mode(cutedsl_mode, monkeypatch)
    op = get_pair_weighted_averaging_op(torch.bfloat16)
    assert isinstance(op, PairWeightedAveragingCuTe)

    H, D, c_m = 8, 32, 64
    Wo = torch.randn(c_m, H * D, dtype=torch.bfloat16, device="cuda")
    executable_ids = []

    shapes = ((1, 7, 64, 64), (2, 11, 96, 128), (1, 7, 33, 33), (2, 11, 47, 47))
    for B, S, I, N in shapes:
        Jp = (N + 7) // 8 * 8
        w = torch.randn(B, H, I, Jp, dtype=torch.bfloat16, device="cuda")
        w[..., N:] = 0
        v_storage = torch.randn(B, S, N, 2 * H * D, dtype=torch.bfloat16, device="cuda")
        g_storage = torch.randn(B, S, I, 2 * H * D, dtype=torch.bfloat16, device="cuda")
        v = v_storage[..., : H * D]
        g = g_storage[..., H * D :]
        out = op(w, v, g, Wo)

        v5 = v.reshape(B, S, N, H, D)
        o = torch.einsum("bhij,bsjhd->bhsid", w[..., :N].float(), v5.float())
        o = o.permute(0, 2, 3, 1, 4).reshape(B, S, I, H * D)
        ref = ((torch.sigmoid(g.float()) * o) @ Wo.float().t()).to(w.dtype)
        rel_l2 = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()
        assert rel_l2 < 2e-2

        config = select_pwa_config(sm_version=op._sm_version, I=I, J=N, S=S, dtype_str="bf16")
        key = pwa_cutedsl._compile_cache_key(op._sm_version, config, torch.bfloat16)
        executable_ids.append(id(op._compiled_cache[key]))

    assert len(set(executable_ids)) == 1


@pytest.mark.skipif(
    set(_PWA_TEST_MODES) != {"source", "cubin"},
    reason="both implementations are required for a bitwise equivalence check",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("D,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
@pytest.mark.parametrize(
    "S,I,N,Jp",
    [(16, 128, 128, 128), (17, 127, 127, 128), (31, 129, 129, 136), (7, 33, 33, 64)],
    ids=["aligned", "mixed-tails", "all-tails", "tile-j-padding"],
)
def test_pwa_source_and_cubin_agree_bitwise(dtype, D, c_m, S, I, N, Jp, monkeypatch):
    """The direct launcher must reproduce the source launch exactly."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    torch.manual_seed(123)
    B, H = 1, 8
    w = torch.randn(B, H, I, Jp, dtype=dtype, device="cuda")
    w[..., N:] = 0
    vg = torch.randn(B, S, max(I, N), 2 * H * D, dtype=dtype, device="cuda")
    v = vg[:, :, :N, : H * D]
    g = vg[:, :, :I, H * D :]
    weight = torch.randn(c_m, H * D, dtype=dtype, device="cuda")

    with monkeypatch.context() as source_patch:
        _configure_pwa_mode("source", source_patch)
        source = PairWeightedAveragingCuTe()(w, v, g, weight)
    with monkeypatch.context() as cubin_patch:
        _configure_pwa_mode("cubin", cubin_patch)
        cubin = PairWeightedAveragingCuTe()(w, v, g, weight)

    torch.testing.assert_close(cubin, source, atol=0.0, rtol=0.0)
