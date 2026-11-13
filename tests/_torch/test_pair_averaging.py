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
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_pair_weighted_averaging_weights,
    load_pair_weighted_averaging_weights_torch,
)
from test_utils.boltz.ref_layers import RefPairWeightedAveraging

from bionemo_ir._torch.custom_ops import pair_weighted_averaging as pwa_ops
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import (
    PairWeightedAveragingCuTe,
    _select_pwa_config_bucket,
    get_pair_weighted_averaging_op,
    is_profitable_pwa_shape,
    select_pwa_config,
)
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import _config as pwa_config
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import cutedsl as pwa_cutedsl
from bionemo_ir._torch.custom_ops.pair_weighted_averaging._config import (
    _select_pwa_config_selection_bucket,
    max_fused_tokens,
)
from bionemo_ir._torch.layers.pair_averaging import PairWeightedAveraging
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryExecutable, load_kernel_configs
from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import SM_VERSION, cutedsl_test_modes, require_cubin_library, skip_if_no_cutedsl

_CUTEDSL_SM = (80, 90, 100, 103, 107, 120, 121)
_PWA_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.pair_weighted_averaging._source"
_PWA_TEST_MODES = cutedsl_test_modes(_PWA_SOURCE_MODULE)
_PWA_CUBIN_INDEX = (
    Path(__file__).resolve().parents[2]
    / "cpp"
    / "kernels"
    / "cutedsl_pair_weighted_averaging"
    / "cubins"
    / "index.json"
)
# (c_h, c_m) tuples with tuned configs and CUBINs, and the models behind them.
_PWA_DIMS = [(32, 64), (8, 64), (8, 128)]
_PWA_DIM_IDS = ["boltz-d32cm64", "of3-d8cm64", "protenix-d8cm128"]
_PWA_MODE_CACHES: dict[str, dict] = {
    "source": {},
    "cubin": {},
}


def _sm107_cubin_published() -> bool:
    """Whether the committed corpus lists an SM107 pack.

    The protected refresh adds ``sm_107a`` after this change lands. Until that
    entry exists, CUBIN mode on SM107 has nothing to launch.
    """
    packs = json.loads(_PWA_CUBIN_INDEX.read_text())["packs"]
    return "sm_107a" in packs


def _configure_pwa_mode(mode: str, monkeypatch) -> None:
    """Force one PWA implementation without allowing silent fallback."""
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(PairWeightedAveragingCuTe, "_compiled_cache", _PWA_MODE_CACHES[mode])

    if mode == "cubin":
        if SM_VERSION == 107 and not _sm107_cubin_published():
            pytest.skip("the SM107 CUBIN pack is not shipped yet; the protected refresh publishes it")
        require_cubin_library()

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


_BITWISE = pytest.mark.skipif(
    set(_PWA_TEST_MODES) != {"source", "cubin"},
    reason="both implementations are required for a bitwise equivalence check",
)


def _skip_unless_packaged(selected, I: int, J: int, S: int, dtype: torch.dtype) -> None:
    """Skip a CUBIN check while the packaged image for this shape predates the selected tile."""
    if not PairWeightedAveragingCuTe()._packaged_image_matches(selected, I, J, S, dtype):
        pytest.skip("packaged CUBIN does not match the tuned tile; protected refresh republishes it")


def _source_and_cubin(monkeypatch, selected, w, v, g, weight) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the op from source and from the packaged image, and check that the image launched."""
    if not PairWeightedAveragingCuTe().is_supported(w, v, weight):
        pytest.skip(f"SM{SM_VERSION} runs this shape eagerly")
    with monkeypatch.context() as source_patch:
        _configure_pwa_mode("source", source_patch)
        source = PairWeightedAveragingCuTe()(w, v, g, weight)
    with monkeypatch.context() as cubin_patch:
        _configure_pwa_mode("cubin", cubin_patch)
        cubin = PairWeightedAveragingCuTe()(w, v, g, weight)
        key = pwa_cutedsl._selection_cache_key(SM_VERSION, selected, w.dtype)
        assert isinstance(PairWeightedAveragingCuTe._compiled_cache.get(key), CuTeDSLKernelLibraryExecutable)
    return source, cubin


def _config_entries(sm: int, D: int, c_m: int) -> dict:
    """The tuned configs of one PWA file, from the loader selection uses."""
    bundle = load_kernel_configs(pwa_config._PWA_CONFIGS_DIR, pwa_config.config_file_name(sm, D, c_m))
    return dict(bundle.configs) if bundle is not None else {}


def _fp32_reference(w, v, g, weight) -> torch.Tensor:
    return pwa_ops._invoke_vanilla_pwa(w.float(), v.float(), g.float(), weight.float())


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

        # One-sided: the fused path keeps z's LayerNorm in fp32 and may beat the half-precision reference.
        assert diff0_max <= 1.5 * diff1_max
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("D,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
@pytest.mark.parametrize("N", [2056, 4100], ids=["n2056", "n4100"])
@pytest.mark.parametrize("cutedsl_mode", _PWA_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_pwa_kernel_covers_every_tile_past_2048_tokens(dtype, D, c_m, N, cutedsl_mode, monkeypatch) -> None:
    """SM90 launches 128-token tiles in groups of 16; a partial last group must still write every tile once."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    if SM_VERSION not in _CUTEDSL_SM:
        pytest.skip(f"PWA ships no CUBINs for SM{SM_VERSION}")
    _configure_pwa_mode(cutedsl_mode, monkeypatch)
    # Exercise the launch order itself, whatever size this SM's limit fuses.
    monkeypatch.setattr(pwa_cutedsl, "is_profitable_shape", lambda *_args, **_kwargs: True)
    torch.manual_seed(0)
    B, H, S = 1, 8, 5
    hd = H * D
    w = torch.softmax(torch.randn(B, H, N, N, device="cuda"), dim=-1)
    w = torch.nn.functional.pad(w, (0, (N + 7) // 8 * 8 - N)).to(dtype)
    vg = torch.randn(B, S, N, 2 * hd, dtype=dtype, device="cuda")
    v, g = vg[..., :hd], vg[..., hd:]
    weight = (torch.randn(c_m, hd, device="cuda") * hd**-0.5).to(dtype)
    op = PairWeightedAveragingCuTe()
    assert op.is_supported(w, v, weight)

    out = op(w, v, g, weight).float()
    ref = _fp32_reference(w, v, g, weight)
    rel_l2 = ((out - ref).norm() / ref.norm()).item()
    assert rel_l2 < 2e-2, f"kernel vs fp32 reference rel_l2={rel_l2:.3e} (dtype={dtype}, N={N})"


def test_pwa_sm90_rows_per_launch(monkeypatch) -> None:
    """From 6144 tokens a D=32 launch covers whole waves of 16-token-tile groups, and a D=8 launch 128 rows."""
    monkeypatch.setattr(pwa_cutedsl, "_sm_count", lambda _device: 132)
    device = torch.device("cuda", 0)

    def rows(D: int, c_m: int, N: int, S: int = 1024) -> int:
        selected = _select_pwa_config_selection_bucket(90, N, N, S, "bf16", H=8, D=D, c_m=c_m)[0]
        return pwa_cutedsl._sm90_rows_per_launch(selected, N, N, S, device)

    for D, c_m in _PWA_DIMS:
        assert rows(D, c_m, 6143) == 1024
    # 64 token tiles by 8 row tiles of 4 rows is 512 CTAs in 4 waves; 6 or 4 row tiles tie, and ties go to fewer
    # launches.
    assert rows(32, 64, 8192) == 32
    # 88 token tiles: 6 row tiles make 528 CTAs, exactly 4 waves; 8 make 704, which end on a third-full sixth wave.
    assert rows(32, 64, 11264) == 24
    assert rows(32, 64, 8192, S=20) == 20
    for c_m in (64, 128):
        assert rows(8, c_m, 6144) == 128
        assert rows(8, c_m, 6144, S=100) == 100
    # A 16-token-tile group fits fewer row tiles into one wave of a smaller GPU.
    monkeypatch.setattr(pwa_cutedsl, "_sm_count", lambda _device: 114)
    assert rows(32, 64, 8192) <= 28


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("D,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
@pytest.mark.parametrize("cutedsl_mode", _PWA_TEST_MODES, ids=lambda mode: f"impl-{mode}")
def test_pwa_sm90_msa_row_blocks_match_one_launch(dtype, D, c_m, cutedsl_mode, monkeypatch) -> None:
    """Launching each batch entry's MSA rows in blocks, the last one partial, writes the bits of one launch."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    if SM_VERSION != 90:
        pytest.skip("only the SM90 kernels launch MSA rows in blocks")
    _configure_pwa_mode(cutedsl_mode, monkeypatch)
    monkeypatch.setattr(pwa_cutedsl, "is_profitable_shape", lambda *_args, **_kwargs: True)
    # Split at a size that runs quickly.
    monkeypatch.setattr(pwa_cutedsl, "_SM90_ROW_SPLIT_TOKENS", 0)
    torch.manual_seed(0)
    B, H, S, N = 2, 8, 300, 520
    hd = H * D
    w = torch.softmax(torch.randn(B, H, N, N, device="cuda"), dim=-1).to(dtype)
    vg = torch.randn(B, S, N, 2 * hd, dtype=dtype, device="cuda")
    v, g = vg[..., :hd], vg[..., hd:]
    weight = (torch.randn(c_m, hd, device="cuda") * hd**-0.5).to(dtype)
    selected = _select_pwa_config_selection_bucket(90, N, N, S, pwa_cutedsl._dtype_str(dtype), H=H, D=D, c_m=c_m)[0]
    rows = pwa_cutedsl._sm90_rows_per_launch(selected, N, N, S, w.device)
    assert rows < S and S % rows, f"{rows} rows per launch must split S={S} with a partial last block"
    op = PairWeightedAveragingCuTe()

    blocks = op(w, v, g, weight)
    monkeypatch.setattr(pwa_cutedsl, "_sm90_rows_per_launch", lambda _selected, _I, _J, S, _device: S)
    assert torch.equal(op(w, v, g, weight), blocks)


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
    w = torch.zeros(1, H, tokens, (tokens + 7) // 8 * 8, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(1, 2, tokens, H * D, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(c_m, H * D, dtype=torch.bfloat16, device="cuda")
    # is_supported() answers for the GPU it runs on, and PWA ships CUBINs for
    # _CUTEDSL_SM alone, so elsewhere it declines whatever the shape. The
    # default crossover rule below is dimensions only and holds everywhere;
    # SM90, SM100 and SM103 move the limits (see the per-SM tests).
    expected_here = is_profitable_pwa_shape(H, D, c_m, tokens, sm_version=SM_VERSION)
    fused_here = expected_here and SM_VERSION in _CUTEDSL_SM
    assert PairWeightedAveragingCuTe().is_supported(w, v, weight) is fused_here
    assert is_profitable_pwa_shape(H, D, c_m, tokens) is expected


@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
@pytest.mark.parametrize("c_m", [64, 128])
@pytest.mark.parametrize("sm", [100, 103, 107])
def test_pair_weighted_averaging_blackwell_d8_runs_the_tcgen05_kernel(sm: int, c_m: int, dtype_str: str) -> None:
    """SM100, SM103 and SM107 D=8 run the tcgen05 kernel at every anchor, in both dtypes, from 33 tokens to 12288."""
    try:
        source = importlib.import_module(_PWA_SOURCE_MODULE)
    except ImportError:  # a build without kernel sources still checks the selection
        source = None

    for N, S in [(33, 7), (128, 1), (256, 16), (512, 512), (1024, 1024), (2048, 2048), (8192, 1024), (12288, 4096)]:
        selected, bucket = _select_pwa_config_selection_bucket(sm, N, N, S, dtype_str, H=8, D=8, c_m=c_m)
        assert (selected.kernel_abi, selected.kernel_variant) == ("sm100", "d8"), (N, S)
        if source is not None:
            for selection in bucket:
                # Building the kernel validates the tile, the cluster, TMEM and shared memory.
                source.resolve_pwa_source(selection, dtype_str)


@pytest.mark.parametrize(
    ("sm", "D", "c_m", "transposed"),
    [
        (100, 8, 64, True),
        (100, 8, 128, True),
        (100, 32, 64, False),
        (90, 8, 64, False),
        (103, 8, 64, True),
        (107, 8, 64, True),
        (80, 8, 128, False),
    ],
)
def test_pwa_only_the_tcgen05_d8_kernel_loads_transposed_v(
    sm: int, D: int, c_m: int, transposed: bool, monkeypatch
) -> None:
    """The layer asks once which v layout to write; every selection of that file agrees."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (sm // 10, sm % 10))
    assert PairWeightedAveragingCuTe().loads_transposed_value(D, c_m) is transposed
    for N, S in [(128, 16), (1024, 1024), (8192, 2048)]:
        selected, bucket = _select_pwa_config_selection_bucket(sm, N, N, S, "bf16", D=D, c_m=c_m)
        assert all(selection.transposed_v is transposed for selection in (selected, *bucket)), (N, S)


def test_pwa_transposed_value_buffer_has_a_tma_pitch() -> None:
    """The producer's v^T keeps its logical N with a j pitch TMA can encode."""
    v = pwa_cutedsl.transposed_value_buffer(2, 3, 64, 37, torch.bfloat16, torch.device("cpu"))
    assert v.shape == (2, 3, 64, 37) and v.stride(-1) == 1 and v.stride(-2) == 40
    assert all(stride * v.element_size() % 16 == 0 for stride in v.stride()[:-1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("c_m", [64, 128])
@pytest.mark.parametrize("S,N", [(7, 33), (37, 300), (16, 1024)], ids=["s7n33", "s37n300", "s16n1024"])
def test_pwa_transposed_value_matches_natural(dtype, c_m: int, S: int, N: int) -> None:
    """v^T from the producer and natural v (relaid out by the op) give the same SM100 D=8 output."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    if SM_VERSION != 100:
        pytest.skip("only the SM100 D=8 kernel loads v transposed")
    torch.manual_seed(0)
    B, H, D = 1, 8, 8
    hd = H * D
    w = torch.softmax(torch.randn(B, H, N, N, device="cuda"), dim=-1)
    w = torch.nn.functional.pad(w, (0, (N + 7) // 8 * 8 - N)).to(dtype)
    v = (torch.randn(B, S, N, hd, device="cuda") * 0.5).to(dtype)
    g = (torch.randn(B, S, N, hd, device="cuda") * 0.5).to(dtype)
    weight = (torch.randn(c_m, hd, device="cuda") * hd**-0.5).to(dtype)
    op = PairWeightedAveragingCuTe()
    assert op.loads_transposed_value(D, c_m)
    vt = pwa_cutedsl.transposed_value_buffer(B, S, hd, N, dtype, v.device)
    vt.copy_(v.transpose(-1, -2))
    out_t = op(w, vt.transpose(-1, -2), g, weight)
    out_n = op(w, v, g, weight)
    assert torch.equal(out_t, out_n)
    # A caller's contiguous v^T has no pitch padding when N % 8 != 0; the op pads it.
    assert torch.equal(op(w, v.transpose(-1, -2).contiguous().transpose(-1, -2), g, weight), out_n)
    ref = _fp32_reference(w, v, g, weight)
    rel_l2 = ((out_t.float() - ref).norm() / ref.norm()).item()
    assert rel_l2 < 2e-2, f"transposed-v kernel vs fp32 reference rel_l2={rel_l2:.3e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("B,N", [(1, 33), (2, 47), (1, 520), (1, 2048)])
def test_pwa_prologue_matches_the_eager_expression(dtype, B: int, N: int) -> None:
    """One pass of bias, softmax and J padding against PyTorch, a fully masked row included."""
    if not torch.cuda.is_available():
        pytest.skip("Triton kernel")
    from bionemo_ir.dsl_kernels.triton.pwa_prologue import pwa_prologue

    torch.manual_seed(0)
    inf = 1e9 if dtype == torch.bfloat16 else 6e4
    b = (torch.randn(B, 8, N, N, device="cuda") * 3).to(dtype)
    mask = (torch.rand(B, N, N, device="cuda") > 0.2).to(dtype)
    mask[:, 0] = 0
    ref = torch.softmax(b + (1 - mask[:, None]) * -inf, dim=-1)
    ref = torch.nn.functional.pad(ref, (0, (N + 7) // 8 * 8 - N))
    out = pwa_prologue(b, mask, inf)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=4e-3, rtol=0)
    assert (out[..., N:] == 0).all()
    assert pwa_prologue(b.float(), mask.float(), inf) is None


@pytest.mark.parametrize("sm", [None, 80])
def test_pair_weighted_averaging_other_sms_keep_the_default_limits(sm):
    """SM80 keeps the defaults: D=8 through 2048 tokens and D=32 through 1536."""
    for c_m in (64, 128):
        assert max_fused_tokens(8, c_m, sm_version=sm) == 2048
        assert is_profitable_pwa_shape(8, 8, c_m, 2048, sm_version=sm)
        assert not is_profitable_pwa_shape(8, 8, c_m, 2049, sm_version=sm)
    assert max_fused_tokens(32, 64, sm_version=sm) == 1536


@pytest.mark.parametrize("sm", [90, 100, 103, 107])
def test_pair_weighted_averaging_fuses_at_any_size_on_hopper_and_blackwell(sm: int) -> None:
    """SM90, SM100, SM103 and SM107 run every shape fused in both dtypes, so none has a token limit.

    The fused layer beats eager at every size eager fits on H100, B200 and Vera Rubin. SM103 runs the SM100 tables.
    """
    for D, c_m in _PWA_DIMS:
        for dtype_str in ("bf16", "fp16"):
            assert max_fused_tokens(D, c_m, sm_version=sm, dtype_str=dtype_str) is None
            assert is_profitable_pwa_shape(8, D, c_m, 65536, sm_version=sm, dtype_str=dtype_str)


@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
@pytest.mark.parametrize("c_m", [64, 128])
def test_pwa_sm90_d8_cluster_grows_with_tokens(c_m: int, dtype_str: str) -> None:
    """D=8 runs unclustered through 512 tokens, then shares w across four MSA-row tiles, and past 2560 tokens, or
    deep MSAs at 2048, shares v across two token tiles as well."""

    def cluster(N: int, S: int = 1024) -> tuple[int, int]:
        return _select_pwa_config_selection_bucket(90, N, N, S, dtype_str, H=8, D=8, c_m=c_m)[0].params.cluster

    assert cluster(512) == cluster(512, S=4096) == (1, 1)
    assert cluster(513) == cluster(2048) == cluster(2560) == (1, 4)
    assert cluster(2048, S=2048) == cluster(2561) == cluster(16384, S=128) == (2, 4)


@pytest.mark.parametrize("sm", [120, 121])
def test_pair_weighted_averaging_sm120_d32_reaches_the_measured_anchor(sm):
    """The N=2048 D=32 entries record a win, so the limit has to let that anchor be selected."""
    assert max_fused_tokens(32, 64, sm_version=sm) == 2048
    assert is_profitable_pwa_shape(8, 32, 64, 2048, sm_version=sm)
    assert not is_profitable_pwa_shape(8, 32, 64, 2049, sm_version=sm)
    for c_m in (64, 128):
        assert max_fused_tokens(8, c_m, sm_version=sm) == 2048


# (tokens, MSA rows) -> (TILE_I, TILE_S, num_threads) where the Vera Rubin sweep replaced the SM100 tile.
_SM107_D32_SMALL_TOKEN_TILES = {
    (128, 256): (128, 1, 192),
    (128, 512): (128, 2, 320),
    (256, 512): (128, 2, 320),
}


@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "tokens,msa_rows,tile_i,tile_s,num_threads",
    [
        (128, 1, 128, 1, 192),
        (128, 128, 128, 1, 192),
        (128, 256, 128, 2, 320),
        (128, 512, 128, 4, 320),
        (256, 16, 128, 1, 192),
        (256, 128, 128, 2, 320),
        (256, 512, 128, 4, 320),
        (768, 512, 128, 4, 320),
        (1024, 16, 256, 4, 192),
        (3072, 1024, 256, 4, 192),
        (5120, 1024, 256, 4, 192),
        (6144, 64, 256, 4, 192),
        (6144, 128, 128, 8, 192),
        (16384, 2048, 128, 8, 192),
    ],
)
@pytest.mark.parametrize("sm", [100, 103, 107])
def test_pair_weighted_averaging_blackwell_d32_tile_follows_tokens_and_msa_rows(
    sm, tokens, msa_rows, tile_i, tile_s, num_threads, dtype_str
):
    """Blackwell D=32 takes fewer MSA rows per CTA below 384 tokens, the pair tile from 1024, the split tile past 5120.

    A small launch has few CTAs, so fewer MSA rows per CTA spread it over more SMs. From 1024 tokens a two-CTA pair
    (TILE_I=256) shares each MMA's v rows; past 5120 tokens and 72 MSA rows the 8-row split tile, which leaves Wo
    to a GEMM, does more work per joule while the power cap holds the clock. SM107 has 212 SMs, so its sweep moved
    three small-token anchors to a narrower tile (see ``_SM107_D32_SMALL_TOKEN_TILES``).
    """
    if sm == 107:
        tile_i, tile_s, num_threads = _SM107_D32_SMALL_TOKEN_TILES.get(
            (tokens, msa_rows), (tile_i, tile_s, num_threads)
        )
    params = select_pwa_config(sm, tokens, tokens, msa_rows, dtype_str, D=32, c_m=64)
    assert (params.TILE_I, params.TILE_S, params.num_threads) == (tile_i, tile_s, num_threads)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("tile_i,tile_s,stages", [(256, 4, 6), (128, 8, 4)], ids=["pair", "split"])
@pytest.mark.parametrize(
    "B,I,N,S", [(1, 33, 33, 11), (2, 300, 260, 9), (1, 2056, 2056, 5)], ids=["n33", "b2-i300-n260", "n2056"]
)
def test_pwa_sm100_pair_and_split_tiles_match_fp32(dtype, tile_i, tile_s, stages, B, I, N, S, monkeypatch) -> None:
    """The pair and split tiles cover ragged token and MSA-row tails, an odd token-tile count and grouped launches.

    The split tile writes the gated value and the op applies Wo; a launch over blocks of 8 MSA rows matches one.
    """
    skip_if_no_cutedsl("pair_weighted_averaging")
    if SM_VERSION not in (100, 103, 107):
        pytest.skip("tcgen05 tiles run on SM100, SM103 and SM107")
    _configure_pwa_mode("source", monkeypatch)
    torch.manual_seed(0)
    H, D, c_m = 8, 32, 64
    hd = H * D
    w = torch.softmax(torch.randn(B, H, I, N, device="cuda"), dim=-1)
    w = torch.nn.functional.pad(w, (0, (N + 7) // 8 * 8 - N)).to(dtype)
    v = (torch.randn(B, S, N, 2 * hd, device="cuda") * 0.5).to(dtype)[..., :hd]
    g = torch.randn(B, S, I, hd, dtype=dtype, device="cuda")
    weight = (torch.randn(c_m, hd, device="cuda") * hd**-0.5).to(dtype)
    params = pwa_config.PWAConfigParams(
        ab_dtype={torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype],
        H=H,
        D=D,
        c_m=c_m,
        TILE_I=tile_i,
        TILE_S=tile_s,
        TILE_J=64,
        num_threads=192,
        num_stages=stages,
        num_stages_w=2,
    )
    selection = pwa_config.PWAConfigSelection(params=params, kernel_abi="sm100")
    op = PairWeightedAveragingCuTe()
    out = op._launch_selection(w, v, g, weight, selection, (selection,)).float()
    ref = _fp32_reference(w, v, g, weight)
    rel_l2 = ((out - ref).norm() / ref.norm()).item()
    assert rel_l2 < 2e-2, f"{tile_i}x{tile_s} tile vs fp32 reference rel_l2={rel_l2:.3e} (I={I}, N={N}, S={S})"
    if tile_s == 8:
        monkeypatch.setattr(pwa_cutedsl, "_SPLIT_Z_BYTES", B * I * hd * w.element_size() * 8)
        blocks = op._launch_selection(w, v, g, weight, selection, (selection,)).float()
        torch.testing.assert_close(blocks, out, atol=1e-2, rtol=1e-2)


@pytest.fixture
def pwa_limit_file(tmp_path, monkeypatch):
    """Read token limits from an empty configs directory; yield the path of its SM90 (D=8, c_m=64) file."""
    monkeypatch.setattr(pwa_config, "_PWA_CONFIGS_DIR", str(tmp_path))
    pwa_config._max_fused_tokens_in_file.cache_clear()
    yield tmp_path / pwa_config.config_file_name(90, 8, 64)
    pwa_config._max_fused_tokens_in_file.cache_clear()


@pytest.mark.parametrize(
    "limit,bf16,fp16",
    [
        (1024, 1024, 1024),
        (4096, 4096, 2048),
        (None, None, 2048),
        ({"bf16": None, "fp16": None}, None, None),
        ({"bf16": 1024, "fp16": 4096}, 1024, 4096),
        ({"bf16": 4096}, 4096, 2048),
        ({"fp16": 1024}, 2048, 1024),
    ],
    ids=["bare-below-default", "bare-above-default", "bare-null", "both-null", "both-set", "bf16-only", "fp16-only"],
)
def test_pair_weighted_averaging_reads_token_limits_by_dtype(pwa_limit_file, limit, bf16, fp16) -> None:
    """A bare limit is bf16's. A dtype the file does not list keeps the default, or a lower bf16 limit."""
    pwa_limit_file.write_text(json.dumps({"max_fused_tokens": limit}))
    assert max_fused_tokens(8, 64, sm_version=90, dtype_str="bf16") == bf16
    assert max_fused_tokens(8, 64, sm_version=90, dtype_str="fp16") == fp16


def test_pair_weighted_averaging_without_a_file_limit_keeps_the_default(pwa_limit_file) -> None:
    """Without a file or a ``max_fused_tokens`` field, both dtypes take the default."""
    assert max_fused_tokens(8, 64, sm_version=90) == 2048
    pwa_limit_file.write_text(json.dumps({"kernel_abi": "sm90"}))
    pwa_config._max_fused_tokens_in_file.cache_clear()
    for dtype_str in ("bf16", "fp16"):
        assert max_fused_tokens(8, 64, sm_version=90, dtype_str=dtype_str) == 2048


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2048", [2048], {"bf16": 0}, {"fp16": "2048"}, {"fp32": 2048}])
def test_pair_weighted_averaging_rejects_malformed_token_limits(pwa_limit_file, limit) -> None:
    """A limit must be a positive integer or null, keyed by bf16 or fp16 when given per dtype."""
    pwa_limit_file.write_text(json.dumps({"max_fused_tokens": limit}))
    with pytest.raises(ValueError, match="max_fused_tokens"):
        max_fused_tokens(8, 64, sm_version=90)


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


def test_fused_pair_weighted_averaging_chunks_msa_rows(monkeypatch) -> None:
    """The fused projection is chunked along S, so a long MSA does not build one [B, S, N, 2*H*D] tensor."""
    layer = PairWeightedAveraging(c_m=64, c_z=128, c_h=32, num_heads=8, dtype=torch.bfloat16).cuda()
    if not isinstance(layer._pwa_op, PairWeightedAveragingCuTe):
        pytest.skip("fused PWA is not selected on this GPU")
    seen: list[int] = []

    def record(_w, v, _g, weight):
        seen.append(v.shape[1])
        return torch.zeros(v.shape[0], v.shape[1], v.shape[2], weight.shape[0], device=v.device, dtype=v.dtype)

    # __call__ is resolved on the class, so an instance assignment would not run.
    monkeypatch.setattr(PairWeightedAveragingCuTe, "__call__", lambda _self, w, v, g, weight: record(w, v, g, weight))
    layer.chunk_policy = ChunkPolicy(chunk_size=4, min_size=0, dim=1)
    n_seq, n_res = 8, 32
    m = torch.zeros(1, n_seq, n_res, 64, dtype=torch.bfloat16, device="cuda")
    z = torch.zeros(1, n_res, n_res, 128, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(1, n_res, n_res, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        out = layer(m, z, mask)
    assert out.shape == (1, n_seq, n_res, 64)
    assert seen == [4, 4]


@pytest.mark.parametrize("sm", [100, 103, 107, 120, 121])
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
def test_pwa_cubin_and_python_select_same_rectangular_bucket(D, c_m, I, J, S, monkeypatch):
    skip_if_no_cutedsl("pair_weighted_averaging")
    library = require_cubin_library()
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
    try:
        config = launcher.make_kernel_config(
            SM_VERSION,
            I,
            J,
            S,
            D,
            c_m,
            launcher.DType.BFLOAT16,
        )
    except (RuntimeError, TypeError, ValueError):
        config = None

    if config is not None:
        packaged = (config.tile_i, config.tile_s, config.tile_j, config.num_threads)
        if packaged == pwa_cutedsl._expected_packaged_tile(selected):
            # An image of the selected kernel was built from these tables, so both must pick the same anchor. An
            # image of another kernel predates a kernel change and is handled below.
            assert config.n_anchor == selected.n_anchor
            assert config.s_anchor == selected.s_anchor
            if pwa_cutedsl._same_packaged_tile(config, selected):
                return

    # The tracked corpus is refreshed on protected main, not in this MR. Until that image matches the
    # tuned tile, cubin mode must run PyTorch rather than the stale kernel.
    _configure_pwa_mode("cubin", monkeypatch)
    op = PairWeightedAveragingCuTe()
    hidden = 8 * D
    padded = (J + 7) // 8 * 8
    torch.manual_seed(0)
    w = torch.randn(1, 8, I, padded, dtype=torch.bfloat16, device="cuda")
    w[..., J:] = 0
    v = torch.randn(1, S, J, hidden, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(1, S, I, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(c_m, hidden, dtype=torch.bfloat16, device="cuda")
    out = op(w, v, g, weight)
    reference = pwa_ops._invoke_vanilla_pwa(w, v, g, weight)
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


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
        selected, _bucket = _select_pwa_config_selection_bucket(op._sm_version, I, N, S, "bf16")
        if cutedsl_mode == "cubin":
            _skip_unless_packaged(selected, I, N, S, torch.bfloat16)
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

        key = pwa_cutedsl._compile_cache_key(op._sm_version, selected.params, selected.kernel_abi, torch.bfloat16)
        executable_ids.append(id(op._compiled_cache[key]))

    assert len(set(executable_ids)) == 1


@_BITWISE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("D,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
@pytest.mark.parametrize(
    "S,I,N,Jp",
    [
        (16, 128, 128, 128),
        (17, 127, 127, 128),
        (31, 129, 129, 136),
        (7, 33, 33, 64),
        (9, 384, 384, 384),
        (9, 513, 513, 520),
        (5, 2600, 2600, 2600),
        # On SM100 these reach the D=32 pair tile, the D=32 split tile (the last two) and D=8 tiles of every kind
        # (g staged in shared memory, w multicast across MSA-row tiles, CTA pairs, v multicast across token tiles);
        # test_pwa_sm100_d8_cubin_matches_source_at_every_tile covers each D=8 tile and cluster.
        (100, 300, 300, 304),
        (130, 400, 400, 400),
        (20, 1100, 1100, 1104),
        (128, 700, 700, 704),
        (400, 1000, 1000, 1000),
        (900, 768, 768, 768),
        (520, 1500, 1500, 1504),
        (520, 260, 260, 264),
        (600, 3500, 3500, 3504),
        (80, 5200, 5200, 5200),
        (77, 5201, 5203, 5208),
    ],
    ids=[
        "aligned",
        "mixed-tails",
        "all-tails",
        "tile-j-padding",
        "n384-bucket",
        "n512-bucket-tails",
        "n3072-bucket",
        "n256-s128",
        "n384-s128",
        "n1024-s16",
        "n768-s128",
        "n1024-s512",
        "n768-s1024",
        "n1536-s512",
        "n256-s512",
        "n4096-s1024",
        "n6144-s128",
        "n6144-s128-tails",
    ],
)
def test_pwa_source_and_cubin_agree_bitwise(dtype, D, c_m, S, I, N, Jp, monkeypatch):
    """The direct launcher must reproduce the source launch exactly."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    require_cubin_library()
    selected, _bucket = _select_pwa_config_selection_bucket(
        SM_VERSION, I, N, S, pwa_cutedsl._dtype_str(dtype), D=D, c_m=c_m
    )
    _skip_unless_packaged(selected, I, N, S, dtype)
    torch.manual_seed(123)
    B, H = 1, 8
    w = torch.randn(B, H, I, Jp, dtype=dtype, device="cuda")
    w[..., N:] = 0
    vg = torch.randn(B, S, max(I, N), 2 * H * D, dtype=dtype, device="cuda")
    v = vg[:, :, :N, : H * D]
    g = vg[:, :, :I, H * D :]
    weight = torch.randn(c_m, H * D, dtype=dtype, device="cuda")
    source, cubin = _source_and_cubin(monkeypatch, selected, w, v, g, weight)
    torch.testing.assert_close(cubin, source, atol=0.0, rtol=0.0)


def _sm100_d8_tile_cases() -> list:
    """One ragged shape per tile and cluster the SM100 D=8 configs select, at the first anchor that selects it, and
    shapes of 16384 tokens, past the last anchor."""
    cases = []
    for c_m in (64, 128):
        configs = _config_entries(100, 8, c_m)
        seen = set()
        for key in sorted(configs, key=pwa_config._parse_key):
            n, s, dtype_str = pwa_config._parse_key(key)
            p = configs[key]["params"]
            tile = (p["TILE_I"], p["TILE_S"], p["num_stages"], p["num_stages_w"], *p["cluster"])
            if (dtype_str, tile) in seen:
                continue
            seen.add((dtype_str, tile))
            # One token past the anchor and a few MSA rows short of it: tails in every dimension.
            S = s - 3 if s > 4 else s
            cases.append(pytest.param(dtype_str, c_m, S, n + 1, tile, id=f"{dtype_str}-cm{c_m}-n{n + 1}-s{S}"))
        for dtype_str in ("bf16", "fp16"):
            for S in (16, 129):
                cases.append(pytest.param(dtype_str, c_m, S, 16384, None, id=f"{dtype_str}-cm{c_m}-n16384-s{S}"))
    return cases


@_BITWISE
@pytest.mark.parametrize("dtype_str,c_m,S,N,tile", _sm100_d8_tile_cases())
def test_pwa_sm100_d8_cubin_matches_source_at_every_tile(dtype_str, c_m, S, N, tile, monkeypatch) -> None:
    """Each D=8 tile and cluster, and 16384 tokens: the CUBIN launch reproduces the source launch bit for bit, from
    natural v (which the op lays out as v^T) and from a caller's contiguous v^T (whose j pitch the op pads to a
    multiple of 8 elements)."""
    if SM_VERSION != 100:
        pytest.skip("only SM100 runs the tcgen05 D=8 kernel")
    skip_if_no_cutedsl("pair_weighted_averaging")
    require_cubin_library()
    H, D = 8, 8
    hd = H * D
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    selected, _bucket = _select_pwa_config_selection_bucket(SM_VERSION, N, N, S, dtype_str, D=D, c_m=c_m)
    params = selected.params
    if tile is not None:
        assert (params.TILE_I, params.TILE_S, params.num_stages, params.num_stages_w, *params.cluster) == tile
    _skip_unless_packaged(selected, N, N, S, dtype)
    torch.manual_seed(7)
    Jp = (N + 7) // 8 * 8
    w = torch.randn(1, H, N, Jp, dtype=dtype, device="cuda")
    w[..., N:] = 0
    vg = torch.randn(1, S, N, 2 * hd, dtype=dtype, device="cuda")
    v, g = vg[..., :hd], vg[..., hd:]
    weight = torch.randn(c_m, hd, dtype=dtype, device="cuda") * hd**-0.5
    source, cubin = _source_and_cubin(monkeypatch, selected, w, v, g, weight)
    torch.testing.assert_close(cubin, source, atol=0.0, rtol=0.0)
    _, cubin_vt = _source_and_cubin(
        monkeypatch, selected, w, v.transpose(-1, -2).contiguous().transpose(-1, -2), g, weight
    )
    torch.testing.assert_close(cubin_vt, source, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("sm", [100, 103, 107])
@pytest.mark.parametrize("D,c_m", _PWA_DIMS, ids=_PWA_DIM_IDS)
def test_pwa_blackwell_images_match_python_at_every_anchor(sm: int, D: int, c_m: int) -> None:
    """At every anchor of a Blackwell config file the launcher accepts the image it picks, and an image of the selected
    tile carries the anchor and cluster Python selects.

    Picking an image checks its launch record for its target, so this covers the SM103 images on any GPU; launching
    them needs SM103.
    """
    skip_if_no_cutedsl("pair_weighted_averaging")
    if sm == 107 and not _sm107_cubin_published():
        pytest.skip("the SM107 CUBIN pack is not shipped yet; the protected refresh publishes it")
    launcher = require_cubin_library().pair_weighted_averaging
    checked = 0
    for key in _config_entries(sm, D, c_m):
        n, s, dtype_str = pwa_config._parse_key(key)
        selected, _bucket = _select_pwa_config_selection_bucket(sm, n, n, s, dtype_str, D=D, c_m=c_m)
        dtype = launcher.DType.BFLOAT16 if dtype_str == "bf16" else launcher.DType.FLOAT16
        # Once the pack is published, a missing anchor image is a failure, not a skip.
        config = launcher.make_kernel_config(sm, n, n, s, D, c_m, dtype)
        if (config.tile_i, config.tile_s, config.tile_j, config.num_threads) != pwa_cutedsl._expected_packaged_tile(
            selected
        ):
            # An image of another tile is stale; the protected refresh republishes it.
            continue
        checked += 1
        assert (config.n_anchor, config.s_anchor) == (selected.n_anchor, selected.s_anchor), key
        assert (config.cluster_i, config.cluster_s) == tuple(selected.params.cluster), key
    if not checked:
        pytest.skip("the packaged images predate these tiles; protected refresh republishes them")


@_BITWISE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,S", [(1, 85), (2, 77)], ids=["b1-s85", "b2-s77"])
def test_pwa_split_tile_cubin_matches_source_over_row_blocks(dtype, B, S, monkeypatch) -> None:
    """The split tile's launches over blocks of MSA rows, a short last block included, reproduce the source launch."""
    skip_if_no_cutedsl("pair_weighted_averaging")
    require_cubin_library()
    H, D, c_m, I, N = 8, 32, 64, 5201, 5203
    hd = H * D
    selected, _bucket = _select_pwa_config_selection_bucket(
        SM_VERSION, I, N, S, pwa_cutedsl._dtype_str(dtype), D=D, c_m=c_m
    )
    if not selected.writes_gated_value:
        pytest.skip(f"SM{SM_VERSION} does not select the split tile")
    _skip_unless_packaged(selected, I, N, S, dtype)
    # Blocks of 24 MSA rows: three full blocks and a short last one.
    monkeypatch.setattr(pwa_cutedsl, "_SPLIT_Z_BYTES", B * I * hd * torch.finfo(dtype).bits // 8 * 24)
    torch.manual_seed(5)
    Jp = (N + 7) // 8 * 8
    w = torch.randn(B, H, I, Jp, dtype=dtype, device="cuda")
    w[..., N:] = 0
    vg = torch.randn(B, S, max(I, N), 2 * hd, dtype=dtype, device="cuda")
    v, g = vg[:, :, :N, :hd], vg[:, :, :I, hd:]
    weight = torch.randn(c_m, hd, dtype=dtype, device="cuda") * hd**-0.5
    source, cubin = _source_and_cubin(monkeypatch, selected, w, v, g, weight)
    torch.testing.assert_close(cubin, source, atol=0.0, rtol=0.0)


def test_pwa_only_the_d32_split_tile_writes_the_gated_value() -> None:
    """An 8-row D=8 tile projects its own output; only the D=32 split tile leaves Wo to the op."""
    d8 = _select_pwa_config_selection_bucket(100, 4096, 4096, 1024, "bf16", D=8, c_m=64)[1]
    assert any(selection.params.TILE_S == 8 for selection in d8)
    assert not any(selection.writes_gated_value for selection in d8)
    split, _bucket = _select_pwa_config_selection_bucket(100, 8192, 8192, 512, "bf16", D=32, c_m=64)
    assert split.params.TILE_S == 8 and split.writes_gated_value


def test_compile_cache_key_separates_kernel_abi() -> None:
    """The Hopper, tcgen05, and Ampere executables of one tile must not share a cache entry, nor two variants of
    one ABI."""
    params = select_pwa_config(sm_version=100, I=256, J=256, S=512, dtype_str="bf16")
    sm80 = pwa_cutedsl._compile_cache_key(100, params, "sm80", torch.bfloat16)
    sm90 = pwa_cutedsl._compile_cache_key(100, params, "sm90", torch.bfloat16)
    sm90_d8 = pwa_cutedsl._compile_cache_key(100, params, "sm90", torch.bfloat16, "d8")
    sm100 = pwa_cutedsl._compile_cache_key(100, params, "sm100", torch.bfloat16)
    assert len({sm80, sm90, sm90_d8, sm100}) == 4
    assert sm80[2] == "sm80" and sm90[2] == "sm90" and sm100[2] == "sm100"
    assert sm90[3] is None and sm90_d8[3] == "d8"


def test_tma_value_rejects_a_token_stride_tma_cannot_encode() -> None:
    """A unit last stride is not enough: TMA also needs 16-byte outer strides and a 16-byte base."""
    storage = torch.empty(1, 2, 3, 257, dtype=torch.bfloat16)
    sliced = storage[..., :256]
    assert sliced.stride(-1) == 1
    assert sliced.stride(2) * sliced.element_size() % 16 != 0
    assert not pwa_cutedsl._tma_value_ready(sliced)

    packed = torch.empty(1, 2, 4, 256, dtype=torch.bfloat16)
    if packed.data_ptr() % 16 == 0:
        assert pwa_cutedsl._tma_value_ready(packed)
    wide = torch.empty(1, 2, 4, 512, dtype=torch.bfloat16)[..., :256]
    if wide.data_ptr() % 16 == 0:
        assert wide.stride(-1) == 1
        assert pwa_cutedsl._tma_value_ready(wide)


def test_tma_value_copy_pads_the_pitch() -> None:
    """A contiguous v^T with N % 8 != 0 is copied into a buffer whose j pitch is a multiple of 8 elements."""
    vt = torch.randn(1, 2, 64, 33).to(torch.bfloat16)
    assert vt.is_contiguous() and not pwa_cutedsl._tma_value_ready(vt)
    copied = pwa_cutedsl._kernel_operand(vt, tma=True)
    assert torch.equal(copied, vt) and copied.stride(-2) == 40
    if copied.data_ptr() % 16 == 0:
        assert pwa_cutedsl._tma_value_ready(copied)
