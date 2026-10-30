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

from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.attention_backend.triangle_attention import claude_kit as claude_kit_module
from bionemo_ir._torch.attention_backend.triangle_attention import heuristic as heuristic_module
from bionemo_ir._torch.attention_backend.triangle_attention.claude_kit import (
    ClaudeKitTriangleAttentionMetadata,
    ClaudeKitTriangleAttentionSM90D32,
    ClaudeKitTriangleAttentionUnavailable,
)
from bionemo_ir._torch.attention_backend.triangle_attention.cuequiv import (
    CuEquivAttention,
    _invoke_triangle_attention_kernel,
    cuequiv_supports,
)
from bionemo_ir._torch.attention_backend.triangle_attention.heuristic import HeuristicTriangleAttention
from tests._torch import SM_VERSION


class _RecordingDelegate:
    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.calls: list[dict] = []

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return q


def _install_fake_delegates(
    backend: HeuristicTriangleAttention,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, _RecordingDelegate]:
    delegates: dict[str, _RecordingDelegate] = {}

    def create(backend_name: str) -> _RecordingDelegate:
        delegate = _RecordingDelegate(backend_name)
        delegates[backend_name] = delegate
        return delegate

    monkeypatch.setattr(backend, "_create_delegate", create)
    monkeypatch.setattr(heuristic_module, "_native_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ClaudeKitTriangleAttentionSM90D32,
        "native_available",
        staticmethod(lambda: True),
    )
    backend._sm_version = 90
    return delegates


def test_claude_kit_extension_import_is_lazy(monkeypatch: pytest.MonkeyPatch):
    def reject_import(_name: str):
        raise AssertionError("constructing the backend imported the native extension")

    monkeypatch.setattr(claude_kit_module.importlib, "import_module", reject_import)
    backend = ClaudeKitTriangleAttentionSM90D32(0, 4, 32, 4)

    assert backend._last_executable is None


def test_claude_kit_executable_matches_native_launcher_contract(monkeypatch: pytest.MonkeyPatch):
    tensor_view = lambda *args: args
    library = SimpleNamespace(Tensor1View=tensor_view, Tensor3View=tensor_view, Tensor4View=tensor_view)
    launched: list[object] = []

    class LaunchParams:
        pass

    launcher = SimpleNamespace(LaunchParams=LaunchParams, launch=launched.append)
    executable = claude_kit_module._ClaudeKitExecutable(library, launcher)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace(cuda_stream=1234))
    q = torch.empty(2, 8, 4, 32, dtype=torch.bfloat16)
    bias = torch.empty(1, 4, 8, 8, dtype=torch.bfloat16)
    lengths = torch.full((2,), 8, dtype=torch.int32)
    output = torch.empty_like(q)
    lse = torch.empty(2, 8, 4, 1, dtype=torch.float32)

    executable(q, q, q, bias, lengths, output, lse, 0.0, 32**-0.5, 2)

    assert len(launched) == 1
    params = launched[0]
    assert params.softmax_scale == 32**-0.5
    assert params.i_dim == 2
    assert params.stream == 1234
    assert params.q[1] == tuple(q.shape)
    assert params.actual_s_kv[1] == tuple(lengths.shape)
    assert params.output[1] == tuple(output.shape)
    assert params.lse[1] == tuple(lse.shape[:3])


@pytest.mark.parametrize(
    "num_heads,batch,i_dim,tokens,dtype,expected",
    [
        # ClaudeKit from 23M attention scores, batch * i_dim * tokens**2 * heads.
        (4, 1, 176, 176, torch.bfloat16, "CuTeDSL"),
        (4, 1, 184, 184, torch.bfloat16, "ClaudeKit"),
        (4, 1, 512, 512, torch.bfloat16, "ClaudeKit"),
        (8, 1, 136, 136, torch.bfloat16, "CuTeDSL"),
        (8, 1, 144, 144, torch.bfloat16, "ClaudeKit"),
        (4, 1, 152, 152, torch.bfloat16, "CuTeDSL"),
        (4, 2, 152, 152, torch.bfloat16, "ClaudeKit"),
        # MSA row attention: i_dim sequences of tokens residues.
        (8, 1, 512, 64, torch.bfloat16, "CuTeDSL"),
        (8, 1, 1024, 64, torch.bfloat16, "ClaudeKit"),
        (4, 1, 512, 512, torch.float16, "CuTeDSL"),
    ],
)
def test_heuristic_routes_at_measured_crossover(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    batch: int,
    i_dim: int,
    tokens: int,
    dtype: torch.dtype,
    expected: str,
):
    backend = HeuristicTriangleAttention(0, num_heads, 32, num_heads)
    delegates = _install_fake_delegates(backend, monkeypatch)
    q = torch.empty((batch, i_dim, tokens, num_heads * 32), dtype=dtype, device="meta")

    result = backend.forward(q, q, q, biases=None)

    assert result is q
    assert backend.last_backend_name == expected
    assert len(delegates[expected].calls) == 1


def test_heuristic_falls_back_when_claude_kit_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    delegates = _install_fake_delegates(backend, monkeypatch)
    monkeypatch.setattr(
        ClaudeKitTriangleAttentionSM90D32,
        "native_available",
        staticmethod(lambda: False),
    )
    q = torch.empty((1, 256, 256, 128), dtype=torch.bfloat16, device="meta")

    assert backend.forward(q, q, q) is q
    assert backend.last_backend_name == "CuTeDSL"
    assert len(delegates["CuTeDSL"].calls) == 1
    assert backend.route_provenance[0]["backend"] == "CuTeDSL"


@pytest.mark.parametrize("native_policy", [None, "ClaudeKit"])
def test_missing_claude_kit_logs_once_across_layers_and_shapes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, native_policy: str | None
) -> None:
    monkeypatch.setattr(heuristic_module, "_warned_missing_claude_kit", False)
    with caplog.at_level("WARNING", logger=heuristic_module.__name__):
        for layer in range(3):
            backend = HeuristicTriangleAttention(layer, 4, 32, 4)
            _install_fake_delegates(backend, monkeypatch)
            monkeypatch.setattr(heuristic_module, "_native_policy", lambda *_args: native_policy)
            monkeypatch.setattr(ClaudeKitTriangleAttentionSM90D32, "native_available", staticmethod(lambda: False))
            for tokens in (256, 373, 959, 1734):
                q = torch.empty((1, tokens, tokens, 128), dtype=torch.bfloat16, device="meta")
                backend.forward(q, q, q)
                assert backend.last_backend_name == "CuTeDSL"
    assert len(caplog.records) == 1
    assert "SM80" in caplog.text
    assert "BIOIR_BUILD_CUTEDSL_KERNELS=1" in caplog.text


@pytest.mark.parametrize(("available", "tokens"), [(True, 373), (False, 29)])
def test_native_and_below_crossover_routes_do_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, available: bool, tokens: int
) -> None:
    monkeypatch.setattr(heuristic_module, "_warned_missing_claude_kit", False)
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    _install_fake_delegates(backend, monkeypatch)
    monkeypatch.setattr(ClaudeKitTriangleAttentionSM90D32, "native_available", staticmethod(lambda: available))
    q = torch.empty((1, tokens, tokens, 128), dtype=torch.bfloat16, device="meta")
    backend.forward(q, q, q)
    assert not caplog.records
    assert not heuristic_module._warned_missing_claude_kit


def test_route_provenance_records_all_selected_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    _install_fake_delegates(backend, monkeypatch)
    assert backend.route_provenance == []
    for tokens in (29, 373, 373):
        q = torch.empty((1, tokens, tokens, 128), dtype=torch.bfloat16, device="meta")
        backend.forward(q, q, q)
    routes = json.loads(json.dumps(backend.route_provenance))
    assert [(r["tokens"], r["backend"]) for r in routes] == [(29, "CuTeDSL"), (373, "ClaudeKit")]
    assert routes[1] == {
        "backend": "ClaudeKit",
        "sm_version": 90,
        "device_index": -1,
        "dtype": "torch.bfloat16",
        "num_heads": 4,
        "head_dim": 32,
        "batch": 1,
        "tokens": 373,
        "i_dim": 373,
        "has_lse": False,
    }
    routes[0]["backend"] = "changed"
    assert backend.route_provenance[0]["backend"] == "CuTeDSL"


def test_heuristic_delegate_receives_buffers_unchanged(monkeypatch: pytest.MonkeyPatch):
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    delegates = _install_fake_delegates(backend, monkeypatch)
    q = torch.empty((1, 128, 512, 128), dtype=torch.bfloat16, device="meta")
    lengths = torch.full((1, 128), 128, dtype=torch.int32)
    pair_bias = torch.empty((1, 4, 512, 512), dtype=torch.bfloat16, device="meta")
    output = torch.empty((128, 512, 4, 32), dtype=torch.bfloat16, device="meta")
    lse = torch.empty((128, 512, 4, 1), dtype=torch.float32, device="meta")
    metadata = heuristic_module.HeuristicTriangleAttentionMetadata()

    result = backend.forward(
        q,
        q,
        q,
        biases=[lengths, pair_bias],
        metadata=metadata,
        output=output,
        output_lse=lse,
    )

    call = delegates["ClaudeKit"].calls[0]
    assert result is q
    assert call["biases"][0] is lengths
    assert call["biases"][1] is pair_bias
    assert call["metadata"] is metadata
    assert call["output"] is output
    assert call["output_lse"] is lse


def test_heuristic_normalizes_legacy_additive_mask():
    additive_mask = torch.tensor([[[[[0.0, 0.0, -1e9, -1e9]]]]])
    pair_bias = torch.zeros(1, 2, 4, 4)

    normalized = heuristic_module._normalize_left_mask_biases([additive_mask, pair_bias])

    assert normalized is not None
    assert normalized[0].shape == (1, 1)
    assert normalized[0].dtype == torch.int32
    torch.testing.assert_close(normalized[0], torch.tensor([[2]], dtype=torch.int32))
    assert normalized[1] is pair_bias


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize(
    ("prefix", "interior"),
    [
        (torch.tensor([[[[[0.0, 0.0, -1e9, -1e9]]]]]), torch.tensor([[[[[0.0, -1e9, 0.0, -1e9]]]]])),
        (torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]), torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])),
    ],
    ids=["additive", "binary"],
)
def test_heuristic_mask_prefix_check_is_debug_only(
    monkeypatch: pytest.MonkeyPatch, prefix: torch.Tensor, interior: torch.Tensor, debug: bool
):
    """The left-aligned check reads the mask back to the host, so production skips it."""
    monkeypatch.setattr(heuristic_module, "DEBUG_ASSERTS", debug)
    (lengths,) = heuristic_module._normalize_left_mask_biases([prefix])
    assert lengths.flatten().tolist() == [2]
    if debug:
        with pytest.raises(ValueError, match="left-aligned"):
            heuristic_module._normalize_left_mask_biases([interior])
    else:
        heuristic_module._normalize_left_mask_biases([interior])


def test_heuristic_refuses_uncached_route_during_capture(monkeypatch: pytest.MonkeyPatch):
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    _install_fake_delegates(backend, monkeypatch)
    q = torch.empty((1, 1, 512, 128), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="not cached before CUDA capture"):
        backend.forward(q, q, q)


def test_heuristic_reuses_cached_route_during_capture(monkeypatch: pytest.MonkeyPatch):
    backend = HeuristicTriangleAttention(0, 4, 32, 4)
    delegates = _install_fake_delegates(backend, monkeypatch)
    q = torch.empty((1, 32, 512, 128), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    backend.forward(q, q, q)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    backend.forward(q, q, q)

    assert len(delegates["ClaudeKit"].calls) == 2
    assert len(backend._route_cache) == 1


def _skip_without_cuequiv() -> None:
    if not cuequiv_supports(torch.bfloat16):
        pytest.skip("cuEquivariance is not importable here")


@pytest.mark.parametrize("cuequiv", [True, False], ids=["cuequiv", "no_cuequiv"])
@pytest.mark.parametrize("sm_version", [100, 103])
@pytest.mark.parametrize(
    ("head_dim", "odd_keys", "keys_of_8"),
    [
        (16, "CuTeDSL", "CuTeDSL"),
        (32, "CuTeDSL", "CuTeDSL"),
        (48, "CuTeDSL", "CuTeDSL"),
        (64, "CuTeDSL", "CuTeDSL"),
        (112, "CuTeDSL", "CuTeDSL"),
        (128, "CuTeDSL", "CuTeDSL"),
        (96, "CUEQUIV", "CUEQUIV"),
        (160, "SDPA", "CUEQUIV"),
        (256, "SDPA", "CUEQUIV"),
        (288, "SDPA", "SDPA"),
    ],
)
def test_heuristic_routes_blackwell_head_dims(
    monkeypatch: pytest.MonkeyPatch, cuequiv: bool, sm_version: int, head_dim: int, odd_keys: str, keys_of_8: str
):
    """Head dims padding to 32, 64 or 128 keep the CuTeDSL kernel on SM100/103. Others, which pad to 96 or
    past 128, take cuEquivariance where its kernels serve them (dense mask up to 128, per-row lengths up to 256
    when the key count is a multiple of 8) and SDPA otherwise, instead of raising on the first call."""
    if cuequiv:
        _skip_without_cuequiv()
    else:
        monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", None)
    backend = HeuristicTriangleAttention(0, 4, head_dim, 4)
    backend._sm_version = sm_version
    for dtype in (torch.float16, torch.bfloat16):
        for batch, tokens, expected in ((1, 37, odd_keys), (5, 37, odd_keys), (1, 384, keys_of_8), (5, 960, keys_of_8)):
            if not cuequiv and expected == "CUEQUIV":
                expected = "SDPA"
            key = heuristic_module._RouteKey(0, dtype, batch, tokens, tokens, False)
            assert backend._select_backend_name(key) == expected


@pytest.mark.parametrize("cuequiv", [True, False], ids=["cuequiv", "no_cuequiv"])
def test_heuristic_fallback_warns_once_per_layer_and_route(monkeypatch: pytest.MonkeyPatch, cuequiv: bool):
    """A layer that leaves ClaudeKit/CuTeDSL raises one ``RuntimeWarning`` per route it takes, however many
    shapes take it; a layer the kernel serves raises none."""
    if cuequiv:
        _skip_without_cuequiv()
    else:
        monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", None)
    keys = [heuristic_module._RouteKey(0, torch.bfloat16, 1, tokens, tokens, False) for tokens in (37, 41, 384, 960)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for head_dim in (256, 256, 128):  # Two layers off the kernel, one on it.
            backend = HeuristicTriangleAttention(0, 4, head_dim, 4)
            backend._sm_version = 100
            for key in keys + keys:
                backend._select_backend_name(key)
    routes = [str(w.message).rsplit(" ", 1)[-1] for w in caught if issubclass(w.category, RuntimeWarning)]
    assert sorted(routes) == sorted((["SDPA", "CUEQUIV"] if cuequiv else ["SDPA"]) * 2)


def test_heuristic_forward_keeps_its_own_route_when_calls_interleave(monkeypatch: pytest.MonkeyPatch) -> None:
    """A concurrent call that routes elsewhere must not change the route of a call already under way: a
    D=256 call with 40 keys stays on cuEquivariance's per-row-length path while a 37-key call, routed to SDPA,
    runs during its mask conversion."""
    backend = HeuristicTriangleAttention(0, 4, 256, 4)
    delegates = _install_fake_delegates(backend, monkeypatch)
    backend._sm_version = 100
    monkeypatch.setattr(
        heuristic_module,
        "cuequiv_supports",
        lambda dtype, head_dim, *, sm_version, keys: keys % 8 == 0,
    )

    def call_inputs(tokens: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        q = torch.zeros(1, 2, tokens, 4 * 256)
        lengths = torch.full((1, 2), tokens, dtype=torch.int32)
        return q, [lengths, torch.zeros(1, 4, tokens, (tokens + 7) // 8 * 8)]

    (q40, biases40), (q37, biases37) = call_inputs(40), call_inputs(37)
    convert = heuristic_module._dense_mask_biases
    interleaved: list[str | None] = []

    def convert_then_interleave(*args: object, **kwargs: object) -> list[torch.Tensor]:
        converted = convert(*args, **kwargs)
        if not interleaved:
            interleaved.append(None)
            backend.forward(q37, q37, q37, biases=biases37)
            interleaved[0] = backend.last_backend_name
        return converted

    monkeypatch.setattr(heuristic_module, "_dense_mask_biases", convert_then_interleave)
    backend.forward(q40, q40, q40, biases=biases40)

    assert interleaved == ["SDPA"]
    assert len(delegates["SDPA"].calls) == 1
    assert [call.get("use_kv_lengths") for call in delegates["CUEQUIV"].calls] == [True]


def _dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    pair_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """FP32 triangle attention over ``[B, I, J, H*D]`` with per-row key counts; returns ``[B, I, J, H, D]``."""
    batch, i_dim, tokens, _ = q.shape

    def heads(operand: torch.Tensor) -> torch.Tensor:
        return operand.float().view(batch, i_dim, tokens, num_heads, head_dim).transpose(2, 3)

    scores = heads(q) @ heads(k).transpose(-1, -2) * head_dim**-0.5 + pair_bias[..., :tokens].float()[:, None]
    masked = torch.arange(tokens, device=q.device) >= lengths[..., None]
    scores = scores.masked_fill(masked[:, :, None, None, :], float("-inf"))
    return (scores.softmax(dim=-1) @ heads(v)).transpose(2, 3)


@pytest.mark.skipif(SM_VERSION not in (80, 86, 89, 90, 100, 103), reason="Heuristic routes SM80-SM103 only")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("chunks", ["rows", "query_tiles"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    ("head_dim", "cuequiv", "tokens", "fallback"),
    [
        (96, True, 37, "CUEQUIV"),
        (96, True, 40, "CUEQUIV"),
        (96, False, 37, "SDPA"),
        (96, False, 40, "SDPA"),
        (256, True, 37, "SDPA"),
        (256, True, 40, "CUEQUIV"),
        (288, True, 37, "SDPA"),
        (288, True, 40, "SDPA"),
    ],
    ids=[
        "d96-cuequiv-j37",
        "d96-cuequiv-j40",
        "d96-sdpa-j37",
        "d96-sdpa-j40",
        "d256-sdpa-j37",
        "d256-cuequiv-j40",
        "d288-sdpa-j37",
        "d288-sdpa-j40",
    ],
)
def test_heuristic_runs_uncovered_head_dims_on_the_dense_backends(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    cuequiv: bool,
    tokens: int,
    fallback: str,
    dtype: torch.dtype,
    chunks: str,
    graph: bool,
):
    """Model callers select Heuristic without a head dim, so a layer whose head dim no kernel serves (D=96 and
    D=288 anywhere, D=256 on SM100/103) used to raise on its first call. It now runs on cuEquivariance or on
    row-chunked SDPA from the left-mask inputs a Heuristic stack prepares: per-row lengths, a pair bias padded
    along the key axis and an output buffer, eagerly and under graph capture. The caller's ``use_kv_lengths``
    does not matter: Heuristic masks are left-aligned."""
    if head_dim == 256 and SM_VERSION not in (100, 103):
        pytest.skip(f"the SM{SM_VERSION} CuTeDSL kernel serves head_dim={head_dim}")
    if cuequiv:
        _skip_without_cuequiv()
    else:
        monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", None)
    torch.manual_seed(0)
    batch, i_dim, num_heads = 2, 12, 2
    # SDPA chunks of five rows, or, when one row exceeds the cap, of seven query positions of one row.
    rows_or_queries = 5 * tokens if chunks == "rows" else 7
    mask_cap = rows_or_queries * num_heads * tokens * dtype.itemsize
    monkeypatch.setattr(heuristic_module, "_SDPA_MASK_BYTES", mask_cap)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    mask_bytes: list[int] = []

    def recording_sdpa(*args: torch.Tensor, attn_mask: torch.Tensor, **kwargs: object) -> torch.Tensor:
        mask_bytes.append(attn_mask.numel() * attn_mask.element_size())
        return sdpa(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    width = num_heads * head_dim
    backend = HeuristicTriangleAttention(0, num_heads, head_dim, num_heads)
    # Slices of one projection, as TriangleAttention passes them.
    qkv = torch.randn(batch, i_dim, tokens, 3 * width, device="cuda", dtype=dtype)
    q, k, v = qkv.split(width, dim=-1)
    lengths = torch.randint(1, tokens + 1, (batch, i_dim), dtype=torch.int32, device="cuda")
    lengths[-1, -1] = 0  # A padding row: no key to attend to.
    padded = (tokens + 7) // 8 * 8
    pair_bias = torch.zeros(batch, num_heads, tokens, padded, device="cuda", dtype=dtype)
    pair_bias[..., :tokens].normal_()
    output = torch.empty(batch * i_dim, tokens, num_heads, head_dim, device="cuda", dtype=dtype)

    def call() -> torch.Tensor:
        return backend.forward(q, k, v, biases=[lengths, pair_bias], output=output, use_kv_lengths=False)

    result = call()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            result = call()
        output.zero_()
        captured.replay()
        torch.cuda.synchronize()

    assert backend.last_backend_name == fallback
    assert result.data_ptr() == output.data_ptr()
    assert torch.isfinite(result).all()
    if fallback == "SDPA":
        assert mask_bytes and max(mask_bytes) <= mask_cap
    expected = _dense_reference(q, k, v, lengths, pair_bias, num_heads, head_dim)
    keyed = lengths > 0  # A row with no key has no defined output.
    torch.testing.assert_close(result.float()[keyed], expected[keyed], atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(SM_VERSION not in (100, 103), reason="cuEquivariance's per-row-length kernel is SM100/SM103 only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    ("head_dim", "keys", "supported"),
    [(256, 40, True), (264, 40, False), (256, 37, False)],
    ids=["d256", "d264", "d256-odd-keys"],
)
def test_cuequiv_supports_matches_its_per_row_length_kernel(
    dtype: torch.dtype, head_dim: int, keys: int, supported: bool
) -> None:
    """Calls cuEquivariance's op directly with per-row lengths, past the wrapper's own eligibility check, so a
    release that widens or narrows the per-row-length kernel (D up to 256, key count a multiple of 8) fails here."""
    _skip_without_cuequiv()
    assert cuequiv_supports(dtype, head_dim, sm_version=SM_VERSION, keys=keys) == supported
    num_heads = 2
    q = torch.randn(1, 4, num_heads, keys, head_dim, device="cuda", dtype=dtype)
    bias = torch.randn(1, 1, num_heads, keys, keys, device="cuda", dtype=dtype)
    lengths = torch.full((1, 4), keys - 3, dtype=torch.int32, device="cuda")

    def attend() -> torch.Tensor:
        return _invoke_triangle_attention_kernel(q, q, q, bias, None, head_dim**-0.5, kv_lengths=lengths)

    if supported:
        assert torch.isfinite(attend()).all()
    else:
        with pytest.raises(RuntimeError, match="hidden_dim"):
            attend()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuEquivariance kernels need a GPU")
@pytest.mark.parametrize(
    ("dtype", "head_dim", "supported"),
    [
        (torch.bfloat16, 128, True),
        (torch.bfloat16, 136, False),
        (torch.float16, 128, True),
        (torch.float16, 136, False),
        (torch.float32, 32, True),
        (torch.float32, 36, False),
    ],
)
def test_cuequiv_supports_matches_its_dense_kernels(dtype: torch.dtype, head_dim: int, supported: bool):
    """``cuequiv_supports`` encodes cuEquivariance's dense-mask head-dim limits; a release that moves them fails
    here rather than routing calls to a kernel that raises, or away from one that runs."""
    _skip_without_cuequiv()
    assert cuequiv_supports(dtype, head_dim) == supported
    num_heads, tokens = 2, 37
    q = torch.randn(1, 4, tokens, num_heads * head_dim, device="cuda", dtype=dtype)
    mask = torch.zeros(1, 4, 1, 1, tokens, device="cuda", dtype=dtype)
    bias = torch.randn(1, num_heads, tokens, tokens, device="cuda", dtype=dtype)
    backend = CuEquivAttention(0, num_heads, head_dim)
    if supported:
        backend.forward(q, q, q, biases=[mask, bias])
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError, match="hidden_dim"):
            backend.forward(q, q, q, biases=[mask, bias])


def test_native_policy_binding_contract(monkeypatch: pytest.MonkeyPatch):
    expected = {
        "target_sm": 90,
        "head_dim": 32,
        "num_heads": 4,
        "batch": 1,
        "tokens": 256,
        "i_dim": 2,
        "dtype": 2,
        "has_lse": True,
    }
    policy = SimpleNamespace(
        DType=SimpleNamespace(FLOAT16=1, BFLOAT16=2, FLOAT32=3),
        TriangleAttentionImplementation=SimpleNamespace(CLAUDE_KIT=10, CUTEDSL=11, UNSUPPORTED=12),
        select_triangle_attention=lambda **request: 10 if request == expected else 12,
    )
    library = SimpleNamespace(heuristic=policy)
    monkeypatch.setattr(heuristic_module.importlib, "import_module", lambda _name: library)

    assert heuristic_module._native_policy(90, torch.bfloat16, 32, 4, 1, 256, 2, True) == "ClaudeKit"


@pytest.mark.parametrize(
    "num_heads,batch,i_dim,tokens,dtype",
    [
        (4, 1, 176, 176, torch.bfloat16),
        (4, 1, 184, 184, torch.bfloat16),
        (8, 1, 144, 144, torch.bfloat16),
        (8, 1, 1024, 64, torch.bfloat16),
        (4, 1, 512, 512, torch.float16),
        (4, 1 << 20, 1 << 20, 1 << 20, torch.bfloat16),
    ],
)
def test_native_policy_matches_python_policy(
    num_heads: int,
    batch: int,
    i_dim: int,
    tokens: int,
    dtype: torch.dtype,
):
    native = heuristic_module._native_policy(90, dtype, 32, num_heads, batch, tokens, i_dim, True)
    if native is None:
        pytest.skip("the native heuristic policy is not built")

    assert native == heuristic_module._python_policy(90, dtype, 32, num_heads, batch, tokens, i_dim)


@pytest.mark.parametrize(
    "sm_version,dtype,head_dim",
    [
        (89, torch.bfloat16, 32),
        (90, torch.float16, 32),
        (90, torch.bfloat16, 64),
    ],
)
def test_claude_kit_refuses_unsupported_variants(
    sm_version: int,
    dtype: torch.dtype,
    head_dim: int,
):
    backend = ClaudeKitTriangleAttentionSM90D32(0, 4, head_dim, 4)
    backend._sm_version = sm_version
    q = torch.empty((1, 1, 8, 4 * head_dim), dtype=dtype, device="cuda")
    lengths = torch.full((1, 1), 8, dtype=torch.int32, device="cuda")
    pair_bias = torch.empty((1, 4, 8, 8), dtype=dtype, device="cuda")

    with pytest.raises(ValueError, match="requires SM90, bfloat16, and head_dim=32"):
        backend.forward(q, q, q, biases=[lengths, pair_bias])


@pytest.fixture
def native_claude_kit() -> tuple[object, object]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("claude_kit native tests require an SM90 CUDA device")
    try:
        return claude_kit_module._load_claude_kit_modules()
    except ClaudeKitTriangleAttentionUnavailable as error:
        pytest.fail(f"SM90 tests require the native ClaudeKit launcher: {error}")


def test_sm90_native_tests_fail_when_launcher_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))

    def unavailable() -> None:
        raise ClaudeKitTriangleAttentionUnavailable("extension has no ClaudeKit submodule")

    monkeypatch.setattr(claude_kit_module, "_load_claude_kit_modules", unavailable)
    try:
        with pytest.raises(pytest.fail.Exception, match="SM90 tests require the native ClaudeKit launcher"):
            native_claude_kit.__wrapped__()
    except pytest.skip.Exception as error:
        pytest.fail(f"fixture skipped on a missing SM90 launcher: {error}")


def _native_inputs(
    *,
    batch: int = 2,
    i_dim: int = 2,
    tokens: int = 256,
    num_heads: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch, i_dim, tokens, num_heads * 32)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pair_bias = torch.randn(batch, num_heads, tokens, tokens, dtype=torch.bfloat16, device="cuda")
    return q, k, v, pair_bias


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    lengths: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, i_dim, tokens, _ = q.shape
    q_heads = q.view(batch, i_dim, tokens, num_heads, 32).float()
    k_heads = k.view(batch, i_dim, tokens, num_heads, 32).float()
    v_heads = v.view(batch, i_dim, tokens, num_heads, 32).float()
    scores = torch.einsum("bijhd,bikhd->bijhk", q_heads, k_heads) * (32**-0.5)
    scores = scores + pair_bias.float().permute(0, 2, 1, 3).unsqueeze(1)
    valid = torch.arange(tokens, device=q.device).view(1, 1, 1, 1, tokens) < lengths.view(batch, i_dim, 1, 1, 1)
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    lse = torch.logsumexp(masked_scores, dim=-1)
    probabilities = torch.softmax(masked_scores, dim=-1)
    output = torch.einsum("bijhk,bikhd->bijhd", probabilities, v_heads)
    return output, lse


@pytest.mark.parametrize("length_shape", ["BI", "flat", "B"])
def test_claude_kit_mask_forms_numerics_and_preallocation(native_claude_kit, length_shape: str):
    del native_claude_kit
    torch.manual_seed(42)
    batch, i_dim, tokens, num_heads = 2, 2, 256, 4
    q, k, v, pair_bias = _native_inputs(
        batch=batch,
        i_dim=i_dim,
        tokens=tokens,
        num_heads=num_heads,
    )
    if length_shape == "B":
        lengths_b = torch.tensor([tokens, tokens - 63], dtype=torch.int32, device="cuda")
        lengths_bi = lengths_b[:, None].expand(batch, i_dim).contiguous()
        actual_s_kv = lengths_b
    else:
        lengths_bi = torch.tensor(
            [[tokens, tokens - 31], [tokens - 63, tokens - 95]],
            dtype=torch.int32,
            device="cuda",
        )
        actual_s_kv = lengths_bi if length_shape == "BI" else lengths_bi.flatten()

    output = torch.empty(batch * i_dim, tokens, num_heads, 32, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch * i_dim, tokens, num_heads, 1, dtype=torch.float32, device="cuda")
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)
    metadata = ClaudeKitTriangleAttentionMetadata()
    metadata.qkv_packed = False

    result = backend.forward(
        q,
        k,
        v,
        biases=[actual_s_kv, pair_bias],
        metadata=metadata,
        output=output,
        output_lse=lse,
    )
    expected, expected_lse = _reference(q, k, v, pair_bias, lengths_bi, num_heads)

    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result.float(), expected, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(
        lse.view(batch, i_dim, tokens, num_heads),
        expected_lse,
        atol=1e-1,
        rtol=1e-2,
    )


def test_claude_kit_rejects_a_bias_staging_grid_past_the_device_limit(native_claude_kit):
    """``batch * heads`` sizes the staging grid's z extent, which CUDA caps at 65535."""
    del native_claude_kit
    batch, num_heads, tokens = 16385, 4, 16
    q, k, v, pair_bias = _native_inputs(batch=batch, i_dim=1, tokens=tokens, num_heads=num_heads)
    lengths = torch.full((batch, 1), tokens, dtype=torch.int32, device="cuda")
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    with pytest.raises(ValueError, match="grid limit"):
        backend.forward(q, k, v, biases=[lengths, pair_bias])


def test_claude_kit_zero_length_rows(native_claude_kit):
    del native_claude_kit
    batch, i_dim, tokens, num_heads = 1, 2, 256, 4
    q, k, v, pair_bias = _native_inputs(
        batch=batch,
        i_dim=i_dim,
        tokens=tokens,
        num_heads=num_heads,
    )
    lengths = torch.tensor([[0, tokens]], dtype=torch.int32, device="cuda")
    output = torch.full(
        (batch * i_dim, tokens, num_heads, 32),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.full(
        (batch * i_dim, tokens, num_heads, 1),
        torch.nan,
        dtype=torch.float32,
        device="cuda",
    )
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    result = backend.forward(
        q,
        k,
        v,
        biases=[lengths, pair_bias],
        output=output,
        output_lse=lse,
    )

    assert torch.count_nonzero(result[:, 0]).item() == 0
    zero_row_lse = lse.view(batch, i_dim, tokens, num_heads)[:, 0]
    expected_lse = torch.full_like(zero_row_lse, -1e9 * (32**-0.5))
    torch.testing.assert_close(zero_row_lse, expected_lse, atol=16, rtol=0)


def test_claude_kit_skips_the_log_sum_exp_store_when_unrequested(native_claude_kit):
    """Omitting ``output_lse`` skips the store without changing the output."""
    del native_claude_kit
    torch.manual_seed(11)
    batch, i_dim, tokens, num_heads = 1, 3, 512, 4
    q, k, v, pair_bias = _native_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    lengths = torch.tensor([[tokens, tokens - 17, 0]], dtype=torch.int32, device="cuda")
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    lse = torch.full((batch * i_dim, tokens, num_heads, 1), torch.nan, dtype=torch.float32, device="cuda")
    with_lse = backend.forward(q, k, v, biases=[lengths, pair_bias], output_lse=lse).clone()
    without_lse = backend.forward(q, k, v, biases=[lengths, pair_bias]).clone()

    assert torch.equal(with_lse, without_lse)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("tokens", [257, 320, 333, 448])
@pytest.mark.parametrize("lengths_kind", ["full", "boundary", "mixed"])
def test_claude_kit_sequence_not_a_multiple_of_the_key_tile(
    native_claude_kit,
    tokens: int,
    lengths_kind: str,
):
    """Sequences that do not fill the kernel's 128-key tile stay correct."""
    del native_claude_kit
    torch.manual_seed(tokens)
    batch, i_dim, num_heads = 1, 3, 4
    q, k, v, pair_bias = _native_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    if lengths_kind == "full":
        lengths_bi = torch.full((batch, i_dim), tokens, dtype=torch.int32, device="cuda")
    elif lengths_kind == "boundary":
        lengths_bi = torch.full((batch, i_dim), tokens - 1, dtype=torch.int32, device="cuda")
    else:
        lengths_bi = torch.tensor([[tokens, (tokens // 128) * 128, tokens - 17]], dtype=torch.int32, device="cuda")

    output = torch.empty(batch * i_dim, tokens, num_heads, 32, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch * i_dim, tokens, num_heads, 1, dtype=torch.float32, device="cuda")
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=output, output_lse=lse)
    expected, expected_lse = _reference(q, k, v, pair_bias, lengths_bi, num_heads)

    torch.testing.assert_close(
        output.view(batch, i_dim, tokens, num_heads, 32).float(),
        expected,
        atol=1e-1,
        rtol=5e-2,
    )
    torch.testing.assert_close(lse.view(batch, i_dim, tokens, num_heads), expected_lse, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("tokens", [256, 333])
@pytest.mark.parametrize("extra_key_padding", [0, 8, 64])
def test_claude_kit_reads_pair_bias_padded_along_the_key_axis(
    native_claude_kit,
    tokens: int,
    extra_key_padding: int,
):
    """A ``[B, H, J, J_padded]`` pair bias is read by its padded stride and never past ``J``."""
    del native_claude_kit
    torch.manual_seed(tokens + extra_key_padding)
    batch, i_dim, num_heads = 1, 3, 4
    padded = (tokens + 7) // 8 * 8 + extra_key_padding
    shape = (batch, i_dim, tokens, num_heads * 32)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pair_bias = torch.randn(batch, num_heads, tokens, padded, dtype=torch.bfloat16, device="cuda")
    # Poison the padding: a kernel that reads past J would produce garbage.
    if padded > tokens:
        pair_bias[..., tokens:] = 1.0e4

    lengths_bi = torch.tensor(
        [[tokens, tokens - 1, (tokens // 128) * 128 or tokens]],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty(batch * i_dim, tokens, num_heads, 32, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch * i_dim, tokens, num_heads, 1, dtype=torch.float32, device="cuda")
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=output, output_lse=lse)
    expected, expected_lse = _reference(q, k, v, pair_bias[..., :tokens].contiguous(), lengths_bi, num_heads)

    torch.testing.assert_close(
        output.view(batch, i_dim, tokens, num_heads, 32).float(),
        expected,
        atol=1e-1,
        rtol=5e-2,
    )
    torch.testing.assert_close(lse.view(batch, i_dim, tokens, num_heads), expected_lse, atol=1e-1, rtol=1e-2)


def _packed_inputs(
    *,
    batch: int,
    i_dim: int,
    tokens: int,
    num_heads: int,
    key_padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q, k, v views of a packed in_proj output and a key-padded pair bias, NaN outside q/k/v."""
    hidden = num_heads * 32
    proj = torch.full((batch, i_dim, tokens, 4 * hidden + 8), torch.nan, dtype=torch.bfloat16, device="cuda")
    proj[..., : 3 * hidden] = torch.randn(batch, i_dim, tokens, 3 * hidden, device="cuda").to(torch.bfloat16)
    q, k, v = proj[..., :hidden], proj[..., hidden : 2 * hidden], proj[..., 2 * hidden : 3 * hidden]
    padded = (tokens + 7) // 8 * 8 + key_padding
    pair_bias = torch.full((batch, num_heads, tokens, padded), torch.nan, dtype=torch.bfloat16, device="cuda")
    pair_bias[..., :tokens] = torch.randn(batch, num_heads, tokens, tokens, device="cuda").to(torch.bfloat16)
    return q, k, v, pair_bias


def _run_and_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    lengths_bi: torch.Tensor,
    num_heads: int,
    *,
    buffers: bool,
) -> None:
    """Run the backend and compare O (and the LSE when requested) with the fp32 reference."""
    batch, i_dim, tokens, _ = q.shape
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)
    lse = None
    if buffers:
        output = torch.full((batch * i_dim, tokens, num_heads, 32), torch.nan, dtype=torch.bfloat16, device="cuda")
        lse = torch.full((batch * i_dim, tokens, num_heads, 1), torch.nan, dtype=torch.float32, device="cuda")
        result = backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=output, output_lse=lse)
        assert result.data_ptr() == output.data_ptr()
    else:
        result = backend.forward(q, k, v, biases=[lengths_bi, pair_bias])
    expected, expected_lse = _reference(q, k, v, pair_bias[..., :tokens], lengths_bi, num_heads)

    actual = result.reshape(batch, i_dim, tokens, num_heads, 32).float()
    live = lengths_bi > 0
    torch.testing.assert_close(actual[live], expected[live], atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[~live]).item() == 0
    if lse is not None:
        actual_lse = lse.view(batch, i_dim, tokens, num_heads)
        torch.testing.assert_close(actual_lse[live], expected_lse[live], atol=1e-2, rtol=1e-4)
        sentinel = torch.full_like(actual_lse[~live], -1e9 * 32**-0.5)
        torch.testing.assert_close(actual_lse[~live], sentinel, atol=16, rtol=0)


def _lengths(kind: str, tokens: int, i_dim: int) -> torch.Tensor:
    if kind == "full":
        lengths = [tokens] * i_dim
    elif kind == "pad14":
        # left-aligned 14 % padding: live rows see the valid prefix, padded rows nothing
        valid = tokens - round(0.14 * tokens)
        lengths = [valid] * (i_dim - 1) + [0]
    else:
        lengths = [tokens, 1, 0, max(tokens - 37, 1)][:i_dim]
    return torch.tensor([lengths], dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("buffers", [True, False], ids=["buffers", "no_buffers"])
@pytest.mark.parametrize("lengths_kind", ["full", "pad14", "ragged"])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("tokens", [127, 129, 255, 256, 257, 300, 383, 511, 512, 513, 777, 1001])
def test_claude_kit_packed_inputs_aligned_and_non_aligned(
    native_claude_kit,
    tokens: int,
    num_heads: int,
    lengths_kind: str,
    buffers: bool,
):
    """Production packed q/k/v views stay correct for any ``J``, not only multiples of 8."""
    del native_claude_kit
    torch.manual_seed(tokens * 10 + num_heads)
    i_dim = 4
    q, k, v, pair_bias = _packed_inputs(batch=1, i_dim=i_dim, tokens=tokens, num_heads=num_heads)

    _run_and_check(q, k, v, pair_bias, _lengths(lengths_kind, tokens, i_dim), num_heads, buffers=buffers)


@pytest.mark.parametrize(
    ("tokens", "num_heads", "lengths"),
    [
        (300, 4, [[280, 280, 243, 0, 1, 280]]),
        (257, 8, [[240, 240, 0, 100], [257, 3, 257, 200]]),
        (129, 4, [[128, 1, 0, 128, 127]]),
        (1032, 4, [[888, 888, 888, 888, 888, 0], [1032, 900, 1032, 0, 17, 1032]]),
    ],
    ids=["shorter_rows", "two_batches", "tile_edge", "padded_and_full_batches"],
)
def test_claude_kit_stages_each_batch_longest_row_as_the_key_end(
    native_claude_kit,
    tokens: int,
    num_heads: int,
    lengths: list[list[int]],
):
    """The staged pair bias is ``-inf`` at and past each batch's longest row."""
    del native_claude_kit
    torch.manual_seed(tokens + num_heads)
    q, k, v, pair_bias = _packed_inputs(batch=len(lengths), i_dim=len(lengths[0]), tokens=tokens, num_heads=num_heads)
    lengths_bi = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    _run_and_check(q, k, v, pair_bias, lengths_bi, num_heads, buffers=True)


@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("tokens", [300, 513])
def test_claude_kit_per_batch_key_limits(native_claude_kit, tokens: int, num_heads: int):
    """Batch elements whose live rows share one length fold that key mask into the staged bias."""
    del native_claude_kit
    torch.manual_seed(tokens + num_heads)
    i_dim = 4
    q, k, v, pair_bias = _packed_inputs(batch=3, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    short, shorter = tokens - 42, tokens // 2 - 3
    lengths_bi = torch.tensor(
        [[short, short, 0, short], [shorter] * i_dim, [tokens, shorter, 0, short]],
        dtype=torch.int32,
        device="cuda",
    )

    _run_and_check(q, k, v, pair_bias, lengths_bi, num_heads, buffers=True)


@pytest.mark.parametrize("pattern", ["rising_slow", "rising_fast", "late_spike", "large_bias", "masked_by_bias"])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("tokens", [300, 512])
def test_claude_kit_reference_max_updates(native_claude_kit, tokens: int, num_heads: int, pattern: str):
    """The lazily advanced reference max stays exact whatever the logit profile."""
    del native_claude_kit
    torch.manual_seed(tokens + num_heads)
    i_dim = 3
    q, k, v, pair_bias = _packed_inputs(batch=1, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    keys = torch.arange(tokens, device="cuda", dtype=torch.float32)
    live_bias = pair_bias[..., :tokens].float()
    if pattern == "rising_slow":
        live_bias = live_bias + 0.05 * keys
    elif pattern == "rising_fast":
        live_bias = live_bias + 0.5 * keys
    elif pattern == "late_spike":
        live_bias[..., tokens - 5] = 60.0
    elif pattern == "large_bias":
        live_bias = torch.empty_like(live_bias).uniform_(-100.0, 100.0)
    else:
        live_bias[:, :, ::2, :40] = -1e9
    pair_bias[..., :tokens] = live_bias.to(torch.bfloat16)
    lengths_bi = torch.tensor([[tokens, tokens - 37, 1]], dtype=torch.int32, device="cuda")

    _run_and_check(q, k, v, pair_bias, lengths_bi, num_heads, buffers=True)


def test_heuristic_additive_mask_matches_selected_claude_kit_delegate(native_claude_kit):
    del native_claude_kit
    batch, i_dim, tokens, num_heads = 1, 32, 512, 4
    q, k, v, pair_bias = _native_inputs(
        batch=batch,
        i_dim=i_dim,
        tokens=tokens,
        num_heads=num_heads,
    )
    lengths = torch.full((batch, i_dim), tokens - 31, dtype=torch.int32, device="cuda")
    valid = torch.arange(tokens, device="cuda").view(1, 1, tokens) < lengths.unsqueeze(-1)
    additive_mask = torch.where(valid, 0.0, -1e9).to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
    direct_output = torch.empty(batch * i_dim, tokens, num_heads, 32, dtype=torch.bfloat16, device="cuda")
    direct_lse = torch.empty(batch * i_dim, tokens, num_heads, 1, dtype=torch.float32, device="cuda")
    routed_output = torch.empty_like(direct_output)
    routed_lse = torch.empty_like(direct_lse)

    direct = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)
    expected = direct.forward(
        q,
        k,
        v,
        biases=[lengths, pair_bias],
        output=direct_output,
        output_lse=direct_lse,
    )
    heuristic = HeuristicTriangleAttention(0, num_heads, 32, num_heads)
    actual = heuristic.forward(
        q,
        k,
        v,
        biases=[additive_mask, pair_bias],
        output=routed_output,
        output_lse=routed_lse,
    )

    assert heuristic.last_backend_name == "ClaudeKit"
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(routed_lse, direct_lse, atol=0, rtol=0)


def test_heuristic_cuda_graph_capture(native_claude_kit):
    del native_claude_kit
    batch, i_dim, tokens, num_heads = 1, 32, 512, 4
    q, k, v, pair_bias = _native_inputs(
        batch=batch,
        i_dim=i_dim,
        tokens=tokens,
        num_heads=num_heads,
    )
    lengths = torch.full((batch, i_dim), tokens, dtype=torch.int32, device="cuda")
    output = torch.empty(batch * i_dim, tokens, num_heads, 32, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch * i_dim, tokens, num_heads, 1, dtype=torch.float32, device="cuda")
    backend = HeuristicTriangleAttention(0, num_heads, 32, num_heads)

    backend.forward(q, k, v, biases=[lengths, pair_bias], output=output, output_lse=lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(q, k, v, biases=[lengths, pair_bias], output=output, output_lse=lse)
    graph.replay()
    torch.cuda.synchronize()

    assert captured.data_ptr() == output.data_ptr()
    assert backend.last_backend_name == "ClaudeKit"


def _mixed_lengths(batch: int, i_dim: int, tokens: int, seed: int) -> torch.Tensor:
    """Per-row live lengths mixing full, partial, length-1 and zero rows, with whole row triples of zeros."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, tokens + 1, (batch, i_dim), generator=generator)
    lengths[:, ::5] = tokens
    lengths[:, 1::7] = 1
    lengths[:, 2::11] = 0
    for start in range(0, i_dim - 2, 13 * 3):
        lengths[:, start : start + 3] = 0  # a dead work tile
    return lengths.to(dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("tokens", [129, 300, 385])
def test_claude_kit_persistent_tile_stream_mixed_rows(native_claude_kit, tokens: int, num_heads: int):
    """Every CTA of the persistent kernel runs many work tiles back to back."""
    del native_claude_kit
    torch.manual_seed(tokens + num_heads)
    batch, i_dim = 2, 96
    q, k, v, pair_bias = _packed_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    lengths_bi = _mixed_lengths(batch, i_dim, tokens, seed=tokens * num_heads)

    _run_and_check(q, k, v, pair_bias, lengths_bi, num_heads, buffers=True)


def _call_buffers(batch: int, i_dim: int, tokens: int, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.full((batch * i_dim, tokens, num_heads, 32), torch.nan, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((batch * i_dim, tokens, num_heads, 1), torch.nan, dtype=torch.float32, device="cuda")
    return output, lse


def test_claude_kit_persistent_tiles_cuda_graph_replay_matches_eager(native_claude_kit):
    """Captured calls replay bit-identically: the tile tickets restart at every call, also inside a graph."""
    del native_claude_kit
    torch.manual_seed(7)
    calls = []
    for batch, i_dim, tokens, num_heads in ((2, 60, 300, 4), (1, 75, 257, 8)):
        q, k, v, pair_bias = _packed_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
        lengths_bi = _mixed_lengths(batch, i_dim, tokens, seed=tokens)
        backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)
        eager_output, eager_lse = _call_buffers(batch, i_dim, tokens, num_heads)
        backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=eager_output, output_lse=eager_lse)
        output, lse = _call_buffers(batch, i_dim, tokens, num_heads)
        calls.append((backend, (q, k, v, lengths_bi, pair_bias), eager_output, eager_lse, output, lse))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for backend, (q, k, v, lengths_bi, pair_bias), _, _, output, lse in calls:
            backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=output, output_lse=lse)
    for _ in range(3):
        for *_, output, lse in calls:
            output.fill_(torch.nan)
            lse.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        for *_, eager_output, eager_lse, output, lse in calls:
            assert torch.equal(output, eager_output)
            assert torch.equal(lse, eager_lse)


def test_claude_kit_concurrent_streams_match_eager(native_claude_kit):
    """Two calls in flight on two streams at once keep separate tile tickets and match their eager results."""
    del native_claude_kit
    torch.manual_seed(11)
    calls = []
    for batch, i_dim, tokens, num_heads in ((2, 90, 385, 4), (1, 120, 300, 8)):
        q, k, v, pair_bias = _packed_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
        lengths_bi = _mixed_lengths(batch, i_dim, tokens, seed=i_dim)
        backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)
        eager_output, eager_lse = _call_buffers(batch, i_dim, tokens, num_heads)
        backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=eager_output, output_lse=eager_lse)
        output, lse = _call_buffers(batch, i_dim, tokens, num_heads)
        calls.append((backend, (q, k, v, lengths_bi, pair_bias), eager_output, eager_lse, output, lse))
    torch.cuda.synchronize()

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(3):
        for *_, output, lse in calls:
            output.fill_(torch.nan)
            lse.fill_(torch.nan)
        torch.cuda.synchronize()
        for stream, (backend, (q, k, v, lengths_bi, pair_bias), _, _, output, lse) in zip(streams, calls, strict=True):
            with torch.cuda.stream(stream):
                backend.forward(q, k, v, biases=[lengths_bi, pair_bias], output=output, output_lse=lse)
        torch.cuda.synchronize()
        for *_, eager_output, eager_lse, output, lse in calls:
            assert torch.equal(output, eager_output)
            assert torch.equal(lse, eager_lse)


def test_claude_kit_cuda_graph_replays_recompute_the_staged_key_end(native_claude_kit):
    """A captured call derives each batch's staged key end on the device at every replay."""
    del native_claude_kit
    torch.manual_seed(7)
    batch, i_dim, tokens, num_heads = 2, 4, 300, 4
    q, k, v, pair_bias = _packed_inputs(batch=batch, i_dim=i_dim, tokens=tokens, num_heads=num_heads)
    # Captured with short rows: a staged key end frozen at capture time would mask live keys of every replay.
    lengths = [
        torch.tensor([[5, 5, 0, 5], [9, 1, 9, 9]], dtype=torch.int32, device="cuda"),
        torch.tensor([[3, 0, 3, 3], [1, 1, 1, 1]], dtype=torch.int32, device="cuda"),
    ]
    shape = (batch * i_dim, tokens, num_heads, 32)
    outputs = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    lses = [torch.empty((*shape[:3], 1), dtype=torch.float32, device="cuda") for _ in range(2)]
    backend = ClaudeKitTriangleAttentionSM90D32(0, num_heads, 32, num_heads)

    def run(output_buffers: list[torch.Tensor], lse_buffers: list[torch.Tensor]) -> None:
        for call in range(2):
            backend.forward(
                q,
                k,
                v,
                biases=[lengths[call], pair_bias],
                output=output_buffers[call],
                output_lse=lse_buffers[call],
            )

    run(outputs, lses)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(outputs, lses)

    replays = [
        ([[280, 280, 0, 280], [300, 300, 300, 300]], [[300, 250, 0, 1], [17, 17, 17, 0]]),
        ([[300, 0, 0, 300], [128, 128, 128, 128]], [[0, 0, 0, 0], [299, 300, 150, 0]]),
        ([[1, 1, 1, 1], [296, 0, 296, 296]], [[264, 264, 264, 264], [100, 256, 0, 129]]),
    ]
    for lengths_a, lengths_b in replays:
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor, dtype=torch.float32).to(torch.bfloat16))
        pair_bias[..., :tokens].copy_(torch.randn_like(pair_bias[..., :tokens], dtype=torch.float32).to(torch.bfloat16))
        lengths[0].copy_(torch.tensor(lengths_a, dtype=torch.int32))
        lengths[1].copy_(torch.tensor(lengths_b, dtype=torch.int32))
        graph.replay()
        eager_outputs = [torch.empty_like(output) for output in outputs]
        eager_lses = [torch.empty_like(lse) for lse in lses]
        run(eager_outputs, eager_lses)
        torch.cuda.synchronize()
        for call in range(2):
            assert torch.equal(outputs[call], eager_outputs[call])
            assert torch.equal(lses[call], eager_lses[call])
