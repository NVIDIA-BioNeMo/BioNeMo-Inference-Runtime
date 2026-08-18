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
"""Module-level CUDA-graph parity for the OpenFold3 ``DiffusionModule`` (B=1).

Captures a real ``DiffusionModule.forward`` call (real checkpoint weights and
real trunk-derived inputs, batch size 1) from an eager pipeline run on a bundled
sample, then drives the same module through a ``CUDAGraphOptimizationTracker``
with ``InputKeyMethod.EXACT``. Once the tracker has captured and verified a
graph, its replayed output must be **byte-identical** to the eager output — the
graph replays the exact same kernels on the same inputs, so parity is bitwise.
"""

import gc

import pytest
import torch

from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
    InputKeyMethod,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.modules.openfold3.diffusion_module import DiffusionModule
from tests.common.test_utils.openfold3.batched_input_tools import (
    AVAILABILITY_EXC,
    capture_and_assemble,
    clone_tree,
    harness_skip_reason,
)

# warmup are calls 1,2,3.  On call 4 is capture, verify, replay. Call 4
# output is used in production
_NUM_DRIVE_CALLS = 4

# All three tests below need real trunk-derived DiffusionModule inputs for this
# same pair of bundled samples.
_SAMPLE_IDS = ("T1038", "T1047s1")


@pytest.fixture(scope="module")
def _of3_diffusion_capture():
    """Module-scoped cache of ``capture_and_assemble(_SAMPLE_IDS)``.

    Each test below only needs a subset of one real eager-pipeline capture:
    the B=1 T1038 input (``per_sample[0]``, T1038's 199 tokens sort before
    T1047s1's 232 — see the token-bucket table in
    ``test_model_forward_with_cuda_graph.py``), the B=2 assembled batch, or
    both — and ``capture_and_assemble`` already computes all of them from one
    pipeline run. Without this cache each test redid that ~30-80s capture
    independently. Returns ``None`` when the checkpoint/metadata is
    unavailable; callers skip on that, same as a direct
    ``capture_and_assemble``/``make_batched_diffusion_inputs`` call would.
    """
    try:
        return capture_and_assemble(_SAMPLE_IDS)
    except AVAILABILITY_EXC:
        return None


@pytest.fixture(autouse=True)
def _free_captured_cuda_graphs():
    """Drop every CUDA graph captured during the test before the next one runs.

    Same hazard as the identically-named fixture in
    ``test_model_forward_with_cuda_graph.py``: capturing a graph registers the
    process-global default CUDA generator with it, and that registration
    outlives the test unless forced free here. Only load-bearing once
    ``_of3_diffusion_capture`` lets ``module`` (and any graphs captured around
    it) survive across tests — each test used to build its own from an
    independent pipeline run.
    """
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _diffusion_graph_config() -> CUDAGraphOptimizationConfig:
    """EXACT-keyed CUDA-graph config for the OF3 DiffusionModule.

    Mirrors what the model builds for the ``diffusion_module`` role: the routing
    config comes straight from the class ``@support_graph_optimization`` default
    (``graph_opt_default``), whose ``num_tokens`` acceptance max (1024) accepts
    the sample's token count.
    """
    return CUDAGraphOptimizationConfig(
        graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
        input_key_method=InputKeyMethod.EXACT,
        input_routing_config=DiffusionModule.graph_opt_default.input_routing_config,
    )


def test_of3_diffusion_module_eager_batched_matches_separate(_of3_diffusion_capture):
    """Eager (no cuda graph): each sample of a B=2 batch of ("T1038", "T1047s1")
    yields the same output as running that sample alone at B=1 — up to bf16
    batch-GEMM non-associativity (a B=2 forward tiles/accumulates GEMMs
    differently than a B=1 forward).

    The batched and per-sample inputs come from a single capture so the random
    ``xl_noisy`` is identical between them (separate pipeline runs would draw
    different noise as RNG advances across samples within a run)."""
    reason = harness_skip_reason(_SAMPLE_IDS)
    if reason is not None:
        pytest.skip(reason)
    if _of3_diffusion_capture is None:
        pytest.skip("openfold3 weights/metadata unavailable")
    module, per_sample, batched = _of3_diffusion_capture

    module = module.eval()
    with torch.no_grad():
        out = module(**clone_tree(batched))  # [2, S, N_atom, 3] (padded to the max shape)
        separate = [module(**clone_tree(kw)) for kw in per_sample]  # each [1, S, N_atom_i, 3]

    assert out.shape[0] == len(per_sample)
    # Each sample's real atoms occupy the first N_atom_i positions of the padded
    # batch; the trailing padding (atom_mask == 0) is inert, so the real slice
    # must match the separate B=1 output.
    for i, single in enumerate(separate):
        n_atom = single.shape[-2]
        got = out[i : i + 1, :, :n_atom, :].float()
        ref = single.float()
        rel = ((got - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6)).item()
        assert rel < 0.02, f"batched sample {i} differs from separate B=1 by mean-relative {rel:.2%} (> 2%)"


def test_of3_diffusion_module_b1_cuda_graph_byte_identical(_of3_diffusion_capture, sample_id="T1038"):
    assert sample_id == _SAMPLE_IDS[0], (
        f"sample_id={sample_id!r} would need its own capture — _of3_diffusion_capture only covers {_SAMPLE_IDS}"
    )
    reason = harness_skip_reason((sample_id,))
    if reason is not None:
        pytest.skip(reason)
    if _of3_diffusion_capture is None:
        pytest.skip("openfold3 weights/metadata unavailable")
    # A single-sample batch is a real B=1 DiffusionModule input (module + kwargs).
    # per_sample is ordered by token count, and T1038 (199 tokens) sorts before
    # T1047s1 (232), so index 0 is always T1038 — see _SAMPLE_IDS above.
    module, per_sample, _ = _of3_diffusion_capture
    kwargs = clone_tree(per_sample[0])

    module = module.eval()
    assert kwargs["xl_noisy"].shape[0] == 1, f"expected a B=1 input, got {tuple(kwargs['xl_noisy'].shape)}"

    # --- Eager reference ------------------------------------------------------
    with torch.no_grad():
        eager_out = module(**kwargs).clone()

    # --- Drive the CUDA-graph tracker: warmup -> capture -> replay ------------
    tracker = CUDAGraphOptimizationTracker(_diffusion_graph_config(), inner_module=module).eval()
    with torch.no_grad():
        for _ in range(_NUM_DRIVE_CALLS):
            graph_out = tracker(**kwargs)

    # Exactly one input shape -> exactly one captured graph, verified, no eager
    # fallback.
    assert len(tracker.graph_state_by_key) == 1, f"expected one captured graph, got {len(tracker.graph_state_by_key)}"
    ((key, state),) = tracker.graph_state_by_key.items()
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED, (
        f"graph did not verify: state={state.preparation_state.name}"
    )
    assert not tracker.fallback_to_eager_by_key.get(key, False), (
        "diffusion module fell back to eager instead of replaying a graph"
    )

    # The captured graph replays the same kernels on the same inputs, so the
    # replayed output is byte-identical to eager.
    assert torch.equal(graph_out, eager_out), (
        "CUDA-graph replay is not byte-identical to eager "
        f"(max|Δ|={(graph_out.float() - eager_out.float()).abs().max().item():.3e})"
    )


def test_of3_diffusion_module_b2_cuda_graph_byte_identical(_of3_diffusion_capture):
    """A B=2 batch of ("T1038", "T1047s1") run eager equals the same batch run
    through the CUDA-graph-optimized DiffusionModule (InputKeyMethod.EXACT),
    byte-for-byte: the tracker captures a graph for the B=2 input shape and its
    replay re-runs the identical kernels on identical inputs."""
    reason = harness_skip_reason(_SAMPLE_IDS)
    if reason is not None:
        pytest.skip(reason)
    if _of3_diffusion_capture is None:
        pytest.skip("openfold3 weights/metadata unavailable")
    module, _, batched = _of3_diffusion_capture
    batched = clone_tree(batched)

    module = module.eval()
    assert batched["xl_noisy"].shape[0] == len(_SAMPLE_IDS), (
        f"expected a B={len(_SAMPLE_IDS)} input, got {tuple(batched['xl_noisy'].shape)}"
    )

    # --- Eager reference ------------------------------------------------------
    with torch.no_grad():
        eager_out = module(**batched).clone()

    # --- Drive the CUDA-graph tracker: warmup -> capture -> replay ------------
    tracker = CUDAGraphOptimizationTracker(_diffusion_graph_config(), inner_module=module).eval()
    with torch.no_grad():
        for _ in range(_NUM_DRIVE_CALLS):
            graph_out = tracker(**batched)

    # Exactly one input shape -> one captured graph, verified, no eager fallback.
    assert len(tracker.graph_state_by_key) == 1, f"expected one captured graph, got {len(tracker.graph_state_by_key)}"
    ((key, state),) = tracker.graph_state_by_key.items()
    assert state.preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED, (
        f"graph did not verify: state={state.preparation_state.name}"
    )
    assert not tracker.fallback_to_eager_by_key.get(key, False), (
        "B=2 diffusion module fell back to eager instead of replaying a graph"
    )

    # The captured graph replays the same kernels on the same inputs, so the
    # replayed output is byte-identical to eager.
    assert torch.equal(graph_out, eager_out), (
        "CUDA-graph replay is not byte-identical to eager "
        f"(max|Δ|={(graph_out.float() - eager_out.float()).abs().max().item():.3e})"
    )
