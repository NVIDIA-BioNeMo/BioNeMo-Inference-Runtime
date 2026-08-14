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
    harness_skip_reason,
    make_batched_diffusion_inputs,
)

# warmup are calls 1,2,3.  On call 4 is capture, verify, replay. Call 4
# output is used in production
_NUM_DRIVE_CALLS = 4


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


def test_of3_diffusion_module_eager_batched_matches_separate():
    """Eager (no cuda graph): each sample of a B=2 batch of ("T1038", "T1047s1")
    yields the same output as running that sample alone at B=1 — up to bf16
    batch-GEMM non-associativity (a B=2 forward tiles/accumulates GEMMs
    differently than a B=1 forward).

    The batched and per-sample inputs come from a single capture so the random
    ``xl_noisy`` is identical between them (separate pipeline runs would draw
    different noise as RNG advances across samples within a run)."""
    sample_ids = ("T1038", "T1047s1")
    reason = harness_skip_reason(sample_ids)
    if reason is not None:
        pytest.skip(reason)
    try:
        module, per_sample, batched = capture_and_assemble(sample_ids)
    except AVAILABILITY_EXC as exc:
        pytest.skip(f"openfold3 weights/metadata unavailable ({type(exc).__name__}: {exc})")

    module = module.eval()
    with torch.no_grad():
        out = module(**batched)  # [2, S, N_atom, 3] (padded to the max shape)
        separate = [module(**kw) for kw in per_sample]  # each [1, S, N_atom_i, 3]

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


def test_of3_diffusion_module_b1_cuda_graph_byte_identical(sample_id="T1038"):
    reason = harness_skip_reason((sample_id,))
    if reason is not None:
        pytest.skip(reason)
    # A single-sample batch is a real B=1 DiffusionModule input (module + kwargs).
    try:
        module, kwargs = make_batched_diffusion_inputs(sample_ids=(sample_id,))
    except AVAILABILITY_EXC as exc:
        pytest.skip(f"openfold3 weights/metadata unavailable ({type(exc).__name__}: {exc})")

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


def test_of3_diffusion_module_b2_cuda_graph_byte_identical():
    """A B=2 batch of ("T1038", "T1047s1") run eager equals the same batch run
    through the CUDA-graph-optimized DiffusionModule (InputKeyMethod.EXACT),
    byte-for-byte: the tracker captures a graph for the B=2 input shape and its
    replay re-runs the identical kernels on identical inputs."""
    sample_ids = ("T1038", "T1047s1")
    reason = harness_skip_reason(sample_ids)
    if reason is not None:
        pytest.skip(reason)
    try:
        module, batched = make_batched_diffusion_inputs(sample_ids)
    except AVAILABILITY_EXC as exc:
        pytest.skip(f"openfold3 weights/metadata unavailable ({type(exc).__name__}: {exc})")

    module = module.eval()
    assert batched["xl_noisy"].shape[0] == len(sample_ids), (
        f"expected a B={len(sample_ids)} input, got {tuple(batched['xl_noisy'].shape)}"
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
