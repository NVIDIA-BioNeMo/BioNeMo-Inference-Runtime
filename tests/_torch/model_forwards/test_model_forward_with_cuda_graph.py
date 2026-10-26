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
"""CUDA-graph parity of the folding models' graph regions.

Verifies that capturing a graph region yields predictions as accurate as the
unoptimized ("original") model, for each model in ``MODEL_SOURCES``.

Both runs go through the public ``build_processor`` API (the same entry point
an application driving the pipeline uses) on the in-process **serial**
backend, over two bundled sample targets from ``examples/data/samples`` (the
CASP14 monomers T1038 and T1047s1). Both runs clear the model's default
graph-region policies; the only difference between them is the
``accelerated_configs`` passed to the engine:

  * **original** — no acceleration; every region runs eagerly.
  * **cuda-graph** — one region (``trunk``, ``diffusion_module`` or
    ``confidence_pairformer``) is enabled by role under a copy of its
    model-owned policy (warmup -> capture -> verify -> replay, with permanent
    eager fallback on any failure).

RNG is seeded identically per request, so a correct graph yields structures
that match the eager run.

The test asserts, per model and region:
  1. the graph engaged and self-verified for every captured key
     (``GRAPH_VERIFIED`` with no eager fallback), one graph per target, and
  2. the cuda-graph prediction is as accurate as the eager one: per target,
     its CA lDDT to the experimental ground truth is within
     ``LDDT_GT_PARITY_TOL`` of the eager prediction's CA lDDT to ground truth.
"""

import gc
import os
import tempfile
from pathlib import Path

import pytest
import torch

import tests
from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir.configs import AcceleratedConfig, BackendType, BaseConfig
from bionemo_ir.data.schemas import InputRequest
from bionemo_ir.models.boltz2 import config as boltz2_config
from bionemo_ir.models.openfold3 import config as of3_config
from bionemo_ir.models.protenix import config as protenix_config
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig
from bionemo_ir.pipeline.stages.configs import WriterStageConfig
from tests._torch import packaged_cubin_kernels
from tests.common.test_utils.basic import path_for_package_in_repo
from tests.common.test_utils.model_forwards import (
    _AVAILABILITY_EXC,
    _find_sample_json,
    _lddt_to_reference,
    _load_request,
    _model_weights_available,
    _run_pipeline,
)

# --- Test configuration ----------------------------------------------------
# Models whose diffusion/token transformer is graph-optimizable. Each is run
# through the full pipeline twice (eager vs cuda-graph) and compared.
MODEL_SOURCES = ("openfold3",)
SEED = 42

# Bundled sample data (no Git LFS — real files shipped in the repo).
REPO_ROOT = path_for_package_in_repo(tests).parent
SAMPLES_DIR = REPO_ROOT / "examples" / "data" / "samples"
# Bundled experimental ground-truth structures (``{sample_id}.pdb``).
GT_DIR = SAMPLES_DIR / "gt"
# Bundled CASP14 monomers and the capture bucket each falls into:
#       id      num_tokens   num_token_bucket
#       T1031   95              (1, 128)
#       T1033   100             (1, 128)
#       T1038   199             (129, 256)
#       T1047s1 232             (129, 256)
# The two tuples below use T1038 and T1047s1, one sample per batch, in
# either order (the order is what varies between them).
SAMPLE_ID_TUPLE_D = (("T1038",), ("T1047s1",))  # tuple of batches each of size 1
SAMPLE_ID_TUPLE_E = (("T1047s1",), ("T1038",))

# Every distinct sample id referenced by any tuple above; used only for the
# module-level data-availability check below. Each test receives its own
# ``sample_ids`` tuple via the ``sample_id_tuple`` parametrization.
_ALL_SAMPLE_IDS = tuple(
    sorted({sid for tup in (SAMPLE_ID_TUPLE_D, SAMPLE_ID_TUPLE_E) for batch in tup for sid in batch})
)

# Diffusion runtime args. num_sampling_steps need only exceed the warmup
# threshold (3 calls) so each target's graph captures and then replays.
RECYCLING_STEPS = 3
NUM_SAMPLING_STEPS = 20
DIFFUSION_SAMPLES = 5

# Max allowed gap between the cuda-graph and eager predictions' CA lDDT to
# ground truth. Padding changes the order of operations in GEMMs (float
# non-associativity), so the two need only agree in accuracy, not bit-for-bit.
LDDT_GT_PARITY_TOL = 0.08

_SAMPLES_AVAILABLE = SAMPLES_DIR.is_dir() and all(_find_sample_json(sid) is not None for sid in _ALL_SAMPLE_IDS)


# Deliberately NOT the tables of the same name in tests.common.test_utils
# .model_forwards: the keys here are this module's model-source names
# (``boltz-2``, plus ``protenix-v2``, which the shared tables omit), and they
# are passed explicitly to ``_model_weights_available`` below. Do not "dedupe"
# these by importing the shared ones -- the keys would no longer match.
_CKPT_ENV = {
    "openfold3": "OPENFOLD3_CKPT",
    "boltz-2": "BOLTZ2_CKPT",
    "protenix-v2": "PROTENIX_V2_CKPT",
}
_HF_CKPT = {
    "openfold3": ("OpenFold/OpenFold3", "checkpoints/of3-p2-155k.pt"),
    "boltz-2": ("boltz-community/boltz-2", "boltz-2_conf.ckpt"),
}


@pytest.fixture(scope="module")
def _eager_baseline_cache(tmp_path_factory):
    """Cache of run 2's ("original"/eager) pipeline output, keyed by
    ``(model_source, sample_ids)``.

    Run 2 in :func:`_assert_cuda_graph_parity` is built with
    ``use_cuda_graph=False``, which gives the engine an empty
    ``accelerated_configs``, so it does not depend on ``module_name``: every
    region's parity test compares against the same unaccelerated pipeline for a
    given ``(model_source, sample_ids)``, which runs once instead of once per test.
    ``tmp_path_factory`` (not a bare
    ``tempfile.TemporaryDirectory``) keeps each cached run's CIF output alive for
    the whole module's test session so a later cache hit can still read it back;
    pytest cleans it up after the session.
    """
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Path]] = {}

    def get_or_run(
        model_source: str, module_name: str, sample_ids: tuple[str, ...], requests: list[InputRequest]
    ) -> dict[str, Path]:
        # module_name is accepted (the caller has one) but deliberately NOT part of
        # the key: nothing is wrapped in the eager run, so it cannot affect output.
        key = (model_source, sample_ids)
        if key not in cache:
            output_dir = tmp_path_factory.mktemp("eager")
            cache[key] = _run_eager_baseline(model_source, module_name, sample_ids, requests, output_dir)
        return cache[key]

    return get_or_run


@pytest.fixture(scope="module", autouse=True)
def _packaged_kernels():
    with packaged_cubin_kernels():
        yield


@pytest.fixture(autouse=True)
def _free_captured_cuda_graphs():
    """Drop every CUDA graph captured during the test before the next one runs.

    ``torch.cuda.graph(...)`` capture (in ``CUDAGraphOptimizationTracker``)
    registers the *process-global* default CUDA generator with the captured
    graph, and that registration lives as long as the graph does. The captured
    graphs are held by the pipeline's in-process model, which is only reachable
    through GC cycles (nn.Module parent<->child refs), so plain refcounting does
    not free them deterministically at test end. If a graph outlives the test,
    the default generator stays graph-registered and a later test in the same
    pytest worker that draws eager RNG from it (e.g. a bare
    ``torch.randn(..., device="cuda")`` in ``tests/ops/test_gated_sigmoid.py``)
    can raise "Offset increment outside graph capture encountered unexpectedly".

    Forcing a collection here frees the model and its graphs, which un-registers
    the default generator, isolating this module's graph state from the rest of
    the suite. Cheap relative to a graph-capture test and safe as a no-op when
    CUDA is unavailable.
    """
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _default_of3_model_config(model_source: str):
    """Pretrained model config with any checkpoint-compatibility overrides.

    Returns ``None`` to use the engine's default pretrained config.

    OpenFold3 ships checkpoints whose atom-transformer pair LayerNorms use
    different layouts, so each config's ``shared_pair_norm`` flag must match
    whichever checkpoint the engine will load (``OPENFOLD3_CKPT`` if set, else
    the HF default ``of3-p2-155k.pt``):

    * **shared** layout: a single ``atom_transformer.layer_norm_z`` per atom
      transformer (e.g. ``of3-p2-155k.pt``). The converter looks for this key
      only when ``shared_pair_norm=True`` (the pretrained-config default).
    * **per-block** layout: a LayerNorm per block
      (``blocks.N.attention_pair_bias.layer_norm_z``), e.g.
      ``of3_ft3_v1.pt``. The converter looks for these only when
      ``shared_pair_norm=False``.

    Matching the wrong layout makes the weight converter look for keys the
    checkpoint does not contain (``KeyError`` at load). Rather than guess the
    layout from the checkpoint filename, inspect the checkpoint's keys and set
    each transformer's flag from what is actually present.
    """
    if model_source != "openfold3":
        return None
    from bionemo_ir.registry import get_model_class

    cfg = get_model_class(model_source).get_pretrained_config(model_source)
    # Only a locally-present ``OPENFOLD3_CKPT`` can be inspected cheaply; when
    # it is unset the engine downloads the HF default (shared layout), which
    # already matches the pretrained-config default, so leave the flags alone.
    ckpt = os.environ.get("OPENFOLD3_CKPT")
    if not ckpt or not Path(ckpt).is_file():
        return cfg
    state_dict = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # (checkpoint prefix, config attr holding that transformer's config)
    transformers = [
        ("input_embedder.atom_attn_enc", cfg.input_embedder_config.atom_transformer_config),
        ("diffusion_module.atom_attn_enc", cfg.diffusion_module_config.atom_transformer_encoder_config),
        ("diffusion_module.atom_attn_dec", cfg.diffusion_module_config.atom_transformer_decoder_config),
    ]
    for prefix, tf_cfg in transformers:
        shared_key = f"{prefix}.atom_transformer.layer_norm_z.weight"
        tf_cfg.shared_pair_norm = shared_key in state_dict
    return cfg


# ===========================================================================
# Pipeline construction + execution (serial backend, via build_processor)
# ===========================================================================
def _eager_model_config(model_source: str) -> BaseConfig:
    """Pretrained config with every default graph-region policy cleared, so only the selected module captures."""
    config = _default_of3_model_config(model_source)
    if config is None:
        from bionemo_ir.registry import get_model_class

        config = get_model_class(model_source).get_pretrained_config(model_source)
    config.disable_cuda_graphs()
    return config


# Config class whose ``graph_optimization_config`` each model gives a graph
# region, by region role.
_REGION_POLICY_OWNERS = {
    "openfold3": {
        "trunk": of3_config.TrunkConfig,
        "diffusion_module": of3_config.DiffusionModuleConfig,
        "confidence_pairformer": of3_config.AuxiliaryHeadsConfig,
    },
    "boltz-2": {
        "trunk": boltz2_config.TrunkConfig,
        "diffusion_module": boltz2_config.ScoreModelConfig,
        "confidence_pairformer": boltz2_config.ConfidenceModuleConfig,
    },
    "protenix-v2": {
        "trunk": protenix_config.TrunkConfig,
        "diffusion_module": protenix_config.DiffusionModuleConfig,
        "confidence_pairformer": protenix_config.ConfidenceHeadConfig,
    },
}


def _region_graph_optimization_config(
    owner_config: type[BaseConfig], sample_ids: tuple[str, ...]
) -> CUDAGraphOptimizationConfig:
    """Copy a graph region's model-owned policy for a parity run.

    The copy keeps the policy's exact routing and first-call capture, verifies
    every capture, and keeps one graph per target with no residency budget, so
    no target's graph is evicted before the assertions read it.
    """
    return owner_config().graph_optimization_config.model_copy(
        update={
            "verify_capture": True,
            "num_graphs_max_for_this_module": len(sample_ids),
            "graph_cache_budget_bytes": None,
        }
    )


def _build_graph_optimization_config(
    model_source: str, module_name: str, sample_ids: tuple[str, ...], use_cuda_graph: bool
) -> CUDAGraphOptimizationConfig | None:
    """Build the CUDA-graph policy for the ``module_name`` region of ``model_source``.

    Returns ``None`` when ``use_cuda_graph`` is False (eager, no acceleration);
    otherwise a copy of the region's model-owned policy (see
    :func:`_region_graph_optimization_config`).
    """
    if not use_cuda_graph:
        return None
    return _region_graph_optimization_config(_REGION_POLICY_OWNERS[model_source][module_name], sample_ids)


def _build_processor_config(
    model_source: str,
    module_name: str,
    output_dir: Path,
    use_cuda_graph: bool,
    sample_ids: tuple[str, ...],
) -> EngineProcessorConfig:
    """Build a serial-backend ``EngineProcessorConfig`` for ``model_source``.

    Both runs clear the model's default graph-region policies first. When
    ``use_cuda_graph`` is set, the engine is given an ``accelerated_configs``
    entry that enables the ``module_name`` region by role under a copy of its
    model-owned policy; the engine applies it via ``model.optimize(...)`` at
    construction time. Otherwise the engine gets an **empty**
    ``accelerated_configs``, so the run is genuinely eager.
    """

    engine_kwargs: dict = {"profile_inference": True}
    engine_kwargs["config"] = _eager_model_config(model_source)
    graph_optimization_config = _build_graph_optimization_config(
        model_source=model_source,
        module_name=module_name,
        sample_ids=sample_ids,
        use_cuda_graph=use_cuda_graph,
    )

    engine_kwargs["accelerated_configs"] = (
        {
            module_name: AcceleratedConfig(
                backend=BackendType.TORCH,
                default=BaseConfig(graph_optimization_config=graph_optimization_config),
            ),
        }
        if graph_optimization_config is not None
        else {}
    )
    # insert engine_kwargs, which include the shape configuration
    engine_processor_config = EngineProcessorConfig(
        model_source=model_source,
        executor_backend=None,  # None -> in-process SerialProcessor
        engine_kwargs=engine_kwargs,
        runtime_args={
            "recycling_steps": RECYCLING_STEPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "diffusion_samples": DIFFUSION_SAMPLES,
        },
        writer_stage=WriterStageConfig(output_path=str(output_dir), format="cif"),
    )
    return engine_processor_config


def _graph_states(processor) -> list[tuple]:
    """Return the per-key ``(preparation_state, fallback_to_eager)`` tuples.

    Reaches into the serial processor's in-process folding engine and searches
    the model for the ``CUDAGraphOptimizationTracker`` of the enabled region (the
    only region with a policy). Searching ``model.modules()`` keeps this
    model-agnostic: each model keeps its regions at different paths.
    """
    tracker = None
    for udf in processor._udf_instances.values():
        folding = getattr(udf, "folding", None)
        if folding is None:
            continue
        for module in folding.engine.model.modules():
            if isinstance(module, CUDAGraphOptimizationTracker):
                tracker = module
                break
        break
    assert isinstance(tracker, CUDAGraphOptimizationTracker), (
        f"expected the enabled region to own a CUDAGraphOptimizationTracker, found {type(tracker).__name__}"
    )
    return [
        (s.preparation_state, tracker.fallback_to_eager_by_key.get(k, False))
        for k, s in tracker.graph_state_by_key.items()
    ]


def _assert_graph_verified_and_no_eager_fallback(states: list[tuple]) -> None:
    """Assert every captured key ended ``GRAPH_VERIFIED`` with no eager fallback.

    Args:
        states: per-key ``(preparation_state, fallback_to_eager)`` tuples as
            returned by :func:`_graph_states`.
    """
    assert states and all(ps == CUDAGraphPreparationState.GRAPH_VERIFIED and not fb for ps, fb in states), (
        "expected every captured key to end GRAPH_VERIFIED with no "
        f"eager fallback, got {[(ps.name, fb) for ps, fb in states]}"
    )


# ===========================================================================
# Structure comparison
# ===========================================================================
def _assert_cudagraph_lddt_to_gt_matches_eager(
    model_source: str,
    original_paths: dict,
    cudagraph_paths: dict,
    sample_ids: tuple[str, ...],
) -> None:
    """Assert the cuda-graph prediction is as accurate as the eager one.

    For each target, computes the CA lDDT of the eager prediction and of the
    cuda-graph prediction, each against the bundled experimental ground-truth
    structure, and asserts the two lDDTs agree within ``LDDT_GT_PARITY_TOL``. A
    correct graph shares weights, inputs, and RNG, so it should be neither
    better nor worse against ground truth than eager — up to float
    non-associativity from padded GEMMs.

    Args:
        model_source: model key, used for diagnostic messages.
        original_paths: ``{sample_id: cif_path}`` from the eager run.
        cudagraph_paths: ``{sample_id: cif_path}`` from the cuda-graph run.
        sample_ids: the targets to compare (this test's ``sample_id_tuple``).
    """
    for sid in sample_ids:
        gt = GT_DIR / f"{sid}.pdb"
        lddt_eager = _lddt_to_reference(original_paths[sid], gt)
        lddt_cudagraph = _lddt_to_reference(cudagraph_paths[sid], gt)
        delta = lddt_cudagraph - lddt_eager
        print(
            f"[parity] {model_source} {sid}: "
            f"cudagraph-vs-GT CA lDDT={lddt_cudagraph:.4f} Δ={delta:.4f}, "
            f"eager-vs-GT CA lDDT={lddt_eager:.4f} "
        )

        assert lddt_cudagraph >= lddt_eager - LDDT_GT_PARITY_TOL, (
            f"{model_source} {sid}: cudagraph-vs-GT lDDT-CA {lddt_cudagraph:.4f}"
            f" from eager-vs-GT {lddt_eager:.4f} by {delta:.4f}"
            f" is worse than tol={LDDT_GT_PARITY_TOL} — the graph changed the prediction's accuracy"
        )


# ===========================================================================
# Test
# ===========================================================================
# Per-model parametrization shared by every module's parity test: each model is
# skipped individually when its checkpoint can't be obtained.
_MODEL_PARAMS = [
    pytest.param(
        m,
        marks=pytest.mark.skipif(
            not _model_weights_available(m, _ckpt_env=_CKPT_ENV, _hf_ckpt=_HF_CKPT),
            reason=f"{m} checkpoint unavailable (set {_CKPT_ENV[m]} or HF auth for {_HF_CKPT[m][0]})",
        ),
    )
    for m in MODEL_SOURCES
]


def _flatten_sample_ids(sample_ids: tuple) -> tuple[str, ...]:
    """Flatten a ``sample_id_tuple`` to a flat tuple of sample-id strings.

    Accepts both forms: a flat tuple of sample ids (e.g. ``("T1047s1",)``) and a
    tuple of per-batch tuples (e.g. ``(("T1038",), ("T1047s1",))``). String
    elements are sample ids; tuple elements are batches whose members are
    flattened out. The pipeline runs one structure per forward
    (``engine_stage`` asserts ``len(rows) == 1``), so batches are size-1 and
    flattening recovers the per-target list the parity body folds into the graph
    cache. Batches of size > 1 would need engine ``B>1`` support to run.
    """
    flat: list[str] = []
    for item in sample_ids:
        if isinstance(item, str):
            flat.append(item)
        else:
            flat.extend(item)
    return tuple(flat)


# ``sample_id_tuple`` parametrizations. Each tuple is one test invocation, a
# tuple of per-batch tuples whose eager and cuda-graph runs fold exactly those
# targets; the test id joins the flattened member sample ids with '-'.
_SAMPLE_ID_TUPLE_PARAMS_ALL = [
    pytest.param(
        t,
        id="-".join(_flatten_sample_ids(t)),
        # Keep eager baselines on one worker.
        marks=pytest.mark.xdist_group("-".join(_flatten_sample_ids(t))),
    )
    for t in (SAMPLE_ID_TUPLE_D, SAMPLE_ID_TUPLE_E)
]

_SAMPLE_ID_TUPLE_PARAMS_D = [pytest.param(t, id="-".join(_flatten_sample_ids(t))) for t in (SAMPLE_ID_TUPLE_D,)]

# Multi-target tuples only (D, E), which every region's parity test runs over.
_SAMPLE_ID_TUPLE_PARAMS_MULTI = _SAMPLE_ID_TUPLE_PARAMS_ALL

# Shared skip/parametrize stack for every region's parity test: CUDA + sample
# data required, run each model. Listed top-to-bottom exactly as the decorators
# would stack.
_CUDA_GRAPH_PARITY_MARKS = (
    pytest.mark.skipif(not torch.cuda.is_available(), reason="pipeline CUDA-graph parity test requires CUDA"),
    pytest.mark.skipif(not _SAMPLES_AVAILABLE, reason=f"sample data not found under {SAMPLES_DIR}"),
    pytest.mark.parametrize("model_source", _MODEL_PARAMS),
)


def _cuda_graph_parity_marks(func):
    """Apply the shared ``_CUDA_GRAPH_PARITY_MARKS`` stack to a parity test.

    Applied innermost-first (bottom of the tuple up) so the result is identical
    to writing the marks as stacked decorators — same skips, same test IDs.
    """
    for mark in reversed(_CUDA_GRAPH_PARITY_MARKS):
        func = mark(func)
    return func


def _run_eager_baseline(
    model_source: str, module_name: str, sample_ids: tuple[str, ...], requests: list[InputRequest], output_dir: Path
) -> dict[str, Path]:
    """Run 2: the original (eager) model. Factored out so it can be called
    either directly (a fresh ``TemporaryDirectory`` per call) or through
    ``_eager_baseline_cache`` (a ``tmp_path_factory`` dir, reused across calls
    with the same ``(model_source, sample_ids)``).

    ``module_name`` only reaches ``_build_processor_config``'s graph-config
    branch, which returns ``None`` for ``use_cuda_graph=False`` before it is
    read, leaving ``accelerated_configs`` empty — so it does not affect the
    result and is not part of the cache key."""
    config = _build_processor_config(
        model_source=model_source,
        module_name=module_name,
        output_dir=output_dir,
        use_cuda_graph=False,
        sample_ids=sample_ids,
    )
    paths, _ = _run_pipeline(config, requests, sample_ids, output_dir)
    torch.cuda.empty_cache()
    return paths


def _assert_cuda_graph_parity(
    model_source: str,
    module_name: str,
    sample_ids: tuple[str, ...],
    eager_baseline_cache=None,
) -> None:
    """Run ``model_source`` eager vs cuda-graph on its ``module_name`` region and
    assert the graph captured/verified for every target and kept the prediction's
    accuracy. Shared body of the per-region parity tests below.

    ``sample_ids`` is this test's ``sample_id_tuple`` — a tuple of per-batch
    tuples; it is flattened to the per-target sample-id list to fold (each batch
    is size-1, the only form the one-structure-per-forward pipeline can run).
    ``eager_baseline_cache`` is the ``_eager_baseline_cache`` fixture, threaded
    through explicitly since this helper is not itself a test function; without
    it run 2 reruns fresh."""
    sample_ids = _flatten_sample_ids(sample_ids)
    requests = [_load_request(sid) for sid in sample_ids]

    with tempfile.TemporaryDirectory() as cudagraph_dir, tempfile.TemporaryDirectory() as fallback_original_dir:
        cudagraph_dir = Path(cudagraph_dir)

        try:
            # --- Run 1: model with the region captured ---------------------
            cudagraph_config = _build_processor_config(
                model_source=model_source,
                module_name=module_name,
                output_dir=cudagraph_dir,
                use_cuda_graph=True,
                sample_ids=sample_ids,
            )
            cudagraph_paths, cudagraph_proc = _run_pipeline(cudagraph_config, requests, sample_ids, cudagraph_dir)
            torch.cuda.empty_cache()

            # --- Run 2: original (eager) model, cached across regions ------
            if eager_baseline_cache is not None:
                original_paths = eager_baseline_cache(model_source, module_name, sample_ids, requests)
            else:
                original_paths = _run_eager_baseline(
                    model_source, module_name, sample_ids, requests, Path(fallback_original_dir)
                )

        except _AVAILABILITY_EXC as exc:
            pytest.skip(f"{model_source}: weights/metadata unavailable ({type(exc).__name__}: {exc})")

        # The region must have captured AND verified one graph per target (each
        # is a distinct input shape), with no eager fallback.
        states = _graph_states(cudagraph_proc)
        _assert_graph_verified_and_no_eager_fallback(states)
        assert len(states) == len(sample_ids), (
            f"expected {len(sample_ids)} captured graphs (one per target), got {len(states)}"
        )

        # --- Parity: cuda-graph predictions must match the original ------------
        _assert_cudagraph_lddt_to_gt_matches_eager(
            model_source=model_source,
            original_paths=original_paths,
            cudagraph_paths=cudagraph_paths,
            sample_ids=sample_ids,
        )


@_cuda_graph_parity_marks
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_MULTI)
def test_cuda_graph_diffusion_module_parity(model_source, sample_id_tuple, _eager_baseline_cache):
    _assert_cuda_graph_parity(
        model_source, "diffusion_module", sample_ids=sample_id_tuple, eager_baseline_cache=_eager_baseline_cache
    )


@_cuda_graph_parity_marks
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_MULTI)
def test_cuda_graph_trunk_parity(model_source, sample_id_tuple, _eager_baseline_cache):
    _assert_cuda_graph_parity(
        model_source, "trunk", sample_ids=sample_id_tuple, eager_baseline_cache=_eager_baseline_cache
    )


@_cuda_graph_parity_marks
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_MULTI)
def test_cuda_graph_confidence_pairformer_parity(model_source, sample_id_tuple, _eager_baseline_cache):
    _assert_cuda_graph_parity(
        model_source,
        "confidence_pairformer",
        sample_ids=sample_id_tuple,
        eager_baseline_cache=_eager_baseline_cache,
    )
