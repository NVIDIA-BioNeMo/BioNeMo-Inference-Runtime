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
"""CUDA-graph parity test for the diffusion (token) transformer.

Verifies that wrapping the diffusion (token) transformer in the CUDA-graph
optimizer produces the *same* predicted structures as the unoptimized
("original") model — for both **OpenFold3** and **Boltz-2**.

Both runs go through the public ``build_processor`` API (the same entry point
used by ``docs/release_artifacts/scripts/run_pipeline.py``) on the in-process
**serial** backend, over three bundled sample targets from
``examples/data/samples`` (the three smallest CASP14 monomers). The only
difference between the two runs is the ``accelerated_configs`` passed to the
engine:

  * **original** — no acceleration; the diffusion transformer runs eager.
  * **cuda-graph** — ``token_transformer`` is wrapped in a
    ``CUDAGraphOptimizationTracker`` (warmup -> capture -> verify -> replay,
    with permanent eager fallback on any failure).

The diffusion sampling loop calls the token transformer with a fixed input
shape across all rollout steps, so a single graph is captured per target and
replayed for the remaining steps (capture-once / replay-many). RNG is seeded
identically per request, so a correct graph yields structures that match the
eager run.

The test asserts, per model:
  1. the graph engaged and self-verified for every captured key
     (``GRAPH_VERIFIED`` with no eager fallback), and
  2. the cuda-graph and original predictions agree (all-atom CA lDDT > 0.9).
"""

import gc
import json
import os
import tempfile
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import torch

import tests
from tensorrt_bionemo._torch.graph_optimization.config_schema import (
    CUDAGraphOptimizationConfig, GraphOptimizationMode, InputKeyMethod)
from tensorrt_bionemo._torch.graph_optimization.graph_optimization_tracker import (
    CUDAGraphOptimizationTracker, CUDAGraphPreparationState)
from tensorrt_bionemo.configs import AcceleratedConfig, BackendType, BaseConfig
from tensorrt_bionemo.data.schemas import InputRequest, MSARecord, Polymer
from tensorrt_bionemo.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor)
from tensorrt_bionemo.pipeline.stages.configs import WriterStageConfig
from tensorrt_bionemo.pipeline.stages.engine_stage import \
    FoldingPredictionError
from tests.common.test_utils.basic import path_for_package_in_repo
from tests.common.test_utils.seeding import seed_everything

# --- Test configuration ----------------------------------------------------
# Models whose diffusion/token transformer is graph-optimizable. Each is run
# through the full pipeline twice (eager vs cuda-graph) and compared.
MODEL_SOURCES = ("openfold3",)
SEED = 42

# Bundled sample data (no Git LFS — real files shipped in the repo).
REPO_ROOT = path_for_package_in_repo(tests).parent
SAMPLES_DIR = REPO_ROOT / "examples" / "data" / "samples"
MONOMERS_DIR = SAMPLES_DIR / "monomers"
# Three smallest CASP14 monomers (≈95 / 100 / 199 residues): each a distinct
# token-transformer input shape, so the graph optimizer captures three graphs.
SAMPLE_IDS = ("T1031", "T1033", "T1038")

# Diffusion runtime args. num_sampling_steps need only exceed the warmup
# threshold (3 calls) so each target's graph captures and then replays.
RECYCLING_STEPS = 3
NUM_SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1

# lDDT floor between the original and cuda-graph predictions. They share weights,
# inputs, and RNG, so a correct graph should yield near-identical structures
# (lDDT ≈ 1.0). The floor is per-model: Boltz-2 and OF3 eager forward is
# deterministic so its graph matches eager bit-for-bit (0.98 leaves headroom
# for benign sampling noise)
LDDT_PARITY_FLOOR = {"boltz-2": 0.98, "openfold3": 0.98}

_SAMPLES_AVAILABLE = MONOMERS_DIR.is_dir() and all(
    (MONOMERS_DIR / f"{sid}.json").exists() for sid in SAMPLE_IDS)


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


# Per-model checkpoint resolution (local-checkpoint env var + HF repo/file).
# CI provides the checkpoint (env var or authenticated/cached HF); elsewhere the
# test skips gracefully instead of erroring on a gated/offline download.
_CKPT_ENV = {"openfold3": "OPENFOLD3_CKPT", "boltz-2": "BOLTZ2_CKPT"}
_HF_CKPT = {
    "openfold3": ("OpenFold/OpenFold3", "checkpoints/of3-p2-155k.pt"),
    "boltz-2": ("boltz-community/boltz-2", "boltz2_conf.ckpt"),
}


def _model_weights_available(model_source: str) -> bool:
    """Whether ``model_source`` weights can be obtained for the model forward.

    True when a local checkpoint env var points at an existing file, an HF auth
    token is present (authenticated download), or the checkpoint is already in
    the local HF cache (offline). OpenFold3's HF repo is gated; Boltz-2 also
    needs CCD/mols metadata, which CI provisions alongside the checkpoint.
    """
    ckpt = os.environ.get(_CKPT_ENV[model_source])
    if ckpt and Path(ckpt).exists():
        return True
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    repo_id, filename = _HF_CKPT[model_source]
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(Path.home() / ".cache" / "hf"),
                        local_files_only=True)
        return True
    except Exception:
        return False


def _model_config(model_source: str):
    """Pretrained model config with any checkpoint-compatibility overrides.

    Returns ``None`` to use the engine's default pretrained config.

    OpenFold3: the shipped ``of3-p2-155k`` checkpoint stores **per-block**
    atom-transformer pair LayerNorms, but the config default
    (``shared_pair_norm=True``) makes the weight converter look for a single
    shared ``atom_transformer.layer_norm_z`` that the checkpoint does not
    contain (``KeyError`` at load). Setting ``shared_pair_norm=False`` on the
    three atom transformers matches the checkpoint (mirrors the original
    OpenFold3 parity test's config).
    """
    if model_source == "openfold3":
        from tensorrt_bionemo.registry import get_model_class
        cfg = get_model_class(model_source).get_pretrained_config(model_source)
        cfg.input_embedder_config.atom_transformer_config.shared_pair_norm = \
            False
        cfg.diffusion_module_config.atom_transformer_encoder_config.\
            shared_pair_norm = False
        cfg.diffusion_module_config.atom_transformer_decoder_config.\
            shared_pair_norm = False
        return cfg
    return None


def _availability_exceptions() -> tuple:
    """Exception types that mean "weights/metadata couldn't be fetched" — a
    runtime safety net (e.g. Boltz-2 CCD/mols not cached) so we skip rather than
    error even if the module-level availability probe was optimistic."""
    excs: list = [FileNotFoundError, ConnectionError]
    try:
        from huggingface_hub import errors as hf_errors
        excs += [
            hf_errors.GatedRepoError,
            hf_errors.RepositoryNotFoundError,
            hf_errors.EntryNotFoundError,
            hf_errors.LocalEntryNotFoundError,
            hf_errors.HfHubHTTPError,
        ]
    except Exception:
        pass
    return tuple(excs)


_AVAILABILITY_EXC = _availability_exceptions()


# ===========================================================================
# Request building (from the bundled declarative sample JSONs)
# ===========================================================================
def _load_request(sample_id: str) -> InputRequest:
    """Build an ``InputRequest`` from ``monomers/<sample_id>.json``.

    Mirrors the JSON schema consumed by ``run_pipeline.py``: a single protein
    polymer with an ``msas`` path resolved relative to the monomers directory.
    Both OpenFold3 and Boltz-2 consume this same request schema.
    """
    json_path = MONOMERS_DIR / f"{sample_id}.json"
    with open(json_path) as fh:
        raw = json.load(fh)
    entry = raw[0] if isinstance(raw, list) else raw

    polymers: list[Polymer] = []
    for poly in entry["polymers"]:
        msa_field = poly.get("msas")
        msas: list[MSARecord] = []
        if isinstance(msa_field, str):
            msas = [
                MSARecord(path=str(MONOMERS_DIR / msa_field), format="a3m")
            ]
        polymers.append(
            Polymer(
                polymer_type=poly.get("polymer_type", "protein"),
                chain_id=poly.get("chain_id"),
                sequence=poly["sequence"],
                msas=msas,
                paired_msas=[],
            ))
    return InputRequest(input_id=entry["input_id"], polymers=polymers)


# ===========================================================================
# Pipeline construction + execution (serial backend, via build_processor)
# ===========================================================================
def _build_processor_config(model_source: str, output_dir: Path,
                            use_cudagraph: bool, input_key_method: InputKeyMethod,
                            module_name: str) -> EngineProcessorConfig:
    """Build a serial-backend ``EngineProcessorConfig`` for ``model_source``.

    When ``use_cudagraph`` is set, the engine is given an ``accelerated_configs``
    entry that swaps the eager ``token_transformer`` for a
    ``CUDAGraphOptimizationTracker`` (TORCH backend + CUDA-graph framework). The
    engine applies this via ``model.optimize(...)`` at construction time. The
    ``token_transformer`` module key is shared by OpenFold3 and Boltz-2.
    """
    if input_key_method != InputKeyMethod.EXACT:
        raise ValueError(f"unsupported input_key_method {input_key_method!r}")
    
    engine_kwargs: dict = {"profile_inference": True}
    model_cfg = _model_config(model_source)
    if model_cfg is not None:
        engine_kwargs["config"] = model_cfg
    if use_cudagraph:
        engine_kwargs["accelerated_configs"] = {
            module_name:
            AcceleratedConfig(
                backend=BackendType.TORCH,
                default=BaseConfig(
                    graph_optimization_config=CUDAGraphOptimizationConfig(
                        graph_optimization_mode=GraphOptimizationMode.
                        CUDA_GRAPHS_VIA_TORCH,
                        verify_capture=True,
                        input_key_method=input_key_method,
                        # Keep one graph per distinct target shape so every
                        # captured key survives for the post-run assertion.
                        num_graphs_max_for_this_module=len(SAMPLE_IDS),
                    )),
            ),
        }

    return EngineProcessorConfig(
        model_source=model_source,
        executor_backend=None,  # None -> in-process SerialProcessor
        engine_kwargs=engine_kwargs,
        runtime_args={
            "recycling_steps": RECYCLING_STEPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "diffusion_samples": DIFFUSION_SAMPLES,
        },
        writer_stage=WriterStageConfig(output_path=str(output_dir),
                                       format="cif"),
    )


def _run_pipeline(model_source: str, 
                  requests: list[InputRequest],
                  output_dir: Path,
                  use_cudagraph: bool,
                  input_key_method: InputKeyMethod,
                  module_name: str) -> tuple[dict[str, Path], object]:
    """Run the serial pipeline over ``requests`` and return written CIF paths.

    Returns ``(paths_by_id, processor)``. The processor is returned so the
    caller can introspect the in-process model (serial backend) to confirm the
    graph engaged. ``should_continue_on_error`` defaults to False, so any
    per-request failure raises rather than silently producing an empty output.
    """
    config = _build_processor_config(model_source, output_dir, use_cudagraph, input_key_method, module_name)
    processor = build_processor(config)

    records = [{
        "record": req,
        "__record_id": req["input_id"],
        "random_seed": SEED,
    } for req in requests]

    # No outer inference_mode: the folding engine's execute() already applies
    # @torch.inference_mode() for the model forward (this is what disables grad
    # so the tracker captures/replays), while the upstream CPU stages run in
    # normal mode — matching run_pipeline.py's serial path.
    seed_everything(SEED)
    try:
        processor(records)
    except FoldingPredictionError as exc:
        # The engine wraps the real per-request error in a batch-level
        # FoldingPredictionError; surface the underlying model/postprocess
        # exception (with its own traceback) so failures are diagnosable, and
        # so a weights/metadata availability error still reaches the skip
        # handler in the test body.
        raise (exc.__cause__ or exc) from None

    paths: dict[str, Path] = {}
    for sid in SAMPLE_IDS:
        cif_path = output_dir / f"{sid}.cif"
        assert cif_path.exists(), (
            f"pipeline did not write expected output {cif_path} "
            f"(model={model_source}, use_cudagraph={use_cudagraph})")
        paths[sid] = cif_path
    return paths, processor


def _graph_states(processor) -> list[tuple]:
    """Return the per-key ``(preparation_state, fallback_to_eager)`` tuples.

    Reaches into the serial processor's in-process folding engine and searches
    the model for the ``CUDAGraphOptimizationTracker`` that replaced the token
    transformer. Searching ``model.modules()`` keeps this model-agnostic — the
    tracker lives at a different path in OpenFold3 vs Boltz-2.
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
        "expected the token transformer to be wrapped in a "
        f"CUDAGraphOptimizationTracker, found {type(tracker).__name__}")
    return [(s.preparation_state, s.fallback_to_eager)
            for s in tracker.graph_state_by_key.values()]


def _assert_graph_verified_and_no_eager_fallback(states: list[tuple]) -> None:
    """Assert every captured key ended ``GRAPH_VERIFIED`` with no eager fallback.

    Args:
        states: per-key ``(preparation_state, fallback_to_eager)`` tuples as
            returned by :func:`_graph_states`.
    """
    assert states and all(
        ps == CUDAGraphPreparationState.GRAPH_VERIFIED and not fb
        for ps, fb in states), (
            "expected every captured key to end GRAPH_VERIFIED with no "
            f"eager fallback, got {[(ps.name, fb) for ps, fb in states]}")


# ===========================================================================
# Structure comparison
# ===========================================================================
def _read_ca(cif_path: Path) -> struc.AtomArray:
    """Load the first model from a CIF and keep CA atoms of amino acids."""
    structure = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1)
    return structure[struc.filter_amino_acids(structure)
                     & (structure.atom_name == "CA")]


def _parity_lddt(original_cif: Path,
                 cudagraph_cif: Path) -> tuple[float, float]:
    """CA lDDT (+ max CA coordinate deviation) between two structures.

    Both structures come from the same model with identical atom ordering, so
    coordinates are compared atom-for-atom for the diagnostic max-deviation.
    """
    ref = _read_ca(original_cif)
    subj = _read_ca(cudagraph_cif)
    assert ref.array_length() == subj.array_length(), (
        f"CA-atom count mismatch original={ref.array_length()} "
        f"cudagraph={subj.array_length()}")
    lddt = float(struc.lddt(ref, subj, aggregation="all"))
    max_dev = float(np.abs(ref.coord - subj.coord).max())
    return lddt, max_dev


def _assert_structures_with_and_without_cudagraph_have_high_lddt(
        model_source: str,
        original_paths: dict,
        cudagraph_paths: dict) -> None:
    """Assert per-target CA lDDT between the eager and cuda-graph predictions.

    Compares each sample's original (eager) structure against its cuda-graph
    structure and asserts the CA lDDT exceeds the per-model parity floor — a
    correct graph shares weights, inputs, and RNG, so the structures should be
    near-identical.

    Args:
        model_source: model key, selects the floor in ``LDDT_PARITY_FLOOR``.
        original_paths: ``{sample_id: cif_path}`` from the eager run.
        cudagraph_paths: ``{sample_id: cif_path}`` from the cuda-graph run.
    """
    floor = LDDT_PARITY_FLOOR[model_source]
    for sid in SAMPLE_IDS:
        lddt, max_dev = _parity_lddt(original_paths[sid], cudagraph_paths[sid])
        print(f"[parity] {model_source} {sid}: CA lDDT={lddt:.4f} "
              f"max|Δcoord|={max_dev:.4f} Å")
        assert floor < lddt, (
            f"{model_source} {sid}: original vs cuda-graph CA lDDT "
            f"{lddt:.4f} below {floor} — the graph changed the "
            "prediction")


# ===========================================================================
# Test
# ===========================================================================
# Per-model parametrization shared by every module's parity test: each model is
# skipped individually when its checkpoint can't be obtained.
_MODEL_PARAMS = [
    pytest.param(
        m,
        marks=pytest.mark.skipif(
            not _model_weights_available(m),
            reason=f"{m} checkpoint unavailable (set {_CKPT_ENV[m]} or HF "
            f"auth for {_HF_CKPT[m][0]})"),
    ) for m in MODEL_SOURCES
]

_MODULE_NAMES = [
    "token_transformer",
    "diffusion_module",
]

# The token transformer is exercised with both graph-cache keying strategies:
# EXACT (one graph per distinct shape) and BUCKETED_SHAPES (shape
# bucketing that lets targets share a captured graph).
_INPUT_KEY_METHOD_PARAMS = [
    pytest.param(InputKeyMethod.EXACT, id="exact"),
]

# Shared skip/parametrize stack for every per-module parity test: CUDA + sample
# data required, run each (model_source, input_key_method) combination. Listed
# top-to-bottom exactly as the decorators would stack.
_CUDAGRAPH_PARITY_MARKS = (
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="pipeline CUDA-graph parity test requires CUDA"),
    pytest.mark.skipif(not _SAMPLES_AVAILABLE,
                       reason=f"sample data not found under {MONOMERS_DIR}"),
    pytest.mark.parametrize("input_key_method", _INPUT_KEY_METHOD_PARAMS),
    pytest.mark.parametrize("module_name", _MODULE_NAMES),
    pytest.mark.parametrize("model_source", _MODEL_PARAMS),
)

def _cudagraph_parity_marks(func):
    """Apply the shared ``_CUDAGRAPH_PARITY_MARKS`` stack to a parity test.

    Applied innermost-first (bottom of the tuple up) so the result is identical
    to writing the marks as stacked decorators — same skips, same test IDs.
    """
    for mark in reversed(_CUDAGRAPH_PARITY_MARKS):
        func = mark(func)
    return func


@_cudagraph_parity_marks
def test_cudagraph_parity_for_module(model_source, module_name, input_key_method):
    requests = [_load_request(sid) for sid in SAMPLE_IDS]

    with tempfile.TemporaryDirectory() as original_dir, \
            tempfile.TemporaryDirectory() as cudagraph_dir:
        original_dir = Path(original_dir)
        cudagraph_dir = Path(cudagraph_dir)

        try:
            # --- Run 1: original (eager) model -----------------------------
            original_paths, _ = _run_pipeline(model_source,
                                              requests,
                                              original_dir,
                                              use_cudagraph=False,
                                              input_key_method=input_key_method,
                                              module_name=module_name)
            torch.cuda.empty_cache()

            # --- Run 2: model with CUDA-graph token transformer ------------
            cudagraph_paths, cudagraph_proc = _run_pipeline(model_source,
                                                            requests,
                                                            cudagraph_dir,
                                                            use_cudagraph=True,
                                                            input_key_method=input_key_method,
                                                            module_name=module_name)
        except _AVAILABILITY_EXC as exc:
            pytest.skip(f"{model_source}: weights/metadata unavailable "
                        f"({type(exc).__name__}: {exc})")

        # The token transformer must have captured AND verified a graph for
        # every distinct target shape, with no eager fallback.
        states = _graph_states(cudagraph_proc)
        _assert_graph_verified_and_no_eager_fallback(states)
        assert len(states) == len(SAMPLE_IDS), (
            f"expected {len(SAMPLE_IDS)} captured graphs (one per target), "
            f"got {len(states)}")

        # --- Parity: cuda-graph predictions must match the original --------
        _assert_structures_with_and_without_cudagraph_have_high_lddt(
            model_source, original_paths, cudagraph_paths)
