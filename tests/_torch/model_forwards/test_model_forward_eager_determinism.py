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
"""Run-to-run determinism baseline for the diffusion (token) transformer path.

Companion to ``model_forwards/test_model_forward_with_cudagraph.py``. Where that
test asks "does the CUDA-graph replay match eager?", this one asks the
prerequisite question: "is
the *eager* model itself reproducible run-to-run?" — because the CUDA-graph
parity premise (graph trajectory == eager trajectory over a 200-step diffusion
rollout) only holds when the underlying kernels are deterministic.

Each model is run through the public ``build_processor`` API on the in-process
**serial** backend **twice**, eager (no acceleration), with the *same* seed and
inputs, over the same three bundled CASP14 monomers. The two runs' predicted
structures are then compared per target (CA lDDT + max coordinate deviation).

Both models are on the deterministic side of this baseline after mr204, and the
test pins that for each:

  * **Boltz-2** — fully deterministic: the two eager runs are **bit-identical**
    (max\\|Δcoord\\| == 0, CA lDDT == 1.0) for every target. This is why its
    CUDA-graph replay matches eager bit-for-bit.
  * **OpenFold3** — fully deterministic: the two eager runs are **bit-identical**
    (max\\|Δcoord\\| == 0, CA lDDT == 1.0) for every target. This is why its
    CUDA-graph replay matches eager bit-for-bit.


"""

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
from tensorrt_bionemo.data.schemas import InputRequest, MSARecord, Polymer
from tensorrt_bionemo.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor)
from tensorrt_bionemo.pipeline.stages.configs import WriterStageConfig
from tensorrt_bionemo.pipeline.stages.engine_stage import FoldingPredictionError
from tests.common.test_utils.basic import path_for_package_in_repo
from tests.common.test_utils.seeding import seed_everything

# --- Test configuration ----------------------------------------------------
# Each model is run eager twice and its two runs compared. Whether the two runs
# must be bit-identical is the property under test (see module docstring).
MODEL_SOURCES = ("openfold3", "boltz-2")
# Expected run-to-run determinism of the eager forward, per model.
EXPECT_DETERMINISTIC = {"openfold3": True, "boltz-2": True}
SEED = 42

# Bundled sample data (no Git LFS — real files shipped in the repo).
REPO_ROOT = path_for_package_in_repo(tests).parent
SAMPLES_DIR = REPO_ROOT / "examples" / "data" / "samples"
MONOMERS_DIR = SAMPLES_DIR / "monomers"
# Three smallest CASP14 monomers (≈95 / 100 residues).
SAMPLE_IDS = ("T1031", "T1033")

# Diffusion runtime args. A long rollout is what would surface any residual
# non-determinism — per-step noise accumulates over the trajectory — so it is
# the strongest setting for this determinism baseline.
RECYCLING_STEPS = 3
NUM_SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1

_SAMPLES_AVAILABLE = MONOMERS_DIR.is_dir() and all(
    (MONOMERS_DIR / f"{sid}.json").exists() for sid in SAMPLE_IDS)

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
    """
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
            msas = [MSARecord(path=str(MONOMERS_DIR / msa_field), format="a3m")]
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
# Pipeline construction + execution (serial backend, eager, via build_processor)
# ===========================================================================
def _build_processor_config(model_source: str,
                            output_dir: Path) -> EngineProcessorConfig:
    """Build a serial-backend, eager ``EngineProcessorConfig`` for the model.

    No ``accelerated_configs`` — the diffusion (token) transformer runs eager.
    """
    engine_kwargs: dict = {"profile_inference": True}
    model_cfg = _model_config(model_source)
    if model_cfg is not None:
        engine_kwargs["config"] = model_cfg

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


def _run_pipeline(model_source: str, requests: list[InputRequest],
                  output_dir: Path) -> dict[str, Path]:
    """Run the eager serial pipeline over ``requests`` and return written CIFs.

    ``should_continue_on_error`` defaults to False, so any per-request failure
    raises rather than silently producing an empty output.
    """
    config = _build_processor_config(model_source, output_dir)
    processor = build_processor(config)

    records = [{
        "record": req,
        "__record_id": req["input_id"],
        "random_seed": SEED,
    } for req in requests]

    # No outer inference_mode: the folding engine's execute() already applies
    # @torch.inference_mode() for the model forward, while the upstream CPU
    # stages run in normal mode — matching run_pipeline.py's serial path.
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
            f"(model={model_source})")
        paths[sid] = cif_path
    return paths


# ===========================================================================
# Structure comparison
# ===========================================================================
def _read_ca(cif_path: Path) -> struc.AtomArray:
    """Load the first model from a CIF and keep CA atoms of amino acids."""
    structure = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1)
    return structure[struc.filter_amino_acids(structure)
                     & (structure.atom_name == "CA")]


def _compare_lddt(cif_a: Path, cif_b: Path) -> tuple[float, float]:
    """CA lDDT (+ max CA coordinate deviation) between two structures.

    Both structures come from the same model with identical atom ordering, so
    coordinates are compared atom-for-atom for the diagnostic max-deviation.
    """
    ref = _read_ca(cif_a)
    subj = _read_ca(cif_b)
    assert ref.array_length() == subj.array_length(), (
        f"CA-atom count mismatch a={ref.array_length()} "
        f"b={subj.array_length()}")
    lddt = float(struc.lddt(ref, subj, aggregation="all"))
    max_dev = float(np.abs(ref.coord - subj.coord).max())
    return lddt, max_dev


# ===========================================================================
# Test
# ===========================================================================
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="eager determinism test requires CUDA")
@pytest.mark.skipif(not _SAMPLES_AVAILABLE,
                    reason=f"sample data not found under {MONOMERS_DIR}")
@pytest.mark.parametrize(
    "model_source",
    [
        pytest.param(
            m,
            marks=pytest.mark.skipif(
                not _model_weights_available(m),
                reason=f"{m} checkpoint unavailable (set {_CKPT_ENV[m]} or HF "
                f"auth for {_HF_CKPT[m][0]})"),
        ) for m in MODEL_SOURCES
    ],
)
def test_eager_run_to_run_determinism(model_source):
    requests = [_load_request(sid) for sid in SAMPLE_IDS]
    expect_deterministic = EXPECT_DETERMINISTIC[model_source]

    with tempfile.TemporaryDirectory() as dir_a, \
            tempfile.TemporaryDirectory() as dir_b:
        dir_a = Path(dir_a)
        dir_b = Path(dir_b)

        try:
            # Two independent eager runs, same seed and inputs.
            paths_a = _run_pipeline(model_source, requests, dir_a)
            torch.cuda.empty_cache()
            paths_b = _run_pipeline(model_source, requests, dir_b)
        except _AVAILABILITY_EXC as exc:
            pytest.skip(f"{model_source}: weights/metadata unavailable "
                        f"({type(exc).__name__}: {exc})")

        results = {}
        for sid in SAMPLE_IDS:
            lddt, max_dev = _compare_lddt(paths_a[sid], paths_b[sid])
            results[sid] = (lddt, max_dev)
            print(f"[determinism] {model_source} {sid}: CA lDDT={lddt:.4f} "
                  f"max|Δcoord|={max_dev:.4f} Å "
                  f"(expect {'identical' if expect_deterministic else 'divergent'})")

        # Bit-identicality (max|Δcoord| == 0) is the reliable per-target
        # determinism discriminator. CA lDDT is a lenient *local* metric: it can
        # stay at 1.0 even when coordinates drift by >1 Å (local geometry is
        # preserved), so a per-target ``lddt < 1.0`` check is not robust for the
        # non-deterministic model — only the aggregate (worst target) is.
        if expect_deterministic:
            # After mr204 both models are deterministic: every target's two
            # eager runs must be bit-identical.
            for sid in SAMPLE_IDS:
                lddt, max_dev = results[sid]
                assert max_dev == 0.0 and lddt == 1.0, (
                    f"{model_source} {sid}: expected bit-identical eager runs "
                    f"(lDDT 1.0, max|Δcoord| 0.0) but got lDDT {lddt:.4f}, "
                    f"max|Δcoord| {max_dev:.4f} Å — the eager forward lost "
                    "determinism")
        else:
            # OpenFold3: per-step non-determinism amplifies over the rollout, so
            # every target's two eager runs must differ (not bit-identical)...
            for sid in SAMPLE_IDS:
                lddt, max_dev = results[sid]
                assert max_dev > 0.0, (
                    f"{model_source} {sid}: expected non-deterministic eager "
                    f"runs (max|Δcoord| > 0) but the two runs were bit-identical "
                    f"(lDDT {lddt:.4f}, max|Δcoord| 0.0) — the forward became "
                    "deterministic (path changed?)")
            # ...and the divergence must be large enough to change the predicted
            # structure for at least the worst-affected target (CA lDDT < 1.0).
            min_lddt = min(lddt for lddt, _ in results.values())
            assert min_lddt < 1.0, (
                f"{model_source}: expected the non-determinism to drop CA lDDT "
                f"below 1.0 for at least one target, but the minimum across "
                f"{list(SAMPLE_IDS)} was {min_lddt:.4f} — the forward became "
                "deterministic (path changed?)")
