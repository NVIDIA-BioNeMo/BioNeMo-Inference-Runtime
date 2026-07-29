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
"""Shared helpers for the model-forward tests (eager + cuda-graph).

Request building from the bundled declarative sample JSONs and CA-lDDT
structure comparison, factored out so the eager-determinism and cuda-graph
parity suites can share a single implementation.
"""

import json
import os
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np

import tests
from tensorrt_bionemo.data.schemas import InputRequest, MSARecord, Polymer
from tests.common.test_utils.basic import path_for_package_in_repo

REPO_ROOT = path_for_package_in_repo(tests).parent
SAMPLES_DIR = REPO_ROOT / "examples" / "data" / "samples"
MONOMERS_DIR = SAMPLES_DIR / "monomers"


# ===========================================================================
# Checkpoint availability (env-var / gated-or-cached HF probe)
# ===========================================================================
# Per-model checkpoint resolution (local-checkpoint env var + HF repo/file).
# CI provides the checkpoint (env var or authenticated/cached HF); elsewhere the
# test skips gracefully instead of erroring on a gated/offline download.
_CKPT_ENV = {"openfold3": "OPENFOLD3_CKPT", "boltz2": "BOLTZ2_CKPT"}
_HF_CKPT = {
    "openfold3": ("OpenFold/OpenFold3", "checkpoints/of3-p2-155k.pt"),
    "boltz2": ("boltz-community/boltz2", "boltz2_conf.ckpt"),
}


def _model_weights_available(model_source: str, _ckpt_env=_CKPT_ENV, _hf_ckpt=_HF_CKPT) -> bool:
    """Whether ``model_source`` weights can be obtained for the model forward.

    True when a local checkpoint env var points at an existing file, an HF auth
    token is present (authenticated download), or the checkpoint is already in
    the local HF cache (offline). OpenFold3's HF repo is gated; boltz2 also
    needs CCD/mols metadata, which CI provisions alongside the checkpoint.
    """
    ckpt = os.environ.get(_ckpt_env[model_source])
    if ckpt and Path(ckpt).exists():
        return True
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    repo_id, filename = _hf_ckpt[model_source]
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(Path.home() / ".cache" / "hf"),
                        local_files_only=True)
        return True
    except Exception:
        return False


def _availability_exceptions() -> tuple:
    """Exception types that mean "weights/metadata couldn't be fetched" — a
    runtime safety net (e.g. boltz2 CCD/mols not cached) so we skip rather than
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
    Both OpenFold3 and boltz2 consume this same request schema.
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
# Structure comparison
# ===========================================================================
def _read_ca(path: Path) -> struc.AtomArray:
    """Load the first model from a CIF or PDB file and keep CA amino-acid atoms.

    Dispatches on the file suffix so predictions (``.cif`` written by the
    pipeline) and the bundled experimental ground truths (``.pdb``) share one
    reader.
    """
    path = Path(path)
    if path.suffix == ".pdb":
        structure = pdb.PDBFile.read(str(path)).get_structure(model=1)
    else:
        structure = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
    return structure[struc.filter_amino_acids(structure)
                     & (structure.atom_name == "CA")]


def _lddt_to_reference(prediction: Path, reference: Path) -> float:
    """CA lDDT of a predicted structure against a reference structure.

    Unlike :func:`_parity_lddt` (which compares two predictions that share atom
    ordering), a prediction and an experimental reference can differ in residue
    composition — the reference may be missing residues resolved in the model.
    CA atoms are therefore matched by residue id and only the shared residues
    are scored. lDDT is superposition-free, so no alignment beyond this residue
    matching is required. The reference supplies the "true" distances, so it is
    passed first.
    """
    pred = _read_ca(prediction)
    ref = _read_ca(reference)
    common = np.intersect1d(pred.res_id, ref.res_id)
    assert common.size, (
        f"no shared CA residues between prediction {prediction} and "
        f"reference {reference}")
    pred_m = pred[np.isin(pred.res_id, common)]
    ref_m = ref[np.isin(ref.res_id, common)]
    pred_m = pred_m[np.argsort(pred_m.res_id, kind="stable")]
    ref_m = ref_m[np.argsort(ref_m.res_id, kind="stable")]
    return float(struc.lddt(ref_m, pred_m, aggregation="all"))


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