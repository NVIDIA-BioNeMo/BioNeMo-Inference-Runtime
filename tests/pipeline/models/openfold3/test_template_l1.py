# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 regression tests for the OpenFold3 direct-CIF template featurization.

Seven fixtures under ``examples/data/samples/templates/``, each exercising
a distinct featurization case:

  7tpu  multi-template top-k + rejection + score-ordered slotting
  8k7x  modified-residue non-standard branch + entity_poly MSE->'M'
  7qsj  altloc="occupancy" distogram/unit-vector value divergence
  8c4d  coverage-drop -> empty templates
  7r6r  mixed molecule types (protein + 2x DNA): pair mask
  8clz  non-standard branch on a symmetric 2-chain dimer
  5xgo  12-chain multichain: asym_id ordering + pair mask at scale

The pairwise tensors are O(n_tokens^2), so a compact signature is committed
instead of the raw tensors: the discrete per-token tensors in full plus stable
reductions of the pairwise ones.

The golden is generated from the current implementation (validated byte-exact,
atol=1e-4, against upstream OpenFold-3 at submodule revision f16647af).
Regenerate after an intentional change with::

    python tests/pipeline/models/openfold3/test_template_l1.py --regen-golden
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
FIXTURE_DIR = _REPO / "examples" / "data" / "samples" / "templates"
GOLDEN_PATH = _HERE / "data" / "template_l1_golden.pt"

TARGETS = ["7tpu", "8k7x", "7qsj", "8c4d", "7r6r", "8clz", "5xgo"]
_UV_ATOL = 1e-2  # tolerance on the summed unit-vector reduction (deterministic)

_PT_MAP = {
    "protein": "PROTEIN", "rna": "RNA", "dna": "DNA",
    "ccd_ligand": "CCD_LIGAND", "smiles_ligand": "SMILES_LIGAND",
}


def _build_template_features(target: str) -> dict[str, torch.Tensor]:
    """Load a fixture JSON (path-based templates + MSA) and run the OF3 structure
    + template feature generators, returning the five ``template_*`` tensors.

    The MSA is loaded for a realistic input shape; template features are
    MSA-independent, so it does not affect the golden."""
    from io import StringIO

    from tensorrt_bionemo.data.parsers import parse_a3m_content
    from tensorrt_bionemo.data.schemas.basic import (
        InputParsed, PolymerParsed, PolymerType, TemplateParsed)
    from tensorrt_bionemo.pipeline.models.openfold3.feature_context import \
        OpenFold3ContextGenerator
    from tensorrt_bionemo.pipeline.models.openfold3.feature_generators import (
        StructureFeatureGenerator, TemplateFeatureGenerator)

    doc = json.loads((FIXTURE_DIR / f"{target}.json").read_text())[0]
    polymers = []
    for p in doc["polymers"]:
        templates = None
        if p.get("templates"):
            # Resolve path -> file content, exactly as the parser stage does.
            templates = [
                TemplateParsed(content=(FIXTURE_DIR / e["path"]).read_text(),
                               format=e.get("format", "cif"),
                               chain_id=e.get("chain_id"))
                for e in p["templates"]]
        msas = None
        if p.get("msas"):
            a3m = (FIXTURE_DIR / p["msas"]).read_text()
            msas = [parse_a3m_content(StringIO(a3m))]
        # Each fixture polymer is a single chain, so take its scalar chain_id.
        cid = p["chain_id"]
        polymers.append(PolymerParsed(
            polymer_type=getattr(PolymerType, _PT_MAP[p["polymer_type"]]),
            chain_id=cid[0] if isinstance(cid, list) else cid,
            sequence=p.get("sequence"), msas=msas, paired_msas=None,
            templates=templates))

    parsed = InputParsed(input_id=target, polymers=polymers)
    row = OpenFold3ContextGenerator(config=None)(parsed)
    ctx = {"_row": row}
    sf = StructureFeatureGenerator(config=None, name="s")({}, ctx)
    trt = TemplateFeatureGenerator(config=None, name="t")(
        {"token_index": sf["token_index"]}, ctx)
    trt["_n_tokens"] = torch.tensor(int(sf["token_index"].shape[0]))
    return trt


def _compact(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce the five template tensors to a small, CI-stable golden signature."""
    restype = feats["template_restype"]                 # [T, N, 32] one-hot int
    assert torch.equal(restype.sum(-1), torch.ones_like(restype[..., 0])), \
        "template_restype expected one-hot over the class axis"
    dg = feats["template_distogram"]                     # [T, N, N, n_bins] one-hot
    valid = dg.sum(-1) > 0                               # [T, N, N] valid pair mask
    uv = feats["template_unit_vector"]                   # [T, N, N, 3] float
    return {
        "n_tokens": feats["_n_tokens"].to(torch.int64),
        "restype_argmax": restype.argmax(-1).to(torch.int16),
        "pseudo_beta_mask": feats["template_pseudo_beta_mask"].to(torch.bool),
        "backbone_frame_mask": feats["template_backbone_frame_mask"].to(torch.bool),
        "distogram_bin_hist": dg.reshape(-1, dg.shape[-1]).sum(0).to(torch.int64),
        "distogram_valid_pairs": valid.sum().to(torch.int64),
        "unit_vector_nonzero": (uv != 0).sum().to(torch.int64),
        "unit_vector_sum": uv.reshape(-1, 3).sum(0).to(torch.float64),
    }


def _assert_matches_golden(sig: dict, gold: dict, target: str) -> None:
    exact = ["n_tokens", "restype_argmax", "pseudo_beta_mask",
             "backbone_frame_mask", "distogram_bin_hist", "distogram_valid_pairs",
             "unit_vector_nonzero"]
    for k in exact:
        assert torch.equal(sig[k], gold[k]), f"{target}: {k} mismatch vs golden"
    assert torch.allclose(sig["unit_vector_sum"], gold["unit_vector_sum"],
                          atol=_UV_ATOL, rtol=0.0), \
        f"{target}: unit_vector_sum drift > {_UV_ATOL}"


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden not generated")
@pytest.mark.parametrize("target", TARGETS)
def test_template_l1_matches_golden(target: str) -> None:
    # weights_only=True: the golden is a tensor-only dict; avoid unpickling.
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
    assert target in golden, f"{target} missing from golden"
    sig = _compact(_build_template_features(target))
    _assert_matches_golden(sig, golden[target], target)


def _regen_golden() -> None:
    golden = {}
    for t in TARGETS:
        golden[t] = _compact(_build_template_features(t))
        print(f"  {t}: n_tokens={int(golden[t]['n_tokens'])} "
              f"pb_mask_nz={int(golden[t]['pseudo_beta_mask'].sum())} "
              f"disto_pairs={int(golden[t]['distogram_valid_pairs'])}")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(golden, GOLDEN_PATH)
    sz = GOLDEN_PATH.stat().st_size / 1024
    print(f"wrote {GOLDEN_PATH} ({sz:.1f} KB)")


if __name__ == "__main__":
    import sys
    if "--regen-golden" in sys.argv:
        _regen_golden()
    else:
        print(__doc__)
