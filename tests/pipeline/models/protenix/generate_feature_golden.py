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

"""Regenerate the multi-molecule feature fixture in the pinned upstream Protenix environment.

Run: python generate_feature_golden.py /path/to/Protenix /path/to/PROTENIX_ROOT_DIR /path/to/output_dir
PROTENIX_ROOT_DIR must hold ``common/components.cif`` and its ``.rdkit_mol.pkl``.
Only the upstream package is imported; BioIR is not used for the reference.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_io import dump_conformers, dump_tensors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_root", type=Path)
    parser.add_argument("protenix_root_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    provenance = json.loads((Path(__file__).parent / "data/multimol_features.json").read_text())
    mols_path = args.protenix_root_dir / "common/components.cif.rdkit_mol.pkl"
    if hashlib.sha256(mols_path.read_bytes()).hexdigest() != provenance["ccd_mols_sha256"]:
        raise ValueError("Conformer pickle does not match the pinned fixture provenance")
    # Upstream reads its CCD paths from this variable at import time.
    os.environ["PROTENIX_ROOT_DIR"] = str(args.protenix_root_dir.resolve())
    sys.path.insert(0, str(args.upstream_root.resolve()))

    import numpy as np
    import torch
    from protenix.data.inference.infer_dataloader import InferenceMSAFeaturizer, InferenceTemplateFeaturizer
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.utils import data_type_transform

    # Upstream augments reference conformers with the global NumPy RNG.
    np.random.seed(0)
    sample = provenance["sample"]
    features, atom_array, _ = SampleDictToFeatures(sample).get_feature_dict()
    features["distogram_rep_atom_mask"] = torch.as_tensor(atom_array.distogram_rep_atom_mask).long()
    for extra in (
        InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=sample["sequences"], atom_array=atom_array, msa_pair_as_unpair=True, use_rna_msa=False
        ),
        InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=sample["sequences"], atom_array=atom_array, use_template=False, online_template_featurizer=None
        ),
    ):
        features.update({key: torch.as_tensor(value) for key, value in extra.items()})
    features = data_type_transform(feat_or_label_dict=features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    golden = {key: value for key, value in features.items() if key in provenance["features"]}
    dump_tensors(golden, args.output_dir / "multimol_features_golden.json")
    with open(mols_path, "rb") as stream:
        mols = pickle.load(stream)
    subset = {name: mols[name] for name in set(atom_array.res_name.tolist()) if name in mols}
    dump_conformers(subset, args.output_dir / "multimol_conformers_golden.json")


if __name__ == "__main__":
    main()
