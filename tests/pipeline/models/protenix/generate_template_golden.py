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

"""Regenerate the 2LWU fixture in the pinned upstream Protenix environment.

Run: python generate_template_golden.py /path/to/Protenix /path/to/output.json
The source revision required by data/2lwu_template.json must be checked out.
Only the upstream package is imported; BioIR is not used for the reference.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_io import dump_tensors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.upstream_root.resolve()
    provenance = json.loads((Path(__file__).parent / "data/2lwu_template.json").read_text())
    source = root / "examples/2lwu.cif"
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["source_sha256"]:
        raise ValueError("2LWU source does not match the pinned fixture provenance")
    sys.path.insert(0, str(root))

    import numpy as np
    import torch
    from protenix.data.constants import PROTEIN_CHAIN
    from protenix.data.template.template_featurizer import TemplateFeatureAssemblyLine
    from protenix.data.template.template_parser import TemplateParser
    from protenix.data.template.template_utils import TemplateHitProcessor

    parsed = TemplateParser.parse(file_id="2lwu", mmcif_string=source.read_text(), catch_all_errors=False).mmcif_object
    sequence = parsed.chain_to_seqres["A"]
    processor = TemplateHitProcessor(str(root / "examples"))
    hit, warning = processor._extract_template_features(
        parsed, "2lwu", dict(enumerate(range(len(sequence)))), sequence, sequence, "A"
    )
    if warning is not None:
        raise ValueError(warning)
    hit["template_release_date"] = parsed.header["release_date"].encode()
    result = (
        TemplateFeatureAssemblyLine(max_templates=4)
        .assemble(
            {
                0: {
                    "entity_id": 0,
                    "chain_id": "A",
                    "sequence": sequence,
                    "chain_entity_type": PROTEIN_CHAIN,
                    "templates": [hit],
                }
            },
            np.arange(len(sequence)),
        )
        .as_protenix_dict()
    )
    dump_tensors({key: torch.as_tensor(value) for key, value in result.items()}, args.output)


if __name__ == "__main__":
    main()
