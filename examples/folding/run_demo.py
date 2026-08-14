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

"""Run a folding model through the BioIR processor pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bionemo_ir.data.schemas import (
    InputRequest,
    MSARecord,
    Polymer,
    Template,
)
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import WriterStageConfig

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "monomers" / "T1031.json"
DIFFUSION_MODELS = {"boltz-1", "boltz-2", "openfold3"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _load_msas(value: Any, base_dir: Path) -> list[MSARecord]:
    records = []
    for item in _as_list(value):
        item = {"path": item} if isinstance(item, str) else item
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported MSA entry: {item!r}")
        path = item.get("path")
        if path is not None:
            path = Path(path)
            if not path.is_absolute():
                path = base_dir / path
            path = str(path)
        records.append(
            MSARecord(
                content=item.get("content"),
                path=path,
                format=item.get("format", "a3m"),
            )
        )
    return records


def _load_templates(value: Any, base_dir: Path) -> list[Template]:
    records = []
    for item in _as_list(value):
        item = {"path": item} if isinstance(item, str) else item
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported template entry: {item!r}")
        path = item.get("path")
        if path is not None:
            path = Path(path)
            if not path.is_absolute():
                path = base_dir / path
            path = str(path)
        records.append(
            Template(
                content=item.get("content"),
                path=path,
                format=item.get("format", "cif"),
                chain_id=item.get("chain_id"),
            )
        )
    return records


def load_requests(path: Path) -> list[InputRequest]:
    """Load the declarative JSON format used by ``examples/data/samples``."""
    with path.open() as handle:
        raw = json.load(handle)
    entries = raw if isinstance(raw, list) else [raw]

    requests = []
    for entry in entries:
        polymers = []
        for polymer in entry.get("polymers", []):
            polymers.append(
                Polymer(
                    polymer_type=polymer.get("polymer_type", "protein"),
                    chain_id=polymer.get("chain_id"),
                    sequence=polymer["sequence"],
                    msas=_load_msas(polymer.get("msas"), path.parent),
                    paired_msas=_load_msas(polymer.get("paired_msas"), path.parent),
                    templates=_load_templates(polymer.get("templates"), path.parent),
                )
            )
        requests.append(InputRequest(input_id=entry["input_id"], polymers=polymers))
    return requests


def _engine_kwargs(model_source: str) -> dict[str, Any]:
    if model_source != "openfold3":
        return {}

    # The published OpenFold3 checkpoint stores per-block atom-transformer
    # pair LayerNorms. Match the model-forward pipeline tests.
    from bionemo_ir.registry import get_model_class

    config = get_model_class(model_source).get_pretrained_config(model_source)
    config.input_embedder_config.atom_transformer_config.shared_pair_norm = False
    config.diffusion_module_config.atom_transformer_encoder_config.shared_pair_norm = False
    config.diffusion_module_config.atom_transformer_decoder_config.shared_pair_norm = False
    return {"config": config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a folding model through BioIR build_processor.")
    parser.add_argument("--model-source", default="boltz-2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-format", choices=("pdb", "cif"), default="cif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    args = parser.parse_args()

    runtime_args = {}
    if args.model_source in DIFFUSION_MODELS:
        runtime_args = {
            "recycling_steps": args.recycling_steps,
            "num_sampling_steps": args.sampling_steps,
            "diffusion_samples": args.diffusion_samples,
        }

    config = EngineProcessorConfig(
        model_source=args.model_source,
        executor_backend=None,
        engine_kwargs=_engine_kwargs(args.model_source),
        runtime_args=runtime_args,
        writer_stage=WriterStageConfig(
            output_path=str(args.output_dir),
            format=args.output_format,
        ),
    )
    processor = build_processor(config)
    requests = load_requests(args.input)
    rows = [
        {
            "record": request,
            "__record_id": request["input_id"],
            "random_seed": args.seed,
        }
        for request in requests
    ]
    outputs = processor(rows)

    failed = [row for row in outputs if (row.get("__inference_error__") or {}).get("error_msg")]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(outputs)} predictions failed: {[row.get('__record_id') for row in failed]}"
        )
    print(f"Wrote {len(outputs)} prediction(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
