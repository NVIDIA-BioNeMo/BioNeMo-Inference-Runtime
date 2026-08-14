<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Test Equivalence Script

Create this script as `test_pipeline_equivalence.py` in the model's test
directory (or `tmp/`) when validating a ported pipeline.

## Usage

```bash
python test_pipeline_equivalence.py --reqs reqs.json --samples samples/ --model <model_name>
```

## Preparation

1. **`reqs.json`**: JSON array of `InputRequest` dicts. Each has `"input_id"`,
   `"polymers"` (with `"sequence"`, `"chain_id"`, `"msas"`, etc.).
1. **`samples/`**: Directory with reference feature dicts saved as
   `<input_id>.pt` via `torch.save()` from the OSS pipeline.
1. **`--model`**: The registered model name (e.g., `"openfold2_ft2"`).

## Full Script

```python
"""
Test BioIR data pipeline equivalence against OSS reference outputs.

Usage:
    python test_pipeline_equivalence.py --reqs reqs.json --samples samples/ --model <model_name>
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from io import StringIO
from typing import Any, Optional

import numpy as np
import torch

from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.data.parsers import parse_a3m_content
from bionemo_ir.data.schemas.basic import (
    InputParsed,
    InputRequest,
    PolymerParsed,
)
from bionemo_ir.registry import (
    get_feature_factory,
    get_model_class,
    get_tokenizer,
)


def parse_input_request(input_request: InputRequest) -> InputParsed:
    """Convert a raw InputRequest dict into a parsed InputParsed structure."""
    polymers = input_request.get("polymers", [])
    polymers_parsed = []
    for polymer in polymers:
        msas = polymer.get("msas")
        msas_parsed = [] if msas is not None else None
        if msas is not None:
            for msa in msas:
                msa_parsed = parse_a3m_content(
                    StringIO(msa.get("content", ""))
                )
                msas_parsed.append(msa_parsed)

        paired_msas = polymer.get("paired_msas")
        paired_msas_parsed = [] if paired_msas is not None else None
        if paired_msas is not None:
            for paired_msa in paired_msas:
                paired_msa_parsed = parse_a3m_content(
                    StringIO(paired_msa.get("content", ""))
                )
                paired_msas_parsed.append(paired_msa_parsed)

        polymers_parsed.append(
            PolymerParsed(
                polymer_type=polymer.get("polymer_type"),
                chain_id=polymer.get("chain_id"),
                sequence=polymer.get("sequence"),
                msas=msas_parsed,
                paired_msas=paired_msas_parsed,
            )
        )
    return InputParsed(
        input_id=input_request.get("input_id"), polymers=polymers_parsed
    )


def generate_feature(
    model_name: str,
    parsed_req: InputParsed,
    config: BaseConfig,
    init_env: Optional[dict[str, Any]] = None,
) -> dict[str, torch.Tensor]:
    """Run the full BioIR feature pipeline: tokenizer -> generators -> collators."""
    tokenizer = get_tokenizer(model_name)
    feature_factory = get_feature_factory(model_name)

    # --- Tokenizer stage ---
    feature_context_generator = tokenizer.context_generator_specs[
        "primary"
    ].generator(config=config)
    context = feature_context_generator(parsed_req)

    for transform_spec in tokenizer.transform_specs:
        transform = transform_spec.transform(
            config=config, name=transform_spec.name, **transform_spec.kwargs
        )
        if transform.is_enabled():
            context = transform(context)

    # Ensure all values are tensors
    context_tensors = {}
    for k, v in context.items():
        if isinstance(v, torch.Tensor):
            context_tensors[k] = v
        elif isinstance(v, np.ndarray):
            context_tensors[k] = torch.from_numpy(v)
        elif isinstance(v, np.generic):
            context_tensors[k] = torch.tensor(v)

    # --- Feature generation stage ---
    env_ = deepcopy(init_env) if init_env is not None else {}
    if feature_factory.pre_init is not None:
        env_ = feature_factory.pre_init(env_)

    features_dict = {}
    for generator_spec in feature_factory.feature_generator_specs:
        generator = generator_spec.functor(
            config=config, name=generator_spec.name, **generator_spec.kwargs
        )
        if generator.is_enabled():
            features_dict[generator.name] = generator(context_tensors, env_)

    merged_feats = feature_factory.features_merger_func(
        contexts=context_tensors, features=features_dict
    )

    # --- Feature collation stage ---
    for collator_spec in feature_factory.feature_collator_specs:
        collator = collator_spec.functor(config=config, **collator_spec.kwargs)
        if collator.is_enabled():
            merged_feats = collator(merged_feats, env_)

    return merged_feats


def compare_features(
    batch: dict[str, torch.Tensor],
    ref_batch: dict[str, torch.Tensor],
    input_id: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> list[str]:
    """Compare two feature dicts. Returns list of error messages (empty = pass)."""
    errors = []

    ref_keys = set(ref_batch.keys())
    batch_keys = set(batch.keys())

    missing = ref_keys - batch_keys
    if missing:
        errors.append(f"[{input_id}] Missing keys: {missing}")

    extra = batch_keys - ref_keys
    if extra:
        errors.append(f"[{input_id}] Extra keys (not in ref): {extra}")

    for k in sorted(ref_keys & batch_keys):
        ref_v = ref_batch[k]
        new_v = batch[k]

        if not isinstance(ref_v, torch.Tensor):
            continue

        if ref_v.shape != new_v.shape:
            errors.append(
                f"[{input_id}] Shape mismatch for '{k}': "
                f"ref={ref_v.shape} vs new={new_v.shape}"
            )
            continue

        try:
            torch.testing.assert_close(
                new_v, ref_v, rtol=rtol, atol=atol, msg=k
            )
        except AssertionError as e:
            max_diff = (new_v.float() - ref_v.float()).abs().max().item()
            errors.append(
                f"[{input_id}] Value mismatch for '{k}': "
                f"max_abs_diff={max_diff:.6e} | {e}"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Test BioIR data pipeline equivalence against reference"
    )
    parser.add_argument(
        "--reqs", required=True,
        help="Path to reqs.json (JSON array of InputRequest dicts)"
    )
    parser.add_argument(
        "--samples", required=True,
        help="Directory with <input_id>.pt reference feature files"
    )
    parser.add_argument(
        "--model", required=True,
        help="Registered model name (e.g., openfold2_ft2)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-5,
        help="Relative tolerance for float comparisons"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5,
        help="Absolute tolerance for float comparisons"
    )
    args = parser.parse_args()

    with open(args.reqs, "r") as f:
        reqs = json.load(f)

    model_cls = get_model_class(args.model)
    pretrained_config = model_cls.get_pretrained_config(args.model)

    total = 0
    passed = 0
    all_errors = []

    for req in reqs:
        parsed_req = parse_input_request(req)
        input_id = parsed_req.get("input_id", "unknown")
        total += 1

        ref_path = os.path.join(args.samples, f"{input_id}.pt")
        if not os.path.exists(ref_path):
            all_errors.append(
                f"[{input_id}] Reference file not found: {ref_path}"
            )
            continue

        batch = generate_feature(
            args.model,
            parsed_req,
            pretrained_config,
            init_env={"random_seed": args.seed},
        )
        ref_batch = torch.load(
            ref_path, map_location="cpu", weights_only=True
        )

        errors = compare_features(
            batch, ref_batch, input_id, rtol=args.rtol, atol=args.atol
        )

        if errors:
            all_errors.extend(errors)
            print(f"  FAIL  {input_id} ({len(errors)} error(s))")
        else:
            passed += 1
            print(f"  PASS  {input_id}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

## Key Functions

### `parse_input_request(input_request) -> InputParsed`

Converts a raw JSON `InputRequest` dict into the `InputParsed` structure the
pipeline expects. Handles:

- Parsing MSA content from A3M strings via `parse_a3m_content(StringIO(...))`
- Parsing paired MSAs for multimer inputs
- Constructing `PolymerParsed` with `polymer_type`, `chain_id`, `sequence`,
  `msas`, `paired_msas`

### `generate_feature(model_name, parsed_req, config, init_env) -> dict`

Runs the complete BioIR feature pipeline manually (outside of Ray Data
stages):

1. **Tokenizer stage**: instantiates `ContextGenerator` from the tokenizer spec,
   runs it on parsed input, then applies each `TransformSpec` in order
1. **Feature generation stage**: calls `pre_init` to set up seeds, runs each
   `FeatureGeneratorSpec`, merges results with context
1. **Feature collation stage**: runs each `FeatureCollatorSpec` (including
   `SampleRepeater` for recycling)

### `compare_features(batch, ref_batch, input_id, rtol, atol) -> list[str]`

Compares two feature dicts and returns error messages:

- Checks for missing and extra keys
- Checks shape equality
- Uses `torch.testing.assert_close` with configurable tolerances
- Reports max absolute difference on failure
- **Never skips any tensor** — every key must be validated

### Stochastic feature validation

Some features are inherently random (e.g., `bert_mask`, random crops, MSA
sampling masks, `ref_pos` with per-residue rotation). **Do NOT skip them.**
Apply ALL of the following in order; the cheap mean/std test is mandatory, the
multi-seed test is optional.

1. **Shape + dtype + value-range checks** — non-negotiable; same as for
   deterministic tensors.
1. **Internal geometry test** (coordinate features only) — pairwise
   intra-residue distances must match within `atol=1e-4` because
   rotation/translation preserves them.
1. **Per-tensor mean & std (single run)** — element-wise mean and std of the
   BioIR tensor must be close to the OSS reference. Cast both to `float64`
   first to avoid precision drift.
1. **Per-axis mean & std (single run)** — for tensors with a known "stochastic
   axis" (e.g. the MSA-row axis for masked-MSA), check the marginal statistics
   along that axis.
1. **Multi-seed KS test (optional)** — only when iteration cost is low. Compare
   per-run mean distributions with `scipy.stats.ks_2samp`. Skip when the OSS
   reference requires GPU inference.

Acceptance thresholds (must be frozen in the test file before debugging, NOT
relaxed afterwards):

| Statistic                                 | Threshold                      |
| ----------------------------------------- | ------------------------------ |
| Shape, dtype                              | exact equality                 |
| Intra-residue pdist (coordinate features) | `atol=1e-4`, `rtol=1e-4`       |
| `\|mean_trt - mean_oss\|`                 | ≤ `0.05 * \|mean_oss\| + 1e-3` |
| `\|std_trt - std_oss\|`                   | ≤ `0.10 * std_oss + 1e-3`      |
| Multi-seed KS p-value (optional)          | > 0.01                         |

```python
def compare_stochastic_stats(
    trt: torch.Tensor,
    oss: torch.Tensor,
    name: str,
    mean_rtol: float = 0.05,
    mean_atol: float = 1e-3,
    std_rtol: float = 0.10,
    std_atol: float = 1e-3,
) -> tuple[bool, str]:
    """Single-run mean/std comparison for one stochastic tensor.

    Casts to float64 before reducing to avoid low-precision summation drift
    on large tensors. Shape / dtype are checked separately upstream.
    """
    a = trt.detach().cpu().to(torch.float64)
    b = oss.detach().cpu().to(torch.float64)
    m_t, m_o = float(a.mean()), float(b.mean())
    s_t, s_o = float(a.std(unbiased=False)), float(b.std(unbiased=False))
    mean_ok = abs(m_t - m_o) <= mean_rtol * abs(m_o) + mean_atol
    std_ok = abs(s_t - s_o) <= std_rtol * abs(s_o) + std_atol
    ok = mean_ok and std_ok
    return ok, (
        f"{name}: mean trt={m_t:+.6f} oss={m_o:+.6f} "
        f"diff={m_t - m_o:+.6f} [{'OK' if mean_ok else 'FAIL'}]  "
        f"std trt={s_t:.6f} oss={s_o:.6f} "
        f"diff={s_t - s_o:+.6f} [{'OK' if std_ok else 'FAIL'}]"
    )


def compare_axis_stats(
    trt: torch.Tensor,
    oss: torch.Tensor,
    axis: int,
    name: str,
    **thresh,
) -> tuple[bool, str]:
    """Per-axis (marginal) mean comparison along a stochastic axis."""
    a = trt.detach().cpu().to(torch.float64)
    b = oss.detach().cpu().to(torch.float64)
    return compare_stochastic_stats(
        a.mean(dim=axis), b.mean(dim=axis),
        f"{name}.mean[axis={axis}]", **thresh)


def compare_stochastic_feature_multiseed(
    trt_fn, oss_fn, key: str, n_runs: int = 20, p_threshold: float = 0.01
) -> list[str]:
    """Optional multi-seed KS test for stochastic features."""
    errors = []
    trt_stats, oss_stats = [], []
    for seed in range(n_runs):
        trt_out = trt_fn(seed=seed)[key].to(torch.float64)
        oss_out = oss_fn(seed=seed)[key].to(torch.float64)
        trt_stats.append(float(trt_out.mean()))
        oss_stats.append(float(oss_out.mean()))

    from scipy.stats import ks_2samp
    stat, pvalue = ks_2samp(trt_stats, oss_stats)
    if pvalue < p_threshold:
        errors.append(
            f"Stochastic feature '{key}' distributions differ across "
            f"{n_runs} runs: KS stat={stat:.4f}, p={pvalue:.4f}"
        )
    return errors
```

Print the mean/std numbers alongside the PASS/FAIL — silent boolean results hide
regressions:

```text
ref_pos:    STOCHASTIC  PASS
  mean trt=+0.000142 oss=+0.000139 diff=+0.000003 [OK]
  std  trt=1.234567 oss=1.234521 diff=+0.000046 [OK]
  intra_geom_pdist max_diff=3.2e-5 [OK]
bert_mask:  STOCHASTIC  FAIL
  mean trt=+0.150412 oss=+0.149987 diff=+0.000425 [OK]
  std  trt=0.357381 oss=0.396103 diff=-0.038722 [FAIL]   ← variance gap
```

## Example `reqs.json`

```json
[
  {
    "input_id": "test_monomer_1",
    "polymers": [
      {
        "polymer_type": "protein",
        "chain_id": ["A"],
        "sequence": "MAGTK...",
        "msas": [
          {"content": ">query\nMAGTK...\n>hit1\nMAGSK...\n"}
        ],
        "paired_msas": null
      }
    ]
  },
  {
    "input_id": "test_multimer_1",
    "polymers": [
      {
        "polymer_type": "protein",
        "chain_id": ["A"],
        "sequence": "MAGTK...",
        "msas": [{"content": ">q\nMAGTK...\n"}],
        "paired_msas": [{"content": ">q\nMAGTK...\n"}]
      },
      {
        "polymer_type": "protein",
        "chain_id": ["B"],
        "sequence": "MKLLV...",
        "msas": [{"content": ">q\nMKLLV...\n"}],
        "paired_msas": [{"content": ">q\nMKLLV...\n"}]
      }
    ]
  }
]
```

## Pattern B Testing Notes

For Pattern B pipelines (Boltz2-style), the test flow differs:

1. **Tokenizer stage produces a row dict**, not a flat tensor dict. Compare the
   row's tensor values and verify non-tensor data shapes/types.
1. **Feature generators read `context["_row"]`**. Set `context = {"_row": row}`
   before running generators.
1. **No single `compute_features()`** to compare against. Instead, compare the
   output of each generator independently, then compare the final collated
   output.
1. **Context caching**: For repeated test runs, save the row via `torch.save()`
   (tensor parts) and `pickle.dump()` (non-tensor parts). Load and inject as
   `context["_row"]` to skip the tokenizer stage.

```python
# Pattern B test flow
context_gen = tokenizer.context_generator_specs["primary"].generator(config=config)
row = context_gen(parsed_req)
context = {"_row": row}
context = feature_factory.pre_init(context)
# Run generators with context["_row"] = row
for gen_spec in feature_factory.feature_generator_specs:
    gen = gen_spec.functor(config=config, **gen_spec.kwargs)
    if gen.is_enabled():
        new_feats = gen(batch, context)
        batch.update(new_feats)
# Run collators
for col_spec in feature_factory.feature_collator_specs:
    col = col_spec.functor(config=config, **col_spec.kwargs)
    if col.is_enabled():
        batch = col(batch, context)
```

## Generating Reference Data from OSS

Use a script like this to save OSS outputs as reference `.pt` files:

```python
import torch
# ... run the OSS pipeline on the same reqs.json ...
for req in reqs:
    input_id = req["input_id"]
    oss_features = run_oss_pipeline(req)  # your OSS pipeline call
    torch.save(oss_features, f"samples/{input_id}.pt")
```
