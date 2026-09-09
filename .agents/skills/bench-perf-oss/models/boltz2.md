---
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
{}
---

# Boltz-2 OSS benchmark profile

This is the model-specific companion to
[`bench-perf-oss`](../SKILL.md). Generic environment isolation,
manifest construction, timing, scoring, synthetic compile probing,
charts, and reporting stay in the parent skill. This profile locks the
Boltz-2 source, checkpoint, data mapping, templates, MSA semantics, and
runtime choices that otherwise fail silently.

Use this profile for the folding key `boltz-2`. Do not use it for
`boltz-2-affinity`; that is a non-folding head with a different
dataset, metric, and output contract.

## Source and checkpoint contract

The OSS reference is:

- repository: `https://github.com/jwohlwend/boltz.git`
- tag: `v2.2.1`
- checkout: `$WORKDIR/oss/boltz`
- official entry: `boltz predict`, implemented by
  `src/boltz/main.py`

Clone the tag and verify it before installing:

```bash
git clone --branch v2.2.1 --depth 1 \
  https://github.com/jwohlwend/boltz.git \
  "$WORKDIR/oss/boltz"
git -C "$WORKDIR/oss/boltz" describe --tags --exact-match
```

The checked-in `examples/boltz2/boltz_private/` tree is not the
benchmark pin. It has different source behavior, including MSA
deletion bookkeeping, and cannot substitute for the official tag.

Both sides load the same folding checkpoint:

- id/file: `boltz2_conf.ckpt`
- hub: `boltz-community/boltz-2`
- expected MD5:
  `2f0a1775bf8fc366a1a85e2019eca288`
- BioIR override: `BOLTZ2_CKPT`

Do not use `boltz2_aff.ckpt`. Stage all required local assets before
timing:

```bash
scripts/fetch_weights.sh --model boltz-2
```

Record and verify:

- checkpoint SHA256 on both sides
- `BOLTZ_CCD_PATH` for `ccd.pkl`
- `BOLTZ_MOL_DIR` for the extracted molecule directory
- one offline strict checkpoint load in each interpreter

A missing or unexpected model key is a blocker. Never use
`strict=False` to turn a different checkpoint into the same variant.

## Environment and kernels

Boltz `v2.2.1` declares NumPy `<2.0` and CUDA 12 cuEquivariance extras,
so use an isolated OSS interpreter on the CUDA 13 PyTorch container.
Do not change BioIR's frozen packages to satisfy the OSS tree.

The upstream CUDA extra names cu12 packages. Apply the generic
CUDA-12-to-CUDA-13 rule:

- `cuequivariance_ops_cu12` → `cuequivariance-ops-cu13`
- `cuequivariance_ops_torch_cu12` →
  `cuequivariance-ops-torch-cu13`
- install a compatible `cuequivariance-torch`

Record every remap in `bench_config.json`, then verify that the OSS
freeze contains no cu12 wheel.

Recommended OSS inference has cuEquivariance enabled:

- load Boltz-2 with `use_kernels=True`
- do not pass `--no_kernels`
- verify the H100/A100 path did not auto-disable kernels

On SM90, cuEquivariance 0.11.1 reports that its fused triangle-attention
kernel targets SM100-family GPUs and uses the generic PyTorch path. This
is not the same as disabling `use_kernels`: retain cuEquivariance for
the supported operations and record the per-operation fallback.

Boltz does not use the DeepSpeed Evoformer attention extension. Record
`deepspeed_evo_attn=false`.

BioIR runs with `CUTEDSL_FORCE_CUBIN=1` and its default optimized
pretrained config, subject only to the template enablement below.

## Locked inference semantics

The benchmark lock is:

```json
{
  "recycling_steps": 3,
  "num_sampling_steps": 200,
  "diffusion_samples": 5
}
```

Map it to OSS prediction arguments as:

- `recycling_steps=3`
- `sampling_steps=200`
- `diffusion_samples=5`
- `max_parallel_samples=5`
- `use_msa_server=False`
- `use_kernels=True`
- seed identical to BioIR

The BioIR registry defaults to one diffusion sample, so the harness
must explicitly overlay five. Both implementations perform
`recycling_steps + 1` trunk passes internally; pass `3` to both. Do
not copy benchmark code that increments the BioIR argument before the
call.

Keep steering and affinity disabled. Lock `subsample_msa`, MSA caps,
precision, and every effective model preset rather than recording only
a CLI name.

BioIR currently has no inference-time MSA subsampling in its Boltz-2
MSA module. For feature-row parity, pass `subsample_msa=False` to OSS
rather than using the OSS CLI default of `True`.

## BioIR Path A and CUDA graph

Boltz-2 folding is Path A:

```python
EngineProcessorConfig(
    model_source="boltz-2",
    executor_backend=None,
    runtime_args={
        "recycling_steps": 3,
        "num_sampling_steps": 200,
        "diffusion_samples": 5,
    },
    engine_kwargs={"profile_inference": True, ...},
)
```

Use one `InputRequest` in each `processor([record])` call. Read
`model_inference_time` for the headline forward latency.

Graph the parent `diffusion_module`; do not separately graph the nested
`token_transformer`. Activate it with
`AcceleratedConfig(backend="torch")` and omit `default=` so the
module-declared safe routine remains intact. Its acceptance profile
covers at most 1024 tokens, so larger samples legitimately fall back
to eager. An explicit graph-optimization config replaces that profile;
do not supply one or raise the limit. Record graph use or fallback per
row rather than claiming every sample was captured.

Use the policy-first
[CUDA-graph audit](../measurement.md#audit-cuda-graph-routing-not-cache-emptiness).
Above the acceptance limit, an empty graph-state map and empty
capture-fallback map are the expected `eager_out_of_range` result.

## Templates must be enabled, not merely featurized

The BioIR pipeline can create Boltz-2 template features while the
default pretrained trunk leaves `use_templates_v2=False`. That is a
silent parity failure: template tensors exist, but `TemplateV2Module`
does not run.

Because `spec_full.json` contains template-bearing samples, construct
the official pretrained config and change only the template enable:

```python
from bionemo_ir.models.boltz2 import Boltz2

config = Boltz2.get_pretrained_config("boltz-2")
config.trunk.use_templates_v2 = True
```

Pass this config to the BioIR engine and record the exception to the
generic `config=None` rule in `bench_config.json`. Do not construct a
base config by hand or copy OSS layer settings into BioIR.

Set the same template cap on both sides. `None` means all supplied
templates; an integer means the identical cap on both. Verify that:

- `T1152`, `T1118v1`, and `T1158v1` each populate every expected slot
- the template module executes on both sides
- samples without templates have zero populated slots

The OSS YAML also needs explicit `templates:` entries. None of the
shipped CASP query YAML files carries those spec templates, so the
staging script must add them from the manifest.

## Dataset-to-OSS mapping

Use all 17 items in `$DATASET_ROOT/spec_full.json`. BioIR consumes
the spec through `InputRequest`, `MSARecord`, and `Template`. OSS
consumes staged Boltz YAML/CSV through the official prediction path.

Only these eight samples ship a `boltz_yaml`:

- `R1117`
- `T1152`
- `T1187`
- `R1136`
- `T1124`
- `T1118v1`
- `T1125`
- `T1158v1`

Their YAML `msa:` values may point into an old local example tree.
Generate `$WORKDIR/oss_data/queries/<id>.yaml` and rewrite each path to
the corresponding manifest-resolved file under `$DATASET_ROOT`.
Never edit the dataset YAML in place.

Generate native Boltz YAML for the other nine samples from the spec:

- preserve every polymer, copy count, CCD/SMILES ligand, RNA, and DNA
- attach every unpaired and paired alignment
- use distinct per-chain MSA paths unless two chains have the same
  sequence and byte-identical MSA content; the official schema requires
  those chains to reference the same path
- add every caller-supplied template

Write all source-to-staged mappings and row counts to
`oss_data/index.json`. Point parser caches and outputs away from the
dataset tree.

## MSA cap and bookkeeping

Lock both unpaired and paired MSA caps to `8192`, matching BioIR's
Boltz-2 predict-path constants. The official `v2.2.1` inference data
module reads `boltz.data.const.max_msa_seqs=16384`; even when
`process_inputs(max_msa_seqs=8192)` caps each chain, paired plus
unpaired construction can therefore exceed 8192 rows. Lock the
featurizer constant to 8192 in the OSS harness before creating the
data module. Do not use 16384 on one side.

Verify after featurization:

- every declared alignment was consumed
- paired rows are present on every chain that declares them
- no stale YAML path fell back to a query-only MSA
- both sides apply the same cap and subsampling seed

### Pinned `v2.2.1` deletion divergence

The official `v2.2.1` tag still rebinds
`chain_deletions = chain_deletions[del_start:del_end]` inside the
sequence loop in `construct_paired_msa`. After the first row, later
deletion slices can become empty, while BioIR preserves the real
per-row deletion counts.

Before the benchmark, search the actual pinned checkout and record the
result:

```bash
rg -n 'chain_deletions = chain_deletions\[' \
  "$WORKDIR/oss/boltz/src/boltz/data/feature/featurizerv2.py"
```

This is a known OSS input-feature divergence, not permission to mutate
either model. Do not zero BioIR deletions to match it. If MSA-bearing
proteins show a quality gap:

- record the affected rows and the source hit
- compare deletion summaries outside the timing window
- report the pinned OSS featurizer behavior explicitly

A later upstream revision may fix the bug, but it is not the declared
`v2.2.1` reference and must not replace the pin mid-benchmark.

## Custom-template parser normalization

The official parser reads every entity in a supplied template mmCIF.
Dataset templates can contain unrelated non-polymers whose author
component identifiers are absent from the bundled CCD (for example,
`A1DEZ`), causing preprocessing to fail before the requested protein
chain is aligned.

For the OSS-only staged template copy:

1. retain only `_atom_site` rows whose `label_entity_id` belongs to an
   `_entity_poly` entry
1. remove `_pdbx_nonpoly_scheme` rows
1. retain only `_struct_conn` rows whose two label asym IDs are both
   polymer entities
1. leave the dataset template untouched
1. hash the ordered polymer atom identity and coordinates before and
   after writing, and fail if either the hash or atom count changes
1. record the transformation and digest in `oss_data/index.json`

This is parser compatibility, not a template-content change: the
protein coordinates consumed by both implementations remain
identical.

## Official OSS harness

Base the harness on `boltz predict`:

1. run `process_inputs` with MSA server disabled, preprocessing threads
   set to one, and the locked MSA cap
1. construct `Boltz2InferenceDataModule`
1. strict-load `Boltz2` with the locked `predict_args` and
   `use_kernels=True`
1. time only the GPU-synchronized model forward/predict step
1. write structures through the official writer

With five diffusion samples, the writer emits multiple ranked CIFs.
Use rank 0 / the highest-confidence structure for the headline lDDT
and DockQ, and retain all per-sample scores in the artifact.

Do not time the CLI wall clock, preprocessing, H2D, writing, or
scoring.

The official model executes its template module on a dummy template
for no-template samples; BioIR detects `has_templates=False` and skips
that mathematically masked call. Record this implementation-level
performance difference. For template-bearing samples, verify the
module executes once per recycle on both sides.

## Fast synthetic compile probe

Boltz-2 is AF3-style. Start with Pairformer and DiffusionModule as
candidate roles, but validate them before any expensive real sample:

1. build deterministic synthetic direct inputs from each selected
   module's actual config
1. call two token/atom sizes in A, A, B, B order
1. test Pairformer and DiffusionModule separately
1. test the combined set only when both roles pass
1. run the real two-sample integration probe once on the largest
   passing set

Compile each selected child with `torch.compile(module)`: omit
`dynamic`, keep Dynamo's global defaults, and do not mark any input
dimensions. Initial A and first B may compile as `dynamic=None`
adapts; immediate repeats must not. Synthetic timings are diagnostics,
never benchmark results. Follow the generic retry ladder and publish
the largest role set that stays finite and measurement-stable. Do not
reject a role because relative L2 versus eager is large:
`torch.compile` may rewrite the graph while still feeding usable
features downstream. Judge that later with lDDT and DockQ. Report
every warmup-only specialization, including same-bin events.

Do not rely on the checkpoint's `compile_pairformer` switch for the
benchmark probe. In `v2.2.1`, eval forward selects
`pairformer_module._orig_mod` when `is_pairformer_compiled` is set,
bypassing the compiled wrapper. Install the benchmark's default-compile
wrapper directly into the parent module slot and leave the
`is_*_compiled` flags false.

On the `v2.2.1` / PyTorch 26.05 H100 stack tested for this profile:

- whole-`PairformerModule` Inductor compilation completed, adapted on
  the second size, and stayed capture-free on immediate repeats. Direct
  output drifted from eager by about 0.27 relative L2; Dynamo's eager
  backend was exact. That drift is diagnostic, not a reject reason.
- `structure_module.score_model` (`DiffusionModule`) also completed
  with finite outputs, one extra graph on the second size, and no
  recapture on immediate repeats.

Publish every role that stays finite and measurement-stable, including
Pairformer. The default compile column compiles both
`pairformer_module` and `structure_module.score_model` with
`torch.compile(module)`. Do not drop Pairformer solely because child
tensors drifted. Re-run the probes when the PyTorch or cuEquivariance
pin changes, and keep lDDT/DockQ as the downstream fitness check.

The published compile column compiles both roles with `dynamic=None`.
Record every warmup-only adaptation, including same-bin events.
Measured forwards must capture zero new graphs. Compile latency versus
eager can be mixed across the residue range; that is an outcome for
the run report, not a compile-validity gate. Put speedup, lDDT, and
DockQ numbers only in `$WORKDIR/results/`, not in this profile.

## Scoring and report checks

Use the generic OpenStructure and DockQ policy. In particular:

- monomers: `dockq=null`, `dockq_status=single_chain`
- RNA/DNA complexes: `dockq_status=unsupported_chain_types`
- protein-ligand complexes: add `--small_molecule`
- normalize missing occupancy and manifest-declared ligand `HETATM`
  metadata only in scorer copies

Before reporting, require:

- 17/17 BioIR and OSS eager forward rows
- exact MSA and template attachment parity
- finite lDDT on every row
- DockQ status on every row
- checkpoint and runtime locks identical
- cuEquivariance enabled in OSS
- compile stats valid for every published compile target
- required latency and speedup graphs generated from result JSON
