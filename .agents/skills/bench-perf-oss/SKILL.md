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
name: bench-perf-oss
description: >-
  Benchmark BioIR against an OSS model's e2e path for folding
  models, measuring model.forward() latency and GPU use on a bench
  set the shipped script rebuilds from RCSB and the MSA NIM — no
  dataset release to download. Serial:
  one sample per processor() call. Score written CIFs with
  OpenStructure lDDT and DockQ on every supported protein interface;
  record an explicit applicability status on every sample.
  Uses build_processor when a pipeline factory
  exists; for compute-only folding models (protenix-v2) uses the
  OSS data pipeline and swaps in the BioIR module. Every sample
  must load its bundled MSAs and templates when any exist, on both
  sides or neither. Use when comparing
  BioIR vs OSS folding performance. Do not use as-is for affinity
  or other non-folding heads — extend the skill first.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Benchmark BioIR vs OSS e2e `model.forward`

**Input:** model key + OSS source path. **Output:** WORKDIR harnesses, locked
config, per-sample `model.forward()` latency + GPU artifacts, OpenStructure
lDDT and DockQ vs `benchmarks/dataset/ground_truth/`,
comparison report.

This file routes the work: scope, the mandatory gates, and what each phase
hands to the next. The procedure for each phase is in
[workflow.md](workflow.md), and the rules behind it are in the references
listed below. Open a reference when its phase starts, not up front.

This is v1 for **structure folding** only (keys with a coordinate
head: Boltz-1/2, OpenFold2 / AlphaFold2, OpenFold3, `protenix-v2`).
The **headline** compared numbers are GPU-synced `model.forward()`
time and GPU memory. Parse, tokenize, featurize, H2D, postprocess,
write, and scoring are **outside** the timing window.

**Two required quality metrics**, neither a substitute for latency
and neither a substitute for the other: OpenStructure lDDT on every
sample, and DockQ on every supported protein-protein or
protein-small-molecule complex. Every row records DockQ applicability;
DockQ 2.1.3 does not support RNA/DNA interfaces. lDDT scores the whole
structure, DockQ scores the interfaces, so a complex can post a
respectable lDDT while docking its chains wrongly. See
[measurement.md](measurement.md#quality-lddt-and-dockq).

Do not run this playbook unchanged on a non-folding key
(`boltz-2-affinity`, or anything whose output is not a structure).
See [Other model kinds](#other-model-kinds).

**Templates on both sides or neither.** Template-bearing spec items
are in scope, and their templates get attached. A side that silently
failed to attach them still predicts — just from less evidence — so
the row looks valid while the structure, and its lDDT, answer a
different question. Where a model sizes its template tensors by
template count, latency diverges too. The two stacks may want
templates in different forms, so follow the pinned model profile and
verify attachment on the featurized batch on both sides before timing. See
[samples.md](samples.md#templates-are-in).

**Serial, one sample per call.** `executor_backend=None`. Feed
`processor([one_record])` — never the whole manifest as one batch
(host-RAM OOM). Same one-at-a-time rule on the OSS side.

Choose a path from the support matrix
(`docs/ref/support-matrix.md`, Pipeline column):

- **Processor path — `build_processor`.** The model has a factory (Boltz-1/2,
  OpenFold2 / AlphaFold2, OpenFold3). BioIR uses serial
  `build_processor`; OSS uses its own e2e. Each side featurizes itself.
- **OSS feature path — OSS pipeline + BioIR module.** The folding model has a
  compute path and no factory (`protenix-v2`). Use the OSS data
  pipeline for features, then replace the OSS `nn.Module` with the
  BioIR module. See [no-pipeline.md](no-pipeline.md).
  `boltz-2-affinity` is **not** a folding bench — see
  [Other model kinds](#other-model-kinds).

Authorities:

- Phase procedures — [workflow.md](workflow.md)
- Production API — [`docs/ref/api.md`][api] (`build_processor`,
  `profile_inference`, `runtime_args`)
- Serial vs Ray — [`docs/ref/architecture.md`][arch]
- Model keys and MSA rules — [`docs/ref/support-matrix.md`][sm]
- Checkpoints — [`docs/ref/model-weights.md`][weights]
- Dataset and sample catalog — [samples.md](samples.md) (built by
  `rebuild_dataset.py`)
- Environment — [environment.md](environment.md)
- Timing protocol — [measurement.md](measurement.md)
- MSA mapping (dataset -> OSS) — [msa.md](msa.md)
- Template mapping (dataset -> OSS) — [templates.md](templates.md)
- No BioIR pipeline — [no-pipeline.md](no-pipeline.md)
- DeepSpeed evoformer (OF2 / OF3) —
  [deepspeed-evoformer.md](deepspeed-evoformer.md)
- Model profiles — [Boltz-2](models/boltz2.md), [OpenFold2 /
  AlphaFold2](models/of2.md), [OpenFold3](models/of3.md),
  [Protenix-v2](models/protenix.md)
- Portable SKU package (optional) — [bundle.md](bundle.md)

[api]: ../../../docs/ref/api.md
[arch]: ../../../docs/ref/architecture.md
[sm]: ../../../docs/ref/support-matrix.md
[weights]: ../../../docs/ref/model-weights.md

## Mandatory gates

If any gate cannot be satisfied, stop and report the blocker. Do not
claim a complete bench.

1. **Both environments are probed and isolated first.** Check BioIR
   and OSS imports, set up whatever is missing, and resolve package
   conflicts before any timed run. See [environment.md](environment.md).
   Do not install OSS pins into the BioIR interpreter.
1. **OSS revision is the pin.** Use the exact revision selected by the
   model profile; when the profile selects a checked-in submodule
   gitlink, that gitlink is the pin. Boltz-1/2 = `v2.2.1`. OpenFold2 =
   [aqlaboratory/openfold `v2.2.0`](https://github.com/aqlaboratory/openfold/tree/v2.2.0).
   A `main` checkout or a different tag is a hard failure. See
   [environment.md](environment.md#oss-checkout-pins). A **recorded,
   reverted, input-side** patch on top of the pin is allowed — only
   to make both sides honour the same inputs, never to change the
   compute path
   ([templates.md](templates.md#temporary-patches-are-allowed)).
1. **No invented numbers.** Every latency, GPU power, clock-rate,
   or speedup figure in the report cites an executed command, exit
   code, and artifact path. Parse tool output — never estimate.
1. **Dataset is the one `rebuild_dataset.py` builds, only.** The builder ships
   in this skill at `dataset/`; local build root: `benchmarks/dataset/`. Use
   `spec_full.json` or `spec_monomer.json` ([samples.md](samples.md)). Do not
   substitute `examples/data/samples/`, FASTA-only inputs, or a synthetic
   residue sweep unless the user explicitly replaces the dataset.
1. **MSAs and templates are mandatory when they exist.** For every
   polymer that lists an unpaired or paired A3M (or Boltz CSV) or a
   template in the spec, both BioIR and OSS must load those files.
   Running without MSAs, with `use_msa_server`, or with empty protein
   MSAs when files exist is a hard failure.
1. **Templates are custom templates: all of them, both sides.** The
   spec's templates are caller-supplied, so both sides featurize
   **every** one, limited only by a cap set identically on each side.
   Turn off the similarity, coverage, date, and keep/drop gates that
   exist to filter *search* output; patch a side if config cannot.
   Verify by counting populated template slots per side — a non-zero
   mask passes while three of four templates are missing. See
   [MSA and template contract](#msa-and-template-contract),
   [msa.md](msa.md), and [templates.md](templates.md).
1. **Same sample set on both sides.** Write
   `$WORKDIR/ref_data/sample_manifest.json` before any timed run. Both
   harnesses assert exact set equality against it.
1. **The OSS input mapping is a script plus an index, not ad-hoc
   files.** When the OSS loader needs another layout or naming, build
   `$WORKDIR/oss_data/` with `bench/stage_oss_data.py` and record
   `oss_data/index.json`; the harness reads the index. Symlink the
   dataset, never copy or edit it, and assert every declared input
   staged with one distinct key per chain (Phase 2).
1. **Config parity is locked first.** Write
   `$WORKDIR/ref_data/bench_config.json` before Phase 4. Recycling,
   sampling, diffusion samples, precision, seed, and GPU id must match.
1. **BioIR uses the default model constructor.** With no config,
   each class uses `get_pretrained_config(self.model_name)`.
   That is the default optimized config
   (dtypes, triangle / pairwise backends). Construct with
   `config=None` (omit the kwarg). Processor path: omit
   `engine_kwargs["config"]` so the engine takes the same
   constructor path. That model captures CUDA graphs by default; an
   eager arm derives the pretrained config and calls
   `disable_cuda_graphs()` ([measurement.md](measurement.md#cuda-graphs)).
   Never pass a handmade `BaseConfig` or override a graph policy.
   Do not copy OSS layer configs onto BioIR. A model profile
   may require one feature enable on top of the official pretrained
   config—for example Boltz-2 custom templates require
   `trunk.use_templates_v2=True`. In that case call
   `get_pretrained_config`, change only the documented flag, pass that
   config, and record the exception. An approximate option a profile
   turns on by default is the same kind of exception
   ([approximate options](#approximate-options)). See [`docs/ref/api.md`][api].
1. **Serial, one GPU, one sample per call.**
   `executor_backend=None`. Pin `CUDA_VISIBLE_DEVICES` to one device.
   Loop the in-scope manifest and call `processor([one_record])` (or
   OSS featurize + `forward` on that one batch). Never
   `processor(all_rows)` — feature generation then holds every
   record in host RAM and OOM-kills before the engine. See
   [measurement.md](measurement.md#executor-serial-not-ray).
1. **Every timed column runs on the same host and the same GPU.** Run
   BioIR and every OSS column on one physical machine and one device,
   not just the same GPU model. Record the hostname, GPU UUID, and
   driver version in `ref_data/bench_config.json`. Latency from
   different hosts is not comparable, because CPU, driver, power limit,
   and thermals differ. If a column must be rerun elsewhere, rerun every
   column there.
1. **Both scorers are installed before any timed run.** Put `ost`
   in its **own conda env** and `DockQ` in its **own venv** (not
   BioIR, not OSS, not Miniforge base) so scorer deps cannot fight
   torch / CUDA pins. Score each written CIF with
   `ost compare-structures`, and score every supported protein
   interface with `DockQ` (`pip install DockQ`). Every row gets a
   `dockq_status`; RNA/DNA interfaces are
   `unsupported_chain_types`. Never invent either number, and never
   write `0.0` DockQ for a monomer — that row is `null` /
   `single_chain`. See
   [environment.md](environment.md#step-7--install-openstructure-after-bioir-and-oss)
   and
   [environment.md](environment.md#step-8--install-dockq-interface-scorer).
1. **OSS uses its recommended inference kernels, and the config has
   to say so.** Do not strip an optimized backend "for fairness", and
   do not assume a shipped default is the recommendation; resolve the
   model profile and upstream kernel documentation. If the selected
   profile requires cuEquivariance, enable it in config—installing the
   wheel is not enabling the kernel. If it requires DeepSpeed
   Evoformer attention, build only that op from source
   (`DS_BUILD_OPS=0 DS_BUILD_EVOFORMER_ATTN=1`), non-editable and
   never `-e`, budgeting 5–10 min for the CUTLASS compile. See
   [environment.md](environment.md#oss-kernel-flags--installing-is-not-enabling)
   and [deepspeed-evoformer.md](deepspeed-evoformer.md).
   A fallback is a different column, and only with user approval.
1. **Warmup 1, measure 1.** One discarded forward (JIT / graph capture
   / `torch.compile`), then one timed forward. That single time is the
   headline. Do not average or take a median.
1. **Five samples before the full sweep.** After preparation, run
   BioIR and OSS eager on five samples spanning the residue range,
   run the fast synthetic direct-module compile probe, then run OSS
   `torch.compile` on the same five when the compile retry ladder
   succeeds. Show the measured results and ask whether to proceed. Do
   not launch the full sweep without the user's explicit approval
   ([five-sample smoke gate](workflow.md#five-sample-preparation-smoke-gate)).
1. **No hidden skips.** A missing sample, missing MSA, OOM, or crash is
   recorded. Do not `continue` past it silently.
1. **Implementation notes are mandatory.** Keep
   `$WORKDIR/implementation-notes.md` current as work happens.

## Approximate options

Some BioIR options are approximate: they change the sampled coordinates
instead of reproducing the exact path bit for bit. Protenix-v2
`edm_sampling_config.frame_reuse_period` is one. A profile marks each
such option as **default on** when the locked config is the one the option
was evaluated with. Rules:

1. **Default on for Protenix-v2.** The profile locks 200 diffusion steps,
   the schedule the option was evaluated with, so the BioIR arm runs with
   `frame_reuse_period = 4`. Build it from `Protenix.get_pretrained_config`,
   change only that field, and record the exception and the value in
   `bench_config.json`. The harness written in Phase 3 takes an `--exact`
   switch that leaves the pretrained config unchanged. Any other step count
   must run exact: the option rejects it.
1. **Run the exact arm as well.** Add one BioIR column with the option off
   (same seed, same inputs, same GPU), time it like every other column, and
   report the approximate column's speedup against it and against OSS.
1. **Judge fidelity on paired scores, not equality.** Score both BioIR
   columns with the same lDDT and DockQ scorers and report the paired
   per-sample difference (approximate minus exact) next to each speedup. Bit
   equality is required only of the exact column and of the option when off.
1. **Label the column.** Name the column `bioir_approx_<option>` and list
   the option and value in the table caption. Never present the approximate
   column as the exact BioIR result.

## MSA and template contract

Mapping the dataset's alignments and templates onto an OSS tree is
its own job, with its own silent failure modes. Full procedure,
per-tree mappings, and verification: [msa.md](msa.md) and
[templates.md](templates.md). The gates below are the short form.

Protein unpaired MSA is **required** for Boltz-1/2 and OpenFold3, and
for every AlphaFold2 / OpenFold2 key (`docs/ref/support-matrix.md`)
when the spec lists A3Ms. Paired MSA is used when the spec lists
`paired_msas`. RNA, DNA, and ligand chains carry empty `msas` —
that is correct; do not invent files.

The spec's templates are **custom templates**: every one is attached
on both sides, capped only by a config value set identically on each,
with the search-oriented filters turned off. Verify by counting
populated slots on the featurized batch rather than reading config —
a tree may need a query→template alignment rather than the structure
the dataset ships, and a fixed slot layout hides a partial drop
([templates.md](templates.md#verification)).

**BioIR load (processor path).** Convert each in-scope spec item to an
`InputRequest` with paths resolved against
`benchmarks/dataset/`
([samples.md](samples.md#load-path-processor)). Do not call
`load_requests` on `spec_*.json`. Attach every listed A3M and
template.

**OSS load (both paths).** Map the same spec item to the OSS
inference script. Boltz: `boltz_yaml` + `boltz_msa_csv`, rewriting
stale absolute `msa:` paths to `$DATASET_ROOT/casp15/msa/`. Other
OSS trees: spec polymers + resolved A3Ms. Do not call
`build_processor` on the OSS feature path.

**After load, before any forward, assert all of:**

1. Every spec `msas` / `paired_msas` (and Boltz CSV) path exists on
   disk under `$DATASET_ROOT`.
1. Every protein polymer whose spec lists MSAs has a non-empty
   `MSARecord` list on the BioIR `InputRequest` (processor path) and
   non-empty alignments on the OSS featurizer input.
1. Every A3M the spec lists for that item is attached (unpaired and
   paired). A listed file that is skipped is a hard failure.
1. The OSS adapter consumes those same alignments. No ColabFold /
   MSA server. No FASTA-only rewrite that drops alignments.
1. Every template the spec lists is attached on both sides:
   populated template slots equal `min(n_supplied, cap)` on each,
   with the same entry and chain per slot
   ([templates.md](templates.md#verification)).

**Forbidden:**

- Feeding `examples/data/samples/` or demo FASTA instead of the
  release specs
- `--use_msa_server` / `use_msa=False` / empty protein MSA when the
  spec lists files
- Timing a no-MSA protein run and comparing it to a with-MSA run
- Using the shipped Boltz YAML's original absolute `msa:` path when
  it does not point at `$DATASET_ROOT`

## Phases

Run the phases in order. Each names what to read, what it must leave in
`$WORKDIR`, and the check that ends it; do not start a phase before the
previous exit check passes. Harnesses live in `$WORKDIR`; do not modify
`bionemo_ir/` or `$OSS_ROOT`.

### Phase 0 — WORKDIR, environments, GPU, OSS root

- **Read:**
  [workflow.md](workflow.md#phase-0--workdir-environments-gpu-oss-root),
  [environment.md](environment.md), and the model profile under `models/`.
  OpenFold2 / OpenFold3 also need
  [deepspeed-evoformer.md](deepspeed-evoformer.md).
- **Produce:** the `$WORKDIR` tree, `$OSS_ROOT` at the pin, the BioIR and OSS
  interpreters, `envs/ost` and `envs/dockq`, `ref_data/gpu_inventory.csv`,
  and the environment fields plus `feature_path` (`processor` | `oss`) in
  `ref_data/bench_config.json`.
- **Exit:** both sides import on the same GPU, and the checkpoint
  strict-loads once in each interpreter from a recorded local path with the
  hub untouched
  ([weights preflight](workflow.md#weights-preflight--prove-one-load-before-any-timed-run)).
  A non-folding key stops here.

### Phase 1 — Manifest + locked config

- **Read:** [workflow.md](workflow.md#phase-1--manifest--locked-config),
  [samples.md](samples.md#manifest-fields),
  [measurement.md](measurement.md#locked-config), and the profile's runtime
  lock.
- **Produce:** `ref_data/sample_manifest.json` with every spec item, the
  in-scope filter by model capability only, and every A3M and template
  resolved; `ref_data/bench_config.json` with the locked runtime args, seed,
  precision, GPU, dataset digests, BioIR default constructor and graph
  activation, OSS entry script, and compile plan.
- **Exit:** the OSS script's expanded effective config matches the lock. The
  config is read-only from here.

### Phase 2 — Map the dataset into `oss_data`

- **Read:** [workflow.md](workflow.md#phase-2--map-the-dataset-into-oss_data),
  [msa.md](msa.md), [templates.md](templates.md), and the profile's mapping
  section.
- **Produce:** `bench/stage_oss_data.py`, an `oss_data/` tree of symlinks, and
  `oss_data/index.json` per sample and chain.
- **Exit:** a rerun is idempotent and silent, every declared input is staged
  with one distinct key per chain, and one monomer and one multi-chain sample
  are checked by hand.

### Phase 3 — Write the harnesses

- **Read:** [workflow.md](workflow.md#phase-3--write-the-harnesses) and
  [measurement.md](measurement.md). Copy the shared helpers from
  [`harness/`](harness/) into `bench/`; do not rewrite them. The OSS feature
  path also needs [no-pipeline.md](no-pipeline.md).
- **Produce:** on the processor path, `bench/run_bioir.py` and
  `bench/run_oss.py`; on the OSS feature path, `bench/dump_oss_features.py`
  plus both runners. Every loop shows a progress
  bar outside the timing window.
- **Exit:** the
  [five-sample smoke gate](workflow.md#five-sample-preparation-smoke-gate)
  passes, its results are shown, and the user explicitly approves the full
  sweep.

### Phase 4 — Run BioIR

- **Read:** [workflow.md](workflow.md#phase-4--run-bioir).
- **Produce:** `results/bioir_forward.json` and logs under `debug/`. The OSS
  feature path writes the feature dumps first. Run with `$BIOIR_PYTHON` and
  `CUTEDSL_FORCE_CUBIN=1`.

### Phase 5 — Run OSS e2e

- **Read:** [workflow.md](workflow.md#phase-5--run-oss-e2e).
- **Produce:** `results/oss_eager.json` always, and `results/oss_compile.json`
  only after a passing probe or retry. Same manifest, locked knobs, and GPU; the
  OSS feature path loads the Phase 4 dumps. Run with `$OSS_PYTHON`. An OSS eager
  kernel crash blocks that backend; never relabel a fallback as `oss_e2e`.

### Phase 6 — Compare and report

- **Read:** [workflow.md](workflow.md#phase-6--compare-and-report) and
  [measurement.md](measurement.md#comparison-table).
- **Produce:** `results/speedup.json` and both charts, then print setup,
  dataset, the latency table, speedup geomean and median overall and per
  residue bin, lDDT, DockQ, GPU power and clocks from in-forward samples,
  commands, and the path to `implementation-notes.md`. Show the chart PNGs
  inline.
- **Exit:** `harness/check_results.py` passes on every result file, every
  number cites an artifact, and latency is in seconds to three decimals.

## Optional — compact a portable `bioir-perf` package

Skip unless the user asks to pack completed benches so they can run
on another GPU SKU or in a fresh container. Follow
[bundle.md](bundle.md).

Packing copies locked harnesses into `bioir-perf/` (host
`bench.sh`, in-container `bench.py`, one directory per model). Do
not name the pack `release/`. Packing is not done until a GPU
smoke of **all** models passes: BioIR, OSS eager, and OSS
`torch.compile` (Protenix never compiles). That smoke is the
harness-bug gate, not a full-dataset SKU sweep. Required pack
outputs are JSON and Markdown (`speedup.json`,
`comparison_report.md`, plus per-column JSON). Supported folding
keys: `boltz2`, `of3`, `protenix`, `of2`.

## Boltz-2 — MSA deletion bug

The pinned OSS `v2.2.1` featurizer zeroes non-query MSA deletions, and BioIR
does not. A large BioIR vs OSS lDDT gap on MSA-bearing proteins may be that,
not the engine. Never zero BioIR deletions to match; triage it with
[models/boltz2.md](models/boltz2.md#pinned-v221-deletion-divergence).

## Other model kinds

v1 is a **folding** bench. If the user names a non-folding key
(`boltz-2-affinity`, embeddings, design), stop and extend this skill (or a
sibling) before writing harnesses. What carries over and what each new kind
must redefine is in
[workflow.md](workflow.md#extending-to-other-model-kinds).

## Out of scope (v1)

- Nsight Systems / Nsight Compute, MFU, SOL%, kernel breakdown
- Ray replica throughput (large-dataset path; not this spec)
- Template **search** (HHsearch / HMMsearch); templates are attached
  from the spec's shipped structures only
  (see [samples.md](samples.md#templates-are-in))
- The demo tree `examples/data/samples/` as a substitute dataset
- DockQ / feature-tensor equivalence (`make-data-pipeline`)
- Isolated same-features microbench of a single stack (`module-onboard`)
  — the OSS feature path is e2e `model.forward` on OSS features, not a
  layer bench
- Non-folding heads (`boltz-2-affinity`, embeddings, design, …)
  until this skill (or a sibling) is extended

Those are separate skills or a later revision.

## Key gotchas

Each of these has cost a full run. The gates above cover the rest.

- **A package name is not a model variant.** Strict-load the same checkpoint
  on both sides; a non-strict or key-rewriting load is model onboarding
  ([weights preflight](workflow.md#weights-preflight--prove-one-load-before-any-timed-run)).
- **Load the checkpoint once before Phase 4.** A gated repo fails every
  sample in the loop.
- **Launch each harness with its own python** — `$BIOIR_PYTHON` or
  `$OSS_PYTHON`, never a bare `python` after Phase 0.
- **`CUTEDSL_FORCE_CUBIN=1` on every BioIR process.** JIT compile time makes
  forward latency unstable.
- **No `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, ever** — it can
  slow `forward`
  ([measurement.md](measurement.md#allocator-flags--leave-pytorch-on-its-default)).
- **CUDA 12 packages → CUDA 13.** Install the `cu13` counterpart of any
  `cu12` pin ([environment.md](environment.md#cuda-12-packages--cuda-13)).
- **Processor path: `profile_inference` already isolates `model.forward`.** Do
  not time `processor()` wall clock and call it the model.
- **H2D is outside the window on both sides.** OSS must move the batch first.
- **Seed goes on the feature-generator stage**, not the tokenizer
  (`docs/ref/api.md`).
- **Paired A3Ms are per chain, same row count** (`docs/ref/api.md`).
  Dropping one side's paired file changes the forward.
- **Let DockQ choose the chain mapping** and record it; a monomer's DockQ is
  `null`, not `0.0`.
- **BioIR graphs are on by default.** The default model needs no
  `optimize()` call; an eager arm uses `disable_cuda_graphs()`. Keep the
  policies and caches, and audit each forward
  ([measurement.md](measurement.md#cuda-graphs)).
- **Sharing the GPU with another job invalidates the run.**
