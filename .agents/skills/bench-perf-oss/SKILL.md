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
See [Later: other model kinds](#later-other-model-kinds).

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

- **Path A — `build_processor`.** The model has a factory (Boltz-1/2,
  OpenFold2 / AlphaFold2, OpenFold3). BioIR uses serial
  `build_processor`; OSS uses its own e2e. Each side featurizes itself.
- **Path B — OSS pipeline + BioIR module.** The folding model has a
  compute path and no factory (`protenix-v2`). Use the OSS data
  pipeline for features, then replace the OSS `nn.Module` with the
  BioIR module. See [no-pipeline.md](no-pipeline.md).
  `boltz-2-affinity` is **not** a folding bench — see
  [Later: other model kinds](#later-other-model-kinds).

Authorities:

- Production API — [`docs/ref/api.md`][api] (`build_processor`,
  `profile_inference`, `runtime_args`)
- Serial vs Ray — [`docs/ref/architecture.md`][arch]
- Model keys and MSA rules — [`docs/ref/support-matrix.md`][sm]
- Checkpoints — [`docs/ref/model-weights.md`][weights]
- Dataset — [samples.md](samples.md) (built by `rebuild_dataset.py`)
- Environment — [environment.md](environment.md)
- Timing protocol — [measurement.md](measurement.md)
- Sample catalog — [samples.md](samples.md)
- MSA mapping (dataset -> OSS) — [msa.md](msa.md)
- Template mapping (dataset -> OSS) — [templates.md](templates.md)
- No BioIR pipeline — [no-pipeline.md](no-pipeline.md)
- DeepSpeed evoformer (OF2 / OF3) —
  [deepspeed-evoformer.md](deepspeed-evoformer.md)
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
1. **BioIR uses the default model constructor.** Every class
   does `self.config = config or self.get_pretrained_config(
   self.model_name)`. That is the default optimized config
   (dtypes, triangle / pairwise backends). Construct with
   `config=None` (omit the kwarg). Path A: omit
   `engine_kwargs["config"]` so the engine takes the same
   constructor path. For Boltz-1/2, OpenFold3, and Protenix, select
   the diffusion module with `AcceleratedConfig(backend="torch")`
   **without** `default=`. The module-declared safe CUDA-graph routine
   accepts at most 1024 tokens and falls back to eager above that
   limit. An explicit `CUDAGraphOptimizationConfig` replaces that
   routine instead of merging with it, removing the safety limit.
   Never pass a handmade `BaseConfig` or override the module's graph
   config. Do not copy OSS layer configs onto BioIR. A model profile
   may require one feature enable on top of the official pretrained
   config—for example Boltz-2 custom templates require
   `trunk.use_templates_v2=True`. In that case call
   `get_pretrained_config`, change only the documented flag, pass that
   config, and record the exception. See [`docs/ref/api.md`][api].
1. **Serial, one GPU, one sample per call.**
   `executor_backend=None`. Pin `CUDA_VISIBLE_DEVICES` to one device.
   Loop the in-scope manifest and call `processor([one_record])` (or
   OSS featurize + `forward` on that one batch). Never
   `processor(all_rows)` — feature generation then holds every
   record in host RAM and OOM-kills before the engine. See
   [measurement.md](measurement.md#executor-serial-not-ray).
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
   ([five-sample smoke gate](#five-sample-preparation-smoke-gate)).
1. **No hidden skips.** A missing sample, missing MSA, OOM, or crash is
   recorded. Do not `continue` past it silently.
1. **Implementation notes are mandatory.** Keep
   `$WORKDIR/implementation-notes.md` current as work happens.

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

**Load path (Path A).** Convert each in-scope spec item to an
`InputRequest` with paths resolved against
`benchmarks/dataset/`
([samples.md](samples.md#load-path-path-a)). Do not call
`load_requests` on `spec_*.json`. Attach every listed A3M and
template.

**Load path (Path B / OSS).** Map the same spec item to the OSS
inference script. Boltz: `boltz_yaml` + `boltz_msa_csv`, rewriting
stale absolute `msa:` paths to `$DATASET_ROOT/casp15/msa/`. Other
OSS trees: spec polymers + resolved A3Ms. Do not call
`build_processor` on Path B.

**After load, before any forward, assert all of:**

1. Every spec `msas` / `paired_msas` (and Boltz CSV) path exists on
   disk under `$DATASET_ROOT`.
1. Every protein polymer whose spec lists MSAs has a non-empty
   `MSARecord` list on the `InputRequest` (Path A) or the OSS
   featurizer input (Path B / OSS).
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

## Phase 0 — WORKDIR, environments, GPU, OSS root

Ask for `$WORKDIR` (default `workdir/bench_perf_<model>/` at the repo
root). Resolve `$OSS_ROOT` from the
[OSS checkout pins](environment.md#oss-checkout-pins) — do not
guess, and do not clone `main`.

- OpenFold3: follow its checkpoint, source, data, and patch profile in
  [models/of3.md](models/of3.md).
- Boltz-2: follow its source pin, template enablement, MSA, and kernel
  profile in [models/boltz2.md](models/boltz2.md).
- OpenFold2 / AlphaFold2: follow the protein-only AF2 model-1
  checkpoint, recycle, MSA/template, and Evoformer-compile profile in
  [models/of2.md](models/of2.md). WORKDIR default `/tmp/openfold2`.
- Protenix-v2: follow the Path B source, checkpoint, recycle mapping,
  and input profile in [models/protenix.md](models/protenix.md).
  WORKDIR default `/tmp/protenix`.
- Boltz-1/2: [jwohlwend/boltz](https://github.com/jwohlwend/boltz) `v2.2.1`
- OpenFold2:
  [aqlaboratory/openfold `v2.2.0`](https://github.com/aqlaboratory/openfold/tree/v2.2.0)

```bash
WORKDIR=${WORKDIR:-workdir/bench_perf_<model>}
mkdir -p "$WORKDIR"/{bench,ref_data,oss_data,results,debug,venv_oss,envs,oss}
```

```text
$WORKDIR/
├── oss/                        # Boltz / OpenFold2 clones at the pinned tag
├── bench/                      # BioIR + OSS harnesses (written here)
├── venv_oss/                   # OSS interpreter when isolation is two_venv
├── envs/ost/                   # dedicated OpenStructure conda env
├── envs/dockq/                 # dedicated DockQ venv
├── miniforge3/                 # bootstrapper only; do not install ost in base
├── ref_data/
│   ├── sample_manifest.json    # canonical sample set + MSA paths
│   ├── bench_config.json       # locked runtime / GPU / checkpoint / pythons
│   ├── gpu_inventory.csv
│   ├── env_bioir.txt           # pip freeze
│   ├── env_oss.txt
│   └── env_conflicts.md
├── oss_data/                   # dataset re-shaped for the OSS loader only
│   ├── msa/                    # alignments regrouped / renamed as it wants
│   ├── templates/              # template alignments or structures likewise
│   ├── queries/                # generated per-sample input files
│   ├── cache/                  # anything the OSS preprocessor writes
│   └── index.json              # generated path -> dataset source, per chain
├── results/
│   ├── bioir_forward.json
│   ├── oss_eager.json              # always
│   ├── oss_compile.json            # only if compile probe passed
│   ├── latency_vs_residues.png     # Phase 6 chart (required)
│   ├── speedup_vs_residues.png     # Phase 6 chart (required)
│   └── speedup.json                # geomean + median, overall and bins
├── implementation-notes.md
└── NOTES.md
```

Harnesses live in `$WORKDIR`. Do not modify `bionemo_ir/` or `$OSS_ROOT`.

`oss_data/` holds the benchmark dataset re-shaped to whatever layout and
naming the OSS loader demands. What that means is per tree — one
directory per chain, a fixed set of recognized filename stems, a native
YAML / CSV / JSON query, a synthesized alignment, a preprocessor cache —
so treat the subdirectories above as a starting shape and keep the index
JSON, which is the part that makes the mapping auditable. Build it with a
script (`bench/stage_oss_*.py`) rather than by hand.

Three rules keep it honest:

- **Symlink, never copy or edit.** Entries point at
  `$DATASET_ROOT`. Same bytes on both sides, and nothing is written into
  the dataset. Point every OSS output directory (caches included) here,
  since some preprocessors create directories next to their inputs.
- **BioIR keeps reading the original dataset paths.** `oss_data/` exists
  because the OSS loader needs a shape, not because the inputs differ. If
  a reshape would change *content* (a parse-time row cap that truncates
  what BioIR reads in full, a template the OSS filters reject), that is a
  parity break: fix it or record it, do not paper over it.
- **Assert the mapping, don't assume it.** Loaders select inputs by
  filename stem and directory name, and the common failure is silent —
  files skipped, or chains collapsed onto one alignment. Verify on the
  featurized batch ([msa.md](msa.md), [templates.md](templates.md)).

### Environments (do this before GPU timing)

Follow [environment.md](environment.md) in order. Do not skip to a
harness because "python is already there".

1. **Container + BioIR install** — default image
   `nvcr.io/nvidia/pytorch:26.05-py3`. **Always**
   `export CUTEDSL_FORCE_CUBIN=1` (CUBIN path; no CuTeDSL JIT on the
   clock). Developer mode: `git lfs pull` the cubin packs, then
   `pip install -e '.[dev]'` — no `--no-build-isolation`, which skips
   the very build requirements (`cmake`, `nanobind`, `setuptools`)
   that the extension needs. Otherwise install the wheel
   (`dist/bionemo_ir-*.whl`) or `pip install bionemo-ir`.
   See [environment.md](environment.md#container-and-bioir-install).
1. **Probe BioIR** — `import bionemo_ir`, CUDA, `_cutedsl_kernels`
   under `CUTEDSL_FORCE_CUBIN=1`. Path A also imports
   `build_processor`. Path B imports the module class
   (`from bionemo_ir.models.protenix import Protenix`).
1. **Probe OSS** — `$OSS_ROOT` is the pin
   ([environment.md](environment.md#oss-checkout-pins)). Verify
   `oss_commit` / tag, then import the e2e package. Read its
   pins. Do not install them yet.
1. **Diff freezes** — classify hard vs soft conflicts
   (torch / CUDA / cuEq / DeepSpeed / numpy major). `flash-attn` is
   optional — uninstall it from either env; do not treat it as a
   conflict.
1. **Set up** — BioIR freeze is the lock. Two venvs when OSS cannot
   run on those pins (`$WORKDIR/venv_oss`). Shared env only when the
   freeze diff is clean; then install OSS with `--no-deps` or
   `PYTHONPATH` so it reuses BioIR packages.
1. **Resolve OSS last** — never change a BioIR package to satisfy
   OSS. If a shared-env install breaks `import bionemo_ir`, revert
   BioIR and split. Keep OSS-only kernels **in the OSS env**.
1. **CUDA 12 wheels → CUDA 13.** Any OSS pin / extra / index that
   names `cu12` or `cuda12` is installed as the `cu13` /
   `cuda13` counterpart. Do not install the CUDA 12 artifact.
   See
   [environment.md](environment.md#cuda-12-packages--cuda-13).
1. **OSS apt packages** — if OSS needs distro libs (RDKit `.so`,
   compilers for DeepSpeed evoformer, …), scan for **every**
   missing soname in one pass, generate
   `$WORKDIR/ref_data/apt_install.sh`, then try `apt-get` and
   passwordless `sudo -n apt-get`. If the agent cannot run either
   (no root, no `sudo`, or a policy that blocks them), **stop**,
   show the generated `sudo apt-get install -y …` command inline,
   and ask the human to run it. Never discover packages one
   `ImportError` at a time. See
   [environment.md](environment.md#oss-system-packages-apt).
1. **OpenFold2 / OpenFold3: source-build DeepSpeed evoformer_attn
   only.** `DS_BUILD_OPS=0 DS_BUILD_EVOFORMER_ATTN=1` into
   `$OSS_PYTHON`. Do not build other DeepSpeed kernels. Do not
   use a PyPI wheel, and do not use `pip install -e` — it exits 0
   having compiled nothing. Budget 5–10 min, then gate on
   `installed_ops` before any forward. See
   [deepspeed-evoformer.md](deepspeed-evoformer.md).
1. **Dry import both sides** on the same visible GPU. Save probes
   under `$WORKDIR/debug/`.
1. **Install OpenStructure last** — after BioIR and OSS are in
   place. Create a **dedicated conda env** (default
   `$WORKDIR/envs/ost`) and install `openstructure` there from
   **`bioconda`** (with `conda-forge` for its dependencies; it is
   not on conda-forge alone). Any existing conda may bootstrap it,
   but never reuse an `ost` from a base env, and never install into
   BioIR or OSS. See
   [environment.md](environment.md#step-7--install-openstructure-after-bioir-and-oss).
1. **Install DockQ** — `pip install DockQ` into its **own venv**
   (default `$WORKDIR/envs/dockq`), never into `$BIOIR_PYTHON` or
   `$OSS_PYTHON`: it pins `numpy < 2.0` and would downgrade numpy
   under the frozen BioIR env. It builds a Cython extension, so it
   needs a C compiler and pip's default build isolation. Probe with
   `--help`; there is no `--version`. See
   [environment.md](environment.md#step-8--install-dockq-interface-scorer).

Record `bioir_python`, `oss_python`, both `torch.__version__` strings,
`isolation` (`two_venv` | `shared`), `ost_cmd`, `ost_version`,
`dockq_cmd`, `dockq_version`, `dockq_args`,
`oss_kernels`, `apt_packages`, `apt_install_script`, and
`cuda12_remaps` in `bench_config.json`.

### GPU inventory

Save before any timed call. See
[measurement.md](measurement.md#gpu-inventory).

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version,uuid,power.limit,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.graphics,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,clocks.current.sm,clocks.current.graphics,clocks.current.memory,power.draw,pstate,clocks_event_reasons.active,clocks_event_reasons.gpu_idle,clocks_event_reasons.sw_power_cap,persistence_mode \
  --format=csv
"$BIOIR_PYTHON" -c "import torch, bionemo_ir; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), bionemo_ir.__file__)"
"$OSS_PYTHON" -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

Power limit and clock rates are part of the inventory, not optional.
Save the CSV as `$WORKDIR/ref_data/gpu_inventory.csv`. Idle SM clocks
in that CSV are nameplate context only — they are often a few hundred
MHz. Harnesses sample power and SM / graphics / memory clocks
**during** every measured forward
([measurement.md](measurement.md#gpu-inventory)).

Look up `model_source` in `docs/ref/support-matrix.md`. Set
`path` in `bench_config.json`:

- Pipeline **Yes** → Path A (folding factory)
- Pipeline **No** and folding (`protenix-v2`) → Path B
  ([no-pipeline.md](no-pipeline.md)). Do not call `build_processor`.
- Pipeline **No** and not folding (`boltz-2-affinity`) → stop.
  Extend the skill first
  ([Later: other model kinds](#later-other-model-kinds)).

### Weights preflight — prove one load before any timed run

Resolve and load the checkpoint **once, offline, before Phase 4**.
Many of these checkpoints sit behind gated or authenticated repos, so
a harness that reaches the hub per sample can fail every sample in
turn and leave a full results file of zeros — after the whole
environment build. Never discover that from inside the bench loop.

First prove that the pin and the BioIR key still name the **same model
variant**. Read the OSS checkpoint registry / compatibility spec and
its current default checkpoint, then compare that with BioIR's
resolved checkpoint. A source release can retire the checkpoint BioIR
still ships or switch its default to a new architecture while keeping
the same package and CLI name. An explicit file path may bypass the
registry check, so validation success is not enough: the subsequent
strict state-dict load is the gate.

If the OSS pin rejects the BioIR checkpoint, or its current checkpoint
has a different architecture / training variant, stop. Do not average
per-block weights into a new shared parameter, add or delete keys, use
`strict=False`, or call two different checkpoints "the same model".
That is model onboarding, not a benchmark input patch. Either onboard
the new variant into BioIR or, with the user's explicit choice, change
the benchmark's declared OSS revision and model variant.

State-dict compatibility is necessary but not sufficient. Compare
non-weight semantics changed by the pin too: tensor orientation,
normalization placement, feature bookkeeping, inference presets, and
default gates. If BioIR combines an old checkpoint with behavior
ported from a newer incompatible release, it is a third, unvalidated
variant; changing only the OSS revision cannot make that comparison
apple-to-apple.

Stage weights with `scripts/fetch_weights.sh --model <family>` when
the hub cannot resolve them, or point BioIR at a checkpoint already
on disk (`BIOIR_CHECKPOINTS=<root>` scans `<root>/<model>/*.pt`; the
per-family variable named by the hub's default repo id takes a direct
file path — `bionemo_ir/hubs/local.py`). Repo checkouts often already
carry the file under `examples/<model>/checkpoints/`; look there
before downloading anything.

The gate: one construct-and-load in the bench interpreter, with the
network unavailable or the hub untouched, logging the resolved local
path, followed by the same strict load in the OSS interpreter. A 401 /
`GatedRepoError`, incompatible-version declaration, missing /
unexpected state-dict keys, or any per-sample hub request is a setup
failure — fix it before timing, and do not paper over it with a token
or a non-strict load in the harness. Record both checkpoint paths,
hashes, registry names, and declared compatibility ranges in
`bench_config.json`. On Protenix pass `include_load_weights=True`.

## Phase 1 — Manifest + locked config

### Sample manifest

Build or reuse the dataset
at `benchmarks/dataset/`
([samples.md](samples.md#build)). Verify
`TARBALL.sha256` and `MANIFEST.json`. Never put a GitHub token in
the harness or notes.

Pick the spec ([samples.md](samples.md#which-spec)). Enumerate
**every** item in that spec (not only in-scope rows). Record:

- `sample_id` (spec `id`), `spec`, `chemical_class`, `seq_len`,
  `token_bin`
- `residues` — sum of all protein / RNA / DNA chains (see
  [measurement.md](measurement.md#result-json)); must equal
  spec `seq_len`
- per-polymer `polymer_type`, `chain_id`, sequence length
- resolved unpaired and paired A3M paths (against `$DATASET_ROOT`)
- `boltz_yaml` / `boltz_msa_csv` when present
- `msa_status`: `attached` | `none_declared` | `missing_file`
  (blocker)
- `gt_path` — `$DATASET_ROOT/<spec gt>`, or `null`
- `has_templates`, `templates[]` (resolved paths + `chain_id`),
  `template_status`
  ([samples.md](samples.md#templates-are-in))
- `in_scope` — see filters below

Filter to what the model can run
(`docs/ref/support-matrix.md`):

- AF2 / OF2 monomer — `spec_monomer.json` (protein monomers only)
- AF2 multimer — `protein-protein` items from `spec_full.json`
  whose polymers are all protein (no RNA / DNA / ligand). See
  [models/of2.md](models/of2.md#protein-only).
- Boltz-1/2, OpenFold3, Path B (`protenix-v2`) — `spec_full.json`

Every filter is about model capability, never about templates. Attach
each item's listed A3Ms / Boltz CSVs and templates.

Write `$WORKDIR/ref_data/sample_manifest.json`. A later harness that
processes a different set is a hard failure.

Default is every **in-scope** spec item, templates included. The user
may narrow it further; do not silently drop RNA/DNA/ligand or
template-bearing samples on Boltz / OpenFold3.

### Locked inference config

**Start from the OSS inference script.** The OSS repo almost
always ships one (`predict.py`, `infer.py`, `run_predict.py`,
`main.py predict`, a CLI entry in `pyproject.toml`). Find it
first. Copy or import that path into `$WORKDIR/bench/` and
build the harness **on top of it** — model load, featurizer,
writer, default flags. Do not rewrite the OSS e2e stack from
the library internals. Record `oss_entry` (path + how you
invoke it) in `bench_config.json`.

Print the resolved OSS config from that script and the BioIR
`get_default_runtime_args(model_source)`. Lock the union in
`$WORKDIR/ref_data/bench_config.json`.

A preset name is not a locked config. Serialize its **expanded,
effective** memory, chunking, offload, kernel, and inference-head
values too. Presets can keep the same name while changing a chunk
ceiling or backend default between pins, which invalidates old timing
and memory rows even when every command-line argument is unchanged.

BioIR `runtime_args` names (`docs/ref/api.md`):

- Boltz-1/2, OpenFold3 — `recycling_steps`, `num_sampling_steps`,
  `diffusion_samples` (OpenFold3 maps those onto cycle / rollout)
- Protenix-v2 — `recycling_steps`, `num_sampling_steps`,
  `diffusion_samples`; this bench locks `recycling_steps=5`
- OpenFold2 / AlphaFold2 — `recycling_steps` only. This bench locks
  `recycling_steps=3` as **exactly three trunk iterations** on both
  sides, with early stopping off. See
  [models/of2.md](models/of2.md#recycle-lock--three-trunk-iterations-no-early-stop).
  Do not leave the multimer default of 20 iters / 0.5 CA-distance
  stop, and do not treat OF2 `recycling_steps` as the AF3-style
  `+ 1` cycle count.

**AF3-style folding config is fixed for this bench.** For Boltz-1,
Boltz-2, and OpenFold3, lock:

```json
{
  "recycling_steps": 3,
  "num_sampling_steps": 200,
  "diffusion_samples": 5
}
```

For Protenix-v2, lock:

```json
{
  "recycling_steps": 5,
  "num_sampling_steps": 200,
  "diffusion_samples": 5
}
```

Use these exact BioIR runtime-argument names. Map the OSS script's
equivalent names onto the same semantics; do not leave its defaults
at another recycle, diffusion-step, or sample count. OpenFold3's
internal `num_cycles` mapping is handled by its `forward`; pass
`recycling_steps=3` to BioIR. For Protenix-v2, pass
`recycling_steps=5` to BioIR and set OSS `model.N_cycle=6`; see
[models/protenix.md](models/protenix.md#runtime-lock-and-recycle-semantics).

Also lock `dataset_root`, `dataset_spec`, `dataset_manifest_sha256` and
`dataset_build_sha256` (BUILD.json — a built tree is not identified by a version
string; see [samples.md](samples.md#provenance--record-the-build-not-a-tag)),
precision, seed, `CUDA_VISIBLE_DEVICES`, warmup/repeats, checkpoint ids,
`bioir_config="get_pretrained_config"`, `accelerated_configs`
(or `null`), `path` (`A` | `B`), `ost_cmd`, `dockq_cmd`, and
`dockq_args`. Path A seeds via
`init_context.random_seed` on the **feature-generator** stage. Path B
seeds the OSS featurizer the way the OSS e2e script does.

OpenFold2 / AlphaFold2 use the same `recycling_steps=3` lock as a
fixed trunk-iteration count (not `+ 1`, not early stop). Details
and the two-field pretrained-config delta are in
[models/of2.md](models/of2.md#recycle-lock--three-trunk-iterations-no-early-stop).
For every family, seed and all supported inference knobs must match
between BioIR and OSS.

**BioIR model config.** Use the default constructor:
`ModelCls(model_name=..., config=None)`. The class fills
`self.config` from `get_pretrained_config(model_name)` — the
default optimized stack (bf16 where the class sets it, auto
triangle / pairwise backends,
`docs/ref/support-matrix.md`). Path A omits
`engine_kwargs["config"]` so `FoldingEngine` does the same.
Do not construct a `BaseConfig` by hand. Do not pass OSS
dtypes or attention backends into BioIR. The only exception is a
minimal feature flag explicitly required by a model profile for input
parity: derive the official pretrained config, change only that flag,
and serialize the delta. Boltz-2's custom-template case is documented
in [models/boltz2.md](models/boltz2.md).

**BioIR CUDA graphs (default on).** After that pretrained config,
for `boltz-1`, `boltz-2`, `openfold3`, and `protenix-v2`, enable a
CUDA graph on `diffusion_module` by selecting only
`AcceleratedConfig(backend="torch")`. Omit `default=` so the
module's safe, exact-shape routine remains active: it accepts
`num_tokens <= 1024` and routes larger inputs to eager. Do not pass
an explicit `CUDAGraphOptimizationConfig`; it replaces rather than
merges with the module default and would erase the routing metadata
and 1024-token guard. Graph that parent only — `token_transformer`
is nested and cannot have its own graph. OpenFold2 / AlphaFold2 have
no graphable module; leave `accelerated_configs` unset.

Audit graph execution from the declared input-routing policy before
reading tracker caches. An out-of-range call bypasses cache-state and
fallback-key creation, so both counters are expected to stay zero; call
that `eager_out_of_range`, not a capture failure. Accepted inputs with a
failed key are `eager_capture_fallback`, and accepted inputs with a graph
state are `cuda_graph`. Audit and serialize the classification before
calling `tracker.reset()` after each sample. Use the reusable procedure
in
[measurement.md](measurement.md#audit-cuda-graph-routing-not-cache-emptiness).

**OSS scenarios.** Always run **eager**. Protenix-v2 is an explicit
compile exception: its pinned combined child compile was prohibitively
slow and collapsed downstream lDDT/DockQ despite stable measured
forwards. Skip that compile column unless a user requests a reprobe
after relevant pins change; see
[models/protenix.md](models/protenix.md#oss-compile-status--skipped).

For other model profiles, compile the **target submodules once** on a
fresh OSS model (OF2: Evoformer; AF3-style: Pairformer +
DiffusionModule — never the whole module). Use
`torch.compile(module)` exactly: omit `dynamic`, leaving its documented
`None` default so Dynamo begins specialized and automatically widens
after shape changes. Do not set `dynamic=True` or `dynamic=False`, and
do not apply manual dynamic/static input marks.

Probe each child with synthetic direct inputs in the sequence
shape-A, shape-A, shape-B, shape-B
([measurement.md](measurement.md#oss-torchcompile)). The first call at
each new shape may compile as the default policy adapts; each immediate
repeat must not. Reproduce the child's production autocast-enabled or
autocast-disabled context in every direct call. Then run the smallest
real sample and a second
different-shape sample. If the probe raises, produces NaN/Inf, or
recompiles on an immediate repeat, walk the
[retry ladder](measurement.md#compile-retry-ladder) without changing
the default dynamic policy. Child-tensor drift versus eager is
expected under `torch.compile` and is not a probe failure; judge
downstream fitness with lDDT and DockQ. If a retry works, **reuse that same
compiled model** for every in-scope sample. Count compile events on
every warmup and measured forward. Later-sample warmup
specializations are reported, not hidden; measured-forward recaptures
fail the compile column. Never drop eager because compile worked and
never `torch.compile` again per sample.

## Phase 2 — Map the dataset into `oss_data`

The dataset is shaped for BioIR's schema. The OSS loader almost never
reads it as-is, and when it disagrees it tends to skip inputs rather
than raise. Do this mapping **before** writing the harnesses, in one
script, so the harness consumes an index instead of guessing layout.

Write `bench/stage_oss_data.py` (or one script per input kind). For
each in-scope sample and each chain it must:

1. **Resolve** the inputs the manifest already recorded — alignments,
   templates, ground truth — against `$DATASET_ROOT`.
1. **Re-shape** them into what the loader accepts, following
   [msa.md](msa.md#3-learn-how-the-loader-selects-and-keys-files) and
   [templates.md](templates.md): the directory that carries chain
   identity, the filename stems the parser recognizes, the native query
   file, the synthesized alignment.
1. **Symlink** rather than copy, so both sides read the same bytes and
   `$DATASET_ROOT` stays untouched. Generate a file only when the
   parser genuinely cannot read the shipped format, and record what it
   was generated from.
1. **Assert what would otherwise pass silently** — every declared input
   staged, one distinct key per chain, and file depth within any
   parse-time cap the loader applies before the other side's global cap
   ([msa.md](msa.md#failure-modes-seen-in-practice)).
1. **Write `oss_data/index.json`** keyed by sample and chain: source
   path, staged path, and the numbers you asserted on (rows, rows used,
   template ids, selected chain). This is what the report cites and
   what the next run reuses instead of re-deriving.

Point every OSS output directory (preprocessor caches, parsed-structure
dumps) into `oss_data/cache/`. Some preprocessors default to a temp
directory or to the input's parent, which either vanishes between runs
or writes into the dataset.

Rerun the script until it is idempotent and silent. Then check the
staged tree once by hand — one monomer and one multi-chain sample,
confirming distinct per-chain inputs — because a mapping bug here looks
exactly like a fast model.

## Phase 3 — Write the harnesses

Read [measurement.md](measurement.md) and copy the timer verbatim.
Write **one** of the two blocks below, matching `path` from Phase 0.

Both harnesses have to turn spec items into their side's input. That
mapping — field names, enable flags, files to synthesize, and the
assertions that prove an alignment or template actually attached — is
in [msa.md](msa.md) and [templates.md](templates.md). Do that work
here, not after the first suspiciously fast row.

### Path A — `$WORKDIR/bench/run_bioir.py`

```python
from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer, Template
from bionemo_ir.configs import AcceleratedConfig
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import (
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)

# Omit engine_kwargs["config"] unless the model profile documents one
# minimal input-parity feature flag. In that case derive the official
# pretrained config and record only that delta.
engine_kwargs = {"profile_inference": True}
if locked["model_source"] in {"boltz-1", "boltz-2", "openfold3"}:
    engine_kwargs["accelerated_configs"] = {
        # Omit default=: preserve the module's <=1024-token safe routine.
        "diffusion_module": AcceleratedConfig(backend="torch"),
    }

config = EngineProcessorConfig(
    model_source="<model_source>",
    executor_backend=None,  # serial: small dataset; Ray is for large jobs only
    runtime_args=locked["runtime_args"],
    engine_kwargs=engine_kwargs,
    feature_generator_stage=FeatureGeneratorStageConfig(
        init_context={"random_seed": locked["seed"]},
    ),
    writer_stage=WriterStageConfig(output_path=str(out_dir), format="cif"),
)
processor = build_processor(config)

# sample.request = spec_item_to_request(item, DATASET_ROOT)  # samples.md
for sample in in_scope_manifest:
    record = {"record": sample.request, "__record_id": sample.sample_id}
    torch.cuda.reset_peak_memory_stats()
    rows = list(processor([record]))
    row = rows[0]
    forward_s = float(row["model_inference_time"])
    # Scoring is AFTER the timed window (writer already ran).
    cif = parse_cif_path(row)
    lddt = dockq = None
    if cif and sample.gt_path:
        lddt = score_lddt(locked["ost_cmd"], cif, sample.gt_path)
        if sample.n_chains > 1:  # monomers have no interface
            dockq = score_dockq(locked["dockq_cmd"], cif, sample.gt_path,
                                args=locked["dockq_args"])
```

- **One `InputRequest` per `processor([record])` call.** Do not
  collect the manifest and call `processor(all_rows)`. That
  materializes every sample's features in host RAM and OOM-kills
  before the engine.
- Assert MSA and template files before the call, and verify on the
  featurized batch that both actually attached
  ([msa.md](msa.md#verification),
  [templates.md](templates.md#verification)).
- Read `row["model_inference_time"]` — that **is** `model.forward()`.
- Ignore `time_taken` and `stage_timing_s` for the headline.
- Reset peak memory before the call; record alloc / reserved after.
- Score the written CIF **after** the call, with
  `ost compare-structures` and — on multi-chain samples — `DockQ`.
  Missing GT → both `null` (still a valid latency row). A scorer
  that failed → record the stderr and leave that metric `null`; do
  not invent a score. A single-chain sample is
  `dockq: null, dockq_status: "single_chain"`, never `0.0`
  ([measurement.md](measurement.md#quality-lddt-and-dockq)).

### Path A — `$WORKDIR/bench/run_oss.py`

**Base this on the OSS inference script**, not a from-scratch
forward. Copy it into `$WORKDIR/debug/` (or import its
helpers) and wrap it. Do not edit `$OSS_ROOT`. Reuse its
input conversion, MSA load, featurizer, checkpoint load, and
CIF writer. You only add: one-sample loop, GPU-sync clock
around `model.forward()`, compile-once column, and the same two
scorers after write (`ost compare-structures`, plus `DockQ` on
supported protein interfaces with the locked `dockq_args`).

1. Convert **one** in-scope spec item at a time using **that
   script's** input format **including** every resolved A3M
   (and Boltz CSV). Do not build one OSS batch from the
   whole manifest. Do not invent a second parser. Do not
   read `examples/data/samples/`.
1. Run OSS featurization untimed (that one sample).
1. Move that batch to GPU untimed.
1. **Eager (required).** Time `model.forward()` (warmup 1, measure 1).
   Write the pred CIF, then `ost compare-structures` vs `gt_path`
   (outside the clock). Write `$WORKDIR/results/oss_eager.json`.
1. **Compile once.** On a **fresh** OSS model, wrap the target
   submodules with `compile_oss_hot_modules` **one time** (OF2:
   Evoformer; AF3-style: Pairformer + DiffusionModule). Each target
   uses `torch.compile(module)` with `dynamic` omitted (`None`), no
   manual shape marks, and unmodified global Dynamo shape settings.
   Do not `torch.compile(model)`.
1. **Fast synthetic compile probe.** Feed deterministic,
   shape-correct synthetic tensors directly to each compiled child at
   two sizes in A, A, B, B order. Initial A and first B may compile;
   immediate repeats must not. Test targets separately, then together.
   Walk compiler retries with this fixture—never rerun the full data
   pipeline merely to reject a compile setting. Synthetic timings are
   not benchmark results.
1. **Real integration probe.** After the synthetic probe passes,
   warmup 1 + measure 1 on the smallest in-scope sample, then on a
   second sample in the **same** `token_bin` when one exists (else the
   next bin). Count Dynamo compiles on both warmups. Probe rules are in
   [measurement.md](measurement.md#oss-torchcompile).
1. **If the probe fails, retry.** Walk the ladder in
   [measurement.md](measurement.md#compile-retry-ladder)
   (real child names, Python-only fences, fewer targets). Every try
   keeps the default `dynamic=None` policy on a fresh model. Do not
   give up after the first adaptation, recapture, or graph break.
1. **Compile column (only after a passing probe or retry).**
   Keep **that same compiled model**. Loop every in-scope
   sample (one batch at a time, warmup 1 / measure 1 per
   sample). Track `warmup_compile_delta` /
   `measure_compile_delta` on every row. Do not compile again.
   Automatic warmup specializations, including same-bin events, stay
   in the result and must be reported. Write
   `$WORKDIR/results/oss_compile.json` only when every published
   measured-forward delta is zero. If the **whole ladder** fails,
   skip this column — eager still stands.

Assert the OSS batch still contains MSA-derived features (non-empty
MSA depth / pair rows). An empty MSA tensor after "successful" load
is a failure.

If OSS must be monkey-patched to expose `forward`, copy the entry
script into `$WORKDIR/debug/`, patch the copy, revert nothing upstream.

### Path B — OSS features, swap the module

Follow [no-pipeline.md](no-pipeline.md). Write three scripts:

1. `$WORKDIR/bench/dump_oss_features.py` (`oss_python`) — call
   the **OSS inference script's** featurizer only; save
   `$WORKDIR/ref_data/oss_features/<id>.pt`.
1. `$WORKDIR/bench/run_oss.py` (`oss_python`) — load dump; run OSS
   eager (one sample at a time); compile target submodules
   **once** on a fresh model; probe; if it works, run **that**
   compiled model on every in-scope dump.
1. `$WORKDIR/bench/run_bioir.py` (`bioir_python`) — load the **same**
   dump, `adapt_oss_batch_to_bioir`, construct BioIR `Protenix`
   (`include_load_weights=True`), select `diffusion_module` with
   `AcceleratedConfig(backend="torch")` and no `default=`, then time
   `model.forward`. This preserves the same 1024-token safe routine as
   Path A.

Do not call `build_processor`. Do not re-featurize on the BioIR side.

### Progress reporting — every harness shows a bar

These runs take tens of minutes and a silent process is
indistinguishable from a hung one. Wrap the sample loop in `tqdm` in
**both** harnesses (and any dump / scoring loop), so the user can see
where a run is and whether it is still moving.

Count the bar in **samples**, so it reads `4/17` — the question
being asked is "which sample is it on", and a residue total answers
a question nobody has. Name the sample and its residue count in the
postfix: the manifest is sorted ascending, so seeing the current
size is what tells the reader whether the expensive tail is still
ahead. Treat the ETA as indicative only, since per-sample cost
grows steeply with size.

```python
from tqdm import tqdm

bar = tqdm(total=len(in_scope), unit="sample", dynamic_ncols=True,
           mininterval=5.0, desc="bioir")  # stderr; results go to stdout
for sample in in_scope:
    bar.set_postfix_str(f"{sample['sample_id']} ({sample['residues']} res)")
    ...  # warmup, then the one measured call
    bar.write(f"{sid}: {forward_s:.3f} s | peak {peak_gb} GB | "
              f"lddt {lddt} | dockq {dockq}")
    bar.update(1)
bar.close()
```

Rules:

- **Nothing inside the timing window.** The bar lives in the loop,
  never between the CUDA syncs; `set_postfix_str` before, `update`
  after.
- **`bar.write()`, not `print()`**, for per-sample lines, so results
  and bar do not interleave into garbage.
- **Keep it on when redirected.** A backgrounded run still needs
  progress in the log; `mininterval=5.0` keeps a redirected bar from
  filling the file with refreshes. Do not gate on `isatty()`.
- **Silence an inner framework bar.** A tree driving a Lightning
  `Trainer` prints its own per-batch bar, which fights the outer one
  at one sample per `predict` call; pass `enable_progress_bar=False`.
  It is display-only and changes no compute.
- **Degrade, never crash.** If `tqdm` is missing from an interpreter,
  fall back to plain prints rather than failing the bench.
- Flush per-sample lines (`flush=True`) so a killed run still shows
  what completed.

### Five-sample preparation smoke gate

After dataset mapping and both harnesses are complete, but before any
full sweep, run a **five-sample BioIR + OSS eager smoke benchmark**,
then an OSS `torch.compile` smoke on the same selection when possible.
This is mandatory: it catches an expensive tail failure while still
testing the real checkpoint, featurizers, writers, scorers, timing
window, and compiled dynamic-shape serving path.

Choose exactly five unique samples spanning the residue range:

1. Sort the in-scope manifest by total residues.
1. Start with ranks nearest 0%, 25%, 50%, 75%, and 100%.
1. Keep the smallest and largest samples fixed.
1. If templates exist but none of the three interior selections has
   templates, replace the nearest interior rank with the
   template-bearing sample closest to the median residue count.
1. Likewise ensure a declared-MSA sample and a multichain sample are
   represented when those classes exist, replacing interior ranks
   only.
1. Record the selected IDs, ranks, residue counts, and replacement
   reasons in `bench_config.json`.

Use the final locked configuration, one warmup and one measured
forward per side. Do not shorten diffusion, recycling, MSA, template,
or sampling settings for the smoke. Write separate artifacts:

```text
results/bioir_smoke.json
results/oss_eager_smoke.json
results/oss_compile_smoke.json
```

Build a fresh OSS model for the compile smoke, call
`compile_oss_hot_modules(...)` once with `torch.compile(module)` and
no `dynamic` argument or manual input marks, and first feed its
children synthetic direct inputs at two sizes. Repeat each size
immediately so the probe distinguishes automatic adaptation from
unstable recompilation. Use that fast fixture for the
[compile retry ladder](measurement.md#compile-retry-ladder).
After a synthetic attempt passes, run the two-real-sample integration
probe once, then the five smoke samples. Never compile per sample or
set `dynamic=True` or `dynamic=False`. If the ladder is
exhausted, omit `oss_compile_smoke.json` and report every attempted
target and failure; the eager gate can still pass.

For each selected sample, show the user:

- residues and chain count
- MSA and populated-template status
- BioIR, OSS eager, and valid OSS compile forward latency in seconds
- peak allocated GPU memory
- lDDT and DockQ when applicable
- explicit success or failure

Any missing BioIR/eager output, attachment mismatch, non-finite value,
scoring failure, or OOM fails the gate. Fix it and rerun all five; do
not quietly replace the failing sample. A compile-only failure follows
the retry ladder and is reported as an unavailable optional column,
not relabeled as eager.

After a successful smoke, print the comparison and **ask the user
whether to launch the full in-scope sweep**. Do not start the full run
until they answer. A declined full run leaves the smoke artifacts as
the handoff.

## Phase 4 — Run BioIR

The full run requires user approval after the five-sample smoke gate.

Path B first:

```bash
cd "$WORKDIR" && "$OSS_PYTHON" bench/dump_oss_features.py
```

Then, either path:

```bash
export CUTEDSL_FORCE_CUBIN=1
cd "$WORKDIR" && "$BIOIR_PYTHON" bench/run_bioir.py
```

Warmup and repeats: [measurement.md](measurement.md#warmup-and-repeats).
Write `$WORKDIR/results/bioir_forward.json`. Preserve stdout/stderr
under `$WORKDIR/debug/`.

## Phase 5 — Run OSS e2e

The full run requires the same approval; do not infer it from a
successful BioIR smoke.

```bash
cd "$WORKDIR" && "$OSS_PYTHON" bench/run_oss.py
```

Same manifest, same locked knobs, same GPU. Path B must load the
dumps from Phase 4, not featurize a second time.

Always write `$WORKDIR/results/oss_eager.json`. Write
`oss_compile.json` only after a passing compile probe **or a
passing retry**. Exhaust the retry ladder before skipping
compile. A compile failure after that does **not** block the
bench — report eager and every attempt.

If an OSS **eager** kernel crashes, that backend is blocked. Do not
disable the kernel and still label the row `oss_e2e`.

## Phase 6 — Compare and report

Print (not only file):

1. **Setup** — model key, path A or B, OSS url / ref / commit
   (must match the pin), both checkpoint
   hashes, GPU name / SM / driver, **power limit (W)**, **max SM
   clock and observed SM clock (MHz)**, `bioir_python` / `oss_python`,
   both torch versions, `isolation`, `ost_cmd`, `dockq_cmd`
   and `dockq_args`, locked
   `runtime_args`, warmup=1 / measure=1, BioIR graph config
   (`module_default`, 1024-token limit, captured and eager-fallback
   sample ids), OSS compile probe pass/fail (two-sample warmup compile
   deltas, targets if pass), `compile_stats.measurement_stable`
1. **Dataset** — `dataset_root`, spec file, MANIFEST digest,
   BUILD.json digest, in-scope count, per-sample MSA
   attached / none and `template_status` (naming any synthesized
   alignment), exclusions with reason
1. **Forward latency** — table from
   [measurement.md](measurement.md#comparison-table)
1. **Speedup** — vs OSS eager, and vs OSS compile when that column
   exists (`oss_s / bioir_s`; `> 1` favors BioIR). Print **geometric
   mean and median** over all in-scope `ok` rows, then the same two
   statistics in residue bins: **short** `< 512`, **medium**
   `512–1024` inclusive, **long** `> 1024`. Name `n` and the sample
   ids in every bin. An empty bin is `n=0`, not an invented ratio.
   See [measurement.md](measurement.md#speedup-aggregates).
1. **lDDT** — OpenStructure scores for BioIR / OSS eager / OSS
   compile. `null` when GT is missing or `ost` failed (cite stderr).
   Never invent a number. On **Boltz-2**, if BioIR and OSS lDDT
   diverge on MSA-bearing proteins, check
   [the known OSS deletion bug](#boltz-2--msa-deletion-bug)
   before blaming the engine.
1. **DockQ** — interface scores for every multi-chain sample, per
   side, with the chain mapping DockQ chose and the per-side mean
   over the samples it covers. `n/a` on single-chain rows, never
   `0.00`. Name the quality band (incorrect / acceptable / medium /
   high) rather than reading DockQ as a percentage, and treat a
   side-to-side gap as an input-parity question first
   ([measurement.md](measurement.md#quality-lddt-and-dockq)).
1. **GPU** — weights footprint, per-sample peak alloc / reserved,
   any OOM, **power limit / enforced / default (W)**, and
   **observed power draw (W)** plus **SM / graphics / memory
   clocks (MHz)** from in-forward samples (min / max / mean per
   scenario, busy samples only). Name max SM vs observed SM.
   Clocks and power are first-class report fields. An idle
   inventory snapshot is not the run clock.
1. **Commands** — exact invocations and result JSON paths
1. **Charts** — latency vs residue count (sum of all chains), then
   speedup vs residue count. Generate them from the result JSONs
   ([measurement.md](measurement.md#charts-latency-vs-residues)).
   **Show the PNGs in the final reply** (read the image files so
   they render for the human). A path-only mention is not enough.
1. **Open questions** — path to `implementation-notes.md`

Report latency in **seconds** (three decimals) everywhere — table,
charts, and per-sample lines. A Phase 6 that prints only overall
geomean, or GPU name without power and clocks, is incomplete.

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

BioIR's Boltz-2 featurizer keeps real per-row MSA deletion counts
(`bionemo_ir/pipeline/models/boltz2/featurizer.py`). An older
upstream `construct_paired_msa` (Boltz
`src/boltz/data/feature/featurizerv2.py`) reassigned
`chain_deletions` to a slice of itself inside the sequence loop:

```python
chain_deletions = chain_msa.deletions
for sequence in chain_msa.sequences:
    ...
    chain_deletions = chain_deletions[del_start:del_end]
```

After the first sequence, every later slice was empty, so
non-query `deletion_value` became all zeros. The benchmark pin
`v2.2.1` still contains this reassignment even though later upstream
code may be fixed. BioIR always uses the real deletions. Zeroing them
on the BioIR side to "match" the pinned OSS bug drops lDDT on
MSA-bearing proteins. See the reproducible contract in
[`models/boltz2.md`](models/boltz2.md).

**When BioIR vs OSS quality looks too far apart** (large lDDT
gap on protein samples that have A3Ms; RNA-only rows are not
this bug):

1. Search the **OSS** tree the bench is using for
   `chain_deletions = chain_deletions[` inside
   `construct_paired_msa`.
1. If that inner-loop reassignment is still there, the OSS
   column is an old featurizer, not a BioIR accuracy
   regression. Record the OSS commit, the hit, and the lDDT
   delta in `implementation-notes.md`. Tell the human.
1. Do **not** zero BioIR `deletion_value` / `deletion_mean` to
   close the gap.
1. If the OSS source already reads the full `all_deletions`
   (or never rebinds `chain_deletions` in the loop), this is
   not the cause — keep looking at config, MSA attach, seed,
   and sampling knobs.

Optional check: dump `deletion_value` (or `deletion_mean`) from
one MSA protein on both sides. OSS all-zero after the query row
plus a large lDDT gap is this bug.

## Later: other model kinds

v1 is a **folding** bench: `InputRequest` → structure CIF →
`ost compare-structures` lDDT plus DockQ on the interfaces,
X-axis = residue count (sum of
all polymer chains). A new
kind of BioIR model needs its own dataset, metric, and size
axis. Reuse the machinery below; do not pretend CIF / lDDT /
DockQ / these folding specs apply.

### Keep as-is for any kind

- BioIR freeze first; isolate OSS; never retune BioIR packages
- Dedicated env per third-party scorer (same idea as
  `$WORKDIR/envs/ost` and `$WORKDIR/envs/dockq`)
- Serial, one sample per `forward` / `processor()` call
- GPU-synced host clock around `model.forward()` only
- Warmup 1, measure 1; no invented numbers
- OSS eager always; compile **target submodules once**, then
  run every sample on that compiled model
- `CUTEDSL_FORCE_CUBIN=1` on every BioIR process
- Allocator left on its default — no `expandable_segments`
- Charts at the end, shown to the human, from result JSONs

**Rewrite per kind** (lock these in `bench_config.json` before
Phase 4):

- **Task / output** — what `forward` returns (coords, affinity
  scalar, embedding, …)
- **Dataset** — not these folding specs unless they actually
  carry the right labels
- **Quality metric** — community tool for that task, not
  hand-rolled. Folding = OpenStructure lDDT + DockQ on
  multi-chain items. Affinity ≠ lDDT.
- **Size axis** — residue count is folding-specific (sum of all
  polymer chains). Pick the
  quantity that drives compute (tokens, atoms, pocket+ligand
  size, …) and plot latency against it
- **Path A vs B** — factory (`build_processor`) or OSS features
  - BioIR module (`docs/ref/support-matrix.md` Pipeline column)
- **MSA / template rules** — only if that task uses them

**Known next key:** `boltz-2-affinity` (`Boltz2Affinity`,
Pipeline = No, `docs/ref/support-matrix.md`). Ligand *structure*
on Boltz-1/2 / OpenFold3 is already in v1. Affinity is a
different head: no CIF writer, no `ost`, no folding GT under
`ground_truth/`. When someone benches it, add a Path B dump of OSS
affinity features, time `model.forward()`, score with the
affinity metric the OSS paper uses, and chart vs a ligand /
complex size — do not reuse this folding manifest.

If the user names a non-folding key, **stop and extend this
skill** (or a sibling) before writing harnesses. Do not score
affinity with lDDT or fold-shaped `runtime_args`.

## Out of scope (v1)

- Nsight Systems / Nsight Compute, MFU, SOL%, kernel breakdown
- Ray replica throughput (large-dataset path; not this spec)
- Template **search** (HHsearch / HMMsearch); templates are attached
  from the spec's shipped structures only
  (see [samples.md](samples.md#templates-are-in))
- The demo tree `examples/data/samples/` as a substitute dataset
- DockQ / feature-tensor equivalence (`make-data-pipeline`)
- Isolated same-features microbench of a single stack (`module-onboard`)
  — Path B is e2e `model.forward` on OSS features, not a layer bench
- Non-folding heads (`boltz-2-affinity`, embeddings, design, …)
  until this skill (or a sibling) is extended

Those are separate skills or a later revision.

## Key gotchas

- **One sample per `processor()` / OSS forward.** Never batch the
  manifest.
- **Custom templates: all of them, both sides, counted.** Filters
  built for search output silently veto caller-supplied templates,
  and `mask.sum() > 0` will not notice. Patching a side to honour the
  input is allowed if recorded and reverted.
- **Load the checkpoint once before Phase 4** (weights preflight). A
  gated repo fails every sample in the loop, and the file is often
  already in the checkout under `examples/<model>/checkpoints/`.
- **A package name is not a model variant.** Read the pin's checkpoint
  registry and strict-load the same checkpoint on both sides. If a
  release changed its default architecture or retired BioIR's
  checkpoint, stop; a non-strict or key-rewriting load is model
  onboarding, not a benchmark patch.
- **Install OpenStructure after BioIR and OSS**, in a dedicated
  conda env (`$WORKDIR/envs/ost`). Never share that env with
  BioIR or OSS. Score with `ost compare-structures` only.
- **DockQ is required on every multi-chain sample**, in its own
  venv (`pip install DockQ` into `$WORKDIR/envs/dockq`). lDDT
  alone cannot see a wrongly docked assembly. Never install it
  into BioIR or OSS — it pins `numpy < 2.0`.
- **A monomer's DockQ is `null`, not `0.0`.** Zero means "the
  interfaces are wrong"; a single chain has none. DockQ exits 1 on
  an interface-free native, which is the answer, not an error.
- **An allocator flag is not free.**
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lowers the
  reserved ceiling but can cost throughput, and it is easy to copy
  from a memory-debugging session into a benchmark command. Keep it
  out of every timed run.
- **Let DockQ choose the chain mapping** and record it. Predicted
  and ground-truth CIFs often disagree about chain names
  (mmCIF chains come from `label_asym_id`), and the mapping search
  is what handles that. Same flags on both sides, always.
- **Path A: `profile_inference` already isolates `model.forward`.**
  Do not time `processor()` wall clock and call it the model.
- **This skill is folding-only.** Affinity / embeddings / design
  need a new dataset, metric, and size axis before any harness.
- **BioIR config is the default constructor** (`config=None` →
  `get_pretrained_config`). Never a handmade `BaseConfig`.
  A model-profile input-parity flag may be changed only on the
  official pretrained config and must be recorded (Boltz-2 templates).
  CUDA-graph `accelerated_configs` only after that, and only on
  families that support it.
- **Path B: do not call `build_processor`.** OSS featurizer + BioIR
  `Protenix(...)`. Dump features once; both forwards load that `.pt`.
- **Graph the BioIR diffusion module by default** on Boltz-1/2, OF3,
  and Protenix with `AcceleratedConfig(backend="torch")` only. Never
  supply `default=` or an explicit graph config: the module-declared
  exact-shape routine includes the `num_tokens <= 1024` safety guard,
  and an override replaces it. Larger inputs intentionally run eager.
  The one warmup forward is the capture; the one measured forward is
  the headline.
- **Base OSS on its inference script.** Find `predict` /
  `infer` / the CLI entry first. Wrap that. Do not reimplement
  OSS featurize + `forward` from internals unless the script
  cannot expose `model.forward()`.
- **OSS eager always; compile after a passing probe or retry.**
  Do not skip eager. Do not publish compile after the **whole
  ladder** failed. Do not give up on the first recapture.
  Automatic warmup specializations are allowed and must be reported;
  do not publish a row whose measured forward recaptures.
- **Compile OSS target submodules once, then run all samples.**
  OF2: Evoformer. AF3-style: Pairformer + DiffusionModule.
  Call `torch.compile(module)` with `dynamic` omitted (`None`), no
  `torch._dynamo.mark_dynamic` / `mark_static`, and no global Dynamo
  shape-policy overrides. Never compile the outer model. Track Dynamo
  compiles on every warmup and measured forward.
- **Reject bad compile settings with synthetic direct inputs first.**
  Exercise each selected child as A, A, B, B without running the full
  data pipeline or rollout. Initial A and adaptation B may compile;
  the immediate repeats must not. Those times are diagnostics only. A
  passing synthetic fixture still needs one real two-sample
  integration probe before the compile sweep.
- **H2D is outside the BioIR window.** OSS must move the batch first
  or the comparison is skewed.
- **Seed goes on the feature-generator stage**, not the tokenizer
  (`docs/ref/api.md`).
- **Paired A3Ms are per chain, same row count** (`docs/ref/api.md`).
  Dropping one side's paired file changes the forward.
- **Boltz-2: a large BioIR vs OSS lDDT gap on MSA proteins** may
  be the old OSS `construct_paired_msa` `chain_deletions` slice
  bug, not the engine. See
  [Boltz-2 — MSA deletion bug](#boltz-2--msa-deletion-bug).
  Do not zero BioIR deletions to match.
- **Boltz YAML `msa:` paths in the release may be stale
  absolutes.** Rewrite them to `$DATASET_ROOT/casp15/msa/`.
- **Sharing the GPU with another job invalidates the run.**
- **Do not `pip install` OSS into the BioIR env** when torch / CUDA /
  cuEq pins differ. Split interpreters (`environment.md`).
- **OSS revision is the pin.** OF3 / Protenix-v2 = `3rdparty/`
  gitlink. Boltz-1/2 = `v2.2.1`. OpenFold2 =
  [aqlaboratory/openfold `v2.2.0`](https://github.com/aqlaboratory/openfold/tree/v2.2.0).
  Do not time `main` or another tag.
- **OpenFold2 / OpenFold3 OSS: source-build DeepSpeed with only
  `evoformer_attn`.** `DS_BUILD_OPS=0 DS_BUILD_EVOFORMER_ATTN=1`.
  Do not build other DeepSpeed kernels. Do not use a PyPI wheel.
- **CUDA 12 packages → CUDA 13.** Do not `pip install` `*-cu12`,
  `[cu12]`, or `+cu12*` on this stack. Use the `cu13` counterpart
  (or BioIR's already-installed CUDA 13 wheel).
- **OSS apt packages: scan once, generate the script, then try
  `apt-get` / `sudo -n apt-get`.** Collect every unresolved soname
  in one ELF `DT_NEEDED` sweep (ignoring auditwheel-bundled
  copies) and write `ref_data/apt_install.sh`. If the agent cannot
  run apt (no root, no `sudo`, or policy-blocked), show that
  command inline and wait. Do not skip a missing `.so`, stub it,
  or discover packages one `ImportError` at a time.
- **Launch each harness with its own python.**
  `$BIOIR_PYTHON bench/run_bioir.py` and
  `$OSS_PYTHON bench/run_oss.py` — never a bare `python` after Phase 0.
- **Phase 6 must show charts**, not only a table. Plot
  `forward_s` (seconds) vs residue count from the result JSONs.
  Do not invent points.
- **Phase 6 must report GPU power and clock rates** from the
  inventory plus **in-forward** samples (power limit, observed
  draw, max SM clock, observed SM clock). A nameplate GPU string,
  or idle inventory clocks, is not enough.
- **Phase 6 speedup is geomean and median**, overall and in
  residue bins short `< 512`, medium `512–1024`, long `> 1024`.
  Empty bins are `n=0`. Do not print only a single overall
  geomean. Write `$WORKDIR/results/speedup.json`.
- **`CUTEDSL_FORCE_CUBIN=1` on every BioIR process.** JIT compile
  time makes forward latency unstable. In developer mode, `git lfs
  pull` the cubin packs before the editable install.
- **No `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, ever.**
  It can slow `forward` down, so it is never valid in a timed run
  on either side. Leave the allocator on its default and record no
  `PYTORCH_CUDA_ALLOC_CONF`
  ([measurement.md](measurement.md#allocator-flags--leave-pytorch-on-its-default)).
