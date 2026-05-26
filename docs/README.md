# TensorRT-BioNeMo — Getting Started
## Introduction

This guide walks first-time users through **pulling the NGC PyTorch container**, **installing the `tensorrt_bionemo` wheel**, and **running a sample inference pipeline end-to-end** (AlphaFold2 monomer + multimer, Boltz-1, Boltz-2, OpenFold3) using the **optimized PyTorch backend** (the recommended default — no engine build required).

> **Looking for backend selection guidance, benchmarking methodology, the programmatic Python API, weight-conversion helpers, the per-model checkpoint catalog, or the env-var reference?** See [`ADVANCED.md`](ADVANCED.md) — it covers backend selection (§0; **TRT engines are on a deprecation track**, PyTorch is the supported default), how to interpret pipeline wall-clock time (§1), the legacy AF2 / OF2 TRT engine-build flow (§2), the programmatic Python API for model construction + `model.optimize()` (§3), upstream weight sourcing (Appendix A), the env-var / `default.yaml` reference (Appendix B), and the data-pipeline support matrix (Appendix C).
>
> **Onboarding a custom module to TRT-BioNeMo?** See [`MODULE_ONBOARDING_GUIDE.md`](MODULE_ONBOARDING_GUIDE.md) instead — it covers the bundled Claude Code skill, the RF3 Pairformer / DiT reference conversions, and the weight-remap walkthroughs. Most users running the bundled scripts do **not** need it.

> **⚠ Known limitations in this EA release — please read before adopting.** This release supports **protein-only** structure prediction (monomers, homo-oligomers, and hetero-oligomers from amino-acid sequences + MSAs). The following inputs are **not supported** in this release, even where the schemas accept them:
>
> - **Templates** — no HHsearch / HMMsearch template features are staged; every bundled sample runs without templates and that is the only validated path.
> - **Small-molecule ligands** — Boltz-1/2 and OpenFold3 ligand inputs (CCD codes, custom SMILES) are not exercised by the bundled data pipeline.
> - **Nucleic acids (DNA / RNA)** — the `Polymer.polymer_type` schema accepts `"dna"` / `"rna"`, but the bundled pipeline assumes `"protein"` end-to-end and does not produce validated structures for nucleic-acid chains.
> - **Ligand-affinity prediction** — `boltz-2-affinity` is **intentionally excluded** from this release; the affinity data pipeline is not bundled and the model key is rejected by [`scripts/run_pipeline.py`](scripts/run_pipeline.py).
>
> If your use case depends on any of the above, contact the TRT-BioNeMo developers before integrating. For the full per-model coverage matrix (which input shapes / MSA configurations each model accepts), see [`ADVANCED.md` Appendix C](ADVANCED.md#appendix-c--data-pipeline-support-matrix).

---

## Prerequisites
- Both x86_64 and aarch64 compute nodes are supported.
- On the host, prerequisites include:
  - NVIDIA GPU Driver compatible with the docker image `nvcr.io/nvidia/pytorch:${NGC_TAG}-py3` (more below)
  - [Docker](https://docs.docker.com/engine/install/)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)
- Standard documentation for installing these prerequisites include
  - For DGX / HGX compute nodes: [NVIDIA DGX OS 7 User Guide](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/additional_software.html#managing-os-and-software-updates)
  - For non-DGX / non-HGX compute nodes with datacenter-class GPUs: [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide)

---

## Quick start (one command, outside Docker)

If you just want to confirm the bundle works on your machine, run the wrapper **from the host** — no manual `docker pull` / `pip install` needed:

```bash
cd release_artifacts   # this directory — the one containing wheel/, scripts/, …
./quick_start.sh       # AF2 monomer, PyTorch backend, runs every bundled monomer sample once
```

The script pulls `nvcr.io/nvidia/pytorch:${NGC_TAG}-py3` (default `25.12`), bind-mounts the bundle to `/workspace/release_artifacts` inside the container, **auto-detects the container CPU architecture** via `uname -m` and installs the matching `x86_64` or `aarch64` wheel from `wheel/pytorch_${NGC_TAG}/`, then runs `scripts/run_pipeline.py` with any extra arguments you pass through. Outputs land under `output/` on the host. Both x86_64 (AMD64) and aarch64 (ARM64 — Grace / GH200 / GB200) hosts are supported.

A few representative invocations:

```bash
./quick_start.sh --model boltz-2                          # Boltz-2, PyTorch backend
./quick_start.sh --model alphafold2_multimer_1 --repeat 3 # each bundled sample 3x
MODEL_NAME=openfold3 HF_TOKEN=hf_xxx ./quick_start.sh     # gated HF repo
NGC_TAG=25.08 ./quick_start.sh --backend serial           # older NGC + serial executor
./quick_start.sh --replicas 4                             # multi-GPU data-parallel
```

**Host prerequisites:** See [Prerequisites](#prerequisites). Nothing else — wheel, checkpoints, and sample data all come from this bundle.

**Configuration:** the most common knobs (`MODEL_NAME`, `OUTPUT_DIR`, `EXECUTOR_BACKEND`, `PIPELINE_REPEAT`, …) live in [`default.yaml`](default.yaml) next to the script. Edit the file for repeatable defaults; environment variables on the command line still take precedence on a per-key basis (`MODEL_NAME=boltz-2 ./quick_start.sh` overrides only that key). For the full set of recognized env vars (per-model `<MODEL>_CKPT` overrides, Boltz metadata paths, `EXTRA_DOCKER_ARGS`, `SKIP_PULL`, …), see the env-var reference table in [`ADVANCED.md`](ADVANCED.md#environment-variable-reference).

When the run finishes (a few minutes on a single H100, longer on first run while images and weights warm up), you should see:

```text
================================================================================
Pipeline Results
================================================================================
Total time:        47.24s
Time per request:  9.45s
Successful:        5/5
Errors:            0/5
…
Pipeline completed successfully!
```

**Congratulations — TensorRT-BioNeMo is up and running.** Five predicted PDB structures plus their per-request confidence scores (`*_scores.json`) are now under `output/` on the host. Open one in PyMOL/ChimeraX (or just `head output/T1031_0.pdb`) to confirm.

If you'd rather walk through the setup step by step (e.g. to poke around inside the container, swap models, or build TRT engines), follow §§1–6 below — `quick_start.sh` is just an automation of exactly those steps plus the §5 pipeline invocation.

---

## 1. Pull the base image

The release bundle ships prebuilt wheels targeted at specific [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) container tags. Each tag has its own `wheel/` subdirectory with one wheel per CPU architecture (`x86_64` / AMD64 and `aarch64` / ARM64):

| NGC image tag | Wheel directory | x86_64 wheel | aarch64 wheel |
|---|---|---|---|
| `nvcr.io/nvidia/pytorch:25.12-py3` | `wheel/pytorch_25.12/` | `tensorrt_bionemo-0.2.0+cu131-cp312-none-manylinux_2_39_x86_64.whl` | `tensorrt_bionemo-0.2.0+cu131-cp312-none-manylinux_2_39_aarch64.whl` |
| `nvcr.io/nvidia/pytorch:25.08-py3` | `wheel/pytorch_25.08/` | `tensorrt_bionemo-0.2.0+cu130-cp312-none-manylinux_2_39_x86_64.whl` | `tensorrt_bionemo-0.2.0+cu130-cp312-none-manylinux_2_39_aarch64.whl` |

NGC PyTorch images are multi-arch manifests, so a single `docker pull` fetches the right variant for your host:

```bash
docker pull nvcr.io/nvidia/pytorch:25.12-py3
```

> Do **not** mix-and-match NGC tags or architectures — a wheel built against 25.12 won't load inside a 25.08 container, and `pip` will refuse the wrong-arch wheel with `not a supported wheel on this platform`.

**Prerequisites on the host:** .  See [Prerequisites](#prerequisites). Both x86_64 and aarch64 GPU nodes are supported, no `--platform` override needed.

---

## 2. Run the container with a single bind mount

Start an interactive GPU container with shared memory and stack limits sufficient for engine builds. Mount the **release bundle directory** to a single container path and `cd` into it — every sub-folder (`wheel/`, `checkpoint/`, `engines/`, `output/`, `scripts/`, `notebooks/`, `data/samples/`) then lives directly under one known root.

Run from the root of this release bundle:

```bash
NGC_TAG=25.12  # must match the NGC tag pulled in §1 (supported: 25.12, 25.08)

docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it \
  -v "$(pwd):/workspace/release_artifacts" \
  -w /workspace/release_artifacts \
  "nvcr.io/nvidia/pytorch:${NGC_TAG}-py3"
```

Inside the container, `pwd` is `/workspace/release_artifacts`, laid out as:

```
/workspace/release_artifacts/
├── wheel/pytorch_<NGC_TAG>/*.whl   # pip installable (see §3)
├── checkpoint/                     # AF2 .pt checkpoints (auto-detected)
├── engines/                        # (optional) TRT engine build tree — see ADVANCED.md
├── output/                         # inference outputs (write-enabled)
├── scripts/run_pipeline.py         # §5 entry point
├── notebooks/openfold2_full_pipeline.ipynb  # §6 entry point
├── data/samples/{monomers,homopolymers,heterooligomers}/   # bundled inputs, auto-detected
├── .claude/skills/module-onboard/          # Claude Code CLI skill (see MODULE_ONBOARDING_GUIDE.md)
├── default.yaml                    # quick_start.sh defaults (MODEL_NAME, OUTPUT_DIR, …)
├── quick_start.sh                  # zero-setup wrapper around §§1–5 (host-side)
├── ADVANCED.md                     # advanced guide (TRT, benchmarking, env vars, …)
├── MODULE_ONBOARDING_GUIDE.md      # onboard a custom module via the Claude Code skill (optional)
└── README.md                       # this document
```

> `scripts/run_pipeline.py` auto-detects `data/samples/` and `checkpoint/` relative to its own location, so the §5 examples need **no extra env vars** with this single mount in place.

Create writable host directories once if missing:

```bash
mkdir -p engines output
```

---

## 3. Install the `tensorrt_bionemo` wheel

Inside the container (already `cd`'d into `/workspace/release_artifacts` from §2), pick the wheel matching the container's CPU architecture from the version-matched subdirectory of `wheel/`:

```bash
# NGC_TAG must match §1 (supported: 25.12, 25.08).
# Architecture is auto-detected from $(uname -m) — x86_64 or aarch64.
ARCH="$(uname -m)"
pip install --no-cache-dir "wheel/pytorch_${NGC_TAG}/tensorrt_bionemo-"*"-manylinux_2_39_${ARCH}.whl"
```

If `pip` errors with `ERROR: <wheel> is not a supported wheel on this platform`, you're installing the wrong-arch file — verify `uname -m` inside the container. If `pip` errors at import time with a missing `libcu*` symbol, the `wheel/pytorch_<NGC_TAG>/` subdirectory doesn't match the NGC image you pulled — re-check the §1 table.

For the exact pinned filenames per NGC tag and architecture, see the table in §1.

---

## 4. Verify the runtime

Confirm GPU, PyTorch CUDA, `tensorrt_bionemo`, and the builder CLI in the **same** container where you installed the wheel:

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
python -c "import tensorrt_bionemo; import tensorrt as trt; print('tensorrt_bionemo OK, TensorRT:', trt.__version__)"
```

> If you bind-mount a `tensorrt_bionemo/` git checkout that shadows the installed package, run the import checks from `cd /tmp` first.

---

## 5. Run the sample pipeline script

The bundled script ([`scripts/run_pipeline.py`](scripts/run_pipeline.py)) is **model-generic**: it supports AlphaFold2 / OpenFold2 monomer + multimer, Boltz-1, Boltz-2, and OpenFold3 (Boltz-2 affinity is out of scope — see [`ADVANCED.md`](ADVANCED.md#appendix-c--data-pipeline-support-matrix)). Select the target via `MODEL_NAME` (or `--model`), and the script:

- Picks the **optimized PyTorch backend by default** — no engine build required.
- **Auto-detects** the bundled `data/samples/` tree and the bundled AF2 `checkpoint/` directory.
- Writes PDBs and per-request `*_scores.json` under `./output/` (override via `OUTPUT_DIR` or `--output-dir`).

All commands below assume the working directory is `/workspace/release_artifacts` (set by `-w` in §2).

```bash
# AlphaFold2 monomer (default)
python scripts/run_pipeline.py

# AlphaFold2 multimer (uses T1151s heterooligomer with paired MSAs)
MODEL_NAME=alphafold2_multimer_1 PIPELINE_REPEAT=3 \
python scripts/run_pipeline.py

# Boltz-1 / Boltz-2 (ccd.pkl + mols/ are auto-downloaded from HuggingFace)
MODEL_NAME=boltz-2 python scripts/run_pipeline.py

# OpenFold3 — needs HuggingFace auth (gated repo). See Appendix A in ADVANCED.md.
MODEL_NAME=openfold3 HF_TOKEN=hf_xxx python scripts/run_pipeline.py

# Multi-GPU data-parallel via Ray replicas (one engine per GPU)
MODEL_NAME=boltz-2 python scripts/run_pipeline.py --replicas 4

# In-process executor (no Ray; easier to debug)
MODEL_NAME=alphafold2_1 python scripts/run_pipeline.py --backend serial
```

> **TensorRT engines.** The TRT engine path is on a deprecation track — the optimized PyTorch backend is the supported default for every model in this release. The **AF2 / OF2 Evoformer** TRT flow is documented end-to-end in [`ADVANCED.md` §2](ADVANCED.md#2-tensorrt-backend-legacy) for benchmarking / reproducibility. For **Boltz-1, Boltz-2, or OpenFold3** TRT engines (where TRT still wins on short-sequence throughput, ≲ 700 residues), engine build and runtime wiring are not documented in this release — contact the TRT-BNM developers.

**Common env vars** (all optional thanks to auto-detection):

| Env var | Purpose |
|---|---|
| `MODEL_NAME` | Model key (default: `alphafold2_1`). See the per-model table in [Appendix A of ADVANCED.md](ADVANCED.md#supported-checkpoints--hf-repo--local-env-var). |
| `PIPELINE_REPEAT` | Number of passes through the bundled sample pool (default: `1`). Use `>1` for warm-up / measurement runs. |
| `OUTPUT_DIR` | PDB output directory. Default: `./output` relative to cwd. |
| `<MODEL>_CKPT` | Per-model local checkpoint override (`ALPHAFOLD2_1_CKPT`, `BOLTZ1_CKPT`, `BOLTZ2_CKPT`, `OPENFOLD3_CKPT`, …). Auto-detected for AF2; optional for Boltz / OF3 (HF fallback). |
| `EXECUTOR_BACKEND` | `"ray"` (default) or `"serial"`. Also exposed as `--backend`. |
| `REPLICAS` | Ray only: data-parallel engine replicas (default: all visible GPUs). Also exposed as `--replicas`. |
| `ENGINE_OUTPUT_DIR` | Optional TRT engine dir (AF2 / OF2 legacy path only — see [`ADVANCED.md` §2](ADVANCED.md#2-tensorrt-backend-legacy)). |

**Illustrative completion summary** (AlphaFold2 monomer, optimized PyTorch backend, default `./output/`):

```text
================================================================================
Pipeline Results
================================================================================
Total time:        47.24s
Time per request:  9.45s
Successful:        5/5
Errors:            0/5

Output artifacts per request:
  T1031_0:
    structure(s): /workspace/release_artifacts/output/T1031_0.pdb
    scores:       /workspace/release_artifacts/output/T1031_0_scores.json
  …
================================================================================
Pipeline completed successfully!
```

The `<rid>_scores.json` files contain the JSON-encoded confidence metrics (`plddt`, `ptm`, `iptm`, `pae`, `max_pae`, …) emitted by the writer stage.

---

## 6. Run the sample notebook (alternative to §5)

The bundled notebook ([`notebooks/openfold2_full_pipeline.ipynb`](notebooks/openfold2_full_pipeline.ipynb)) is the OpenFold2-monomer counterpart of §5 with extra cells for Ray configuration, single-GPU vs replica execution, and optional TensorRT-accelerated Evoformer wiring.

Execute it non-interactively (install `nbconvert` + `ipykernel` first if missing):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/openfold2_full_pipeline.ipynb \
  --output /workspace/release_artifacts/output/openfold2_full_pipeline_executed.ipynb
```

---

## Quick checklist

- [ ] Base image pulled; container started with `--gpus all` and the **single bundle mount** from §2.
- [ ] `tensorrt_bionemo` installed from the matching `wheel/pytorch_<NGC_TAG>/…` directory.
- [ ] `nvidia-smi`, PyTorch CUDA, `import tensorrt_bionemo`, `trtbnm-build --help` all succeed.
- [ ] `data/samples/` and `checkpoint/` populated (bundled — auto-detected by the script).
- [ ] §5 sample script and/or §6 notebook run; outputs appear under `output/`.

---

## Where to next

- **Backend choice (PyTorch vs TensorRT), per-model recommendations, SM-version caveats** → [`ADVANCED.md` §0](ADVANCED.md#0-choose-a-backend-pytorch-vs-tensorrt)
- **Benchmarking methodology** (cold vs warm, JIT bucket compilation, Ray vs serial timings) → [`ADVANCED.md` §1](ADVANCED.md#1-benchmarking-methodology--interpreting-wall-clock-time)
- **TensorRT engine build flow — AF2 / OF2 Evoformer** (env vars, checkpoint conversion, `trtbnm-build`; on the deprecation track. **Boltz / OpenFold3 TRT — including the short-seqlen throughput regime where TRT still wins — is not documented; contact the TRT-BNM developers**) → [`ADVANCED.md` §2](ADVANCED.md#2-tensorrt-backend-legacy)
- **Programmatic Python API** (call OF2 / AF2 / Boltz-1/2 / OF3 from your own code: `get_model_class`, `model.optimize()`, single-request inference recipe — what `scripts/run_pipeline.py` wraps) → [`ADVANCED.md` §3](ADVANCED.md#3-programmatic-api--model-construction--inference)
- **Per-model checkpoint catalog & weight conversion** (HF repos, env-var names, JAX→PT helper) → [`ADVANCED.md` Appendix A](ADVANCED.md#appendix-a--upstream-weights--per-model-checkpoint-catalog)
- **Environment variable reference** (every `quick_start.sh` / `run_pipeline.py` knob in one table; YAML-key + env-var + default columns) → [`ADVANCED.md` Appendix B](ADVANCED.md#appendix-b--environment-variable-reference)
- **Data-pipeline support matrix** (monomer/multimer, MSA, templates, ligands) → [`ADVANCED.md` Appendix C](ADVANCED.md#appendix-c--data-pipeline-support-matrix)
- **Onboarding a custom module to TRT-BioNeMo** (Claude Code skill, RF3 Pairformer + DiT reference conversions, weight-remap walkthroughs) → [`MODULE_ONBOARDING_GUIDE.md`](MODULE_ONBOARDING_GUIDE.md)
