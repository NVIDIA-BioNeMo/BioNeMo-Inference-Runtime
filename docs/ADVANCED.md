# TensorRT-BioNeMo — Advanced Guide

Companion to [`README.md`](README.md). The README covers first-time setup (image → container → wheel install → verify → run a sample). This document covers everything that goes beyond a default PyTorch happy-path run:

- §0 — **Choose a backend** (PyTorch vs TensorRT)
- §1 — **Benchmarking methodology** — interpreting wall-clock time
- §2 — **TensorRT backend, end-to-end** (env vars → checkpoint conversion → `trtbnm-build` → re-running §5 with `ENGINE_OUTPUT_DIR`)
- §3 — **Programmatic API** — model construction, `model.optimize()`, and a single-request inference recipe for OF2 / AF2 / Boltz-1/2 / OF3
- Appendix A — **Upstream weights & per-model checkpoint catalog** (HF repos, env-var names, JAX→PT helper)
- Appendix B — **Environment variable reference** (every knob exposed by `quick_start.sh` and `scripts/run_pipeline.py`, with YAML-key + env-var + default columns)
- Appendix C — **Data-pipeline support matrix** (monomer/multimer, MSA, templates, ligands)

If you have not yet pulled the image, run the container, installed the wheel, or run the sample script, start with [`README.md`](README.md) §§1–6 first.

> **Onboarding a custom module to TRT-BioNeMo?** That's covered in a separate document — see [`MODULE_ONBOARDING_GUIDE.md`](MODULE_ONBOARDING_GUIDE.md) for the bundled Claude Code skill, RF3 Pairformer / DiT reference conversions, and weight-remap walkthroughs. Most early-access customers running the bundled scripts do **not** need it.

> **⚠ Known limitations in this EA release.** **Protein-only** structure prediction is supported. The bundled pipeline does **not** support: templates (no HHsearch / HMMsearch staging), small-molecule ligands (Boltz / OpenFold3 CCD or custom SMILES), nucleic acids (DNA / RNA accepted by the schema but not validated end-to-end), or ligand-affinity prediction (`boltz-2-affinity` is intentionally excluded — rejected by `scripts/run_pipeline.py`). Per-model details: [Appendix C](#appendix-c--data-pipeline-support-matrix). The same callout appears at the top of [`README.md`](README.md) so first-time users see it before pulling images.

---

## 0. Choose a backend (PyTorch vs TensorRT)

> **Strong recommendation: use the optimized PyTorch backend.** The TensorRT engine path is on a deprecation track (see the callout below); the PyTorch backend is the supported default for every model in this release.

> **TensorRT-engine deprecation notice.** Starting with this release (version 0.2.0), the **optimized PyTorch backend** ships with attention kernels (triangle attention, pairwise attention) that match or beat the TensorRT engines on **Ampere and Hopper datacenter GPUs**. The TRT engine path is therefore being **phased out** in upcoming releases — the PyTorch backend is the supported default for every model. The AF2 / OF2 Evoformer TRT flow is preserved end-to-end in §2 for reproducibility / benchmarking. **For Boltz-1, Boltz-2, or OpenFold3 TRT engines — including the short-sequence regime where TRT still wins on throughput — engine build and runtime wiring are not documented in this release; contact the TRT-BNM developers.**

> **GPU naming used in this guide.** Where this guide talks about "the target GPUs" or "Hopper / Ampere", the precise compute-capability mapping is:
>
> | Architecture | Compute capability | Datacenter GPUs in scope |
> |---|---|---|
> | **Ampere** | SM80 | A100, A40, A30 |
> | **Ada Lovelace** | SM89 | L40, L40S |
> | **Hopper** | SM90 | H100, H200, GH200 (Grace-Hopper) |
> | **Blackwell** *(not yet covered)* | SM100+ | B100, B200, GB200 |
>
> "SM90 and earlier" / "Hopper / Ampere" in this doc means the Ampere–Hopper family in the first three rows. Blackwell-class GPUs (B100 / B200 / GB200) are **not yet covered** by the optimized PyTorch backend in this release — both backends still run, but the kernel-level guarantees in §0 do not apply yet. You can check the SM version on your host with `python -c 'import torch; print(torch.cuda.get_device_capability())'` (a `(major, minor)` tuple — `(9, 0)` is H100 / Hopper, `(8, 0)` is A100 / Ampere, `(10, 0)` is Blackwell).

Behavior of the PyTorch backend differs by model family:

- **OpenFold2 / AlphaFold2 (monomer + multimer):** the optimized PyTorch backend **outperforms the Evoformer TensorRT engine across the full supported sequence-length range** on Ampere–Hopper (SM90 and earlier). There is no crossover region where TRT wins — use PyTorch unconditionally.
- **Boltz-1, Boltz-2, OpenFold3:** the optimized PyTorch backend outperforms TRT **beyond ~700 residues**; TRT still has a throughput edge on **short sequences (≲ 700 residues)** where per-step Python/dispatch overhead dominates. The TRT engine path for those models is on the deprecation track above — see "Boltz/OF3 short-sequence high-throughput" row below for the recommended action.

Use this table to pick a path:

| Deployment profile | Recommended backend | Why |
|---|---|---|
| **OpenFold2 / AlphaFold2 (any sequence length)** on Ampere / Hopper (A100 / H100 / H200 / L40S, SM90 and earlier) | **Optimized PyTorch backend** | Beats the Evoformer TRT engine across the full sequence-length range on Ampere / Hopper; no engine build; no profile-bound fallbacks. |
| **Boltz-1/2 or OpenFold3, long sequences (≳ 700 residues)** on Ampere / Hopper (A100 / H100 / H200, SM90 and earlier) | **Optimized PyTorch backend** | Higher throughput than TRT at long seqlen on Ampere / Hopper-class GPUs; no engine build; easiest to ship. |
| **Boltz-1/2 or OpenFold3, high-throughput short sequences (≲ 700 residues)** | **TensorRT engines** — *technically faster, but on the deprecation track* | Lower Python/dispatch overhead per step still wins here. **Engine build for these models is not documented in this release** — contact the TRT-BNM developers if you have a hard requirement. New deployments should plan for the PyTorch backend (the gap will close in upcoming releases). |
| **AF2 / OF2 reproducibility or benchmarking** (TRT vs PyTorch comparison) | TensorRT (Evoformer) — **legacy / deprecated path** | Documented end-to-end in §2 for back-compat. Not recommended for new production deployments. |
| **Mixed workloads** | **PyTorch first, TRT only where strictly required** | Start with PyTorch; only consider TRT where profiling shows a real win and deprecation timelines are acceptable (practically: Boltz / OF3 at short seqlen — OF2 always stays on PyTorch). |

**What this means in practice:**

- For a **PyTorch-backend** run (the recommended default for **every** model), [`README.md`](README.md) §§1–6 are sufficient. You can ignore §2 of this doc entirely.
- For an **AF2 / OF2 TensorRT** comparison run, complete [`README.md`](README.md) §§1–4 first (image, container, wheel, verify), then continue with §2 of this doc (env vars → conversion → `trtbnm-build`), then return to [`README.md` §5](README.md#5-run-the-sample-pipeline-script) and re-run with `ENGINE_OUTPUT_DIR=…`.
- For a **Boltz / OpenFold3 TensorRT** run (short-sequence throughput), the multi-module engine build and runtime wiring are **not in this release** — reach out to the TRT-BNM developers.

---

## 1. Benchmarking methodology — interpreting wall-clock time

The **"Total time"** line printed at the end of a `quick_start.sh` (or `scripts/run_pipeline.py`) run is the end-to-end pipeline wall-clock — **not** a pure model-forward benchmark. On a cold host it bundles in costs that are amortized across runs and do not represent steady-state model performance:

- **Python import & framework init** — `import torch`, `import tensorrt`, `import ray`, CUDA context creation.
- **Ray cluster bring-up** (`--backend ray` default): starting the object store, scheduler, and one actor per replica.
- **Checkpoint & metadata load** — `alphafold2_1.pt` / HF Hub fetch for Boltz / OF3 weights, **plus** the data-pipeline metadata bundle (Boltz CCD + tokenizer blobs, OF3 atom-type tables, …) under `~/.cache/huggingface/` and `~/.cache/trt-bionemo/`. These warm up on the first run; subsequent runs skip the download and just `torch.load` from local disk.
- **First-forward JIT / kernel autotune** — CUTLASS-DSL and Triton kernel compilation (written to `/tmp/.../bionemo_kernel_cache/{dsl,triton}`), `torch.compile` graph capture, TensorRT tactic selection (if `ENGINE_OUTPUT_DIR` is set). Most are cached on disk after the first forward pass and skipped on repeat runs.

### Recommended workflow for steady-state model-forward throughput

1. **Use `--backend serial --repeat N` with `N ≥ 2`** in a single process. Thanks to the pool-cycling semantics, `--repeat 2` submits every bundled sample twice *back-to-back in the same process*, so the **first pass** through the pool absorbs the per-process JIT + DSL/Triton-kernel compile cost (and the one-time `torch.compile` graph capture / TRT tactic selection), and every **subsequent pass** runs against warmed caches. The serial backend prints **per-request model time** (and residue counts) for every request, so you can read cold-vs-warm model-forward seconds directly off the log without re-paying Python/Ray startup.
2. **Read the warm lines, not `Total time`.** Every request is suffixed with its per-sample repeat index: IDs ending in `_0` are the cold pass (one per sample), `_1` is the first warm pass, `_2` the next, … Use **`sum model`** / **`mean model`** at the bottom of *Model inference time (serial, per request)* — `Total time` also counts parser/featurizer/writer overhead. Bump `--repeat` to dilute the cold `_0` entries further.

> **Note on JIT granularity.** The CUTLASS-DSL and Triton kernels are compiled **per residue-range bucket**, not per exact sequence length, so the cold pass warms one kernel per bucket and all subsequent samples that land in an already-seen bucket pay *no* extra JIT cost (even if their exact residue count differs). In the bundled monomer pool (95, 100, 199, …), the first pass usually covers every bucket the warm passes will hit — which is why `--repeat 2` is sufficient for a stable `mean model`.

3. **For production / end-to-end throughput, use `--backend ray --replicas N`.** The serial backend is only intended for debugging or quick single-process measurements; Ray is the right fit for real deployments because its staged `map_batches` pipeline hides parser / featurizer / MSA-load / writer latency behind GPU inference (and scales the GPU stage with replicas). As a side effect, per-row timings in Ray overlap across stages and aren't directly comparable to the serial per-request numbers — use serial for *"what does the model alone cost?"* and Ray for *"what does the system deliver?"*.

### Concrete recipe

```bash
# One-shot cold+warm measurement: the first pool-pass warms up caches, the
# subsequent passes give you steady-state per-request timings. Ignore the
# first N per-request lines (where N = pool_size), then read "sum model"
# and "mean model" — or bump --repeat further (3, 5, …) to dilute the
# cold first pass in the sum/mean.
./quick_start.sh --backend serial --repeat 2
```

This applies equally to `quick_start.sh` and direct `scripts/run_pipeline.py` invocations inside the container; `quick_start.sh` just adds Docker pull + wheel install to the outer wall-clock (not to anything reported inside the pipeline's own log).

---

## 2. TensorRT backend (legacy)

> **Deprecation notice.** This section documents the **AlphaFold2 / OpenFold2 Evoformer** TRT flow only and is preserved for back-compat / benchmarking. The optimized PyTorch backend (§0) is the supported default for every model in this release.
>
> **For Boltz-1, Boltz-2, or OpenFold3 TRT engines, contact the TRT-BNM developers.** Engine build, multi-module wiring, and runtime support for those models are not documented here.

The AF2 / OF2 TRT flow has three steps:

1. **Export environment variables** (§2.1) — pin model name, paths, sequence-length bounds, and dtype.
2. **Convert the upstream checkpoint** to the TRT-BNM Safetensors layout (§2.2).
3. **Build the Evoformer TensorRT engine** with `trtbnm-build` (§2.3).

Once the engine exists, return to [`README.md` §5](README.md#5-run-the-sample-pipeline-script) and re-run with `ENGINE_OUTPUT_DIR=…` (see §2.4 for the AF2-monomer / multimer layout).

### 2.1 Export environment variables

Set variables to match your model and paths under `engines/` and `checkpoint/` (both relative to `/workspace/release_artifacts` from `README.md` §2). Names below are illustrative for an OpenFold-style Evoformer flow; for Boltz or other models, follow your release's `trtbnm-build` and conversion docs for `--model`, `--module`, and checkpoint layout.

```bash
# Logical model id passed to conversion / trtbnm-build (supported names depend on release).
export MODEL_NAME=alphafold2_1

# Directory that will hold (or already holds) the converted Safetensors / TRT-BNM checkpoint
# tree after the conversion step (input to trtbnm-build --checkpoint_dir).
export SAFETENSORS_PATH=engines/evoformer_safetensors

# Directory where trtbnm-build writes TensorRT engines (package-specific layout under this root).
export ENGINES_PATH=engines/evoformer_engines

# Upstream PyTorch (or equivalent) checkpoint file on disk.
# See Appendix A for where these come from per model.
export CHECKPOINT_PATH=checkpoint/alphafold2_1.pt

# Dynamic sequence-length range used at build time. Supported range is model-dependent;
# many stacks support roughly 4–2048. Building for a **narrower** band (e.g. 16–1536)
# is usually **faster and lighter** than spanning the full range if your deployment fits it.
export MAX_SEQLEN=1536
export MIN_SEQLEN=16

# Module name for trtbnm-build (evoformer, pairformer, etc.—see release docs).
export MODULE_NAME=evoformer

# Precision hint for the builder (e.g. bfloat16 weakly-typed engine where supported).
export DTYPE=bfloat16
```

`alphafold2_1.pt` is shipped under `checkpoint/` in this bundle, so `${CHECKPOINT_PATH}` already points at a valid file. Drop additional AF2 checkpoints (e.g. `alphafold2_2.pt` … `alphafold2_5.pt`, or `alphafold2_multimer_*.pt`) into the same directory before §2.2 if you need to run other model variants. For Boltz / OF3 sourcing options, see [Appendix A](#appendix-a--upstream-weights--per-model-checkpoint-catalog).

### 2.2 Convert checkpoints to the TRT-BNM layout (Safetensors)

Conversion is **model-specific**. The package ships examples under `tensorrt_bionemo`'s `EXAMPLES_DIR` (override with `TENSORRT_BIONEMO_EXAMPLES_DIR` if you mount examples elsewhere).

**Example** (OpenFold2 Evoformer): resolve the script path from the installed package, then run:

```bash
python "$(python -c 'from tensorrt_bionemo import EXAMPLES_DIR; print(EXAMPLES_DIR / "openfold2" / "convert_evoformer_checkpoint.py")')" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${SAFETENSORS_PATH}" \
  --triangle_attn_backend CUEQUIV \
  --local_checkpoint "${CHECKPOINT_PATH}"
```

**Illustrative log lines:**

```text
2026-04-11 08:28:32,087 - tensorrt_bionemo.hubs.local - INFO - Loading alphafold2_1 from local filesystem /workspace/release_artifacts/checkpoint/alphafold2_1.pt
Total time of converting checkpoints: 00:00:01
```

For **other models**, use the conversion entrypoint and flags from your TensorRT-BioNeMo release (Boltz pairformer / token transformer, multimer flags, etc.).

**Verify:** under `${SAFETENSORS_PATH}` you should see the layout your release expects (commonly a `trt/` subtree with `config.json` and weight shards — see package docs).

### 2.3 Build TensorRT engines (`trtbnm-build`)

Point `--checkpoint_dir` at the **converted** directory from §2.2 (here `${SAFETENSORS_PATH}`). Use `--module "${MODULE_NAME}"` (not the model name unless they coincide in your release).

```bash
trtbnm-build \
  --model "${MODEL_NAME}" \
  --module "${MODULE_NAME}" \
  --checkpoint_dir "${SAFETENSORS_PATH}" \
  --max_seqlen "${MAX_SEQLEN}" \
  --min_seqlen "${MIN_SEQLEN}" \
  --output_dir "${ENGINES_PATH}" \
  --weakly_dtype "${DTYPE}"
```

**Illustrative log excerpts:**

```text
[TRT-LLM] [I] Building module evoformer for backend trt
[TRT-LLM] [W] Overriding # of builder profiles <= 2.
[TRT-LLM] [I] Module config dtype: float32, weakly_dtype: bfloat16
[TRT-LLM] [I] Building weakly-typed engine with dtype bfloat16.
[TRT] [I] [MemUsageChange] Init CUDA: ...
[TRT-LLM] [I] Dynamic input m with shape: [None, 516, None, 256], dtype: DataType.BF16
...
```

**Quick sanity check** (optional): add `--dry_run` and a temporary `--output_dir` per `trtbnm-build --help`.

> **Boltz-1, Boltz-2, OpenFold3 TRT engines: not documented in this release.** These models have multiple TRT-targetable modules (e.g. `structure_pairformer` + `confidence_pairformer` + `token_transformer` for Boltz; `pairformer` + `token_transformer` for OpenFold3) and their build/runtime wiring is on the deprecation track described in §0. **Contact the TRT-BNM developers** if you have a hard requirement to evaluate them.

### 2.4 Re-run the sample script with the TRT Evoformer engine

Once the engine exists under `${ENGINES_PATH}`, re-run the pipeline from [`README.md` §5](README.md#5-run-the-sample-pipeline-script) with `ENGINE_OUTPUT_DIR` pointed at the engine tree. The script auto-detects `trt/config.json` and wires the Evoformer module to TRT.

**Expected `ENGINE_OUTPUT_DIR` layout** (AF2 / OF2 — single module):

`<dir>/trt/config.json` **or** `<dir>/evoformer/trt/config.json` (the dir itself may be the engine dir for the single-module Evoformer case).

```bash
# AlphaFold2 monomer with the Evoformer TRT engine.
# NOTE: §0 recommends the PyTorch backend for all OF2 seqlens on Ampere /
# Hopper (A100 / H100 / H200, SM90 and earlier) — this example is here for
# reproducing the TRT path / benchmarking.
MODEL_NAME=alphafold2_1 \
ENGINE_OUTPUT_DIR=engines/evoformer_engines \
python scripts/run_pipeline.py

# AlphaFold2 multimer with TRT Evoformer (same layout as monomer).
MODEL_NAME=alphafold2_multimer_1 PIPELINE_REPEAT=3 \
ENGINE_OUTPUT_DIR=engines/evoformer_multimer_engines \
python scripts/run_pipeline.py
```

> **Boltz-1, Boltz-2, OpenFold3 with TRT engines: not documented here.** Re-running the pipeline with `ENGINE_OUTPUT_DIR=…` for those models is on the deprecation track described in §0. Use the optimized PyTorch backend (the [`README.md` §5](README.md#5-run-the-sample-pipeline-script) examples already do this by default), or contact the TRT-BNM developers if you need to evaluate the legacy multi-module TRT path.

### 2.5 TensorRT-only checklist (AF2 / OF2 reproducibility)

- [ ] Conversion produces the expected layout under `SAFETENSORS_PATH` (commonly `trt/config.json` + weight shards).
- [ ] `trtbnm-build` completes; engine artifacts present under `ENGINES_PATH` (often `trt/` with `rank0.engine` + `config.json`).
- [ ] Sample script re-run with `ENGINE_OUTPUT_DIR=…` succeeds; logs confirm the TRT Evoformer engine is wired (otherwise the script transparently falls back to PyTorch).

---

## 3. Programmatic API — model construction & inference

[`scripts/run_pipeline.py`](scripts/run_pipeline.py) is a thin wrapper around the Python API exposed by `tensorrt_bionemo`. If you are integrating one of the models into your own pipeline (training-loop eval, batch-prediction service, custom post-processor, …), construct the model directly instead of shelling out to the CLI. This section is the "hello world" for that path.

> **When to use this vs the CLI.** Use the CLI (`scripts/run_pipeline.py`) when you want a complete request → PDB pipeline with parser / tokenizer / featurizer / writer / Ray scaling already wired. Use the API below when you want a plain `nn.Module` you can call `forward()` on, hook into your own data loader, or compose with other models in the same Python process.

### 3.1 Hello-world: load a model and inspect it

```python
import torch
import tensorrt_bionemo  # noqa: F401  — import side-effect registers all model factories
from tensorrt_bionemo.registry import get_model_class

ModelCls = get_model_class("boltz-2")
model = ModelCls().cuda().eval()
print(type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e6, "M params")
```

Calling the constructor with no arguments builds the model from the factory's pretrained config **and** auto-loads weights via the hub resolver (local env var → Hugging Face). Pass `include_load_weights=False` to construct an empty `nn.Module` and load weights manually later.

> **No explicit registration needed.** `import tensorrt_bionemo` runs a one-shot `_init()` that registers every model factory and loads the CUDA / TensorRT plugin libraries. Any module that pulls in `tensorrt_bionemo.registry`, `tensorrt_bionemo.models.*`, etc. transitively imports the package, so the `get_*` helpers below work without further setup.

### 3.2 Registry tour — what each model exposes

The package-level `_init()` populates a global `ModelRegistry` (in `tensorrt_bionemo/registry.py`) with one factory per supported model. Five getters cover everything you need to build a from-scratch inference loop:

| Helper (in `tensorrt_bionemo.registry`) | Returns | Use it for |
|---|---|---|
| `get_model_class(name)` | `Type[nn.Module]` — the folding model itself | `model = Cls()`; supports `(config, model_name, include_load_weights)` |
| `get_tokenizer(name)` | A `TokenizerBase` instance | Convert an `InputRequest` into a tokenized record |
| `get_feature_factory(name)` | A `FeatureFactoryBase` instance | Build the feature dict the model's `forward()` expects |
| `get_postprocessor(name)` | `Type[PostProcessorBase]` | Turn raw model outputs into PDB / scores |
| `get_default_runtime_args(name)` | `dict` — recycling / sampling defaults | Forward-time kwargs (e.g. `recycling_steps=3` for Boltz/OF3) |

Per-model summary (kept in sync with each factory in `tensorrt_bionemo/registry.py`):

| Model key (`MODEL_NAME`) | `nn.Module` class | Default runtime args | TRT-accelerated modules (for §3.4) |
|---|---|---|---|
| `alphafold2_1` … `alphafold2_5` | `OpenFold2` | `{}` (none) | `evoformer` |
| `alphafold2_multimer_1` … `_5` | `OpenFold2` (multimer config) | `{}` | `evoformer` |
| `boltz-1` | `Boltz1` | `{recycling_steps: 3, num_sampling_steps: 200, diffusion_samples: 1}` | `structure_pairformer`, `confidence_pairformer`, `token_transformer` |
| `boltz-2` | `Boltz2` | same as Boltz-1 | same as Boltz-1 |
| `openfold3` | `OpenFold3` | same as Boltz-1 (mapped to `num_cycles` / `no_rollout_steps` / `no_rollout_samples`) | `pairformer`, `token_transformer` |

`boltz-2-affinity` is registered for completeness but its tokenizer / feature factory / post-processor raise `NotImplementedError` — it is **not** a supported pipeline target in this release.

### 3.3 Constructing each model family

All four model classes take the same `(config=None, model_name=None, include_load_weights=True)` constructor signature, so the snippets below differ only in the model key. Weights resolve via `tensorrt_bionemo.hubs.load_weights` (`tensorrt_bionemo/hubs/checkpoint.py`): local env var first, Hugging Face on miss. Set `<MODEL>_CKPT` (e.g. `ALPHAFOLD2_1_CKPT`, `BOLTZ2_CKPT`, `OPENFOLD3_CKPT` — see Appendix A) to pin a local file and skip the HF call.

```python
import os, torch
import tensorrt_bionemo  # noqa: F401  — auto-registers factories on import
from tensorrt_bionemo.registry import get_model_class

# 1. AlphaFold2 / OpenFold2 monomer
os.environ["ALPHAFOLD2_1_CKPT"] = "/checkpoints/alphafold2_1.pt"   # required (JAX-only upstream)
af2 = get_model_class("alphafold2_1")().cuda().eval()

# 2. AlphaFold2 multimer (same class, multimer-flagged config)
os.environ["ALPHAFOLD2_MULTIMER_1_CKPT"] = "/checkpoints/alphafold2_multimer_1.pt"
af2m = get_model_class("alphafold2_multimer_1")().cuda().eval()

# 3. Boltz-1 / Boltz-2 (auto-download from HF on first run; or pin BOLTZ{1,2}_CKPT)
b2 = get_model_class("boltz-2")().cuda().eval()

# 4. OpenFold3 (gated HF repo — see Appendix A; or pin OPENFOLD3_CKPT to skip HF)
of3 = get_model_class("openfold3")().cuda().eval()
```

`OpenFold3.__init__` also accepts `diffusion_samples=N` to fix the rollout sample count at construction time; for Boltz-1/2 the same knob lives in `runtime_args` (see §3.5).

### 3.4 Wiring TRT engines into a model in-process (`model.optimize`)

If you have already built TRT engines (§2 for AF2 / OF2; contact developers for Boltz / OF3 — see §0 deprecation notice), you can swap them into a live model **in-place** without re-creating it. Every supported model mixes in `OptimizedModuleSetterMixin` and exposes `.optimize(accelerated_configs, context_memory_allocator=None)` (defined in `tensorrt_bionemo/models/helper.py`).

```python
import tensorrt_bionemo  # noqa: F401  — auto-registers factories on import
from tensorrt_bionemo.configs import AcceleratedConfig
from tensorrt_bionemo.registry import get_model_class

model = get_model_class("alphafold2_1")().cuda().eval()

model.optimize({
    "evoformer": AcceleratedConfig(
        backend="trt",
        checkpoint="/engines/alphafold2_1",   # dir with trt/config.json + rank0.engine
    ),
})
```

For multi-module models (Boltz-1/2: `structure_pairformer` + `confidence_pairformer` + `token_transformer`; OpenFold3: `pairformer` + `token_transformer`), pass one entry per module. Modules omitted from the dict stay on PyTorch — that is exactly the partial-acceleration pattern `scripts/run_pipeline.py` uses (see `_build_processor_config` and `TRT_MODULE_BY_MODEL`). There is no separate `tensorrt_bionemo.compile()` entry point; `model.optimize(...)` is the single hook.

### 3.5 Forward pass — full Workflow-2 recipe

Once the model is loaded, you still need (a) a tokenized + featurized input dict and (b) a post-processor. The factory exposes both, so a complete in-process inference loop for one request looks like this — this is what [`scripts/run_pipeline.py`](scripts/run_pipeline.py) wraps in stages:

```python
import torch
import tensorrt_bionemo  # noqa: F401  — auto-registers factories on import
from tensorrt_bionemo.data.schemas import InputRequest, Polymer, MSARecord
from tensorrt_bionemo.registry import (
    get_model_class, get_tokenizer,
    get_feature_factory, get_postprocessor, get_default_runtime_args,
    load_metadata,
)

NAME = "boltz-2"

model     = get_model_class(NAME)().cuda().eval()
tokenizer = get_tokenizer(NAME)
features  = get_feature_factory(NAME)
post_cls  = get_postprocessor(NAME)
runtime_args = get_default_runtime_args(NAME)
# Boltz-1/2 only: ccd.pkl + mols/ — auto-downloaded on first call.
metadata = load_metadata(NAME)

request: InputRequest = {
    "input_id": "demo",
    "polymers": [Polymer(
        polymer_type="protein", chain_id=["A"],
        sequence="GSHMSL...",                      # your sequence
        msas=[MSARecord(path="msa.a3m", format="a3m")],
        paired_msas=[],
    )],
}

tokenized = tokenizer(request)
feats = features.generate_features(tokenized, metadata=metadata)
feats = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feats.items()}

with torch.inference_mode():
    raw = model(feats, **runtime_args)

structures = post_cls()(raw, request, output_dir="output", format="pdb")
print(structures)   # {"pdb": "output/demo.pdb", ...}
```

Notes:

- **OpenFold2 / AlphaFold2** use `get_feature_factory("alphafold2_1")` (monomer) or `get_feature_factory("alphafold2_multimer_1")` (multimer); the multimer feature factory expects paired MSAs on every chain. `runtime_args` is `{}` — recycling is controlled by `OpenFold2.config.no_recycling_iter` instead.
- **Boltz-1 / Boltz-2** require `metadata=load_metadata(name)` (CCD + mols). Set `BOLTZ_CCD_PATH` / `BOLTZ_MOL_DIR` to point at local copies, otherwise the loader downloads from `boltz-community/boltz-{1,2}` on first call.
- **OpenFold3** maps `runtime_args` to its own forward kwargs (`recycling_steps → num_cycles`, `num_sampling_steps → no_rollout_steps`, `diffusion_samples → no_rollout_samples`); see `OpenFold3Factory.get_default_runtime_args` for the contract.

### 3.6 When to step back up to the high-level pipeline

The recipe above is single-process and single-request. For batched / multi-GPU inference, parser-level retries, MSA caching, automatic engine fallbacks, and the `output/` writer (PDB + per-request `*_scores.json`), use `EngineProcessorConfig` + `build_processor` directly (both in `tensorrt_bionemo.pipeline.processor.engine_proc`) — the pattern is exactly the one in [`scripts/run_pipeline.py`](scripts/run_pipeline.py) `_build_processor_config` (search for `EngineProcessorConfig(`). That is the supported entry point for production deployments; the §3.5 recipe is intended for debugging, fine-tuning eval, and single-call integration into a larger Python codebase.

---

## Appendix A — Upstream weights & per-model checkpoint catalog

For reproducible or air-gapped setups, keep large checkpoints under `checkpoint/` inside the release bundle. With the single mount from [`README.md` §2](README.md#2-run-the-container-with-a-single-bind-mount), they are accessible at `/workspace/release_artifacts/checkpoint/` inside the container. Do not commit multi-gigabyte blobs to git.

### Where checkpoints come from (per model family)

| Model | Upstream source | How `tensorrt_bionemo` consumes it |
|---|---|---|
| **AlphaFold2 / OpenFold2** | Upstream OpenFold2 only ships **JAX** parameter files — no PyTorch checkpoint exists in the public repo. | **We ship a pre-converted PyTorch checkpoint** (e.g. `alphafold2_1.pt`, included under `checkpoint/` in this bundle). Use it directly via `${CHECKPOINT_PATH}`. To regenerate yourself, use the helper below. |
| **Boltz-1, Boltz-2** | Hugging Face Hub (public — `boltz-community/boltz-1`, `boltz-community/boltz-2`). | **Auto-downloaded** on first use when `${CHECKPOINT_PATH}` / `--local_checkpoint` is **not** provided. Set `HF_HOME` / `HUGGINGFACE_HUB_CACHE` to persist the cache under the mounted bundle (e.g. `checkpoint/hf_cache`). Provide a local `.pt` only if you need to pin a specific file or run air-gapped. |
| **OpenFold3** | Hugging Face Hub, **gated** repo (`OpenFold/OpenFold3`). | Same auto-download flow as Boltz, **but** the HF repo is access-controlled: first-time use fails with `GatedRepoError: 401 ... Access to model OpenFold/OpenFold3 is restricted` unless you (1) request access at <https://huggingface.co/OpenFold/OpenFold3> and are approved, then (2) authenticate the container — either `huggingface-cli login` (writes `~/.cache/huggingface/token`) or `export HF_TOKEN=<your-token>` before running [`README.md` §5](README.md#5-run-the-sample-pipeline-script). Air-gapped alternative: download once on a connected host, copy `checkpoints/of3-p2-155k.pt` to `checkpoint/`, and set `OPENFOLD3_CKPT=checkpoint/of3-p2-155k.pt` so the HF call is skipped entirely. |

> **Air-gapped / pinned deployments:** always pre-stage the weights under `checkpoint/` in the bundle and pass them explicitly via `${CHECKPOINT_PATH}` so nothing tries to reach Hugging Face at runtime. (For **OpenFold3**, this is the simplest workaround for the gated HF repo — see the OF3 row above for the one-time download + `OPENFOLD3_CKPT` flow.)

### JAX → PyTorch helper (AlphaFold2 only)

To regenerate a PyTorch checkpoint from upstream AF2 JAX params, use **`examples/openfold2/jax_to_pt.py`** shipped with this release, which calls OpenFold2's `import_jax_weights_` and writes a matching `.pt`:

```bash
# Inside the container (cwd = /workspace/release_artifacts), with the openfold2 Python package available:
python "$(python -c 'from tensorrt_bionemo import EXAMPLES_DIR; print(EXAMPLES_DIR / "openfold2" / "jax_to_pt.py")')" \
  --jax_path checkpoint/params/params_model_1.npz \
  --config_preset model_1 \
  --output_dir checkpoint
# Produces checkpoint/params_model_1.pt — point ALPHAFOLD2_1_CKPT at it.
```

The `config_preset` / JAX filename suffix (`model_1` … `model_5`, or the `_multimer*` variants) must match the upstream OpenFold2 config. The script itself is small and self-contained — reproduced below for reference (`examples/openfold2/jax_to_pt.py`):

```python
import argparse
import os

import torch
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_


def get_model_basename(model_path: str) -> str:
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def main(args):
    config = model_config(args.config_preset)
    model = AlphaFold(config)
    model_basename = get_model_basename(args.jax_path)
    model_version = "_".join(model_basename.split("_")[1:])
    import_jax_weights_(model, args.jax_path, version=model_version)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"{model_basename}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_path",
                        type=str,
                        help="Path to JAX checkpoint file",
                        default="params_model_1.npz")
    parser.add_argument("--config_preset",
                        type=str,
                        help="The corresponding config preset",
                        default="model_1")
    parser.add_argument("--output_dir",
                        type=str,
                        help="Path for output directory",
                        default="output")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
```

Requires the upstream [**OpenFold** Python package](https://github.com/aqlaboratory/openfold) (for `openfold.config`, `openfold.model.model.AlphaFold`, `openfold.utils.import_weights.import_jax_weights_`) and AF2 JAX params in `params_model_{1..5}.npz` / `params_model_{1..5}_multimer.npz` form. The output filename mirrors the JAX basename (e.g. `params_model_1.npz` → `params_model_1.pt`) — rename or symlink to `alphafold2_1.pt` (etc.) if you want it to line up with the env var naming used elsewhere in this guide.

### Supported checkpoints — HF repo + local env var

Name resolution order inside `tensorrt_bionemo`: **local env var (if set) → Hugging Face Hub** (see `tensorrt_bionemo/hubs/local.py` and `tensorrt_bionemo/hubs/hf.py`).

| Model key | HF repo_id | HF filename | Local env var |
|---|---|---|---|
| `alphafold2_1` | — (JAX-only upstream; ship local) | — | `ALPHAFOLD2_1_CKPT` |
| `alphafold2_2` | — | — | `ALPHAFOLD2_2_CKPT` |
| `alphafold2_3` | — | — | `ALPHAFOLD2_3_CKPT` |
| `alphafold2_4` | — | — | `ALPHAFOLD2_4_CKPT` |
| `alphafold2_5` | — | — | `ALPHAFOLD2_5_CKPT` |
| `alphafold2_multimer_1` | — | — | `ALPHAFOLD2_MULTIMER_1_CKPT` |
| `alphafold2_multimer_2` | — | — | `ALPHAFOLD2_MULTIMER_2_CKPT` |
| `alphafold2_multimer_3` | — | — | `ALPHAFOLD2_MULTIMER_3_CKPT` |
| `alphafold2_multimer_4` | — | — | `ALPHAFOLD2_MULTIMER_4_CKPT` |
| `alphafold2_multimer_5` | — | — | `ALPHAFOLD2_MULTIMER_5_CKPT` |
| `openfold2_ft2` | `nz/OpenFold` | `finetuning_2.pt` | `OPENFOLD2_FINETUNING_2_CKPT` |
| `openfold2_ft3` | `nz/OpenFold` | `finetuning_3.pt` | `OPENFOLD2_FINETUNING_3_CKPT` |
| `openfold2_ft4` | `nz/OpenFold` | `finetuning_4.pt` | `OPENFOLD2_FINETUNING_4_CKPT` |
| `openfold2_ft5` | `nz/OpenFold` | `finetuning_5.pt` | `OPENFOLD2_FINETUNING_5_CKPT` |
| `openfold2_no_templ_1` | `nz/OpenFold` | `finetuning_no_templ_1.pt` | `OPENFOLD2_NO_TEMPL_1_CKPT` |
| `openfold2_no_templ_2` | `nz/OpenFold` | `finetuning_no_templ_2.pt` | `OPENFOLD2_NO_TEMPL_2_CKPT` |
| `openfold2_no_templ_ptm_1` | `nz/OpenFold` | `finetuning_no_templ_ptm_1.pt` | `OPENFOLD2_NO_TEMPL_PTM_1_CKPT` |
| `openfold2_ptm_1` | `nz/OpenFold` | `finetuning_ptm_1.pt` | `OPENFOLD2_PTM_1_CKPT` |
| `openfold2_ptm_2` | `nz/OpenFold` | `finetuning_ptm_2.pt` | `OPENFOLD2_PTM_2_CKPT` |
| `boltz1` | `boltz-community/boltz-1` | `boltz1_conf.ckpt` | `BOLTZ1_CKPT` |
| `boltz2` | `boltz-community/boltz-2` | `boltz2_conf.ckpt` | `BOLTZ2_CKPT` |
| `boltz2_affinity` | `boltz-community/boltz-2` | `boltz2_aff.ckpt` | `BOLTZ2_AFFINITY_CKPT` |
| `openfold3` | `OpenFold/OpenFold3` | `checkpoints/of3-p2-155k.pt` | `OPENFOLD3_CKPT` |

**Usage:**

```bash
# Inside the container (cwd = /workspace/release_artifacts), relative paths resolve under the bundle.
export ALPHAFOLD2_1_CKPT=checkpoint/alphafold2_1.pt          # bundled with this release; no HF fallback
export OPENFOLD3_CKPT=checkpoint/of3-p2-155k.pt              # optional override; HF fallback available
export HF_HOME=checkpoint/hf_cache                           # persist HF downloads under the mounted bundle
```

If your organization publishes weights via an internal registry, download with your approved method and verify size/checksums per release notes.

---

## Appendix B — Environment variable reference

Every knob exposed by [`quick_start.sh`](quick_start.sh) and [`scripts/run_pipeline.py`](scripts/run_pipeline.py), in one place. The table is grouped by purpose; per-row values reflect the **resolution order**:

> **Precedence (lowest → highest):** built-in defaults in the script → values in [`default.yaml`](default.yaml) → environment variables → CLI flags forwarded to `run_pipeline.py` (`--model`, `--backend`, `--replicas`, …).
>
> Editing `default.yaml` is the recommended way to set repeatable defaults; environment variables are best for one-shot overrides (`MODEL_NAME=boltz-2 ./quick_start.sh`). Set `CONFIG_FILE=""` (empty) on the command line to bypass YAML loading entirely.

### Container & wheel (host-side, `quick_start.sh` only)

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `CONFIG_FILE` | — | `<script-dir>/default.yaml` | Path to YAML defaults file. Set to empty to skip YAML loading. |
| `NGC_TAG` | `ngc_tag` | `25.12` | NGC PyTorch image tag. Must match a `wheel/pytorch_<tag>/` subdir. Supported: `25.12`, `25.08`. |
| `WHEEL_ARCH` | `wheel_arch` | `$(uname -m)` | Override wheel CPU architecture (`x86_64` / `aarch64`). |
| `RUN_AS_ROOT` | `run_as_root` | unset | If non-empty, skip the host-uid `--user` remap. |
| `SKIP_PULL` | `skip_pull` | unset | If non-empty, skip `docker pull` (use locally cached image). |
| `EXTRA_DOCKER_ARGS` | `extra_docker_args` | unset | Extra args passed verbatim to `docker run` (e.g. `-v /host:/mnt`). |

### Pipeline target (forwarded into the container)

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `MODEL_NAME` | `model_name` | `alphafold2_1` | Model key. See [Appendix A](#supported-checkpoints--hf-repo--local-env-var) for the full list. Also exposed as `--model`. Boltz-2 affinity is out of scope. |
| `OUTPUT_DIR` | `output_dir` | `output` (relative to bundle) | PDB / scores output directory. Also exposed as `--output-dir`. |
| `PIPELINE_REPEAT` | `pipeline_repeat` | `1` | Number of passes through the bundled sample pool. Also exposed as `--repeat`. |
| `SAMPLES_DIR` | `samples_dir` | auto-detect (`data/samples/`) | Override sample-data root. |
| `REQUESTS_JSON` | `requests_json` | unset | Override the JSON request file (advanced; bypasses sample auto-detection). |

### Executor (Ray vs serial)

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `EXECUTOR_BACKEND` | `executor_backend` | `ray` | `"ray"` (data-parallel) or `"serial"` (in-process). Also exposed as `--backend`. |
| `REPLICAS` | `replicas` | all visible CUDA devices | Ray only: data-parallel engine actors. Also exposed as `--replicas`. |
| `NUM_GPUS_PER_REPLICA` | `num_gpus_per_replica` | `1.0` | Ray only: GPUs reserved per engine replica. Also exposed as `--num-gpus-per-replica`. |

### TensorRT (legacy AF2 / OF2 path — see §2)

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `ENGINE_OUTPUT_DIR` | `engine_output_dir` | unset (→ PyTorch backend) | TRT engine dir. Must contain `trt/config.json` (or `evoformer/trt/config.json`). AF2 / OF2 only — Boltz-1/2 / OpenFold3 multi-module TRT is not documented here. |
| `MODEL_NAME` | `model_name` | (above) | Build-time `trtbnm-build --model`. |
| `MODULE_NAME` | — | `evoformer` | Build-time `trtbnm-build --module`. (Set on the build command line; no quick-start equivalent.) |
| `SAFETENSORS_PATH` | — | (your choice, e.g. `engines/evoformer_safetensors`) | Build-time `--checkpoint_dir` source for `trtbnm-build`. |
| `ENGINES_PATH` | — | (your choice, e.g. `engines/evoformer_engines`) | Build-time `--output_dir` for `trtbnm-build`. Re-exported as `ENGINE_OUTPUT_DIR` at run time. |
| `CHECKPOINT_PATH` | — | bundled `checkpoint/<model>.pt` | Build-time upstream checkpoint path. **Same as the per-model `<MODEL>_CKPT` below** — the two coexist for back-compat: `CHECKPOINT_PATH` is conventional in the §2 conversion / build commands, `<MODEL>_CKPT` is what `tensorrt_bionemo`'s hub loader reads at run time. Set the latter for run-time overrides. |
| `MAX_SEQLEN`, `MIN_SEQLEN` | — | (your choice, e.g. `1536` / `16`) | Build-time dynamic seqlen bounds for `trtbnm-build`. |
| `DTYPE` | — | (your choice, e.g. `bfloat16`) | Build-time `--weakly_dtype` precision hint. |

### Hugging Face Hub

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `HF_TOKEN` | `hf_token` | unset | HF auth token. **Required** for the gated `OpenFold/OpenFold3` repo (see [Appendix A](#where-checkpoints-come-from-per-model-family)). Equivalent to a `huggingface-cli login`. |
| `HF_HOME` | `hf_home` | `<bundle>/.cache/huggingface` | HF cache override. The default persists downloads on the host across runs. |
| `HUGGING_FACE_HUB_TOKEN` | — | unset | Alternate auth env var read by `huggingface_hub`. |

### Per-model checkpoint overrides (run-time hub loader)

Set only if you want to bypass the bundled `checkpoint/` auto-detection (AF2) or HF auto-download (Boltz / OF3). All paths must be reachable from inside the container. The full per-model list is in [Appendix A](#supported-checkpoints--hf-repo--local-env-var); the most common are below.

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `ALPHAFOLD2_1_CKPT` … `ALPHAFOLD2_5_CKPT` | `alphafold2_1_ckpt` … `alphafold2_5_ckpt` | bundled `checkpoint/alphafold2_<n>.pt` | Override AF2 monomer weights. |
| `ALPHAFOLD2_MULTIMER_1_CKPT` … `ALPHAFOLD2_MULTIMER_5_CKPT` | `alphafold2_multimer_<n>_ckpt` | bundled (if present) | Override AF2 multimer weights. |
| `OPENFOLD2_*_CKPT` (FT2..5, no-templ, ptm) | `openfold2_*_ckpt` | HF auto-download (`nz/OpenFold`) | Override OpenFold2 fine-tuned variants. |
| `BOLTZ1_CKPT` | `boltz1_ckpt` | HF auto-download | Override Boltz-1. |
| `BOLTZ2_CKPT` | `boltz2_ckpt` | HF auto-download | Override Boltz-2. |
| `OPENFOLD3_CKPT` | `openfold3_ckpt` | HF auto-download (gated) | Override OpenFold3. **Recommended for air-gapped / pinned deployments** (avoids the gated HF repo). |

### Boltz metadata (auto-downloaded from HF if unset)

| Env var | YAML key | Default | Purpose |
|---|---|---|---|
| `BOLTZ_CCD_PATH` | `boltz_ccd_path` | HF auto-download (`boltz-community/boltz-1/ccd.pkl`) | CCD pickle for Boltz-1/2. |
| `BOLTZ_MOL_DIR` | `boltz_mol_dir` | HF auto-download (`boltz-community/boltz-2/mols.tar`, auto-extracted) | Mols directory for Boltz-2. |

### `CHECKPOINT_PATH` vs `<MODEL>_CKPT` — when to use which

These two env vars coexist by convention but cover slightly different lifecycle steps:

| Variable | Used by | When |
|---|---|---|
| `CHECKPOINT_PATH` | `trtbnm-build` and the §2.2 conversion commands | **Build-time only.** A convenient shell variable referenced by the §2 examples. Has no runtime effect on the pipeline. |
| `<MODEL>_CKPT` (e.g. `ALPHAFOLD2_1_CKPT`, `BOLTZ2_CKPT`) | `tensorrt_bionemo.hubs.local` at runtime — `scripts/run_pipeline.py`, the notebook | **Run-time hub loader.** Read every time the pipeline resolves weights for `MODEL_NAME=<name>`. |

For most §5/§6 PyTorch-backend runs you only ever need `<MODEL>_CKPT`. `CHECKPOINT_PATH` only matters during the legacy AF2 / OF2 TRT engine build (§2). They typically point at the same file.

---

## Appendix C — Data-pipeline support matrix

Coverage of the bundled data pipeline (`scripts/run_pipeline.py` + the `data/samples/` tree) across supported models. "Supported" means the pipeline in this release bundle accepts the input and produces a structure end-to-end; unsupported rows generally require extra features (templates, ligands, nucleic acids) that are **out of scope** for this release.

| Model | Monomer | Homo-oligomer | Heterooligomer (multimer) | Unpaired MSA | Paired MSA | Templates | Nucleic acids (DNA / RNA) | Small-molecule ligands | Ligand affinity |
|---|---|---|---|---|---|---|---|---|---|
| `alphafold2_{1..5}` (AF2 / OF2 monomer) | Yes | — | — | Required | — | **Not supported** | — | — | — |
| `alphafold2_multimer_{1..5}` (AF2 multimer) | — | Yes | Yes | Required | Optional | **Not supported** | — | — | — |
| `boltz-1` | Yes | Yes | Yes | Required | Optional | **Not supported** | **Not supported** | **Not supported** | — |
| `boltz-2` | Yes | Yes | Yes | Required | Optional | **Not supported** | **Not supported** | **Not supported** | — |
| `openfold3` | Yes | Yes | Yes | Required | Optional | **Not supported** | **Not supported** | **Not supported** | — |
| `boltz-2-affinity` | — | — | — | — | — | — | — | — | **Not supported** (out of scope for this release) |

### Limitations (apply to all supported models)

- **Protein polymers only.** Every `Polymer.polymer_type` in the bundled samples and every downstream stage assumes `"protein"`; nucleic-acid chains (DNA / RNA) are accepted by the schema but not exercised or validated by this bundle.
- **No templates.** Every `Polymer.templates` field in the bundled JSONs is `null`; the release pipeline does **not** stage HHsearch / HMMsearch template features, and running without templates is the only tested path.
- **No ligand / affinity inputs.** The Boltz-2 affinity pipeline (ligand-affinity prediction with CCD or custom small molecules) is **intentionally excluded** from this release; its data pipeline is not bundled, so `boltz-2-affinity` is rejected by `scripts/run_pipeline.py`. Boltz-1/2 and OpenFold3 protein-only predictions are fully supported.
- **MSA expectations per family.** AF2 monomer takes a single unpaired a3m per chain; AF2 multimer additionally **requires** paired a3m MSAs on every chain (enforced by `scripts/run_pipeline.py::create_sample_requests_of2_multimer`); Boltz-1/2 and OpenFold3 accept both shapes and treat paired MSAs as optional.

---

## Further reading

- Package builder: `trtbnm-build --help`
- Wheel layout and naming: see [`README.md` §1](README.md#1-pull-the-base-image)
- Checkpoint policy: see [Appendix A](#appendix-a--upstream-weights--per-model-checkpoint-catalog)
- Backend choice: see [§0](#0-choose-a-backend-pytorch-vs-tensorrt)
