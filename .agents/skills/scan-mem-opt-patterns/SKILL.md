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
name: scan-mem-opt-patterns
description: Diagnose and reduce activation memory in pairwise-representation structure models using precision, lifetime, never-materialize, compute-once conditioning, row-born pair state, and chunking; profile OOMs and measure large-N capacity with reproducible boundaries and provenance. Use when fitting longer inputs or porting memory optimizations across Boltz, OpenFold, Protenix, and related models. Use bench-perf-oss for full BioIR-versus-OSS folding comparisons.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Scan for Pairwise-Representation Memory-Optimization Patterns

Protein-structure models (Boltz, OpenFold2/3, Protenix, and most
AlphaFold-lineage ports) blow up on the **pairwise `[B, N, N, c]`
representation** at large token count `N`. The peak is a *stack* of `[N,N,*]`
activations, each `N²·c·sizeof(elem)` bytes (e.g. `[N,N,256]` fp32 ≈ 16 GB at
N≈4000). Fitting a longer sequence is almost never one big win — it is removing
that stack **one `[N,N,*]` tensor at a time**.

This skill is the checklist developed against **Boltz2**, **ProtenixV2**, and
**OpenFold3** H100 and A100 80 GB capacity work, written to **transfer**: walk
all four levers even when no keyword matches. The core profiler and chunk
engine are inlined. For a measured GPU campaign, also read
[benchmarking and profiling](references/benchmarking-and-profiling.md); it
defines claim boundaries, source pins, isolation, metrics, boundary closure,
numerical evidence, and experiment stopping rules.

## When to use

- "Reduce `<model>`'s memory" / "fit longer sequences / more residues" / "why
  does `<model>` OOM at large N".
- "Apply the Boltz / Protenix memory optimizations to `<model>`" / porting a new
  pairwise model.
- Multi-sample diffusion / confidence OOM (`diffusion_samples` / `N_sample` > 1)
  even when single-sample fits.
- "Profile memory", "find the next allocation wall", "measure maximum input
  size", or "establish a safe production limit" for a pairwise model.

## Choose the operating mode

- **Source scan only:** use the four levers, pattern catalog, and report
  template. Do not invent measured savings from tensor-size estimates.
- **OOM diagnosis or memory profile:** also read the profiling, allocator, and
  evidence sections of
  [benchmarking and profiling](references/benchmarking-and-profiling.md).
- **Optimization and capacity campaign:** read that reference in full before
  launching GPU attempts. Freeze the workload and source identity, change one
  lever at a time, and close a repeat-confirmed pass/OOM bracket.
- **BioIR-versus-OSS folding latency or quality:** use
  [`bench-perf-oss`](../bench-perf-oss/SKILL.md). A synthetic capacity workload
  has no folding ground truth and cannot support a quality claim.

## The four transferable levers

Every fix below is one of these. When nothing in the catalog matches, ask which
lever a given `[N,N,*]` tensor is missing:

- **L1 — Precision.** Store/compute the `[N,N,*]` tensor in **bf16, not fp32**
  (½ the bytes). Especially when a fp32 producer feeds a bf16 consumer (the fp32
  tensor is cast down anyway).
- **L2 — Lifetime.** A `[N,N,*]` tensor should be **resident only while it is
  read**. Shorten its life: accumulate **in place**, **recompute** a cheap
  tensor instead of holding it, build heavy temporaries in a **helper scope** so
  they free on return, **drop** feed_dict tensors once dead, **`del`** a stage's
  outputs before the next heavy stage.
- **L3 — Never materialize.** Don't build the big intermediate at all, and don't
  carry it in a form wider than anyone reads: **`one_hot@W → embedding`
  gather**, **broadcast/python-loop → `bmm`/`einsum`**, **inline** a use-once
  tensor into its consumer, **reduce at the producer** when every consumer only
  reads a reduction, feed a fused kernel a **strided** input instead of a
  padded/contiguous copy.
- **L4 — Chunking.** A **position-wise** op over `[N,N,*]` (`f(cat(a,b)) ==
  cat(f(a),f(b))`) can run in **row slices** → peak transient ≈ `chunk/N` of
  dense, **bit-identical**. Use the chunk engine (below).

Cross-cutting facts worth internalizing:

- **Shared code pays off.** Many fixes land in layer/module code reused across
  models — one edit can cover several models (e.g. Boltz1/2, OpenFold2/3,
  Protenix) at once. Prefer fixing the shared layer.
- **The peak is one or two stages.** Usually the trunk/MSA stack, the
  diffusion-conditioning stage, the multi-sample sampler / DiT, and the
  confidence stage. Profile per-stage first; spend the levers where the wall is.
- **`S` (diffusion samples) multiplies any `[N,N,*]` you expand over samples.**
  Prefer sample-independent pair tensors + broadcast in attention (P13) over
  materializing `[B·S,N,N,*]`.
- **Inference-only assumptions are allowed** here (no autograd), which unlocks
  in-place / `del` / `pop` that training could not do. Destructive feature
  ownership must be **opt-in** when the caller may retain the input dict (P6).
- **Do not hoist recycle-dependent modules** (template / MSA that read current
  `z`) outside the recycle loop for "memory" — holding their activations across
  cycles can *raise* the resident base. Only static feature-side projections are
  hoist candidates; profile first.
- **Caching is not automatically a memory optimization.** Computing a
  noise-invariant pair path once can remove repeated work, but its output stays
  resident throughout the rollout. It lowers memory only when the composition
  also shortens or stages another overlapping lifetime (P21).

## Workflow (any model)

Copy this checklist and track progress:

```text
- [ ] 1. Classify the claim: source scan, diagnostic profile, hard capacity,
         recommended capacity, latency, or quality. Do not merge them.
- [ ] 2. For GPU measurements, read the campaign reference and freeze hardware,
         source/diff, checkpoint, inputs, full workload, allocator, and outputs.
- [ ] 3. Map the target: the model forward; the trunk / MSA / pairformer
         stack; the diffusion-conditioning + token DiT + confidence stages;
         the rel-pos encoder; the featurizer; input and recycle pair
         construction. Note `N_sample` / `diffusion_samples`. Separate
         noise-invariant pair conditioning from time-dependent single
         conditioning. Identify the pairwise rep tensor(s) and which stage(s)
         drive the peak (profiler below).
- [ ] 4. Grep the model package for each pattern's `Detect` markers below
         -> a candidate list.
- [ ] 5. For each candidate: confirm it applies (read the hit + trace
         dtype/dataflow/lifetime), estimate GiB saved, note the fix and its
         lever. Do NOT edit yet.
- [ ] 6. Write the report (template below), ranked by GiB and by which
         stage's wall it moves.
- [ ] 7. Apply one top candidate; prove the optimized dispatch/lifetime path
         executes, verify numerically, then re-run the exact failed input and
         preceding pass under the frozen contract.
- [ ] 8. Preserve raw attempts, repeat only the terminal adjacent endpoints,
         and report the next wall plus any unverified quality or latency limit.
```

The `Detect` markers are grep hints — every hit is a *candidate*. Confirm by
reading the code: several patterns need dtype / dataflow / lifetime judgment
(e.g. "fp32 feeding a bf16 consumer", "still referenced when the next heavy
stage runs"), not just a keyword match.

## Profiling tools (find the peak stage / attribute the OOM)

Use the full campaign protocol in
[benchmarking and profiling](references/benchmarking-and-profiling.md) when a
number will be published. Three model-agnostic diagnostic techniques follow;
the first is inlined as a ready-to-use exploratory harness:

- **Per-submodule memory attribution** — register `forward_pre` / `forward`
  hooks that log `memory_allocated` + `max_memory_allocated` at each module
  boundary; the boundary where the running peak jumps to its max is the dominant
  consumer, and on OOM the last module entered is the culprit. At that boundary,
  dump the live CUDA tensors (deduped by storage) to name the resident set, and
  probe the feed_dict by tensor size for dead entries. (Harness below.)
- **Capacity boundary** — run the complete frozen workload once per fresh,
  OOM-isolated subprocess. Search coarse anchors, refine on a declared grid,
  then repeat both adjacent pass/OOM endpoints. A shortened recycle, diffusion,
  sample, head, or export schedule is a diagnostic only, never a capacity row.
- **Per-submodule timing tree** — NVTX ranges around the live submodules + CUDA
  events (or an nsys capture) for an inclusive per-module time tree with a true
  CPU-vs-GPU split — the compute hot stage, complementary to the memory view.

### Memory-hook harness (copy/adapt)

Hooks every module boundary, logs current + running-peak allocation, dumps the
live resident set on demand, and on OOM points at the last module entered. Adapt
`deep` to your model's peak stage; the wrappers touch nothing in the source tree
(they `register_*_hook` on live instances). This instrumentation is intrusive:
use a separate run for latency and remove the hooks before timing.

```python
import gc, torch
GIB = 1 / (1 << 30)
_LOG = []  # (phase, name, cur_gib, peak_gib)

def _mem():
    # Do not synchronize at every module boundary: that changes scheduling.
    return (torch.cuda.memory_allocated() * GIB,
            torch.cuda.max_memory_allocated() * GIB)

def _hooks(name):
    def pre(_m, _i):      _LOG.append(("enter", name, *_mem()))
    def post(_m, _i, _o): _LOG.append(("exit",  name, *_mem()))
    return pre, post

# deep = top-level stage names to also hook one level into
def install(model, deep=()):
    for name, child in model.named_children():
        p, q = _hooks(name)
        child.register_forward_pre_hook(p)
        child.register_forward_hook(q)
        if name in deep:
            for cn, gch in child.named_children():
                a, b = _hooks(f"{name}.{cn}")
                gch.register_forward_pre_hook(a)
                gch.register_forward_hook(b)
    # A stage invoked as a *method* (not __call__), e.g. a `.sample()`
    # sampler, is NOT caught by forward hooks -- wrap it instead:
    #   orig = sm.sample
    #   def w(*a, **k):
    #       _LOG.append(("enter", "sample", *_mem()))
    #       try: return orig(*a, **k)
    #       finally: _LOG.append(("exit", "sample", *_mem()))
    #   sm.sample = w

# attribute the resident set; call from a pre-hook at a boundary
def dump_cuda_tensors(tag, topn=30):
    seen = {}
    # gc only finds Python-referenced tensors (ok in inference)
    for o in gc.get_objects():
        try:
            if not (torch.is_tensor(o) and o.is_cuda):
                continue
            st = o.untyped_storage()  # dedupe by storage: views count once
            seen[st.data_ptr()] = (st.nbytes(), tuple(o.shape), str(o.dtype))
        except Exception:
            continue
    rows = sorted(seen.values(), reverse=True)
    total = sum(r[0] for r in rows) * GIB
    print(f"[cuda @ {tag}] {len(rows)} storages, {total:.2f} GiB")
    for nb, shp, dt in rows[:topn]:
        print(f"  {nb * GIB:7.3f} GiB  {str(shp):28s} {dt}")

def report(oom=None):
    peak = max(_LOG, key=lambda r: r[3], default=None)
    for ph, nm, cur, pk in _LOG:
        hit = peak and (ph, nm, pk) == (peak[0], peak[1], peak[3])
        star = "  <-- PEAK" if hit else ""
        print(f"{ph:>5} {nm:<32} cur {cur:6.1f}GiB  peak {pk:6.1f}GiB{star}")
    if peak:
        print(f"global peak {peak[3]:.1f} GiB at {peak[1]} ({peak[0]})")
    if oom is not None:
        entered = [r[1] for r in _LOG if r[0] == "enter"]
        print(f"OOM -- last module entered: {entered[-1] if entered else '?'}")

# install(model, deep=("trunk", "confidence_module"))
# try:     model(feed_dict, **runtime_args)
# except torch.OutOfMemoryError as e:
#     report(e)   # partial log + culprit stage
# else:
#     report()    # full per-module peak table
```

The `cur` where the running `peak` jumps to its max is the dominant consumer;
`dump_cuda_tensors` at that boundary names the tensors (shape/dtype) in the
resident base. Reproduce the wall with the default allocator first. A separate
`expandable_segments` run may test an allocator-history hypothesis, but cannot
define the production boundary or share a timing comparison.

Because that dump reports **storage** bytes against the **view's** shape, a row
whose byte count is far larger than its shape implies is a slice pinning a
bigger parent (P15) — worth chasing, since the fix is usually a one-line
reduce-at-producer rather than a code lever.

## Pattern catalog

Grouped by lever. Each entry: **Symptom → Detect (grep markers) → Fix →
Savings**. Sizes assume `[N,N,c]` at N≈4-5k on an 80 GB GPU. The names in
`Detect` are example symbols from the Boltz/OpenFold lineage — adapt to your
model's names.

### L1 — Precision (fp32 → bf16)

#### P1 — Trimul forced fp32 → vanilla dual-GEMM

- **Symptom:** a pairformer/MSA `TriangleMultiplication` runs fp32 (vanilla
  dual-GEMM, ~2× the bf16 fused path) because a `high_precision` flag was left
  at its `True` default (config propagation dropped).
- **Detect:** `trimul_high_precision`, `high_precision`, the pairformer-layer
  builder. Red flag: a module builds pairformer layers WITHOUT threading the
  precision flag from its config. Confirm at runtime: the trimul's
  high-precision dtype is `float32` and it dispatches to the vanilla (not fused)
  dual-GEMM.
- **Fix:** thread the precision flag (default `False`) config → module → layer;
  give the intermediate layer a `False` default as a safety net.
- **Savings:** ~½ the trimul transient + unblocks the fused path.

#### P2 — fp32 pair cond / bias / accumulator / cache feeding a bf16 consumer

- **Symptom:** a `[N,N,*]` producer, accumulator, or **long-lived cache** runs
  fp32 while its consumer is bf16, so the fp32 tensor is cast down anyway.
  Shapes:
  - **producer** — pair transitions or precomputed per-layer biases built fp32
    for a bf16 diffusion transformer.
  - **assembled accumulator** — an fp32 `[N,N,c_z]` built at `self.dtype` (e.g.
    the confidence z-init: norm + rel-pos + bonds + contact + single-to-pair)
    then fed to a bf16 pairformer — worse, held at fp32 across a per-sample
    loop.
  - **rollout-long cache** — diffusion shared-vars / pair conditioning cached
    once then held across hundreds of denoise steps (e.g. `pair_z`); if
    consumers are bf16, caching fp32 doubles the resident cost for the whole
    rollout.
- **Detect:** the pair-conditioning module, per-layer bias projections,
  distogram embedding, the pairformer dtype attr, shared-vars / prepare_cache
  paths, `dtype=`. Red flag: a `[N,N,*]` fp32 producer/accumulator/cache whose
  value is `.to(bf16)`'d downstream (or fed into a bf16 stack).
- **Fix:** build/cast at the consumer dtype. Producer: cast inputs in at the
  module boundary, expose the dtype as a config field. Accumulator: `z =
  z.to(<pairformer dtype>)` right after the last `+=` — before it's held/looped.
  Cache: **compute** pair conditioning in fp32 if quality requires it, then
  **store** the cached tensor in the consumer dtype (bf16). Do **not** blindly
  cast the conditioning transitions themselves to bf16 — that has caused
  measurable long-sequence lDDT regressions; keep compute fp32, cache bf16, and
  A/B before flipping persistent pair-*state* dtype (opt-in only).
- **Savings:** ½ the tensor (e.g. a per-layer `token_trans_bias`
  `[N,N,depth·heads]` 22→11 GB; the confidence z-init `[N,N,c_z]` ~8→4 GB held
  across the per-sample loop; cached `pair_z` ~16→8 GB resident for the whole
  diffusion rollout).

### L2 — Lifetime (free ASAP)

#### P3 — Out-of-place `[N,N,*]` accumulation

- **Symptom:** chained `z = z + X` on the pair rep keeps old + term + new
  simultaneously live (≈3×).
- **Detect:** `z = z +`, `z_init = z_init +`, `zij = zij +`.
- **Fix (inference):** in-place `z += X` on a **freshly-owned** tensor (e.g.
  right after a norm / broadcast-add that already allocated it). Keep shared
  modules out-of-place by default and thread an explicit ownership opt-in from
  the model-specific caller. Gate the mutation off when CUDA graph capture is
  active. Do **not** infer ownership from execution mode alone, and do **not**
  in-place a `forward()` input the caller reuses (it silently corrupts a second
  call).
- **Verify:** assert destination storage is reused only in the opted-in
  inference case; cover default and capture guards; compare against
  the original sequence of residual additions rather than an algebraically
  collapsed expression whose floating-point rounding differs. A global
  allocator peak may remain unchanged when an earlier stage dominates, so
  retain phase telemetry and an adjacent capacity-grid test.
- **Savings:** ~1 `[N,N,c]` per residual add lifetime. Reusing the accumulator
  can remove a later-stage spike and move capacity even when the whole-forward
  `max_memory_allocated()` is unchanged.

#### P4 — A tensor computed early but held across the whole trunk

- **Symptom:** a rel-pos encoding (or similar) is computed once for `z_init`,
  then kept live until the diffusion-conditioning stage — resident across the
  entire trunk / all recycles.
- **Detect:** `relative_position_encoding` (or the rel-pos symbol) used in the
  z-init AND passed to a later stage. More generally: any `[N,N,*]` local whose
  first and last uses straddle a heavy stage.
- **Fix:** fold it into `z_init`, free it (`= None`), and **recompute** it
  cheaply (nearly free after P8) just before the later consumer. Gate behind a
  `recompute_rel_pos` config flag. Deterministic → output unchanged.
- **Savings:** one `[N,N,token_z]` (~15 GB fp32 at N≈5k).

#### P5 — Heavy intermediate held in the caller scope (RAII) / multi-sample

- **Symptom:** a heavy `[N,N,*]` intermediate is created early and consumed late
  (or a small tensor is derived from it), so it stays resident across unrelated
  heavy work in between. Examples:
  - confidence pairformer inputs (distogram-embed + `pair_z`, ~16 GB fp32) built
    inline before the pairformer stack, which only needs the small bf16 derived
    tensor;
  - the confidence-heads `pae_logits` / `pde_logits` (`[N,N,num_bins]` ~4 GB
    each **+** their softmax aggregation temporaries) created at the top of the
    heads but consumed at the very end — resident across the whole plddt /
    complex-metric section (this is where the pae aggregation OOM'd at N~4k);
  - **multi-sample stack:** all `N_sample` PAE/PDE logit tensors (and full PAE
    probability volumes) retained until after the last sample / summary — at S=5
    this is ~`S`× one sample's logits overlapping trunk `z` / distogram (~90 GB
    class peak).
- **Detect:** a big `[N,N,*]` local whose creation and last use straddle other
  heavy work; `pae_logits`, `pde_logits`, distogram embedding,
  `compute_aggregated_metric`; confidence / summary loops that `stack` / `cat`
  over samples before reducing.
- **Fix:** move create+consume into a **helper** that returns only the small
  results; the heavy locals free on return (Python scope = RAII). If a
  *late-computed dependency* forces the late placement (e.g. a fallback that
  reused a value computed further down), **break it** (derive the fallback shape
  from an already-available tensor) so the helper can run up front and free
  early. For multi-sample confidence: **prepare once** (sample-independent
  `s`/`z`/masks), then **per-sample** `logits → summary_one_sample` (heavy
  `pae_prob` is a local), retain only compact scores; keep a batched `forward`
  only for tests / `return_raw_logits` debug. Optionally move compact results to
  CPU before the next sample.
- **Savings:** the heavy intermediate(s) — ~16 GB (pairformer inputs);
  ~2×`[N,N,num_bins]` + the aggregation temporaries per head (pae/pde);
  multi-sample path drops from `S`× logits to **one sample** at a time (~`S`×
  reduction of the confidence wall).

#### P6 — Dead feed_dict tensors / shallow-copy vs destructive ownership

- **Symptom:** the feed_dict carries big tensors that inference never reads, or
  reads once early then never again, yet they stay resident the whole forward.
  Two sub-cases:
  - **never-read training targets:** loss targets / atom-set maps
    (`disto_target` — one-hot distogram `[N,N,ens,bins]` ≈ 8 GB; center-atom /
    rep-atom maps ≈ 1 GB each).
  - **read-once-early feats:** raw pair feats folded into the z-init then dead
    (`contact_conditioning`, `contact_threshold`, `token_bonds`, `type_bonds`);
    also MSA/template / restype/profile fields after input embed; **`relp` / RPE
    one-hot** after its last consumer (diffusion pair conditioning) — otherwise
    it rides through the whole denoise + confidence wall.
- **Detect:** for each large key, grep the forward + confidence + postprocessor
  for *reads* (not the featurizer construction). Never-read ⇒ pop up front;
  read-once ⇒ pop right after the last read.
- **Fix (three sites):** (1) in the model forward, `feed_dict.pop(key, None)` —
  up front for never-read, after the last consumer for read-once; handles
  feed_dicts from an upstream (e.g. OSS) data pipeline that still emits them.
  (2) in **your featurizer**, stop building/emitting the never-read ones
  entirely. (3) **ownership:** popping a *shallow shell copy* does **not** free
  tensors the caller still holds — expose an opt-in `consume_input_features` /
  destructive mode that takes ownership of the caller's dict; default remains
  non-destructive for API compatibility. Pair with opt-in **compact output**
  (omit trunk `s`/`z`, raw logits, large contact matrices; keep writer-facing
  coords + summary) so stage outputs are not re-retained in the return dict
  (P7).
- **Savings:** ~10 GB (training targets) + a few `[N,N,c]` (read-once feats) +
  `relp` (~9 GB at N≈4k `c=139`) + trunk `z` / logits when compact (~16+4 GB).
  Destructive mode required to realize feature savings when the caller retains
  the input dict.
- **Caveat:** `pop`/in-place mutate the caller's dict — fine for one-shot
  inference (fresh feats per request), not for re-invoking the same module with
  the same dict. Keep defaults non-destructive.

#### P7 — Stage outputs held across a later heavy stage

- **Symptom:** a stage's big outputs (e.g. diffusion-conditioning
  `token_trans_bias` + `q`/`c`/atom biases; the sampler output dict) stay
  referenced (locals + a kwargs dict + the returned dict) through a later
  memory-heavy stage (confidence) that never reads them.
- **Detect:** find each big output's **last** read; check whether it's still
  referenced when the next heavy stage runs. Watch for a tensor kept alive by
  **three** refs (local, kwargs dict, output dict).
- **Fix:** extract only what the later stage / return dict needs, then `del` the
  tensors **and** the dicts holding them right after their last use. (If the
  later stage reads part of the dict, pull those tensors out first, then `del`
  the dict.)
- **Savings:** ~12 GB (conditioning tensors + sampler dict residue).

### L3 — Never materialize / layout

#### P8 — `one_hot` + `cat` + `Linear` (RPE / MSA / template features)

- **Symptom:** `F.one_hot(…) → torch.cat → Linear` builds huge int64 one-hots +
  a wide fp32/bf16 concat before a projection. Classic sites: relative-position
  encoder; **MSA symbol** projection (`one_hot(msa, 32)` + deletion cols →
  `[B,S,N,34]`); **template** pair features (`dgram + restype_i/j + …` → wide
  `[B,N,N,C_feat]`).
- **Detect:** `F.one_hot`, the rel-pos encoder, `relpos` / `relp`, MSA embedder,
  template embedder, `torch.cat` into a `Linear` whose first dim matches a
  one-hot width.
- **Fix:** `one_hot(idx) @ W ≡ gather of W's rows` → `F.embedding(idx,
  W.t()[slice])` (or `weight[:, :K].T`), accumulate geometric / continuous
  columns by multiplying their weight columns in place. Weight-row / column
  slices must match the **original concat column order** and leave checkpoint
  keys unchanged (slice at runtime). Prefer this over retaining compact bucket
  indices when recomputing pairwise integer differences is cheap.
- **Savings:** RPE int64 one-hots + fp32 concat (tens of GB); MSA at large `S`
  (e.g. S≈16k, N≈4k: int64 one-hot ~16 GB + bf16 copies); template
  `[B,N,N,C_feat]` per template (~7 GB at C≈108, N≈4k).

#### P9 — Broadcast / python-loop materialization of `[N,N,*]`

- **Symptom:** an `[N,N]`/`[N,N,*]` intermediate is built by broadcasting (`a[:,
  :, None] * b[:, None, :]`) or a python loop, only to be **reduced/contracted**
  afterwards.
- **Detect:** `[:, :, None]`, `[:, None, :]`, `torch.bmm`, `einsum`; mask
  products / norm counts / pair sums built then summed.
- **Fix:** fold into a single matmul / `einsum` / fused reduction that never
  forms the full broadcast (e.g. `num_mask = bmm(mask.T, mask)` instead of
  `(mask[:,:,None]*mask[:,None,:]).sum`). Cast integer masks to float for `bmm`.
- **Savings:** the `[N,N,*]` broadcast temporary.

#### P10 — Use-once big intermediate bound to a variable

- **Symptom:** a large fp32 feature (e.g. a Fourier embedding `[N,N,~120]`) is
  bound to a local, used exactly once (in a `cat`/add), but kept alive longer;
  or final masking builds several `[N,N,*]` temporaries via chained out-of-place
  ops.
- **Detect:** a big `[N,N,*]` local read exactly once downstream;
  `FourierEmbedding`, `fourier_embedding`, chained `* mask` / `+ enc` on the
  pair rep.
- **Fix:** **inline** the expression into its single consumer so it frees
  immediately after; do the final masking **in place** (`x *= m; x += a; x +=
  b`). Fuse elementwise (`.mul_().cos_()`).
- **Savings:** the inlined tensor(s) (~15 GB fp32 fourier) + the mask
  temporaries.

#### P11 — Fused-op input copies (`.contiguous()` / `F.pad`)

- **Symptom:** a fused kernel forces `.contiguous()` or `F.pad` on a large
  `[B,S,N,*]` input (a full-size copy just to satisfy the kernel).
- **Detect:** `.contiguous()`, `F.pad` near a fused-op call on S/N-sized
  tensors.
- **Fix:** make the kernel read the **strided / natural-extent** input and
  **predicate** the ragged tile instead of copying (usually only last-dim
  contiguity is actually required).
- **Savings:** the redundant full-size copies (tens of GB in deep-MSA).

### L4 — Chunking

#### P12 — Unchunked position-wise op over `[N,N,*]`

- **Symptom:** a `Transition`/FFN, outer-product-mean, pair-weighted-averaging,
  or triangle attention materializes its full internal activation
  (`[N,N,2·hidden]` ~25-30 GB; OPM's `[N,N,c_hidden²]`; PWA's `[H,S,N,D]`) in
  one shot. Also: legacy MSA `PairTransition` (ReLU) left dense while pairformer
  `transition_z` is already row-chunked.
- **Detect:** `Transition(`, `PairTransition`, `transition_z`,
  `OuterProductMean`, `PairWeightedAveraging`, `TriangleAttention`. Red flag: a
  pair-rep op called with no chunking wrapper; MSA stack using a different
  Transition class than the pairformer.
- **Fix:** wrap the op with the chunk engine (below). The op must be
  position-wise along the chunked dim → numerically identical. Give MSA
  `PairTransition` and diffusion pair-path `transition_z` their own row
  `ChunkPolicy`; keep dense fast path below threshold. **Do not** replace ReLU
  PairTransition with SwiGLU Transition (math + checkpoint differ). **Do not**
  chunk every diffusion Transition — invalid when axis 1 is the sample
  dimension.
- **Savings:** ~`chunk/N` of the op's transient (e.g. a pair FFN ~10×; MSA
  `[N,N,4·c_z]` at N≈4k ~33 GB → ~chunk/N).

### L2/L3 — Multi-sample diffusion & joint projection (Protenix-proven)

#### P13 — Sample-replicated pairwise conditioning for multi-sample diffusion

- **Symptom:** before a token DiT / pair-bias attention stack, the model expands
  sample-independent `z_pair` (or pair bias) over `diffusion_samples` /
  `N_sample` to `[B·S,N,N,c]`, then projects all layers' biases in one mega-GEMM
  and often `.contiguous()`-copies each layer. Peak scales with `S` even though
  pair geometry does not depend on the sample noise.
- **Detect:** `z_pair` / `pair_z` reshape or `expand` / `repeat` with
  `diffusion_samples`; `precompute_pair_biases` / mega bias projection; token
  transformer taking `[B*S,N,N,*]` pair bias while token activations are the
  only sample-varying input.
- **Fix:** keep pair conditioning **sample-independent** `[B,N,N,c]`; precompute
  one bias per layer as `[B,H,N,N]` (or equivalent); keep token activations as
  `[B,S,N,C]` through the global token DiT; **broadcast / reuse** pair bias and
  mask over `S` inside attention (SDPA/VANILLA: `[B,1,H,N,N]`; custom kernels:
  infer `mult=S` from Q's flattened batch vs unexpanded bias/mask batch);
  flatten `B*S` only after the token DiT for the atom decoder. Do not expand `z`
  just to feed the bias projector.
- **Savings:** at N≈4k, S=5, c=256 bf16: removes ~41 GB replicated `z_bs`,
  shrinks mega output ~61→12 GB, and avoids ~61 GB per-layer contiguous-copy
  lists (~100+ GB class). Remaining B=1 mega tensor (all layer biases as views)
  is a separate streaming tradeoff.

#### P14 — Joint `LayerNorm(cat(a,b))` + `Linear` materializing the concat

- **Symptom:** pair conditioning (or similar) does
  `Linear(LayerNorm(torch.cat([z_trunk, relpe], -1)))` and materializes a wide
  fp32 `[N,N,c_a+c_b]` (e.g. 512-ch → ~33 GB at N≈4k) before the GEMM.
- **Detect:** `torch.cat` of two `[N,N,*]` sources into `LayerNorm` then
  `Linear`; diffusion / pair-conditioning modules; channel count = sum of
  sources.
- **Fix:** compute **joint** LayerNorm statistics over both sources (without
  concatenating), fold the LN scale into each half of the Linear weight, run two
  GEMMs, and add the shared-mean correction. Checkpoint layout stays a single
  Linear; slice weights at runtime. Keep the projection / transitions in fp32 if
  bf16 transitions regress quality (see P2 cache note). Pair with row-chunking
  on the pair-path `transition_z` (P12) — do **not** apply the generic chunk
  policy to single-conditioning transitions whose axis 1 is the **sample** axis.
- **Savings:** the full concat (`[N,N,c_a+c_b]` fp32, e.g. ~33 GB) plus easier
  headroom for the following transition activation.

### L2/L3 — Reduce at the producer (Boltz-proven)

#### P15 — A wide tensor crosses stages, read only as a reduction

- **Symptom:** an early stage produces `[N,N,K]` and the model threads it
  through diffusion into a late stage, but every consumer immediately collapses
  the trailing dim to `[N,N]` (masked softmax sum, expectation over bins,
  argmax). The wide form is resident for the whole span while `1/K` of it is
  read. Canonical case: Boltz's `pred_distogram_logits` `[B,N,N,64]`, which the
  confidence heads read *only* as `(softmax(logits) * contact_mask).sum(-1)` — a
  per-pair contact probability.
- **Detect:** a producer output passed down a call chain (model forward → stage
  → module → head) where the first thing each consumer does is reduce the last
  dim; a head owning a bin-mask buffer (`contacts`, `boundaries`) that it
  applies to an argument it did not create; any `[N,N,K]` parameter whose only
  uses sit inside one reduction expression.
- **Fix:** compute the reduction at the producer (a
  `compute_contact_prob(logits)` next to `compute_distogram`), thread the
  `[N,N]` result through the signatures in place of the wide tensor, and delete
  the reduction plus its mask buffer from every consumer. Four details decide
  whether this is safe and whether it actually pays off:
  1. **Chunk the reduction** (L4) — a dense `softmax(logits.float())`
     transiently allocates a full fp32 copy of the very tensor you are
     eliminating. Softmax is independent per pair row, so row-chunk it and the
     transient drops to `chunk/N`.
  2. **Watch for slice aliasing** — when the producer returns
     `[B,N,N,n_distograms,K]` and the code keeps `out[:, :, :, 0]`, that slice
     is a **view** and pins the entire parent storage, so the real resident cost
     is `n_distograms×` the apparent shape. Reduce the slice on the spot and
     never bind the parent to a name.
  3. **Reduce before the sample repeat** — the reduction commutes with the
     multiplicity repeat, so reducing first turns an `S`× replication of
     `[N,N,K]` into an `S`× replication of `[N,N]`.
  4. **Keep the mask full width** — a `K`-long 0/1 vector, not a `[:k]` slice,
     so the sum over the reduced dim stays bit-identical to the consumers'
     original mask-then-sum.
- **Savings:** `K`× on the resident tensor, held across every stage in between —
  Boltz2 at N=3936, K=64 fp32: **3.97 GB → 62 MB**, freed across the whole
  diffusion rollout and the confidence stack, with peak transient bounded by the
  chunk.

### L2/L3/L4 — H100-proven composition and dispatch

#### P16 — Chunk executor retains chunks, then duplicates the full output

- **Symptom:** row chunking bounds the operator's internal activation, but the
  executor stores every completed chunk in a list and calls `torch.cat` at the
  end. The final concatenation needs a second full output while all chunks
  remain live, so the program can OOM immediately after the chunked operator
  succeeds.
- **Detect:** list comprehensions or `append` inside a chunk loop followed by
  `torch.cat(outs, ...)`; an OOM requesting approximately one full
  `[N,N,c]` output at the concatenation line.
- **Fix:** in this inference-only runtime, evaluate the first chunk, allocate
  the final contiguous output once with `new_empty`, copy that chunk into its
  destination slice, and write every later chunk directly into the same storage.
  Do not retain a separate backward implementation. Validate output extent,
  negative `cat_dim`, and tuple/list output type and arity.
- **Proof:** make `torch.cat` raise for compatible one-to-one inference outputs;
  weak-reference the first and previous copied chunks and require them dead
  before the next operator call. Cover tensor, tuple, list, negative
  `cat_dim`, row-expanding, mismatched-shape, and channels-last outputs.
- **Savings:** removes the final full-output copy and releases each temporary
  chunk after its destination copy.

#### P17 — A destination-writing fused op rejects a logically compatible layout

- **Symptom:** a fused elementwise/gated epilogue exists and can write into an
  owned destination, yet a dispatcher rejects tensors whose ranks or leading
  shapes differ before flattening. A logically row-aligned input such as
  `[1,N,N,K]` and destination `[N²,K]` silently falls back to
  `sigmoid(gate) * output`, materializing a full gate or product.
- **Detect:** shape/rank equality checks before `flatten`/`view`, a broad
  fallback around a fused op, and an OOM at the vanilla sigmoid/multiply rather
  than inside the fused kernel. Inspect the actual dispatch choice at runtime.
- **Fix:** first validate the kernel's real contract—device, dtype, inner
  dimensions, supported architecture, destination ownership, and required
  stride—then compare flattened leading row counts. Dispatch the existing
  destination-writing kernel when those logical counts match; preserve the
  fallback for unsupported cases.
- **Proof:** monkeypatch the vanilla fallback to raise, assert that the returned
  tensor aliases the supplied destination, and compare the fused result with the
  reference using a frozen dtype-specific tolerance. Keep an unsupported-shape
  fallback test.
- **Savings:** one full gate/product tensor, often several GiB at large `N`,
  without adding a kernel or changing model precision.

#### P18 — A cheap full-pair output is allocated before an independent heavy phase

- **Symptom:** a head computes a full `[N,N,K]` output early and retains it
  across an unrelated pairformer, confidence stack, or offload phase. The output
  is required eventually but not by the intervening work; its lifetime overlap
  can expose a later, non-monotonic OOM.
- **Detect:** trace creation and first consumer of distogram, PAE/PDE, contact,
  or other pair logits. Flag a creation that precedes a heavy stage which never
  reads it, especially when a cast makes the retained copy wider.
- **Fix:** move the projection after the heavy phase and recreate only a cheap
  cast or view when needed. Gate schedule changes to the measured large-input
  inference path; preserve prior behavior for CPU execution, small inputs, and
  CUDA graph capture unless each is independently validated.
  Clearing the allocator cache alone is not a fix while the output remains live.
- **Proof:** record call order in a focused test, compare the deferred and prior
  outputs exactly when operation order is unchanged, test every guard, and
  re-profile the previously failing input.
- **Savings:** removes the full output from the heavy phase's resident base; the
  peak reduction depends on whether that overlap was on the critical path.

#### P19 — CPU offload is collapsed into the default performance lane

- **Symptom:** a memory-optimized build appears slower than a GPU-resident
  baseline because large pair representations or pair-head logits cross PCIe
  inside `model.forward()`. The same run can still improve end-to-end pipeline
  time when the next consumer is on the CPU, so one timing boundary hides the
  tradeoff seen by the other.
- **Detect:** `to("cpu")`, host `copy_`, or a host archive followed by a later
  `to(device=cuda)` around `[S,N,N,C]` tensors; flags such as
  `offload_pairformer_outputs`; CPU-resident PAE/PDE outputs. Account for every
  direction. An archive copied GPU→CPU, restored once for projection, then
  emitted as CPU logits moves approximately
  `S*N^2*(2*C_pair*bytes_pair + C_heads*bytes_out)` bytes.
- **Fix:** keep explicit GPU-resident latency and CPU-offload capacity modes.
  Select a default only after a matched A/B establishes the relevant GPU
  headroom, forward latency, pipeline latency, and host-memory cost. Do not
  silently enable offload only in the candidate lane or treat the two modes as
  one aggregate. Consider a size gate only after measuring its crossover.
- **Proof:** on the same source revision and inputs, run the two modes in fresh,
  interleaved processes; assert the effective flags and output devices, verify
  numerical or file identity, and report peak allocated/reserved GPU memory,
  host RSS, synchronized forward time, and end-to-end pipeline time separately.
- **Savings/cost:** releases the offloaded pair archive and logits from the GPU
  resident set, but consumes host memory and interconnect bandwidth. The cost
  scales quadratically with `N` and linearly with sample count `S`.

#### P20 — CPU offload archives every sample before projecting final heads

- **Symptom:** sequential confidence offload still peaks in host RSS or spends
  avoidable time on transfers. Each completed BF16 `[N,N,C_pair]` sample is
  copied to an all-sample CPU archive, then copied back to the GPU for PAE/PDE
  projection, even though the final FP32 pair logits belong on the CPU.
- **Detect:** a per-sample Pairformer iterator followed by `copy_` into a host
  pair archive, then a second loop that selects each host sample, calls
  `to(cuda)`, projects the heads, and copies logits to CPU. Size the removable
  archive as `S*N^2*C_pair*bytes_pair`; account separately for the unavoidable
  final CPU outputs.
- **Fix:** in inference-only CPU-offload mode, project each completed GPU pair
  immediately and copy its logits into preallocated final-dtype CPU output
  slices before advancing the Pairformer iterator. Retain only small single
  representations needed by later heads. Preserve the old path for CPU
  execution, graph capture, and the explicit GPU-resident lane.
- **Proof:** make the all-sample archive builder raise, assert the streamed path
  executes and returns CPU final-dtype outputs, compare outputs exactly when
  arithmetic order is unchanged, and measure both owned RSS and device-wide
  GPU high-water on the prior host-guard input. A completed run may still be
  GPU-unsafe; classify host guard, hard OOM, and recommended margin separately.
- **Savings/cost:** removes one BF16 all-sample host archive plus its CPU-to-GPU
  restore, while final CPU logits remain `O(S*N^2*C_heads)`. It can lower host
  RSS and forward time without moving a trunk- or diffusion-limited GPU-safe
  boundary.

### L2/L3/L4 — Compute-once and row-born pair state (A100-proven)

#### P21 — Noise-invariant pair conditioning repeats every denoising step

- **Symptom:** diffusion conditioning computes the same `[N,N,*]` pair path
  on every denoising step even though only the single path depends on time or
  noise. Hundreds of identical pair transitions dominate runtime, while the
  repeated allocation schedule can keep an avoidable high-water mark.
- **Detect:** trace pair and single conditioning dependencies separately.
  Look for the combined conditioning module inside the per-step denoiser, pair
  projections or transitions that read only the trunk pair and static batch
  features, and no rollout-scoped prepared-pair argument.
- **Fix:** split the API into `prepare_pair(...)` and
  `forward_single(..., time_or_noise)` operations. Let the sampler or rollout
  owner build the pair result once, pass it to every step, and delete it after
  the last step. Keep the cache request-scoped; do not use a module-global
  dictionary that can outlive or cross-contaminate requests. For a long
  CPU-offload lane whose confidence stage later needs the original trunk pair,
  snapshot that original on the host, release the GPU source before building
  the prepared pair by rows, then delete the prepared pair before restoring
  the original. Apply P19 host guards and keep GPU-resident mode distinct.
- **Proof:** count one pair preparation per rollout and one single preparation
  per step; test consecutive requests for stale-cache reuse; exercise dense,
  CPU, and capture guards; compare complete outputs under the declared
  exact or tolerance contract. Measure initial cache construction and rollout
  peaks separately because the first can remain the local wall.
- **Savings/cost:** caching alone primarily removes repeated compute; the cache
  remains resident and is a memory win only when another overlapping lifetime
  is shortened or staged. In one pinned OpenFold3 A100 80 GB exact-mode run,
  the compute-once revision kept all five N=4,096 CIF hashes byte-identical and
  reduced synchronized forward time from 697.57 to 550.58 seconds (21.07%).
  Do not transfer that timing or threshold to another workload or GPU SKU.

#### P22 — Pair state is assembled or recycled through dense full-size outputs

- **Symptom:** an input embedder materializes full token-bond,
  relative-position, projection, cast, and out-of-place sum tensors before
  returning the final pair state; or each recycle computes
  `z_init + linear(norm(z))` into another full `[N,N,C]` output even though
  inference owns `z`.
- **Detect:** full-size `token_bonds_emb` or relative-position features followed
  by `z = z + ...`; a producer result cast immediately to the consumer dtype;
  dense recycle expressions whose old `z` is dead after the assignment.
- **Fix:** preallocate the final pair storage once in the consumer dtype. Build
  each row block at the producer precision, preserve the dense operation order,
  and copy the completed rows directly into final storage. For recycling, write
  normalized and projected rows back into inference-owned `z` only outside
  CUDA graph capture. Keep a dense path below a validated threshold and give
  each operation an independently tuned chunk policy; do not copy another
  runtime's fixed chunk size.
- **Allocator caveat:** a large local allocated-memory reduction can still raise
  the global reserved or device-wide peak when row-build allocation classes
  survive into the trunk. Reclaim the allocator cache only at a proven phase
  boundary after the row temporaries are dead, retain dense/capture behavior,
  and remeasure the complete workload before claiming capacity.
- **Numerical boundary:** row-wise execution can select a different backend path
  even when the expression is algebraically unchanged. Test the exact
  32-bit-index boundary and both sides when a tensor approaches `2^31`
  elements. OpenFold3 `[1,4096,4096,128]` changed end-to-end hashes on its row
  recycle path; that establishes path-dependent numerics, not a generally
  incorrect kernel. Its exact lane therefore keeps the dense path through
  N=4,096 and enables row recycling only above it.
- **Proof:** make the dense long-input fallback raise, assert final storage or
  owned recycle storage is reused, and cover dtype, row extent, CPU, and
  capture guards. Compare local rows and complete outputs where a dense
  reference fits, then remeasure the global peak and repeat adjacent capacity
  endpoints. Do not claim dense-reference equivalence above the tested range.
- **Savings/cost:** in the pinned OpenFold3 A100 80 GB CPU-offload campaign,
  row-born input construction reduced its isolated stage from 57.149 to
  21.166 GB. Row recycling added 0.11% forward time at N=4,480 and 0.31% at
  N=4,608 versus the conditioning-only revision. The final P21/P22 composition
  moved recommended-safe capacity from 4,288 to 4,736 tokens (+10.45%, or
  +21.99% pair area); N=4,800 completed but violated the declared headroom.
  This is a workload-specific capacity result, not a GPU-resident boundary.

## Chunk engine (inline, copy/adapt)

The reusable core for **L4**. `chunk_apply` runs a *position-wise* op in row
slices and keeps the dense result unchanged. During inference it allocates the
final result once and copies each completed chunk directly into its destination;
it never retains all chunks for a second full-output `torch.cat`. This skill
targets inference only; callers must not use this helper as a backward contract.
Inputs whose sliced extent does not match the primary pass through untouched,
and inputs below the threshold take one dense call.

```python
import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class ChunkPolicy:
    chunk_size: int = 512   # rows per slice along `dim` (<= 0 disables)
    # min_size: only chunk when the sliced-dim extent exceeds this (see note)
    # min_rank: only chunk tensors with >= this rank (skips rank-3 reps)
    min_size:   int = 2560
    dim:        int = 1     # slice inputs / concat outputs along this dim
    min_rank:   int = 4
    enabled:    bool = True

    def should_chunk(self, x):
        return (self.enabled and self.chunk_size > 0 and torch.is_tensor(x)
                and x.dim() >= self.min_rank and x.dim() > self.dim
                and x.shape[self.dim] > self.min_size)

def chunk_apply(fn, *chunked, policy, cat_dim=None, **passthrough):
    """Run a position-wise fn in row slices along policy.dim."""
    primary = chunked[0] if chunked else None
    if primary is None or not policy.should_chunk(primary):
        return fn(*chunked, **passthrough)  # dense fast path

    dim = policy.dim
    out_dim = policy.dim if cat_dim is None else cat_dim
    n = primary.shape[dim]

    def _slice(t, start, length):
        # Misaligned tensors, scalars, and None pass through untouched.
        if (t is None or not torch.is_tensor(t) or t.dim() <= dim
                or t.shape[dim] != n):
            return t
        return t.narrow(dim, start, length)

    spans = iter(
        (start, min(policy.chunk_size, n - start))
        for start in range(0, n, policy.chunk_size)
    )
    first_start, first_length = next(spans)
    first = fn(
        *[_slice(t, first_start, first_length) for t in chunked],
        **passthrough,
    )

    container_type = type(first) if isinstance(first, (tuple, list)) else None
    arity = len(first) if container_type is not None else 1

    def _outputs(value):
        if container_type is None:
            if not torch.is_tensor(value):
                raise TypeError("chunk outputs must be tensors")
            return [value]
        if not isinstance(value, container_type) or len(value) != arity:
            raise TypeError("chunk outputs changed type or length")
        if not all(torch.is_tensor(output) for output in value):
            raise TypeError("chunk output containers must hold tensors")
        return list(value)

    first_outputs = _outputs(first)

    def _output_dim(output):
        normalized_dim = out_dim if out_dim >= 0 else output.dim() + out_dim
        if not 0 <= normalized_dim < output.dim():
            raise IndexError(
                f"cat_dim {out_dim} is out of range for rank {output.dim()}"
            )
        return normalized_dim

    result_dims = [_output_dim(output) for output in first_outputs]
    reference_shapes = [tuple(output.shape) for output in first_outputs]

    def _validate(outputs):
        for output, result_dim, reference in zip(
            outputs, result_dims, reference_shapes, strict=True
        ):
            if output.dim() != len(reference):
                raise ValueError("chunk output rank changed")
            if any(
                output.shape[axis] != extent
                for axis, extent in enumerate(reference)
                if axis != result_dim
            ):
                raise ValueError("non-concatenated output extent changed")

    def _pack(outputs):
        return outputs[0] if container_type is None else container_type(outputs)

    def _cat_remaining(collected, remaining):
        for start, length in remaining:
            outputs = _outputs(fn(
                *[_slice(t, start, length) for t in chunked],
                **passthrough,
            ))
            _validate(outputs)
            for values, output in zip(collected, outputs, strict=True):
                values.append(output)
        return _pack([
            torch.cat(outputs, dim=result_dim)
            for outputs, result_dim in zip(collected, result_dims, strict=True)
        ])

    _validate(first_outputs)
    compatible = all(
        output.layout == torch.strided
        and output.is_contiguous()
        and output.shape[result_dim] == first_length
        for output, result_dim in zip(first_outputs, result_dims, strict=True)
    )
    if not compatible:
        # Preserve torch.cat's general contract for row-expanding functions and
        # alternate layouts such as channels-last.
        return _cat_remaining([[output] for output in first_outputs], spans)

    # Write compatible inference results directly into final storage instead
    # of retaining chunks for a later concatenation.
    def _allocate(output, result_dim, chunk_length):
        shape = list(output.shape)
        shape[result_dim] = n
        result = output.new_empty(shape)
        result.narrow(
            result_dim, first_start, chunk_length
        ).copy_(output)
        return result

    results = [
        _allocate(output, result_dim, first_length)
        for output, result_dim in zip(first_outputs, result_dims, strict=True)
    ]
    written = [(first_start, first_length)]
    del first_outputs, first

    for start, length in spans:
        current = fn(
            *[_slice(t, start, length) for t in chunked],
            **passthrough,
        )
        outputs = _outputs(current)
        _validate(outputs)
        compatible = all(
            output.is_contiguous()
            and output.shape[result_dim] == length
            and output.dtype == result.dtype
            and output.device == result.device
            for output, result, result_dim in zip(
                outputs, results, result_dims, strict=True
            )
        )
        if not compatible:
            collected = [
                [
                    result.narrow(result_dim, old_start, old_length).clone()
                    for old_start, old_length in written
                ] + [output]
                for result, result_dim, output in zip(
                    results, result_dims, outputs, strict=True
                )
            ]
            return _cat_remaining(collected, spans)
        for result, result_dim, output in zip(
            results, result_dims, outputs, strict=True
        ):
            result.narrow(result_dim, start, length).copy_(output)
        written.append((start, length))
        # Python otherwise keeps the previous RHS bound while evaluating the
        # next fn(...) call, retaining an extra row block.
        del output, outputs, current
    return _pack(results)
```

**To make an op chunkable:**

1. Split the dense body into a pure `_forward_impl(self, x, ...)` that is
   position-wise along one output dim.
1. In `forward`, `return chunk_apply(self._forward_impl, x,
   policy=self.chunk_policy, cat_dim=..., **kw)`.
1. Pick `dim`: the axis where output row `i` depends only on input row `i` (pair
   row `N`, sequence `S`, query row `I`). Verify `f(cat(a,b)) == cat(f(a),f(b))`
   before trusting it.

Notes: give each op its own `ChunkPolicy`; a central
`dict[name -> ChunkPolicy]` lets you retune thresholds globally instead of
hardcoding each layer. Resolve device-scaled defaults lazily so import never
initializes CUDA. Pair activations are O(N²), so `sqrt(device memory)` scaling
is a useful starting estimate for `min_size`, not a measured capacity claim;
validate each GPU SKU.

Start chunking disabled when a fused kernel already bounds the dominant
intermediate. Still inspect allocations *before* that kernel: a dense QKV
projection can OOM before memory-bounded attention runs. Enable guarded
query-row chunking only after the traceback and shape arithmetic identify that
specific wall, then measure its launch/latency cost.

## Allocator fragmentation: a separate diagnostic

Not all reserved memory is live tensor storage. The caching allocator can hold
reserved-but-unallocated blocks that do not satisfy a differently sized
request. A large reserved-but-unallocated value is an allocator-history
**signature**, not proof of fragmentation: live allocations can pin split
segments, and changing lifetime or scheduling can change reuse.

First reproduce the OOM under the default allocator in a fresh subprocess. If
the exact attempt reports a large unused reserve, run one distinct,
explicitly-labeled diagnostic with:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Use that result only to ask whether allocator policy moves the wall. Never mix
it into a production hard/recommended boundary, a before/after code delta, or a
latency comparison. Expanding segments can add virtual-memory remapping work
and change throughput. Record the exact environment and
`diagnostic_only=true`.

Do not claim that the flag lowered the live set merely because a run passed.
Compare allocated peaks and the resident-tensor inventory. A smaller input
failing after a larger input passes is likewise an allocator-history/lifetime
signal, not a monotonic capacity boundary and not proof that fragmentation
alone caused the failure.

For comparative latency work, keep both implementations on the default
allocator as required by [the benchmark protocol][bench-alloc].

[bench-alloc]: ../bench-perf-oss/measurement.md#allocator-flags--leave-pytorch-on-its-default

## Reduce-then-verify (when applying a found fix)

1. **Gate** dtype, chunk, in-place, destructive, and scheduling changes so
   unaffected small-input, CPU, capture, or default-API paths retain their prior
   behavior. Destructive ownership remains opt-in.
1. Apply **one lever at a time** on a dedicated branch or immutable source
   snapshot. Do not bundle a second optimization merely because the first
   exposes another wall.
1. **Prove the new path executes.** Make a fallback raise or count dispatches;
   assert storage aliasing for destination-writing paths; record call order for
   lifetime scheduling. Output equality alone can pass while the optimized path
   silently falls back.
1. **Verify numerically** on randomized, representative shapes before profiling.
   Use `torch.equal` when operation order is unchanged; otherwise freeze an
   operator- and dtype-specific `torch.testing.assert_close` contract before
   examining the capacity result. There is no universal BF16 tolerance.
   Initialize random weights, and clone inputs per call when either path mutates
   storage. Keep checkpoint keys unchanged.
1. Exercise the guards and API semantics: dense/small path, unsupported fused
   shape, CPU, batch shape, output type, CUDA graph capture, and
   caller-owned inputs as applicable. For the chunk engine, prove inference
   avoids `torch.cat`.
1. **Re-profile the exact prior failure and its preceding pass** under the same
   default-allocator contract. Record raw peak allocated, reserved, device-wide
   used memory, host RSS, output status, and the new failure site. Run allocator
   experiments only as separately labeled diagnostics.
1. When a wall moves, close the new adjacent grid endpoints with two fresh
   subprocesses each. Do not repeatedly scan an unaffected recommendation
   region, and do not run beyond the predeclared ceiling merely because the last
   point passed.
1. For quality-sensitive precision, fused-math, or operation-order changes, run
   a real-checkpoint full-workload A/B. Finite deterministic synthetic outputs
   establish execution integrity, not folding quality.

## Report template

Title it `# <Model> memory-opt scan`, then the findings table:

| #   | lever    | pattern                                       | location    | applies? | est. GB | fix (1 line)                     | risk |
| --- | -------- | --------------------------------------------- | ----------- | -------- | ------- | -------------------------------- | ---- |
| P1  | L1       | fp32 trimul                                   | <file:line> | yes/no   | ~X      | thread high_precision flag       | low  |
| P2  | L1       | fp32 pair cond / bias / accum / cache         | ...         | ...      | ...     | build/cast bf16; cache consumer  | low  |
| P3  | L2       | out-of-place [N,N,*] add                      | ...         |          |         | z += X (freshly-owned)           | low  |
| P4  | L2       | tensor held across trunk                      | ...         |          |         | recompute flag                   | low  |
| P5  | L2       | heavy intermediate / multi-sample             | ...         |          |         | RAII helper; stream per sample   | low  |
| P6  | L2       | dead feed_dict / ownership                    | ...         |          |         | pop + opt-in destructive/compact | low  |
| P7  | L2       | stage outputs held                            | ...         |          |         | del before next stage            | low  |
| P8  | L3       | one-hot+cat Linear (RPE/MSA/tmpl)             | ...         |          |         | embedding-gather; slice weights  | low  |
| P9  | L3       | broadcast/loop [N,N,*]                        | ...         |          |         | bmm / einsum                     | low  |
| P10 | L3       | use-once big intermediate                     | ...         |          |         | inline + in-place mask           | low  |
| P11 | L3       | fused-op input copy                           | ...         |          |         | strided read + predicate         | med  |
| P12 | L4       | unchunked [N,N,*] op                          | ...         |          |         | chunk_apply (pair-row only)      | low  |
| P13 | L2/L3    | sample-replicated pair bias                   | ...         |          |         | sample-indep bias; broadcast S   | med  |
| P14 | L3       | joint LN+Linear via cat                       | ...         |          |         | joint stats; dual GEMM; no cat   | med  |
| P15 | L2/L3    | wide tensor crosses stages, read as reduction | ...         |          |         | reduce at producer; chunk it     | low  |
| P16 | L2/L4    | chunk list + final cat                        | ...         |          |         | preallocate; copy each chunk     | low  |
| P17 | L3       | fused op rejects logical layout               | ...         |          |         | flatten rows; write destination  | med  |
| P18 | L2       | pair output spans heavy phase                 | ...         |          |         | defer output past heavy work     | low  |
| P19 | L2       | CPU offload hidden in performance lane        | ...         |          |         | split latency/capacity modes     | low  |
| P20 | L2/L3    | CPU pair archive restored before projection   | ...         |          |         | project sample into CPU outputs  | low  |
| P21 | L2       | repeated noise-invariant pair conditioning    | ...         |          |         | prepare once per rollout         | low  |
| P22 | L2/L3/L4 | dense input/recycle pair construction         | ...         |          |         | row-born final storage           | med  |

Close with a `## Notes` section answering:

- Pairwise rep tensor(s): `<name/shape>`. Peak-driving stage(s): `<trunk MSA /
  diffusion cond / DiT / confidence>`.
- Multi-sample factor S: `<N_sample>`. Any `[N,N,*]` expanded over S?
- Cross-stage args: any `[N,N,K]` threaded between stages that every consumer
  only reads reduced (P15)?
- Conditioning invariance: which pair work is noise-independent, who owns its
  rollout-scoped cache, and are construction and rollout peaks reported
  separately (P21)?
- Row-born storage: is the final pair destination inference-owned and in the
  consumer dtype? Are allocator phase boundaries measured globally (P22)?
- Large-index numerics: does any path approach `2^31` elements, and were the
  exact boundary and both sides tested?
- Shared vs model-specific: which fixes land in shared layer/module code (cover
  multiple models)?
- Ownership / compact-output flags: default-safe vs opt-in destructive?
- Claim type and boundary: estimate, diagnostic profile, hard capacity,
  recommended capacity, latency, or quality.
- Frozen identity: source revision/diff, checkpoint, workload, input, allocator,
  hardware, and harness.
- If measured: immutable attempt artifacts, repeat counts, adjacent pass/OOM
  bracket, headroom rule, observed deltas, cleanup, and next failure site.
- Numerical scope: exact, locally tolerance-bounded, deterministic/finite, or
  real-data quality-tested; never silently promote one level to another.
- Recommended order (biggest / earliest wall first).
- Rejected, with the reason: hoist recycle-dependent template/MSA; global
  conditioning cache; copied donor chunk sizes; unscored BF16 conditioning;
  chunk sample-axis transitions; atom coords as the main wall.
