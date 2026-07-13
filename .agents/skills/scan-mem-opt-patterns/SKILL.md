---
name: scan-mem-opt-patterns
description: Scan a pairwise-representation structure model (Boltz1/2, OpenFold2/3, or a new port) for the activation-memory patterns proven while fitting Boltz2 ~4000 residues on 80 GB, and port the fixes. Organized around four transferable levers — precision (fp32→bf16 trimul / pair-conditioning / per-layer bias / assembled fp32 accumulators cast to the consumer dtype), lifetime (in-place [N,N,*] accumulation, recompute-instead-of-hold, RAII-scoped heavy intermediates incl. head logits, drop dead feed_dict tensors, del stage outputs), never-materialize (one-hot→embedding-gather, broadcast/loop→bmm/einsum, inline-into-consumer, strided fused-op inputs), and chunking (position-wise [N,N,*] ops in row-slices) — so the same reasoning transfers to any model whose peak is the pairwise [N,N,c] tensor. Also covers the CUDA caching-allocator fragmentation lever (PYTORCH_CUDA_ALLOC_CONF=expandable_segments) and a per-submodule memory-attribution profiler. All reusable code is inlined; no repo-specific paths. Use when asked to reduce a model's activation memory, fit longer sequences, diagnose an OOM at large N, or apply the Boltz memory optimizations to another model.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Scan for Pairwise-Representation Memory-Optimization Patterns

Protein-structure models (Boltz, OpenFold2/3, and most AlphaFold-lineage ports) blow up on the
**pairwise `[B, N, N, c]` representation** at large token count `N`. The peak is a *stack* of
`[N,N,*]` activations, each `N²·c·sizeof(elem)` bytes (e.g. `[N,N,128]` fp32 ≈ 8 GB at N≈4000).
Fitting a longer sequence is almost never one big win — it is removing that stack **one `[N,N,*]`
tensor at a time**.

This skill is the checklist that let Boltz2 reach ~4000 residues on an 80 GB GPU, written to
**transfer**: the pattern catalog is grouped under four *levers*, and for a new model you walk the
levers even when no keyword matches. It is self-contained — the reusable code (profiler harness,
chunk engine) is inlined; nothing here depends on a specific repo layout.

## When to use

- "Reduce `<model>`'s memory" / "fit longer sequences / more residues" / "why does `<model>` OOM at large N".
- "Apply the Boltz memory optimizations to `<model>`" / porting a new pairwise model.

## The four transferable levers

Every fix below is one of these. When nothing in the catalog matches, ask which lever a given
`[N,N,*]` tensor is missing:

- **L1 — Precision.** Store/compute the `[N,N,*]` tensor in **bf16, not fp32** (½ the bytes).
  Especially when a fp32 producer feeds a bf16 consumer (the fp32 tensor is cast down anyway).
- **L2 — Lifetime.** A `[N,N,*]` tensor should be **resident only while it is read**. Shorten its
  life: accumulate **in place**, **recompute** a cheap tensor instead of holding it, build heavy
  temporaries in a **helper scope** so they free on return, **drop** feed_dict tensors once dead,
  **`del`** a stage's outputs before the next heavy stage.
- **L3 — Never materialize.** Don't build the big intermediate at all: **`one_hot@W → embedding`
  gather**, **broadcast/python-loop → `bmm`/`einsum`**, **inline** a use-once tensor into its
  consumer, feed a fused kernel a **strided** input instead of a padded/contiguous copy.
- **L4 — Chunking.** A **position-wise** op over `[N,N,*]` (`f(cat(a,b)) == cat(f(a),f(b))`) can run
  in **row slices** → peak transient ≈ `chunk/N` of dense, **bit-identical**. Use the chunk engine
  (below).

Cross-cutting facts worth internalizing:

- **Shared code pays off.** Many fixes land in layer/module code reused across models — one edit can
  cover several models (e.g. Boltz1/2 and OpenFold2/3) at once. Prefer fixing the shared layer.
- **The peak is one or two stages.** Usually the trunk/MSA stack, the diffusion-conditioning stage,
  and the confidence stage. Profile per-stage first; spend the levers where the wall is.
- **Inference-only assumptions are allowed** here (no autograd), which unlocks in-place / `del` /
  `pop` that training could not do.

## Workflow (any model)

Copy this checklist and track progress:

```
- [ ] 1. Map the target: the model forward; the trunk / MSA / pairformer stack; the
         diffusion-conditioning + confidence stages; the rel-pos encoder; the featurizer.
         Identify the pairwise rep tensor(s) and which stage(s) drive the peak (profiler below).
- [ ] 2. Grep the model package for each pattern's `Detect` markers below -> a candidate list.
- [ ] 3. For each candidate: confirm it applies (read the hit + trace dtype/dataflow/lifetime),
         estimate GB saved, note the fix and its lever. Do NOT edit yet.
- [ ] 4. Write the report (template below), ranked by GB and by which stage's wall it moves.
- [ ] 5. Apply top candidates one at a time; verify numerically, then re-profile (see below).
```

The `Detect` markers are grep hints — every hit is a *candidate*. Confirm by reading the code:
several patterns need dtype / dataflow / lifetime judgment (e.g. "fp32 feeding a bf16 consumer",
"still referenced when the next heavy stage runs"), not just a keyword match.

## Profiling tools (find the peak stage / attribute the OOM)

Three model-agnostic techniques; the first is inlined below as a ready-to-use harness:

- **Per-submodule memory attribution** — register `forward_pre` / `forward` hooks that log
  `memory_allocated` + `max_memory_allocated` at each module boundary; the boundary where the running
  peak jumps to its max is the dominant consumer, and on OOM the last module entered is the culprit.
  At that boundary, dump the live CUDA tensors (deduped by storage) to name the resident set, and
  probe the feed_dict by tensor size for dead entries. (Harness below.)
- **Fit / max-length test** — run the real pipeline at increasing problem sizes, each in an
  **OOM-isolated subprocess** (so an OOM can't corrupt the driver), reporting peak alloc/reserved;
  descend from the largest and stop at the first that fits. A few sampling/recycle steps suffice —
  the peak is per-step.
- **Per-submodule timing tree** — NVTX ranges around the live submodules + CUDA events (or an nsys
  capture) for an inclusive per-module time tree with a true CPU-vs-GPU split — the compute hot
  stage, complementary to the memory view.

### Memory-hook harness (copy/adapt)

Hooks every module boundary, logs current + running-peak allocation, dumps the live resident set on
demand, and on OOM points at the last module entered. Adapt `deep` to your model's peak stage; the
wrappers touch nothing in the source tree (they `register_*_hook` on live instances).

```python
import gc, torch
GB = 1 / 1e9
_LOG = []  # (phase, name, cur_gb, peak_gb)

def _mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() * GB, torch.cuda.max_memory_allocated() * GB

def _hooks(name):
    def pre(_m, _i):      _LOG.append(("enter", name, *_mem()))
    def post(_m, _i, _o): _LOG.append(("exit",  name, *_mem()))
    return pre, post

def install(model, deep=()):  # deep = top-level stage names to also hook one level into
    for name, child in model.named_children():
        p, q = _hooks(name); child.register_forward_pre_hook(p); child.register_forward_hook(q)
        if name in deep:
            for cn, gch in child.named_children():
                a, b = _hooks(f"{name}.{cn}")
                gch.register_forward_pre_hook(a); gch.register_forward_hook(b)
    # A stage invoked as a *method* (not __call__), e.g. a `.sample()` sampler, is NOT caught by
    # forward hooks -- wrap it instead:
    #   orig = sm.sample
    #   def w(*a, **k):
    #       _LOG.append(("enter", "sample", *_mem()))
    #       try: return orig(*a, **k)
    #       finally: _LOG.append(("exit", "sample", *_mem()))
    #   sm.sample = w

def dump_cuda_tensors(tag, topn=30):  # attribute the resident set; call from a pre-hook at a boundary
    seen = {}
    for o in gc.get_objects():                       # only finds Python-referenced tensors (ok in inference)
        try:
            if torch.is_tensor(o) and o.is_cuda:
                st = o.untyped_storage()             # dedupe by storage so views count once
                seen[st.data_ptr()] = (st.nbytes(), tuple(o.shape), str(o.dtype))
        except Exception:
            continue
    rows = sorted(seen.values(), reverse=True)
    print(f"[cuda @ {tag}] {len(rows)} storages, {sum(r[0] for r in rows) * GB:.2f} GB")
    for nb, shp, dt in rows[:topn]:
        print(f"  {nb * GB:7.3f} GB  {str(shp):28s} {dt}")

def report(oom=None):
    peak = max(_LOG, key=lambda r: r[3], default=None)
    for ph, nm, cur, pk in _LOG:
        star = "  <-- PEAK" if peak and (ph, nm, pk) == (peak[0], peak[1], peak[3]) else ""
        print(f"{ph:>5} {nm:<32} cur {cur:6.1f}G  peak {pk:6.1f}G{star}")
    if peak:
        print(f"global peak {peak[3]:.1f} GB at {peak[1]} ({peak[0]})")
    if oom is not None:
        entered = [r[1] for r in _LOG if r[0] == "enter"]
        print(f"OOM -- last module entered: {entered[-1] if entered else '?'}")

# install(model, deep=("trunk", "confidence_module"))
# try:     model(feed_dict, **runtime_args)
# except torch.OutOfMemoryError as e:  report(e)     # partial log + culprit stage
# else:    report()                                  # full per-module peak table
```

The `cur` where the running `peak` jumps to its max is the dominant consumer; `dump_cuda_tensors` at
that boundary names the tensors (shape/dtype) in the resident base. Run under `expandable_segments`
(below) so the peak reflects live tensors, not fragmentation.

## Pattern catalog

Grouped by lever. Each entry: **Symptom → Detect (grep markers) → Fix → Savings**. Sizes assume
`[N,N,c]` at N≈4-5k on an 80 GB GPU. The names in `Detect` are example symbols from the Boltz/OpenFold
lineage — adapt to your model's names.

### L1 — Precision (fp32 → bf16)

#### P1 — Trimul forced fp32 → vanilla dual-GEMM

- **Symptom:** a pairformer/MSA `TriangleMultiplication` runs fp32 (vanilla dual-GEMM, ~2× the bf16
  fused path) because a `high_precision` flag was left at its `True` default (config propagation dropped).
- **Detect:** `trimul_high_precision`, `high_precision`, the pairformer-layer builder. Red flag: a
  module builds pairformer layers WITHOUT threading the precision flag from its config. Confirm at
  runtime: the trimul's high-precision dtype is `float32` and it dispatches to the vanilla (not fused)
  dual-GEMM.
- **Fix:** thread the precision flag (default `False`) config → module → layer; give the intermediate
  layer a `False` default as a safety net.
- **Savings:** ~½ the trimul transient + unblocks the fused path.

#### P2 — fp32 pair conditioning / bias / accumulator feeding a bf16 consumer

- **Symptom:** a `[N,N,*]` producer or accumulator runs fp32 while its consumer is bf16, so the fp32
  tensor is cast down anyway. Two shapes:
  - **producer** — pair transitions or precomputed per-layer biases built fp32 for a bf16 diffusion
    transformer.
  - **assembled accumulator** — an fp32 `[N,N,c_z]` built at `self.dtype` (e.g. the confidence z-init:
    norm + rel-pos + bonds + contact + single-to-pair) then fed to a bf16 pairformer — worse, held at
    fp32 across a per-sample loop.
- **Detect:** the pair-conditioning module, per-layer bias projections, distogram embedding, the
  pairformer dtype attr, `dtype=`. Red flag: a `[N,N,*]` fp32 producer/accumulator whose value is
  `.to(bf16)`'d downstream (or fed into a bf16 stack).
- **Fix:** build/cast at the consumer dtype. Producer: cast inputs in at the module boundary, expose
  the dtype as a config field. Accumulator: `z = z.to(<pairformer dtype>)` right after the last `+=` —
  before it's held/looped.
- **Savings:** ½ the tensor (e.g. a per-layer `token_trans_bias` `[N,N,depth·heads]` 22→11 GB; the
  confidence z-init `[N,N,c_z]` ~8→4 GB held across the per-sample loop).

### L2 — Lifetime (free ASAP)

#### P3 — Out-of-place `[N,N,*]` accumulation

- **Symptom:** chained `z = z + X` on the pair rep keeps old + term + new simultaneously live (≈3×).
- **Detect:** `z = z +`, `z_init = z_init +`, `zij = zij +`.
- **Fix (inference):** in-place `z += X` on a **freshly-owned** tensor (e.g. right after a norm /
  broadcast-add that already allocated it). Verify the accumulator isn't aliased/needed elsewhere —
  do **not** in-place a `forward()` input the caller reuses (it silently corrupts a second call).
- **Savings:** ~1 `[N,N,c]` per chained add (clears the z-init spike).

#### P4 — A tensor computed early but held across the whole trunk

- **Symptom:** a rel-pos encoding (or similar) is computed once for `z_init`, then kept live until
  the diffusion-conditioning stage — resident across the entire trunk / all recycles.
- **Detect:** `relative_position_encoding` (or the rel-pos symbol) used in the z-init AND passed to a
  later stage. More generally: any `[N,N,*]` local whose first and last uses straddle a heavy stage.
- **Fix:** fold it into `z_init`, free it (`= None`), and **recompute** it cheaply (nearly free after
  P8) just before the later consumer. Gate behind a `recompute_rel_pos` config flag. Deterministic →
  output unchanged.
- **Savings:** one `[N,N,token_z]` (~15 GB fp32 at N≈5k).

#### P5 — Build-once heavy intermediate held in the caller scope (RAII)

- **Symptom:** a heavy `[N,N,*]` intermediate is created early and consumed late (or a small tensor
  is derived from it), so it stays resident across unrelated heavy work in between. Examples:
  - confidence pairformer inputs (distogram-embed + `pair_z`, ~16 GB fp32) built inline before the
    pairformer stack, which only needs the small bf16 derived tensor;
  - the confidence-heads `pae_logits` / `pde_logits` (`[N,N,num_bins]` ~4 GB each **+** their softmax
    aggregation temporaries) created at the top of the heads but consumed at the very end — resident
    across the whole plddt / complex-metric section (this is where the pae aggregation OOM'd at N~4k).
- **Detect:** a big `[N,N,*]` local whose creation and last use straddle other heavy work;
  `pae_logits`, `pde_logits`, distogram embedding, `compute_aggregated_metric`.
- **Fix:** move create+consume into a **helper** that returns only the small results; the heavy
  locals free on return (Python scope = RAII). If a *late-computed dependency* forces the late
  placement (e.g. a fallback that reused a value computed further down), **break it** (derive the
  fallback shape from an already-available tensor) so the helper can run up front and free early.
- **Savings:** the heavy intermediate(s) — ~16 GB (pairformer inputs); ~2×`[N,N,num_bins]` + the
  aggregation temporaries per head (pae/pde).

#### P6 — Dead feed_dict tensors resident for the whole forward

- **Symptom:** the feed_dict carries big tensors that inference never reads, or reads once early then
  never again, yet they stay resident the whole forward. Two sub-cases:
  - **never-read training targets:** loss targets / atom-set maps (`disto_target` — one-hot distogram
    `[N,N,ens,bins]` ≈ 8 GB; center-atom / rep-atom maps ≈ 1 GB each).
  - **read-once-early feats:** raw pair feats folded into the z-init then dead
    (`contact_conditioning`, `contact_threshold`, `token_bonds`, `type_bonds`).
- **Detect:** for each large key, grep the forward + confidence + postprocessor for *reads* (not the
  featurizer construction). Never-read ⇒ pop up front; read-once ⇒ pop right after the last read.
- **Fix (two sites):** (1) in the model forward, `feed_dict.pop(key, None)` — up front for never-read,
  after the z-init for read-once; handles feed_dicts from an upstream (e.g. OSS) data pipeline that
  still emits them. (2) in **your featurizer**, stop building/emitting the never-read ones entirely.
- **Savings:** ~10 GB (training targets) + a few `[N,N,c]` (read-once feats).
- **Caveat:** `pop`/in-place mutate the caller's dict — fine for one-shot inference (fresh feats per
  request), not for re-invoking the same module with the same dict.

#### P7 — Stage outputs held across a later heavy stage

- **Symptom:** a stage's big outputs (e.g. diffusion-conditioning `token_trans_bias` + `q`/`c`/atom
  biases; the sampler output dict) stay referenced (locals + a kwargs dict + the returned dict)
  through a later memory-heavy stage (confidence) that never reads them.
- **Detect:** find each big output's **last** read; check whether it's still referenced when the next
  heavy stage runs. Watch for a tensor kept alive by **three** refs (local, kwargs dict, output dict).
- **Fix:** extract only what the later stage / return dict needs, then `del` the tensors **and** the
  dicts holding them right after their last use. (If the later stage reads part of the dict, pull
  those tensors out first, then `del` the dict.)
- **Savings:** ~12 GB (conditioning tensors + sampler dict residue).

### L3 — Never materialize / layout

#### P8 — `one_hot` + `cat` + `Linear` (relative-position encoder)

- **Symptom:** `F.one_hot(d_*) → torch.cat → Linear` builds huge int64 one-hots + a fp32 concat.
- **Detect:** `F.one_hot`, the rel-pos encoder, `relpos`.
- **Fix (tp_size==1):** `one_hot(idx) @ W ≡ gather of W's rows` → `F.embedding(idx, W.t()[slice])`,
  accumulate in place; keep the one-hot path as the tensor-parallel fallback. Weight-row slices must
  match the original concat column order.
- **Savings:** the int64 one-hots (`[N,N,2·r_max+2]` ×2) + fp32 concat — tens of GB.

#### P9 — Broadcast / python-loop materialization of `[N,N,*]`

- **Symptom:** an `[N,N]`/`[N,N,*]` intermediate is built by broadcasting
  (`a[:, :, None] * b[:, None, :]`) or a python loop, only to be **reduced/contracted** afterwards.
- **Detect:** `[:, :, None]`, `[:, None, :]`, `torch.bmm`, `einsum`; mask products / norm counts /
  pair sums built then summed.
- **Fix:** fold into a single matmul / `einsum` / fused reduction that never forms the full broadcast
  (e.g. `num_mask = bmm(mask.T, mask)` instead of `(mask[:,:,None]*mask[:,None,:]).sum`). Cast
  integer masks to float for `bmm`.
- **Savings:** the `[N,N,*]` broadcast temporary.

#### P10 — Use-once big intermediate bound to a variable

- **Symptom:** a large fp32 feature (e.g. a Fourier embedding `[N,N,~120]`) is bound to a local, used
  exactly once (in a `cat`/add), but kept alive longer; or final masking builds several `[N,N,*]`
  temporaries via chained out-of-place ops.
- **Detect:** a big `[N,N,*]` local read exactly once downstream; `FourierEmbedding`,
  `fourier_embedding`, chained `* mask` / `+ enc` on the pair rep.
- **Fix:** **inline** the expression into its single consumer so it frees immediately after; do the
  final masking **in place** (`x *= m; x += a; x += b`). Fuse elementwise (`.mul_().cos_()`).
- **Savings:** the inlined tensor(s) (~15 GB fp32 fourier) + the mask temporaries.

#### P11 — Fused-op input copies (`.contiguous()` / `F.pad`)

- **Symptom:** a fused kernel forces `.contiguous()` or `F.pad` on a large `[B,S,N,*]` input (a
  full-size copy just to satisfy the kernel).
- **Detect:** `.contiguous()`, `F.pad` near a fused-op call on S/N-sized tensors.
- **Fix:** make the kernel read the **strided / natural-extent** input and **predicate** the ragged
  tile instead of copying (usually only last-dim contiguity is actually required).
- **Savings:** the redundant full-size copies (tens of GB in deep-MSA).

### L4 — Chunking

#### P12 — Unchunked position-wise op over `[N,N,*]`

- **Symptom:** a `Transition`/FFN, outer-product-mean, pair-weighted-averaging, or triangle attention
  materializes its full internal activation (`[N,N,2·hidden]` ~25-30 GB; OPM's `[N,N,c_hidden²]`;
  PWA's `[H,S,N,D]`) in one shot.
- **Detect:** `Transition(`, `transition_z`, `OuterProductMean`, `PairWeightedAveraging`,
  `TriangleAttention`. Red flag: a pair-rep op called with no chunking wrapper.
- **Fix:** wrap the op with the chunk engine (below). The op must be position-wise along the chunked
  dim → numerically identical.
- **Savings:** ~`chunk/N` of the op's transient (e.g. a pair FFN ~10×).

## Chunk engine (inline, copy/adapt)

The reusable core for **L4**. `chunk_apply` runs a *position-wise* op in row-slices along one dim and
concatenates — bit-identical to the dense call, but bounds the internal activation to `chunk_size`
rows. It only slices tensors whose sliced-dim extent matches the primary input; `None` / scalars /
already-reduced biases pass through untouched, and it falls back to a single dense call below the
size threshold (identical graph, zero overhead).

```python
import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class ChunkPolicy:
    chunk_size: int = 512   # rows per slice along `dim` (<= 0 disables)
    min_size:   int = 2560  # only chunk when the sliced-dim extent exceeds this (see note)
    dim:        int = 1     # slice inputs / concat outputs along this dim
    min_rank:   int = 4     # only chunk tensors with >= this many dims (skips cheap rank-3 reps)
    enabled:    bool = True

    def should_chunk(self, x):
        return (self.enabled and self.chunk_size > 0 and torch.is_tensor(x)
                and x.dim() >= self.min_rank and x.dim() > self.dim
                and x.shape[self.dim] > self.min_size)

def chunk_apply(fn, *chunked, policy, cat_dim=None, **passthrough):
    """Evaluate a POSITION-WISE fn in row-slices along policy.dim and concatenate."""
    primary = chunked[0] if chunked else None
    if primary is None or not policy.should_chunk(primary):
        return fn(*chunked, **passthrough)                     # dense fast path
    dim = policy.dim
    out_dim = policy.dim if cat_dim is None else cat_dim
    n = primary.shape[dim]

    def _slice(t, start, length):
        if (t is None or not torch.is_tensor(t) or t.dim() <= dim
                or t.shape[dim] != n):                          # misaligned / scalar / None -> pass through
            return t
        return t.narrow(dim, start, length)

    outs = [fn(*[_slice(t, s, min(policy.chunk_size, n - s)) for t in chunked], **passthrough)
            for s in range(0, n, policy.chunk_size)]
    if isinstance(outs[0], (tuple, list)):                      # concat element-wise for multi-output fns
        return type(outs[0])(torch.cat([o[i] for o in outs], dim=out_dim)
                             for i in range(len(outs[0])))
    return torch.cat(outs, dim=out_dim)
```

**To make an op chunkable:**

1. Split the dense body into a pure `_forward_impl(self, x, ...)` that is position-wise along one
   output dim.
1. In `forward`, `return chunk_apply(self._forward_impl, x, policy=self.chunk_policy, cat_dim=..., **kw)`.
1. Pick `dim`: the axis where output row `i` depends only on input row `i` (pair row `N`, sequence
   `S`, query row `I`). Verify `f(cat(a,b)) == cat(f(a),f(b))` before trusting it.

Notes: give each op its own `ChunkPolicy`; a central `dict[name -> ChunkPolicy]` lets you retune
thresholds globally at runtime instead of hardcoding per layer. **Memory-scale `min_size`** rather
than fixing it: pair activations are O(N²), so the largest N that fits before chunking scales
`~sqrt(total_mem)` — anchor at a reference GPU (e.g. 2560 residues @ 80 GB) and scale to the device,
resolved lazily on the first forward so import never initializes CUDA. Disable chunking for ops that
already have a memory-bounded kernel (e.g. flash-attention triangle attention).

## Runtime lever: allocator fragmentation (`expandable_segments`)

Not all "used" memory is live tensors — the CUDA caching allocator also holds **reserved-but-
unallocated** blocks it can't reuse for a differently-sized request (fragmentation), which at large
`N` can be **tens of GB**. Before (or alongside) any code lever, try:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It lets the allocator grow/relocate segments instead of stranding them. In the Boltz2 4k run it
reclaimed **~19 GB** (reserved-but-unallocated `19 GB → 0.1 GB`) and moved the OOM wall from the
*first* confidence pairformer layer all the way into the confidence heads — turning a fragmentation
failure into a true allocation ceiling. **Read the OOM message:** a large `"X GB reserved but unallocated"` ⇒ fragmentation (try this flag first); a small one ⇒ genuinely out of live memory (need
a code lever from the catalog). Zero-code and a sensible runtime default here — keep it on while
profiling so the per-stage peak reflects live tensors, not fragmentation.

## Reduce-then-verify (when applying a found fix)

1. **Gate** dtype/chunk/in-place changes so the aligned / `tp_size>1` / small-N path stays
   byte-identical (compile-time constant, `if`-gate, or size threshold).
1. **Verify numerically** on a small shape *before* profiling: a hermetic equivalence test comparing
   the new path to the original (`torch.testing.assert_close`, fp32 `~1e-4`, bf16 `~3e-3`). **Init
   random weights** — zero/default-initialized layers make equivalence tests vacuously pass. If a
   module accumulates into its inputs in place, **clone inputs per call** in the test.
1. **Re-profile** per-stage peak (`torch.cuda.max_memory_allocated`) to record the GB delta and
   confirm the wall actually moved (with `expandable_segments` to isolate fragmentation).
1. Apply **one lever at a time** on a dedicated branch so each delta is attributable.

## Report template

```
# <Model> memory-opt scan

| #   | lever | pattern                         | location     | applies? | est. GB | fix (1 line)                | risk |
|-----|-------|---------------------------------|--------------|----------|---------|-----------------------------|------|
| P1  | L1    | fp32 trimul                     | <file:line>  | yes/no   | ~X      | thread high_precision flag  | low  |
| P2  | L1    | fp32 pair cond / bias / accum   | ...          | ...      | ...     | build/cast bf16             | low  |
| P3  | L2    | out-of-place [N,N,*] add        | ...          |          |         | z += X (freshly-owned)      | low  |
| P4  | L2    | tensor held across trunk        | ...          |          |         | recompute flag              | low  |
| P5  | L2    | heavy intermediate in scope     | ...          |          |         | RAII helper                 | low  |
| P6  | L2    | dead feed_dict tensors          | ...          |          |         | pop (fwd + featurizer)      | low  |
| P7  | L2    | stage outputs held              | ...          |          |         | del before next stage       | low  |
| P8  | L3    | rel-pos one-hot                 | ...          |          |         | embedding-gather            | low  |
| P9  | L3    | broadcast/loop [N,N,*]          | ...          |          |         | bmm / einsum                | low  |
| P10 | L3    | use-once big intermediate       | ...          |          |         | inline + in-place mask      | low  |
| P11 | L3    | fused-op input copy             | ...          |          |         | strided read + predicate    | med  |
| P12 | L4    | unchunked [N,N,*] op            | ...          |          |         | chunk_apply                 | low  |

## Notes
- Pairwise rep tensor(s): <name/shape>. Peak-driving stage(s): <trunk MSA / diffusion cond / confidence>.
- Shared vs model-specific: which fixes land in shared layer/module code (cover multiple models)?
- Recommended order (biggest / earliest wall first): ...
```
