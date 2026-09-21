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

# Benchmark and profile pairwise activation memory

Read this reference for GPU profiling, an optimization campaign, or a capacity
claim. A source-only pattern scan does not need it. Prefer a model-specific
repository runbook when one exists; this file defines the transferable
invariants it must preserve.

## Keep claim types separate

Name the claim before collecting data:

- **Size estimate:** tensor shape times element size. Useful for ranking, but
  not a measured saving or capacity result.
- **Diagnostic profile:** instrumented run used to locate a stage, live tensor,
  allocation, transfer, or allocator-history problem.
- **Hard capacity:** largest tested grid point that completes the full workload,
  adjacent to a repeat-confirmed CUDA OOM under the same contract.
- **Recommended capacity:** largest repeat-confirmed pass that also meets
  predeclared GPU and host headroom margins.
- **Latency:** GPU-synchronized `model.forward()` timing after the declared
  warmup, without profiling instrumentation.
- **Quality:** real-checkpoint, real-input scoring appropriate to the model.

Do not merge these claims. In particular:

- a synthetic stress input has no folding ground truth;
- a zero-warmup capacity duration is not steady-state latency;
- an instrumented profiler run is not a latency sample;
- an allocator-diagnostic pass is not a production capacity result; and
- finite, deterministic outputs do not establish folding quality.

Use [bench-perf-oss](../../bench-perf-oss/SKILL.md) for a complete
BioIR-versus-OSS folding latency and quality comparison.

## Freeze the experiment contract

Freeze the contract before the first expensive attempt. If an identity field
changes, start a new campaign revision; do not reuse an earlier pass.

### Source and runtime identity

Record:

- branch role, commit, tree, imported package path, and whether the worktree is
  clean;
- an exact binary diff digest when testing an uncommitted candidate;
- submodule revisions, checkpoint path and SHA-256, strict-load status, and
  model/config identity;
- Python, Torch, CUDA, driver, container or environment lock, and dependency
  versions;
- native extension, kernel pack, CUBIN, and benchmark-harness digests when they
  affect dispatch; and
- the exact allocator environment, command, thread counts, and relevant
  environment variables.

For an immutable candidate, pin a baseline commit plus a binary patch, a
tracked-source manifest, and their digests. Build manifests from tracked and
intentional candidate files; exclude ignored caches such as `__pycache__`,
compiled leftovers, logs, and raw outputs. Verify source identity before and
after every attempt.

### Hardware identity

Record actual values rather than assuming the SKU name implies them:

- GPU name, UUID, compute capability, visible bytes, driver, and topology;
- MIG, ECC, persistence mode, power limit, maximum/application clocks, and
  in-forward power, clocks, temperature, and throttling when available; and
- host `MemTotal`, preflight `MemAvailable`, NUMA placement, swap state, and
  pinned-memory limit when CPU offload is involved.

An H100 replay is a new SKU campaign, not a scaled A100/Ada threshold. A
`sqrt(memory)` estimate may choose the first probe, but only measurements set
the boundary.

### Workload and input identity

Freeze every multiplier and output obligation:

- effective pretrained config and its delta from the official default;
- batch size, serial/parallel mode, precision, kernels, allocator, and seed;
- recycles/model cycles, diffusion steps, diffusion samples, and confidence
  mode;
- all required heads, postprocessing, exports, and output validation;
- input or sequence digest, tokens, residues, atoms, chains, MSA rows, paired
  MSA rows, templates, ligands, and chemical classes as applicable; and
- the search grid, reviewed upper ceiling, timeout, host guard, and recommended
  headroom formula.

Baseline and candidate must run the same full workload. Reducing recycles,
steps, samples, MSA depth, templates, heads, or exports is a separate scenario,
not a memory optimization.

When a real-MSA input is lengthened synthetically, preserve every source MSA
row and transform query and aligned columns by the same deterministic rule.
Record source and transformed row counts, aligned widths, and digests for each
chain. Capture the processed model-input MSA and mask shapes too, and reject a
completion or OOM whose processed depth collapses to one; attaching an A3M
path alone does not prove the pipeline consumed non-query rows. Label the
result synthetic capacity evidence: tiled MSA columns do not create structural
ground truth or support a folding-quality claim.

## Isolate every capacity attempt

One attempt owns one input size and one fresh subprocess.

1. Acquire one absolute GPU lock shared by every lane.
2. Inside the lock, require no unexpected compute PID, sufficient free GPU
   memory, and a predeclared host-memory floor.
3. Launch a new process group. Capture stdout, stderr, exit code, timestamps,
   exact command/environment, and a sidecar GPU/host telemetry stream.
4. Run the complete request once. Never retry after OOM in the same process.
5. Write to a unique immutable attempt directory. Never overwrite a success or
   failure.
6. On exit, terminate only the owned process group if necessary, wait for
   cleanup, and prove that no owned child or GPU process remains.
7. Recheck source identity and classify the terminal state.

Only a genuine CUDA allocator failure is `oom`. Keep these distinct:
`ok`, `oom`, `timeout`, host guard, external GPU process, setup failure,
source drift, invalid output, and supervisor failure. Infrastructure failures
cannot close a capacity bracket.

## Measure complementary memory domains

Store raw bytes in artifacts; convert to binary GiB only for display.

- `max_memory_allocated`: peak tensor/storage bytes known to PyTorch.
- `max_memory_reserved`: peak bytes held by the PyTorch caching allocator.
- Device-wide used-memory high-water: catches CUDA context, kernels, workspaces,
  graph pools, and allocations outside PyTorch.
- Owned process-group RSS high-water and minimum system `MemAvailable`: required
  for CPU offload and machine safety.
- Model-load allocated bytes: separates weights/context from request
  activations.
- OOM request bytes, free bytes, allocated bytes, and reserved-but-unallocated
  bytes from the exception.
- Failure stage, recycle/step/sample, operation, source line, requested shape,
  dtype, and calculated tensor bytes.

Use:

```text
tensor_bytes = product(shape) * element_size
gpu_headroom = torch_visible_device_bytes - device_wide_peak_bytes
```

Do not infer headroom from `reserved`, and do not call
`reserved - allocated` reclaimable memory. Live allocations can pin allocator
segments, and external/native allocations are absent from allocated bytes.

Reset PyTorch peak statistics immediately before the measured scope. Device-wide
and host peaks require sidecar polling throughout that scope; one
`nvidia-smi` sample after synchronization can miss the peak.

Polling is discrete even when it spans the whole scope. Repeats can have
identical PyTorch peaks but different device-wide high-water values when a
short native or workspace transient aligns with only one poll. Preserve the
repeat vector and range, use every attempt's own maximum for its headroom
classification, and report the conservative worst-observed delta alongside
any median. Do not present the best or median sampled reduction as a stable
saving when the repeats disagree.

## Time a separate clean run

Capacity and profiling runs answer fit and attribution. Measure latency in a
separate process without hooks, allocator history capture, tensor enumeration,
CUPTI, NVTX-heavy tracing, or debug logging.

Unless a model-specific contract says otherwise, use one discarded warmup and
one measured forward. Construct and load the model outside the window, and keep
preprocessing, host-to-device transfer, postprocessing, writing, scoring, and
progress output outside both synchronization points:

```python
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    output = model(device_batch, **runtime_args)
torch.cuda.synchronize()
forward_seconds = time.perf_counter() - start
```

Use the same window for baseline and candidate. Record pipeline wall time only
as a supplementary metric. If CUDA graphs, compilation, or shape
specialization apply, declare their warmup/capture rules and verify that the
measured call does not compile or recapture.

A single timing observed incidentally during a capacity attempt supports no
latency claim. For a latency trade-off, use already recommended-safe inputs and
the repository's benchmark protocol.

## Close a capacity boundary efficiently

Choose coarse anchors, refinement grid, maximum ceiling, and repeat rule before
launching the sweep.

1. Run a known-safe anchor.
2. Ascend through predeclared coarse points and stop that lane at the first
   genuine OOM.
3. Refine only between the largest pass `L` and smallest OOM `F`, aligned to
   the declared grid.
4. When `F = L + grid`, rerun `L` and `F` in two independent fresh
   subprocesses and distinct directories.
5. Call `L` the hard maximum only when both `L` attempts pass completely and
   both `F` attempts OOM at a consistent site.
6. If the reviewed ceiling passes, report a lower bound and obtain a new ceiling
   before testing larger inputs.

A larger pass followed by a smaller OOM is not a valid monotonic bracket. Treat
it as an allocator-history, scheduling, or lifetime signal; preserve both
attempts and diagnose the changed failure stage. Do not blindly average or
discard the inconvenient row.

Define the recommended margin before observing the boundary. A useful default,
when no product rule exists, is GPU headroom of at least
`max(4 GiB, 10% of visible bytes)` plus a host margin of at least
`max(16 GiB, 10% of MemTotal)`. Report the observed raw margins so an operator
can apply a stricter rule. The hard maximum is not automatically safe.

## Profile the measured wall

Escalate instrumentation only as far as the current evidence requires.

1. **Read the OOM.** Capture its allocation size and allocator state. Calculate
   candidate tensor sizes from the live shapes and dtype.
2. **Add coarse stage boundaries.** Reset the peak before the target scope and
   record enter/exit allocated and running peak around trunk, MSA, diffusion,
   sampling, and confidence.
3. **Instrument the implicated submodule.** Use the inlined hook harness in
   `SKILL.md`. Hooks miss methods called directly instead of through
   `Module.__call__`; wrap those methods explicitly. Fused/native kernels may
   allocate between Python boundaries.
4. **Inspect live storage once.** Deduplicate by storage pointer. A small view
   can pin a much larger parent. Python GC sees Python-referenced tensors, not
   every native allocation, and must not be used in a timing run.
5. **Capture allocator history only when needed.** A PyTorch memory snapshot can
   explain segment splits and allocation ancestry, but it is intrusive and can
   be very large. Bound the capture to one diagnostic attempt around the failing
   phase.
6. **Use timeline profilers for timeline questions.** NVTX plus Nsight Systems
   can separate CPU gaps, kernels, and transfers. CUPTI transfer accounting is a
   separate profiled scenario; summed copy durations are descriptive because
   transfers may overlap compute.

A memory-bounded attention kernel does not guarantee the projection before it
is bounded. For example, a dense QKV projection can allocate
`[B,N,N,3c]` before flash/fused attention sees query rows. If the traceback
lands at that projection, guarded query-row chunking targets the wall more
directly than changing the attention kernel.

`torch.cuda.empty_cache()` releases wholly unused allocator segments; it does
not free a live tensor or free blocks inside a segment pinned by another live
allocation. Use it only at a proven phase boundary after shortening lifetimes,
measure device-wide high-water as well as allocated/reserved bytes, and
preserve graph-capture behavior.

## Diagnose allocator history separately

First reproduce with the default allocator. A large
reserved-but-unallocated value suggests an allocator-history experiment but
does not prove fragmentation. Live tensors can pin split segments.

If supported by a separately tested harness, repeat only the failing size with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and mark
`diagnostic_only=true`. Never use that row for:

- the hard or recommended production boundary;
- a baseline/candidate memory delta;
- a latency comparison; or
- one side of an implementation comparison.

If the wall moves, inspect which allocation becomes next. Do not claim that
live memory fell unless allocated peaks and the resident inventory demonstrate
it.

## Choose the next code lever

Rank candidates by direct evidence:

1. the exact failed allocation or unsafe peak it removes;
2. calculated bytes removed from the critical overlap;
3. reuse of an existing kernel, buffer, chunk path, or proven schedule;
4. semantic and numerical risk;
5. likely latency cost; and
6. whether a shared-layer fix benefits multiple models.

Prefer eliminating the measured tensor over broad tuning. Examples:

- destination-writing fused dispatch before adding a new kernel;
- lifetime rescheduling before calling `empty_cache()`;
- output preallocation before shrinking every global chunk;
- guarded query-row chunking when full QKV fails before fused attention; and
- consumer-dtype storage before changing the producer's compute precision.

Record rejected alternatives and why: wrong failure stage, estimate too small,
unsupported kernel contract, wider numerical risk, allocator-only workaround,
or launch overhead. Change one lever, validate it, and reprofile before choosing
the next.

## Use a numerical-validation ladder

State exactly which level each change proves.

1. **Path activation:** make the fallback raise or record dispatch/call order.
   For destination-writing operations, assert returned storage aliases the
   supplied buffer.
2. **Local operator:** use randomized nonzero weights and representative shapes.
   Require exact equality when operation order is unchanged. Otherwise freeze an
   operator- and dtype-specific tolerance before the capacity run.
3. **Guards and semantics:** cover small/dense, unsupported layout, CPU, batch
   shape, output containers, caller ownership, and CUDA graph capture as
   applicable.
4. **End-to-end integrity:** run the full checkpoint/workload; require every
   expected head, shape, dtype, device, finite value, sample, and export.
   Preserve hashes and compact confidence summaries for exact comparisons.
5. **Quality:** when math, precision, kernel order, or model scheduling can
   change results, score representative real inputs. Synthetic finiteness and
   deterministic repeat hashes do not replace quality metrics.

Separate statements such as “bitwise identical,” “within local BF16
tolerance,” “finite and repeat-deterministic,” and “quality-neutral on the
scored set.” Never summarize all four as “numerically identical.”

## Spend GPU time deliberately

Use the cheapest evidence that can answer the next decision:

- source trace and shape arithmetic before a GPU run;
- focused randomized tests before full-checkpoint execution;
- one diagnostic at the exact wall before a sweep;
- the prior pass and failure after each single code lever;
- two repeats only when closing terminal adjacent endpoints; and
- a real quality set only when the change can affect model math.

Do not rescan an unchanged recommended region when an optimization only targets
a later hard wall. Do not repeatedly test a known interior pass, rerun a
deterministic infrastructure failure, or auto-ascend beyond the reviewed
ceiling. When a result is non-monotonic, run one targeted diagnostic based on
its new failure site instead of accumulating blind repeats.

Stop the campaign immediately for source drift, changed checkpoint/input,
unexpected GPU process, host-safety breach, invalid output, or cleanup failure.

## Preserve evidence another engineer can audit

Each raw attempt should contain at least:

- attempt/campaign ID, timestamps, terminal status, and exact command;
- frozen source/runtime/hardware/workload identity;
- input dimensions and digests;
- allocated, reserved, device-wide, RSS, and host-availability peaks;
- synchronized timing fields with their scope, or explicit `null`;
- output validation, output signature and hashes;
- OOM allocation, allocator state, stage, traceback, and log;
- preflight/postflight GPU process lists and owned-process cleanup; and
- source identity before and after.

The campaign aggregate should bind attempt paths and SHA-256 digests, source
pins, endpoint repeats, hard and recommended decisions, numerical scope,
rejected candidates, and the next wall. Keep bulky logs, traces, environments,
and outputs outside Git; commit a compact curated JSON and a terse journey that
links immutable raw evidence.

Do not let ignored files pollute a production manifest. Verify every digest
referenced by the curated artifact independently before publishing it.

## Completion checklist

- [ ] Claim type and full workload are explicit.
- [ ] Source, checkpoint, input, environment, hardware, and allocator are
      pinned.
- [ ] Each capacity attempt used a fresh isolated subprocess and unique output.
- [ ] Metrics cover PyTorch, device-wide GPU, and host memory.
- [ ] Optimized path execution and numerical scope are proven.
- [ ] The exact prior wall was re-profiled under the unchanged contract.
- [ ] Adjacent hard endpoints are repeat-confirmed, or the result is a lower
      bound.
- [ ] Recommended capacity applies predeclared GPU and host margins.
- [ ] Raw artifacts are immutable, hash-bound, and source-clean.
- [ ] The report names rejected alternatives, unverified claims, and the next
      allocation wall.
