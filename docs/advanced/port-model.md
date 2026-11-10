---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Port a Model to BioIR

A native model port implements an upstream pretrained network with BioNeMo
Inference Runtime (BioIR) components and converts its checkpoint. The port
preserves the upstream inference behavior while using BioIR's attention
backends, fused operations, and graph regions where supported.

The result includes the model, its config and checkpoint converter, tests,
and evidence of correctness. Full pipeline support also connects the model
to BioIR's tokenizer, feature generators, postprocessor, and factory
registry. Decide whether that integration is in scope before starting.

For an agent-driven port, use the [model-onboard skill][skill]. If you want
to replace one module inside your existing model, follow [Accelerate a
Custom Model][modules]. If the network already exists and you need its data
path, follow [Port a Data Pipeline][pipeline].

[skill]: ../../.agents/skills/model-onboard/SKILL.md
[modules]: accelerate-custom-model.md
[pipeline]: port-data-pipeline.md

## Fix the Reference and Scope

Start with the following resources:

- The upstream source at the revision that matches the checkpoint, including
  its inference entrypoint and the configuration the release uses for
  inference. Training defaults can differ from inference settings.
- The released checkpoint and supporting metadata, identified by revision
  and digest. Random weights can test wiring, but cannot validate a port of
  a pretrained model.
- The licenses of the upstream code and the checkpoint. Port only what they
  allow, and cite the upstream file in the docstring of each ported module.
- A writable BioIR checkout on a branch, prepared as in [Development
  Workflow][dev]. Keep the upstream checkout read-only and install its
  dependencies in a separate environment.
- Representative inputs for the requested features, plus experimental
  structures or other task references for final quality checks.
- A working directory outside the package for reference captures, validation
  scripts, logs, and reports. Keep those large artifacts out of Git.

Record the supported model variant, input types, output fields, precision
modes, and runtime settings. For folding models, include MSA and template
use, recycles, diffusion steps, sample count, seed, and ranking policy.
For other tasks, define the corresponding input and output contracts and
quality metrics before adapting a folding-specific procedure.

Record available hardware and memory. A host without a supported GPU can
still inspect sources, implement the port, and run static or CPU checks.
Prepare reproducible commands for later GPU validation and report that work
as pending. Do not replace numerical comparisons with import checks or
report performance without running the model.

[dev]: ../dev.md

## Map the Upstream Computation

Trace inference from input features through the final returned outputs.
Inventory the embedders, trunk, recycling, sampler, denoiser, prediction
heads, and postprocessing that the chosen variant actually runs. Include
operations outside the main module, such as coordinate augmentation, sample
ranking, and corrections applied to the written structure.

For each component, record its input and output shapes, token or atom axes,
masks, parameters, buffers, and precision. Compare its math with the nearest
BioIR component, then classify the work:

- Reuse an existing component or converter with a different config.
- Extend a shared component with an option whose default preserves current
  models.
- Compose existing primitives into a new family-specific component.
- Implement an operation that BioIR does not yet provide.

Start with the reusable layers and family composites described in
[Architecture][architecture]. The [module conversion guide][modules]
explains config mapping, fused weights, and common mismatches. Matching
class names or tensor sizes is insufficient: compare normalization,
projection biases, residual order, attention layout, and mask semantics.
Derived constants also change with the config. For example, a distogram
with more bins moves the bin edges that a contact probability sums over.

Inspect precision across the whole inference path. Casting a recycled pair
state, changing accumulation precision, or moving an autocast boundary can
change both results and memory. A release can also switch precision by input
size, for example running diffusion in `float32` up to a token count and in
`bfloat16` above it. Record each threshold and which branches the validation
covers.

A model can also change token space after the trunk, for example by
splitting residues into finer tokens for diffusion. That stage then needs
explicit feature and output mappings. Its CUDA graph limits, chunking, and
other size-keyed policies must use the token count of the stage that runs.

Write the mapping and gaps in a feasibility report before implementation.
State the supported scope, the closest existing family, shared changes, and
the validation plan. Resolve missing upstream semantics before choosing an
approximation.

[architecture]: ../ref/architecture.md

## Implement the Native Network

Use the existing family nearest to the upstream architecture as a reference. For
example, [Protenix modeling][protenix] composes an embedder, trunk, sampler, and
heads from [family modules][protenix-modules]. Its [converter][converter] shows
how to reuse conversion helpers across those components.

Place the model config, assembly, and converter under
`bionemo_ir/models/<model>/`. Put family-specific composites under
`bionemo_ir/_torch/modules/<model>/`, and reusable primitives under
`bionemo_ir/_torch/layers/`. Reuse an existing family directly when its
semantics match; implement only the differences. When the upstream is a fork
of a BioIR family, subclass that family's model and config classes. Override
the widths and stages that differ, and inherit the rest.

For a model that runs through the folding engine, follow
[Model Constructor and `forward`][model-api]:

- `get_pretrained_config(model_name)` returns a `BaseConfig` subclass with
  the model's dimensions, dtype and backend choices, and batch limits.
- The constructor accepts `config` and `model_name`, resolves the checkpoint
  through `bionemo_ir.hubs`, and loads the converted parameters.
- `forward(batch, **runtime_args)` consumes the feature contract expected by
  the model and returns the outputs its postprocessor requires. Follow the
  family's seed contract, including `sampling_seed` when supplied by the
  pipeline.

The constructor's hub lookup needs the model's entries before the first
load, even in a compute-only port. Complete the first two steps of [Register
the Model][registry]: add the name to the support matrix and the checkpoint
entry under `bionemo_ir/hubs/`. An unregistered name fails with `KeyError`.

Match upstream recycling, sampling schedules, conditioning caches, and
confidence computation. Preserve the upstream default inference settings;
record deliberate precision or algorithm changes separately. Translate count
conventions: a release that sets `N_cycle` to its total number of trunk
passes corresponds to BioIR's `recycling_steps = N_cycle - 1`.

A reused parent `forward` can consume input features after their last use
there. Keep any feature that the port's later stages still read. Reject
unsupported inputs, such as a batch size the model cannot run, with an
explicit error.

Choose backends supported by the actual dtype, GPU, and dimensions. New
channel widths can exceed a fused kernel's supported shapes even when the
math matches. A kernel does not always reject such a shape: a tiling sized
for narrower channels can drop channels without an error. Check the kernel's
supported shapes, confirm that backend selection falls back, and verify the
fallback. Report its performance separately from an unimplemented kernel
optimization. Chunk thresholds tuned for the parent's widths can also
exhaust memory at wider ones, so revisit them.

When extending shared layers, keep the existing defaults and add regression
coverage for the affected families. Training-only code such as optimizers,
losses, and activation checkpointing does not belong in an inference port.

[protenix]: ../../bionemo_ir/models/protenix/modeling.py
[protenix-modules]: ../../bionemo_ir/_torch/modules/protenix/
[converter]: ../../bionemo_ir/models/protenix/convert.py
[model-api]: ../ref/api.md#model-constructor-and-forward

## Convert and Account for the Checkpoint

Load the model state from the upstream checkpoint format and map it into the
BioIR components. Load with `torch.load(..., weights_only=True)`. A training
checkpoint also holds optimizer state, so read its model entry and
memory-map the file. Strip a data-parallel `module.` prefix from the keys.
Reuse existing conversion helpers where layouts match. Check concatenation
order, transposes, bias fusion, and casts against the actual target layers.

Account for every source parameter and buffer. Record tensors that are
converted, intentionally shared, or omitted with a reason. Detect missing
mappings and unintended duplicate conversion; legitimate tied weights need
an explicit sharing rule. Non-strict loading must not conceal missing
parameters.

A fork that keeps its parent's module names can reuse the parent's
converters. To keep the accounting complete, wrap the state dict in a
mapping that records each key read. Run every converter on it, then fail
when a tensor is unread or read twice. Reads made inside the reused
converters are then counted too.

Check buffers as well as trainable parameters. Distance-bin edges and other
constants may be checkpoint state even when the target recomputes them.
Compare the checkpoint's values with the recomputed ones. When they differ,
load the checkpoint's values: the upstream may have computed them on another
device or in another dtype. Document any removed zero-update branch and
verify its effect using the released weights.

Test conversion on real checkpoint tensors. Synthetic weights can exercise
layout and shape edge cases, but do not establish checkpoint coverage or
pretrained-model accuracy.

## Verify the Compute Path

Establish numerical equivalence independently of the BioIR data pipeline.
Generate reference features and intermediate inputs with upstream code in
its own environment. Record the source revision, checkpoint digest, resolved
settings, seed, and commands with each capture.

Replay the same inputs through the matching BioIR components. Start with the
smallest changed component, then compare a whole block, the complete stack,
the trunk across recycles, a denoising step, and the output heads as
applicable. Include the full compute path and every architectural difference
listed in the feasibility report.

Use the same weights, masks, and conditioning on both sides. Match or replay
the random draws when comparing stochastic components; a common seed alone
does not guarantee identical draws across different RNG implementations or
call orders. Replaying the upstream's generator, draw order, and sample
chunking removes sampling noise from whole-model comparisons. Record
unavoidable stochastic differences and measure upstream repeatability before
interpreting a whole-model coordinate difference.

Set tolerances before comparisons, with separate expectations for the
precision modes being tested. Use the [numerical verification guide][numerics]
for module comparisons. Save failing inputs, outputs, and error summaries;
fix the first divergence and repeat the affected comparisons. Do not loosen
a tolerance merely to pass a failing port.

A passing final structure score does not establish module parity. A changed
recycle or conditioning computation can still predict a similar structure.
Before relying on a task-level gate, confirm that it rejects deliberately
wrong edits of the upstream, such as scaled coordinates or a zeroed recycled
pair.

Conversely, passing module comparisons do not establish whole-model parity.
Errors within each module's tolerance can compound through recycles and
sampling. Compare the whole model's output with the upstream's own spread
between repeated runs on the same input. A port outside that spread still
has a divergence to find, often in the precision of state carried between
recycles. Retain both intermediate comparisons and task-level quality checks.

[numerics]: accelerate-custom-model.md#verify-the-numerics

## Connect the Data Path

When full pipeline support is requested, port or adapt the tokenizer,
features, collators, and postprocessor as in [Port a Data Pipeline][pipeline].
An existing pipeline is reusable only when token definitions, atom ordering,
MSA and template handling, metadata, and output semantics match the new model.
For a fork, subclass the parent's tokenizer, feature generators, and
postprocessor, and override only the rules that differ, such as molecule-type
assignment, kept bonds, or corrections to terminal atoms. Document
unsupported input types and reject them explicitly.

Finish [Register the Model][registry] with the residue vocabulary and the
factory. A fork's factory can subclass the parent's and override its runtime
defaults. Register model-specific reference data, such as a chemical
component dictionary, in `bionemo_ir/hubs/metadata.py` with its SHA-256
digest. A fork's reference files can differ from the parent's under the same
names, so compare digests before reusing the parent's entries.

Follow [Validate a Ported Data Pipeline][validate] for feature comparisons
and complete folding runs. Run the full sample manifest through serial and
Ray execution when qualifying that pipeline; a one-file demo is only a
smoke check. Compare the written structures and scores with upstream under
the same settings and ground truth.

A compute-only port can be useful when explicitly requested. Report that
boundary and its feature-loading entrypoint; do not claim `build_processor`
support until the data path, registry, and writer have been exercised.

[registry]: port-data-pipeline.md#register-the-model
[validate]: validate-data-pipeline.md

## Qualify and Report the Port

Run tests for the new model and for shared components and existing families
that changed. Include supported input types, masks, precision boundaries,
and any new dimension or token-space behavior. Keep the upstream baseline
independent of the implementation being tested.

Commit tests that run without the upstream package. Upstream captures and
comparison scripts stay in the working directory. Cover the following:

- The registry and hub entries.
- Checkpoint accounting on a small synthetic state dict.
- Each overridden feature, ranking, or postprocessing rule on small inputs.
- New modules, and every shared component you extended, against their
  previous behavior.

Document the model in a page under `docs/ref/`: what differs from its
nearest family, how to run it, its weights and metadata, defaults,
precision, and known limits. Add its rows to the [support matrix][support]
and [model weights][weights] pages. A page reaches the documentation site
only after it is listed in `fern/index.yml` and in `PAGE_ROUTES` in
`fern/src/common.py`.

After correctness passes, measure speed and memory if performance is in
scope. Follow the [benchmark skill][benchmark] for supported folding models;
its dataset, timing boundary, and scoring rules apply. A model without a
profile under that skill's `models/` directory needs a new profile, or the
report must describe its own timing method. Other tasks need their own
metrics. A GPU OOM or an unavailable dependency remains a reported limit,
not a passing result.

Finish with the following evidence:

- Upstream and BioIR revisions, checkpoint identity, supported scope, and
  instructions to construct and run the model. When the measured revision
  differs from the submitted one, show that the difference leaves the model
  code unchanged.
- The component mapping, shared extensions, and checkpoint accounting.
- Each check as PASS, FAIL, or PENDING, with its command, artifact path, input
  set, and hardware. Give the exit code for executed checks and the reason
  for pending checks.
- Module comparisons, complete prediction quality, regression results, and
  performance measurements when requested.
- Deliberate differences, unsupported features, validation limits, and the
  external working directory containing the evidence.

[support]: ../ref/support-matrix.md
[weights]: ../ref/model-weights.md
[benchmark]: ../../.agents/skills/bench-perf-oss/SKILL.md
