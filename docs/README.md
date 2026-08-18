<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BioNeMo Inference Runtime Developer Guide

How to set up a local environment, build from source, and run the tests. This
is the source of truth for local development;
[`CONTRIBUTING.md`](../CONTRIBUTING.md) covers contribution _policy_ (issues,
sign-off, PR expectations) and links here for the mechanics.

> GPUs are required to run the test suite — there is no CPU-only path.

## Environment setup & build

Build and run the development image (repository and dev dependencies are
installed in editable mode inside it). See the top-level [README](../README.md)
for the current Docker build and run commands.

In a checkout with no `bionemo_ir/dsl_kernels/cute/` sources, the CUBIN
launcher extension must exist before anything imports `bionemo_ir` — the
package requires a kernel backend at import time, and pytest then aborts the
session before collection. Build it with
`pip install --no-build-isolation -v -e '.[dev]'`.
`BIOIR_BUILD_CUTEDSL_KERNELS=0` (env or an untracked `build.env`) skips that
build, so use it only where an extension is already installed. `prek run` does
not import the package, so the lint loop works without it.

<!-- TODO: expand — native build, submodule init (3rdparty/), CUDA/compiler
     matrix, devcontainer usage. -->

## Code style

<!-- TODO: document the coding guideline and formatting/linting workflow once
     finalized. -->

## Testing

Run the suite the way CI does:

```bash
pytest -s tests/
```

Model weights for tests and benchmarks are staged from NGC — see
[ref/model-weights.md](ref/model-weights.md). Keep weights and checkpoints off
git; they live on NGC, never in the repository.

<!-- TODO: expand — GPU/distributed test markers, ciflow:* CI labels, coverage
     expectations, C++/CUDA test invocation, benchmarking. -->

## Further development guidelines

<!-- Placeholder for deeper project development guidelines: architecture
     overview, adding a new model, kernel development, debugging. Add sections
     here as they are written. -->
