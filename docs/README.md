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
[model-weights.md](model-weights.md). Keep weights and checkpoints off git; they
live on NGC, never in the repository.

<!-- TODO: expand — GPU/distributed test markers, ciflow:* CI labels, coverage
     expectations, C++/CUDA test invocation, benchmarking. -->

## Release artifacts

[nv/release.md](nv/release.md) covers how to build a wheel from any branch — for
testing on a host with no checkout — the version scheme, when to bump it, and
what a `release/*` push runs.

## Further development guidelines

<!-- Placeholder for deeper project development guidelines: architecture
     overview, adding a new model, kernel development, debugging. Add sections
     here as they are written. -->
