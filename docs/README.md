<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BioNeMo Inference Runtime documentation

What lives where. [`CONTRIBUTING.md`](../CONTRIBUTING.md) covers contribution
_policy_ — issues, sign-off, what a reviewable pull request looks like — and
links here for the mechanics.

**Start at [dev.md](dev.md).** It is the source of truth for local development:
prerequisites, getting the source, building the extension, running a model,
testing, linting, and what to run before opening a pull request. Staging
checkpoints is its own topic — see
[ref/model-weights.md](ref/model-weights.md). For style and commit conventions
see [coding.md](coding.md); for the container images and what each target is
for see [`../docker/README.md`](../docker/README.md).

> The test suite needs an NVIDIA GPU — there is no CPU-only path through it.
> Tests whose weights are unavailable skip rather than fail, so a run without
> them still exercises the kernels and the pipeline scaffolding.

[`ref/`](ref/) is the technical reference: the API, the pipeline architecture,
the config tree, the support matrix and checkpoint resolution. The top-level
[README](../README.md) lists each entry with a description.
