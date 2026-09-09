---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# BioNeMo Inference Runtime Documentation

Public Markdown for BioNeMo Inference Runtime (BioIR). The published site is
at [docs.nvidia.com/bionemo/inference-runtime][site].

- [`install.md`](install.md) — requirements and release-wheel installation
- [`quickstart.md`](quickstart.md) — serial Boltz-2 prediction
- [`ray.md`](ray.md) — multi-GPU Boltz-2 prediction
- [`dev.md`](dev.md) — build, test, and contribute
- [`coding.md`](coding.md) — style, naming, and tooling
- [`contributing.md`](contributing.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and [`SECURITY.md`](SECURITY.md) —
  community policies
- [`ref/`](ref/) — API, architecture, config, images, support, model weights,
  and benchmarks
- [`fern/pages/`](fern/pages/) — Fern-native pages used by the published site
- [`assets/`](assets/) — images and other media referenced by these pages

The linked Markdown files and `ref/` are the public sources. `fern/` holds site
configuration and Fern-native MDX.

[site]: https://docs.nvidia.com/bionemo/inference-runtime
