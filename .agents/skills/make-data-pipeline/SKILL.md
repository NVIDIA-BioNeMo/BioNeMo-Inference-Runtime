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
name: make-data-pipeline
description: Port an open-source bioinformatics data pipeline into the BioIR pipeline architecture. Use when the user asks to create, port, convert, or write a new data pipeline from OSS code, or add a new model's data processing to the BioIR system.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Port an Open-Source Data Pipeline to BioIR

**Input:** the open-source model's name and source location. **Output:** a
registered BioIR pipeline, validation evidence, and a summary report.

The procedure is in two guides. Read both in full before you start, and
follow them in order — every step of the first, then every step of the
second, starting a step only after the previous one passes:

1. [Port a Data Pipeline][port]
1. [Validate a Ported Data Pipeline][validate]

This file lists only what an agent must do on top of them.

[port]: ../../../docs/advanced/port-data-pipeline.md
[validate]: ../../../docs/advanced/validate-data-pipeline.md

## Hard Gates

Treat the [rules for trustworthy results][rules] as hard gates. In addition:

- Every PASS you report cites the command, its exit code, the artifact path,
  and the exact sample set. Never report a result from memory or partial
  output.
- A gate you cannot meet is a blocker. Stop and report it, and do not claim
  completion.
- Changing a frozen tolerance, threshold, or feature category, or narrowing
  the sample scope, needs user approval and a rationale in `NOTES.md`.

[rules]: ../../../docs/advanced/validate-data-pipeline.md#rules-for-trustworthy-results

## Ask the User

- **First:** where the upstream source lives, and the WORKDIR (default
  `workdir/<model_name>/`). Do not guess which version or fork is
  authoritative.
- **Before:** extending an existing pipeline instead of creating one,
  adding a dependency, or downloading a checkpoint that `bionemo_ir.hubs`
  does not resolve.
- **Immediately:** when you find an upstream bug.

The user might step away. Gather every resource — source, checkpoint, samples,
scorers — at the start.

## Working Directory

Set up the environment as in [Development Workflow][dev], and the WORKDIR as
in the [validation guide][validate]. Never commit the WORKDIR.

Keep `$WORKDIR/implementation-notes.md` as a running log, written when a
decision happens rather than at the end. Each entry has a timestamp, a
category (design decision, deviation, tradeoff, or open question), the
context, the decision, the reason, alternatives considered, the expected
impact, and whether it needs user confirmation.

[dev]: ../../../docs/dev.md#create-the-development-environment

## Final Report

Consolidate `implementation-notes.md` so a future agent can resume from it,
then print a summary:

1. Model, upstream location and commit, and pipeline shape.
1. The function inventory: upstream function to BioIR class.
1. Files created or modified.
1. Level 1 and Level 2 results, including serial and Ray.
1. Limitations, and anything that needs human review.
1. The WORKDIR, the path to `implementation-notes.md`, and every open
   question.
