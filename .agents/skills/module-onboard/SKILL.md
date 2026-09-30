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
name: module-onboard
description: Converts a source model's fundamental module (Pairformer, DiffusionTransformer, etc.) to use BioIR optimized layers, with weight conversion and numerical validation.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Module Onboarding

**Input:** a source module (an `nn.Module` subclass) and, optionally, a
checkpoint. **Output:** a feasibility report, a weight converter, an adapter
and swap function, equivalence tests, and a summary report.

The procedure is [Accelerate a Custom Model with BioIR Modules][guide]. Read
it in full and follow its steps in order. Use the RF3 Pairformer conversion in
the [module onboarding example][examples] as a template when the source module
matches its layout. This file lists only what an agent must do on top of
the guide.

[guide]: ../../../docs/advanced/accelerate-custom-model.md
[examples]: ../../../examples/module_onboarding/

## Confirm with the User

Before creating any file, confirm:

1. The source module file, and the class if the file defines several.
1. The checkpoint, if any. Without one, use random weights and state in the
   summary that real-checkpoint validation is still pending.
1. The working directory, by default `/workspace/onboard_<source>_<module>/`.

## Rules

- Write every artifact under the working directory, in the guide's layout.
  The BioIR repository or install and the source repository are read-only.
- Stop and report when the module's parameters (count × bytes per element)
  exceed the GPU memory from `nvidia-smi`, or when a source dependency
  conflicts with BioIR's, such as a different PyTorch version. Never modify
  the BioIR environment.
- With only a BioIR wheel installed, read the sources with
  `inspect.getsource`.

## Feasibility Report

After the guide's [Map Your Module Onto BioIR Layers][map] section, write
`$WORKDIR/feasibility_report.md` and show it to the user before converting
anything. Cover, writing "None" where a section is empty:

1. Module overview: class, file, checkpoint, parameter count, layer count,
   dtype, and the closest BioIR module.
1. Support matrix: each submodule with its BioIR class or a gap, the gap's
   reason from the guide's list, and whether conversion is full or partial.
1. Hyperparameter mapping and `forward` signature differences.
1. Gaps that need BioIR development, with estimated effort.
1. Training-only features to strip.
1. The plan and open questions.

[map]: ../../../docs/advanced/accelerate-custom-model.md#map-your-module-onto-bioir-layers

## Summary Report

Benchmarks are optional. Run them as in [Benchmark the Swap][bench] only after
every test passes, and save them to `$WORKDIR/results/bench_<timestamp>.csv`
and `.json`.
Then print:

1. What was converted, and any gaps.
1. The working directory and the files created.
1. Fusions applied, and every discarded weight with its reason.
1. Each test, what it validates, and PASS or FAIL.
1. Limitations and recommended next steps.

[bench]: ../../../docs/advanced/accelerate-custom-model.md#benchmark-the-swap
