---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Third-Party Notices

BioNeMo Inference Runtime (BioIR) uses the third-party software listed below.
Each component remains governed by its own license. The Apache License 2.0 for
NVIDIA-authored BioIR code does not replace those terms.

Dependency constraints remain authoritative in the project manifests. The
release SBOM records the resolved dependency versions. This notice records a
version or revision only when it identifies source copied into BioIR, code
compiled into a distributed artifact, or a pinned reference submodule.

The root [`LICENSE`][bioir-license] contains Apache-2.0. Other reusable license
texts are under [`LICENSES/`][licenses]. Component-specific license and notice
files distributed by an upstream package remain authoritative for that package.

## Distributed Source and Derived Code

- **OpenFold and AlphaFold source excerpts**
  - Version: source excerpts; no upstream release pin
  - Copyright: AlQuraishi Laboratory and DeepMind Technologies Limited
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [OpenFold][openfold] and [AlphaFold][alphafold]
  - Distribution: adapted files ship in the public source tree and wheel
  - Modifications: integration with BioIR data structures, optimized layers,
    checkpoint conversion, and output handling

- **Ray source excerpt**
  - Version: Ray 2.53.0 source excerpt
  - Copyright: Ray Authors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [Ray pipeline stage][ray-stage]
  - Distribution: adapted code ships in
    `bionemo_ir/pipeline/stages/base.py`

- **vLLM source excerpt**
  - Version: source excerpt; no upstream release pin
  - Copyright: vLLM contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [vLLM][vllm]
  - Distribution: adapted code ships in
    `bionemo_ir/dsl_kernels/triton/fused_swiglu.py`

- **PyTorch3D source excerpt**
  - Version: source excerpt; no upstream release pin
  - Copyright: Meta Platforms, Inc. and affiliates
  - License: BSD-3-Clause; see
    [`LICENSES/BSD-3-Clause.txt`][bsd-3-clause]
  - Source: [PyTorch3D][pytorch3d]
  - Distribution: copied code ships in
    `bionemo_ir/_torch/layers/random_augmentation.py`

- **Quack source excerpt**
  - Version: source excerpt; no upstream release pin
  - Copyright: Wentao Guo, Ted Zadouri, and Tri Dao
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [Quack cache implementation][quack]
  - Distribution: adapted code ships in
    `bionemo_ir/dsl_kernels/cute_cache.py`

- **cuEquivariance source excerpt**
  - Version: source excerpt from the cuEquivariance implementation
  - Copyright: NVIDIA Corporation and affiliates
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [cuEquivariance][cuequivariance]
  - Distribution: adapted code ships in
    `bionemo_ir/dsl_kernels/triton/fused_ln_proj_moveaxis_pad.py`

- **CUTLASS-derived CuTe DSL source**
  - Version: source excerpts associated with nvidia-cutlass-dsl 4.5.2
  - Copyright: NVIDIA Corporation and affiliates
  - License: BSD-3-Clause or NVIDIA Software License, as marked per source file
  - Source: [CUTLASS][cutlass] and [CuTe DSL license][cutlass-dsl-license]
  - Distribution: BSD-3-Clause files may ship; proprietary CuTe DSL source is
    removed from public artifacts before release

- **TensorRT-LLM source excerpt**
  - Version: source excerpt; no upstream release pin
  - Copyright: NVIDIA Corporation and affiliates
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source: [TensorRT-LLM][tensorrt-llm]
  - Distribution: adapted layout code ships in
    `bionemo_ir/_torch/layers/linear.py`

## Compiled into BioIR

- **nanobind**
  - Version: 2.10.2
  - Copyright: Wenzel Jakob and contributors
  - License: BSD-3-Clause; see
    [`LICENSES/BSD-3-Clause.txt`][bsd-3-clause]
  - Source and license: [nanobind 2.10.2][nanobind]
  - Distribution: statically compiled into BioIR extension modules
  - Modifications: none

## Runtime Dependencies

These packages are declared in `requirements.txt` and downloaded separately by
the recipient's package installer. They are not included in the BioIR source
distribution or wheel. Their package-provided license and notice files remain
authoritative, and the release SBOM records the resolved versions.

- **apache-tvm-ffi**
  - Copyright: Apache TVM contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Apache TVM FFI][apache-tvm-ffi]

- **torch**
  - Copyright: PyTorch contributors
  - License: compound upstream license bundle, including Apache-2.0,
    BSD-2-Clause, BSD-3-Clause, BSL-1.0, MIT, and LLVM exceptions
  - Source and license: [PyTorch][pytorch]
  - Notice: preserve the `LICENSE`, `NOTICE`, and third-party license files from
    the installed distribution

- **cuequivariance**
  - Copyright: NVIDIA Corporation and affiliates
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [cuEquivariance][cuequivariance]

- **triton**
  - Copyright: OpenAI and Triton contributors
  - License: MIT; see [`LICENSES/MIT.txt`][mit]
  - Source and license: [Triton][triton]

- **biopython**
  - Copyright: Biopython contributors
  - License: Biopython License Agreement; some files are dual-licensed under
    BSD-3-Clause
  - License: [`LICENSES/Biopython-License-Agreement.txt`][biopython-license]
  - Source: [Biopython][biopython]

- **ray**
  - Copyright: Ray Authors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Ray][ray]

- **uvloop**
  - Copyright: MagicStack Inc. and uvloop contributors
  - License: MIT OR Apache-2.0; BioIR records the MIT option
  - License: [`LICENSES/MIT.txt`][mit]
  - Source: [uvloop][uvloop]

- **modelcif**
  - Copyright: ModelCIF contributors
  - License: MIT; see [`LICENSES/MIT.txt`][mit]
  - Source and license: [python-modelcif][modelcif]

- **rdkit**
  - Copyright: RDKit contributors
  - License: BSD-3-Clause; see
    [`LICENSES/BSD-3-Clause.txt`][bsd-3-clause]
  - Source and license: [RDKit][rdkit]

- **biotite**
  - Copyright: Biotite contributors
  - License: BSD-3-Clause; see
    [`LICENSES/BSD-3-Clause.txt`][bsd-3-clause]
  - Source and license: [Biotite][biotite]

- **datasets**
  - Copyright: Hugging Face and contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Datasets][datasets]

- **huggingface-hub**
  - Copyright: Hugging Face and contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Hugging Face Hub][huggingface-hub]

- **lru-dict**
  - Copyright: Amit Dev and contributors
  - License: MIT; see [`LICENSES/MIT.txt`][mit]
  - Source and license: [lru-dict][lru-dict]

- **kalign-python**
  - Copyright: Timo Lassmann and contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Kalign][kalign]

- **scipy**
  - Copyright: SciPy developers
  - License: BSD-3-Clause with bundled third-party notices
  - Source and license: [SciPy][scipy]
  - Notice: preserve the package-provided license bundle from the installed
    distribution

- **gemmi**
  - Copyright: Global Phasing Ltd.
  - License: MPL-2.0 OR LGPL-3.0-or-later; BioIR records the MPL-2.0 option
  - License: [`LICENSES/MPL-2.0.txt`][mpl-2]
  - Source: [Gemmi][gemmi]
  - Distribution: obtained independently by recipients; not included in the
    BioIR source distribution or wheel

- **einops**
  - Copyright: Alex Rogozhnikov and contributors
  - License: MIT; see [`LICENSES/MIT.txt`][mit]
  - Source and license: [einops][einops]

- **pydantic**
  - Copyright: Pydantic contributors
  - License: MIT; see [`LICENSES/MIT.txt`][mit]
  - Source and license: [Pydantic][pydantic]

## NVIDIA-Licensed Runtime Dependencies

These NVIDIA packages are included in the component inventory but are not
third-party OSS. Their package-specific NVIDIA terms control their use and
distribution.

- **cuda-python** — LicenseRef-NVIDIA-SOFTWARE-LICENSE;
  [package and terms][cuda-python]
- **cuequivariance-ops-cu13** — LicenseRef-NVIDIA-MATH-LIBRARIES-SDK;
  [package][cueq-ops-cu13]
- **cuequivariance-ops-torch-cu13** —
  LicenseRef-NVIDIA-MATH-LIBRARIES-SDK; [package][cueq-ops-torch-cu13]
- **nvidia-cutlass-dsl** — LicenseRef-NVIDIA-SOFTWARE-LICENSE;
  [package and terms][cutlass-dsl-license]

## Reference Submodules

The following pinned submodules support parity and reference tests. They do not
ship in the BioIR wheel or runtime container.

- **OpenFold3**
  - Commit: `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`
  - Copyright: AlQuraishi Laboratory and contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [OpenFold3 commit][openfold3]

- **Protenix**
  - Commit: `2475421477ab414b571149ad4a875c390ff8a35d`
  - Copyright: ByteDance and Protenix contributors
  - License: Apache-2.0; see [`LICENSE`][bioir-license]
  - Source and license: [Protenix commit][protenix]

## Build and Development Dependencies

These tools are used to build or test BioIR. They do not ship in the wheel or
runtime container unless an artifact-specific SBOM states otherwise.

- **CMake** — BSD-3-Clause; [source and license][cmake]
- **setuptools** — MIT; [source and license][setuptools]
- **prek** — MIT; [source and license][prek]
- **pytest** — MIT; [source and license][pytest]
- **pytest-cov** — MIT; [source and license][pytest-cov]
- **pytest-xdist** — MIT; [source and license][pytest-xdist]
- **parameterized** — BSD-2-Clause; [source and license][parameterized]
- **biotraj** — LGPL-2.1-or-later; [source and license][biotraj]
- **ml-collections** — Apache-2.0;
  [source and license][ml-collections]

## Base Images

- **NVIDIA PyTorch container** — selected in `docker/Dockerfile`; used only for
  build and development stages
- **NVIDIA CUDA container** — selected in `docker/Dockerfile`; used when a
  recipient builds the runtime image

The image-provided OSS notices and license files must remain in derivative
images. The release SBOM, rather than this repository inventory, records the
resolved operating-system and base-image package closure.

[alphafold]: https://github.com/google-deepmind/alphafold
[apache-tvm-ffi]: https://github.com/apache/tvm-ffi
[bioir-license]: LICENSE
[biopython]: https://github.com/biopython/biopython
[biopython-license]: LICENSES/Biopython-License-Agreement.txt
[biotite]: https://github.com/biotite-dev/biotite
[biotraj]: https://github.com/biotite-dev/biotraj
[bsd-3-clause]: LICENSES/BSD-3-Clause.txt
[cmake]: https://github.com/Kitware/CMake
[cuda-python]: https://pypi.org/project/cuda-python/
[cueq-ops-cu13]: https://pypi.org/project/cuequivariance-ops-cu13/
[cueq-ops-torch-cu13]: https://pypi.org/project/cuequivariance-ops-torch-cu13/
[cuequivariance]: https://github.com/NVIDIA/cuEquivariance
[cutlass]: https://github.com/NVIDIA/cutlass
[cutlass-dsl-license]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
[datasets]: https://github.com/huggingface/datasets
[einops]: https://github.com/arogozhnikov/einops
[gemmi]: https://github.com/project-gemmi/gemmi
[huggingface-hub]: https://github.com/huggingface/huggingface_hub
[kalign]: https://github.com/TimoLassmann/kalign
[licenses]: LICENSES
[lru-dict]: https://github.com/amitdev/lru-dict
[mit]: LICENSES/MIT.txt
[ml-collections]: https://github.com/google/ml_collections
[modelcif]: https://github.com/ihmwg/python-modelcif
[mpl-2]: LICENSES/MPL-2.0.txt
[nanobind]: https://github.com/wjakob/nanobind/tree/v2.10.2
[openfold]: https://github.com/aqlaboratory/openfold
[openfold3]: https://github.com/aqlaboratory/openfold-3/tree/c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
[parameterized]: https://github.com/wolever/parameterized
[prek]: https://github.com/j178/prek
[protenix]: https://github.com/bytedance/Protenix/tree/2475421477ab414b571149ad4a875c390ff8a35d
[pydantic]: https://github.com/pydantic/pydantic
[pytest]: https://github.com/pytest-dev/pytest
[pytest-cov]: https://github.com/pytest-dev/pytest-cov
[pytest-xdist]: https://github.com/pytest-dev/pytest-xdist
[pytorch]: https://github.com/pytorch/pytorch
[pytorch3d]: https://github.com/facebookresearch/pytorch3d
[quack]: https://github.com/Dao-AILab/quack/blob/c8ec3170057987da0ec99883736f381ea1937cf3/quack/cache/jit.py
[ray]: https://github.com/ray-project/ray
[ray-stage]: https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/batch/stages/base.py
[rdkit]: https://github.com/rdkit/rdkit
[scipy]: https://github.com/scipy/scipy
[setuptools]: https://github.com/pypa/setuptools
[tensorrt-llm]: https://github.com/NVIDIA/TensorRT-LLM
[triton]: https://github.com/triton-lang/triton
[uvloop]: https://github.com/MagicStack/uvloop
[vllm]: https://github.com/vllm-project/vllm
