# syntax=docker/dockerfile:1
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
#
# See docs/ref/docker-images.md.

# nvcr.io/nvidia/pytorch is part of the public NGC catalog and pulls anonymously
# -- no NGC key, no docker login. This project's GPU dependencies are tuned for
# its framework stack. The image-specific lock policy preserves the base's torch
# and Triton while hash-locking every package layered on top.
#
# The default tag is digest-pinned because `devdeps` bakes that same digest in as
# the compile identity of the tracked CUBIN packs; see the ENV below.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
ARG BASE_TAG=26.05-py3@sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
ARG RUNTIME_IMAGE=nvidia/cuda
ARG RUNTIME_TAG=13.0.2-base-ubuntu24.04

# --------------------------------------------------------------------- base
FROM ${BASE_IMAGE}:${BASE_TAG} AS base
ARG UV_VERSION=0.12.9

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The PyTorch base images install torch into the distro's Python, which
    # Debian marks externally-managed (PEP 668). The dependency stages install
    # into that interpreter so they retain the NGC framework stack.
    PIP_BREAK_SYSTEM_PACKAGES=1

ENV UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_CACHE_DIR=/cache/uv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# What the extension build actually needs. nvcc is NOT required to compile CUDA
# here -- the kernels ship precompiled -- it is only read to stamp the wheel's
# +cuXYZ local version, and CUDA_TAG overrides that. The C++ side needs cuda.h
# and the libcuda stub to link against the driver API.
RUN test -f "${CUDA_HOME}/include/cuda.h" \
    && test -f "${CUDA_HOME}/lib64/stubs/libcuda.so"

# Install Git and its LFS transport here, ahead of any stage that may carry
# credentials in the environment, so no apt maintainer script ever runs beside
# them.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git git-lfs ccache curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system --skip-repo

RUN python -m pip install "uv==${UV_VERSION}"

# ccache needs a compiler launcher to do anything; setting CCACHE_DIR alone is
# inert. Both languages the extension uses are covered (cpp/CMakeLists.txt
# declares CXX and ASM). CCACHE_BASEDIR rewrites absolute paths relative to the
# checkout, without which every container hashes its own paths and never hits.
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_ASM_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/cache/ccache \
    CCACHE_BASEDIR=/bioir \
    CCACHE_MAXSIZE=10G \
    PIP_CACHE_DIR=/cache/pip

# Sticky-writable like /tmp. The compiler launcher above is set for every stage,
# so a build running as an arbitrary UID -- which is the whole point of the dev
# stage, and is also how CI runs the devdeps image -- must be able to write here.
# Without this, CMake's compiler check fails with "ccache: error: Permission
# denied" and the extension never builds.
RUN mkdir -p /cache/ccache /cache/pip /cache/uv && chmod -R 1777 /cache

# --------------------------------------------------------------------- deps
FROM base AS deps
WORKDIR /opt/bioir-deps
# Install into the NGC Python environment so its tested framework and CUDA stack
# remain in place. Pruning torch and Triton also removes their PyPI CUDA wheel
# closure; every remaining runtime and build package comes from the lock export
# with hashes. Reinstall PyYAML because the base's distro-owned copy has no pip
# RECORD metadata and cannot be upgraded in place.
COPY pyproject.toml uv.lock requirements.txt ./
RUN --mount=type=cache,target=/cache/uv \
    uv export --quiet --locked --no-dev --no-emit-project \
        --prune torch --prune triton \
        --output-file /tmp/bioir-runtime.txt \
    && uv export --quiet --locked --only-group build --no-emit-project \
        --output-file /tmp/bioir-build.txt \
    && uv pip install --system --no-deps --require-hashes \
        --reinstall-package pyyaml \
        --requirements /tmp/bioir-runtime.txt \
    && uv pip install --system --no-deps --require-hashes \
        --requirements /tmp/bioir-build.txt \
    && rm -f /tmp/bioir-runtime.txt /tmp/bioir-build.txt

# ------------------------------------------------------------------ devdeps
FROM deps AS devdeps
# Dependencies only, no source: invalidated by the project metadata and lock,
# which is what makes it usable as CI's rolling build cache.
#
# The immutable compile identity of the tracked CUBIN packs: the digest of the
# anonymously pullable base above. Regenerating a pack is only reproducible from
# that exact base, so the digest is pinned here rather than derived at build
# time.
ENV BIOIR_CUBIN_COMPILE_ENVIRONMENT=sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
RUN --mount=type=cache,target=/cache/uv \
    uv export --quiet --locked --only-group dev --no-emit-project \
        --prune torch --prune triton \
        --output-file /tmp/bioir-dev.txt \
    && uv pip install --system --no-deps --require-hashes \
        --requirements /tmp/bioir-dev.txt \
    && rm -f /tmp/bioir-dev.txt

# ---------------------------------------------------------------------- dev
FROM devdeps AS dev
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# A dev image may be rebuilt on any BASE_TAG, so it must not inherit devdeps'
# claim to that identity. Empty means "not built from the pinned base".
ENV BIOIR_CUBIN_COMPILE_ENVIRONMENT=

# Match the host UID so files written to the bind mount are owned by the user
# who started the container. A mismatch leaves root-owned build/ and *.so on the
# host, and makes git report the checkout as dubiously owned.
RUN if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd -g "${USER_GID}" "${USERNAME}"; \
    fi \
    && if getent passwd "${USER_UID}" >/dev/null; then \
        usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m \
            "$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
    else \
        useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}"; \
    fi \
    && python -m venv --system-site-packages /opt/bioir-venv \
    && mkdir -p /cache/ccache /cache/pip /cache/uv /cache/bioir-kernels /bioir \
    && chown -R "${USER_UID}:${USER_GID}" /cache /bioir /opt/bioir-venv

ENV BIOIR_KERNEL_CACHE_DIR=/cache/bioir-kernels \
    HISTFILE=/cache/.bash_history \
    VIRTUAL_ENV=/opt/bioir-venv
ENV PATH=/opt/bioir-venv/bin:${PATH}
USER ${USERNAME}
WORKDIR /bioir
# Deliberately does not install the project: the source arrives as a bind mount
# at run time. The writable venv inherits the image dependencies, so
# `uv pip install --no-deps -e .` adds only the checkout on first attach.
# An interactive `docker run` gets a shell. The long-lived containers that need
# to outlive it -- Compose for VS Code, docker/dev.sh -- pass `sleep infinity`
# themselves rather than relying on a default that looks hung when a human runs
# the image directly.
CMD ["bash"]

# -------------------------------------------------------------------- wheel
FROM devdeps AS wheel
WORKDIR /src
COPY . /src
# Make wheel ZIP metadata derive from the source revision rather than build time.
ARG SOURCE_DATE_EPOCH
RUN test -n "${SOURCE_DATE_EPOCH}"
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
# State the CUDA tag explicitly so the wheel's local version does not depend on
# nvcc being present or on which toolkit the base image happens to carry.
ARG CUDA_TAG=
ENV CUDA_TAG=${CUDA_TAG}
# Compile jobs. Empty means "ask the machine", which is right locally and on a
# runner whose daemon sees the host CPUs; docker/Makefile fills it from the
# CI allocation, where nproc inside a job container reports 1 and would compile
# the extension serially.
ARG BUILD_JOBS=
# The CMake tree and the materialized CUBIN payloads both live under build/cmake
# -- setup.py keeps them out of build_temp so they outlive the install -- and
# .dockerignore excludes build/, so without this cache every rebuild re-configures
# CMake, re-materializes the packs and re-links from scratch. ccache only ever
# covered the compile step. Mounted at build/cmake, not build/, so build_lib
# stays fresh: setuptools copies changed sources into it but never deletes
# removed ones, and a persistent copy would carry a deleted module into the
# wheel. sharing=locked because one CMake tree cannot take two concurrent builds.
#
# CMAKE_BUILD_PARALLEL_LEVEL because the default generator is Unix Makefiles,
# which builds serially unless told otherwise: setup.py passes `--parallel` only
# when setuptools' own `parallel` option is set, and build frontends do not.
RUN --mount=type=cache,target=/cache/uv \
    --mount=type=cache,target=/cache/ccache \
    --mount=type=cache,target=/src/build/cmake,sharing=locked \
    uv lock --check \
    && CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        uv build --wheel --no-build-isolation --out-dir /dist .

# ------------------------------------------------------------- wheel-export
# `--output type=local` writes the whole filesystem of the target stage, so
# exporting `wheel` directly would dump the entire PyTorch image onto the host.
# This is the target `make -C docker wheel` builds; it holds the wheel alone.
FROM scratch AS wheel-export
COPY --from=wheel /dist/ /

# ------------------------------------------------------------------ runtime
FROM ${RUNTIME_IMAGE}:${RUNTIME_TAG} AS runtime
#
# Deliberately a `base` CUDA image, not `runtime` or `devel`:
#   * the extension links exactly one CUDA library, libcuda.so.1 (the driver
#     API), which the container toolkit injects from the host;
#   * it loads already-compiled CUBINs via cuModuleLoadData, so the driver never
#     JITs and neither nvrtc nor nvJitLink is needed;
#   * torch brings its own cudart/cublas/cudnn as PyPI wheels, so a CUDA
#     `runtime` base would ship a second copy of all of them.
#
# The Triton kernels do compile at run time: their launcher stubs are host C
# extensions, so gcc and python3.12-dev below are what make the model buildable.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/cache/pip

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# Installed ahead of the wheel and on its own, so this layer is keyed by
# requirements.txt alone. Bundled into the wheel install it was invalidated by
# every source change and re-resolved the whole closure -- torch, ray, rdkit --
# from PyPI each rebuild. requirements.txt is exactly the wheel's Requires-Dist:
# pyproject.toml derives the dynamic dependencies from this file. This stage
# shares nothing with the deps chain -- different base, different interpreter --
# so it installs them itself.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/cache/pip \
    pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Bind-mounted rather than COPYed: a COPY keeps the wheel in a layer for the life
# of the image, and deleting it afterwards only writes a whiteout. `wheel-export`
# holds nothing but the wheel, so mounting it keys this layer on the wheel's bytes
# alone -- mounting `wheel` would key it on that stage's whole filesystem.
# --no-deps because requirements.txt is installed above; --no-cache-dir rather
# than a cache mount because there is nothing to cache for a local file.
RUN --mount=type=bind,from=wheel-export,target=/wheel \
    pip install --no-deps --no-cache-dir /wheel/*.whl

# PEP 639 installs the wheel's legal material under its dist-info/licenses
# directory. Copy that exact payload to the conventional system location so a
# container recipient does not need Python packaging knowledge to find it.
RUN set -eux; \
    license_dir="$(find /opt/venv/lib/python3.12/site-packages -path '*/bionemo_ir-*.dist-info/licenses' -type d -print -quit)"; \
    test -n "${license_dir}"; \
    test -f "${license_dir}/LICENSE"; \
    test -f "${license_dir}/THIRD_PARTY_NOTICES.md"; \
    test -f "${license_dir}/LICENSES/MPL-2.0.txt"; \
    mkdir -p /usr/share/licenses/bionemo-ir; \
    cp -a "${license_dir}/." /usr/share/licenses/bionemo-ir/

# Check what can actually be checked without a GPU: that the wheel carried the
# compiled extension, and that the only CUDA library it needs is the driver's
# libcuda.so.1. Importing it here is impossible -- dlopen has to resolve
# DT_NEEDED libcuda.so.1, and the container toolkit injects that from the host
# only at `docker run --gpus`, not at build time. The import is verified at run
# time instead.
# Runtime details: docs/ref/docker-images.md.
# Asserting libcuda.so.1 is present says nothing about what else crept in, so
# also fail if the closure picks up a toolkit or math library this base
# deliberately does not ship. Written as `if ... exit 1`, not `! grep`:
# `set -e` is specified to ignore the exit status of a !-negated command, so the
# negated form would report every leak as a clean build.
RUN set -eux; \
    so="$(find /opt/venv -name '_cutedsl_kernels*.so' -print -quit)"; \
    test -n "${so}"; \
    deps="$(ldd "${so}")"; \
    echo "${deps}"; \
    echo "${deps}" | grep -q 'libcuda\.so\.1'; \
    if echo "${deps}" | grep -Eq 'lib(cudart|cublas|cublasLt|cudnn|nvrtc|nvJitLink|cufft|curand|cusparse|cusolver)\.so'; then \
        echo "extension links a CUDA library this image does not ship" >&2; \
        exit 1; \
    fi; \
    echo "extension installed: ${so}"

# Ubuntu 24.04 already ships a UID 1000 (`ubuntu`), so creating one blindly
# fails with "UID already in use". Reuse whoever holds it instead.
ARG USER_UID=1000
RUN if getent passwd "${USER_UID}" >/dev/null; then \
        echo "reusing existing uid ${USER_UID}: $(getent passwd "${USER_UID}" | cut -d: -f1)"; \
    else \
        useradd -m -u "${USER_UID}" bioir; \
    fi
USER ${USER_UID}
WORKDIR /tmp
