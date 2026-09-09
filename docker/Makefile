# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Image guide: docs/ref/docker-images.md.

SHELL := /bin/bash
PWD := $(shell pwd)
PROJECT_DIR ?= $(PWD)/..

# Base images. Both pull anonymously -- no NGC key, no docker login. Override to
# build against a different PyTorch/CUDA combination, but check that its torch
# and triton satisfy requirements.txt without a downgrade.
# Compatibility notes: docs/ref/docker-images.md.
BASE_IMAGE ?= nvcr.io/nvidia/pytorch
BASE_TAG ?= 26.05-py3
RUNTIME_IMAGE ?= nvidia/cuda
RUNTIME_TAG ?= 13.0.2-base-ubuntu24.04

REGISTRY_IMAGE ?= bionemo-ir

COMMIT_SHORT_SHA ?= $(shell git rev-parse --short=7 HEAD)
BIOIR_BRANCH_ROUGH := $(shell git rev-parse --abbrev-ref HEAD)
BIOIR_BRANCH := $(subst /,_,$(BIOIR_BRANCH_ROUGH))
ifdef CI
TAG ?= bioir$(COMMIT_SHORT_SHA)
else
TAG ?= $(BIOIR_BRANCH)_bioir$(COMMIT_SHORT_SHA)
endif

DEV_IMAGE_AND_TAG := $(REGISTRY_IMAGE):$(TAG)
# Dependency-only image, keyed by the base it was built from: no source copy, no
# editable install, no CMake tree. Invalidated only by requirements*.txt, which
# is what makes it usable as a rolling CI cache.
DEPS_IMAGE_AND_TAG ?= $(REGISTRY_IMAGE):deps-$(BASE_TAG)
RUNTIME_IMAGE_AND_TAG ?= $(REGISTRY_IMAGE):runtime-$(TAG)

# Match the host UID/GID so the bind-mounted checkout does not collect
# root-owned build artifacts.
USER_UID ?= $(shell id -u)
USER_GID ?= $(shell id -g)

# Layer caching. Default: reuse the local BuildKit cache. CI additionally passes
# DOCKER_CACHE_FROM=<image:tag> to seed from a registry image;
# BUILDKIT_INLINE_CACHE embeds the metadata a cold runner needs to warm up.
#
# That metadata is asked for in CI only. Preparing it walks every layer of a
# ~20GB image on each build, so a fully cached local rebuild -- what
# `docker/dev.sh` does on every attach -- would pay a long export for something
# no local build reads.
DOCKER_CACHE_FROM ?=
_CACHE_ARGS := $(if $(or $(CI),$(DOCKER_CACHE_FROM)),--build-arg BUILDKIT_INLINE_CACHE=1,) \
	$(if $(DOCKER_CACHE_FROM),--cache-from $(DOCKER_CACHE_FROM),)

# Compile jobs for the wheel stage. In a CI job container nproc reports 1
# whatever the scheduler granted, so pass the scheduler's allocation through
# when there is one. Locally, cap at 32 rather than letting the Dockerfile ask
# the machine; override BUILD_JOBS on a host with fewer cores or less RAM.
ifdef CI
BUILD_JOBS ?= $(CI_SLURM_CPUS_PER_TASK)
else
BUILD_JOBS ?= 32
endif
_JOBS_ARG := $(if $(BUILD_JOBS),--build-arg BUILD_JOBS=$(BUILD_JOBS),)

CPU_ARCH := $(shell uname -m)
ifeq ($(CPU_ARCH),x86_64)
BUILD_PLATFORM := linux/amd64
else ifneq ($(filter aarch64 arm64,$(CPU_ARCH)),)
BUILD_PLATFORM := linux/arm64
else
$(error Unsupported architecture: $(CPU_ARCH))
endif

_BUILD := docker build $(_CACHE_ARGS) --platform $(BUILD_PLATFORM) \
	--file $(PROJECT_DIR)/docker/Dockerfile

# Deliberately not part of _BUILD: `deps` must NOT pass them, so that the
# Dockerfile's digest-pinned BASE_TAG default stands and the image keeps the
# compile identity the tracked CUBIN packs were built against.
_BASE_ARGS := --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	--build-arg BASE_TAG=$(BASE_TAG)

.PHONY: help deps dev wheel runtime submodules print-build-platform

help:
	@echo "Targets:"
	@echo "  deps      dependency-only image, no source  -> $(DEPS_IMAGE_AND_TAG)"
	@echo "  dev       daily dev image, bind-mount source -> $(DEV_IMAGE_AND_TAG)"
	@echo "  wheel     build the wheel into ./dist"
	@echo "  runtime   minimal wheel-only runtime image  -> $(RUNTIME_IMAGE_AND_TAG)"
	@echo "  submodules  git submodule update --init --recursive"

# Builds the Dockerfile's `devdeps` stage: runtime deps plus the dev/test deps
# the suite imports. Named for what it is used for, not for the stage.
deps:
	$(_BUILD) --target devdeps --tag $(DEPS_IMAGE_AND_TAG) $(PROJECT_DIR)

# docker/dev.sh drives this target with its own REGISTRY_IMAGE/TAG, then owns
# the container and the mounts around it.
dev:
	$(_BUILD) $(_BASE_ARGS) --target dev \
		--build-arg USER_UID=$(USER_UID) \
		--build-arg USER_GID=$(USER_GID) \
		--tag $(DEV_IMAGE_AND_TAG) $(PROJECT_DIR)

# Exports the built wheel to ./dist on the host rather than leaving it in an
# image layer. Targets wheel-export, not wheel: see the Dockerfile.
wheel:
	$(_BUILD) $(_BASE_ARGS) --target wheel-export $(_JOBS_ARG) \
		$(if $(CUDA_TAG),--build-arg CUDA_TAG=$(CUDA_TAG),) \
		--output type=local,dest=$(PROJECT_DIR)/dist \
		$(PROJECT_DIR)
	@echo "wheel written to $(PROJECT_DIR)/dist"

runtime:
	$(_BUILD) $(_BASE_ARGS) --target runtime $(_JOBS_ARG) \
		--build-arg RUNTIME_IMAGE=$(RUNTIME_IMAGE) \
		--build-arg RUNTIME_TAG=$(RUNTIME_TAG) \
		$(if $(CUDA_TAG),--build-arg CUDA_TAG=$(CUDA_TAG),) \
		--tag $(RUNTIME_IMAGE_AND_TAG) $(PROJECT_DIR)

submodules:
	git -C $(PROJECT_DIR) submodule update --init --recursive

# The platform the images are built for, so CI can key its layer cache on the
# same value this file derives rather than re-deriving `uname -m` beside it.
print-build-platform:
	@echo $(BUILD_PLATFORM)
