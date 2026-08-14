SHELL := /bin/bash
PWD := $(shell pwd)
PROJECT_DIR ?= $(PWD)/..

# PyTorch development base
TORCH_BASE_IMAGE ?= nvcr.io/nvidia/pytorch
TORCH_BASE_TAG ?= 26.05-py3

# BioIR image parameters
REGISTRY_IMAGE ?= bionemo-ir

COMMIT_SHORT_SHA ?= $(shell git rev-parse --short=7 HEAD)
BIOIR_BRANCH_ROUGH := $(shell git rev-parse --abbrev-ref HEAD)
BIOIR_BRANCH := $(subst /,_,$(BIOIR_BRANCH_ROUGH))
ifdef CI_COMMIT_BRANCH
TAG ?= bioir$(COMMIT_SHORT_SHA)_from_pytorch$(TORCH_BASE_TAG)
else
TAG ?= $(BIOIR_BRANCH)_bioir$(COMMIT_SHORT_SHA)_from_pytorch$(TORCH_BASE_TAG)
endif
REGISTRY_IMAGE_AND_TAG := ${REGISTRY_IMAGE}:${TAG}

# Dependency-only image: the Dockerfile's `ci_stage`, with no source copy, no
# editable install and no CMake build tree. CI runs build:extension and the test
# shards on this instead of the full dev image.
REQS_IMAGE_AND_TAG ?= ${REGISTRY_IMAGE}:reqs-${TORCH_BASE_TAG}

# Docker layer caching (CI build:image follow-up).
#   Default: reuse the local BuildKit layer cache. The CI runner talks to a
#   persistent host docker daemon, so unchanged layers — notably the uv dependency
#   install in requirements_stage (before COPY .) — survive between builds.
#   CI also passes DOCKER_CACHE_FROM=<image:tag> to seed from a registry cache
#   image; BUILDKIT_INLINE_CACHE embeds cache metadata into the built image so a
#   cold/rotated runner can warm up from that registry image.
DOCKER_CACHE_FROM ?=
_DOCKER_CACHE_ARGS := --build-arg BUILDKIT_INLINE_CACHE=1 $(if $(DOCKER_CACHE_FROM),--cache-from $(DOCKER_CACHE_FROM),)

CPU_ARCH := $(shell uname -m)
ifeq ($(CPU_ARCH),x86_64)
BUILD_PLATFORM := linux/amd64
else ifneq ($(filter aarch64 arm64,$(CPU_ARCH)),)
BUILD_PLATFORM := linux/arm64
else
$(error Unsupported architecture: $(CPU_ARCH))
endif

.PHONY: submodules bioir_dev bioir_reqs

bioir_reqs:
	@echo "Build ${REQS_IMAGE_AND_TAG} (ci_stage), begin"
	pushd $(PROJECT_DIR) \
		&& docker build \
			$(_DOCKER_CACHE_ARGS) \
			--platform $(BUILD_PLATFORM) \
			--target ci_stage \
			--file $(PROJECT_DIR)/docker/Dockerfile.with-source \
			--build-arg BASE_IMAGE=$(TORCH_BASE_IMAGE) \
			--build-arg BASE_TAG=$(TORCH_BASE_TAG) \
			--tag $(REQS_IMAGE_AND_TAG) \
			$(PROJECT_DIR) \
		&& popd
	@echo "Build ${REQS_IMAGE_AND_TAG}, end"

bioir_dev:
	@echo "Make target build_with_source, begin"
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& echo "pwd=$$(pwd)" \
		&& printenv \
		&& echo "Build ${REGISTRY_IMAGE_AND_TAG} with source for commit $(COMMIT_SHORT_SHA), begin" \
		&& docker build \
			$(_DOCKER_CACHE_ARGS) \
			--platform $(BUILD_PLATFORM) \
			--target dev_stage \
			--file $(PROJECT_DIR)/docker/Dockerfile.with-source \
			--build-arg BASE_IMAGE=$(TORCH_BASE_IMAGE) \
			--build-arg BASE_TAG=$(TORCH_BASE_TAG) \
			--tag $(REGISTRY_IMAGE_AND_TAG) \
			$(PROJECT_DIR) \
		&& echo "Build ${REGISTRY_IMAGE_AND_TAG}, end" \
		&& popd
	@echo "Make target build_with_source, end"

submodules:
	@echo "Initializing and updating Git submodules..."
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& git submodule update --init --recursive \
	 	&& popd
	@echo "Git submodules updated."
