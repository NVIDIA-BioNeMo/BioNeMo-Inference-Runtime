SHELL := /bin/bash
PWD := $(shell pwd)
PROJECT_DIR ?= $(PWD)/..

# PyTorch development base
TORCH_BASE_IMAGE ?= nvcr.io/nvidia/pytorch
TORCH_BASE_TAG ?= 25.12-py3

# TRTBNM image parameters
REGISTRY_IMAGE ?= tensorrt-bionemo

COMMIT_SHORT_SHA ?= $(shell git rev-parse --short=7 HEAD)
TRTBNM_BRANCH_ROUGH := $(shell git rev-parse --abbrev-ref HEAD)
TRTBNM_BRANCH := $(subst /,_,$(TRTBNM_BRANCH_ROUGH))
ifdef CI_COMMIT_BRANCH
TAG ?= trtbnm$(COMMIT_SHORT_SHA)_from_pytorch$(TORCH_BASE_TAG)
else
TAG ?= $(TRTBNM_BRANCH)_trtbnm$(COMMIT_SHORT_SHA)_from_pytorch$(TORCH_BASE_TAG)
endif
REGISTRY_IMAGE_AND_TAG := ${REGISTRY_IMAGE}:${TAG}

# Docker layer caching (CI build:docker-image follow-up).
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

.PHONY: submodules trtbnm_dev

trtbnm_dev:
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
