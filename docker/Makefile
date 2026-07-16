SHELL := /bin/bash
PWD := $(shell pwd)
PROJECT_DIR ?= $(PWD)/..
PACKAGE_DIR ?= $(PROJECT_DIR)/packages
TRTLLM_DIR := $(PROJECT_DIR)/3rdparty/TensorRT-LLM

# TRTLLM image parameters
TRT_LLM_VER ?= 1.2.0
TRTLLM_COMMIT ?= $(shell cd ${PROJECT_DIR} && git ls-tree --abbrev=7 HEAD 3rdparty/TensorRT-LLM | awk '{print $$3}')
TRTLLM_IMAGE ?= tensorrt-llm
TRTLLM_TAG ?= trtllm$(TRT_LLM_VER)_$(TRTLLM_COMMIT)
TORCH_BASE_IMAGE ?= nvcr.io/nvidia/pytorch
TORCH_BASE_TAG ?= 25.12-py3

# TRTBNM base parameters
TRTBNM_BASE_IMAGE := trtbnm_base
TRTBNM_BASE_TAG ?= trtllm$(TRT_LLM_VER)_$(TRTLLM_COMMIT)_with_cuda_deps

# TRTBNM image parameters
RELEASE_IMAGE ?= trtbnm
RELEASE_TAG ?= Release
REGISTRY_IMAGE ?= tensorrt-bionemo

COMMIT_SHORT_SHA ?= $(shell git rev-parse --short=7 HEAD)
TRTBNM_BRANCH_ROUGH := $(shell git rev-parse --abbrev-ref HEAD)
TRTBNM_BRANCH := $(subst /,_,$(TRTBNM_BRANCH_ROUGH))
ifdef CI_COMMIT_BRANCH
TAG ?= trtbnm$(COMMIT_SHORT_SHA)_from_$(TRTBNM_BASE_TAG)
else
TAG ?= $(TRTBNM_BRANCH)_trtbnm$(COMMIT_SHORT_SHA)_from_$(TRTBNM_BASE_TAG)
endif
REGISTRY_IMAGE_AND_TAG := ${REGISTRY_IMAGE}:${TAG}

CPU_ARCH := $(shell uname -m)
ifeq ($(CPU_ARCH),x86_64)
	BUILD_PLATFORM := linux/amd64
else ifeq ($(CPU_ARCH),aarch64)
	BUILD_PLATFORM := linux/arm64
else
	$(error Unsupported architecture: $(CPU_ARCH))
endif

.PHONY: submodules trtllm trtbnm_base trtbnm_dev

trtbnm_dev:
	@echo "Make target build_with_source, begin"
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& echo "pwd=$$(pwd)" \
		&& printenv \
		&& echo "Build ${REGISTRY_IMAGE_AND_TAG} with source for commit $(COMMIT_SHORT_SHA), begin" \
		&& docker build \
			--no-cache \
			--platform $(BUILD_PLATFORM) \
			--target dev_stage \
			--file $(PROJECT_DIR)/docker/Dockerfile.with-source \
			--build-arg BASE_IMAGE=$(REGISTRY_IMAGE)/$(TRTBNM_BASE_IMAGE) \
			--build-arg BASE_TAG=$(TRTBNM_BASE_TAG) \
			--tag $(REGISTRY_IMAGE_AND_TAG) \
			$(PROJECT_DIR) \
		&& echo "Build ${REGISTRY_IMAGE_AND_TAG}, end" \
		&& popd
	@echo "Make target build_with_source, end"

trtbnm_base:
	@echo "Make target trtbnm_base, begin"
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& echo "pwd=$$(pwd)" \
		&& echo "Build $(REGISTRY_IMAGE)/$(TRTBNM_BASE_IMAGE):$(TRTBNM_BASE_TAG) as base for commit $(COMMIT_SHORT_SHA), begin" \
		&& docker build \
			--no-cache \
			--platform $(BUILD_PLATFORM) \
			--file $(PROJECT_DIR)/docker/Dockerfile.trtbnm_base \
			--build-arg BASE_IMAGE=$(REGISTRY_IMAGE)/$(TRTLLM_IMAGE) \
			--build-arg BASE_TAG=$(TRTLLM_TAG) \
			--tag $(REGISTRY_IMAGE)/$(TRTBNM_BASE_IMAGE):$(TRTBNM_BASE_TAG) \
			$(PROJECT_DIR) \
		&& echo "Build ${REGISTRY_IMAGE}/${TRTBNM_BASE_IMAGE}:${TRTBNM_BASE_TAG}, end" \
		&& popd
	@echo "Make target trtbnm_base, end"

trtllm:
	@echo "Make target trtllm, begin"
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& echo "pwd=$$(pwd)" \
		&& echo "Build ${REGISTRY_IMAGE}/${TRTLLM_IMAGE}:${TRTLLM_TAG}, begin" \
		&& docker build --pull \
			--platform $(BUILD_PLATFORM) \
			--target devel \
			--file $(TRTLLM_DIR)/docker/Dockerfile.multi \
			--build-arg BASE_TAG=$(TORCH_BASE_TAG) \
			--build-arg BASE_IMAGE=$(TORCH_BASE_IMAGE) \
			--build-arg TRT_LLM_VER=$(TRT_LLM_VER) \
			--build-arg TARGETARCH=$(CPU_ARCH) \
			--tag $(REGISTRY_IMAGE)/$(TRTLLM_IMAGE):$(TRTLLM_TAG) \
			$(TRTLLM_DIR) \
		&& echo "Build ${REGISTRY_IMAGE}/${TRTLLM_IMAGE}:${TRTLLM_TAG}, end" \
		&& popd
	@echo "Make target trtllm, end"

submodules:
	@echo "Initializing and updating Git submodules..."
	@echo "PROJECT_DIR=${PROJECT_DIR}"
	pushd $(PROJECT_DIR) \
		&& git submodule update --init --recursive \
	 	&& popd
	@echo "Git submodules updated."
