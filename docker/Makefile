PWD := $(shell pwd)
PROJECT_DIR := $(PWD)/..
PACKAGE_DIR ?= $(PROJECT_DIR)/packages
TRTLLM_DIR := $(PROJECT_DIR)/3rdparty/TensorRT-LLM
BASE_IMAGE := tensorrt-llm
BASE_TAG := base
RELEASE_IMAGE ?= trtbnm
RELEASE_TAG ?= release
TORCH_BASE_TAG ?= 25.12-py3
CPU_ARCH := $(shell uname -m)

ifeq ($(CPU_ARCH),x86_64)
	BUILD_PLATFORM := linux/amd64
else ifeq ($(CPU_ARCH),aarch64)
	BUILD_PLATFORM := linux/arm64
else
	$(error Unsupported architecture: $(CPU_ARCH))
endif

.PHONY: base submodules

base: submodules
	docker build --pull \
		--platform $(BUILD_PLATFORM) \
		--target devel \
		--file $(TRTLLM_DIR)/docker/Dockerfile.multi \
		--build-arg BASE_TAG=$(TORCH_BASE_TAG) \
		--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch \
		--build-arg TRT_LLM_VER=1.2.1 \
		--build-arg TARGETARCH=$(CPU_ARCH) \
		--tag $(BASE_IMAGE):$(BASE_TAG) $(TRTLLM_DIR)

submodules:
	@echo "Initializing and updating Git submodules..."
	cd $(PROJECT_DIR)
	git submodule update --init --recursive
	@echo "Git submodules updated."
