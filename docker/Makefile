#
# Notes:
#	- Before running the 'dev' make target
#		- An ssh agent must be running.
#		- An ssh private key file must be on path for git@<internal-git-host>:12051
#		- This will be checked by the command 'ssh -T -p 12051 git@<internal-git-host>'
#

PWD := $(shell pwd)
PROJECT_DIR := $(PWD)/..
PACKAGE_DIR ?= $(PROJECT_DIR)/packages
TRTLLM_DIR := $(PROJECT_DIR)/3rdparty/TensorRT-LLM
DOCKERFILE_DEV := $(PROJECT_DIR)/docker/Dockerfile_dev
TRTLLM_COMMIT := $(shell cd .. && git ls-tree --abbrev HEAD 3rdparty/TensorRT-LLM | awk '{print $$3}')
BASE_IMAGE := <internal-registry>/tensorrt-llm
TRT_LLM_VER := 1.2.0
BASE_TAG := $(TRT_LLM_VER)_$(TRTLLM_COMMIT)
DEV_IMAGE := <internal-git-host>:5005/the internal repository:dev_from_trtllm_$(BASE_TAG)
RELEASE_IMAGE ?= trtbnm
RELEASE_TAG ?= release
TORCH_BASE_TAG ?= 25.10-py3
CPU_ARCH := $(shell uname -m)


ifeq ($(CPU_ARCH),x86_64)
	BUILD_PLATFORM := linux/amd64
else ifeq ($(CPU_ARCH),aarch64)
	BUILD_PLATFORM := linux/arm64
else
	$(error Unsupported architecture: $(CPU_ARCH))
endif

.PHONY: dev trtllm submodules

dev: trtllm
	@echo "Build dev image ${DEV_IMAGE}, begin"
	eval $(ssh-agent)
	ssh -T -p 12051 git@<internal-git-host>
	docker build \
		--ssh default \
		--build-arg BASE_FOR_DEV_IMAGE=$(BASE_IMAGE):$(BASE_TAG) \
		--file Dockerfile_dev \
		--tag $(DEV_IMAGE) \
		.
	@echo "Build dev image ${DEV_IMAGE}, end"


trtllm: submodules
	@echo "Build TRTLLM docker image with commit $(TRTLLM_COMMIT), begin"
	@echo "The output docker image will be $(BASE_IMAGE):$(BASE_TAG)"
	docker build --pull \
		--platform $(BUILD_PLATFORM) \
		--target devel \
		--file $(TRTLLM_DIR)/docker/Dockerfile.multi \
		--build-arg BASE_TAG=$(TORCH_BASE_TAG) \
		--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch \
		--build-arg TRT_LLM_VER=$(TRT_LLM_VER) \
		--build-arg TARGETARCH=$(CPU_ARCH) \
		--tag $(BASE_IMAGE):$(BASE_TAG) $(TRTLLM_DIR)
	@echo "Build TRTLLM docker image with commit $(TRTLLM_COMMIT), end"


submodules:
	@echo "Initializing and updating Git submodules..."
	cd $(PROJECT_DIR)
	git submodule update --init --recursive
	@echo "Git submodules updated."
