PWD := $(shell pwd)
PROJECT_DIR := $(PWD)/..
PACKAGE_DIR ?= $(PROJECT_DIR)/packages
TRTLLM_DIR := $(PROJECT_DIR)/3rdparty/TensorRT-LLM
BASE_IMAGE := tensorrt-llm
BASE_TAG := base
RELEASE_IMAGE ?= trtbnm
RELEASE_TAG ?= release

.PHONY: trtbnm_wheel release base submodules

release: trtbnm_wheel
	docker build \
		--target release \
		--file $(PROJECT_DIR)/docker/Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BASE_TAG=$(BASE_TAG) \
		--tag $(RELEASE_IMAGE):$(RELEASE_TAG) $(PROJECT_DIR)

trtbnm_wheel: base
	mkdir -p $(PACKAGE_DIR) && \
	docker build \
		--target builder \
		--file $(PROJECT_DIR)/docker/Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BASE_TAG=$(BASE_TAG) \
		--tag trtbnm:wheel $(PROJECT_DIR) && \
	docker run --gpus all \
		-v $(PACKAGE_DIR):/packages \
		trtbnm:wheel \
		bash /src/docker/build_wheels.sh

base: submodules
	docker build --pull \
		--target devel \
		--file $(TRTLLM_DIR)/docker/Dockerfile.multi \
		--tag $(BASE_IMAGE):$(BASE_TAG) $(TRTLLM_DIR)

submodules:
	@echo "Initializing and updating Git submodules..."
	cd $(PROJECT_DIR)
	git submodule update --init --recursive
	@echo "Git submodules updated."
