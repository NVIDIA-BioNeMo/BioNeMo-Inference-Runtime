ARG TARGETPLATFORM

ARG BASE_IMAGE=tensorrt-llm
ARG BASE_TAG=base

FROM ${BASE_IMAGE}:${BASE_TAG} AS builder
COPY . /src

FROM ${BASE_IMAGE}:${BASE_TAG} as release
# Example: Conditionally set a variable based on architecture
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Building for AMD64 architecture"; \
        ARCHITECTURE="x86_64"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Building for ARM64 architecture"; \
        ARCHITECTURE="aarch64"; \
    else \
        echo "Unknown or unsupported architecture: $TARGETPLATFORM"; \
        ARCHITECTURE="unknown"; \
    fi && \
        echo "Detected architecture: $ARCHITECTURE"
COPY ./packages /packages
COPY ./docker/dependencies.txt /packages/dependencies.txt
RUN cd /packages && \
    pip install --no-cache-dir -r /packages/dependencies.txt && \
    bash -c "pip install /packages/tensorrt_llm-*$ARCHITECTURE.whl --no-cache-dir --no-deps" && \
    bash -c "pip install cuequivariance_ops-*$ARCHITECTURE.whl cuequivariance_ops_torch-*$ARCHITECTURE.whl --no-cache-dir" && \
    bash -c "pip install cuequivariance_torch==0.6.1 cuequivariance==0.6.1 --no-cache-dir" && \
    bash -c "pip install tensorrt_bionemo-*$ARCHITECTURE.whl --no-cache-dir --no-deps" && \
    rm -rf /packages && \
    TRTLLM_PACKAGE_PATH=$(python -c "import importlib.util; spec = importlib.util.find_spec('tensorrt_llm'); print(spec.submodule_search_locations[0])") && \
    bash -c "sed -i 's/mode=ctypes.RTLD_GLOBAL/mode=ctypes.RTLD_LOCAL/' $TRTLLM_PACKAGE_PATH/plugin/plugin.py"
