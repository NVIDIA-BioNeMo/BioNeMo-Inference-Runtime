ARG TARGETPLATFORM

ARG BASE_IMAGE=tensorrt-llm
ARG BASE_TAG=base
ARG CUE_VERSION=0.7.0


FROM ${BASE_IMAGE}:${BASE_TAG} AS nvfatbin_devel
RUN pip install nvidia-nvfatbin==13.0.39 --no-cache-dir && \
    NVFATBIN_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cu13 && \
    cp $NVFATBIN_DIR/include/*.h /usr/local/cuda/include && \
    cp $NVFATBIN_DIR/lib/libnvfatbin.so.13 /usr/local/cuda/lib64/libnvfatbin.so

FROM nvfatbin_devel AS builder
# Install nvfatbin libs for building cuequivariance >= 0.7.0
COPY . /src

FROM nvfatbin_devel AS release
ARG CUE_VERSION=0.7.0
# Example: Conditionally set a variable based on architecture
COPY ./docker/etc/0001-Patch-to-support-PTX-9.0.patch /tmp/0001-Patch-to-support-PTX-9.0.patch

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
        echo "Detected architecture: $ARCHITECTURE" && \
    git clone --single-branch --branch v3.4.0 https://github.com/triton-lang/triton.git /opt/triton && \
    cd /opt/triton && \
    git apply /tmp/0001-Patch-to-support-PTX-9.0.patch && \
    rm -rf /tmp/0001-Patch-to-support-PTX-9.0.patch && \
    pip install -r python/requirements.txt --no-cache-dir && \
    pip install -e .

# Mount the host's 'my_data' directory to '/mnt/data' in the build container
# NOTE: no-deps flags is necessary to avoid conflicts with the dependencies in the tensorrt-llm requirements.txt file
COPY ./docker/dependencies.txt /tmp/dependencies.txt
COPY ./docker/etc/*.json /tmp/etc/
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
RUN --mount=type=bind,source=./packages,target=/packages \
    cd /packages && \
    pip install --no-cache-dir -r /tmp/dependencies.txt && \
    bash -c "pip install cuequivariance_ops*-$CUE_VERSION-*$ARCHITECTURE.whl cuequivariance_ops_torch*-$CUE_VERSION-*$ARCHITECTURE.whl --no-cache-dir" && \
    bash -c "pip install cuequivariance_torch==$CUE_VERSION cuequivariance==$CUE_VERSION --no-cache-dir" && \
    bash -c "pip install /packages/tensorrt_llm-*$ARCHITECTURE.whl --no-cache-dir --no-deps" && \
    bash -c "pip install tensorrt_bionemo-*$ARCHITECTURE.whl --no-cache-dir --no-deps" && \
    CUEQ_OPS_PACKAGE_PATH=$(python -c "import importlib.util; spec = importlib.util.find_spec('cuequivariance_ops'); print(spec.submodule_search_locations[0])") && \
    cp /tmp/etc/*.json $CUEQ_OPS_PACKAGE_PATH/triton/cache/ && \
    rm -rf /tmp/dependencies.txt && \
    rm -rf /tmp/etc/
