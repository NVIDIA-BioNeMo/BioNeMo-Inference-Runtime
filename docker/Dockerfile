ARG BASE_IMAGE=tensorrt-llm
ARG BASE_TAG=base

FROM ${BASE_IMAGE}:${BASE_TAG} AS builder
COPY . /src

FROM ${BASE_IMAGE}:${BASE_TAG} as release
COPY ./packages /packages
COPY ./docker/dependencies.txt /packages/dependencies.txt
RUN cd /packages && \
    # WAR, this work for B200 and pytorch:25.04-py3
    pip install --no-cache-dir -r /packages/dependencies.txt && \
    pip install tensorrt-llm==1.0.0rc0 --no-cache-dir --no-deps && \
    pip install tensorrt_bionemo-*.whl --no-cache-dir --no-deps && \
    rm -rf /packages && \
    TRTLLM_PACKAGE_PATH=$(python -c "import importlib.util; spec = importlib.util.find_spec('tensorrt_llm'); print(spec.submodule_search_locations[0])") && \
    bash -c "sed -i 's/mode=ctypes.RTLD_GLOBAL/mode=ctypes.RTLD_LOCAL/' $TRTLLM_PACKAGE_PATH/plugin/plugin.py"
