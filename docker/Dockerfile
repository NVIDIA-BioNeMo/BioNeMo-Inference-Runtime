ARG BASE_IMAGE=tensorrt-llm
ARG BASE_TAG=base

FROM ${BASE_IMAGE}:${BASE_TAG} AS builder
COPY . /src

FROM ${BASE_IMAGE}:${BASE_TAG} as release
COPY ./packages /packages
RUN cd /packages && \
    pip install tensorrt_bionemo-*.whl --no-cache-dir && \
    rm -rf /packages && \
    python -c "import importlib.util; spec = importlib.util.find_spec('tensorrt_llm'); print(spec.submodule_search_locations[0])" > /tmp/package_path.txt
RUN TRTLLM_PACKAGE_PATH=$(cat /tmp/package_path.txt) && \
    bash -c "sed -i 's/mode=ctypes.RTLD_GLOBAL/mode=ctypes.RTLD_LOCAL/' $TRTLLM_PACKAGE_PATH/plugin/plugin.py" && \
    rm -rf /tmp/package_path.txt