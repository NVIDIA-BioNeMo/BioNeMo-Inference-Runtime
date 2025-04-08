#!/bin/bash
CPU_COUNT=$(nproc --all)
cd cpp/build && \
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTRT_LIB_DIR=/usr/local/tensorrt/lib \
    -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include \
    -DFAST_BUILD=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -S .. && \
cmake --build $(pwd) \
    --config Debug \
    --parallel $CPU_COUNT \
    --target nvinfer_plugin_tensorrt_bionemo && \
cd ../.. && \
    mkdir -p tensorrt_bionemo/libs && \
    cp cpp/build/tensorrt_bionemo/plugins/libnvinfer_plugin_tensorrt_bionemo.so tensorrt_bionemo/libs/

# Patch load TensorRT-LLM as local scope to avoid conflicts with TensorRT-LLM
sed -i 's/mode=ctypes.RTLD_GLOBAL/mode=ctypes.RTLD_LOCAL/' 3rdparty/TensorRT-LLM/tensorrt_llm/plugin/plugin.py
