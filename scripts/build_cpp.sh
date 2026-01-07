#!/bin/bash
CPU_COUNT=$(nproc --all)
TRT_ROOT_DIR=${TRT_ROOT_DIR:-"/usr/local/tensorrt"}

mkdir -p cpp/build && cd cpp/build && \
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTRT_ROOT_DIR=${TRT_ROOT_DIR} \
    -DFAST_BUILD=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -S .. && \
cmake --build $(pwd) \
    --config Debug \
    --parallel $CPU_COUNT && make install
