#!/bin/bash
CPU_COUNT=$(nproc --all)

pip install cuequivariance-ops-torch-cu13==0.10.0
mkdir -p cpp/build && cd cpp/build && \
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DFAST_BUILD=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -S .. && \
cmake --build $(pwd) \
    --config Debug \
    --parallel $CPU_COUNT && make install
