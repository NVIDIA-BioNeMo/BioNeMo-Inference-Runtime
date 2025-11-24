#!/bin/bash
arch=$(uname -i)
OUTPUT_DIR=$1
CUE_VERSION=${2:-0.7.0}

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
TENSORRT_LLM_PATH=/src/3rdparty/TensorRT-LLM
pip install --no-cache-dir -r /src/docker/dependencies.txt

# TODO: download tensorrt-llm from artifactory
# build tensorrt-llm from source
cd $TENSORRT_LLM_PATH
sed -i 's/mode=ctypes.RTLD_GLOBAL/mode=ctypes.RTLD_LOCAL/' $TENSORRT_LLM_PATH/tensorrt_llm/plugin/plugin.py
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --fast
cp $TENSORRT_LLM_PATH/build/tensorrt_llm*$arch.whl $OUTPUT_DIR/

# Install for TRT-BNM cpp
pip install $OUTPUT_DIR/tensorrt_llm*$arch.whl --no-cache-dir --no-deps


CUEQUIV_OPS_PATH=/src/3rdparty/cuequiv-ops

# copy triton tuning cache for trimul kernel
cp /src/docker/etc/*.json $CUEQUIV_OPS_PATH/cuequivariance_ops/cuequivariance_ops/triton/cache


NVFATBIN_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
cp $NVFATBIN_DIR/include/*.h /usr/local/cuda/include
cp $NVFATBIN_DIR/lib/libnvfatbin.so.13 /usr/local/cuda/lib64/libnvfatbin.so

pip install nanobind pynvml scikit-build-core build
cd $CUEQUIV_OPS_PATH/cuequivariance_ops && \
    sed -i 's/70-real//g' CMakeLists.txt && \
    SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release;-DCUEQUIV_OPS_CMAKE_CUDA_ARCHITECTURES=CUEQUIV_OPS;-DCMAKE_CUDA_FLAGS=-I/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include" \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR/ && \
    pip install $OUTPUT_DIR/cuequivariance_ops*$arch.whl
cd $CUEQUIV_OPS_PATH/cuequivariance_ops_torch && \
    SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR/

cd /src && \
    TRT_ROOT_DIR=/usr/local/tensorrt ./scripts/build_cpp.sh && \
cd /src && \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR
