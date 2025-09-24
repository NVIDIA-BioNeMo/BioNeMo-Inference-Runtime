#!/bin/bash
arch=$(uname -i)
OUTPUT_DIR=$1

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
TENSORRT_LLM_PATH=/src/3rdparty/TensorRT-LLM
pip install --no-cache-dir -r /src/docker/dependencies.txt

# TODO: download tensorrt-llm from artifactory
# build tensorrt-llm from source
cd $TENSORRT_LLM_PATH
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --fast
cp $TENSORRT_LLM_PATH/build/tensorrt_llm*$arch.whl $OUTPUT_DIR/

pip install $OUTPUT_DIR/tensorrt_llm*$arch.whl --no-cache-dir --no-deps

CUEQUIV_OPS_PATH=/src/3rdparty/cuequiv-ops
pip install nanobind pynvml scikit-build-core build
cd $CUEQUIV_OPS_PATH/cuequivariance_ops && \
    sed -i 's/70-real//g' CMakeLists.txt && \
    SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release;-DCUEQUIV_OPS_CMAKE_CUDA_ARCHITECTURES=CUEQUIV_OPS" \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR/ && \
    pip install $OUTPUT_DIR/cuequivariance_ops*$arch.whl
cd $CUEQUIV_OPS_PATH/cuequivariance_ops_torch && \
    SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR/

cd /src && \
    TRT_ROOT_DIR=/usr/local/tensorrt ./scripts/build_cpp.sh && \
cd /src && \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir $OUTPUT_DIR


