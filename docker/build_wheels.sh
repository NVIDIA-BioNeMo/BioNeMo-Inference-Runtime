#!/bin/bash

pip install --no-cache-dir -r /src/docker/dependencies.txt
pip install tensorrt-llm==1.0.0rc0 --no-cache-dir --no-deps

if [ ! -d /packages ]; then
    mkdir -p /packages
else
    rm -rf /packages/*.whl
fi

cd /src && \
    TRT_ROOT_DIR=/usr/local/tensorrt ./scripts/build_cpp.sh && \
cd /src && \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir /packages
