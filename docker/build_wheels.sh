#!/bin/bash

pip install -r /src/requirements.txt --no-cache-dir

if [ ! -d /packages ]; then
    mkdir -p /packages
else
    rm -rf /packages/*.whl
fi

cd /src && \
    TRT_ROOT_DIR=/usr/local/tensorrt ./scripts/build_cpp.sh && \
cd /src && \
    python -m build . --wheel --skip-dependency-check --no-isolation --outdir /packages 

