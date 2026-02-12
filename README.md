<div align="center">

# TensorRT-BioNemo

<h4> A TensorRT Toolbox for Optimized Structure Prediction Model Inference, inspired by TensorRT-LLM </h4>
<div align="left">

## Getting Started

### Develop Docker

Create the base image by the following command, the command will create the TRT-LLM base image, contain TensorRT development packages:

```bash
$ make -C docker base
```

Or pull from: `<internal-registry>/tensorrt-llm:base`

Run:

```bash
$ docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it \
  <internal-registry>/tensorrt-llm:base
```

Inside docker container, just:

```bash
$ git clone <internal-repository>
# Note: --no-build-isolation must have to avoid re-install torch
$ cd tensorrt-bionemo && pip install --no-build-isolation -v -e .[dev]
```

To create a develop environment, with needed dependencies for running tests.

#### Testing

```bash
$ pytest -s $(pwd)/tests
```

### Release Docker
Use the same development docker to build the wheel package:
```bash
# clean building caches.
$ rm -rf cpp/build 
$ pip install build
# Set RECOMPILE_CPP=1 to avoid caches
$ RECOMPILE_CPP=1 python -m build --wheel --no-isolation --outdir packages/
```

## Troubleshoots

```
+ TensorRT bug on chunking loop (fixed for version 10.11): NVBug 5190992

+ TRT activation's memory are not reused for the trifast custom kernel: an internal discussion
```
