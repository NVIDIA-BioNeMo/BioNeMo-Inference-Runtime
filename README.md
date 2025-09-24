<div align="center">

# TensorRT-BioNemo

<h4> A TensorRT Toolbox for Optimized Structure Prediction Model Inference, inspired by TensorRT-LLM </h4>
<div align="left">

## Getting Started

### Develop

Please use the `.devcontainer` for `vscode` or `cursor`. In the devcontainer, run as the following:

```bash
$ git submodule update --init --recursive
$ cd 3rdparty/TensorRT-LLM
# Build TensorRT-LLM from source
$ export PATH=$PATH:$HOME/.local/bin # this for conan executable
$ python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --fast # add `-b Debug` for build debug with trt-llm
$ pip install -e . && cd ../..
# Install cuequiv
$ pip install cuequivariance-ops-cu12==0.6.1
# Build TRT plugins for TensorRT-BioNemo
$ ./scripts/build_cpp.sh
# Install as develop mode
$ pip install -e .
```

### Docker

```bash
$ make -C docker release
```

Only build wheels for TensorRT-LLM and TensorRT-BNM:

```bash
$ export PACKAGE_DIR=__path_to_save_wheel_on_host__
$ make -C docker trtbnm_wheel
```

## Testing

```bash
$ pytest -s $(pwd)/tests
```

## Troubleshoots

```
+ TensorRT bug on chunking loop (fixed for version 10.11): NVBug 5190992

+ TRT activation's memory are not reused for the trifast custom kernel: an internal discussion
```
