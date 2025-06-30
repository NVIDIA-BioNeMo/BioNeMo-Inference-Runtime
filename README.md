<div align="center">

# TensorRT-BioNemo

<h4> A TensorRT Toolbox for Optimized Structure Prediction Model Inference, inspired by TensorRT-LLM </h4>
<div align="left">

## Getting Started

Please use the `.devcontainer` for `vscode` or `cursor`. In the devcontainer, run as the following:

```bash
$ git submodule update --init --recursive
$ cd 3rdparty/TensorRT-LLM
# Build TensorRT-LLM from source
$ export PATH=$PATH:$HOME/.local/bin # this for conan executable
$ git checkout v1.0.0rc0
$ python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --fast # add `-b Debug` for build debug with trt-llm
$ pip install -e . && cd ../..
# Install cuequiv
$ cd 3rdparty/cuequiv-ops && pip install nanobind pynvml scikit-build-core && ./build.sh cue-ops && cd ../..
# Build TRT plugins for TensorRT-BioNemo
$ ./scripts/build_cpp.sh
# Install as develop mode
$ pip install -e .
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
