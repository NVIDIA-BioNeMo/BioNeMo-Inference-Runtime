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
$ git checkout c384d26736acafa8fb9db0b87ab07da12fad8a06
$ python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --fast # add `-b Debug` for build debug with trt-llm
$ pip install -e .
$ cd ../..
$ pip install -e .
```

## Testing

```bash
$ pytest -s $(pwd)/tests
```
