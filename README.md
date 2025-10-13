<div align="center">

# TensorRT-BioNemo

<h4> A TensorRT Toolbox for Optimized Structure Prediction Model Inference, inspired by TensorRT-LLM </h4>
<div align="left">

## Getting Started

### Model hubs

Currently, the TRT-BNM is supporting local and huggingface model hubs. The `load_weights` function will load checkpoints from the local hub, if not found in the local hub, it will fallback to the huggingface remote hub. You can specify the hub by setting the `hub` option in this function.

```python
from tensorrt_bionemo.hubs import load_weights
os.environ["BOLTZ2_CKPT"] = "boltz2.ckpt"                   # see environment variables in tensorrt_bionemo/hubs/local.py
boltz2_state_dict = load_weights("boltz-2")                 # priority: local -> hf
boltz2_state_dict = load_weights("boltz-2", hub="hf")       # only hf
boltz2_state_dict = load_weights("boltz-2", cache_path="boltz2.ckpt") # local checkpoint from a specific path
```

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
