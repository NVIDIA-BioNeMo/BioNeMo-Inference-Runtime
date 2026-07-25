<div align="center">

# TensorRT-BioNemo

<h4>GPU-accelerated structure prediction model inference</h4>
<div align="left">

## Getting Started

### Develop Docker

Build the development image directly from the NVIDIA PyTorch base:

```bash
$ make -C docker trtbnm_dev REGISTRY_IMAGE=tensorrt-bionemo TAG=dev
```

Run the resulting image:

```bash
$ docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it \
  tensorrt-bionemo:dev
```

The repository and development dependencies are already installed in editable
mode.

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
